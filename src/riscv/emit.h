/* RISC-V (RV64G + RVV 1.0) machine-code emitter for the RAID JIT.
 *
 * Supports exactly the instruction subset the code generator needs:
 * integer ALU (add/sub/addi/slli), loads/stores (ld/sd), branches
 * (beq/bne), lui/auipc/jalr, vsetvli/vsetivli, unit-stride byte vector
 * loads/stores (vle8.v/vse8.v), and the integer vector ALU subset used
 * for nibble-table GF(2^8) multiplies: vxor.vv, vand.vv, vand.vx,
 * vand.vi, vsrl.vi, vrgather.vv, vmv.v.v, vmv.v.x; plus the Zvbc/Zvkg
 * carryless multiplies vclmul.vv/vclmulh.vv used by the MUL_ZVBC twin;
 * plus the Zvbb bitmanip instructions vandn.vv and vbrev8.v (detected
 * but not yet consumed by any backend path; see rvv_zvbb).
 *
 * Labels with fixups (B-type branches, J-type jumps, PC-relative
 * auipc+addi pairs for
 * constant-pool addressing), NOP-sled alignment. Only 32-bit
 * instructions are emitted (no C extension), so every instruction is 4
 * bytes. Vector instruction encodings were verified byte-for-byte
 * against GNU as (riscv64-linux-gnu-as -march=rv64gcv; the Zvbc/Zvkg
 * carryless multiplies against -march=rv64gcv_zvbc_zvkg).
 */
#pragma once
#include <stddef.h>
#include <stdint.h>

/* GPR ids (ABI names) */
enum {
    X0 = 0, RA = 1, SP = 2, GP = 3, TP = 4,
    T0 = 5, T1 = 6, T2 = 7,
    S0 = 8, S1 = 9,
    A0 = 10, A1 = 11, A2 = 12, A3 = 13, A4 = 14, A5 = 15, A6 = 16, A7 = 17,
    S2 = 18, S3 = 19, S4 = 20, S5 = 21, S6 = 22, S7 = 23,
    S8 = 24, S9 = 25, S10 = 26, S11 = 27,
    T3 = 28, T4 = 29, T5 = 30, T6 = 31
};

/* SEW / LMUL selectors for vsetvli/vsetivli (the JIT uses e8 with
 * m1/m2/m4/m8 per the LMUL tuning axis) */
enum { VSEW8 = 0, VSEW16 = 1, VSEW32 = 2, VSEW64 = 3 };
enum { VLMUL_M1 = 0, VLMUL_M2 = 1, VLMUL_M4 = 2, VLMUL_M8 = 3 };

typedef struct {
    uint8_t *buf;
    size_t cap, len;
    size_t *lpos;   /* bound position per label, or SIZE_MAX */
    int nlab, clab;
    struct {
        size_t at;  /* offset of the first instruction to patch */
        int lab;
        int add;    /* extra addend (constant-pool offset) */
        int kind;   /* 0 = B-type branch, 1 = auipc+addi pair, 2 = J-type */
    } *fix;
    int nfix, cfix;
} em_t;

void em_init(em_t *e, uint8_t *buf, size_t cap);
void em_byte(em_t *e, uint8_t v); /* raw byte emit (constant pool) */
int  em_label(em_t *e);           /* returns label id */
void em_bind(em_t *e, int lab);
void em_resolve(em_t *e); /* patch all fixups; call after binding all */
void em_free(em_t *e);    /* release label/fixup tables (not the code buf) */
void em_align(em_t *e, unsigned a); /* NOP sled to alignment a (power of 2) */

/* --- integer --- */
void em_add(em_t *e, int rd, int rs1, int rs2);
void em_sub(em_t *e, int rd, int rs1, int rs2);
void em_mul(em_t *e, int rd, int rs1, int rs2);   /* M extension */
void em_divu(em_t *e, int rd, int rs1, int rs2);  /* M extension */
void em_addi(em_t *e, int rd, int rs1, int imm12); /* -2048..2047 */
void em_slli(em_t *e, int rd, int rs1, unsigned sh); /* 0..63 */
void em_srli(em_t *e, int rd, int rs1, unsigned sh); /* 0..63 */
void em_srai(em_t *e, int rd, int rs1, unsigned sh); /* 0..63, arithmetic */
void em_and(em_t *e, int rd, int rs1, int rs2);
void em_ld(em_t *e, int rd, int rs1, int off12);   /* rd = [rs1+off] */
void em_sd(em_t *e, int rs2, int rs1, int off12);  /* [rs1+off] = rs2 */
void em_beq(em_t *e, int rs1, int rs2, int lab);   /* rel, label fixup */
void em_bne(em_t *e, int rs1, int rs2, int lab);   /* rel, label fixup */
void em_blt(em_t *e, int rs1, int rs2, int lab);   /* rel, label fixup */
void em_bge(em_t *e, int rs1, int rs2, int lab);   /* rel, label fixup */
/* jal rd, label: J-type, +-1 MiB label fixup (for jumps beyond the
 * B-type +-4 KiB range, e.g. the strip-mining loop-back of a large
 * unrolled body) */
void em_jal(em_t *e, int rd, int lab);
void em_lui(em_t *e, int rd, int imm20);
void em_jalr(em_t *e, int rd, int rs1, int off12); /* ret = jalr x0, ra, 0 */
void em_unimp(em_t *e); /* all-zero word: permanently illegal (trap sled) */
void em_trap_align(em_t *e, unsigned a); /* UNIMP sled to alignment a (power
                                          * of 2); dead space after ret only */
/* rd = address of (label + add); emits auipc+addi with a paired fixup */
void em_lla(em_t *e, int rd, int lab, int add);

/* --- vector (all unmasked, vm=1, unless the _m masked forms) --- */
void em_vsetvli(em_t *e, int rd, int rs1, int vsew, int vlmul, int vta,
                int vma);
void em_vsetivli(em_t *e, int rd, unsigned uimm, int vsew, int vlmul, int vta,
                 int vma);
void em_vle8(em_t *e, int vd, int rs1);  /* vd = *(rs1), unit stride */
void em_vse8(em_t *e, int vs3, int rs1); /* *(rs1) = vs3, unit stride */
/* Masked unit-stride byte load/store (vm=0): the mask register is
 * always v0. Used by the maskedtail twin's tail; inactive lanes do not
 * fault (no over-read/over-write past the tail). */
void em_vle8_v_m(em_t *e, int vd, int rs1);  /* vd = *(rs1), v0.t */
void em_vse8_v_m(em_t *e, int vs3, int rs1); /* *(rs1) = vs3, v0.t */
/* vid.v vd: vd[i] = i (element index). OPMVV, funct6=010100, vs1 field
 * = 10001 selects vid, vm=1 (unmasked), vs2 field = 00000 (reserved;
 * the V spec requires it to be v0). NOTE: the destination group must
 * not overlap the mask register (v0) when LMUL > 1 -- the V spec
 * raises illegal-insn there -- so the maskedtail twin writes the index
 * vector into a body-clobbered temp, never v0. GAS-verified (vid.v v9
 * -> 0x5208a4d7). */
void em_vid_v(em_t *e, int vd);
/* vmsltu.vx vd, vs2, rs1: vd[i] = (vs2[i] <u rs1) unsigned, the mask
 * producer for the maskedtail twin. OPIVX, funct6=011010, vm=1
 * (unmasked: the old mask is dead at each rewrite). vd MUST be v0
 * (the mask register); the emitter asserts this. Unsigned, not
 * vmslt.vx (funct6=011011 signed): at SEW=8 the scalar truncates to 8
 * bits (V spec §11.1), so tmp >= 128 would read negative under the
 * signed compare. GAS-verified (vmsltu.vx v0,v9,a3 -> 0x6a96c057;
 * note 011100 is vmsleu.vx, unsigned <=). */
void em_vmsltu_vx(em_t *e, int vd, int vs2, int rs1);
/* Zihintntl NTL.ALL: non-temporal locality hint for the immediately
 * following memory instruction (streaming: no exploitable temporal
 * locality). Encoded as ADD x0,x0,x5 -- a HINT, i.e. architecturally a
 * NOP on every implementation, so no feature gating is needed. */
void em_ntl_all(em_t *e);
/* Zicbop prefetch.r: prefetch for read off bytes ahead of rs1.
 * off must be a multiple of 32, 0 <= off <= 2016 (the GAS-verified
 * encodable forward range; the JIT backend only requests 32..2016).
 * Encoded as an ORI x0,rs1,imm HINT. */
void em_prefetch_r(em_t *e, int rs1, unsigned off);
/* operand fields are the raw encoding fields: vs1 = bits 19:15,
 * vs2 = bits 24:20 (for .vv both are vector regs; for .vx/.vi the vs1
 * field carries the scalar register / immediate). */
void em_vxor_vv(em_t *e, int vd, int vs1, int vs2);
void em_vand_vv(em_t *e, int vd, int vs1, int vs2);
void em_vand_vx(em_t *e, int vd, int vs2, int rs1);
void em_vand_vi(em_t *e, int vd, int vs2, unsigned uimm);
/* Zvbb vector bitmanip (detected via rvv_zvbb; no backend path emits
 * these yet -- see the I5 analysis):
 *   vandn.vv vd, vs1, vs2: vd[i] = vs1[i] & ~vs2[i] (OPIVV,
 *     funct6=000001; note this is NOT 000101 -- that is vmin).
 *   vbrev8.v vd, vs: vd[i] = brev8(vs[i]) (bit-reverse each byte;
 *     VXUNARY0: funct6=010010, vs1 field = 01000 selects vbrev8).
 * Encodings per the ratified Zvbb spec, cross-checked against the
 * wasmtime Zvbb implementation's encoding table (funct6/vs1 values);
 * NOT yet verified against GNU as (no RISC-V toolchain on this host).
 * The shape test pins the exact words. */
void em_vandn_vv(em_t *e, int vd, int vs1, int vs2); /* Zvbb */
void em_vbrev8_v(em_t *e, int vd, int vs);           /* Zvbb */
void em_vsrl_vi(em_t *e, int vd, int vs2, unsigned uimm);
void em_vsll_vi(em_t *e, int vd, int vs2, unsigned uimm);
void em_vrgather_vv(em_t *e, int vd, int vtab, int vidx); /* vd[i] =
                          vtab[vidx[i]]; table reg in vs2 field, index
                          reg in vs1 field */
/* Zvbc/Zvbc32e carryless multiply (SEW=8 lanes):
 *   vclmul.vv  vd, vs1, vs2: vd[i] = low 8 bits of clmul(vs1[i],vs2[i])
 *   vclmulh.vv vd, vs1, vs2: vd[i] = bits 15..8 of clmul(vs1[i],vs2[i])
 * Both instructions are Zvbc (Vector Crypto §36.2/36.3); the SEW=8
 * forms need the separate Zvbc32e extension (NOT Zvkg: Zvkg defines
 * vghsh/vgmul).
 * OPMVV class (funct3=010). Operand placement follows the standard
 * convention -- the first source register goes in bits[24:20], the
 * second in bits[19:15] -- verified byte-for-byte against GAS
 * (vclmul.vv v8,v9,v10 -> 0x32952457; vclmulh.vv v11,v9,v10 ->
 * 0x369525d7). The operation is commutative, so operand order is
 * functionally irrelevant; what matters is byte-exactness, which the
 * shape test pins. A dedicated encoder is used instead of em_opv so
 * the field assignment is explicit and independently testable. */
void em_vclmul_vv(em_t *e, int vd, int vs1, int vs2);  /* Zvbc */
void em_vclmulh_vv(em_t *e, int vd, int vs1, int vs2); /* Zvbc (SEW=8 form: Zvbc32e) */
void em_vmv_v_v(em_t *e, int vd, int vs1);
void em_vmv_v_x(em_t *e, int vd, int rs1);
