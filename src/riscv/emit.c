/* RISC-V (RV64G + RVV 1.0) machine-code emitter. See emit.h.
 *
 * Encoding notes (all verified against GNU as):
 * - vsetvli:  bit31=0,  bits30:20 = vtypei, rs1, funct3=111, rd, 0x57.
 * - vsetivli: bit31=1,  bits30:20 = vtypei, uimm, funct3=111, rd, 0x57.
 *   vtypei[10:0] = {1'b1 for vsetivli (0 for vsetvli), 2'b00, vma, vta,
 *   vsew[2:0], vlmul[2:0]}.
 * - vle8.v/vse8.v (unit stride): nf=000, mop=00, vm=1, rs1 in the
 *   vs1 field (bits 19:15), lumop/sumop=00000, width=000, vd/vs3;
 *   major opcode 0000111 for loads, 0100111 for stores. The maskedtail
 *   twin's _m forms are the same words with the vm bit (bit 25) = 0.
 * - vid.v: OPMVV (funct3=010), funct6=010100, vs1 field = 10001, vm=1,
 *   vs2 field = 00000.
 * - vmsltu.vx: OPIVX (funct3=100), funct6=011010, vm=1; vd is always v0.
 *   (NOT 011100: that is vmsleu.vx, unsigned <=; NOT 011011: that is
 *   vmslt.vx, signed <. See emit.h.)
 * - blt/bge: B-type, funct3=100/101 (for the tail's blez synthesis).
 * - OPIVV (funct3=000): vxor.vv funct6=001011, vand.vv funct6=001001,
 *   vrgather.vv funct6=001100 (table in vs2 field, indices in vs1 field),
 *   vmv.v.v funct6=010111 (source in vs1 field, vs2 field = 0).
 * - OPIVX (funct3=100): vand.vx funct6=001001, vmv.v.x funct6=010111
 *   (scalar in the vs1 field, vs2 field = 0).
 * - OPIVI (funct3=011): vsrl.vi funct6=101000 (uimm in vs1 field).
 * - Zvbb (OPIVV): vandn.vv funct6=000001 (vd[i] = vs1[i] & ~vs2[i];
 *   NOT 000101, which is vmin); vbrev8.v is VXUNARY0 funct6=010010
 *   with vs1 field = 01000 selecting vbrev8 (source in vs2 field).
 *   Per the ratified Zvbb spec; cross-checked against the wasmtime
 *   Zvbb encoding table. NOT GAS-verified (no RISC-V toolchain here);
 *   the shape test pins the words.
 */
#include "emit.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>

void em_init(em_t *e, uint8_t *buf, size_t cap)
{
    memset(e, 0, sizeof(*e));
    e->buf = buf;
    e->cap = cap;
}

void em_byte(em_t *e, uint8_t v)
{
    assert(e->len < e->cap);
    e->buf[e->len++] = v;
}

static void emit32(em_t *e, uint32_t w)
{
    assert(e->len + 4 <= e->cap);
    e->buf[e->len++] = (uint8_t)w;
    e->buf[e->len++] = (uint8_t)(w >> 8);
    e->buf[e->len++] = (uint8_t)(w >> 16);
    e->buf[e->len++] = (uint8_t)(w >> 24);
}

static uint32_t read32(em_t *e, size_t at)
{
    assert(at + 4 <= e->len);
    return (uint32_t)e->buf[at] | ((uint32_t)e->buf[at + 1] << 8) |
           ((uint32_t)e->buf[at + 2] << 16) | ((uint32_t)e->buf[at + 3] << 24);
}

static void patch32(em_t *e, size_t at, uint32_t w)
{
    assert(at + 4 <= e->len);
    e->buf[at] = (uint8_t)w;
    e->buf[at + 1] = (uint8_t)(w >> 8);
    e->buf[at + 2] = (uint8_t)(w >> 16);
    e->buf[at + 3] = (uint8_t)(w >> 24);
}

int em_label(em_t *e)
{
    if (e->nlab == e->clab) {
        e->clab = e->clab ? e->clab * 2 : 32;
        e->lpos = realloc(e->lpos, (size_t)e->clab * sizeof(*e->lpos));
        assert(e->lpos);
    }
    e->lpos[e->nlab] = (size_t)-1;
    return e->nlab++;
}

void em_bind(em_t *e, int lab)
{
    assert(lab >= 0 && lab < e->nlab);
    assert(e->lpos[lab] == (size_t)-1); /* bind once */
    e->lpos[lab] = e->len;
}

static void add_fix(em_t *e, size_t at, int lab, int add, int kind)
{
    if (e->nfix == e->cfix) {
        e->cfix = e->cfix ? e->cfix * 2 : 32;
        e->fix = realloc(e->fix, (size_t)e->cfix * sizeof(*e->fix));
        assert(e->fix);
    }
    e->fix[e->nfix].at = at;
    e->fix[e->nfix].lab = lab;
    e->fix[e->nfix].add = add;
    e->fix[e->nfix].kind = kind;
    e->nfix++;
}

void em_resolve(em_t *e)
{
    for (int i = 0; i < e->nfix; i++) {
        size_t labpos = e->lpos[e->fix[i].lab];
        assert(labpos != (size_t)-1); /* every referenced label is bound */
        long target = (long)(labpos + (size_t)e->fix[i].add);
        long at = (long)e->fix[i].at;
        if (e->fix[i].kind == 0) {
            /* B-type branch: patch the 12-bit signed offset (in units of
             * 2 bytes) into the already-emitted instruction. */
            long off = target - at;
            assert(off >= -4096 && off < 4096 && (off & 1) == 0);
            uint32_t w = read32(e, (size_t)at);
            uint32_t u = (uint32_t)off;
            w |= ((u >> 12) & 1u) << 31; /* imm[12] */
            w |= ((u >> 5) & 0x3fu) << 25; /* imm[10:5] */
            w |= ((u >> 1) & 0xfu) << 8; /* imm[4:1] */
            w |= ((u >> 11) & 1u) << 7; /* imm[11] */
            patch32(e, (size_t)at, w);
        } else if (e->fix[i].kind == 2) {
            /* J-type: patch the 20-bit signed offset (units of 2 bytes). */
            long off = target - at;
            assert(off >= -(1L << 20) && off < (1L << 20) && (off & 1) == 0);
            uint32_t w = read32(e, (size_t)at);
            uint32_t u = (uint32_t)off;
            w |= ((u >> 20) & 1u) << 31; /* imm[20] */
            w |= ((u >> 1) & 0x3ffu) << 21; /* imm[10:1] */
            w |= ((u >> 11) & 1u) << 20; /* imm[11] */
            w |= ((u >> 12) & 0xffu) << 12; /* imm[19:12] */
            patch32(e, (size_t)at, w);
        } else {
            /* auipc+addi pair: PC-relative address of (label + add). */
            long off = target - at;
            long hi = (off + 0x800) >> 12;
            long lo = off - (hi << 12);
            assert(hi >= -(1L << 19) && hi < (1L << 19));
            assert(lo >= -2048 && lo < 2048);
            uint32_t au = read32(e, (size_t)at);
            au |= ((uint32_t)hi & 0xfffffu) << 12;
            patch32(e, (size_t)at, au);
            uint32_t ad = read32(e, (size_t)at + 4);
            ad |= ((uint32_t)lo & 0xfffu) << 20;
            patch32(e, (size_t)at + 4, ad);
        }
    }
}

void em_free(em_t *e)
{
    free(e->lpos);
    free(e->fix);
    e->lpos = NULL;
    e->fix = NULL;
}

void em_align(em_t *e, unsigned a)
{
    assert(a && (a & (a - 1)) == 0);
    /* NOP = addi x0, x0, 0 */
    while (e->len & (a - 1))
        emit32(e, 0x00000013u);
}

/* The all-zero word is architecturally guaranteed illegal on RISC-V
 * (UNIMP); it traps even where the C extension is present (c.unimp). */
void em_unimp(em_t *e) { emit32(e, 0x00000000u); }

/* Pad to alignment a with UNIMP words. Dead space after ret only: a
 * fall-through past ret must trap immediately, never slide through NOPs
 * into the const pool. The 16-byte pool alignment is always a whole
 * number of 4-byte words. */
void em_trap_align(em_t *e, unsigned a)
{
    assert(a && (a & (a - 1)) == 0);
    assert((e->len & 3) == 0); /* all emits are 32-bit; no sub-word tail */
    while (e->len & (a - 1))
        em_unimp(e);
}

/* ---------------- integer ---------------- */

void em_add(em_t *e, int rd, int rs1, int rs2)
{
    emit32(e, (0x00u << 25) | ((uint32_t)(rs2 & 31) << 20) |
                  ((uint32_t)(rs1 & 31) << 15) | (0x0u << 12) |
                  ((uint32_t)(rd & 31) << 7) | 0x33u);
}

void em_sub(em_t *e, int rd, int rs1, int rs2)
{
    emit32(e, (0x20u << 25) | ((uint32_t)(rs2 & 31) << 20) |
                  ((uint32_t)(rs1 & 31) << 15) | (0x0u << 12) |
                  ((uint32_t)(rd & 31) << 7) | 0x33u);
}

/* M extension: mul/divu. funct7=0000001. */
void em_mul(em_t *e, int rd, int rs1, int rs2)
{
    emit32(e, (0x01u << 25) | ((uint32_t)(rs2 & 31) << 20) |
                  ((uint32_t)(rs1 & 31) << 15) | (0x0u << 12) |
                  ((uint32_t)(rd & 31) << 7) | 0x33u);
}

void em_divu(em_t *e, int rd, int rs1, int rs2)
{
    emit32(e, (0x01u << 25) | ((uint32_t)(rs2 & 31) << 20) |
                  ((uint32_t)(rs1 & 31) << 15) | (0x5u << 12) |
                  ((uint32_t)(rd & 31) << 7) | 0x33u);
}

void em_addi(em_t *e, int rd, int rs1, int imm12)
{
    assert(imm12 >= -2048 && imm12 < 2048);
    emit32(e, (((uint32_t)imm12 & 0xfffu) << 20) |
                  ((uint32_t)(rs1 & 31) << 15) | (0x0u << 12) |
                  ((uint32_t)(rd & 31) << 7) | 0x13u);
}

void em_slli(em_t *e, int rd, int rs1, unsigned sh)
{
    assert(sh < 64);
    emit32(e, (0x00u << 25) | ((sh & 0x3fu) << 20) |
                  ((uint32_t)(rs1 & 31) << 15) | (0x1u << 12) |
                  ((uint32_t)(rd & 31) << 7) | 0x13u);
}

void em_srli(em_t *e, int rd, int rs1, unsigned sh)
{
    assert(sh < 64);
    emit32(e, (0x00u << 25) | ((sh & 0x3fu) << 20) |
                  ((uint32_t)(rs1 & 31) << 15) | (0x5u << 12) |
                  ((uint32_t)(rd & 31) << 7) | 0x13u);
}

/* srai: arithmetic right shift; imm[11:5] = 0100000 distinguishes it
 * from srli. GAS-verified (srai t3,t2,63 -> 0x43f3de13). */
void em_srai(em_t *e, int rd, int rs1, unsigned sh)
{
    assert(sh < 64);
    emit32(e, (0x20u << 25) | ((sh & 0x3fu) << 20) |
                  ((uint32_t)(rs1 & 31) << 15) | (0x5u << 12) |
                  ((uint32_t)(rd & 31) << 7) | 0x13u);
}

/* and: OP, funct3=111, funct7=0000000.
 * GAS-verified (and t2,t2,t3 -> 0x01c3f3b3). */
void em_and(em_t *e, int rd, int rs1, int rs2)
{
    emit32(e, (0x00u << 25) | ((uint32_t)(rs2 & 31) << 20) |
                  ((uint32_t)(rs1 & 31) << 15) | (0x7u << 12) |
                  ((uint32_t)(rd & 31) << 7) | 0x33u);
}

void em_ld(em_t *e, int rd, int rs1, int off12)
{
    assert(off12 >= -2048 && off12 < 2048);
    emit32(e, (((uint32_t)off12 & 0xfffu) << 20) |
                  ((uint32_t)(rs1 & 31) << 15) | (0x3u << 12) |
                  ((uint32_t)(rd & 31) << 7) | 0x03u);
}

void em_sd(em_t *e, int rs2, int rs1, int off12)
{
    assert(off12 >= -2048 && off12 < 2048);
    uint32_t u = (uint32_t)off12 & 0xfffu;
    emit32(e, (((u >> 5) & 0x7fu) << 25) | ((uint32_t)(rs2 & 31) << 20) |
                  ((uint32_t)(rs1 & 31) << 15) | (0x3u << 12) |
                  ((u & 0x1fu) << 7) | 0x23u);
}

static void em_branch(em_t *e, int funct3, int rs1, int rs2, int lab)
{
    size_t at = e->len;
    emit32(e, ((uint32_t)(rs2 & 31) << 20) | ((uint32_t)(rs1 & 31) << 15) |
                  ((uint32_t)(funct3 & 7) << 12) | 0x63u);
    add_fix(e, at, lab, 0, 0);
}

void em_beq(em_t *e, int rs1, int rs2, int lab)
{
    em_branch(e, 0x0, rs1, rs2, lab);
}

void em_bne(em_t *e, int rs1, int rs2, int lab)
{
    em_branch(e, 0x1, rs1, rs2, lab);
}

void em_blt(em_t *e, int rs1, int rs2, int lab)
{
    em_branch(e, 0x4, rs1, rs2, lab);
}

void em_bge(em_t *e, int rs1, int rs2, int lab)
{
    em_branch(e, 0x5, rs1, rs2, lab);
}

void em_jal(em_t *e, int rd, int lab)
{
    size_t at = e->len;
    emit32(e, ((uint32_t)(rd & 31) << 7) | 0x6fu);
    add_fix(e, at, lab, 0, 2);
}

void em_lui(em_t *e, int rd, int imm20)
{
    emit32(e, (((uint32_t)imm20 & 0xfffffu) << 12) |
                  ((uint32_t)(rd & 31) << 7) | 0x37u);
}

void em_jalr(em_t *e, int rd, int rs1, int off12)
{
    assert(off12 >= -2048 && off12 < 2048);
    emit32(e, (((uint32_t)off12 & 0xfffu) << 20) |
                  ((uint32_t)(rs1 & 31) << 15) | (0x0u << 12) |
                  ((uint32_t)(rd & 31) << 7) | 0x67u);
}

void em_lla(em_t *e, int rd, int lab, int add)
{
    size_t at = e->len;
    emit32(e, ((uint32_t)(rd & 31) << 7) | 0x17u); /* auipc rd, 0 */
    emit32(e, ((uint32_t)(rd & 31) << 15) | ((uint32_t)(rd & 31) << 7) |
                  0x13u); /* addi rd, rd, 0 */
    add_fix(e, at, lab, add, 1);
}

/* ---------------- vector ---------------- */

static uint32_t vtypei(int ivli, int vsew, int vlmul, int vta, int vma)
{
    return ((uint32_t)(ivli & 1) << 10) | ((uint32_t)(vma & 1) << 7) |
           ((uint32_t)(vta & 1) << 6) | ((uint32_t)(vsew & 7) << 3) |
           (uint32_t)(vlmul & 7);
}

void em_vsetvli(em_t *e, int rd, int rs1, int vsew, int vlmul, int vta, int vma)
{
    emit32(e, (0u << 31) | (vtypei(0, vsew, vlmul, vta, vma) << 20) |
                  ((uint32_t)(rs1 & 31) << 15) | (0x7u << 12) |
                  ((uint32_t)(rd & 31) << 7) | 0x57u);
}

void em_vsetivli(em_t *e, int rd, unsigned uimm, int vsew, int vlmul, int vta,
                 int vma)
{
    assert(uimm < 32);
    emit32(e, (1u << 31) | (vtypei(1, vsew, vlmul, vta, vma) << 20) |
                  ((uimm & 31u) << 15) | (0x7u << 12) |
                  ((uint32_t)(rd & 31) << 7) | 0x57u);
}

/* Unit-stride byte load/store skeleton with an explicit vm bit, so the
 * masked (vm=0) forms differ from the unmasked ones in exactly one bit
 * (reviewable by diff). nf=000, mop=00 (unit stride), rs1 in the vs1
 * field (bits 19:15), lumop/sumop=00000, width=000. */
static void em_vle8_v(em_t *e, int vm, int vd, int rs1)
{
    emit32(e, (0u << 29) | ((uint32_t)(vm & 1) << 25) | (0u << 20) |
                  ((uint32_t)(rs1 & 31) << 15) | (0x0u << 12) |
                  ((uint32_t)(vd & 31) << 7) | 0x07u);
}

static void em_vse8_v(em_t *e, int vm, int vs3, int rs1)
{
    /* note the STORE major opcode 0100111 */
    emit32(e, (0u << 29) | ((uint32_t)(vm & 1) << 25) | (0u << 20) |
                  ((uint32_t)(rs1 & 31) << 15) | (0x0u << 12) |
                  ((uint32_t)(vs3 & 31) << 7) | 0x27u);
}

void em_vle8(em_t *e, int vd, int rs1)
{
    em_vle8_v(e, 1, vd, rs1);
}

void em_vse8(em_t *e, int vs3, int rs1)
{
    em_vse8_v(e, 1, vs3, rs1);
}

void em_vle8_v_m(em_t *e, int vd, int rs1)
{
    /* vm=0: masked by v0; inactive lanes do not fault. */
    em_vle8_v(e, 0, vd, rs1);
}

void em_vse8_v_m(em_t *e, int vs3, int rs1)
{
    /* vm=0: masked by v0; inactive lanes are not stored. */
    em_vse8_v(e, 0, vs3, rs1);
}

void em_ntl_all(em_t *e)
{
    /* GAS-verified (`ntl.all`, -march=rv64g_zihintntl): 0x00500033.
     * ADD x0,x0,x5 is a HINT encoding: it executes as a NOP on every
     * RISC-V implementation, including ones without Zihintntl. */
    emit32(e, 0x00500033u);
}

void em_prefetch_r(em_t *e, int rs1, unsigned off)
{
    /* Zicbop prefetch.r is an ORI x0,rs1,imm HINT: imm[4:0]=0b00001
     * selects .r, imm[11:5] is the offset in 32-byte units.
     * GAS-verified (`-march=rv64gcv_zicbop`):
     *   prefetch.r 0(a0)     -> 0x00156013
     *   prefetch.r 32(t2)     -> 0x0213e013
     *   prefetch.r 2016(t2)   -> 0x7e13e013
     * GAS rejects offsets above 2016, so the emitter caps there too. */
    assert((off & 31u) == 0 && off <= 2016);
    uint32_t imm = (off & 0x7e0u) | 0x001u;
    emit32(e, (imm << 20) | ((uint32_t)(rs1 & 31) << 15) |
                      (0x6u << 12) | (0x0u << 7) | 0x13u);
}

/* OP-V ALU skeleton: funct6 | vm | vs2 | vs1 | funct3 | vd | 0x57 */
static void em_opv(em_t *e, int funct6, int vs2, int vs1, int funct3, int vd)
{
    emit32(e, ((uint32_t)(funct6 & 0x3f) << 26) | (1u << 25) |
                  ((uint32_t)(vs2 & 31) << 20) | ((uint32_t)(vs1 & 31) << 15) |
                  ((uint32_t)(funct3 & 7) << 12) | ((uint32_t)(vd & 31) << 7) |
                  0x57u);
}

/* vid.v vd: vd[i] = i (element index), zero-extended to SEW. OPMVV
 * (funct3=010), funct6=010100, vs1 field = 10001 selects vid, vm=1
 * (unmasked), vs2 field = 00000 (the V spec requires the vs2 field to
 * be v0, otherwise the encoding is reserved). NOTE: the destination
 * group must not overlap the mask register (v0) when LMUL > 1 (V spec
 * §16.9: illegal-insn); the maskedtail twin therefore targets a
 * body-clobbered temp, never v0. GAS-verified: vid.v v9 -> 0x5208a4d7
 * (funct3=010, vs1=10001). */
void em_vid_v(em_t *e, int vd)
{
    em_opv(e, 0x14 /*010100*/, 0 /*vs2 field = 0*/, 0x11 /*10001=vid*/,
           0x2 /*OPMVV*/, vd);
}

/* vmsltu.vx vd, vs2, rs1: vd[i] = (vs2[i] <u rs1), unsigned -- the
 * mask producer for the maskedtail twin. OPIVX (funct3=100),
 * funct6=011010, vm=1 (unmasked: the old mask is dead at each rewrite,
 * so no masked-off-lane hazard). vd MUST be v0: every masked operation
 * hardwires v0 as THE mask register, so a mask produced anywhere else
 * would need a copy through v0 anyway; the assert makes misuse loud.
 * Unsigned (not vmslt.vx): at SEW=8 the scalar operand truncates to 8
 * bits (V spec §11.1), so a signed compare would read tmp >= 128 as
 * negative; vmsltu.vx compares the low 8 bits unsigned instead. Mask
 * math is exact when VLMAX <= 255 (vid.v indexes mod 256); the
 * backend rejects wider configs (see jit_build_version).
 * GAS-verified: vmsltu.vx v0,v9,a3 -> 0x6a96c057 (funct6=011010,
 * funct3=100). Note 011100 is vmsleu.vx (unsigned <=), not vmsltu.vx. */
void em_vmsltu_vx(em_t *e, int vd, int vs2, int rs1)
{
    assert(vd == 0); /* v0 is the mask register */
    em_opv(e, 0x1a /*011010*/, vs2, rs1, 0x4 /*OPIVX*/, vd);
}

void em_vxor_vv(em_t *e, int vd, int vs1, int vs2)
{
    em_opv(e, 0x0b /*001011*/, vs2, vs1, 0x0 /*OPIVV*/, vd);
}

void em_vand_vv(em_t *e, int vd, int vs1, int vs2)
{
    em_opv(e, 0x09 /*001001*/, vs2, vs1, 0x0 /*OPIVV*/, vd);
}

void em_vand_vx(em_t *e, int vd, int vs2, int rs1)
{
    em_opv(e, 0x09 /*001001*/, vs2, rs1, 0x4 /*OPIVX*/, vd);
}

/* vd[i] = vs2[i] & sext(imm): the 5-bit immediate is sign-extended to
 * SEW, so imm=15 is the 0x0F nibble mask for SEW=8. OPIVI class. */
void em_vand_vi(em_t *e, int vd, int vs2, unsigned uimm)
{
    assert(uimm < 32);
    em_opv(e, 0x09 /*001001*/, vs2, (int)uimm, 0x3 /*OPIVI*/, vd);
}

void em_vsrl_vi(em_t *e, int vd, int vs2, unsigned uimm)
{
    assert(uimm < 32);
    em_opv(e, 0x28 /*101000*/, vs2, (int)uimm, 0x3 /*OPIVI*/, vd);
}

void em_vsll_vi(em_t *e, int vd, int vs2, unsigned uimm)
{
    assert(uimm < 32);
    em_opv(e, 0x25 /*100101*/, vs2, (int)uimm, 0x3 /*OPIVI*/, vd);
}

/* Zvbb vector bitmanip. OPIVV class (funct3=000); operand placement
 * follows the standard convention (first assembly source in
 * bits[19:15], second in bits[24:20]), like em_vand_vv above.
 *   vandn.vv vd, vs1, vs2: vd[i] = vs1[i] & ~vs2[i].
 *     funct6=000001 (NOT 000101: that is vmin).
 *   vbrev8.v vd, vs: vd[i] = brev8(vs[i]) (bit-reverse each byte).
 *     VXUNARY0: funct6=010010, vs1 field = 01000 selects vbrev8, source
 *     in the vs2 field (bits[24:20]).
 * Encodings per the ratified Zvbb spec, cross-checked against the
 * wasmtime Zvbb implementation's encoding table. NOT GAS-verified (no
 * RISC-V toolchain on this host); the shape test pins the exact words
 * so a future GAS check can confirm or correct them. */
void em_vandn_vv(em_t *e, int vd, int vs1, int vs2)
{
    em_opv(e, 0x01 /*000001*/, vs2, vs1, 0x0 /*OPIVV*/, vd);
}

void em_vbrev8_v(em_t *e, int vd, int vs)
{
    em_opv(e, 0x12 /*010010*/, vs, 0x08 /*01000=vbrev8*/, 0x0 /*OPIVV*/,
           vd);
}

void em_vrgather_vv(em_t *e, int vd, int vtab, int vidx)
{
    /* note: OPIVV class (funct3=000), not OPMVV */
    em_opv(e, 0x0c /*001100*/, vtab, vidx, 0x0 /*OPIVV*/, vd);
}

/* Zvbc/Zvkg carryless multiply. OPMVV class (funct3=010); the first
 * source register goes in bits[24:20], the second in bits[19:15], the
 * standard placement (GAS-verified words:
 *   vclmul.vv  v8,v9,v10  -> 0x32952457
 *   vclmulh.vv v11,v9,v10 -> 0x369525d7).
 * A dedicated encoder (not em_opv) keeps the field assignment explicit
 * and independently testable; the shape test pins it byte-for-byte. */
static void em_vclmul(em_t *e, int funct6, int vd, int vs1, int vs2)
{
    emit32(e, ((uint32_t)(funct6 & 0x3f) << 26) | (1u << 25) |
                  ((uint32_t)(vs1 & 31) << 20) | ((uint32_t)(vs2 & 31) << 15) |
                  (0x2u << 12) | ((uint32_t)(vd & 31) << 7) | 0x57u);
}

void em_vclmul_vv(em_t *e, int vd, int vs1, int vs2)
{
    em_vclmul(e, 0x0c /*001100*/, vd, vs1, vs2);
}

void em_vclmulh_vv(em_t *e, int vd, int vs1, int vs2)
{
    em_vclmul(e, 0x0d /*001101*/, vd, vs1, vs2);
}

void em_vmv_v_v(em_t *e, int vd, int vs1)
{
    /* note: OPIVV class (funct3=000), not OPMVV */
    em_opv(e, 0x17 /*010111*/, 0, vs1, 0x0 /*OPIVV*/, vd);
}

void em_vmv_v_x(em_t *e, int vd, int rs1)
{
    /* note: OPIVX class (funct3=100), not OPMVX */
    em_opv(e, 0x17 /*010111*/, 0, rs1, 0x4 /*OPIVX*/, vd);
}
