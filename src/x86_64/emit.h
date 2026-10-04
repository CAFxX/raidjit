/* Minimal x86-64 assembler for the RAID JIT.
 * Supports exactly the instruction subset the code generator needs:
 * integer ALU/control flow, VEX-encoded SSE/AVX/AVX2 vector ops and
 * EVEX-encoded AVX512 ops, RIP-relative constant-pool loads, labels with
 * rel32 fixups, NOP-sled alignment, ENDBR64 and UD2. */
#pragma once
#include <stddef.h>
#include <stdint.h>

/* GPR ids */
enum {
    RAX = 0, RCX, RDX, RBX, RSP, RBP, RSI, RDI,
    R8, R9, R10, R11, R12, R13, R14, R15
};

/* condition codes for jcc (as the low 4 bits of the opcode extension) */
enum {
    CC_O = 0, CC_NO, CC_B, CC_AE, CC_E, CC_NE, CC_BE, CC_A,
    CC_S, CC_NS, CC_P, CC_NP, CC_L, CC_GE, CC_LE, CC_G
};

/* vector length in bytes: 16 (SSE), 32 (AVX2), 64 (AVX512) */
enum { VL16 = 16, VL32 = 32, VL64 = 64 };

typedef struct {
    uint8_t *buf;
    size_t cap, len;
    size_t *lpos;   /* bound position per label, or SIZE_MAX */
    int nlab, clab;
    struct {
        size_t at;  /* offset of the disp32 to patch */
        int lab;
        int add;    /* extra addend (constant-pool offset) */
    } *fix;
    int nfix, cfix;
} em_t;

void em_init(em_t *e, uint8_t *buf, size_t cap);
void em_byte(em_t *e, uint8_t v); /* raw byte emit (constant pool) */
int  em_label(em_t *e);          /* returns label id */
void em_bind(em_t *e, int lab);
void em_resolve(em_t *e);        /* patch all fixups; call after binding all */
void em_free(em_t *e);           /* release label/fixup tables (not the code buf) */
void em_align(em_t *e, unsigned a); /* NOP sled to alignment a (power of 2) */

/* --- control / misc --- */
void em_endbr64(em_t *e);
void em_setssbsy(em_t *e);
void em_clrssbsy(em_t *e);
void em_ud2(em_t *e);
void em_ret(em_t *e);
void em_push_r(em_t *e, int r);
void em_pop_r(em_t *e, int r);
void em_jcc(em_t *e, int cc, int lab); /* rel32 */
void em_jmp(em_t *e, int lab);         /* rel32 */

/* --- integer --- */
/* mem operands: base b, index x (-1 = none), scale sc (0..3), disp.
 * Use b = -2 for RIP-relative (only in the _rip variants). */
void em_mov_rr(em_t *e, int d, int s);                       /* r64, r64 */
void em_mov_ri(em_t *e, int d, uint64_t v);                   /* r64, imm */
void em_mov_rm(em_t *e, int d, int b, int x, int sc, int disp); /* r64,[m] */
void em_mov_mr(em_t *e, int b, int x, int sc, int disp, int s); /* [m],r64 */
void em_movzx_rm8(em_t *e, int d, int b, int x, int sc, int disp); /* r32,[m8] */
void em_mov_r_m8(em_t *e, int d, int b, int x, int sc, int disp);  /* r8,[m8] */
void em_mov_m8_r(em_t *e, int b, int x, int sc, int disp, int s);  /* [m8],r8 */
void em_lea(em_t *e, int d, int b, int x, int sc, int disp);
void em_lea_rip(em_t *e, int d, int lab, int add);
void em_add_ri(em_t *e, int d, int v);   /* r64, imm */
void em_sub_ri(em_t *e, int d, int v);
void em_cmp_ri(em_t *e, int d, int v);
void em_add_r32_ri(em_t *e, int d, int v); /* r32, imm */
void em_cmp_r32_ri(em_t *e, int d, int v);
void em_add_rr(em_t *e, int d, int s);  /* r64, r64 */
void em_sub_rr(em_t *e, int d, int s);  /* r64, r64 */
void em_xor_rr(em_t *e, int d, int s);   /* r64, r64 */
void em_xor_r32(em_t *e, int d, int s);  /* r32, r32 */
void em_xor_r8(em_t *e, int d, int s);   /* r8, r8 */
void em_shr_ri(em_t *e, int d, unsigned n); /* r64, imm */
void em_test_rr(em_t *e, int d, int s); /* r64, r64 */
void em_dec_r(em_t *e, int d);
void em_inc_r(em_t *e, int d);

/* --- vector (vd/s regs are 0..31; vl in {16,32,64}) --- */
void em_vmovdqa_ld(em_t *e, int vd, int b, int x, int sc, int disp, int vl);
void em_vmovdqa_st(em_t *e, int b, int x, int sc, int disp, int vs, int vl);
void em_vmovdqa_rr(em_t *e, int vd, int vs, int vl);
/* Non-temporal stores: same alignment contract as the aligned forms.
 * Weakly ordered. The jitted code emits NO fence; a caller that reads
 * the NT-written data (or needs it visible to other agents) must
 * execute SFENCE itself after the jitted function returns. */
void em_vmovntdq_st(em_t *e, int b, int x, int sc, int disp, int vs, int vl);
void em_sfence(em_t *e);
/* Software prefetch: hint 0=NTA, 1=T0, 2=T1, 3=T2. 0F 18 /r. */
void em_prefetch(em_t *e, int hint, int b, int x, int sc, int disp);
void em_vpxor(em_t *e, int vd, int s1, int s2, int vl); /* vd = s1 ^ s2 */
void em_vpxor_rr(em_t *e, int vd, int vs, int vl);      /* vd ^= vs */
void em_vpxor_mem(em_t *e, int vd, int vs1,
                  int b, int x, int sc, int disp, int vl); /* vd = vs1 ^ [m] */
void em_vpshufb(em_t *e, int vd, int s1, int s2, int vl);
void em_vpshufb_masked(em_t *e, int vd, int s1, int s2, int vl, int k, int z);
void em_vmovdqu8_ld_masked(em_t *e, int vd, int b, int x, int sc, int disp,
                          int vl, int k, int z);
void em_vmovdqu8_st_masked(em_t *e, int b, int x, int sc, int disp, int vs,
                          int vl, int k);
void em_kmovq_km(em_t *e, int k, int b, int x, int sc, int disp);
void em_vpor_mem(em_t *e, int vd, int s1, int b, int x, int sc, int disp,
                 int vl);
void em_vpermi2b(em_t *e, int vd_idx, int vtbl1, int vtbl2, int vl);
void em_vpand(em_t *e, int vd, int s1, int s2, int vl);
void em_vpsrlw(em_t *e, int vd, int s1, unsigned imm, int vl);
/* 512-bit only (GFNI): per-byte GF(2) affine transform y = A.x ^ b, one
 * independent 8x8-bit-matrix transform per byte lane (SDM: VGF2P8AFFINEQB).
 * s2 holds the matrix rows and imm8 the b vector; see gf_affine_matrix()
 * in jit.c for the constant layout (including the bit-reversal) that
 * turns this primitive into a GF(2^8) multiply-by-constant. */
void em_vgf2p8affineqb(em_t *e, int vd, int s1, int s2, unsigned imm, int vl);
void em_vpternlog(em_t *e, int vd, int s1, int s2, unsigned imm, int vl);
/* Memory-source form: vd = lut[vd, s1, mem] (dword elements, bitwise op).
 * imm=0x96 gives vd ^= s1 ^ mem. AVX-512 only. */
void em_vpternlog_mem(em_t *e, int vd, int s1, int b, int x, int sc, int disp,
                      unsigned imm, int vl);
void em_vzero(em_t *e, int vd, int vl); /* vd ^= vd */
void em_vzeroupper(em_t *e); /* VEX.128.0F 77: clear upper vector state */

/* --- legacy SSE encodings (no VEX; for the SSE ISA level) --- */
void em_movdqa_ld_sse(em_t *e, int vd, int b, int x, int sc, int disp);
void em_movdqa_st_sse(em_t *e, int b, int x, int sc, int disp, int vs);
/* Non-temporal store: 66 0F E7 /r. Same 16-byte alignment requirement as
 * MOVDQA; weakly ordered. The jitted code emits NO fence; a caller that
 * reads the NT-written data must execute SFENCE itself after return. */
void em_movntdq_st_sse(em_t *e, int b, int x, int sc, int disp, int vs);
void em_movdqa_rr_sse(em_t *e, int vd, int vs);
void em_pxor_sse(em_t *e, int vd, int vs); /* vd ^= vs */
void em_pshufb_sse(em_t *e, int vd, int vs);
void em_pand_sse(em_t *e, int vd, int vs);
void em_psrlw_sse(em_t *e, int vd, unsigned imm);
/* --- small integer extras --- */
void em_cmp_rr(em_t *e, int d, int s);   /* cmp r64, r64 */
void em_mov_r8r8(em_t *e, int d, int s); /* mov r8, r8 */
