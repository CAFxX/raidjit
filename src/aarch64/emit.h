/* raidjit AArch64 machine-code emitter (NEON + SVE/SVE2).
 *
 * Encodes a fixed subset of A64 + SVE instructions needed by the JIT.
 * Every encoding below was verified word-for-word against
 * aarch64-linux-gnu-as (see tests/test_aarch64_emit.c).
 *
 * Register conventions used by the JIT:
 *   x0..x2  incoming args (in[], out[], len), then x8=in, x9=out
 *   x10     pointer temp, x11 byte offset, x12/x13/x14 temps / VL / group
 *   x15     log-table base (scalar tail), x16 pool base, x17 exp-table base
 *   v0..    NEON accumulators/temps (v8..v15 saved/restored when touched)
 *   z0-z7,z16-z31 SVE accumulators/temps (logical reg r maps to z(16+r)
 *   for r<16, z(r-16) for r in [16,24); z8-z15 callee-saved, never used)
 *   p0      all-true predicate (SVE)
 */
#pragma once
#include <stddef.h>
#include <stdint.h>

/* GPR ids; 31 is XZR/SP depending on context */
enum {
    X0 = 0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15,
    X16, X17, X18, X19, X20, X21, X22, X23, X24, X25, X26, X27, X28, X29, X30
};
#define XZR 31
#define XSP 31 /* stack pointer (reg 31 in SP context) */

/* A64 condition codes */
enum {
    CC_EQ = 0, CC_NE = 1, CC_HS = 2, CC_LO = 3,
    CC_HI = 8, CC_LS = 9, CC_GE = 10, CC_LT = 11, CC_GT = 12, CC_LE = 13
};

/* PRFM prfop values (type/target/policy) */
#define PRF_PLDL1KEEP 0u
#define PRF_PLDL1STRM 1u

typedef struct {
    int lab;
    size_t at;
    int kind;
} fixup_t;

typedef struct {
    int *pos;          /* label -> byte offset, -1 = unbound */
    int n, cap;
    fixup_t *fx;
    int nfx, fxcap;
} em_labels_t;

typedef struct {
    uint8_t *buf;
    size_t len, cap;
    int err;
    em_labels_t lb;
} em_t;

void em_init(em_t *e, uint8_t *buf, size_t cap);
void em_free(em_t *e);
void em_byte(em_t *e, uint8_t v);
void em_word(em_t *e, uint32_t w);          /* one A64/SVE instruction */
void em_align(em_t *e, unsigned a);         /* NOP sled to 2^a alignment */
void em_trap_align(em_t *e, unsigned a);    /* UDF sled to 2^a alignment;
                                             * dead space after ret only */

int  em_label(em_t *e);
void em_bind(em_t *e, int lab);
void em_resolve(em_t *e);                   /* patch fixups; sets e->err */

/* control flow */
void em_ret(em_t *e);
void em_b(em_t *e, int lab);
void em_bcc(em_t *e, int cc, int lab);
void em_cbz(em_t *e, int rt, int lab);      /* 64-bit */
void em_cbz_w(em_t *e, int rt, int lab);    /* 32-bit */
void em_cbnz(em_t *e, int rt, int lab);     /* 64-bit */
void em_adr(em_t *e, int rd, int lab);      /* pc-relative, +/-1MB */
void em_udf(em_t *e);                       /* permanently undefined (trap sled) */

/* 64-bit integer */
void em_mov_rr(em_t *e, int rd, int rn);              /* orr xD,xzr,xN */
void em_movz(em_t *e, int rd, unsigned imm16);        /* movz xD,#imm */
void em_add_ri(em_t *e, int rd, int rn, unsigned imm); /* imm < 4096 */
void em_add_lsl12(em_t *e, int rd, int rn, unsigned imm); /* add xD,xN,#imm,lsl#12 */
void em_add_rr(em_t *e, int rd, int rn, int rm);
void em_sub_rr(em_t *e, int rd, int rn, int rm);
void em_subs_rr(em_t *e, int rd, int rn, int rm); /* subs xRd,xRn,xRm */
void em_subs_ri(em_t *e, int rd, int rn, unsigned imm);
void em_cmp_rr(em_t *e, int rn, int rm);              /* subs xzr,xN,xM */
void em_cmp_ri(em_t *e, int rn, unsigned imm);
void em_lsl_ri(em_t *e, int rd, int rn, unsigned sh);  /* ubfm alias, sh in 1..63 */
void em_lsr_ri(em_t *e, int rd, int rn, unsigned sh);  /* ubfm alias, sh in 1..63 */
void em_mul(em_t *e, int rd, int rn, int rm);
void em_udiv(em_t *e, int rd, int rn, int rm);
void em_ldr64(em_t *e, int rt, int rn, unsigned off);  /* off%8==0, off<32768 */
void em_str64_pre(em_t *e, int rt, int rn, int off);   /* str xRt,[xRn,#off]! */
void em_ldr64_post(em_t *e, int rt, int rn, int off);  /* ldr xRt,[xRn],#off */
void em_ldrb(em_t *e, int rt, int rn, int rm);         /* wRt=[xRn+xRm] */
void em_ldrb_ri(em_t *e, int rt, int rn, unsigned off);/* wRt=[xRn+#off], off<4096 */
void em_strb(em_t *e, int rt, int rn, int rm);         /* [xRn+xRm]=wRt */
void em_strb_ri(em_t *e, int rt, int rn, unsigned off);/* [xRn+#off]=wRt */

/* 32-bit integer (scalar tail) */
void em_mov_wr(em_t *e, int rd, int rn);              /* orr wD,wzr,wN */
void em_mov_wi(em_t *e, int rd, unsigned imm16);      /* movz wD,#imm */
void em_eor_w(em_t *e, int rd, int rn, int rm);
void em_add_wr_i(em_t *e, int rd, int rn, unsigned imm); /* imm < 4096 */
void em_sub_wr_i(em_t *e, int rd, int rn, unsigned imm);
void em_cmp_wr_i(em_t *e, int rn, unsigned imm);

/* prefetch: PRFM <prfop>, [xRn, #off]; off must be 8-aligned, off <= 32760.
 * The backend (jit_build_version) validates this and fails the twin
 * build cleanly on unencodable distances; the emitter itself just
 * encodes. */
void em_prfm(em_t *e, unsigned prfop, int rn, unsigned off);

/* NEON (all .16b unless noted) */
void em_neon_ldr(em_t *e, int qt, int rn, unsigned off); /* off%16==0, off<65536 */
void em_neon_str(em_t *e, int qt, int rn, unsigned off);
void em_neon_stnp(em_t *e, int qt1, int qt2, int rn, int off); /* off%16==0, -1024..1008 */
void em_neon_eor(em_t *e, int vd, int vn, int vm);
void em_neon_and(em_t *e, int vd, int vn, int vm);
void em_neon_orr(em_t *e, int vd, int vn, int vm);     /* vector copy */
void em_neon_ushr(em_t *e, int vd, int vn, unsigned sh); /* sh in 1..8 */
void em_neon_tbl(em_t *e, int vd, int vn, int vm);     /* vd = tbl({vn},vm) */
void em_neon_movi(em_t *e, int vd, unsigned imm8);
void em_neon_stp_pre(em_t *e, int qt1, int qt2, int rn, int off);  /* [xRn,#off]! */
void em_neon_ldp_post(em_t *e, int qt1, int qt2, int rn, int off); /* [xRn],#off */

/* SVE (.b; pg is a P register id, zdn/z physical ids) */
void em_sve_whilelo(em_t *e, int pd, int rn, int rm);  /* whilelo pd.b, xRn, xRm */
void em_sve_whilelt(em_t *e, int pd, int rn, int rm);  /* whilelt pd.b, xRn, xRm */
void em_sve_ptrue(em_t *e, int pd);                    /* ptrue pd.b, all */
void em_sve_cntb(em_t *e, int xd);                     /* cntb xD, all, mul #1 */
void em_sve_ld1b(em_t *e, int zt, int pg, int rn);     /* ld1b zt.b, pg/z, [xRn] */
void em_sve_ld1b_off(em_t *e, int zt, int pg, int rn, int off); /* ld1b zt.b, pg/z, [xRn, #off, MUL VL]; off in -8..7 */
void em_sve_ld1rqb(em_t *e, int zt, int pg, int rn);   /* replicate 16B across VL */
void em_sve_st1b(em_t *e, int zt, int pg, int rn);
void em_sve_st1b_off(em_t *e, int zt, int pg, int rn, int off); /* st1b zt.b, pg, [xRn, #off, MUL VL]; off in -8..7 */
void em_sve_stnt1b(em_t *e, int zt, int pg, int rn);   /* non-temporal store */
void em_sve_stnt1b_off(em_t *e, int zt, int pg, int rn, int off); /* stnt1b zt.b, pg, [xRn, #off, MUL VL]; off in -8..7 */
void em_sve_eor(em_t *e, int zdn, int pg, int zm);     /* destructive: zdn^=zm */
void em_sve_eor_unpred(em_t *e, int zd, int zn, int zm); /* eor zd.d,zn.d,zm.d, unpredicated */
void em_sve_eor3(em_t *e, int zdn, int zm, int za);   /* SVE2: zdn^=zm^za */
void em_sve_and(em_t *e, int zdn, int pg, int zm);
void em_sve_and_unpred(em_t *e, int zd, int zn, int zm); /* and zd.d,zn.d,zm.d, unpredicated */
void em_sve_orr(em_t *e, int zdn, int pg, int zm);      /* destructive: zDn|=zM */
void em_sve_mov(em_t *e, int zd, int zn);               /* zD.d = zN.d, true copy */
void em_sve_lsr(em_t *e, int zdn, int pg, unsigned sh);/* sh in 1..7 */
void em_sve_lsr_unpred(em_t *e, int zd, int zn, unsigned sh); /* lsr zd.b,zn.b,#sh, unpredicated; sh in 1..7 */
void em_sve_tbl(em_t *e, int zd, int zn, int zm);      /* zd = tbl({zn},zm) */
void em_sve_dup(em_t *e, int zd, int imm8);            /* dup zd.b, #imm8 (signed) */
