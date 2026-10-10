/* Code generation: plan + candidate -> executable machine code.
 *
 * Generated function (System V):
 *   void fn(const void *in[], void *out[], size_t len);
 *   rdi = in, rsi = out, rdx = len
 *
 * Layout: vectorized main loop over groups of U*VL bytes, then a scalar
 * tail for len % (U*VL) bytes. All GF constants live in a constant pool
 * after the code, addressed via r15 (set once in the prologue).
 */
#include "jit.h"
#include "emit.h"
#include "gf.h"
#include "cpu.h"
#include "xorsched.h"
#include "fused.h" /* fused_crc_interleave_ok: the crc_fold uarch gate */

#include <sys/mman.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <assert.h>

/* constant-pool entry kinds */
enum { CK_NIBLO, CK_NIBHI, CK_AFF, CK_MASK, CK_LOG, CK_EXP,
         CK_NIBBOTH, CK_OR10, CK_TAILMASK, CK_ONE, CK_MSHIFT, CK_CRCFOLD };

typedef struct {
    em_t em;
    const plan_t *pl;
    cand_t cand;
    const rj_options *opts;
    int vl, U, group;
    int pool_lab;
    uint8_t *pool;
    size_t pool_len, pool_cap;
    struct {
        int kind;
        uint8_t c;
        size_t off;
    } *cent;
    int ncent, ccent;
    /* register assignment */
    int t, tlo, thi, r1, r2, tbl, tbl2, t2, mask;
    /* REDUX crc_fold register pool (cand->crc_fold only; -1 otherwise).
     * The fold needs N+5 128-bit lanes, disjoint from the encode's
     * 0..E-1 (E = encode regs used): 5 VEX-bound temps/consts at E..E+4
     * (VEX.128 PCLMULQDQ addresses only xmm0-15, so the backend rejects
     * E > 11 cleanly) followed by the N stream states at fold_xs0..E+N+4
     * (EVEX-only access: em_vpxor_evex128 / em_vmovdqa_ld/st_evex128).
     * This is bottom-up continuation, not the spec's top-down zmm31
     * placement: the backend's vmax+1 == plan_need_regs lockstep assert
     * forces the fold pool to extend the encode's 0-based assignment,
     * and VEX cannot address the spec's zmm_{32-(N+5)}..zmm31 anyway.
     * Disjointness, determinism, exact N+5 accounting and fail-closed
     * filtering are preserved; see the backend gate below. */
    int fold_xc, fold_xa, fold_xb, fold_xk1, fold_xk2; /* E..E+4 */
    int fold_xs0;  /* first stream-state reg (E+5); N = n_in + n_out states */
    int fold_n;
    /* VPMULTISHIFTQB high-nibble control (track I2, mshift twin): the
     * loop-invariant per-lane shift counts (byte j of each 64-bit lane
     * holds 8*j+4). Register-resident: VPMULTISHIFTQB's control is
     * EVEX.vvvv with no memory form. Allocated only when
     * cand->mshift && need_mask (0 = unallocated). */
    int msh;
    int *hreg; /* HORNER: 2 regs per row (nibble) or 1 (gfni), -1 if unused */
    /* Bit-program (xorbit) twin registers: input-byte reg, two bit-temp
     * regs, and the 0x01 extract mask. Assigned as n_out*U + {0..3}
     * (see xorbit_need_regs); the nibble tables are dead on this path. */
    int xbt_in, xbt_r1, xbt_r2, xbt_mask;
    /* Software-pipelined loop twin (track I8, cand->pipe): base vector
     * regs of the per-column double-buffered datasets D (pipe_d, U
     * regs) and Dn (pipe_dn, U regs). In pipe mode the single input
     * temp t is dead (replaced by D/Dn). -1 when pipe is off. */
    int pipe_d, pipe_dn;
    /* MULACC/DELTA GFNI table-load dedup (pure JIT-time state): the affine
     * table in tbl depends only on the coefficient c = mat[j][i], so it is
     * loop-invariant across the u-loop and across input columns i; nothing
     * else writes tbl on the GFNI path. Track the last-loaded pool offset
     * and skip redundant c_ldpool calls (cf. P_HORNER remat tbl_pool_off).
     * pool_get() dedups by (kind, c) with append-only offsets, so offset
     * comparison is sound. Lives in cg_t (not a gen_body local) because
     * mulacc_column is shared by the P_MULACC and P_DELTA paths. Skipping
     * the call also skips its uop/mem accounting. */
    size_t mulacc_pool_off;
    int mulacc_pool_valid;
    /* PDELTA nibble NIBHI pinning (item 7, AVX512 only): tbl2 holds the
     * NIBHI table for the last-used coefficient c. tbl is reloaded with
     * NIBLO on every use (scratch); tbl2 is pinned, so the NIBHI pool
     * load is skipped when c matches the last use. Saves 1 load uop
     * per (i,u,j) when c is stable. tbl2 is written only here, never
     * clobbered elsewhere on the PDELTA AVX512 nibble path. */
    uint8_t mulacc_nib_c;
    int mulacc_nib_valid;
    /* static cost model: per one group iteration */
    double uops, mem;
    int need_logexp; /* scalar tail needs log/exp tables */
    int need_mask;   /* nibble vector path needs the 0x0F mask */
    /* Pointer hoisting: number of in[] pointers held in registers for
     * the whole call (0 = disabled). in[k] is loop-invariant, so the
     * prologue loads it once into HH[k] instead of reloading [R8+k*8]
     * every group iteration. */
    int nhoist;
    int nt;          /* cand->nt: non-temporal vector output stores */
    size_t prefetch_dist; /* cand->prefetch_dist: PREFETCHT0 distance, 0=off */
    int notail;      /* length-dispatch twin: omit the scalar tail entirely;
                      * only valid for len % group == 0 (see
                      * docs/length_dispatch.md) */
    int maskedtail;  /* length-dispatch twin: masked AVX-512 vector tail
                      * (track B1x86). Disjoint from notail. */
    int permi2b;     /* VPERMI2B nibble twin (track B2) */
    uint8_t logtab[256], exptab[512];
    int vpternlogd_emitted; /* ternlog twin: count of VPTERNLOGD insns
                             * emitted. If cand->ternlog was requested but
                             * zero were emitted, the pin was a no-op and
                             * jit_build_version rejects it (phantom twin
                             * fix, BUG_t1_80). */
} cg_t;

static int is_sse(const cg_t *cg) { return cg->cand.isa == ISA_SSE; }
static int is_avx512(const cg_t *cg)
{
    return cg->cand.isa == ISA_AVX512;
}

/* ---------------- pointer hoisting ---------------- */

/* The first PTR_NHOIST in[] pointers are loop-invariant, so the prologue
 * loads them once into HH[] instead of the body reloading [R8+k*8] every
 * group iteration. Measured +2.2% on cache-resident RAID-6 encode
 * (AVX-512, core-bound); ~0% when DRAM-bound; never negative.
 * P_DELTA is excluded: its [old,new,oldpar] pointer layout needs its own
 * design (follow-up). */
/* Max input pointers held in registers. HH[0..1] are caller-saved
 * (RSI/RDI: dead in the vector body -- the scalar tail sets them up
 * itself); HH[2..4] are callee-saved R12-R14 (pushed/popped by the
 * prologue/epilogue when used). */
#define PTR_NHOIST 5
static const int HH[PTR_NHOIST] = { RSI, RDI, R12, R13, R14 };

/* How many in[] pointers to hoist for this plan. */
static int ptr_hoist_count(const cg_t *cg)
{
    const plan_t *pl = cg->pl;
    if (pl->noop || pl->kind == P_DELTA || pl->n_in < 2)
        return 0;
    return pl->n_in < PTR_NHOIST ? pl->n_in : PTR_NHOIST;
}

/* Register holding in[i]'s pointer, or -1 if i is not hoisted. */
static int hoist_inptr(const cg_t *cg, int i)
{
    if (i < 0 || i >= cg->nhoist)
        return -1;
    return HH[i];
}

/* ---------------- constant pool ---------------- */

static size_t pool_get(cg_t *cg, int kind, uint8_t c)
{
    for (int i = 0; i < cg->ncent; i++)
        if (cg->cent[i].kind == kind && cg->cent[i].c == c)
            return cg->cent[i].off;
    uint8_t tmp[1024]; /* CK_TAILMASK needs 65*8 = 520 */
    size_t len = 0;
    if (kind == CK_NIBLO || kind == CK_NIBHI) {
        uint8_t lo[16], hi[16];
        gf_build_nibble(c, lo, hi);
        uint8_t *src = (kind == CK_NIBLO) ? lo : hi;
        for (int r = 0; r < cg->vl / 16; r++) {
            memcpy(tmp + len, src, 16);
            len += 16;
        }
    } else if (kind == CK_AFF) {
        uint64_t m = gf_affine_matrix(c);
        for (int r = 0; r < 8; r++) {
            memcpy(tmp + len, &m, 8);
            len += 8;
        }
    } else if (kind == CK_NIBBOTH) {
        /* Interleaved nibble tables for the VPERMI2B twin (track B2):
         * [NIBLO(16B), NIBHI(16B)] x2. One 64B load serves both lookups:
         * vpermi2b(tlo, tbl, tbl) indexes bytes [0,16) = NIBLO, and
         * vpermi2b(thi|0x10, tbl, tbl) indexes bytes [16,32) = NIBHI
         * (bit 6 = 0 selects the first table operand in both cases). */
        uint8_t lo[16], hi[16];
        gf_build_nibble(c, lo, hi);
        for (int r = 0; r < 2; r++) {
            memcpy(tmp + len, lo, 16);
            len += 16;
            memcpy(tmp + len, hi, 16);
            len += 16;
        }
    } else if (kind == CK_OR10) {
        /* 64 bytes of 0x10: the high-nibble index offset for the
         * VPERMI2B twin, applied via memory-source VPORB (track B2). */
        memset(tmp, 0x10, 64);
        len = 64;
    } else if (kind == CK_TAILMASK) {
        /* 65 qwords of tail masks for the masked-tail twin (track B1x86):
         * mask[i] = (1ULL << i) - 1, i.e. the low i bits set; mask[64] =
         * ~0ULL. The tail chunk loop loads mask[rdx] with KMOVQ for a
         * chunk of rdx bytes (1 <= rdx <= 64; mask[0] = 0 is never
         * used). */
        for (int i = 0; i <= 64; i++) {
            uint64_t m = (i == 64) ? ~0ULL
                         : (i == 0) ? 0
                                    : ((1ULL << i) - 1);
            memcpy(tmp + len, &m, 8);
            len += 8;
        }
    } else if (kind == CK_MASK) {
        memset(tmp, 0x0F, (size_t)cg->vl);
        len = (size_t)cg->vl;
    } else if (kind == CK_ONE) {
        /* Vector of 0x01 bytes: the bit-extract mask for the bit-program
         * (xorbit) emitter's ((in_i >> k) & 1) materialization. */
        memset(tmp, 0x01, (size_t)cg->vl);
        len = (size_t)cg->vl;
    } else if (kind == CK_MSHIFT) {
        /* VPMULTISHIFTQB high-nibble control (track I2): per 64-bit lane,
         * byte j holds the shift count 8*j+4, so output byte k (lane
         * q, lane-relative j = k-8q) is {(t[k+1] & 0x0F), (t[k] >> 4)}:
         * bits [8j+11:8j+4] of the lane. The trailing 0x0F AND stays
         * (bits [7:4] carry the neighbor nibble; VPSHUFB zeroes its lane
         * when the index bit 7 is set). One entry serves every vl (the
         * 8-byte lane pattern repeats). */
        for (int r = 0; r < cg->vl / 8; r++)
            for (int j = 0; j < 8; j++)
                tmp[len++] = (uint8_t)(8 * j + 4);
    } else if (kind == CK_LOG) {
        memcpy(tmp, cg->logtab, 256);
        len = 256;
    } else if (kind == CK_CRCFOLD) {
        /* REDUX crc_fold multiply constants: 16 bytes holding the 64-bit
         * constant in both lanes ({K,K}, little-endian), so imm=0x00 and
         * imm=0x11 both read the right half. c=0 -> K1 = 0xae689191
         * (x^-128 mod QREF), c=1 -> K2 = 0xccaa009e (x^-64 mod QREF).
         * Loaded once at kernel entry into fold_xk1/xk2 (hoisted). */
        uint64_t k = (c == 0) ? 0xae689191ULL : 0xccaa009eULL;
        memcpy(tmp, &k, 8);
        memcpy(tmp + 8, &k, 8);
        len = 16;
    } else { /* CK_EXP */
        memcpy(tmp, cg->exptab, 256);
        len = 256;
    }
    size_t off = (cg->pool_len + 63) & ~(size_t)63;
    if (off + len > cg->pool_cap) {
        cg->pool_cap = (off + len + 4095) & ~(size_t)4095;
        cg->pool = realloc(cg->pool, cg->pool_cap);
        assert(cg->pool);
    }
    memcpy(cg->pool + off, tmp, len);
    cg->pool_len = off + len;
    if (cg->ncent == cg->ccent) {
        cg->ccent = cg->ccent ? cg->ccent * 2 : 32;
        cg->cent = realloc(cg->cent, (size_t)cg->ccent * sizeof(*cg->cent));
        assert(cg->cent);
    }
    cg->cent[cg->ncent].kind = kind;
    cg->cent[cg->ncent].c = c;
    cg->cent[cg->ncent].off = off;
    cg->ncent++;
    return off;
}

/* ---------------- cost-counting emit wrappers ---------------- */

static void c_ld(cg_t *cg, int vd, int b, int x, int sc, int disp)
{
    em_t *e = &cg->em;
    if (is_sse(cg))
        em_movdqa_ld_sse(e, vd, b, x, sc, disp);
    else
        em_vmovdqa_ld(e, vd, b, x, sc, disp, cg->vl);
    cg->uops += 1;
    cg->mem += cg->vl;
}

/* pool load [r15+off]; counted (L1 traffic, honest uop cost) */
static void c_ldpool(cg_t *cg, int vd, size_t off)
{
    em_t *e = &cg->em;
    assert(vd >= 0); /* catches the remat hreg=-1 hazard at codegen time */
    if (is_sse(cg))
        em_movdqa_ld_sse(e, vd, R15, -1, 0, (int)off);
    else
        em_vmovdqa_ld(e, vd, R15, -1, 0, (int)off, cg->vl);
    cg->uops += 1;
    cg->mem += cg->vl;
}

/* prologue (pre-loop) pool load: not counted, amortized away */
static void raw_ldpool(cg_t *cg, int vd, size_t off)
{
    em_t *e = &cg->em;
    assert(vd >= 0); /* catches the remat hreg=-1 hazard at codegen time */
    if (is_sse(cg))
        em_movdqa_ld_sse(e, vd, R15, -1, 0, (int)off);
    else
        em_vmovdqa_ld(e, vd, R15, -1, 0, (int)off, cg->vl);
}

/* ---------------- REDUX crc_fold helpers ----------------
 * Per-stream 128-bit fold (spec hidden_files/crc_fold_v2_SPEC.md §2):
 *   W  = S XOR C            (C = 16-byte message chunk)
 *   S' = CLMUL(W[63:0], K1) XOR CLMUL(W[127:64], K2)
 * States live in fold_xs0.. (one xmm per stream, whole kernel);
 * xc/xa/xb/xk1/xk2 are the shared temps/consts. Only called when
 * cand->crc_fold (implies ISA_AVX512; the backend gate enforces it). */

/* Fold one 16B chunk already in fold_xc into stream s's state. */
static void gen_fold_chunk(cg_t *cg, int s)
{
    em_t *e = &cg->em;
    int xs = cg->fold_xs0 + s;
    assert(cg->cand.crc_fold && is_avx512(cg));
    em_vpxor_evex128(e, cg->fold_xc, cg->fold_xc, xs); /* W = C ^ S */
    em_pclmulqdq(e, cg->fold_xa, cg->fold_xc, cg->fold_xk1, 0x00);
    em_pclmulqdq(e, cg->fold_xb, cg->fold_xc, cg->fold_xk2, 0x11);
    em_vpxor_evex128(e, xs, cg->fold_xa, cg->fold_xb); /* S' = A ^ B */
    cg->uops += 4;
}

/* Fold the 64B vector in zmm zreg into stream s (4 x 16B chunks). */
static void gen_fold_zmm(cg_t *cg, int zreg, int s)
{
    em_t *e = &cg->em;
    assert(cg->cand.crc_fold && is_avx512(cg));
    for (int k = 0; k < 4; k++) {
        em_vextracti32x4(e, cg->fold_xc, zreg, (unsigned)k);
        gen_fold_chunk(cg, s);
    }
    cg->uops += 4; /* 4 extracts; the chunks' uops counted above */
}

/* Fold 64B at [b + x*sc + disp] into stream s (4 x 16B loads).
 * Used where the body consumes an input vector through a memory operand
 * (no zmm materialization) or skips an unused input column: the bytes
 * are in memory either way, so the fold observes exactly what the
 * encode consumes. */
static void gen_fold_mem(cg_t *cg, int b, int x, int sc, int disp, int s)
{
    em_t *e = &cg->em;
    assert(cg->cand.crc_fold && is_avx512(cg));
    for (int k = 0; k < 4; k++) {
        em_vmovdqa_ld_evex128(e, cg->fold_xc, b, x, sc, disp + k * 16);
        gen_fold_chunk(cg, s);
    }
    cg->uops += 4; /* 4 loads; the chunks' uops counted in the helper */
    cg->mem += 64;
}

static void c_st(cg_t *cg, int b, int x, int sc, int disp, int vs)
{
    em_t *e = &cg->em;
    if (is_sse(cg)) {
        if (cg->nt)
            em_movntdq_st_sse(e, b, x, sc, disp, vs);
        else
            em_movdqa_st_sse(e, b, x, sc, disp, vs);
    } else if (cg->nt) {
        em_vmovntdq_st(e, b, x, sc, disp, vs, cg->vl);
    } else {
        em_vmovdqa_st(e, b, x, sc, disp, vs, cg->vl);
    }
    cg->uops += 1;
    cg->mem += cg->vl;
}

/* Software prefetch for an input stream: PREFETCHT0 [ptr_reg + R11 + dist].
 * Call right after the input pointer is loaded into ptr_reg. No-op when
 * prefetching is disabled. One prefetch per stream per group iteration;
 * the benchmark decides whether the extra uops pay off. */
static void c_prefetch_in(cg_t *cg, int ptr_reg)
{
    if (cg->prefetch_dist) {
        em_prefetch(&cg->em, 1 /*T0*/, ptr_reg, R11, 0,
                    (int)cg->prefetch_dist);
        cg->uops += 1;
    }
}

/* Body pointer load: in[i]'s pointer into a register. Returns the hoisted
 * register when i < cg->nhoist, otherwise emits the legacy per-iteration
 * `mov tmp,[R8+i*8]` (accounted in the static model: 1 uop, 8 mem bytes;
 * the hoisted path costs nothing per iteration -- the prologue loads are
 * amortized over the trip count). */
static int body_inptr(cg_t *cg, int i, int tmp)
{
    int pr = hoist_inptr(cg, i);
    if (pr >= 0)
        return pr;
    em_mov_rm(&cg->em, tmp, R8, -1, 0, i * 8);
    cg->uops += 1;
    cg->mem += 8;
    return tmp;
}

static void c_vpxor(cg_t *cg, int vd, int s1, int s2)
{
    em_t *e = &cg->em;
    if (is_sse(cg)) {
        if (vd != s1) {
            em_movdqa_rr_sse(e, vd, s1);
            cg->uops += 1;
        }
        em_pxor_sse(e, vd, s2);
    } else {
        em_vpxor(e, vd, s1, s2, cg->vl);
    }
    cg->uops += 1;
}

/* vd ^= s1 ^ s2 via VPTERNLOGD (AVX-512 only). 0x96 is the 3-input XOR
 * truth table. Saves one uop vs two VPXORDs but keeps s1+s2 live. */
static void c_vpternlog(cg_t *cg, int vd, int s1, int s2)
{
    em_t *e = &cg->em;
    assert(cg->cand.isa == ISA_AVX512);
    em_vpternlog(e, vd, s1, s2, 0x96, cg->vl);
    cg->uops += 1;
    cg->vpternlogd_emitted++;
}

/* vd ^= s1 ^ [mem] via VPTERNLOGD (AVX-512 only). Counted as one uop
 * in the static model (same convention as c_vpxor_mem: one vector
 * instruction with a folded memory operand); the memory operand folds
 * the second input's load into the ternary logic op, so this costs the
 * same model-uops as VPXORD-with-memory but spends one ALU uop where
 * two VPXORDs would spend two. The 3-input XOR is symmetric, so 0x96
 * is correct regardless of which operand is the memory one. */
static void c_vpternlog_mem(cg_t *cg, int vd, int s1, int b, int x, int sc,
                           int disp)
{
    em_t *e = &cg->em;
    assert(cg->cand.isa == ISA_AVX512);
    em_vpternlog_mem(e, vd, s1, b, x, sc, disp, 0x96, cg->vl);
    cg->uops += 1;
    cg->mem += cg->vl;
    cg->vpternlogd_emitted++;
}

/* Folded XOR-with-memory (one uop on every x86 ISA: pxor m128 on SSE,
 * vpxor/vpxord with a memory operand on AVX/AVX-512). The static model
 * in src/xorsched.c counts this as 1 uop on all three ISAs; keep in
 * lockstep (item 6: the SSE folded form was re-introduced together
 * with the model update). */
static void c_vpxor_mem(cg_t *cg, int vd, int vs1, int b, int x, int sc,
                        int disp)
{
    assert(vd == vs1); /* all our uses accumulate */
    if (is_sse(cg)) {
        /* PXOR-with-memory is valid legacy SSE (66 0F EF /r): fold the
         * load into the xor, one uop instead of movdqa+pxor's two. */
        em_pxor_mem_sse(&cg->em, vd, b, x, sc, disp);
        cg->uops += 1;
        cg->mem += cg->vl;
    } else {
        em_t *e = &cg->em;
        em_vpxor_mem(e, vd, vs1, b, x, sc, disp, cg->vl);
        cg->uops += 1;
        cg->mem += cg->vl;
    }
}

static void c_shuf(cg_t *cg, int vd, int s1, int s2)
{
    em_t *e = &cg->em;
    if (is_sse(cg)) {
        if (vd != s1) {
            em_movdqa_rr_sse(e, vd, s1);
            cg->uops += 1;
        }
        em_pshufb_sse(e, vd, s2);
    } else {
        em_vpshufb(e, vd, s1, s2, cg->vl);
    }
    cg->uops += 1;
}

static void c_and(cg_t *cg, int vd, int s1, int s2)
{
    em_t *e = &cg->em;
    if (is_sse(cg)) {
        if (vd != s1) {
            em_movdqa_rr_sse(e, vd, s1);
            cg->uops += 1;
        }
        em_pand_sse(e, vd, s2);
    } else {
        em_vpand(e, vd, s1, s2, cg->vl);
    }
    cg->uops += 1;
}

static void c_srlw(cg_t *cg, int vd, int s1, unsigned imm)
{
    em_t *e = &cg->em;
    if (is_sse(cg)) {
        if (vd != s1) {
            em_movdqa_rr_sse(e, vd, s1);
            cg->uops += 1;
        }
        em_psrlw_sse(e, vd, imm);
    } else {
        em_vpsrlw(e, vd, s1, imm, cg->vl);
    }
    cg->uops += 1;
}

/* High-nibble extract via VPMULTISHIFTQB (track I2, AVX-512 only):
 * vd[k] = (s1's lane >> vctrl[k]) & 0xFF per 64-bit lane, with the
 * loop-invariant vctrl holding 8*j+4 per lane-relative byte j. One uop
 * like VPSRLW, but issued to the shuffle port; the caller still applies
 * the trailing 0x0F AND (bits [7:4] of the output carry the neighbor
 * nibble, and VPSHUFB zeroes its lane when the index bit 7 is set). */
static void c_mshift(cg_t *cg, int vd, int s1, int vctrl)
{
    assert(is_avx512(cg));
    em_vpmultishiftqb(&cg->em, vd, vctrl, s1, cg->vl);
    cg->uops += 1;
}

/* Nibble split of vsrc into vd_lo/vd_hi: vd_lo = vsrc & 0x0F,
 * vd_hi = (vsrc >> 4) & 0x0F. The I2 (mshift) twin extracts the high
 * nibble with VPMULTISHIFTQB instead of VPSRLW (AVX-512 only); the two
 * ANDs are unchanged, so the split stays 3 uops either way. Single
 * home for the three body sites below (P_MULACC/P_DELTA/P_HORNER). */
static void nibble_split(cg_t *cg, int vd_lo, int vd_hi, int vsrc)
{
    c_and(cg, vd_lo, vsrc, cg->mask);
    if (cg->cand.mshift)
        c_mshift(cg, vd_hi, vsrc, cg->msh);
    else
        c_srlw(cg, vd_hi, vsrc, 4);
    c_and(cg, vd_hi, vd_hi, cg->mask);
}

/* Word shift left by immediate. Two-operand on SSE (mov when vd != s1),
 * three-operand on AVX. Used by the bit-program emitter to move a
 * bit-0-normalized bit value to its output bit position. */
static void c_sllw(cg_t *cg, int vd, int s1, unsigned imm)
{
    em_t *e = &cg->em;
    if (is_sse(cg)) {
        if (vd != s1) {
            em_movdqa_rr_sse(e, vd, s1);
            cg->uops += 1;
        }
        em_psllw_sse(e, vd, imm);
    } else {
        em_vpsllw(e, vd, s1, imm, cg->vl);
    }
    cg->uops += 1;
}

static void c_aff(cg_t *cg, int vd, int s1, int s2)
{
    assert(!is_sse(cg));
    em_vgf2p8affineqb(&cg->em, vd, s1, s2, 0, cg->vl);
    cg->uops += 1;
}

static void c_vzero(cg_t *cg, int vd)
{
    em_t *e = &cg->em;
    if (is_sse(cg))
        em_pxor_sse(e, vd, vd);
    else
        em_vzero(e, vd, cg->vl);
    cg->uops += 1;
}

/* Vector register copy. Used to fold "xor with a compile-time-known
 * zero" (first-touch accumulators, Horner c==0) into a single move.
 * A self-move is a no-op and emits nothing. */
static void c_vmov(cg_t *cg, int vd, int vs)
{
    em_t *e = &cg->em;
    if (vd == vs)
        return;
    if (is_sse(cg))
        em_movdqa_rr_sse(e, vd, vs);
    else
        em_vmovdqa_rr(e, vd, vs, cg->vl);
    cg->uops += 1;
}

/* ---------------- codegen ---------------- */

/* REDUX crc_fold entry: load the N stream states from fold_state and
 * init the hoisted K1/K2 constants. Called from gen_const_setup before
 * the tile loop. On entry [RSP] holds the fold_state pointer (pushed by
 * gen_function's prologue); R15 is the constant-pool base (need_r15 is
 * forced on for fold kernels). Between entry and exit the states live
 * only in the fold_xs0.. xmms. */
static void gen_fold_setup(cg_t *cg)
{
    em_t *e = &cg->em;
    assert(cg->cand.crc_fold && is_avx512(cg));
    /* R10 = fold_state (16-byte aligned per the driver contract). */
    em_mov_rm(e, R10, RSP, -1, 0, 0);
    for (int s = 0; s < cg->fold_n; s++)
        em_vmovdqa_ld_evex128(e, cg->fold_xs0 + s, R10, -1, 0, s * 16);
    /* Hoisted constants: xk1 = {K1,K1}, xk2 = {K2,K2} (both lanes, so
     * imm=0x00 and imm=0x11 both read the right half). */
    em_vmovdqa_ld_evex128(e, cg->fold_xk1, R15, -1, 0,
                          (int)pool_get(cg, CK_CRCFOLD, 0));
    em_vmovdqa_ld_evex128(e, cg->fold_xk2, R15, -1, 0,
                          (int)pool_get(cg, CK_CRCFOLD, 1));
}

static void gen_const_setup(cg_t *cg)
{
    const plan_t *pl = cg->pl;
    if (cg->cand.crc_fold)
        gen_fold_setup(cg);
    if (cg->cand.xorbit) {
        /* Bit-program twin: the 0x01 bit-extract mask (the nibble
         * tables are dead on this path). */
        raw_ldpool(cg, cg->xbt_mask, pool_get(cg, CK_ONE, 0));
        return;
    }
    if ((pl->kind == P_MULACC || pl->kind == P_DELTA) &&
        cg->cand.mul == MUL_NIBBLE) {
        /* With all coefficients in {0,1} (P_DELTA with an all-{0,1} delta
         * matrix; P_MULACC only when hand-built) the nibble split never
         * executes, so the mask load, its pool entry, and the mask
         * register itself are dead. */
        if (cg->need_mask)
            raw_ldpool(cg, cg->mask, pool_get(cg, CK_MASK, 0));
    } else if (pl->kind == P_HORNER) {
        if (cg->cand.mul == MUL_NIBBLE) {
            /* The nibble split needs the 0x0F mask; with all coefficients
             * in {0,1} the split never executes and the load is dead. */
            if (cg->need_mask)
                raw_ldpool(cg, cg->mask, pool_get(cg, CK_MASK, 0));
            /* In remat mode hreg[] stays -1 (no persistent table regs);
             * tables are reloaded into tbl on each use instead. */
            if (!cg->cand.remat) {
                for (int j = 0; j < pl->n_out; j++) {
                    uint8_t c = pl->horner_c[j];
                    if (c == 0 || c == 1)
                        continue;
                    raw_ldpool(cg, cg->hreg[2 * j],
                               pool_get(cg, CK_NIBLO, c));
                    raw_ldpool(cg, cg->hreg[2 * j + 1],
                               pool_get(cg, CK_NIBHI, c));
                }
            }
        } else if (!cg->cand.remat) {
            for (int j = 0; j < pl->n_out; j++) {
                uint8_t c = pl->horner_c[j];
                if (c == 0 || c == 1)
                    continue;
                raw_ldpool(cg, cg->hreg[2 * j], pool_get(cg, CK_AFF, c));
            }
        }
    }
    if (cg->cand.mshift && cg->need_mask) {
        /* VPMULTISHIFTQB high-nibble control (track I2): loop-invariant,
         * loaded once into the msh register. Dead when the nibble split
         * never executes (need_mask == 0), like the 0x0F mask. */
        raw_ldpool(cg, cg->msh, pool_get(cg, CK_MSHIFT, 0));
    }
}

/* Scan column `col` of an n_out x ncols row-major matrix: is it all-zero?
 * Does any entry need the nibble tables (c != 0,1)? */
static void mulacc_col_info(const plan_t *pl, int col, int ncols,
                            int *any, int *neednib)
{
    int a = 0, nn = 0;
    for (int j = 0; j < pl->n_out; j++) {
        uint8_t c = pl->mat[(size_t)j * (size_t)ncols + (size_t)col];
        if (c) {
            a = 1;
            if (c != 1)
                nn = 1;
        }
    }
    *any = a;
    *neednib = nn;
}

/* Multiply-accumulate one matrix column into the accumulators.
 * cg->t (and cg->tlo/cg->thi on the nibble path) must already hold the
 * per-lane multiplicand; mat is n_out x ncols row-major.
 * first_col[j] is the first column with a nonzero entry in row j, or -1;
 * pass NULL when accumulators are not provably zero (P_DELTA seeds from
 * oldpar). At the first contributing column the accumulator is known
 * zero, so the update is emitted as a move/load instead of an
 * xor-with-zero. */
static void mulacc_column(cg_t *cg, int col, int ncols, int u,
                          const int *first_col)
{
    const plan_t *pl = cg->pl;
    int n_out = pl->n_out, U = cg->U;
    int gfni = cg->cand.mul == MUL_GFNI;
    for (int j = 0; j < n_out; j++) {
        uint8_t c = pl->mat[(size_t)j * (size_t)ncols + (size_t)col];
        if (!c)
            continue;
        int acc = j * U + u;
        int first = first_col && col == first_col[j];
        if (c == 1) {
            if (first)
                c_vmov(cg, acc, cg->t); /* acc was 0 */
            else
                c_vpxor(cg, acc, acc, cg->t);
        } else if (gfni) {
            /* The affine table depends only on c, so it is loop-invariant
             * across u and i: skip the pool load if tbl already holds it
             * (tracked in cg->mulacc_pool_off). The nibble branch below
             * ping-pongs tbl per (i,j,u) and cannot use this (F3). */
            size_t poff = pool_get(cg, CK_AFF, c);
            if (!cg->mulacc_pool_valid || cg->mulacc_pool_off != poff) {
                c_ldpool(cg, cg->tbl, poff);
                cg->mulacc_pool_off = poff;
                cg->mulacc_pool_valid = 1;
            }
            c_aff(cg, cg->t2, cg->t, cg->tbl);
            if (first)
                c_vmov(cg, acc, cg->t2); /* acc was 0 */
            else
                c_vpxor(cg, acc, acc, cg->t2);
        } else if (cg->permi2b) {
            /* VPERMI2B twin (track B2): 2-table nibble lookup. The two
             * 16-entry tables travel interleaved in one 64-byte pool entry
             * (CK_NIBBOTH = [lo16,hi16]×2). lo indices (tlo, 0-15) select
             * the lo table; hi indices are ORed with 0x10 (via memory-source
             * VPORB with the CK_OR10 pool entry) to select the hi table at
             * bytes 16-31. VPERMI2B overwrites its index register, so copy
             * tlo/thi to r1/r2 first. Static uops: 1 load + 1 vmov + 1 vporb
             * + 2 vpermi2b + 1-2 xor = 6-7, vs 2 loads + 2 shuf + 1-2 xor
             * = 5-6 for the VPSHUFB path (the model says the twin is
             * marginally worse; measurement decides anyway). */
            assert(cg->cand.isa == ISA_AVX512);
            c_ldpool(cg, cg->tbl, pool_get(cg, CK_NIBBOTH, c));
            c_vmov(cg, cg->r1, cg->tlo);
            em_vpermi2b(&cg->em, cg->r1, cg->tbl, cg->tbl, cg->vl);
            cg->uops += 1;
            /* r2 = thi | 0x10 (memory-source VPORB, no extra register) */
            size_t or10off = pool_get(cg, CK_OR10, 0);
            em_vpor_mem(&cg->em, cg->r2, cg->thi, R15, -1, 0, (int)or10off,
                        cg->vl);
            cg->uops += 1;
            cg->mem += 64;
            em_vpermi2b(&cg->em, cg->r2, cg->tbl, cg->tbl, cg->vl);
            cg->uops += 1;
            if (first) {
                c_vpxor(cg, acc, cg->r1, cg->r2);
            } else if (cg->cand.ternlog) {
                c_vpternlog(cg, acc, cg->r1, cg->r2);
            } else {
                c_vpxor(cg, acc, acc, cg->r1);
                c_vpxor(cg, acc, acc, cg->r2);
            }
        } else {
            /* Nibble VPSHUFB path. On PDELTA/AVX512 the NIBHI table is
             * pinned in tbl2 (item 7): NIBLO is reloaded into tbl (scratch)
             * on every use, but the NIBHI pool load is skipped when the
             * coefficient matches the last use (tracked in
             * cg->mulacc_nib_c). Saves 1 load uop per (i,u,j) when c is
             * stable. On other paths the tables ping-pong through tbl. */
            c_ldpool(cg, cg->tbl, pool_get(cg, CK_NIBLO, c));
            c_shuf(cg, cg->r1, cg->tbl, cg->tlo);
            if (pl->kind == P_DELTA && is_avx512(cg)) {
                if (!cg->mulacc_nib_valid || cg->mulacc_nib_c != c) {
                    c_ldpool(cg, cg->tbl2, pool_get(cg, CK_NIBHI, c));
                    cg->mulacc_nib_c = c;
                    cg->mulacc_nib_valid = 1;
                }
                c_shuf(cg, cg->r2, cg->tbl2, cg->thi);
            } else {
                c_ldpool(cg, cg->tbl, pool_get(cg, CK_NIBHI, c));
                c_shuf(cg, cg->r2, cg->tbl, cg->thi);
            }
            if (first) {
                /* acc was 0: acc = r1^r2 in one uop. (On the ternlog
                 * candidate this replaces the 1-uop vpternlog, which
                 * would otherwise read the stale acc.) */
                c_vpxor(cg, acc, cg->r1, cg->r2);
            } else if (cg->cand.ternlog) {
                c_vpternlog(cg, acc, cg->r1, cg->r2);
            } else {
                c_vpxor(cg, acc, acc, cg->r1);
                c_vpxor(cg, acc, acc, cg->r2);
            }
        }
    }
}

/* Maximum-weight column pairing for the P_XOR VPTERNLOGD twin.
 *
 * A column pair (a,b) lets every row using BOTH columns fold its two
 * XORs into one VPTERNLOGD (in[b] as the memory operand); rows using a
 * single column of the pair take a plain XOR either way. So
 * weight(a,b) = #{rows using a and b}, and the uop-optimal pairing is
 * a maximum-weight matching on the complete graph over the n_in
 * columns. Computed at JIT-compile time -- zero cost in the generated
 * code. Exact DP over subsets for n_in <= 20; greedy by descending
 * weight above that (correctness never depends on optimality: a
 * suboptimal pairing just folds fewer XORs).
 *
 * Output: pa[k], pb[k] for k in [0, npairs); pb[k] == -1 marks an
 * unpaired (leftover) column, emitted as a plain single-column step.
 * Every column appears exactly once. Returns npairs. */
static int ternlog_pair_columns(const plan_t *pl, int *pa, int *pb)
{
    int n = pl->n_in, n_out = pl->n_out;
    int npairs = 0;
    if (n <= 0)
        return 0;
    if (n == 1) {
        pa[0] = 0;
        pb[0] = -1;
        return 1;
    }
    int *w = malloc((size_t)n * (size_t)n * sizeof(int));
    if (!w) { /* OOM: degrade to the natural (0,1),(2,3),... pairing */
        for (int i = 0; i < n; i += 2) {
            pa[npairs] = i;
            pb[npairs] = (i + 1 < n) ? i + 1 : -1;
            npairs++;
        }
        return npairs;
    }
    for (int a = 0; a < n; a++)
        for (int b = 0; b < n; b++) {
            int c = 0;
            if (a != b) {
                for (int j = 0; j < n_out; j++)
                    c += (pl->mat[(size_t)j * (size_t)n + (size_t)a] != 0 &&
                          pl->mat[(size_t)j * (size_t)n + (size_t)b] != 0);
            }
            w[(size_t)a * (size_t)n + (size_t)b] = c;
        }
    if (n <= 20) {
        /* Exact maximum-weight matching by DP over subsets:
         * dp[mask] = best achievable weight on the vertex set mask. */
        size_t N = (size_t)1 << n;
        int *dp = malloc(N * sizeof(int));
        int8_t *dec = malloc(N * sizeof(int8_t));
        if (dp && dec) {
            dp[0] = 0;
            for (size_t mask = 1; mask < N; mask++) {
                int i = 0; /* lowest set bit */
                while (!(mask & ((size_t)1 << i)))
                    i++;
                size_t m0 = mask ^ ((size_t)1 << i);
                int best = dp[m0]; /* leave i unpaired */
                int8_t bj = -1;
                for (int j = i + 1; j < n; j++) {
                    if (mask & ((size_t)1 << j)) {
                        int v = w[(size_t)i * (size_t)n + (size_t)j] +
                                dp[m0 ^ ((size_t)1 << j)];
                        if (v > best) {
                            best = v;
                            bj = (int8_t)j;
                        }
                    }
                }
                dp[mask] = best;
                dec[mask] = bj;
            }
            size_t mask = N - 1;
            while (mask) {
                int i = 0;
                while (!(mask & ((size_t)1 << i)))
                    i++;
                int8_t j = dec[mask];
                if (j < 0) {
                    pa[npairs] = i;
                    pb[npairs] = -1;
                    mask ^= (size_t)1 << i;
                } else {
                    pa[npairs] = i;
                    pb[npairs] = j;
                    mask ^= ((size_t)1 << i) | ((size_t)1 << j);
                }
                npairs++;
            }
            free(dp);
            free(dec);
            free(w);
            return npairs;
        }
        free(dp);
        free(dec);
        /* fall through to greedy */
    }
    /* Greedy fallback (n_in > 20, or DP allocation failed): take the
     * heaviest available pair first. O(n^3); JIT-compile time only. */
    {
        char *used = calloc((size_t)n, 1);
        if (!used) { /* ultimate fallback: natural pairing */
            for (int i = 0; i < n; i += 2) {
                pa[npairs] = i;
                pb[npairs] = (i + 1 < n) ? i + 1 : -1;
                npairs++;
            }
            free(w);
            return npairs;
        }
        for (;;) {
            int ba = -1, bb = -1, bw = 0;
            for (int a = 0; a < n; a++) {
                if (used[a])
                    continue;
                for (int b = a + 1; b < n; b++) {
                    if (used[b])
                        continue;
                    int wij = w[(size_t)a * (size_t)n + (size_t)b];
                    if (wij > bw) {
                        bw = wij;
                        ba = a;
                        bb = b;
                    }
                }
            }
            if (ba < 0)
                break; /* no positive-weight pair left; rest stay single */
            pa[npairs] = ba;
            pb[npairs] = bb;
            npairs++;
            used[ba] = used[bb] = 1;
        }
        for (int a = 0; a < n; a++) {
            if (!used[a]) {
                pa[npairs] = a;
                pb[npairs] = -1;
                npairs++;
            }
        }
        free(used);
    }
    free(w);
    return npairs;
}

/* Bit-program (xorbit) body for GF-multiply rows (track A1). Drives the
 * vector body from pl->bit_sched instead of the naive nibble emitter.
 * Input bits are materialized as bit-0-normalized byte vectors
 * ((in_i >> k) & 1); each temp is one vector XOR; output bytes are
 * assembled as out_j = XOR_b (P(j,b) << b). Temps are fanned out to
 * their user output bits immediately after formation (no spilling);
 * the 1-entry input-byte cache (cur_in) avoids reloading in[i].
 * u is outermost so the per-byte first-touch state (touched[]) spans
 * all temps/input bits of the u-slice: the first contribution to a
 * byte moves (acc was zero), later ones xor. P_DELTA seeds accs from
 * oldpar, so every contribution xors. All uop accounting mirrors
 * xorbit_version_uops in src/xorsched.c exactly (lockstep-tested). */

/* Fan out one bit-0-normalized value (vsrc) to a temp's / input bit's
 * user output bits, grouped by bit position b (users sorted ascending
 * by output-bit id). */
static void xorbit_fanout(cg_t *cg, const int *users, int nusers, int vsrc,
                          int vtmp, int u, int *touched, int is_delta)
{
    int U = cg->U;
    int w = 0;
    while (w < nusers) {
        int b = users[w] & 7;
        int w1 = w;
        while (w1 < nusers && (users[w1] & 7) == b)
            w1++;
        int nj = w1 - w;
        int j0 = users[w] >> 3;
        int acc0 = j0 * U + u;
        if (nj == 1 && !is_delta && !touched[j0]) {
            /* single first-touch user: shift/place directly into acc */
            if (b == 0)
                c_vmov(cg, acc0, vsrc);
            else
                c_sllw(cg, acc0, vsrc, (unsigned)b);
            touched[j0] = 1;
        } else if (b == 0) {
            for (int x = w; x < w1; x++) {
                int j = users[x] >> 3;
                int acc = j * U + u;
                if (!is_delta && !touched[j]) {
                    c_vmov(cg, acc, vsrc);
                    touched[j] = 1;
                } else {
                    c_vpxor(cg, acc, acc, vsrc);
                }
            }
        } else {
            c_sllw(cg, vtmp, vsrc, (unsigned)b);
            for (int x = w; x < w1; x++) {
                int j = users[x] >> 3;
                int acc = j * U + u;
                if (!is_delta && !touched[j]) {
                    c_vmov(cg, acc, vtmp);
                    touched[j] = 1;
                } else {
                    c_vpxor(cg, acc, acc, vtmp);
                }
            }
        }
        w = w1;
    }
}

/* Load in[i]'s pointer for the xorbit body. P_DELTA uses a raw mov
 * (uncounted), mirroring the naive DELTA body; otherwise body_inptr
 * (counted mov when i is not hoisted). */
static int xorbit_inptr(cg_t *cg, int i, int is_delta)
{
    if (is_delta) {
        em_mov_rm(&cg->em, R10, R8, -1, 0, i * 8);
        return R10;
    }
    return body_inptr(cg, i, R10);
}

static void gen_body_xorbit(cg_t *cg)
{
    const plan_t *pl = cg->pl;
    const xorbit_t *s = pl->bit_sched;
    assert(s && s->ntemps > 0);
    assert(s->kind == pl->kind);
    int n_out = pl->n_out, U = cg->U, vl = cg->vl;
    int ni_sched = s->n_in_sched;
    int is_delta = (pl->kind == P_DELTA);
    int inreg = cg->xbt_in, r1 = cg->xbt_r1, r2 = cg->xbt_r2;
    int mask1 = cg->xbt_mask;

    /* Temp-stack liveness (emitter redesign, track A3). Compute def/lastuse
     * for the temp DAG. For flat temps (current scheduler), all lastuse
     * are -1 (no temp is used by another temp). The liveness info is
     * available for future nested-temp support; the flat emission path
     * below is unchanged and byte-identical to the original. */
    int *t_def = NULL, *t_lastuse = NULL;
    int maxlive = 0;
    if (xorbit_temp_liveness(s, &t_def, &t_lastuse) == 0) {
        maxlive = xorbit_maxlive(s, t_def, t_lastuse);
        /* For flat temps, maxlive must be 0. If not, the scheduler
         * emitted nested temps (future), which the current emission
         * code does not handle. This is a plan-time invariant check. */
        assert(maxlive == 0);
    }
    free(t_def);
    free(t_lastuse);

    if (!is_delta) {
        /* zero dead output bytes (no atoms in any of their 8 bits) */
        for (int j = 0; j < n_out; j++) {
            int live = 0;
            for (int b = 0; b < 8; b++)
                if (s->bit_natoms[8 * j + b] > 0) {
                    live = 1;
                    break;
                }
            if (!live)
                for (int u = 0; u < U; u++)
                    c_vzero(cg, j * U + u);
        }
    } else {
        /* seed accumulators from oldpar (like the naive DELTA body) */
        int nc = pl->nchanged;
        for (int j = 0; j < n_out; j++) {
            em_mov_rm(&cg->em, R10, R8, -1, 0, (2 * nc + j) * 8);
            for (int u = 0; u < U; u++)
                c_ld(cg, j * U + u, R10, R11, 0, u * vl);
        }
    }

    int pf_done[RJ_MAX_DISKS];
    for (int i = 0; i < ni_sched; i++)
        pf_done[i] = 0;
    int touched[RJ_MAX_DISKS];

    for (int u = 0; u < U; u++) {
        for (int j = 0; j < n_out; j++)
            touched[j] = 0;
        int cur_in = -1;
        /* phase 1: form each temp, fan out immediately */
        for (int m = 0; m < s->ntemps; m++) {
            int a = s->temps[m].a, b = s->temps[m].b;
            int ia = a >> 3, ka = a & 7;
            int ib = b >> 3, kb = b & 7;
            if (cur_in != ia) {
                int pr = xorbit_inptr(cg, ia, is_delta);
                if (!pf_done[ia]) {
                    c_prefetch_in(cg, pr);
                    pf_done[ia] = 1;
                }
                c_ld(cg, inreg, pr, R11, 0, u * vl);
                cur_in = ia;
            }
            c_srlw(cg, r1, inreg, (unsigned)ka);
            c_and(cg, r1, r1, mask1);
            if (cur_in != ib) {
                int pr = xorbit_inptr(cg, ib, is_delta);
                if (!pf_done[ib]) {
                    c_prefetch_in(cg, pr);
                    pf_done[ib] = 1;
                }
                c_ld(cg, inreg, pr, R11, 0, u * vl);
                cur_in = ib;
            }
            c_srlw(cg, r2, inreg, (unsigned)kb);
            c_and(cg, r2, r2, mask1);
            c_vpxor(cg, r1, r1, r2);
            xorbit_fanout(cg, s->temps[m].users, s->temps[m].nusers, r1,
                          r2, u, touched, is_delta);
        }
        /* phase 2: surviving input bits, fanned out directly */
        for (int q = 0; q < 8 * ni_sched; q++) {
            if (!s->inbit_nusers[q])
                continue;
            int i = q >> 3, k = q & 7;
            if (cur_in != i) {
                int pr = xorbit_inptr(cg, i, is_delta);
                if (!pf_done[i]) {
                    c_prefetch_in(cg, pr);
                    pf_done[i] = 1;
                }
                c_ld(cg, inreg, pr, R11, 0, u * vl);
                cur_in = i;
            }
            c_srlw(cg, r1, inreg, (unsigned)k);
            c_and(cg, r1, r1, mask1);
            xorbit_fanout(cg, s->inbit_users[q], s->inbit_nusers[q], r1,
                          r2, u, touched, is_delta);
        }
    }
}

/* ---------------- recovery closed-form body (algos #4, rec_cf twin) ---------------- */

/* r1 = NIBLO(c)[tlo], r2 = NIBHI(c)[thi]: one nibble multiply's two
 * table lookups, with the tables loaded straight into the lookup
 * registers (no tbl intermediary -- c_shuf's vd==s1 form needs no mov,
 * so this is also one uop cheaper than the ping-pong on SSE). tbl
 * stays free as a value temp, which the e=2 combine needs: it has four
 * live vectors (p, q, both inners) but only t is non-scratch otherwise.
 * tlo/thi must hold the nibble split of the multiplicand (done once by
 * the caller when the input is shared). c != 0,1. */
static void rec_cf_nib_parts(cg_t *cg, uint8_t c)
{
    assert(c != 0 && c != 1);
    c_ldpool(cg, cg->r1, pool_get(cg, CK_NIBLO, c));
    c_shuf(cg, cg->r1, cg->r1, cg->tlo);
    c_ldpool(cg, cg->r2, pool_get(cg, CK_NIBHI, c));
    c_shuf(cg, cg->r2, cg->r2, cg->thi);
}

/* vd = c * vs (c != 0,1) via the GFNI affine, sharing mulacc_column's
 * last-value table dedup (the affine table depends only on c). Unlike
 * the nibble path's fixed r1/r2 outputs, the affine writes straight to
 * vd, so it never disturbs the combine's live temps. */
static void rec_cf_aff(cg_t *cg, int vd, int vs, uint8_t c)
{
    assert(c != 0 && c != 1);
    size_t poff = pool_get(cg, CK_AFF, c);
    if (!cg->mulacc_pool_valid || cg->mulacc_pool_off != poff) {
        c_ldpool(cg, cg->tbl, poff);
        cg->mulacc_pool_off = poff;
        cg->mulacc_pool_valid = 1;
    }
    c_aff(cg, vd, vs, cg->tbl);
}

/* vd = c * vs. 0 → zero, 1 → move, else the nibble or GFNI multiply.
 * Clobbers: nibble → tlo, thi, r1, r2; GFNI → tbl. vd/vs must not
 * alias those temps (vd == vs is fine). */
static void rec_cf_scale(cg_t *cg, int vd, int vs, uint8_t c)
{
    if (c == 0) {
        c_vzero(cg, vd);
        return;
    }
    if (c == 1) {
        c_vmov(cg, vd, vs);
        return;
    }
    if (cg->cand.mul == MUL_GFNI) {
        rec_cf_aff(cg, vd, vs, c);
    } else {
        nibble_split(cg, cg->tlo, cg->thi, vs);
        rec_cf_nib_parts(cg, c);
        c_vpxor(cg, vd, cg->r1, cg->r2);
    }
}

/* acc ^= c * t, where t holds the loaded survivor input and tlo/thi its
 * nibble split (split by the caller iff neednib -- the split is shared
 * by both e=2 syndromes). first = 1 when acc is provably zero (first
 * contributing column): the update is a move, not an xor-with-zero.
 * The per-coefficient shapes mirror mulacc_column (nibble lookups,
 * GFNI affine with table dedup, ternlog fold), so the syndrome phase
 * emits the same operations the generic path performs on the baked R
 * matrix, just factored per equation row. */
static void rec_cf_synd_accum(cg_t *cg, uint8_t c, int acc, int first,
                              int neednib)
{
    int gfni = cg->cand.mul == MUL_GFNI;
    if (!c)
        return;
    if (c == 1) {
        if (first)
            c_vmov(cg, acc, cg->t);
        else
            c_vpxor(cg, acc, acc, cg->t);
        return;
    }
    if (gfni) {
        rec_cf_aff(cg, cg->t2, cg->t, c);
        if (first)
            c_vmov(cg, acc, cg->t2);
        else
            c_vpxor(cg, acc, acc, cg->t2);
        return;
    }
    assert(neednib);
    rec_cf_nib_parts(cg, c);
    if (first) {
        c_vpxor(cg, acc, cg->r1, cg->r2);
    } else if (cg->cand.ternlog) {
        c_vpternlog(cg, acc, cg->r1, cg->r2);
    } else {
        c_vpxor(cg, acc, acc, cg->r1);
        c_vpxor(cg, acc, acc, cg->r2);
    }
}

/* Recovery closed-form vector body (algos #4, rec_cf twin).
 *
 * e=1: x_f = cinv * syndrome, syndrome = XOR_s ecoef[s]*x_s.
 * e=2: p = XOR_s ecoef1[s]*x_s, q = XOR_s ecoef2[s]*x_s, then
 *      x1 = detinv*(d*p ^ b*q), x2 = detinv*(c*p ^ a*q).
 *
 * The syndrome phase mirrors the generic P_MULACC column loop, so the
 * emitted per-coefficient operations are the same ones the generic path
 * performs on the baked R matrix, factored per equation row. The win:
 * e=1 replaces folding cinv into every coefficient with one
 * strength-reduced final multiply; e=2 evaluates the shared p/q
 * syndromes once (pure XOR when a row is all-0/1) instead of emitting
 * 2*ns baked multiplies plus the 2x2 combine.
 *
 * Register use is exactly the generic P_MULACC assignment (accs, t,
 * tlo/thi/r1/r2/tbl/mask or t2/tbl), so plan_need_regs is unchanged.
 * The scalar tail is the shared generic P_MULACC tail over the baked R
 * matrix (bit-identical to the closed form by the phase-1 guarantee);
 * the stores below are shared with the generic body. */
static void gen_body_rec_cf(cg_t *cg)
{
    const plan_t *pl = cg->pl;
    const struct rec_cf_data *cfd = pl->rec_cf;
    assert(pl->kind == P_MULACC);
    assert(cfd && (cfd->nf == 1 || cfd->nf == 2));
    assert(cfd->ns == pl->n_in && cfd->nf == pl->n_out);
    /* The planner excludes the other body twins from rec_cf bases;
     * jit_build_version rejects them outright. */
    assert(!cg->cand.xorbit && !cg->cand.xorsched && !cg->cand.permi2b &&
           !cg->cand.remat && !cg->cand.mshift && !cg->cand.pipe);
    /* P_MULACC kind guarantees a nontrivial baked R, hence need_mask:
     * the nibble temps (tlo/thi/r1/r2/tbl/mask) or GFNI temps (t2/tbl)
     * are allocated. */
    assert(cg->need_mask);
    int ns = cfd->ns, nf = cfd->nf, U = cg->U, vl = cg->vl;
    int gfni = cg->cand.mul == MUL_GFNI;

    /* First contributing survivor per syndrome (-1: none → the
     * accumulator is zeroed up front). */
    int first1 = -1, first2 = -1;
    for (int s = 0; s < ns; s++) {
        uint8_t c1 = (nf == 1) ? cfd->ecoef[s] : cfd->ecoef1[s];
        if (first1 < 0 && c1)
            first1 = s;
        if (nf == 2 && first2 < 0 && cfd->ecoef2[s])
            first2 = s;
    }
    for (int u = 0; u < U; u++) {
        if (first1 < 0)
            c_vzero(cg, u);
        if (nf == 2 && first2 < 0)
            c_vzero(cg, U + u);
    }

    /* Syndrome phase: column-major over survivors like the generic
     * body (load + optional nibble split shared by both e=2 syndromes,
     * then per-syndrome accumulate). */
    for (int s = 0; s < ns; s++) {
        uint8_t c1 = (nf == 1) ? cfd->ecoef[s] : cfd->ecoef1[s];
        uint8_t c2 = (nf == 2) ? cfd->ecoef2[s] : 0;
        if (!c1 && !c2)
            continue;
        int neednib =
            !gfni && ((c1 != 0 && c1 != 1) || (c2 != 0 && c2 != 1));
        int pr = body_inptr(cg, s, R10);
        c_prefetch_in(cg, pr);
        for (int u = 0; u < U; u++) {
            c_ld(cg, cg->t, pr, R11, 0, u * vl);
            if (neednib)
                nibble_split(cg, cg->tlo, cg->thi, cg->t);
            rec_cf_synd_accum(cg, c1, u, s == first1, neednib);
            if (nf == 2)
                rec_cf_synd_accum(cg, c2, U + u, s == first2, neednib);
        }
    }

    if (nf == 1) {
        /* x_f = cinv * syndrome. cinv == 1: nothing to do. */
        uint8_t cinv = cfd->cinv;
        if (cinv != 1 && first1 >= 0) {
            if (!gfni) {
                /* Strength reduction via loop-invariant hoisting: the
                 * NIBLO/NIBHI tables for the loop-invariant cinv live
                 * in r1/r2 across the u-loop, 2+6U uops/group vs 8U for
                 * the naive per-u ping-pong. (An xtime chain would cost
                 * 3+6U for cinv=2 and 3+6kU for cinv=2^k generally --
                 * the hoisted nibble dominates it for every cheap cinv,
                 * with no new constants or code paths.) */
                c_ldpool(cg, cg->r1, pool_get(cg, CK_NIBLO, cinv));
                c_ldpool(cg, cg->r2, pool_get(cg, CK_NIBHI, cinv));
            }
            for (int u = 0; u < U; u++) {
                if (gfni) {
                    rec_cf_scale(cg, u, u, cinv);
                } else {
                    nibble_split(cg, cg->tlo, cg->thi, u);
                    c_shuf(cg, cg->t, cg->r1, cg->tlo);
                    c_shuf(cg, cg->tbl, cg->r2, cg->thi);
                    c_vpxor(cg, u, cg->t, cg->tbl);
                }
            }
        }
    } else {
        /* e=2 combine: x1 = detinv*(d*p ^ b*q) → acc set 0,
         * x2 = detinv*(c*p ^ a*q) → acc set 1.
         * K-trick (exact by GF(2^8) distributivity): with k1 = b^-1*d
         * (b != 0), d*p ^ b*q = b*(k1*p ^ q); with k2 = a^-1*c,
         * c*p ^ a*q = a*(k2*p ^ q) (a != 0 always: pivot). The direct
         * form would need p, q, both inners, and multiply scratch live
         * at once -- five values, but GFNI only has t, t2, tbl beyond
         * the accs. The k-trick needs just two live temps (t and tt),
         * so one shape serves both multiply methods. tt must not alias
         * the multiply scratch: the nibble multiply is tbl-free (tables
         * go straight to r1/r2), so tt is tbl there; the GFNI affine
         * writes straight to its destination, so tt is t2 there. */
        uint8_t a = cfd->a, b = cfd->b, c = cfd->c, d = cfd->d;
        uint8_t detinv = cfd->detinv;
        assert(a != 0 && detinv != 0);
        uint8_t k1 = b ? gf_mul(gf_inv(b), d) : 0;
        uint8_t k2 = gf_mul(gf_inv(a), c);
        int tt = gfni ? cg->t2 : cg->tbl;
        for (int u = 0; u < U; u++) {
            int p_acc = u, q_acc = U + u;
            /* inner2 = a*(k2*p ^ q) → tt (c == 0: inner2 = a*q) */
            if (c) {
                rec_cf_scale(cg, tt, p_acc, k2);
                c_vpxor(cg, tt, tt, q_acc);
            } else {
                c_vmov(cg, tt, q_acc);
            }
            rec_cf_scale(cg, tt, tt, a);
            /* inner1 = b*(k1*p ^ q) → t (b == 0: inner1 = d*p, d != 0:
             * det = a*d ^ b*c = a*d != 0 with a != 0) */
            if (b) {
                rec_cf_scale(cg, cg->t, p_acc, k1);
                c_vpxor(cg, cg->t, cg->t, q_acc);
                rec_cf_scale(cg, cg->t, cg->t, b);
            } else {
                assert(d != 0);
                rec_cf_scale(cg, cg->t, p_acc, d);
            }
            /* x2 = detinv*inner2 → acc set 1; x1 = detinv*inner1 → acc
             * set 0. Both inners are done before either accumulator is
             * written: p/q stay live in the accs until here. */
            rec_cf_scale(cg, q_acc, tt, detinv);
            rec_cf_scale(cg, p_acc, cg->t, detinv);
        }
    }
}

static void gen_body(cg_t *cg)
{
    em_t *e = &cg->em;
    const plan_t *pl = cg->pl;
    int n_in = pl->n_in, n_out = pl->n_out, U = cg->U, vl = cg->vl;
    int gfni = cg->cand.mul == MUL_GFNI;

    /* Accumulator seeding. The prologue only zeroes rows that no input
     * column touches; every other accumulator is overwritten before its
     * first read, so xoring it with zero would be a wasted operation:
     *  - P_DELTA: every acc is seeded by a vector load from oldpar[j].
     *  - P_HORNER: the i==0 iteration collapses acc = 0*c + t to a
     *    plain load of t into each acc (n_in >= 1 for real plans; the
     *    degenerate n_in==0 case keeps its zeroes below).
     *  - P_XOR / P_MULACC: the first contributing column moves/loads
     *    into the accumulator instead of xoring with zero. */
    if (pl->kind == P_HORNER && n_in == 0) {
        for (int j = 0; j < n_out; j++)
            for (int u = 0; u < U; u++)
                c_vzero(cg, j * U + u);
    } else if (((pl->kind == P_XOR && !cg->cand.ternlog) ||
                pl->kind == P_MULACC) &&
               !cg->cand.xorbit && !cg->cand.rec_cf) {
        /* P_XOR's ternlog twin zeroes every accumulator up front in its
         * own prologue below (no first-input move logic), so the
         * dead-row-only zeroing here is skipped for it. The xorbit
         * twin zeroes its own dead bytes (bit-program-based) inside
         * gen_body_xorbit. The rec_cf twin does its own
         * syndrome-based seeding inside gen_body_rec_cf. */
        for (int j = 0; j < n_out; j++) {
            int live = 0;
            for (int i = 0; i < n_in; i++)
                if (pl->mat[(size_t)j * (size_t)n_in + (size_t)i]) {
                    live = 1;
                    break;
                }
            if (!live)
                for (int u = 0; u < U; u++)
                    c_vzero(cg, j * U + u);
        }
    }

    if (cg->cand.xorbit) {
        /* Bit-program twin (track A1): the body follows pl->bit_sched
         * (cross-row/coefficient CSE at bit granularity) instead of the
         * naive nibble emitter. gen_body_xorbit handles its own
         * accumulator seeding/zeroing; stores below are shared. */
        gen_body_xorbit(cg);
    } else if (pl->kind == P_XOR) {
        if (cg->cand.ternlog) {
            /* VPTERNLOGD twin (AVX-512 only): fold two input XORs into
             * one ternary-logic uop per row. Columns are processed in
             * maximum-weight pairs (ternlog_pair_columns): weight(a,b)
             * is the number of rows using both columns, so the pairing
             * maximizes the rows that take the single-uop
             * c_vpternlog_mem. in[a] is loaded once into t and shared
             * across rows; in[b]'s pointer is prb (hoisted reg when b <
             * nhoist, else RDX loaded per iteration). For a row using
             * both columns,
             * acc ^= t ^ in[b] becomes a single VPTERNLOGD with in[b]
             * as the memory operand ([prb+R11+u*vl]), vs one load + two
             * XOR uops in the plain candidate. Rows using only one
             * column of the pair take a plain XOR (register form for
             * in[a], memory form for in[b]); a leftover unpaired column
             * (pb[k] == -1) degrades to a plain single-column step. All
             * accumulators are zeroed up front -- this also covers
             * zero-input rows -- so no first-input move logic is needed.
             * Vector register use is accs + t, identical to the plain
             * candidate, so the n_out*U+1 pressure formula in
             * plan_need_regs() is unchanged. */
            assert(cg->cand.isa == ISA_AVX512);
            for (int j = 0; j < n_out; j++)
                for (int u = 0; u < U; u++)
                    c_vzero(cg, j * U + u);
            int tpa[RJ_MAX_DISKS], tpb[RJ_MAX_DISKS];
            int npairs = ternlog_pair_columns(pl, tpa, tpb);
            for (int pi = 0; pi < npairs; pi++) {
                int a = tpa[pi], b = tpb[pi];
                int ua = 0, ub = 0;
                for (int j = 0; j < n_out; j++) {
                    ua |= pl->mat[(size_t)j * (size_t)n_in + (size_t)a];
                    if (b >= 0)
                        ub |= pl->mat[(size_t)j * (size_t)n_in + (size_t)b];
                }
                if (!ua && !ub)
                    continue;
                int pra = -1, prb = -1;
                if (ua) {
                    pra = body_inptr(cg, a, R10);
                    c_prefetch_in(cg, pra);
                }
                if (b >= 0 && ub) {
                    prb = body_inptr(cg, b, RDX);
                    c_prefetch_in(cg, prb);
                }
                for (int u = 0; u < U; u++) {
                    if (ua)
                        c_ld(cg, cg->t, pra, R11, 0, u * vl);
                    for (int j = 0; j < n_out; j++) {
                        int ma =
                            !!pl->mat[(size_t)j * (size_t)n_in + (size_t)a];
                        int mb = b >= 0 &&
                                 !!pl->mat[(size_t)j * (size_t)n_in +
                                           (size_t)b];
                        int acc = j * U + u;
                        if (ma && mb)
                            c_vpternlog_mem(cg, acc, cg->t, prb, R11, 0,
                                            u * vl);
                        else if (ma)
                            c_vpxor(cg, acc, acc, cg->t);
                        else if (mb)
                            c_vpxor_mem(cg, acc, acc, prb, R11, 0, u * vl);
                    }
                }
            }
        } else if (cg->cand.xorsched) {
        /* XOR-schedule twin: the body follows pl->xor_sched (cross-row
         * CSE) instead of the naive per-row column loop. Phase 1 forms
         * each temp t_k = in_a ^ in_b and fans it out to its user rows;
         * phase 2 streams the surviving input atoms column-major like
         * the naive body. Every accumulator is overwritten (move /
         * direct load) on first touch, never XORed with zero; each
         * shared subexpression is computed once. Dead rows were zeroed
         * by the prologue above (shared with the naive body).
         * Every ISA forms the temp with a folded memory XOR (SSE: pxor
         * m128, one uop); only one live temp is needed. RDX is dead
         * here (len was moved to RCX in the prologue; the ternlog
         * branch that uses RDX as prb is not taken). */
        {
            const xorsched_t *xs = pl->xor_sched;
            assert(xs && xs->ntemps > 0);
            int pf_done[RJ_MAX_DISKS];
            for (int i = 0; i < n_in; i++)
                pf_done[i] = 0;
            /* phase 1: temp formation + fan-out */
            for (int k = 0; k < xs->ntemps; k++) {
                int a = xs->temps[k].a, b = xs->temps[k].b;
                int pra = body_inptr(cg, a, R10);
                int prb = body_inptr(cg, b, RDX);
                if (!pf_done[a]) {
                    c_prefetch_in(cg, pra);
                    pf_done[a] = 1;
                }
                if (!pf_done[b]) {
                    c_prefetch_in(cg, prb);
                    pf_done[b] = 1;
                }
                for (int u = 0; u < U; u++) {
                    c_ld(cg, cg->t, pra, R11, 0, u * vl);
                    c_vpxor_mem(cg, cg->t, cg->t, prb, R11, 0, u * vl);
                    for (int wi = 0; wi < xs->temps[k].nusers; wi++) {
                        int uw = xs->temps[k].users[wi];
                        int acc = xs->first_row[uw] * U + u;
                        if (xs->row_first[uw] == k)
                            c_vmov(cg, acc, cg->t);
                        else
                            c_vpxor(cg, acc, acc, cg->t);
                    }
                }
            }
            /* phase 2: surviving input atoms, column-major */
            for (int i = 0; i < n_in; i++) {
                int nusers = xs->in_nusers[i];
                if (!nusers)
                    continue;
                int pr = body_inptr(cg, i, R10);
                if (!pf_done[i]) {
                    c_prefetch_in(cg, pr);
                    pf_done[i] = 1;
                }
                int uw0 = xs->in_users[i][0];
                int is_first =
                    (nusers == 1 && xs->row_first[uw0] == xs->ntemps + i);
                for (int u = 0; u < U; u++) {
                    if (is_first) {
                        /* acc dead: direct load */
                        int acc = xs->first_row[uw0] * U + u;
                        c_ld(cg, acc, pr, R11, 0, u * vl);
                    } else if (nusers == 1) {
                        /* acc live, single user: folded memory XOR
                         * (1 uop on every ISA) */
                        int acc = xs->first_row[uw0] * U + u;
                        c_vpxor_mem(cg, acc, acc, pr, R11, 0, u * vl);
                    } else {
                        c_ld(cg, cg->t, pr, R11, 0, u * vl);
                        for (int wi = 0; wi < nusers; wi++) {
                            int uw = xs->in_users[i][wi];
                            int acc = xs->first_row[uw] * U + u;
                            if (xs->row_first[uw] == xs->ntemps + i)
                                c_vmov(cg, acc, cg->t);
                            else
                                c_vpxor(cg, acc, acc, cg->t);
                        }
                    }
                }
            }
            /* duplicate rows: vector moves from the canonical row */
            for (int j = 0; j < n_out; j++) {
                int uw = xs->urow_of[j];
                if (j == xs->first_row[uw] || xs->row_first[uw] < 0)
                    continue;
                for (int u = 0; u < U; u++)
                    c_vmov(cg, j * U + u, xs->first_row[uw] * U + u);
            }
        }
        } else {
        /* first_in[j]: first input with a 1 in row j (-1 if none). The
         * accumulator is provably zero there: load/move, don't xor. */
        int first_in[RJ_MAX_DISKS];
        for (int j = 0; j < n_out; j++) {
            first_in[j] = -1;
            for (int i = 0; i < n_in; i++)
                if (pl->mat[(size_t)j * (size_t)n_in + (size_t)i]) {
                    first_in[j] = i;
                    break;
                }
        }
        for (int i = 0; i < n_in; i++) {
            int users[RJ_MAX_DISKS], nu = 0;
            for (int j = 0; j < n_out; j++)
                if (pl->mat[j * n_in + i])
                    users[nu++] = j;
            int pr = -1;
            if (!nu) {
                /* REDUX crc_fold: the encode skips this input column,
                 * but the CRC still covers the stream's bytes. Fold
                 * from memory (the vector is never materialized). */
                if (cg->cand.crc_fold) {
                    pr = body_inptr(cg, i, R10);
                    for (int u = 0; u < U; u++)
                        gen_fold_mem(cg, pr, R11, 0, u * vl, i);
                }
                continue;
            }
            pr = body_inptr(cg, i, R10);
            c_prefetch_in(cg, pr);
            for (int u = 0; u < U; u++) {
                if (nu == 1) {
                    int acc = users[0] * U + u;
                    if (i == first_in[users[0]]) {
                        c_ld(cg, acc, pr, R11, 0, u * vl);
                        /* REDUX crc_fold: acc holds input i's bytes. */
                        if (cg->cand.crc_fold)
                            gen_fold_zmm(cg, acc, i);
                    } else {
                        c_vpxor_mem(cg, acc, acc, pr, R11, 0, u * vl);
                        /* REDUX crc_fold: memory-operand consume; fold
                         * input i's bytes from their address. */
                        if (cg->cand.crc_fold)
                            gen_fold_mem(cg, pr, R11, 0, u * vl, i);
                    }
                } else {
                    c_ld(cg, cg->t, pr, R11, 0, u * vl);
                    /* REDUX crc_fold: fold input i's vector once (the
                     * fan-out below only reads it). */
                    if (cg->cand.crc_fold)
                        gen_fold_zmm(cg, cg->t, i);
                    for (int k = 0; k < nu; k++) {
                        int acc = users[k] * U + u;
                        if (i == first_in[users[k]])
                            c_vmov(cg, acc, cg->t);
                        else
                            c_vpxor(cg, acc, acc, cg->t);
                    }
                }
            }
        }
        } /* end non-ternlog P_XOR */
    } else if (pl->kind == P_MULACC) {
        if (cg->cand.rec_cf) {
            /* Recovery closed-form twin (algos #4): the syndrome +
             * combine form instead of the generic MULACC over the baked
             * R matrix. Stores below are shared. */
            gen_body_rec_cf(cg);
        } else {
        /* first_col[j]: first column with a nonzero entry in row j. */
        int first_col[RJ_MAX_DISKS];
        for (int j = 0; j < n_out; j++) {
            first_col[j] = -1;
            for (int i = 0; i < n_in; i++)
                if (pl->mat[(size_t)j * (size_t)n_in + (size_t)i]) {
                    first_col[j] = i;
                    break;
                }
        }
        for (int i = 0; i < n_in; i++) {
            int any, neednib;
            mulacc_col_info(pl, i, n_in, &any, &neednib);
            if (!any) {
                /* REDUX crc_fold: the encode skips this column, but the
                 * CRC still covers the stream's bytes. */
                if (cg->cand.crc_fold) {
                    int pr0 = body_inptr(cg, i, R10);
                    for (int u = 0; u < U; u++)
                        gen_fold_mem(cg, pr0, R11, 0, u * vl, i);
                }
                continue;
            }
            int pr = body_inptr(cg, i, R10);
            c_prefetch_in(cg, pr);
            for (int u = 0; u < U; u++) {
                c_ld(cg, cg->t, pr, R11, 0, u * vl);
                /* REDUX crc_fold: t holds input i's bytes (fold before
                 * the nibble split reuses t's lanes). */
                if (cg->cand.crc_fold)
                    gen_fold_zmm(cg, cg->t, i);
                if (neednib && !gfni)
                    nibble_split(cg, cg->tlo, cg->thi, cg->t);
                mulacc_column(cg, i, n_in, u, first_col);
            }
        }
        } /* end generic P_MULACC (non-rec_cf) */
    } else if (pl->kind == P_DELTA) {
        /* new_p[j] = old_p[j] ^ Σ_i ((old[i] ^ new[i]) * M[j][changed[i]]).
         * in[] = [old_0..old_{nc-1}, new_0..new_{nc-1},
         *         oldpar_0..oldpar_{np-1}]. RDX is dead after the prologue
         * (R8=in[], R9=out[], RCX=len) and serves as the 2nd pointer. */
        int nc = pl->nchanged;
        /* 2nd delta temp: not yet live at delta time (used only inside
         * the per-row multiply below). On SSE c_vpxor_mem would clobber
         * cg->t, so the two-operand form is needed there. */
        int dt = gfni ? cg->t2 : cg->r1;
        for (int j = 0; j < n_out; j++) {
            em_mov_rm(e, R10, R8, -1, 0, (2 * nc + j) * 8);
            c_prefetch_in(cg, R10);
            for (int u = 0; u < U; u++)
                c_ld(cg, j * U + u, R10, R11, 0, u * vl);
        }
        for (int i = 0; i < nc; i++) {
            int any, neednib;
            mulacc_col_info(pl, i, nc, &any, &neednib);
            if (!any)
                continue;
            em_mov_rm(e, R10, R8, -1, 0, i * 8);
            c_prefetch_in(cg, R10);
            em_mov_rm(e, RDX, R8, -1, 0, (nc + i) * 8);
            c_prefetch_in(cg, RDX);
            for (int u = 0; u < U; u++) {
                if (is_sse(cg)) {
                    c_ld(cg, cg->t, R10, R11, 0, u * vl);
                    c_ld(cg, dt, RDX, R11, 0, u * vl);
                    c_vpxor(cg, cg->t, cg->t, dt);
                } else {
                    c_ld(cg, cg->t, R10, R11, 0, u * vl);
                    c_vpxor_mem(cg, cg->t, cg->t, RDX, R11, 0, u * vl);
                }
                if (neednib && !gfni)
                    nibble_split(cg, cg->tlo, cg->thi, cg->t);
                /* NULL: accumulators hold oldpar, not zero -- always xor */
                mulacc_column(cg, i, nc, u, NULL);
            }
        }
    } else { /* P_HORNER */
        /* Remat table-load dedup (pure JIT-time state): the table held in
         * tbl depends only on horner_c[j], so it is loop-invariant across
         * the u-loop (and across input columns i); nothing clobbers tbl
         * between a load and its use. Track the last-loaded pool offset
         * and skip redundant c_ldpool calls. pool_get() dedups by
         * (kind, c) with append-only offsets, so offset comparison is
         * sound. Skipping the call also skips its uop/mem accounting,
         * keeping the static cost model post-dedup. */
        size_t tbl_pool_off = 0;
        int tbl_pool_valid = 0;
        for (int i = 0; i < n_in; i++) {
            int pr = body_inptr(cg, i, R10);
            c_prefetch_in(cg, pr);
            for (int u = 0; u < U; u++) {
                if (i == 0) {
                    /* Accumulators are provably zero: acc = 0*c + t = t
                     * for any c. Load directly into acc[0] and move to
                     * the remaining rows: one load + (n_out-1) moves per
                     * u instead of load-to-t + n_out moves, saving one
                     * uop per u. acc[0] = u never aliases t (= n_out*U),
                     * and the `continue` keeps t dead. */
                    c_ld(cg, 0 * U + u, pr, R11, 0, u * vl);
                    /* REDUX crc_fold: acc holds input 0's bytes. */
                    if (cg->cand.crc_fold)
                        gen_fold_zmm(cg, 0 * U + u, 0);
                    for (int j = 1; j < n_out; j++)
                        c_vmov(cg, j * U + u, 0 * U + u);
                    continue;
                }
                c_ld(cg, cg->t, pr, R11, 0, u * vl);
                /* REDUX crc_fold: fold input i's vector once. */
                if (cg->cand.crc_fold)
                    gen_fold_zmm(cg, cg->t, i);
                for (int j = 0; j < n_out; j++) {
                    uint8_t c = pl->horner_c[j];
                    int acc = j * U + u;
                    if (c == 1) {
                        c_vpxor(cg, acc, acc, cg->t);
                    } else if (c == 0) {
                        /* acc = 0*acc + t = t: a move, not zero+xor */
                        c_vmov(cg, acc, cg->t);
                    } else if (gfni) {
                        if (cg->cand.remat) {
                            size_t poff = pool_get(cg, CK_AFF, c);
                            if (!tbl_pool_valid || tbl_pool_off != poff) {
                                c_ldpool(cg, cg->tbl, poff);
                                tbl_pool_off = poff;
                                tbl_pool_valid = 1;
                            }
                            c_aff(cg, cg->t2, acc, cg->tbl);
                        } else {
                            c_aff(cg, cg->t2, acc, cg->hreg[2 * j]);
                        }
                        c_vpxor(cg, acc, cg->t2, cg->t);
                    } else {
                        nibble_split(cg, cg->tlo, cg->thi, acc);
                        if (cg->cand.remat) {
                            /* NIBLO/NIBHI ping-pong through the single tbl
                             * scratch: each u iteration needs both tables,
                             * so the 2 loads per (i,j,u) are inherent and
                             * last-value tracking cannot help. A second
                             * scratch would allow hoisting one table out
                             * of the u-loop, but it would change the
                             * proven pressure formula from n_out*U+7 to
                             * n_out*U+8 (requiring re-proof and a
                             * plan_need_regs update); the extra L1 loads
                             * are accepted instead. */
                            c_ldpool(cg, cg->tbl,
                                     pool_get(cg, CK_NIBLO, c));
                            c_shuf(cg, cg->r1, cg->tbl, cg->tlo);
                            c_ldpool(cg, cg->tbl,
                                     pool_get(cg, CK_NIBHI, c));
                            c_shuf(cg, cg->r2, cg->tbl, cg->thi);
                        } else {
                            c_shuf(cg, cg->r1, cg->hreg[2 * j], cg->tlo);
                            c_shuf(cg, cg->r2, cg->hreg[2 * j + 1],
                                   cg->thi);
                        }
                        c_vpxor(cg, acc, cg->r1, cg->r2);
                        c_vpxor(cg, acc, acc, cg->t);
                    }
                }
            }
        }
    }

    for (int j = 0; j < n_out; j++) {
        em_mov_rm(e, R10, R9, -1, 0, j * 8);
        for (int u = 0; u < U; u++) {
            /* REDUX crc_fold: the parity vector is complete in the acc;
             * fold it before the store (or the zmm's reuse). */
            if (cg->cand.crc_fold)
                gen_fold_zmm(cg, j * U + u, n_in + j);
            c_st(cg, R10, R11, 0, u * vl, j * U + u);
        }
    }
}

/* ---------------- software-pipelined loop (track I8) ---------------- */

/* Columns whose input data the pipe body stages: P_MULACC stages only
 * columns with a nonzero coefficient somewhere (dead columns are
 * skipped, mirroring gen_body); P_HORNER stages every input column.
 * Returns the count (0 for degenerate plans: no loads to pipeline). */
static int pipe_used_columns(cg_t *cg, int *cols)
{
    const plan_t *pl = cg->pl;
    int ncols = 0;
    if (pl->kind == P_MULACC) {
        for (int i = 0; i < pl->n_in; i++) {
            int any, neednib;
            mulacc_col_info(pl, i, pl->n_in, &any, &neednib);
            if (any)
                cols[ncols++] = i;
        }
    } else { /* P_HORNER */
        for (int i = 0; i < pl->n_in; i++)
            cols[ncols++] = i;
    }
    return ncols;
}

/* Pipe prologue: issue the first iteration's first-column loads into
 * Dn. Runs after the len<group branch (a full group fits), before the
 * loop label; R11 == 0 here. Uncounted: prologue setup is not
 * per-iteration work. */
static void gen_pipe_prologue(cg_t *cg, int col0)
{
    int vl = cg->vl;
    double save_uops = cg->uops, save_mem = cg->mem;
    int pr = body_inptr(cg, col0, R10);
    for (int u = 0; u < cg->U; u++)
        c_ld(cg, cg->pipe_dn + u, pr, R11, 0, u * vl);
    cg->uops = save_uops;
    cg->mem = save_mem;
}

/* One HORNER column's compute from the staged dataset (pipe twin).
 * Mirrors gen_body's P_HORNER nibble path with the input lanes in
 * [data, data+U) instead of freshly loaded t (non-remat, non-GFNI;
 * both are rejected for the pipe twin). */
static void pipe_horner_compute(cg_t *cg, int col, int data)
{
    const plan_t *pl = cg->pl;
    int n_out = pl->n_out, U = cg->U;
    assert(!cg->cand.remat);
    for (int u = 0; u < U; u++) {
        if (col == 0) {
            /* acc = d0: plain moves from the staged dataset */
            for (int j = 0; j < n_out; j++)
                c_vmov(cg, j * U + u, data + u);
            continue;
        }
        for (int j = 0; j < n_out; j++) {
            uint8_t c = pl->horner_c[j];
            int acc = j * U + u;
            if (c == 1) {
                c_vpxor(cg, acc, acc, data + u);
            } else if (c == 0) {
                c_vmov(cg, acc, data + u);
            } else {
                nibble_split(cg, cg->tlo, cg->thi, acc);
                c_shuf(cg, cg->r1, cg->hreg[2 * j], cg->tlo);
                c_shuf(cg, cg->r2, cg->hreg[2 * j + 1], cg->thi);
                c_vpxor(cg, acc, cg->r1, cg->r2);
                c_vpxor(cg, acc, acc, data + u);
            }
        }
    }
}

/* Software-pipelined (double-buffered) loop body (track I8, pipe twin).
 *
 * Per-column double buffering over the D/Dn datasets (U regs each):
 *   - compute(cols[cidx], iter) reads (cidx == 0 ? Dn : D). Dn holds
 *     (cols[0], iter): the prologue for iter == 0, the boundary load
 *     of the previous iteration otherwise. mulacc_column reads the
 *     input through cg->t, so the per-u loop rebinds cg->t to the
 *     dataset lane (restored after the column).
 *   - then, if cidx+1 < ncols: load(cols[cidx+1], iter) -> D at R11;
 *     else (last column): if R11+group <= rax (a full next iteration
 *     exists): load(cols[0], iter+1) -> Dn at RDX = R11+group. The
 *     guard is exact (compare-based loop has no trip counter) and
 *     keeps the load in-bounds; the last iteration would otherwise
 *     over-read up to group bytes into/past the tail. RDX doubles as
 *     the next-offset index register (free in the pipe MULACC/HORNER
 *     bodies; P_DELTA, its other user, is excluded from pipe).
 *     The boundary load skips the software prefetch: the demand load
 *     is already issued a full iteration early, so a prefetch would
 *     only add uops.
 * Every issued load is consumed by the next column step's compute, so
 * the pipeline drains naturally; the "epilogue" is the final
 * iteration's stores. ncols == 0 falls back to gen_body. */
static void gen_body_pipe(cg_t *cg)
{
    em_t *e = &cg->em;
    const plan_t *pl = cg->pl;
    int n_in = pl->n_in, n_out = pl->n_out, U = cg->U, vl = cg->vl;
    assert(cg->cand.pipe);
    assert(pl->kind == P_MULACC || pl->kind == P_HORNER);
    assert(!cg->cand.remat);

    int cols[RJ_MAX_DISKS];
    int ncols = pipe_used_columns(cg, cols);
    if (ncols == 0) {
        gen_body(cg); /* degenerate: no loads to pipeline */
        return;
    }

    int first_col[RJ_MAX_DISKS];
    if (pl->kind == P_MULACC) {
        for (int j = 0; j < n_out; j++) {
            first_col[j] = -1;
            for (int i = 0; i < n_in; i++)
                if (pl->mat[(size_t)j * (size_t)n_in + (size_t)i]) {
                    first_col[j] = i;
                    break;
                }
        }
    }

    int save_t = cg->t;
    for (int cidx = 0; cidx < ncols; cidx++) {
        int col = cols[cidx];
        int data = (cidx == 0) ? cg->pipe_dn : cg->pipe_d;
        /* compute column col from the staged dataset */
        if (pl->kind == P_MULACC) {
            int any, neednib;
            mulacc_col_info(pl, col, n_in, &any, &neednib);
            for (int u = 0; u < U; u++) {
                cg->t = data + u;
                if (neednib)
                    nibble_split(cg, cg->tlo, cg->thi, cg->t);
                mulacc_column(cg, col, n_in, u, first_col);
            }
        } else {
            pipe_horner_compute(cg, col, data);
        }
        /* issue the next loads into the other dataset */
        if (cidx + 1 < ncols) {
            int pr = body_inptr(cg, cols[cidx + 1], R10);
            c_prefetch_in(cg, pr);
            for (int u = 0; u < U; u++)
                c_ld(cg, cg->pipe_d + u, pr, R11, 0, u * vl);
        } else {
            /* boundary: (cols[0], iter+1) -> Dn at R11+group, iff a
             * full next iteration exists. */
            int skip = em_label(e);
            em_mov_rr(e, RDX, R11);
            em_add_ri(e, RDX, cg->group);
            em_cmp_rr(e, RDX, RAX);
            em_jcc(e, CC_A, skip);
            {
                int pr0 = body_inptr(cg, cols[0], R10);
                for (int u = 0; u < U; u++)
                    c_ld(cg, cg->pipe_dn + u, pr0, RDX, 0, u * vl);
            }
            em_bind(e, skip);
        }
    }
    cg->t = save_t;

    for (int j = 0; j < n_out; j++) {
        em_mov_rm(e, R10, R9, -1, 0, j * 8);
        for (int u = 0; u < U; u++)
            c_st(cg, R10, R11, 0, u * vl, j * U + u);
    }
}

/* Scalar-tail helper: RBX ^= RDX * c (GF(2^8)); RDI/RSI = log/exp tables.
 * RDX holds the multiplicand byte on entry; RAX is scratch. */
static void tail_mul_byte(cg_t *cg, uint8_t c)
{
    em_t *e = &cg->em;
    if (c == 1) {
        em_xor_r8(e, RBX, RDX);
        return;
    }
    int lsk = em_label(e), lns = em_label(e);
    em_test_rr(e, RDX, RDX);
    em_jcc(e, CC_E, lsk);
    em_movzx_rm8(e, RAX, RDI, RDX, 0, 0); /* log[d] */
    em_add_r32_ri(e, RAX, cg->logtab[c]);
    em_cmp_r32_ri(e, RAX, 255);
    em_jcc(e, CC_B, lns);
    em_add_r32_ri(e, RAX, -255);
    em_bind(e, lns);
    em_movzx_rm8(e, RAX, RSI, RAX, 0, 0); /* exp -> mul */
    em_xor_r8(e, RBX, RAX);
    em_bind(e, lsk);
}

/* scalar tail for len % group bytes. r11 = tail_start, rcx = len. */
/* REDUX crc_fold: fold the whole 16B chunks of the scalar tail region.
 * Spec §3.4: the kernel folds exactly floor(tl/16)*16 bytes per call;
 * the vector body covers 64B multiples, so the tail's whole chunks
 * ([tail_start, tail_start + nchunks*16)) must be folded here -- the
 * bytes the encode consumed/produced, observed exactly once:
 *   - input streams: 16B chunks from in[i] (memory; inputs are never
 *     modified by the kernel);
 *   - parity streams: 16B chunks from out[j] (the byte loop above
 *     already computed and stored every tail byte; reloading observes
 *     exactly the produced bytes).
 * On entry R11 == RCX (byte loop finished) and [RSP] = tail_start.
 * The sub-16B remainder is the driver's rj_crc_fold_tail (spec §6);
 * the kernel never folds partial chunks. */
static void gen_fold_tail_chunks(cg_t *cg)
{
    em_t *e = &cg->em;
    const plan_t *pl = cg->pl;
    int n_in = pl->n_in, n_out = pl->n_out;
    int done = em_label(e), chunk = em_label(e);
    /* RBX = tail_start; RAX = nchunks = (RCX - tail_start) >> 4. */
    em_mov_rm(e, RBX, RSP, -1, 0, 0);
    em_mov_rr(e, RAX, RCX);
    em_sub_rr(e, RAX, RBX);
    em_shr_ri(e, RAX, 4);
    em_test_rr(e, RAX, RAX);
    em_jcc(e, CC_E, done);
    em_xor_rr(e, R10, R10); /* R10 = chunk byte offset */
    em_bind(e, chunk);
    for (int i = 0; i < n_in; i++) {
        /* RDX = in[i] + tail_start; fold 16B at [RDX + R10]. */
        em_mov_rm(e, RDX, R8, -1, 0, i * 8);
        em_add_rr(e, RDX, RBX);
        em_vmovdqa_ld_evex128(e, cg->fold_xc, RDX, R10, 0, 0);
        gen_fold_chunk(cg, i);
    }
    for (int j = 0; j < n_out; j++) {
        em_mov_rm(e, RDX, R9, -1, 0, j * 8);
        em_add_rr(e, RDX, RBX);
        em_vmovdqa_ld_evex128(e, cg->fold_xc, RDX, R10, 0, 0);
        gen_fold_chunk(cg, n_in + j);
    }
    em_add_ri(e, R10, 16);
    em_dec_r(e, RAX);
    em_jcc(e, CC_NE, chunk);
    em_bind(e, done);
}

static void gen_tail(cg_t *cg)
{
    em_t *e = &cg->em;
    const plan_t *pl = cg->pl;
    int n_in = pl->n_in, n_out = pl->n_out;

    em_push_r(e, R11); /* [rsp] = tail_start */
    if (cg->need_logexp) {
        size_t loff = pool_get(cg, CK_LOG, 0);
        size_t eoff = pool_get(cg, CK_EXP, 0);
        em_lea(e, RDI, R15, -1, 0, (int)loff);
        em_lea(e, RSI, R15, -1, 0, (int)eoff);
    }
    for (int j = 0; j < n_out; j++) {
        int lj = em_label(e);
        em_mov_rm(e, R10, R9, -1, 0, j * 8);
        em_mov_rm(e, R11, RSP, -1, 0, 0);
        if (pl->kind == P_DELTA)
            em_bind(e, lj); /* re-seed the accumulator every tail byte */
        if (pl->kind == P_DELTA) {
            /* seed the accumulator from the old parity byte, not zero */
            em_mov_rm(e, RAX, R8, -1, 0, (2 * pl->nchanged + j) * 8);
            em_movzx_rm8(e, RBX, RAX, R11, 0, 0);
        } else if (pl->kind == P_HORNER && n_in > 0) {
            /* No zero: the i==0 iteration below overwrites RBX with a
             * plain move (RBX = 0*c + d = d for any c). */
        } else if (pl->kind == P_XOR) {
            /* Zero only if the row has no contributors; otherwise the
             * first contributor moves its byte into BL (see below) and
             * the zero would be overwritten before any read. */
            int has = 0;
            for (int i = 0; i < n_in; i++)
                if (pl->mat[j * n_in + i]) {
                    has = 1;
                    break;
                }
            if (!has)
                em_xor_r32(e, RBX, RBX);
        } else {
            em_xor_r32(e, RBX, RBX);
        }
        if (pl->kind != P_DELTA)
            em_bind(e, lj);
        if (pl->kind == P_XOR) {
            int first = 1;
            for (int i = 0; i < n_in; i++) {
                if (!pl->mat[j * n_in + i])
                    continue;
                em_mov_rm(e, RAX, R8, -1, 0, i * 8);
                em_mov_r_m8(e, RDX, RAX, R11, 0, 0);
                if (first) {
                    /* RBX is not zeroed above; BL = DL is a move,
                     * not an xor-with-zero. Upper RBX bits stay stale
                     * but are never read (store/xors only touch BL). */
                    em_mov_r8r8(e, RBX, RDX);
                    first = 0;
                } else {
                    em_xor_r8(e, RBX, RDX);
                }
            }
        } else if (pl->kind == P_MULACC) {
            for (int i = 0; i < n_in; i++) {
                uint8_t c = pl->mat[j * n_in + i];
                if (!c)
                    continue;
                em_mov_rm(e, RAX, R8, -1, 0, i * 8);
                em_movzx_rm8(e, RDX, RAX, R11, 0, 0);
                tail_mul_byte(cg, c);
            }
        } else if (pl->kind == P_DELTA) {
            /* acc seeded from oldpar above; per changed disk:
             * RDX = old[i][k] ^ new[i][k], then RBX ^= RDX * M[j][i] */
            int nc = pl->nchanged;
            for (int i = 0; i < nc; i++) {
                uint8_t c = pl->mat[(size_t)j * (size_t)nc + (size_t)i];
                if (!c)
                    continue;
                em_mov_rm(e, RAX, R8, -1, 0, i * 8);
                em_movzx_rm8(e, RDX, RAX, R11, 0, 0);
                em_mov_rm(e, RAX, R8, -1, 0, (nc + i) * 8);
                em_movzx_rm8(e, RAX, RAX, R11, 0, 0);
                em_xor_r8(e, RDX, RAX);
                tail_mul_byte(cg, c);
            }
        } else { /* P_HORNER */
            uint8_t c = pl->horner_c[j];
            for (int i = 0; i < n_in; i++) {
                em_mov_rm(e, RAX, R8, -1, 0, i * 8);
                em_movzx_rm8(e, RDX, RAX, R11, 0, 0);
                if (i == 0) {
                    /* RBX is not zeroed above (see per-j setup): at i==0
                     * RBX = 0*c + d = d for any c, a plain move. This
                     * replaces the whole log/exp/test/branch sequence. */
                    em_mov_r8r8(e, RBX, RDX);
                    continue;
                }
                if (c == 1) {
                    em_xor_r8(e, RBX, RDX);
                    continue;
                }
                if (c == 0) {
                    em_mov_r8r8(e, RBX, RDX); /* acc = d */
                    continue;
                }
                int lz = em_label(e), lnx = em_label(e), lns = em_label(e);
                em_test_rr(e, RBX, RBX);
                em_jcc(e, CC_E, lz);
                em_movzx_rm8(e, RAX, RDI, RBX, 0, 0); /* log[acc] */
                em_add_r32_ri(e, RAX, cg->logtab[c]);
                em_cmp_r32_ri(e, RAX, 255);
                em_jcc(e, CC_B, lns);
                em_add_r32_ri(e, RAX, -255);
                em_bind(e, lns);
                em_movzx_rm8(e, RAX, RSI, RAX, 0, 0);
                em_xor_r8(e, RAX, RDX);
                em_mov_r8r8(e, RBX, RAX);
                em_jmp(e, lnx);
                em_bind(e, lz);
                em_mov_r8r8(e, RBX, RDX);
                em_bind(e, lnx);
            }
        }
        em_mov_m8_r(e, R10, R11, 0, 0, RBX);
        /* The zero is dead for P_DELTA: lj is bound before the seed, so
         * every iteration re-seeds RBX with a full movzx (which clears
         * the whole register) before any use. */
        if (pl->kind != P_DELTA)
            em_xor_r32(e, RBX, RBX);
        em_inc_r(e, R11);
        em_cmp_rr(e, R11, RCX);
        em_jcc(e, CC_B, lj);
    }
    if (cg->cand.crc_fold)
        gen_fold_tail_chunks(cg);
    em_add_ri(e, RSP, 8);
}

/* Masked AVX-512 tail (track B1x86, SOTA item 14). Replaces the scalar
 * tail loop when cand->maskedtail. Entered with r11 = group-aligned byte
 * offset and rcx = len, and only when r11 < rcx (the caller checks), so
 * the tail (rcx - r11 bytes) is in 1..group-1.
 *
 * The tail is processed in 64-byte masked chunks reusing the u=0
 * accumulator slots (j*U). Per chunk:
 *   rdx = min(r10, 64); k1 = mask[rdx] (KMOVQ from the CK_TAILMASK pool);
 *   zero/seed the accumulators; masked body; masked stores;
 *   r11 += rdx; r10 -= rdx.
 * Chunk starts are always 64-byte aligned (r11 starts group-aligned and
 * only the last chunk is short), so the aligned masked forms are safe.
 * Masked-off lanes never fault (verified empirically on AMD EPYC 9D25:
 * see the maskfault probe, track Bx86); loads use {k1}{z} (zeroing) and
 * stores use {k1} (merging), so bytes past the tail are bit-exact.
 *
 * The arithmetic (xor / gfni affine / nibble shuffle) runs UNMASKED on
 * the zero-padded masked loads: masked-off lanes compute garbage, but
 * the masked store never writes them, so the result is bit-exact. (For
 * GFNI this relies on the store mask, not on masking the affine: the
 * affine's constant term b lands in masked-off lanes of the accumulator
 * but is never stored.)
 *
 * Evaluation order (each its own benchmark gate): P_XOR, then GFNI,
 * then nibble. Nibble masked tails are not implemented (the split +
 * dual-table lookup needs tlo/thi/r1/r2 live in the tail; see the gap
 * note in gen_tail_masked). */

/* Masked load of in[i] (or R8[idx*8] for P_DELTA's computed indices)
 * into cg->t, {k1}{z}. */
static void tail_masked_ld(cg_t *cg, int idx)
{
    em_t *e = &cg->em;
    int pr = hoist_inptr(cg, idx);
    if (pr >= 0) {
        em_vmovdqu8_ld_masked(e, cg->t, pr, R11, 0, 0, VL64, 1, 1);
    } else {
        em_mov_rm(e, RAX, R8, -1, 0, idx * 8);
        em_vmovdqu8_ld_masked(e, cg->t, RAX, R11, 0, 0, VL64, 1, 1);
    }
}

/* GFNI masked-tail multiply-accumulate for one matrix column:
 * acc[j] ^= t * mat[j][col] (c==1: plain xor; else affine via tbl).
 * Tables come from the pool (raw_ldpool, uncounted) with last-value
 * dedup tracked in tbl_off/tbl_valid. t2 is the affine temp. Raw em_
 * calls only: the tail is not part of the per-group static model. */
static void tail_gfni_mulacc(cg_t *cg, int col, int ncols,
                             size_t *tbl_off, int *tbl_valid)
{
    em_t *e = &cg->em;
    const plan_t *pl = cg->pl;
    int n_out = pl->n_out;
    for (int j = 0; j < n_out; j++) {
        uint8_t c = pl->mat[(size_t)j * (size_t)ncols + (size_t)col];
        if (!c)
            continue;
        int acc = j * cg->U;
        if (c == 1) {
            em_vpxor(e, acc, acc, cg->t, VL64);
        } else {
            size_t poff = pool_get(cg, CK_AFF, c);
            if (!*tbl_valid || *tbl_off != poff) {
                raw_ldpool(cg, cg->tbl, poff);
                *tbl_off = poff;
                *tbl_valid = 1;
            }
            em_vgf2p8affineqb(e, cg->t2, cg->t, cg->tbl, 0, VL64);
            em_vpxor(e, acc, acc, cg->t2, VL64);
        }
    }
}

/* P_MULACC GFNI masked tail body (one 64-byte chunk). */
static void gen_tail_masked_mulacc_gfni(cg_t *cg)
{
    em_t *e = &cg->em;
    const plan_t *pl = cg->pl;
    int n_in = pl->n_in, n_out = pl->n_out;
    for (int j = 0; j < n_out; j++)
        em_vzero(e, j * cg->U, VL64);
    size_t tbl_off = 0;
    int tbl_valid = 0;
    for (int i = 0; i < n_in; i++) {
        int any = 0;
        for (int j = 0; j < n_out; j++)
            if (pl->mat[(size_t)j * (size_t)n_in + (size_t)i]) {
                any = 1;
                break;
            }
        if (!any)
            continue;
        tail_masked_ld(cg, i);
        tail_gfni_mulacc(cg, i, n_in, &tbl_off, &tbl_valid);
    }
}

/* P_DELTA GFNI masked tail body (one 64-byte chunk). Accumulators are
 * seeded from oldpar (masked loads); per changed disk the delta
 * old[i]^new[i] is formed in t (via t2 when a nontrivial coefficient
 * needs the affine temp; t2/tbl are only assigned then) and
 * multiply-accumulated. */
static void gen_tail_masked_delta_gfni(cg_t *cg)
{
    em_t *e = &cg->em;
    const plan_t *pl = cg->pl;
    int n_out = pl->n_out, nc = pl->nchanged;
    for (int j = 0; j < n_out; j++) {
        em_mov_rm(e, RAX, R8, -1, 0, (2 * nc + j) * 8);
        em_vmovdqu8_ld_masked(e, j * cg->U, RAX, R11, 0, 0, VL64, 1, 1);
    }
    size_t tbl_off = 0;
    int tbl_valid = 0;
    for (int i = 0; i < nc; i++) {
        int any1 = 0, anyn = 0;
        for (int j = 0; j < n_out; j++) {
            uint8_t c = pl->mat[(size_t)j * (size_t)nc + (size_t)i];
            if (c == 1)
                any1 = 1;
            else if (c)
                anyn = 1;
        }
        if (!any1 && !anyn)
            continue;
        if (anyn) {
            /* Nontrivial coefficient: form the delta in t via t2
             * (assigned: anyn implies plan_any_nontrivial). */
            em_mov_rm(e, RAX, R8, -1, 0, i * 8);
            em_vmovdqu8_ld_masked(e, cg->t, RAX, R11, 0, 0, VL64, 1, 1);
            em_mov_rm(e, RAX, R8, -1, 0, (nc + i) * 8);
            em_vmovdqu8_ld_masked(e, cg->t2, RAX, R11, 0, 0, VL64, 1, 1);
            em_vpxor(e, cg->t, cg->t, cg->t2, VL64);
            tail_gfni_mulacc(cg, i, nc, &tbl_off, &tbl_valid);
        } else {
            /* All-trivial column: acc ^= old[i] ^ new[i] via two
             * masked loads and XORs; no t2 (unassigned here). */
            em_mov_rm(e, RAX, R8, -1, 0, i * 8);
            em_vmovdqu8_ld_masked(e, cg->t, RAX, R11, 0, 0, VL64, 1, 1);
            for (int j = 0; j < n_out; j++)
                if (pl->mat[(size_t)j * (size_t)nc + (size_t)i] == 1)
                    em_vpxor(e, j * cg->U, j * cg->U, cg->t, VL64);
            em_mov_rm(e, RAX, R8, -1, 0, (nc + i) * 8);
            em_vmovdqu8_ld_masked(e, cg->t, RAX, R11, 0, 0, VL64, 1, 1);
            for (int j = 0; j < n_out; j++)
                if (pl->mat[(size_t)j * (size_t)nc + (size_t)i] == 1)
                    em_vpxor(e, j * cg->U, j * cg->U, cg->t, VL64);
        }
    }
}

/* P_HORNER GFNI masked tail body (one 64-byte chunk). Horner recurrence
 * with masked input loads; tables from hreg (non-remat) or the pool
 * (remat, last-value dedup). acc=0 ^ t = t, so no i==0 special case is
 * needed (the affine's constant term is 0 for GF multiplies). */
static void gen_tail_masked_horner_gfni(cg_t *cg)
{
    em_t *e = &cg->em;
    const plan_t *pl = cg->pl;
    int n_in = pl->n_in, n_out = pl->n_out;
    for (int j = 0; j < n_out; j++)
        em_vzero(e, j * cg->U, VL64);
    size_t tbl_off = 0;
    int tbl_valid = 0;
    for (int i = 0; i < n_in; i++) {
        tail_masked_ld(cg, i);
        for (int j = 0; j < n_out; j++) {
            uint8_t c = pl->horner_c[j];
            int acc = j * cg->U;
            if (c == 1) {
                em_vpxor(e, acc, acc, cg->t, VL64);
            } else if (c == 0) {
                em_vmovdqa_rr(e, acc, cg->t, VL64);
            } else {
                int treg;
                if (cg->cand.remat) {
                    size_t poff = pool_get(cg, CK_AFF, c);
                    if (!tbl_valid || tbl_off != poff) {
                        raw_ldpool(cg, cg->tbl, poff);
                        tbl_off = poff;
                        tbl_valid = 1;
                    }
                    treg = cg->tbl;
                } else {
                    treg = cg->hreg[2 * j];
                }
                em_vgf2p8affineqb(e, cg->t2, acc, treg, 0, VL64);
                em_vpxor(e, acc, cg->t2, cg->t, VL64);
            }
        }
    }
}

/* Masked tail dispatcher: P_XOR and GFNI multiply tails share the
 * chunk loop (r10/rdx/k1 setup and masked stores); the per-chunk body
 * is kind-specific above. Masked-off lanes never fault (verified
 * empirically on AMD EPYC 9D25, track Bx86); loads use {k1}{z}
 * (zeroing) and stores use {k1} (merging), so bytes past the tail are
 * bit-exact. */
static void gen_tail_masked(cg_t *cg)
{
    em_t *e = &cg->em;
    const plan_t *pl = cg->pl;
    int n_in = pl->n_in, n_out = pl->n_out;

    assert(cg->cand.isa == ISA_AVX512);
    assert(cg->vl == 64);
    int gfni = cg->cand.mul == MUL_GFNI;
    /* Gate: P_XOR (plain body), or GFNI multiply tails. Nibble masked
     * tails are a separate, not-yet-implemented gate. */
    assert((pl->kind == P_XOR && !cg->cand.xorsched && !cg->cand.ternlog) ||
           (gfni && !cg->cand.permi2b &&
            (pl->kind == P_MULACC || pl->kind == P_DELTA ||
             pl->kind == P_HORNER)));

    size_t moff = pool_get(cg, CK_TAILMASK, 0);

    /* r10 = remaining tail bytes = rcx - r11 (> 0 on entry) */
    em_mov_rr(e, R10, RCX);
    em_sub_rr(e, R10, R11);

    int chunk_lab = em_label(e);
    int done_lab = em_label(e);
    em_bind(e, chunk_lab);
    /* rdx = min(r10, 64) */
    em_mov_rr(e, RDX, R10);
    em_cmp_ri(e, RDX, 64);
    int full_lab = em_label(e);
    em_jcc(e, CC_BE, full_lab);
    em_mov_ri(e, RDX, 64);
    em_bind(e, full_lab);
    /* k1 = mask[rdx]: the low rdx bits set */
    em_kmovq_km(e, 1, R15, RDX, 3, (int)moff);

    if (pl->kind == P_XOR) {
        /* zero the u=0 accumulators (registers j*U; masked-off lanes
         * stay zero through the unmasked XORs below). cg->t is the
         * temp. */
        for (int j = 0; j < n_out; j++)
            em_vzero(e, j * cg->U, VL64);
        /* per input column: masked load, xor into contributing rows */
        for (int i = 0; i < n_in; i++) {
            int any = 0;
            for (int j = 0; j < n_out; j++)
                if (pl->mat[(size_t)j * (size_t)n_in + (size_t)i]) {
                    any = 1;
                    break;
                }
            if (!any)
                continue;
            tail_masked_ld(cg, i);
            for (int j = 0; j < n_out; j++) {
                if (!pl->mat[(size_t)j * (size_t)n_in + (size_t)i])
                    continue;
                em_vpxor(e, j * cg->U, j * cg->U, cg->t, VL64);
            }
        }
    } else if (pl->kind == P_MULACC) {
        gen_tail_masked_mulacc_gfni(cg);
    } else if (pl->kind == P_DELTA) {
        gen_tail_masked_delta_gfni(cg);
    } else { /* P_HORNER */
        gen_tail_masked_horner_gfni(cg);
    }
    /* masked (merging) stores */
    for (int j = 0; j < n_out; j++) {
        em_mov_rm(e, RAX, R9, -1, 0, j * 8);
        em_vmovdqu8_st_masked(e, RAX, R11, 0, 0, j * cg->U, VL64, 1);
    }
    em_add_rr(e, R11, RDX);
    em_sub_rr(e, R10, RDX);
    em_jcc(e, CC_NE, chunk_lab);
    em_bind(e, done_lab);
}

static void gen_function(cg_t *cg)
{
    em_t *e = &cg->em;
    const rj_options *o = cg->opts;
    int loop_lab = em_label(e), tail_lab = em_label(e), done_lab = em_label(e);

    /* R15 (constant-pool base) is only needed when the build emits pool
     * loads. P_XOR never does; P_MULACC always does (planner guarantees
     * a coefficient outside {0,1}); P_DELTA/P_HORNER do iff the nibble
     * mask or log/exp tables are needed. The masked tail (track B1x86)
     * needs R15 for its tail-mask table. Skipping the push/lea/pop saves
     * 3 uops per call when the pool is empty. */
    int need_r15 = cg->pl->kind == P_MULACC ||
                   ((cg->pl->kind == P_DELTA || cg->pl->kind == P_HORNER) &&
                    (cg->need_mask || cg->need_logexp)) ||
                   cg->maskedtail || cg->cand.crc_fold;
    /* crc_fold needs R15 for the K1/K2 constant-pool loads at entry. */

    /* Pointer hoisting: decided before the pushes because HH[2..4]
     * (R12-R14) are callee-saved. */
    cg->nhoist = ptr_hoist_count(cg);

    if (o->emit_endbr)
        em_endbr64(e);
    if (o->emit_shadow_stack)
        em_setssbsy(e);
    em_push_r(e, RBX);
    if (need_r15)
        em_push_r(e, R15);
    if (cg->nhoist > 2) {
        em_push_r(e, R12);
        em_push_r(e, R13);
        em_push_r(e, R14);
    }
    /* REDUX crc_fold: the 4th argument (fold_state, RCX on entry) must
     * survive the whole kernel (entry loads + exit stores). No GPR is
     * free across body+tail (RSI/RDI are hoisted input pointers, RBX is
     * the tail accumulator), so spill it to the stack. It is pushed last
     * and popped first (into R10) at the single done_lab exit. */
    if (cg->cand.crc_fold)
        em_push_r(e, RCX);
    /* NOTE: the pushes leave rsp%16 as 0 or 8 inside the body depending
     * on the count (RBX [+R15] [+R12-R14 when nhoist>2]: 1, 2, 4 or 5
     * pushes -> rsp%16 is 0, 8, 8 or 0). This is safe
     * because generated code never calls out and never spills vector
     * registers to the stack -- every vector memory op is an aligned
     * MOVDQA to a caller buffer or the constant pool. If a call is ever
     * added, insert sub rsp,8 here (and add rsp,8 at the epilogue) to
     * restore the ABI 16-byte alignment. */
    em_mov_rr(e, R8, RDI);   /* in[] */
    em_mov_rr(e, R9, RSI);   /* out[] */
    em_mov_rr(e, RCX, RDX);  /* len */
    if (need_r15)
        em_lea_rip(e, R15, cg->pool_lab, 0); /* const-pool base */

    if (!cg->pl->noop) {
        /* Hoisted input pointers: in[k] never changes across iterations,
         * so load it once here. The body uses HH[k] via hoist_inptr(). */
        for (int k = 0; k < cg->nhoist; k++)
            em_mov_rm(e, HH[k], R8, -1, 0, k * 8);
        em_xor_rr(e, R11, R11); /* byte offset (before branch: tail needs it) */
        gen_const_setup(cg);
        /* Hoisted loop bound: rax = len - group, with CF set iff
         * len < group (no full group fits). The main loop is then a
         * do-while: body; r11 += group; cmp r11, rax; jbe loop. Per
         * iteration this is add + fused cmp/jcc = 2 uops, vs lea +
         * fused cmp/jcc + add + jmp = 4 uops for the old test-at-top
         * form. The bound is compare-based (no len/group division), so
         * group = U*VL need not be a power of two. RAX is dead in
         * gen_body (only the scalar tail uses it), so the bound
         * survives the loop. */
        em_mov_rr(e, RAX, RCX);       /* rax = len */
        em_sub_ri(e, RAX, cg->group); /* rax = len - group; CF iff len < group */
        /* A notail twin is only ever called with len % group == 0 (the
         * dispatcher guarantees it), so the loop always exits with
         * offset == len: route the exit straight to done_lab. The tail
         * path is not emitted. */
        em_jcc(e, CC_B, cg->notail ? done_lab : tail_lab);
        /* Pipe prologue (track I8): issue the first iteration's
         * first-column loads. After the len<group branch (a full group
         * fits), before the loop label; R11 == 0 here. */
        if (cg->cand.pipe) {
            int cols[RJ_MAX_DISKS];
            if (pipe_used_columns(cg, cols) > 0)
                gen_pipe_prologue(cg, cols[0]);
        }
        int al = o->loop_align ? o->loop_align
                               : (cg->vl == 64 ? 64 : cg->vl == 32 ? 32 : 16);
        em_align(e, (unsigned)al);
        em_bind(e, loop_lab);
        if (cg->cand.pipe)
            gen_body_pipe(cg);
        else
            gen_body(cg);
        em_add_ri(e, R11, cg->group); /* offset += group */
        em_cmp_rr(e, R11, RAX);       /* vs len - group */
        em_jcc(e, CC_BE, loop_lab);   /* while offset <= len - group */
        /* No SFENCE here: NT stores are weakly ordered and the jitted
         * code deliberately emits no fence. Callers that need the NT
         * data to be globally visible before subsequent reads must
         * execute SFENCE themselves after the call returns. */
        if (!cg->notail) {
            em_bind(e, tail_lab);
            em_cmp_rr(e, R11, RCX);
            em_jcc(e, CC_AE, done_lab);
            if (cg->maskedtail)
                gen_tail_masked(cg);
            else
                gen_tail(cg);
        }
        em_bind(e, done_lab);
    }

    /* REDUX crc_fold exit: store the N stream states back to fold_state
     * (the driver owns the buffer; the kernel only borrows it). R10 =
     * fold_state, popped LIFO before the other saved registers. */
    if (cg->cand.crc_fold) {
        em_pop_r(e, R10);
        for (int s = 0; s < cg->fold_n; s++)
            em_vmovdqa_st_evex128(e, R10, -1, 0, s * 16, cg->fold_xs0 + s);
    }
    if (cg->nhoist > 2) {
        em_pop_r(e, R14);
        em_pop_r(e, R13);
        em_pop_r(e, R12);
    }
    if (need_r15)
        em_pop_r(e, R15);
    em_pop_r(e, RBX);
    if (o->emit_shadow_stack)
        em_clrssbsy(e);
    /* Clear the upper vector state before returning: without this, a
     * caller executing legacy SSE after rj_fn pays the AVX-SSE transition
     * penalty (~tens of cycles, once per call). Gated on !is_sse — the SSE
     * path is legacy-encoded and never dirties the upper state. */
    if (!is_sse(cg))
        em_vzeroupper(e);
    em_ret(e);
    /* Trap sled after ret, unconditional: a fall-through past ret is a
     * codegen bug and must fault immediately (#UD), never slide through
     * NOPs into the const pool. */
    em_ud2(e);
}

/* ---------------- version build ---------------- */

int jit_build_version(const plan_t *pl, const cand_t *cand,
                      const rj_cpu_info *cpu, const rj_options *opts,
                      version_t *v, char *err, size_t errlen)
{
    /* REDUX crc_fold gate (spec §3.6; defense in depth -- the planner's
     * fused_crc_interleave_ok check in plan_candidates is the primary
     * filter). A crc_fold=1 candidate must never be built when the gate
     * is closed on this CPU (e.g. a hand-crafted plan ID reaching
     * rj_compile_pinned, or a stale planner): reject cleanly, never
     * wrong code, never an assertion abort. */
    if (cand->crc_fold && !fused_crc_interleave_ok(cpu)) {
        snprintf(err, errlen,
                 "x86_64 backend: crc_fold=1 but the interleaved-CRC "
                 "uarch gate is closed on this CPU");
        return -1;
    }
    /* REDUX crc_fold backend requirements. The fold is implemented for
     * the plain AVX-512 vector bodies only (P_XOR naive, P_MULACC,
     * P_HORNER with MUL_NIBBLE/MUL_GFNI): the fold hooks sit at the
     * input-vector loads and the parity stores, which the exotic body
     * twins (pipe/xorbit/xorsched/permi2b/mshift/remat/ternlog and the
     * masked/predicated tails) restructure or bypass. P_DELTA is out of
     * scope (spec §3.5: PARITY_UPDATE targets take the scalar CRC path).
     * nt/prefetch/notail are orthogonal (store policy / prefetch /
     * empty tail) and stay allowed. Reject incompatible combinations
     * cleanly -- never wrong code. */
    if (cand->crc_fold) {
        if (cand->isa != ISA_AVX512) {
            snprintf(err, errlen,
                     "x86_64 backend: crc_fold=1 requires ISA_AVX512");
            return -1;
        }
        /* Noop plans have no vector body to fold into; the exit
         * state-stores would write uninitialized registers. */
        if (pl->noop) {
            snprintf(err, errlen,
                     "x86_64 backend: crc_fold=1 incompatible with noop");
            return -1;
        }
        if (cand->mul != MUL_NIBBLE && cand->mul != MUL_GFNI) {
            snprintf(err, errlen,
                     "x86_64 backend: crc_fold=1 requires MUL_NIBBLE or "
                     "MUL_GFNI (got %d)", cand->mul);
            return -1;
        }
        if (pl->kind == P_DELTA) {
            snprintf(err, errlen,
                     "x86_64 backend: crc_fold=1 is ENCODE-only "
                     "(PARITY_UPDATE takes the scalar CRC path)");
            return -1;
        }
        if (cand->pipe || cand->xorbit || cand->xorsched || cand->permi2b ||
            cand->mshift || cand->remat || cand->maskedtail ||
            cand->ternlog || cand->predtail) {
            snprintf(err, errlen,
                     "x86_64 backend: crc_fold=1 incompatible with "
                     "pipe/xorbit/xorsched/permi2b/mshift/remat/maskedtail/"
                     "ternlog/predtail twins");
            return -1;
        }
    }
    (void)cpu;
    /* Rematerialization is implemented only for P_HORNER. User-reachable
     * via pinned candidates (WASM API): fail cleanly, never abort. */
    if (cand->remat && pl->kind != P_HORNER) {
        snprintf(err, errlen,
                 "x86_64 backend: remat twin needs P_HORNER");
        return -1;
    }
    /* Software-pipelined loop twin (track I8): implemented for the
     * P_MULACC/P_HORNER nibble bodies only (non-remat, no other twins).
     * Anything else fails the build cleanly -- never wrong code. The
     * planner's ISA gate keeps pipe off x86-64 (closed pending
     * quiet-box benchmarks); force-built here for differential
     * testing. */
    if (cand->pipe &&
        ((pl->kind != P_MULACC && pl->kind != P_HORNER) ||
         cand->mul != MUL_NIBBLE || cand->remat || cand->xorbit ||
         cand->xorsched || cand->permi2b || cand->mshift ||
         cand->ternlog || cand->maskedtail)) {
        snprintf(err, errlen,
                 "x86_64 backend: pipe twin needs P_MULACC/P_HORNER "
                 "nibble, non-remat, no other twins");
        return -1;
    }
    /* Recovery closed-form emitter twin (algos #4): implemented for
     * P_MULACC recovery plans; never combined with the other body
     * twins (xorbit/xorsched/permi2b/remat/mshift/pipe -- the planner
     * excludes those bases at enumeration). Anything else fails the
     * build cleanly -- never wrong code. */
    if (cand->rec_cf &&
        (pl->kind != P_MULACC || pl->rec_cf == NULL || cand->xorbit ||
         cand->xorsched || cand->permi2b || cand->remat || cand->mshift ||
         cand->pipe)) {
        snprintf(err, errlen,
                 "x86_64 backend: rec_cf twin needs P_MULACC without "
                 "other body twins");
        return -1;
    }
    /* Length-dispatch tail twins are disjoint: a version is never both.
     * The masked tail needs AVX-512 (opmask registers); the vpermi2b
     * twin needs the AVX-512 MULACC/DELTA nibble path. User-reachable
     * via pinned candidates: fail cleanly, never abort. */
    if (cand->maskedtail && cand->notail) {
        snprintf(err, errlen,
                 "x86_64 backend: maskedtail and notail are disjoint");
        return -1;
    }
    if (cand->maskedtail && cand->isa != ISA_AVX512) {
        snprintf(err, errlen,
                 "x86_64 backend: maskedtail needs AVX-512");
        return -1;
    }
    /* Masked tail plan-kind gate (mirrors api.c's twin gate and
     * gen_tail_masked's assertion): P_XOR plain body, or GFNI multiply.
     * Nibble masked tails are not implemented. */
    if (cand->maskedtail) {
        int gfni = cand->mul == MUL_GFNI;
        int ok = (pl->kind == P_XOR && !cand->xorsched && !cand->ternlog) ||
                 (gfni && !cand->permi2b &&
                  (pl->kind == P_MULACC || pl->kind == P_DELTA ||
                   pl->kind == P_HORNER));
        if (!ok) {
            snprintf(err, errlen,
                     "x86_64 backend: maskedtail needs P_XOR plain or GFNI multiply");
            return -1;
        }
    }
    if (cand->permi2b && (cand->isa != ISA_AVX512 ||
                          cand->mul != MUL_NIBBLE ||
                          (pl->kind != P_MULACC && pl->kind != P_DELTA))) {
        snprintf(err, errlen,
                 "x86_64 backend: permi2b needs AVX-512 MULACC/DELTA nibble");
        return -1;
    }
    /* Bit-program twin (track A1): only for GF-multiply rows, only from
     * a MUL_NIBBLE base, never combined with the other body twins.
     * User-reachable via pinned candidates: fail cleanly. */
    if (cand->xorbit &&
        (pl->kind != P_MULACC && pl->kind != P_DELTA &&
         pl->kind != P_HORNER)) {
        snprintf(err, errlen,
                 "x86_64 backend: xorbit needs GF-multiply rows");
        return -1;
    }
    if (cand->xorbit && cand->mul != MUL_NIBBLE) {
        snprintf(err, errlen,
                 "x86_64 backend: xorbit needs MUL_NIBBLE");
        return -1;
    }
    if (cand->xorbit &&
        (cand->ternlog || cand->permi2b || cand->remat ||
         cand->xorsched)) {
        snprintf(err, errlen,
                 "x86_64 backend: xorbit excludes other body twins");
        return -1;
    }
    /* The xorbit body follows pl->bit_sched; xorbit_build() returns NULL
     * when there is nothing to share. A NULL schedule would abort in
     * gen_body_xorbit's assertion. User-reachable via pinned candidates:
     * fail cleanly, never abort. */
    if (cand->xorbit && !pl->bit_sched) {
        snprintf(err, errlen,
                 "x86_64 backend: xorbit needs a plan with bit-level sharing");
        return -1;
    }
    /* VPMULTISHIFTQB nibble twin (track I2): AVX-512 MUL_NIBBLE
     * GF-multiply rows only (P_MULACC/P_DELTA/P_HORNER), never combined
     * with the other body twins (xorbit/permi2b/remat/xorsched are
     * different bodies/splits). Independent of ternlog (split vs fold).
     * The instruction is AVX512_VBMI; the api/planner gate additionally
     * requires cpu->vbmi before the twin can be selected.
     * User-reachable via pinned candidates: fail cleanly. */
    if (cand->mshift &&
        (cand->isa != ISA_AVX512 || cand->mul != MUL_NIBBLE ||
         (pl->kind != P_MULACC && pl->kind != P_DELTA &&
          pl->kind != P_HORNER))) {
        snprintf(err, errlen,
                 "x86_64 backend: mshift needs AVX-512 MULACC/DELTA/HORNER nibble");
        return -1;
    }
    if (cand->mshift &&
        (cand->xorbit || cand->permi2b || cand->remat ||
         cand->xorsched)) {
        snprintf(err, errlen,
                 "x86_64 backend: mshift excludes other body twins");
        return -1;
    }
    /* VPTERNLOGD twin: AVX-512 only (emits vpternlogd). The P_XOR
     * ternlog fold is selected by the planner; a pinned candidate can
     * request it on any ISA, so gate here. User-reachable via pinned
     * candidates: fail cleanly, never abort. */
    if (cand->ternlog && cand->isa != ISA_AVX512) {
        snprintf(err, errlen,
                 "x86_64 backend: ternlog needs AVX-512");
        return -1;
    }
    memset(v, 0, sizeof(*v));
    v->cand = *cand;
    v->kind = pl->kind;
    v->plan = pl; /* borrowed: bench validates against the owning plan */
    v->gbps = -1;
    v->vl = isa_vl(cand->isa);
    v->gbps_measured = 0;
    v->notail = cand->notail;
    v->maskedtail = cand->maskedtail;

    cg_t cg;
    memset(&cg, 0, sizeof(cg));
    cg.pl = pl;
    cg.cand = *cand;
    cg.opts = opts;
    cg.vl = isa_vl(cand->isa);
    cg.U = cand->U;
    cg.nt = cand->nt;
    cg.prefetch_dist = cand->prefetch_dist;
    cg.group = cg.U * cg.vl;
    cg.notail = cand->notail;
    cg.maskedtail = cand->maskedtail;
    /* predtail is an SVE-only twin (track B1sve, aarch64 backend):
     * reject cleanly rather than silently building a general twin. */
    if (cand->predtail) {
        snprintf(err, errlen,
                 "predtail twin requires SVE (not implemented for x86-64)");
        return -1;
    }
    /* lmul is an RVV-only axis: reject cleanly rather than silently
     * ignoring it. Only 0 (means 1) and 1 are valid here; the explicit
     * != form also rejects negative values (defense in depth). */
    if (cand->lmul != 0 && cand->lmul != 1) {
        snprintf(err, errlen,
                 "lmul=%d requires RVV (not implemented for x86-64)",
                 cand->lmul);
        return -1;
    }
    cg.permi2b = cand->permi2b;
    v->group_bytes = (size_t)cg.group;
    gf_build_logexp(cg.logtab, cg.exptab);

    /* scalar tail needs log/exp iff any real GF multiply is required.
     * The nibble vector path needs the 0x0F mask under the same
     * condition; P_DELTA with an all-{0,1} delta matrix (e.g. RAID-5)
     * needs neither. Single source of truth: plan_any_nontrivial (see
     * plan.h); the P_DELTA scan covers the n_out x nchanged delta
     * columns, not n_out x n_in.
     * A notail/maskedtail twin never emits the scalar tail, so it never
     * needs the log/exp tables (need_mask stays: the vector body needs
     * it). */
    if (cand->xorbit) {
        /* Bit-program twin: the vector body needs no 0x0F nibble mask,
         * but the scalar tail still needs log/exp for nontrivial
         * coefficients. */
        if (plan_any_nontrivial(pl))
            cg.need_logexp = !cand->notail && !cand->maskedtail;
    } else if (pl->kind == P_MULACC || pl->kind == P_DELTA ||
               pl->kind == P_HORNER) {
        if (plan_any_nontrivial(pl)) {
            cg.need_logexp = !cand->notail && !cand->maskedtail;
            cg.need_mask = 1;
        }
    }

    /* register assignment */
    int n_out = pl->n_out, U = cand->U;
    cg.t = n_out * U;
    int vmax = cg.t; /* highest assigned vector register index */
    if (cand->pipe) {
        /* Software-pipelined shape (track I8): the D/Dn datasets (U
         * regs each) replace the single input temp t (dead: -1).
         * Validation above rejects unsupported combos, so only
         * P_MULACC/P_HORNER x MUL_NIBBLE, non-remat, reach here. Must
         * stay in lockstep with plan_need_regs' pipe formulas
         * (asserted below). */
        assert(pl->kind == P_MULACC || pl->kind == P_HORNER);
        assert(cand->mul == MUL_NIBBLE && !cand->remat);
        cg.t = -1;
        cg.pipe_d = n_out * U;
        cg.pipe_dn = n_out * U + U;
        cg.tlo = n_out * U + 2 * U;
        cg.thi = cg.tlo + 1;
        cg.r1 = cg.tlo + 2;
        cg.r2 = cg.tlo + 3;
        if (pl->kind == P_MULACC) {
            cg.tbl = cg.tlo + 4;
            cg.mask = cg.tlo + 5;
            vmax = cg.mask;
        } else {
            /* P_HORNER: tables come from hreg (hoisted); the tbl
             * scratch is dead (remat, its only other user, is rejected
             * for the pipe twin). Left -1 so any accidental use trips
             * loudly. */
            cg.tbl = -1;
            cg.mask = cg.tlo + 4;
            vmax = cg.mask;
            cg.hreg = calloc((size_t)n_out * 2, sizeof(int));
            if (!cg.hreg) {
                snprintf(err, errlen, "out of memory");
                return -1;
            }
            for (int j = 0; j < n_out * 2; j++)
                cg.hreg[j] = -1;
            int r = vmax + 1;
            for (int j = 0; j < n_out; j++) {
                uint8_t c = pl->horner_c[j];
                if (c == 0 || c == 1)
                    continue;
                cg.hreg[2 * j] = r++;
                cg.hreg[2 * j + 1] = r++;
            }
            vmax = r - 1;
        }
    } else if (cand->xorbit) {
        /* Bit-program twin (track A1): n_out*U accumulators + input-byte
         * reg + two bit temps + 0x01 mask. The nibble tables are dead.
         * Must stay in lockstep with xorbit_need_regs. */
        assert(pl->bit_sched && pl->bit_sched->ntemps > 0);
        assert(pl->kind == P_MULACC || pl->kind == P_DELTA ||
               pl->kind == P_HORNER);
        cg.xbt_in = n_out * U;
        cg.xbt_r1 = n_out * U + 1;
        cg.xbt_r2 = n_out * U + 2;
        cg.xbt_mask = n_out * U + 3;
        vmax = cg.xbt_mask;
    }
    if (pl->kind == P_XOR && cand->xorsched && is_sse(&cg)) {
        /* XOR-schedule twin on SSE: t2 is allocated (the pressure
         * formula in xorsched.c counts it) but currently unused -- the
         * folded pxor-m128 form needs only the single temp t. Kept to
         * keep the backend's vmax in lockstep with plan_need_regs. */
        cg.t2 = cg.t + 1;
        vmax = cg.t2;
    }
    if (!cand->pipe && !cand->xorbit &&
        (pl->kind == P_MULACC || pl->kind == P_DELTA)) {
        if (cand->mul == MUL_NIBBLE) {
            if (cg.need_mask) {
                cg.tlo = cg.t + 1; cg.thi = cg.t + 2; cg.r1 = cg.t + 3;
                cg.r2 = cg.t + 4; cg.tbl = cg.t + 5;
                if (pl->kind == P_DELTA && is_avx512(&cg)) {
                    /* NIBHI pinning (item 7): tbl2 holds NIBHI for the
                     * last-used coefficient; mask shifts to t+7. */
                    cg.tbl2 = cg.t + 6; cg.mask = cg.t + 7;
                } else {
                    cg.mask = cg.t + 6;
                }
                vmax = cg.mask;
                if (cand->mshift) {
                    /* VPMULTISHIFTQB control (track I2): one more live
                     * register than the VPSRLW form. */
                    cg.msh = vmax + 1;
                    vmax = cg.msh;
                }
            } else {
                /* All coefficients in {0,1} (only P_DELTA reaches here:
                 * the planner guarantees a nontrivial coefficient for
                 * P_MULACC). The nibble split never executes, so
                 * tlo/thi/r2/tbl/mask are dead and not allocated. r1
                 * survives on SSE as the delta temp dt (the two-operand
                 * old^new); on AVX the delta folds into a memory XOR
                 * and only t remains. */
                if (pl->kind == P_DELTA && is_sse(&cg)) {
                    cg.r1 = cg.t + 1;
                    vmax = cg.r1;
                } else {
                    vmax = cg.t;
                }
            }
        } else {
            if (cg.need_mask) {
                cg.t2 = cg.t + 1; cg.tbl = cg.t + 2;
                vmax = cg.tbl;
            } else {
                /* All-trivial: the affine table and its temp are dead,
                 * and the delta temp dt (= t2) is only read on the SSE
                 * path, which GFNI never takes. */
                vmax = cg.t;
            }
        }
    } else if (!cand->pipe && pl->kind == P_HORNER && !cand->xorbit) {
        cg.hreg = calloc((size_t)n_out * 2, sizeof(int));
        if (!cg.hreg) {
            snprintf(err, errlen, "out of memory");
            return -1;
        }
        for (int j = 0; j < n_out * 2; j++)
            cg.hreg[j] = -1;
        if (cand->mul == MUL_NIBBLE) {
            cg.tlo = cg.t + 1; cg.thi = cg.t + 2; cg.r1 = cg.t + 3;
            cg.r2 = cg.t + 4; cg.mask = cg.t + 5;
            vmax = cg.mask;
            if (cand->mshift) {
                /* VPMULTISHIFTQB control (track I2): one more live
                 * register; the hreg table regs start after it.
                 * (remat is disjoint from mshift by assertion.) */
                cg.msh = vmax + 1;
                vmax = cg.msh;
            }
            if (cand->remat) {
                /* Rematerialize: tables reloaded from pool into tbl
                 * on each use; no hreg. */
                cg.tbl = cg.t + 6;
                vmax = cg.tbl;
            } else {
                int r = vmax + 1;
                for (int j = 0; j < n_out; j++) {
                    uint8_t c = pl->horner_c[j];
                    if (c == 0 || c == 1)
                        continue;
                    cg.hreg[2 * j] = r++;
                    cg.hreg[2 * j + 1] = r++;
                }
                vmax = r - 1;
            }
        } else {
            cg.t2 = cg.t + 1;
            if (cand->remat) {
                cg.tbl = cg.t + 2;
                vmax = cg.tbl;
            } else {
                vmax = cg.t2;
                int r = cg.t + 2;
                for (int j = 0; j < n_out; j++) {
                    uint8_t c = pl->horner_c[j];
                    if (c == 0 || c == 1)
                        continue;
                    cg.hreg[2 * j] = r++;
                }
                vmax = r - 1;
            }
        }
    }
    /* REDUX crc_fold register pool: N+5 lanes continuing the encode's
     * 0-based assignment (the vmax+1 == want_regs lockstep below forces
     * bottom-up continuation; see the cg_t field comment for why the
     * spec's top-down zmm31 placement is not used). E = encode regs.
     * The 5 VEX-bound temps/consts take E..E+4: VEX.128 PCLMULQDQ
     * addresses only xmm0-15, so E > 11 is rejected cleanly here
     * (fail-closed; the planner's 32-bound filter does not know about
     * the VEX limit). The N stream states take E+5... (EVEX-only). */
    cg.fold_xc = cg.fold_xa = cg.fold_xb = -1;
    cg.fold_xk1 = cg.fold_xk2 = -1;
    cg.fold_xs0 = -1;
    cg.fold_n = 0;
    if (cand->crc_fold) {
        int E = vmax + 1;
        if (E > 11) {
            snprintf(err, errlen,
                     "x86_64 backend: crc_fold=1 needs the 5 VEX temps in "
                     "xmm0-15 (encode uses %d regs)", E);
            free(cg.hreg);
            return -1;
        }
        cg.fold_xc = E;
        cg.fold_xa = E + 1;
        cg.fold_xb = E + 2;
        cg.fold_xk1 = E + 3;
        cg.fold_xk2 = E + 4;
        cg.fold_n = pl->n_in + pl->n_out;
        cg.fold_xs0 = E + 5;
        vmax = E + 4 + cg.fold_n;
    }
    /* Single source of truth: the planner's filter must agree exactly
     * with this assignment. A drift here once admitted candidates past
     * the regfile, and the emitter wraps reg 32 to 0 silently. The
     * xorsched twin has its own pressure formula (extra temp on SSE).
     * The mshift twin's control register (cg.msh, 0 when unallocated)
     * adds one: plan_need_regs has no mshift dimension because the
     * planner never enumerates the twin (I2 gate closed) -- if the gate
     * ever opens, the enumeration's pressure filter must add the same
     * +1 (see the plan_candidates I2 block). */
    int want_regs = cand->xorbit ? xorbit_need_regs(cand->isa, pl, U)
                    : cand->xorsched ? xorsched_need_regs(cand->isa, pl, U)
                                     : plan_need_regs(cand->isa, cand->mul, pl,
                                                     U, cand->remat, cand->pipe,
                                                     0, cand->crc_fold) +
                                       (cg.msh != 0 ? 1 : 0);
    assert(vmax + 1 == want_regs);
    /* Over-pressure must fail cleanly (return -1), not abort or emit
     * code that overruns the regfile. The planner's filter normally
     * prevents this, but force-built candidates (tests, future twins)
     * must also be safe. */
    if (vmax >= isa_nregs(cand->isa)) {
        snprintf(err, errlen,
                 "x86_64 backend: register pressure (needs %d regs, "
                 "isa has %d)", vmax + 1, isa_nregs(cand->isa));
        free(cg.hreg);
        return -1;
    }

    uint8_t *buf = malloc(1 << 20);
    if (!buf) {
        free(cg.hreg);
        snprintf(err, errlen, "out of memory");
        return -1;
    }
    em_init(&cg.em, buf, 1 << 20);
    cg.pool_lab = em_label(&cg.em);
    cg.pool_cap = 8192;
    cg.pool = malloc(cg.pool_cap);
    if (!cg.pool) {
        free(buf);
        free(cg.hreg);
        snprintf(err, errlen, "out of memory");
        return -1;
    }

    gen_function(&cg);

    /* constant pool after the code. The dead space between ret and the
     * pool is trap-filled (never NOPs): see em_trap_align. */
    em_trap_align(&cg.em, 64);
    em_bind(&cg.em, cg.pool_lab);
    size_t code_size = cg.em.len;
    for (size_t i = 0; i < cg.pool_len; i++)
        em_byte(&cg.em, cg.pool[i]);
    em_resolve(&cg.em);
    size_t total = cg.em.len;

    long ps = sysconf(_SC_PAGESIZE);
    size_t memsz = (total + (size_t)ps - 1) & ~((size_t)ps - 1);
#ifdef RJ_WASM
    /* Compile-only (WebAssembly, tools/wasm/SPEC.md §S6.2): no
     * mmap/mprotect in the browser. The emitters are
     * position-independent, so the bytes are the contract; keep them in
     * a plain malloc'd block. Native builds never define RJ_WASM. */
    (void)ps;
    void *mem = malloc(total ? total : 1);
    if (!mem) {
        snprintf(err, errlen, "malloc failed");
        goto fail;
    }
    memsz = total;
    memcpy(mem, buf, total);
#else
    void *mem = mmap(NULL, memsz, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) {
        snprintf(err, errlen, "mmap failed");
        goto fail;
    }
    memcpy(mem, buf, total);
    if (mprotect(mem, memsz, PROT_READ | PROT_EXEC) != 0) {
        snprintf(err, errlen, "mprotect RX failed");
        munmap(mem, memsz);
        goto fail;
    }
#endif

    v->mem = mem;
    v->memsz = memsz;
    v->fn = (rj_fn)mem;
    v->uops_per_byte = cg.uops / cg.group;
    v->mem_per_byte = cg.mem / cg.group;
    v->code_size = code_size;
    v->const_size = cg.pool_len;

    const char *strat = pl->kind == P_XOR    ? "xor"
                        : pl->kind == P_HORNER ? "horner"
                        : pl->kind == P_DELTA  ? "delta"
                                               : "mulacc";
    if (pl->kind == P_XOR)
    /* %.65s on the plan name: the provable -Wformat-truncation bound
     * for the 192B desc (fixed suffix <= 97B: " mulacc avx512+nibble"
     * + " U=8" + "+ternlog+nt+pf+remat" + " notail" + " maskedtail" +
     * "+permi2b" + "+xorbit" + "+mshift" + "+pipe" + "+reccf"; plan names can reach
     * 127B since the A5 "[rows ...]" suffix, capped at 65 by %.65s).
     * 65 + 97 = 162 < 192, so the bound holds with margin. */
        snprintf(v->desc, sizeof(v->desc), "%.65s %s %s U=%d%s%s%s%s%s%s%s%s%s%s%s", pl->name, strat,
                 isa_name(cand->isa), cand->U, cand->ternlog ? "+ternlog" : "",
                 cand->xorsched ? "+xsched" : "",
                 cand->xorbit ? "+xorbit" : "",
                 cand->nt ? "+nt" : "",
                 cand->prefetch_dist ? "+pf" : "",
                 cand->remat ? "+remat" : "",
                 cand->notail ? " notail" : "",
                 cand->maskedtail ? " maskedtail" : "",
                 cand->permi2b ? "+permi2b" : "",
                 cand->mshift ? "+mshift" : "",
                 cand->pipe ? "+pipe" : "");
    else
        snprintf(v->desc, sizeof(v->desc), "%.65s %s %s%s U=%d%s%s%s%s%s%s%s%s%s%s%s", pl->name, strat,
                 isa_name(cand->isa),
                 cand->mul == MUL_GFNI ? "+gfni" : "+nibble", cand->U,
                 cand->ternlog ? "+ternlog" : "",
                 cand->xorbit ? "+xorbit" : "",
                 cand->nt ? "+nt" : "",
                 cand->prefetch_dist ? "+pf" : "",
                 cand->remat ? "+remat" : "",
                 cand->notail ? " notail" : "",
                 cand->maskedtail ? " maskedtail" : "",
                 cand->permi2b ? "+permi2b" : "",
                 cand->mshift ? "+mshift" : "",
                 cand->pipe ? "+pipe" : "",
                 cand->rec_cf ? "+reccf" : "");

    /* Phantom twin check (BUG_t1_80): if ternlog was requested but zero
     * VPTERNLOGD instructions were emitted, the pin was a no-op. Reject
     * with a clean error instead of recording a false twin label. */
    if (cand->ternlog && cg.vpternlogd_emitted == 0) {
        if (err && errlen > 0)
            snprintf(err, errlen,
                     "ternlog twin requested but no VPTERNLOGD emitted "
                     "(plan has no foldable triples)");
        goto fail;
    }

    free(buf);
    free(cg.pool);
    free(cg.cent);
    free(cg.hreg);
    em_free(&cg.em);
    return 0;

fail:
    free(buf);
    free(cg.pool);
    free(cg.cent);
    free(cg.hreg);
    em_free(&cg.em);
    return -1;
}

void jit_free_version(version_t *v)
{
    if (v->mem) {
#ifdef RJ_WASM
        free(v->mem);
#else
        munmap(v->mem, v->memsz);
#endif
    }
    memset(v, 0, sizeof(*v));
}
