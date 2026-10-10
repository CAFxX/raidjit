/* raidjit AArch64 code generator (NEON + SVE/SVE2).
 *
 * Implements jit_build_version() from src/jit.h. Mirrors the x86-64
 * backend's algorithms (P_XOR / P_MULACC / P_DELTA / P_HORNER, constant
 * pool, scalar tail) with AArch64 instruction selection:
 *
 *  - NEON: 128-bit vectors, TBL (nibble method) for GF multiplies.
 *  - SVE:  VL-agnostic, z0-z7 + z16-z31 (logical r -> z(16+r) for r<16,
 *    z(r-16) for r in [16,24); z8-z15 callee-saved, never used), p0 only.
 *    LD1RQB replicates 16-byte pool tables across the vector length.
 *
 * Register conventions (also see src/aarch64/emit.h):
 *   x8=in[], x9=out[], x2=len, x11=byte offset, x16=pool base,
 *   x10=pointer temp, x14=2nd pointer (P_DELTA) / scratch,
 *   x15=SVE pool-addr temp / tail log-table base, x17=tail exp-table base,
 *   x12/x13=loop temps (SVE: x12=VL bytes, x13=group bytes).
 *   p1=predicated-tail chunk predicate (predtail twin only; p0 stays the
 *   main loop's all-true predicate).
 *
 * Non-temporal stores: NEON uses STNP pairs (an odd trailing vector falls
 * back to a temporal STR); SVE uses STNT1B. The jitted code executes NO
 * fence after NT stores: callers that need the NT data to be globally
 * visible before subsequent reads must execute DMB ISHST themselves
 * after the call returns.
 */
#include "jit.h"
#include "plan.h"
#include "emit.h"
#include "gf.h"
#include "xorsched.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>

/* constant-pool entry kinds (16-byte tables, 64-byte aligned) */
enum { CK_NIBLO, CK_NIBHI, CK_LOG, CK_EXP };

typedef struct {
    int kind;
    uint8_t c;
    size_t off;
} poolent_t;

typedef struct {
    const plan_t *pl;
    cand_t cand;
    const rj_options *opts;
    const rj_cpu_info *cpu;
    int is_sve;
    int vl;          /* vector length in bytes: 16 (NEON) or sve_vl (bytes) */
    int U, group;    /* group = U*vl bytes per vector-loop iteration */
    int t, tlo, thi, r1, r2, tbl_lo, tbl_hi, mask; /* logical vector regs */
    int t2; /* XOR-schedule 2nd temp (P_XOR xorsched twin only) */
    /* Bit-program (xorbit) twin registers: input-byte reg, two bit-temp
     * regs, and the 0x01 mask reg (see xorbit_need_regs). */
    int xbt_in, xbt_r1, xbt_r2, xbt_mask;
    /* Software-pipelined loop twin (track I8, cand->pipe): base logical
     * regs of the per-column double-buffered datasets D (pipe_d, U
     * regs) and Dn (pipe_dn, U regs). In pipe mode the single input
     * temp t is dead (replaced by D/Dn); tlo/thi/r1/r2, the table regs
     * and mask keep their roles. -1 when the pipe twin is off. */
    int pipe_d, pipe_dn;
    int *hreg;       /* P_HORNER table regs, 2 per row */
    int vmax;        /* highest logical vector reg touched (NEON save/restore) */
    em_t em;
    int pool_lab;
    uint8_t *pool;
    size_t pool_len, pool_cap;
    poolent_t *cent;
    int ncent, ccent;
    int need_mask, need_logexp;
    int nt;
    unsigned prefetch_dist;
    int notail;      /* length-dispatch twin: omit the scalar tail entirely;
                      * only valid for len % group == 0 (see
                      * docs/length_dispatch.md) */
    int predtail;    /* length-dispatch twin: replace the scalar tail with a
                      * fully predicated vector tail (SVE whilelt; track
                      * B1sve). Disjoint from notail (asserted in
                      * jit_build_version). Serves tail lengths only
                      * (TAIL_MASKED, see docs/length_dispatch.md §12). */
    int pred;        /* predicated single-lane body mode: set only around
                      * gen_pred_tail. Leaf helpers then use predicate p1
                      * (generated per chunk by whilelt), run the u=0 lane
                      * only, and skip per-lane pointer bumps (the chunk
                      * loop recomputes pointers from x11 each iteration). */
    /* use_eor3: SVE2 EOR3 codegen for the nibble multiply reduction
     * (set in jit_build_version; implies is_sve && cpu->sve2). When set,
     * the HORNER body folds lo^hi^input into one EOR3 and the
     * MULACC/DELTA accumulate path folds acc^r1^r2 into one EOR3. The
     * gate is exact: it is set iff at least one EOR3 will be emitted, so
     * the "+eor3" desc tag means "this build contains EOR3 instructions".
     * Register allocation is unchanged (no new live regs), so the
     * pressure formulas hold. */
    int use_eor3;
    uint8_t logtab[256], exptab[512];
    double uops, mem;
} cg_t;

/* logical vector reg -> physical: NEON identity; SVE: r in [0,15] ->
 * z(16+r), r in [16,23] -> z(r-16), i.e. z0-z7. z8-z15 are callee-saved
 * per AAPCS64 and are never touched. This gives SVE 24 usable vector
 * registers (z0-z7 + z16-z31), raising the (n_out,U) ceiling by 50%
 * over the old z16-z31-only mapping. */
static int V(const cg_t *cg, int r)
{
    if (!cg->is_sve)
        return r;
    assert(r >= 0 && r < 24);
    return r < 16 ? 16 + r : r - 16;
}

/* ---------------- constant pool ---------------- */

static size_t pool_get(cg_t *cg, int kind, uint8_t c)
{
    for (int i = 0; i < cg->ncent; i++)
        if (cg->cent[i].kind == kind && cg->cent[i].c == c)
            return cg->cent[i].off;
    uint8_t tmp[512];
    size_t len = 0;
    if (kind == CK_NIBLO || kind == CK_NIBHI) {
        uint8_t lo[16], hi[16];
        gf_build_nibble(c, lo, hi);
        memcpy(tmp, (kind == CK_NIBLO) ? lo : hi, 16);
        len = 16;
    } else if (kind == CK_LOG) {
        memcpy(tmp, cg->logtab, 256);
        len = 256;
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

/* dst = x16 (pool base) + off */
static void c_pooladdr(cg_t *cg, int dst, size_t off)
{
    em_t *e = &cg->em;
    if (off < 4096) {
        em_add_ri(e, dst, X16, (unsigned)off);
    } else {
        unsigned hi = (unsigned)(off >> 12), lo = (unsigned)(off & 0xfff);
        em_movz(e, dst, hi);
        em_lsl_ri(e, dst, dst, 12);
        if (lo)
            em_add_ri(e, dst, dst, lo);
        em_add_rr(e, dst, dst, X16);
    }
}

static void c_ldpool(cg_t *cg, int r, int kind, unsigned c)
{
    assert(r >= 0); /* catches the remat hreg=-1 hazard at codegen time */
    em_t *e = &cg->em;
    size_t off = pool_get(cg, kind, (uint8_t)c);
    if (cg->is_sve) {
        /* LD1RQB has no offset: materialize the address in x15 */
        c_pooladdr(cg, X15, off);
        em_sve_ld1rqb(e, V(cg, r), 0, X15);
        cg->uops += 2;
    } else {
        assert(off < 65536);
        em_neon_ldr(e, r, X16, (unsigned)off);
        cg->uops += 1;
    }
    cg->mem += 16;
}

/* ---------------- vector helpers ---------------- */

/* load lane u of the vector at ptr into logical reg r (ptr holds the
 * lane-0 address: base+offset, never bumped). SVE addresses lane u with
 * the contiguous [xRn, #u, MUL VL] offset form, so the base pointer stays
 * fixed and no per-lane ADD executes in the loop. The MUL VL immediate
 * is a signed 4-bit value (-8..7); U <= 8 always fits (asserted in the
 * emitter). In predicated-tail mode (cg->pred) the load runs under p1
 * with u == 0 (plain [xRn] form). */
static void c_ld_to(cg_t *cg, int r, int u, int ptr)
{
    em_t *e = &cg->em;
    if (cg->is_sve) {
        if (cg->pred)
            em_sve_ld1b(e, V(cg, r), 1, ptr);
        else
            em_sve_ld1b_off(e, V(cg, r), 0, ptr, u);
    } else {
        assert(!cg->pred); /* predicated tails are SVE-only */
        em_neon_ldr(e, r, ptr, (unsigned)u * 16);
    }
    cg->uops += 1;
    cg->mem += cg->vl;
}

static void c_st_to(cg_t *cg, int r, int u, int ptr)
{
    em_t *e = &cg->em;
    if (cg->is_sve) {
        if (cg->pred) {
            if (cg->nt)
                em_sve_stnt1b(e, V(cg, r), 1, ptr);
            else
                em_sve_st1b(e, V(cg, r), 1, ptr);
        } else if (cg->nt) {
            em_sve_stnt1b_off(e, V(cg, r), 0, ptr, u);
        } else {
            em_sve_st1b_off(e, V(cg, r), 0, ptr, u);
        }
    } else {
        assert(!cg->pred); /* predicated tails are SVE-only */
        em_neon_str(e, r, ptr, (unsigned)u * 16);
    }
    cg->uops += 1;
    cg->mem += cg->vl;
}

/* d ^= s. Predicated-tail mode uses the merging predicated EOR under p1:
 * active lanes get d^s, inactive lanes keep d (never stored). */
static void c_xor(cg_t *cg, int d, int s)
{
    if (cg->is_sve)
        em_sve_eor(&cg->em, V(cg, d), cg->pred ? 1 : 0, V(cg, s));
    else {
        assert(!cg->pred); /* predicated tails are SVE-only */
        em_neon_eor(&cg->em, d, d, s);
    }
    cg->uops += 1;
}

/* d ^= x ^ y via a single SVE2 EOR3 (Zdn ^= Zm ^ Za, unpredicated).
 * d must hold a LIVE value: EOR3 folds d's current contents in, so it
 * is only valid on accumulate paths, never where d is dead (e.g. the
 * mulacc_column first-column path, where the destination is merely dead
 * rather than zero). The assert tripwires emission to builds that
 * selected the SVE2 codegen variant (implies cpu->sve2). */
static void c_eor3(cg_t *cg, int d, int x, int y)
{
    assert(cg->use_eor3);
    em_sve_eor3(&cg->em, V(cg, d), V(cg, x), V(cg, y));
    cg->uops += 1;
}

/* d = s (copy) */
static void c_copy(cg_t *cg, int d, int s)
{
    if (cg->is_sve)
        em_sve_mov(&cg->em, V(cg, d), V(cg, s));
    else
        em_neon_orr(&cg->em, d, s, s);   /* vD = vS|vS = vS */
    cg->uops += 1;
}

/* d = s & 0x0F. SVE: single unpredicated AND. Sound: the old
 * predicated form ran under an all-true p0 at every call site (all 6
 * are in gen_body; p0 = whilelo(xzr, x12=cntb) is set once in the
 * prologue and never modified in the loop), so the intended semantics
 * is a full-vector AND, which the unpredicated form delivers without
 * touching p0. Saves 1 uop here (the c_copy), 2 per nibble-split site.
 * Predicated-tail mode keeps the unpredicated 3-operand form: it is
 * lane-independent, and every consumer of d is lane-independent (TBL)
 * or predicated-merging (XOR/EOR) feeding a predicated store, so the
 * inactive lanes ((inactive s) & mask) are never observed. (The
 * destructive predicated AND, d &= mask, cannot express d = s & mask.) */
static void c_and3(cg_t *cg, int d, int s)
{
    em_t *e = &cg->em;
    if (cg->is_sve) {
        em_sve_and_unpred(e, V(cg, d), V(cg, s), V(cg, cg->mask));
    } else {
        assert(!cg->pred); /* predicated tails are SVE-only */
        em_neon_and(e, d, s, cg->mask);
    }
    cg->uops += 1;
}

/* d = s >> 4 (upper bits zero). SVE: single unpredicated LSR;
 * same p0 argument as c_and3. Predicated-tail mode keeps the
 * unpredicated form for the same lane-independence reason as c_and3:
 * every consumer of d is TBL (lane-independent) or predicated, so
 * inactive lanes are never observed. (The destructive predicated LSR,
 * d >>= 4, cannot express d = s >> 4.) */
static void c_lsr4(cg_t *cg, int d, int s)
{
    em_t *e = &cg->em;
    if (cg->is_sve) {
        em_sve_lsr_unpred(e, V(cg, d), V(cg, s), 4);
    } else {
        assert(!cg->pred); /* predicated tails are SVE-only */
        em_neon_ushr(e, d, s, 4);
    }
    cg->uops += 1;
}

/* ---- Bit-program (xorbit) twin helpers ----
 * Mirror the x86 c_srlw/c_and/c_sllw trio for the bit-program body.
 * All forms are 1 uop (no SSE-style mov+shift penalty: NEON has
 * 3-operand shifts via the new emitters, SVE uses unpredicated forms).
 * The xorbit body is lane-independent, so unpredicated is sound (same
 * argument as c_and3/c_lsr4). */

/* d = s >> k, k in 1..7. Caller skips k=0 (no shift needed). */
static void c_xbt_srl(cg_t *cg, int d, int s, unsigned k)
{
    em_t *e = &cg->em;
    assert(k >= 1 && k <= 7);
    if (cg->is_sve) {
        em_sve_lsr_unpred(e, V(cg, d), V(cg, s), k);
    } else {
        assert(!cg->pred);
        em_neon_ushr(e, d, s, k);
    }
    cg->uops += 1;
}

/* d = s & 0x01, using the xorbit 0x01 mask register (xbt_mask). */
static void c_xbt_and1(cg_t *cg, int d, int s)
{
    em_t *e = &cg->em;
    if (cg->is_sve) {
        em_sve_and_unpred(e, V(cg, d), V(cg, s), V(cg, cg->xbt_mask));
    } else {
        assert(!cg->pred);
        em_neon_and(e, d, s, cg->xbt_mask);
    }
    cg->uops += 1;
}

/* d = s << b, b in 1..7. Caller uses c_copy for b=0. */
static void c_xbt_sll(cg_t *cg, int d, int s, unsigned b)
{
    em_t *e = &cg->em;
    assert(b >= 1 && b <= 7);
    if (cg->is_sve) {
        em_sve_lsl_unpred(e, V(cg, d), V(cg, s), b);
    } else {
        assert(!cg->pred);
        em_neon_shl(e, d, s, b);
    }
    cg->uops += 1;
}

/* ---- Bit-program (xorbit) body ---- */

/* Zero a vector register (for xorbit dead-byte zeroing). */
static void c_xbt_zero(cg_t *cg, int r)
{
    em_t *e = &cg->em;
    if (cg->is_sve)
        em_sve_dup(e, V(cg, r), 0);
    else {
        assert(!cg->pred);
        em_neon_movi(e, r, 0);
    }
    cg->uops += 1;
}

/* Forward declarations (defined below, used by xorbit code) */
static void c_inptr(cg_t *cg, int dst, int idx);
static void c_inptr_np(cg_t *cg, int dst, int idx);

/* Fan out one bit-0-normalized value (vsrc) to a temp's / input bit's
 * user output bits, grouped by bit position b (users sorted ascending
 * by output-bit id). Mirrors x86 xorbit_fanout exactly. */
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
                c_copy(cg, acc0, vsrc);
            else
                c_xbt_sll(cg, acc0, vsrc, (unsigned)b);
            touched[j0] = 1;
        } else if (b == 0) {
            for (int x = w; x < w1; x++) {
                int j = users[x] >> 3;
                int acc = j * U + u;
                if (!is_delta && !touched[j]) {
                    c_copy(cg, acc, vsrc);
                    touched[j] = 1;
                } else {
                    c_xor(cg, acc, vsrc);
                }
            }
        } else {
            c_xbt_sll(cg, vtmp, vsrc, (unsigned)b);
            for (int x = w; x < w1; x++) {
                int j = users[x] >> 3;
                int acc = j * U + u;
                if (!is_delta && !touched[j]) {
                    c_copy(cg, acc, vtmp);
                    touched[j] = 1;
                } else {
                    c_xor(cg, acc, vtmp);
                }
            }
        }
        w = w1;
    }
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
    (void)vl; /* vl used via c_ld_to's u parameter */
    /* Predicated-tail mode processes one lane per chunk (the u=0
     * slice); the chunk loop re-enters gen_body per VL-sized chunk.
     * Computing the other lanes would be dead work: gen_stores only
     * stores the u=0 slice in pred mode. */
    int ulanes = cg->pred ? 1 : U;

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
                for (int u = 0; u < ulanes; u++)
                    c_xbt_zero(cg, j * U + u);
        }
    } else {
        /* seed accumulators from oldpar (like the naive DELTA body) */
        int nc = pl->nchanged;
        for (int j = 0; j < n_out; j++) {
            c_inptr(cg, X10, 2 * nc + j);
            for (int u = 0; u < ulanes; u++)
                c_ld_to(cg, j * U + u, u, X10);
        }
    }

    int pf_done[RJ_MAX_DISKS];
    for (int i = 0; i < ni_sched; i++)
        pf_done[i] = 0;
    int touched[RJ_MAX_DISKS];

    for (int u = 0; u < ulanes; u++) {
        for (int j = 0; j < n_out; j++)
            touched[j] = 0;
        int cur_in = -1;
        /* phase 1: form each temp, fan out immediately */
        for (int m = 0; m < s->ntemps; m++) {
            int a = s->temps[m].a, b = s->temps[m].b;
            int ia = a >> 3, ka = a & 7;
            int ib = b >> 3, kb = b & 7;
            if (cur_in != ia) {
                if (!pf_done[ia]) {
                    c_inptr(cg, X10, ia);
                    pf_done[ia] = 1;
                } else {
                    c_inptr_np(cg, X10, ia);
                }
                c_ld_to(cg, inreg, u, X10);
                cur_in = ia;
            }
            /* Note: always shift+and, even for ka=0, to stay in lockstep
             * with xorbit_version_uops (which counts 2 uops here for
             * non-SSE). A shift-by-0 is a legal no-op. */
            if (ka == 0)
                c_copy(cg, r1, inreg);
            else
                c_xbt_srl(cg, r1, inreg, (unsigned)ka);
            c_xbt_and1(cg, r1, r1);
            if (cur_in != ib) {
                if (!pf_done[ib]) {
                    c_inptr(cg, X10, ib);
                    pf_done[ib] = 1;
                } else {
                    c_inptr_np(cg, X10, ib);
                }
                c_ld_to(cg, inreg, u, X10);
                cur_in = ib;
            }
            if (kb == 0) {
                c_copy(cg, r2, inreg);
                c_xbt_and1(cg, r2, r2);
            } else {
                c_xbt_srl(cg, r2, inreg, (unsigned)kb);
                c_xbt_and1(cg, r2, r2);
            }
            c_xor(cg, r1, r2);
            xorbit_fanout(cg, s->temps[m].users, s->temps[m].nusers, r1,
                          r2, u, touched, is_delta);
        }
        /* phase 2: surviving input bits, fanned out directly */
        for (int q = 0; q < 8 * ni_sched; q++) {
            if (!s->inbit_nusers[q])
                continue;
            int i = q >> 3, k = q & 7;
            if (cur_in != i) {
                if (!pf_done[i]) {
                    c_inptr(cg, X10, i);
                    pf_done[i] = 1;
                } else {
                    c_inptr_np(cg, X10, i);
                }
                c_ld_to(cg, inreg, u, X10);
                cur_in = i;
            }
            if (k == 0) {
                c_copy(cg, r1, inreg);
                c_xbt_and1(cg, r1, r1);
            } else {
                c_xbt_srl(cg, r1, inreg, (unsigned)k);
                c_xbt_and1(cg, r1, r1);
            }
            xorbit_fanout(cg, s->inbit_users[q], s->inbit_nusers[q], r1,
                          r2, u, touched, is_delta);
        }
    }
}

/* d = table[idx]; table is the logical reg tblr */
static void c_tbl_into(cg_t *cg, int d, int tblr, int idx)
{
    em_t *e = &cg->em;
    if (cg->is_sve)
        em_sve_tbl(e, V(cg, d), V(cg, tblr), V(cg, idx));
    else
        em_neon_tbl(e, d, tblr, idx);
    cg->uops += 1;
}

/* dst = in[idx] + x11; prefetch dist bytes ahead if enabled.
 * Prefetch is suppressed in predicated-tail mode: the tail is shorter
 * than one group and needs no lookahead. */
static void c_inptr(cg_t *cg, int dst, int idx)
{
    em_t *e = &cg->em;
    em_ldr64(e, dst, X8, (unsigned)idx * 8);
    em_add_rr(e, dst, dst, X11);
    if (cg->prefetch_dist && !cg->pred) {
        em_prfm(e, PRF_PLDL1STRM, dst, cg->prefetch_dist);
        cg->uops += 1;
    }
}

/* x10 = out[j] + x11 */
static void c_outptr(cg_t *cg, int j)
{
    em_t *e = &cg->em;
    em_ldr64(e, X10, X9, (unsigned)j * 8);
    em_add_rr(e, X10, X10, X11);
}

/* dst = in[idx] + x11, without the prefetch (for inputs already
 * prefetched by an earlier c_inptr in the xorsched body) */
static void c_inptr_np(cg_t *cg, int dst, int idx)
{
    em_t *e = &cg->em;
    em_ldr64(e, dst, X8, (unsigned)idx * 8);
    em_add_rr(e, dst, dst, X11);
}

/* ---------------- const setup ---------------- */

static void gen_const_setup(cg_t *cg)
{
    const plan_t *pl = cg->pl;
    em_t *e = &cg->em;
    if (cg->cand.xorbit) {
        /* Bit-program twin: set up the 0x01 mask register. Prologue
         * const setup is not per-iteration work: uncounted, by the
         * same convention as the nibble mask/table setup (and x86's
         * mask load). */
        if (cg->is_sve)
            em_sve_dup(e, V(cg, cg->xbt_mask), 1);
        else
            em_neon_movi(e, cg->xbt_mask, 1);
    } else if ((pl->kind == P_MULACC || pl->kind == P_DELTA) && cg->need_mask) {
        if (cg->is_sve)
            em_sve_dup(e, V(cg, cg->mask), 15);
        else
            em_neon_movi(e, cg->mask, 15);
        cg->uops += 1;
    } else if (pl->kind == P_HORNER) {
        /* The nibble mask is only needed when some row coefficient is
         * nontrivial (outside {0,1}); gate the dup on the planner's
         * nontrivial predicate (same source of truth as MULACC/DELTA's
         * need_mask). Saves 1 prologue uop for all-trivial Horner. */
        if (plan_any_nontrivial(pl)) {
            if (cg->is_sve)
                em_sve_dup(e, V(cg, cg->mask), 15);
            else
                em_neon_movi(e, cg->mask, 15);
            cg->uops += 1;
        }
        /* In remat mode hreg[] stays -1 (no persistent table regs);
         * tables are reloaded into tbl on each use instead. */
        if (!cg->cand.remat) {
            for (int j = 0; j < pl->n_out; j++) {
                uint8_t c = pl->horner_c[j];
                if (c == 0 || c == 1)
                    continue;
                c_ldpool(cg, cg->hreg[2 * j], CK_NIBLO, c);
                c_ldpool(cg, cg->hreg[2 * j + 1], CK_NIBHI, c);
            }
        }
    }
}

/* ---------------- body ---------------- */

/* Column scan info: does any row use column col, and does any need the
 * nibble path (coefficient outside {0,1})? */
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
 * mat is n_out x ncols row-major.
 *
 * Loop nest is (column -> row -> lane): for each row j with a nonzero
 * coefficient, the nibble tables are loaded ONCE per (j,col) into the
 * two persistent table regs (tbl_lo/tbl_hi) -- they depend only on
 * (j,col), not on the lane -- and then the u-lane loop runs the input
 * load, nibble split, TBL and accumulate. This hoisting saves 4*(U-1)
 * table-load uops per nontrivial (j,col) on SVE (2*(U-1) on NEON) at
 * the price of reloading the input lane per row (L1 hits; the input
 * pointer X10/X3 stays fixed across lanes via MUL VL addressing, set
 * up once per column by the caller).
 *
 * first_col[j] is the first column with a nonzero entry in row j, or
 * NULL when accumulators are not provably zero (P_DELTA seeds from
 * oldpar). At the first contributing column the accumulator is known
 * zero, so the update is a move instead of an xor-with-zero.
 * is_delta selects the P_DELTA input formation (t = old^new via X10/X3)
 * versus P_MULACC (t = input via X10).
 * HORNER never reaches this function: it uses the inline i-loop in
 * gen_body, not mulacc_column.
 * staged >= 0 selects the software-pipelined mode (track I8): the input
 * lanes for this column were issued by an earlier load step into the
 * dataset at logical regs [staged, staged+U), so the per-(j,u) loads
 * below are skipped and the data is read from staged+u instead of t.
 * Pass -1 for the classic load-per-(j,u) form. */
static void mulacc_column(cg_t *cg, int col, int ncols,
                          const int *first_col, int is_delta, int staged)
{
    const plan_t *pl = cg->pl;
    int n_out = pl->n_out;
    int U = cg->U;
    /* Predicated-tail mode processes one lane per chunk (the u=0 slice). */
    int ulanes = cg->pred ? 1 : U;
    assert(pl->kind != P_HORNER);
    assert(staged < 0 || !is_delta); /* pipe excludes P_DELTA */
    for (int j = 0; j < n_out; j++) {
        uint8_t c = pl->mat[(size_t)j * (size_t)ncols + (size_t)col];
        if (!c)
            continue;
        int first = first_col && col == first_col[j];
        if (c != 1) {
            /* Hoisted table load: depends only on (j,col). */
            c_ldpool(cg, cg->tbl_lo, CK_NIBLO, c);
            c_ldpool(cg, cg->tbl_hi, CK_NIBHI, c);
        }
        for (int u = 0; u < ulanes; u++) {
            int acc = j * U + u;
            /* In staged (pipe) mode the data register replaces t; the
             * load below is skipped (issued earlier into the dataset). */
            int t = (staged >= 0) ? staged + u : cg->t;
            if (staged < 0) {
                if (is_delta) {
                    c_ld_to(cg, cg->t, u, X10);
                    c_ld_to(cg, cg->r1, u, X3);
                    c_xor(cg, cg->t, cg->r1);
                } else {
                    c_ld_to(cg, cg->t, u, X10);
                }
            }
            if (c == 1) {
                if (first)
                    c_copy(cg, acc, t); /* acc was 0 */
                else
                    c_xor(cg, acc, t);
            } else {
                /* Nontrivial: split t into nibbles, then TBL. The
                 * split is per (j,u) -- only rows with c != 1 need it. */
                c_and3(cg, cg->tlo, t);
                c_lsr4(cg, cg->thi, t);
                c_tbl_into(cg, cg->r1, cg->tbl_lo, cg->tlo);
                c_tbl_into(cg, cg->r2, cg->tbl_hi, cg->thi);
                if (first) {
                    /* acc is dead on this path (first contributing
                     * column; the accumulator is overwritten before its
                     * first read), so acc = r1^r2 is a single
                     * unpredicated EOR on SVE -- no mov needed, and the
                     * unpredicated form never reads the dead acc. EOR3
                     * is NOT usable here: it computes Zdn^Zm^Za, i.e. it
                     * folds acc's current value in, but acc is merely
                     * dead (uninitialized), not zero. NEON keeps mov+eor
                     * (its EOR is already 3-operand but the c_xor helper
                     * is destructive). */
                    if (cg->is_sve) {
                        em_sve_eor_unpred(&cg->em, V(cg, acc), V(cg, cg->r1),
                                          V(cg, cg->r2));
                        cg->uops += 1;
                    } else {
                        c_copy(cg, acc, cg->r1);
                        c_xor(cg, acc, cg->r2);
                    }
                } else if (cg->use_eor3) {
                    /* acc live: acc ^= r1 ^ r2 in one instruction */
                    c_eor3(cg, acc, cg->r1, cg->r2);
                } else {
                    c_xor(cg, acc, cg->r1);
                    c_xor(cg, acc, cg->r2);
                }
            }
        }
    }
}

/* ---------------- recovery closed-form body (algos #4, rec_cf twin) -- */

static void gen_stores(cg_t *cg);

/* Accumulate one syndrome column: acc ^= c*in[s] with acc = acc0+u per
 * lane. first = 1 when this is the accumulator's first contributing
 * column (acc dead: copy/move instead of xor-with-zero). Mirrors
 * mulacc_column's per-(j,col) logic for a single accumulator row, with
 * the coefficient array swapped from pl->mat to the cf_data row. */
static void cf_syndrome_column(cg_t *cg, int s, uint8_t c, int acc0,
                               int first)
{
    int U = cg->U;
    /* Predicated-tail mode processes one lane per chunk (the u=0 slice). */
    int ulanes = cg->pred ? 1 : U;
    if (!c)
        return;
    if (c != 1) {
        c_ldpool(cg, cg->tbl_lo, CK_NIBLO, c);
        c_ldpool(cg, cg->tbl_hi, CK_NIBHI, c);
    }
    c_inptr(cg, X10, s);
    for (int u = 0; u < ulanes; u++) {
        int acc = acc0 + u;
        c_ld_to(cg, cg->t, u, X10);
        if (c == 1) {
            if (first)
                c_copy(cg, acc, cg->t);
            else
                c_xor(cg, acc, cg->t);
        } else {
            c_and3(cg, cg->tlo, cg->t);
            c_lsr4(cg, cg->thi, cg->t);
            c_tbl_into(cg, cg->r1, cg->tbl_lo, cg->tlo);
            c_tbl_into(cg, cg->r2, cg->tbl_hi, cg->thi);
            if (first) {
                /* acc dead: unpredicated EOR on SVE, mov+eor on NEON --
                 * exactly mulacc_column's first-column path (the
                 * destination is dead, not zero, so EOR3 is unusable;
                 * in pred mode the unpredicated form is exact on active
                 * lanes, same argument as gen_pred_tail documents). */
                if (cg->is_sve) {
                    em_sve_eor_unpred(&cg->em, V(cg, acc), V(cg, cg->r1),
                                      V(cg, cg->r2));
                    cg->uops += 1;
                } else {
                    c_copy(cg, acc, cg->r1);
                    c_xor(cg, acc, cg->r2);
                }
            } else if (cg->use_eor3) {
                c_eor3(cg, acc, cg->r1, cg->r2);
            } else {
                c_xor(cg, acc, cg->r1);
                c_xor(cg, acc, cg->r2);
            }
        }
    }
}

/* dst = c*src over GF(2^8); dst's old value is dead (consumed by the
 * multiply). c==0: zero dst. c==1: copy. Otherwise the nibble-split
 * TBL idiom with the dead-dst reduction. src is preserved. */
static void cf_mul_dead(cg_t *cg, int dst, int src, uint8_t c)
{
    if (c == 0) {
        if (cg->is_sve)
            em_sve_dup(&cg->em, V(cg, dst), 0);
        else
            em_neon_movi(&cg->em, dst, 0);
        cg->uops += 1;
        return;
    }
    if (c == 1) {
        c_copy(cg, dst, src);
        return;
    }
    c_ldpool(cg, cg->tbl_lo, CK_NIBLO, c);
    c_ldpool(cg, cg->tbl_hi, CK_NIBHI, c);
    c_and3(cg, cg->tlo, src);
    c_lsr4(cg, cg->thi, src);
    c_tbl_into(cg, cg->r1, cg->tbl_lo, cg->tlo);
    c_tbl_into(cg, cg->r2, cg->tbl_hi, cg->thi);
    if (cg->is_sve) {
        em_sve_eor_unpred(&cg->em, V(cg, dst), V(cg, cg->r1), V(cg, cg->r2));
        cg->uops += 1;
    } else {
        c_copy(cg, dst, cg->r1);
        c_xor(cg, dst, cg->r2);
    }
}

/* dst ^= c*src over GF(2^8); dst is live. c==0: no-op. c==1: plain
 * xor. Otherwise the nibble idiom; on SVE2 (use_eor3) the reduction
 * folds into one EOR3. src is preserved. */
static void cf_mulxor_live(cg_t *cg, int dst, int src, uint8_t c)
{
    if (c == 0)
        return;
    if (c == 1) {
        c_xor(cg, dst, src);
        return;
    }
    c_ldpool(cg, cg->tbl_lo, CK_NIBLO, c);
    c_ldpool(cg, cg->tbl_hi, CK_NIBHI, c);
    c_and3(cg, cg->tlo, src);
    c_lsr4(cg, cg->thi, src);
    c_tbl_into(cg, cg->r1, cg->tbl_lo, cg->tlo);
    c_tbl_into(cg, cg->r2, cg->tbl_hi, cg->thi);
    if (cg->use_eor3)
        c_eor3(cg, dst, cg->r1, cg->r2);
    else {
        c_xor(cg, dst, cg->r1);
        c_xor(cg, dst, cg->r2);
    }
}

/* Zero one accumulator row (acc0+u per lane). Used for an all-zero
 * syndrome (no contributing column): the accumulator would otherwise
 * stay dead. Unreachable via the planner for a P_MULACC plan's primary
 * syndrome, but a force-built candidate must not read dead accs. */
static void cf_zero_acc(cg_t *cg, int acc0)
{
    int ulanes = cg->pred ? 1 : cg->U;
    for (int u = 0; u < ulanes; u++) {
        int acc = acc0 + u;
        if (cg->is_sve)
            em_sve_dup(&cg->em, V(cg, acc), 0);
        else
            em_neon_movi(&cg->em, acc, 0);
        cg->uops += 1;
    }
}

/* e=2 2x2 combine for one lane (unfolded form: 6 GF multiplies):
 *   x1 = (d*p ^ b*q)*detinv into P's reg,
 *   x2 = (c*p ^ a*q)*detinv into Q's reg.
 * P = acc0+u holds p, Q = acc1+u holds q (both live on entry, dead on
 * exit). Temps: t holds x1_pre, r1 holds x2_pre; tlo/thi/r1/r2 are the
 * multiply working set. p is dead after its c-term is extracted, q
 * after its a-term; the P/Q homes are dead then (their old values
 * consumed), so the detinv multiplies use the dead-dst form -- with the
 * Q-home multiply first, since it consumes r1 while the P-home
 * multiply would clobber it. The a-term accumulates through r2 only
 * (r1 holds c*p live, t holds x1_pre live): the TBL writes r2 fully, so
 * the sequential xor-into-r1 is exact. */
static void cf_combine_lane(cg_t *cg, int u, int acc0, int acc1,
                            const struct rec_cf_data *cfd)
{
    int P = acc0 + u, Q = acc1 + u;
    int t = cg->t;
    /* x1_pre = d*p ^ b*q into t */
    cf_mul_dead(cg, t, P, cfd->d);
    cf_mulxor_live(cg, t, Q, cfd->b);
    /* x2_pre = c*p ^ a*q into r1 */
    cf_mul_dead(cg, cg->r1, P, cfd->c);
    if (cfd->a == 1) {
        c_xor(cg, cg->r1, Q);
    } else if (cfd->a != 0) {
        c_ldpool(cg, cg->tbl_lo, CK_NIBLO, cfd->a);
        c_ldpool(cg, cg->tbl_hi, CK_NIBHI, cfd->a);
        c_and3(cg, cg->tlo, Q);
        c_lsr4(cg, cg->thi, Q);
        c_tbl_into(cg, cg->r2, cg->tbl_lo, cg->tlo);
        c_xor(cg, cg->r1, cg->r2);
        c_tbl_into(cg, cg->r2, cg->tbl_hi, cg->thi);
        c_xor(cg, cg->r1, cg->r2);
    }
    /* P = detinv*x1_pre, Q = detinv*x2_pre (P/Q dead: p/q consumed).
     * Q first: it consumes r1, which the P multiply would clobber. */
    cf_mul_dead(cg, Q, cg->r1, cfd->detinv);
    cf_mul_dead(cg, P, t, cfd->detinv);
}

/* Recovery closed-form body (algos #4, rec_cf twin).
 *
 * e=1: x_f = cinv * syndrome, syndrome = XOR_s (ecoef[s] * in[s]).
 * e=2: p = XOR_s (ecoef1[s]*in[s]), q = XOR_s (ecoef2[s]*in[s]), then
 * the per-lane 2x2 combine above.
 *
 * Register usage is exactly compute_regs' P_MULACC nibble assignment
 * (accs, t, tlo, thi, r1, r2, tbl_lo, tbl_hi, mask): the twin is only
 * enumerated on P_MULACC plans, which always have a nontrivial baked
 * coefficient, so the nibble set is always allocated (asserted). The
 * scalar tail stays on the generic baked-R path (gen_tail): GF(2^8)
 * distributivity makes it bit-identical to the closed form. */
static void gen_body_rec_cf(cg_t *cg)
{
    const struct rec_cf_data *cfd = cg->pl->rec_cf;
    int U = cg->U;
    assert(cfd && (cfd->nf == 1 || cfd->nf == 2));
    assert(cg->need_mask); /* nibble reg set allocated (see above) */
    if (cfd->nf == 1) {
        int first = -1;
        for (int s = 0; s < cfd->ns; s++)
            if (cfd->ecoef[s]) {
                first = s;
                break;
            }
        if (first < 0)
            cf_zero_acc(cg, 0);
        else
            for (int s = 0; s < cfd->ns; s++)
                cf_syndrome_column(cg, s, cfd->ecoef[s], 0, s == first);
        /* x_f = cinv * syndrome */
        if (cfd->cinv != 1) {
            int ulanes = cg->pred ? 1 : U;
            for (int u = 0; u < ulanes; u++)
                cf_mul_dead(cg, u, u, cfd->cinv);
        }
    } else {
        /* p into acc row 0, q into acc row 1 */
        int first1 = -1, first2 = -1;
        for (int s = 0; s < cfd->ns; s++)
            if (cfd->ecoef1[s]) {
                first1 = s;
                break;
            }
        for (int s = 0; s < cfd->ns; s++)
            if (cfd->ecoef2[s]) {
                first2 = s;
                break;
            }
        if (first1 < 0)
            cf_zero_acc(cg, 0);
        else
            for (int s = 0; s < cfd->ns; s++)
                cf_syndrome_column(cg, s, cfd->ecoef1[s], 0, s == first1);
        if (first2 < 0)
            cf_zero_acc(cg, U);
        else
            for (int s = 0; s < cfd->ns; s++)
                cf_syndrome_column(cg, s, cfd->ecoef2[s], U, s == first2);
        {
            int ulanes = cg->pred ? 1 : U;
            for (int u = 0; u < ulanes; u++)
                cf_combine_lane(cg, u, 0, U, cfd);
        }
    }
    gen_stores(cg);
}

static void gen_body(cg_t *cg)
{
    /* Recovery closed-form twin (algos #4): syndrome+combine body
     * instead of the generic MULACC over the baked R matrix. */
    if (cg->cand.rec_cf) {
        gen_body_rec_cf(cg);
        return;
    }
    em_t *e = &cg->em;
    const plan_t *pl = cg->pl;
    int n_in = pl->n_in, n_out = pl->n_out, U = cg->U;
    /* Predicated-tail mode processes one lane per chunk (the u=0 slice);
     * the chunk loop re-enters gen_body per VL-sized chunk. */
    int ulanes = cg->pred ? 1 : U;

    /* Accumulator seeding. Every live accumulator is overwritten before
     * its first read (see x86 backend for the argument); dead rows are
     * zeroed here. */
    if (cg->cand.xorbit) {
        /* Bit-program twin: the body follows pl->bit_sched instead of
         * the naive emitter. gen_body_xorbit handles its own
         * accumulator seeding/zeroing; stores below are shared. */
        gen_body_xorbit(cg);
    } else if (pl->kind == P_HORNER && n_in == 0) {
        for (int j = 0; j < n_out; j++)
            for (int u = 0; u < ulanes; u++) {
                int r = j * U + u;
                if (cg->is_sve)
                    em_sve_dup(e, V(cg, r), 0);
                else
                    em_neon_movi(e, r, 0);
                cg->uops += 1;
            }
        /* fall through: store the zeroed accumulators below */
    } else if (pl->kind == P_XOR) {
        if (cg->cand.xorsched) {
        /* XOR-schedule twin: the body follows pl->xor_sched (cross-row
         * CSE). Phase 1 forms each temp t_k = in_a ^ in_b into t/t2 and
         * fans out; phase 2 streams surviving inputs via t. Every
         * accumulator is overwritten (copy) on first touch, never
         * XORed with zero; each shared subexpression is computed once.
         * X3 is the second input pointer (free in P_XOR bodies). */
        {
            const xorsched_t *xs = pl->xor_sched;
            assert(xs && xs->ntemps > 0);
            int pf_done[RJ_MAX_DISKS];
            for (int i = 0; i < n_in; i++)
                pf_done[i] = 0;
            /* phase 1: temp formation + fan-out */
            for (int k = 0; k < xs->ntemps; k++) {
                int a = xs->temps[k].a, b = xs->temps[k].b;
                if (!pf_done[a]) {
                    c_inptr(cg, X10, a);
                    pf_done[a] = 1;
                } else {
                    c_inptr_np(cg, X10, a);
                }
                if (!pf_done[b]) {
                    c_inptr(cg, X3, b);
                    pf_done[b] = 1;
                } else {
                    c_inptr_np(cg, X3, b);
                }
                for (int u = 0; u < ulanes; u++) {
                    c_ld_to(cg, cg->t, u, X10);
                    c_ld_to(cg, cg->t2, u, X3);
                    c_xor(cg, cg->t, cg->t2);
                    for (int wi = 0; wi < xs->temps[k].nusers; wi++) {
                        int uw = xs->temps[k].users[wi];
                        int acc = xs->first_row[uw] * U + u;
                        if (xs->row_first[uw] == k)
                            c_copy(cg, acc, cg->t);
                        else
                            c_xor(cg, acc, cg->t);
                    }
                }
            }
            /* phase 2: surviving input atoms */
            for (int i = 0; i < n_in; i++) {
                int nusers = xs->in_nusers[i];
                if (!nusers)
                    continue;
                if (!pf_done[i]) {
                    c_inptr(cg, X10, i);
                    pf_done[i] = 1;
                } else {
                    c_inptr_np(cg, X10, i);
                }
                for (int u = 0; u < ulanes; u++) {
                    c_ld_to(cg, cg->t, u, X10);
                    for (int wi = 0; wi < nusers; wi++) {
                        int uw = xs->in_users[i][wi];
                        int acc = xs->first_row[uw] * U + u;
                        if (xs->row_first[uw] == xs->ntemps + i)
                            c_copy(cg, acc, cg->t);
                        else
                            c_xor(cg, acc, cg->t);
                    }
                }
            }
            /* dead rows: zero */
            for (int uw = 0; uw < xs->nunique; uw++) {
                if (xs->row_first[uw] >= 0)
                    continue;
                for (int j = 0; j < n_out; j++) {
                    if (xs->urow_of[j] != uw)
                        continue;
                    for (int u = 0; u < ulanes; u++) {
                        int acc = j * U + u;
                        if (cg->is_sve)
                            em_sve_dup(e, V(cg, acc), 0);
                        else
                            em_neon_movi(e, acc, 0);
                        cg->uops += 1;
                    }
                }
            }
            /* duplicate rows: vector moves from the canonical row */
            for (int j = 0; j < n_out; j++) {
                int uw = xs->urow_of[j];
                if (j == xs->first_row[uw] || xs->row_first[uw] < 0)
                    continue;
                for (int u = 0; u < ulanes; u++)
                    c_copy(cg, j * U + u, xs->first_row[uw] * U + u);
            }
        }
        } else {
        /* first_i[j] = first input with a nonzero coefficient in row j
         * (-1 if the row is all zero). The accumulator for (j,u) is
         * initialized by loading the first nonzero input and XORs the
         * rest; this must persist across the i loop. */
        int first_i[n_out];
        for (int j = 0; j < n_out; j++) {
            first_i[j] = -1;
            for (int i = 0; i < n_in; i++)
                if (pl->mat[(size_t)j * (size_t)n_in + (size_t)i]) {
                    first_i[j] = i;
                    break;
                }
        }
        for (int i = 0; i < n_in; i++) {
            int nu = 0;
            int sole_j = -1;
            for (int j = 0; j < n_out; j++)
                if (pl->mat[(size_t)j * (size_t)n_in + (size_t)i]) {
                    nu++;
                    sole_j = j;
                }
            if (!nu)
                continue;
            c_inptr(cg, X10, i);   /* X10 = in[i] + x11 */
            /* First-source direct load (item 8): when the input feeds a
             * single row (nu==1) and is that row's first contributing
             * column, load directly into the accumulator (1 uop) instead
             * of load-to-t + copy (2 uops). Mirrors the PDELTA seeding
             * pattern above (and x86-64's first-source load). The
             * xorsched model (xs_body_naive) counts this case as 1 uop. */
            int direct = (nu == 1 && i == first_i[sole_j]);
            for (int u = 0; u < ulanes; u++) {
                if (direct) {
                    c_ld_to(cg, sole_j * U + u, u, X10);
                    continue;
                }
                /* Load once per (i,u) into t; each row then either
                 * initializes its accumulator from t (first contributing
                 * column) or XORs t in. t = n_out*U never aliases an acc
                 * and is dead across the j-loop except as the shared
                 * source. */
                c_ld_to(cg, cg->t, u, X10);
                for (int j = 0; j < n_out; j++) {
                    if (!pl->mat[(size_t)j * (size_t)n_in + (size_t)i])
                        continue;
                    int acc = j * U + u;
                    if (i == first_i[j])
                        c_copy(cg, acc, cg->t);
                    else
                        c_xor(cg, acc, cg->t);
                }
            }
        }
        /* zero accumulators for all-zero rows */
        for (int j = 0; j < n_out; j++) {
            if (first_i[j] != -1)
                continue;
            for (int u = 0; u < ulanes; u++) {
                int acc = j * U + u;
                if (cg->is_sve)
                    em_sve_dup(e, V(cg, acc), 0);
                else
                    em_neon_movi(e, acc, 0);
                cg->uops += 1;
            }
        }
        } /* end xorsched/naive P_XOR */
    } else if (pl->kind == P_DELTA) {
        int nc = pl->nchanged;
        /* Seed accumulators from old parity: in[2*nc + j], NOT out[j].
         * (out[j] may alias in[2*nc+j] for in-place update, but the
         * old parity value must be loaded before any stores.) */
        for (int j = 0; j < n_out; j++) {
            c_inptr(cg, X10, 2 * nc + j);
            for (int u = 0; u < ulanes; u++)
                c_ld_to(cg, j * U + u, u, X10);
        }
        for (int i = 0; i < nc; i++) {
            int any, neednib;
            mulacc_col_info(pl, i, nc, &any, &neednib);
            if (!any)
                continue;
            /* Input pointers set once per column; the MUL VL lane
             * addressing keeps them fixed across the u-loop inside
             * mulacc_column. */
            c_inptr(cg, X10, i);
            /* NOTE: the 2nd delta pointer MUST NOT use x14: on the
             * SVE path x14 is the loop iteration counter
             * (udiv x14, x2, x13 in the prologue; subs x14 at loop
             * end). x15 is also unusable here: c_ldpool clobbers it
             * on the SVE path (called by mulacc_column below for
             * nontrivial coefficients). x3 is otherwise unused in
             * generated code (no calls emitted), so it is safe. */
            em_ldr64(e, X3, X8, (unsigned)(nc + i) * 8);
            em_add_rr(e, X3, X3, X11);
            mulacc_column(cg, i, nc, NULL, 1, -1);
        }
    } else { /* P_MULACC, P_HORNER */
        int *first_col = NULL;
        if (pl->kind == P_MULACC) {
            first_col = malloc((size_t)n_out * sizeof(*first_col));
            assert(first_col);
            for (int j = 0; j < n_out; j++) {
                first_col[j] = -1;
                for (int i = 0; i < n_in; i++)
                    if (pl->mat[(size_t)j * (size_t)n_in + (size_t)i]) {
                        first_col[j] = i;
                        break;
                    }
            }
        }
        for (int i = 0; i < n_in; i++) {
            int any, neednib;
            mulacc_col_info(pl, i, n_in, &any, &neednib);
            if (!any)
                continue;
            if (pl->kind == P_MULACC) {
                /* Hoisted-table nest (column -> row -> lane): the input
                 * pointer is set once per column and stays fixed across
                 * lanes via MUL VL addressing; mulacc_column loads each
                 * row's tables once per (j,i) and runs the lane loop. */
                c_inptr(cg, X10, i);
                mulacc_column(cg, i, n_in, first_col, 0, -1);
                continue;
            }
            /* P_HORNER keeps its (column -> lane -> row) nest: tables
             * live in hreg (or are rematerialized per use). */
            for (int u = 0; u < ulanes; u++) {
                if (u == 0)
                    c_inptr(cg, X10, i);
                if (i > 0) {
                    c_ld_to(cg, cg->r1, u, X10);
                    /* acc = acc*c_j ^ r1, per row j */
                    for (int j = 0; j < n_out; j++) {
                        uint8_t c = pl->horner_c[j];
                        int acc = j * U + u;
                        if (c == 1) {
                            c_xor(cg, acc, cg->r1);
                        } else if (c == 0) {
                            c_copy(cg, acc, cg->r1);
                        } else {
                            c_and3(cg, cg->tlo, acc);
                            c_lsr4(cg, cg->thi, acc);
                            /* SVE2: fold lo^hi^input with one EOR3. The lo
                             * TBL targets acc directly -- acc is dead after
                             * the nibble split -- and r2 takes hi, so the
                             * MOV and one EOR disappear (drops 2 uops:
                             * 9 -> 7 on the remat path, 7 -> 5 vector insns
                             * with hoisted tables; same live regs). */
                            /* The lo TBL targets acc directly on both paths:
                             * acc is dead after the nibble split above
                             * (its old value was consumed into tlo/thi),
                             * r1 holds the input (read by the final xor),
                             * and r2 is the hi temp -- so acc is the only
                             * valid lo target. This drops the mov on the
                             * non-EOR3 path (EOR3 already did this). */
                            int lo_dst = acc;
                            if (cg->cand.remat) {
                                /* NIBLO/NIBHI ping-pong through the single
                                 * tbl_lo scratch: every (i,j,u) needs both
                                 * tables, and the scratch is shared across
                                 * rows, so the 2 loads per use are inherent
                                 * -- the table cannot be hoisted out of the
                                 * u-loop (same tradeoff as the x86 backend:
                                 * a second scratch would change the proven
                                 * n_out*U+6 pressure formula). */
                                c_ldpool(cg, cg->tbl_lo, CK_NIBLO, c);
                                c_tbl_into(cg, lo_dst, cg->tbl_lo, cg->tlo);
                                c_ldpool(cg, cg->tbl_lo, CK_NIBHI, c);
                                c_tbl_into(cg, cg->r2, cg->tbl_lo, cg->thi);
                            } else {
                                c_tbl_into(cg, lo_dst, cg->hreg[2 * j],
                                           cg->tlo);
                                c_tbl_into(cg, cg->r2, cg->hreg[2 * j + 1],
                                           cg->thi);
                            }
                            if (cg->use_eor3)
                                c_eor3(cg, acc, cg->r2, cg->r1);
                            else {
                                c_xor(cg, acc, cg->r2);
                                c_xor(cg, acc, cg->r1);
                            }
                        }
                    }
                } else {
                    /* Horner i==0: acc = d0 (no multiply) */
                    c_ld_to(cg, cg->t, u, X10);
                    for (int j = 0; j < n_out; j++) {
                        int acc = j * U + u;
                        c_copy(cg, acc, cg->t);
                    }
                }
            }
        }
        free(first_col);
    }

    gen_stores(cg);
}

/* Output stores for one group iteration (shared by gen_body and the
 * pipe twin's gen_body_pipe). Pure extraction: no behavior change. */
static void gen_stores(cg_t *cg)
{
    em_t *e = &cg->em;
    const plan_t *pl = cg->pl;
    int n_out = pl->n_out, U = cg->U;
    /* Predicated-tail mode processes one lane per chunk (the u=0 slice). */
    int ulanes = cg->pred ? 1 : U;

    /* stores: NEON nt pairs adjacent lanes with STNP; SVE uses STNT1B */
    for (int j = 0; j < n_out; j++) {
        c_outptr(cg, j);
        if (cg->nt && !cg->is_sve && U >= 2) {
            int u;
            for (u = 0; u + 1 < U; u += 2) {
                em_neon_stnp(e, j * U + u, j * U + u + 1, X10,
                             (unsigned)u * 16);
                cg->uops += 1;
                cg->mem += 2 * cg->vl;
            }
            if (u < U)
                c_st_to(cg, j * U + u, u, X10);
        } else {
            for (int u = 0; u < ulanes; u++)
                c_st_to(cg, j * U + u, u, X10);
        }
    }
    /* NOTE: no DMB is executed for NT stores. The caller must execute
     * DMB ISHST itself if the NT data must be visible to other agents
     * before subsequent reads. */
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
 * Dn. Runs after the empty-loop branch (N >= 1 guaranteed) with
 * x11 == 0, before the loop label. Uncounted: prologue setup is not
 * per-iteration work (same convention as gen_const_setup). */
static void gen_pipe_prologue(cg_t *cg, int col0)
{
    double save_uops = cg->uops, save_mem = cg->mem;
    c_inptr(cg, X10, col0);
    for (int u = 0; u < cg->U; u++)
        c_ld_to(cg, cg->pipe_dn + u, u, X10);
    cg->uops = save_uops;
    cg->mem = save_mem;
}

/* One HORNER column's compute from the staged dataset (pipe twin).
 * Mirrors gen_body's P_HORNER i-loop with the input lanes in
 * [data, data+U) instead of freshly loaded r1/t. remat is rejected for
 * the pipe twin (asserted in jit_build_version), so only the hoisted
 * hreg table form exists here. */
static void pipe_horner_column(cg_t *cg, int col, int data)
{
    const plan_t *pl = cg->pl;
    int n_out = pl->n_out, U = cg->U;
    assert(!cg->cand.remat);
    if (col == 0) {
        /* acc = d0: plain moves from the staged dataset */
        for (int u = 0; u < U; u++)
            for (int j = 0; j < n_out; j++)
                c_copy(cg, j * U + u, data + u);
        return;
    }
    for (int u = 0; u < U; u++) {
        for (int j = 0; j < n_out; j++) {
            uint8_t c = pl->horner_c[j];
            int acc = j * U + u;
            if (c == 1) {
                c_xor(cg, acc, data + u);
            } else if (c == 0) {
                c_copy(cg, acc, data + u);
            } else {
                c_and3(cg, cg->tlo, acc);
                c_lsr4(cg, cg->thi, acc);
                /* The lo TBL targets acc directly (mirrors gen_body:
                 * acc is dead after the nibble split, r1's input role
                 * is taken by the staged dataset). */
                c_tbl_into(cg, acc, cg->hreg[2 * j], cg->tlo);
                c_tbl_into(cg, cg->r2, cg->hreg[2 * j + 1], cg->thi);
                if (cg->use_eor3)
                    c_eor3(cg, acc, cg->r2, data + u);
                else {
                    c_xor(cg, acc, cg->r2);
                    c_xor(cg, acc, data + u);
                }
            }
        }
    }
}

/* Software-pipelined (double-buffered) loop body (track I8, pipe twin).
 *
 * Per-column double buffering: the input lanes for column cols[cidx]
 * of the current iteration are staged in a U-register dataset; the
 * compute for (cidx, iter) reads the dataset while the loads for the
 * NEXT (column, iteration) are issued into the other dataset:
 *   - compute(cols[cidx], iter) reads (cidx == 0 ? Dn : D). Dn holds
 *     (cols[0], iter): the prologue for iter == 0, the boundary load of
 *     the previous iteration otherwise.
 *   - then, if cidx+1 < ncols: load(cols[cidx+1], iter) -> D at the
 *     current offset x11;
 *     else (last column): if iter+1 < N: load(cols[0], iter+1) -> Dn at
 *     x11+group, guarded by (niter_reg > 1) so the load never
 *     over-reads into/past the tail region.
 * The compute consumes each dataset fully before it is overwritten, so
 * D/Dn are the only extra live set. Every issued load is consumed by
 * the next column step's compute, so the pipeline drains naturally;
 * the "epilogue" is the final iteration's stores. ncols == 0
 * (degenerate: no input columns) falls back to gen_body. */
static void gen_body_pipe(cg_t *cg)
{
    em_t *e = &cg->em;
    const plan_t *pl = cg->pl;
    int n_in = pl->n_in, n_out = pl->n_out, U = cg->U;
    assert(cg->cand.pipe);
    assert(pl->kind == P_MULACC || pl->kind == P_HORNER);
    assert(!cg->cand.remat);

    int cols[RJ_MAX_DISKS];
    int ncols = pipe_used_columns(cg, cols);
    if (ncols == 0) {
        gen_body(cg); /* degenerate: no loads to pipeline */
        return;
    }

    int *first_col = NULL;
    if (pl->kind == P_MULACC) {
        first_col = malloc((size_t)n_out * sizeof(*first_col));
        assert(first_col);
        for (int j = 0; j < n_out; j++) {
            first_col[j] = -1;
            for (int i = 0; i < n_in; i++)
                if (pl->mat[(size_t)j * (size_t)n_in + (size_t)i]) {
                    first_col[j] = i;
                    break;
                }
        }
    }

    for (int cidx = 0; cidx < ncols; cidx++) {
        int col = cols[cidx];
        int data = (cidx == 0) ? cg->pipe_dn : cg->pipe_d;
        /* compute column col from the staged dataset */
        if (pl->kind == P_MULACC)
            mulacc_column(cg, col, n_in, first_col, 0, data);
        else
            pipe_horner_column(cg, col, data);
        /* issue the next loads into the other dataset */
        if (cidx + 1 < ncols) {
            c_inptr(cg, X10, cols[cidx + 1]);
            for (int u = 0; u < U; u++)
                c_ld_to(cg, cg->pipe_d + u, u, X10);
        } else {
            /* boundary: (cols[0], iter+1) -> Dn at x11+group, iff a
             * full next iteration exists. x14 (SVE) / x13 (NEON) is the
             * countdown niter = N - iter, so iter+1 < N iff reg > 1.
             * The guard keeps the load in-bounds; the last iteration
             * would otherwise over-read up to group bytes into/past
             * the tail. */
            int skip = em_label(e);
            em_cmp_ri(e, cg->is_sve ? X14 : X13, 1);
            em_bcc(e, CC_LE, skip);
            c_inptr(cg, X10, cols[0]);
            /* group < 4096 (NEON: U*16 <= 128; SVE: U*vl <= 8*256):
             * fits the 12-bit add immediate. */
            assert((unsigned)cg->group < 4096);
            em_add_ri(e, X10, X10, (unsigned)cg->group);
            cg->uops += 1;
            for (int u = 0; u < U; u++)
                c_ld_to(cg, cg->pipe_dn + u, u, X10);
            em_bind(e, skip);
        }
    }
    free(first_col);

    gen_stores(cg);
}

/* ---------------- scalar tail ---------------- */

/* acc(w12) ^= d(w14) * c over GF(2^8), log/exp tables at x15/x17.
 * x13 is the 64-bit byte counter; w12 is the 32-bit accumulator. */
static void tail_mul_byte(cg_t *cg, uint8_t c)
{
    em_t *e = &cg->em;
    if (c == 1) {
        em_eor_w(e, 12, 12, 14);
        return;
    }
    int lsk = em_label(e), lns = em_label(e);
    em_cbz_w(e, 14, lsk);
    em_ldrb(e, 14, 15, 14);      /* w14 = log[d]; x14 (pointer) dead here */
    em_add_wr_i(e, 14, 14, cg->logtab[c]);
    em_cmp_wr_i(e, 14, 255);
    em_bcc(e, CC_LO, lns);
    em_sub_wr_i(e, 14, 14, 255);
    em_bind(e, lns);
    em_ldrb(e, 14, 17, 14);       /* w14 = exp[...] */
    em_eor_w(e, 12, 12, 14);
    em_bind(e, lsk);
}

/* Scalar tail.
 *
 * Register convention:
 *   x13 = byte counter (64-bit, never truncated),
 *   w12 = accumulator (32-bit),
 *   x14 = input pointer; after the byte load w14 holds the data byte,
 *   w19 = second data byte (P_HORNER only; x19 saved/restored),
 *   x15/x17 = log/exp table bases, x11 = tail start (preserved).
 *
 * P_DELTA uses GF(2^8) linearity: (old^new)*c = old*c ^ new*c, so the two
 * products are accumulated separately and no extra register is needed. */
static void gen_tail(cg_t *cg)
{
    em_t *e = &cg->em;
    const plan_t *pl = cg->pl;
    int n_in = pl->n_in, n_out = pl->n_out;

    if (cg->need_logexp) {
        c_pooladdr(cg, X15, pool_get(cg, CK_LOG, 0));
        c_pooladdr(cg, X17, pool_get(cg, CK_EXP, 0));
    }
    for (int j = 0; j < n_out; j++) {
        int lj = em_label(e);
        em_ldr64(e, X10, X9, (unsigned)j * 8);
        em_mov_rr(e, X13, X11);   /* x13 = tail start */
        if (pl->kind == P_DELTA) {
            em_bind(e, lj);
            /* seed from the old parity byte */
            em_ldr64(e, X14, X8,
                     (unsigned)(2 * pl->nchanged + j) * 8);
            em_ldrb(e, 12, 14, 13);
        } else if (pl->kind == P_XOR) {
            int has = 0;
            for (int i = 0; i < n_in; i++)
                if (pl->mat[(size_t)j * (size_t)n_in + (size_t)i]) {
                    has = 1;
                    break;
                }
            if (!has)
                em_mov_wi(e, 12, 0);
            em_bind(e, lj);
        } else if (pl->kind == P_HORNER && n_in > 0) {
            em_bind(e, lj);       /* i == 0 overwrites w12 below */
        } else {
            /* P_MULACC (and P_HORNER with n_in == 0): the accumulator
             * must be re-zeroed for EVERY tail byte, so the zeroing is
             * inside the loop (bind first). The old order (mov before
             * bind) zeroed w12 only for the first tail byte; bytes 1..
             * accumulated onto the previous byte's result. */
            em_bind(e, lj);
            em_mov_wi(e, 12, 0);
        }
        if (pl->kind == P_XOR) {
            int first = 1;
            for (int i = 0; i < n_in; i++) {
                if (!pl->mat[(size_t)j * (size_t)n_in + (size_t)i])
                    continue;
                em_ldr64(e, X14, X8, (unsigned)i * 8);
                em_ldrb(e, 14, 14, 13);
                if (first) {
                    em_mov_wr(e, 12, 14);
                    first = 0;
                } else {
                    em_eor_w(e, 12, 12, 14);
                }
            }
        } else if (pl->kind == P_MULACC) {
            for (int i = 0; i < n_in; i++) {
                uint8_t c = pl->mat[(size_t)j * (size_t)n_in + (size_t)i];
                if (!c)
                    continue;
                em_ldr64(e, X14, X8, (unsigned)i * 8);
                em_ldrb(e, 14, 14, 13);
                tail_mul_byte(cg, c);
            }
        } else if (pl->kind == P_DELTA) {
            int nc = pl->nchanged;
            for (int i = 0; i < nc; i++) {
                uint8_t c = pl->mat[(size_t)j * (size_t)nc + (size_t)i];
                if (!c)
                    continue;
                em_ldr64(e, X14, X8, (unsigned)i * 8);
                em_ldrb(e, 14, 14, 13);
                tail_mul_byte(cg, c);            /* acc ^= old*c */
                em_ldr64(e, X14, X8, (unsigned)(nc + i) * 8);
                em_ldrb(e, 14, 14, 13);
                tail_mul_byte(cg, c);            /* acc ^= new*c */
            }
        } else { /* P_HORNER */
            uint8_t c = pl->horner_c[j];
            for (int i = 0; i < n_in; i++) {
                em_ldr64(e, X14, X8, (unsigned)i * 8);
                em_ldrb(e, 19, 14, 13);          /* w19 = d */
                if (i == 0) {
                    em_mov_wr(e, 12, 19);
                    continue;
                }
                if (c == 1) {
                    em_eor_w(e, 12, 12, 19);
                    continue;
                }
                if (c == 0) {
                    em_mov_wr(e, 12, 19);
                    continue;
                }
                int lz = em_label(e), lnx = em_label(e), lns = em_label(e);
                em_cbz_w(e, 12, lz);
                em_ldrb(e, 14, 15, 12);
                em_add_wr_i(e, 14, 14, cg->logtab[c]);
                em_cmp_wr_i(e, 14, 255);
                em_bcc(e, CC_LO, lns);
                em_sub_wr_i(e, 14, 14, 255);
                em_bind(e, lns);
                em_ldrb(e, 14, 17, 14);
                em_eor_w(e, 14, 14, 19);
                em_mov_wr(e, 12, 14);
                em_b(e, lnx);
                em_bind(e, lz);
                em_mov_wr(e, 12, 19);
                em_bind(e, lnx);
            }
        }
        em_strb(e, 12, 10, 13);   /* [x10 + x13] = w12 */
        /* No w12 zeroing here: the loop head re-zeroes (or overwrites)
         * w12 for the next byte. P_DELTA's accumulator persists. */
        em_add_ri(e, X13, X13, 1);
        em_cmp_rr(e, X13, X2);
        em_bcc(e, CC_LO, lj);
    }
}

/* Fully predicated vector tail (SVE only; predtail twin, track B1sve).
 *
 * Replaces the scalar byte tail: the remaining rem = len - x11 bytes
 * (1 <= rem < group) are processed in VL-sized chunks, each chunk running
 * one single-lane predicated pass of gen_body under p1 =
 * whilelt(xzr, rem) (element i active iff i < rem; a full chunk when
 * rem >= VL).
 *
 * Register convention: x11 = chunk base (advanced by VL per chunk; dead
 * after the tail), x13 = remaining count, x12 = VL bytes (preserved),
 * x16 = pool base (preserved). p1 is the chunk predicate; p0 (the main
 * loop's all-true predicate) is untouched.
 *
 * Exactness: per-lane data flow is independent, so every active lane
 * computes exactly the scalar tail's byte. Inactive lanes keep stale
 * register values but are never stored (stores are p1-predicated) and
 * never read by a predicated op. LD1B/ST1B under p1 perform no memory
 * access for inactive elements, so no over-read/over-write past the
 * buffer end is possible (verified by guard-page tests). The one
 * exception to "predicated everything" is mulacc_column's first-column
 * path, which keeps its unpredicated EOR: the destination is dead there,
 * and a merging predicated EOR would fold the dead value into active
 * lanes; unpredicated EOR computes the exact r1^r2 on active lanes. */
static void gen_pred_tail(cg_t *cg)
{
    em_t *e = &cg->em;
    assert(cg->is_sve && !cg->notail);
    int chunk = em_label(e);
    /* x13 = rem = len - x11 (1 .. group-1; the tail is entered only when
     * x11 < x2, so rem >= 1 and the first whilelt is never all-false). */
    em_sub_rr(e, X13, X2, X11);
    em_bind(e, chunk);
    em_sve_whilelt(e, 1, XZR, X13);
    /* The tail is steady-state-excluded from the uops/mem model, exactly
     * like the scalar tail (which emits raw em_ calls with zero
     * accounting): twins compare on the identical vector body. */
    double save_uops = cg->uops, save_mem = cg->mem;
    cg->pred = 1;
    gen_body(cg);
    cg->pred = 0;
    cg->uops = save_uops;
    cg->mem = save_mem;
    em_add_rr(e, X11, X11, X12);
    em_subs_rr(e, X13, X13, X12);
    /* Continue while more chunks remain: after the subs, x13 = old_rem
     * - VL; old_rem > VL (unsigned) iff x13 > 0 (signed). b.gt reads
     * the signed result; equivalently b.hi would test old_rem > VL via
     * the carry flag. Both are correct for rem in 1..group-1. */
    em_bcc(e, CC_GT, chunk);
}

/* ---------------- prologue / epilogue ---------------- */

static void gen_function(cg_t *cg)
{
    em_t *e = &cg->em;
    const plan_t *pl = cg->pl;
    const rj_options *o = cg->opts;
    int loop = em_label(e), tail = em_label(e), done = em_label(e);

    /* x19 is the P_HORNER tail's data-byte register (callee-saved).
     * A notail twin never emits the scalar tail, so it needs no x19; a
     * predtail twin's vector tail needs none either. */
    int save_x19 = !pl->noop && !cg->notail && !cg->predtail &&
                   pl->kind == P_HORNER;
    if (save_x19)
        em_str64_pre(e, 19, XSP, -16);

    /* callee-saved NEON regs (v8..vmax, rounded up to a pair) */
    int vmaxr = -1;
    if (!cg->is_sve && cg->vmax >= 8) {
        vmaxr = cg->vmax | 1;
        for (int r = 8; r <= vmaxr; r += 2)
            em_neon_stp_pre(e, r, r + 1, XSP, -32);
    }

    em_mov_rr(e, X8, X0);
    em_mov_rr(e, X9, X1);

    /* pool address. Whether the body/tail reference the pool is known
     * statically from the plan kind (pool entries are allocated lazily
     * during body emission, so ncent is not usable here yet). */
    int need_pool = pl->kind == P_MULACC ||
                    ((pl->kind == P_DELTA || pl->kind == P_HORNER) &&
                     (cg->need_mask || cg->need_logexp));
    if (need_pool)
        em_adr(e, X16, cg->pool_lab);

    if (!pl->noop) {
        em_mov_rr(e, X11, XZR);   /* x11 = byte offset */
        /* A notail twin is only ever called with len % group == 0 (the
         * dispatcher guarantees it), so the counted loop always ends
         * with offset == len: route the empty-loop exit straight to
         * done. The tail path is not emitted. */
        int empty_lab = cg->notail ? done : tail;
        if (cg->is_sve) {
            em_sve_cntb(e, X12);              /* x12 = VL bytes */
            /* x13 = x12 * U = group bytes. U is a JIT-time constant:
             * fold the multiply at compile time. Powers of two use a
             * single LSL, U=1 is a mov; only non-pow2 U needs movz+mul.
             * (The prologue is not uop-counted; this saves 1 executed
             * instruction per kernel call.) */
            if (cg->U == 1) {
                em_mov_rr(e, X13, X12);
            } else if (cg->U == 2 || cg->U == 4 || cg->U == 8) {
                em_lsl_ri(e, X13, X12,
                          (unsigned)(cg->U == 2 ? 1 : cg->U == 4 ? 2 : 3));
            } else {
                em_movz(e, X13, (unsigned)cg->U);
                em_mul(e, X13, X12, X13);
            }
            em_udiv(e, X14, X2, X13);        /* x14 = niter */
            em_sve_whilelo(e, 0, XZR, X12);  /* p0 = all true */
            /* Vector constants (mask, nibble tables) are set up before
             * the empty-loop branch: when len < group the main loop is
             * skipped, but the predicated tail is a vector tail and
             * needs them. (The scalar tail never touches vector
             * registers, so skipping the setup was harmless before the
             * predtail twin existed.) */
            gen_const_setup(cg);
            em_cbz(e, X14, empty_lab);
        } else {
            /* x13 = len / group. Group not necessarily pow2: use udiv. */
            em_movz(e, X14, (unsigned)cg->group);
            em_udiv(e, X13, X2, X14);
            em_cbz(e, X13, empty_lab);
            gen_const_setup(cg);
        }
        /* Pipe prologue (track I8): issue the first iteration's
         * first-column loads. After the empty-loop branch (N >= 1),
         * before the loop label; x11 == 0 here. */
        if (cg->cand.pipe) {
            int cols[RJ_MAX_DISKS];
            if (pipe_used_columns(cg, cols) > 0)
                gen_pipe_prologue(cg, cols[0]);
        }
        em_bind(e, loop);
        {
            int al = o->loop_align ? o->loop_align : 16;
            int lg = 0;
            while ((1 << lg) < al)
                lg++;
            em_align(e, (unsigned)lg);
        }
        if (cg->cand.pipe)
            gen_body_pipe(cg);
        else
            gen_body(cg);
        if (cg->is_sve) {
            em_add_rr(e, X11, X11, X13);
            em_subs_ri(e, X14, X14, 1);
            em_bcc(e, CC_NE, loop);
        } else {
            em_add_ri(e, X11, X11, (unsigned)cg->group);
            em_subs_ri(e, X13, X13, 1);
            em_bcc(e, CC_NE, loop);
        }
        /* NT stores above used no fence; the caller must execute
         * DMB ISHST if cross-agent visibility is required. */
        if (!cg->notail) {
            em_bind(e, tail);
            em_cmp_rr(e, X11, X2);
            em_bcc(e, CC_HS, done);
            if (cg->predtail)
                gen_pred_tail(cg);
            else
                gen_tail(cg);
        }
        em_bind(e, done);
    }

    if (vmaxr >= 8) {
        for (int r = vmaxr - 1; r >= 8; r -= 2)
            em_neon_ldp_post(e, r, r + 1, XSP, 32);
    }
    if (save_x19)
        em_ldr64_post(e, 19, XSP, 16);
    em_ret(e);
    /* Trap sled after ret, unconditional: a fall-through past ret is a
     * codegen bug and must fault immediately, never slide through NOPs
     * into the const pool. */
    em_udf(e);
    /* pool_lab is bound by the caller after 64-byte alignment */
}

/* ---------------- entry ---------------- */

static void compute_regs(cg_t *cg)
{
    const plan_t *pl = cg->pl;
    int n_out = pl->n_out, U = cg->U;
    if (cg->cand.xorbit) {
        /* Bit-program twin: n_out*U accumulators + input-byte reg +
         * two bit temps + 0x01 mask. The nibble tables are dead.
         * Must stay in lockstep with xorbit_need_regs. */
        assert(pl->bit_sched && pl->bit_sched->ntemps > 0);
        assert(pl->kind == P_MULACC || pl->kind == P_DELTA ||
               pl->kind == P_HORNER);
        cg->xbt_in = n_out * U;
        cg->xbt_r1 = n_out * U + 1;
        cg->xbt_r2 = n_out * U + 2;
        cg->xbt_mask = n_out * U + 3;
        cg->vmax = cg->xbt_mask;
    } else if (cg->cand.pipe) {
        /* Software-pipelined shape (track I8): the D/Dn datasets
         * (U regs each) replace the single input temp t (dead: -1).
         * jit_build_version rejects unsupported (kind, mul, remat)
         * combos, so only P_MULACC/P_HORNER x MUL_NIBBLE, non-remat,
         * reach here. Must stay in lockstep with plan_need_regs'
         * pipe formulas (asserted below). */
        assert(pl->kind == P_MULACC || pl->kind == P_HORNER);
        assert(!cg->cand.remat);
        cg->t = -1;
        cg->pipe_d = n_out * U;
        cg->pipe_dn = n_out * U + U;
        cg->tlo = n_out * U + 2 * U;
        cg->thi = cg->tlo + 1;
        if (pl->kind == P_MULACC) {
            cg->r1 = cg->tlo + 2;
            cg->r2 = cg->tlo + 3;
            cg->tbl_lo = cg->tlo + 4;
            cg->tbl_hi = cg->tlo + 5;
            cg->mask = cg->tlo + 6;
            cg->vmax = cg->mask;
        } else {
            /* HORNER: r1's input-temp role is taken by the staged
             * dataset; left unassigned (-1) so any accidental use
             * trips loudly instead of silently encoding v0/z0. */
            cg->r1 = -1;
            cg->r2 = cg->tlo + 2;
            cg->mask = cg->tlo + 3;
            cg->hreg =
                malloc((size_t)2 * (size_t)n_out * sizeof(*cg->hreg));
            assert(cg->hreg);
            int r = cg->tlo + 4;
            for (int j = 0; j < n_out; j++) {
                uint8_t c = pl->horner_c[j];
                if (c == 0 || c == 1) {
                    cg->hreg[2 * j] = cg->hreg[2 * j + 1] = -1;
                    continue;
                }
                cg->hreg[2 * j] = r++;
                cg->hreg[2 * j + 1] = r++;
            }
            cg->vmax = n_out ? r - 1 : cg->mask;
        }
    } else if (pl->kind == P_XOR) {
        cg->t = n_out * U;
        cg->vmax = cg->t;
        if (cg->cand.xorsched) {
            /* XOR-schedule twin: temp formation needs two live vector
             * temps (t, t2); AArch64 has no vector XOR-with-memory. */
            cg->t2 = n_out * U + 1;
            cg->vmax = cg->t2;
        }
    } else if (pl->kind == P_HORNER) {
        cg->r1 = n_out * U;
        cg->t = cg->r1;   /* t aliases r1 for HORNER */
        cg->tlo = n_out * U + 1;
        cg->thi = n_out * U + 2;
        cg->r2 = n_out * U + 3;
        cg->mask = n_out * U + 4;
        cg->vmax = cg->mask;
        cg->hreg = malloc((size_t)2 * (size_t)n_out * sizeof(*cg->hreg));
        assert(cg->hreg);
        if (cg->cand.remat) {
            /* Rematerialize: no persistent table regs; hreg[] stays -1
             * and is never read (gen_const_setup and the body both
             * branch on cand.remat; the c_ldpool assert would fire if a
             * -1 ever reached the emitter). Tables are reloaded from the
             * pool into tbl_lo on each use.
             * Pressure proof: n_out*U accumulators (0..n_out*U-1) plus
             * t/r1 (n_out*U), tlo (+1), thi (+2), r2 (+3), mask (+4),
             * tbl_lo (+5): exactly n_out*U+6 vector regs, matching the
             * plan_need_regs P_HORNER remat branch (asserted below). */
            for (int j = 0; j < n_out; j++)
                cg->hreg[2 * j] = cg->hreg[2 * j + 1] = -1;
            cg->tbl_lo = n_out * U + 5;
            cg->vmax = cg->tbl_lo;
        } else {
            /* Table registers are compacted to nontrivial rows only:
             * rows with coefficient 0/1 are pure XOR and need no table.
             * Unused slots stay -1 and are never read (gen_const_setup
             * and the body both skip trivial rows). */
            int r = n_out * U + 5;
            for (int j = 0; j < n_out; j++) {
                uint8_t c = pl->horner_c[j];
                if (c == 0 || c == 1) {
                    cg->hreg[2 * j] = cg->hreg[2 * j + 1] = -1;
                    continue;
                }
                cg->hreg[2 * j] = r++;
                cg->hreg[2 * j + 1] = r++;
            }
            if (n_out)
                cg->vmax = r - 1;
        }
    } else { /* P_MULACC, P_DELTA */
        cg->t = n_out * U;
        if (cg->need_mask) {
            /* Hoisted tables: tbl_lo/tbl_hi hold one row's NIBLO/NIBHI
             * tables across the lane loop (mulacc_column), so both are
             * live together with t,tlo,thi,r1,r2,mask: n_out*U+8 regs. */
            cg->tlo = n_out * U + 1;
            cg->thi = n_out * U + 2;
            cg->r1 = n_out * U + 3;
            cg->r2 = n_out * U + 4;
            cg->tbl_lo = n_out * U + 5;
            cg->tbl_hi = n_out * U + 6;
            cg->mask = n_out * U + 7;
            cg->vmax = cg->mask;
        } else {
            /* All coefficients in {0,1} (only P_DELTA reaches here: the
             * planner guarantees a nontrivial coefficient for
             * P_MULACC). The nibble split, its tables and the mask never
             * execute, so tlo/thi/r2/tbl_lo/tbl_hi/mask are dead and not
             * allocated. r1 survives as the delta temp: the body forms
             * old^new into r1 for every column on both NEON and SVE. */
            if (pl->kind == P_DELTA) {
                cg->r1 = n_out * U + 1;
                cg->vmax = cg->r1;
            } else {
                cg->vmax = cg->t;
            }
        }
    }
    assert(cg->vmax < 16 || cg->is_sve);
    /* SVE uses 24 vector regs (z0-z7 + z16-z31; z8-z15 callee-saved, never
     * touched). The logical count must fit: plan_need_regs (via
     * isa_nregs) enforces the same bound in the planner filter. */
    if (cg->is_sve)
        assert(cg->vmax < 24);
    /* Single source of truth: the planner's filter must agree exactly
     * with this assignment, or candidates could be wrongly dropped or
     * (worse) admitted past the regfile. The xorsched twin has its own
     * pressure formula (extra temp). */
    int want_regs = cg->cand.xorbit
                        ? xorbit_need_regs(cg->is_sve ? ISA_SVE : ISA_NEON,
                                           pl, U)
                        : cg->cand.xorsched
                        ? xorsched_need_regs(cg->is_sve ? ISA_SVE : ISA_NEON,
                                             pl, U)
                        : plan_need_regs(cg->is_sve ? ISA_SVE : ISA_NEON,
                                         MUL_NIBBLE, pl, U, cg->cand.remat,
                                         cg->cand.pipe, 0, cg->cand.crc_fold);
    assert(cg->vmax + 1 == want_regs);
}

int jit_build_version(const plan_t *pl, const cand_t *cand,
                      const rj_cpu_info *cpu, const rj_options *opts,
                      version_t *v, char *err, size_t errlen)
{
    cg_t cg;
    memset(&cg, 0, sizeof(cg));
    cg.pl = pl;
    cg.cand = *cand;
    /* The planner never generates remat twins for other plan kinds, and
     * the backend only implements remat for P_HORNER. User-reachable
     * via pinned candidates: fail cleanly, never abort. */
    if (cand->remat && pl->kind != P_HORNER) {
        snprintf(err, errlen,
                 "aarch64 backend: remat twin needs P_HORNER");
        return -1;
    }
    /* Software-pipelined loop twin (track I8): implemented for the
     * P_MULACC/P_HORNER nibble bodies only (non-remat). Anything else
     * fails the build cleanly -- never wrong code; the planner only
     * enumerates the supported combinations. */
    if (cand->pipe &&
        ((pl->kind != P_MULACC && pl->kind != P_HORNER) ||
         cand->remat || cand->xorsched)) {
        snprintf(err, errlen,
                 "aarch64 backend: pipe twin needs P_MULACC/P_HORNER "
                 "nibble, non-remat");
        return -1;
    }
    /* Recovery closed-form emitter twin (algos #4): the backend emits
     * the syndrome+combine body (gen_body_rec_cf) instead of the generic
     * MULACC over the baked R matrix. The planner only enumerates it on
     * P_MULACC recovery plans carrying rec_cf_data for nf in {1,2},
     * never combined with another body twin; anything else (including
     * force-built candidates) fails cleanly here -- never wrong code. */
    if (cand->rec_cf) {
        const struct rec_cf_data *cfd = pl->rec_cf;
        if (!cfd || pl->kind != P_MULACC ||
            (cfd->nf != 1 && cfd->nf != 2) ||
            pl->n_out != cfd->nf || pl->n_in != cfd->ns ||
            !plan_any_nontrivial(pl) ||
            cand->pipe || cand->remat || cand->xorsched ||
            cand->xorbit || cand->permi2b || cand->mshift) {
            snprintf(err, errlen,
                     "aarch64 backend: rec_cf twin needs a P_MULACC "
                     "recovery plan with rec_cf_data (nf 1..2, "
                     "n_out == nf, n_in == ns), a nontrivial baked "
                     "coefficient, and no other body twin");
            return -1;
        }
    }
    cg.opts = opts;
    cg.cpu = cpu;
    cg.is_sve = (cand->isa == ISA_SVE);
    cg.vl = cg.is_sve ? cpu->sve_vl : 16; /* sve_vl is in bytes */
    if (cg.is_sve) {
        /* Build-time VL must equal the runtime VL the generated code
         * reads via cntb (x13 = cntb * U below). PR_SVE_GET_VL is
         * unreliable under qemu-user: qemu 8.2.2 interprets
         * -cpu max,sve-default-vector-length=N as N *bytes* (not bits),
         * so prctl reports N/8 bytes while the true emulated VL is N
         * bytes (verified 2026-10-02: for N=128, cntb=128 and an SVE
         * ld1b/st1b round-trip under whilelo(cntb) moves 128 bytes;
         * rdvl agrees with prctl's wrong 16). Read the true VL
         * directly via cntb — this is a build-time constant (same CPU
         * runs the JIT output), not a runtime computation. On native
         * hardware prctl and cntb agree, so this changes nothing there.
         * See docs/length_dispatch.md §11.0 and
         * tests/test_boundaries.c:sve_vl_bytes (same pattern). */
        size_t true_vl;
#ifdef RJ_WASM
        /* Compile-only (WebAssembly, tools/wasm/SPEC.md §S6.2b): no
         * hardware to query; the VL comes from the synthetic CPU bundle
         * (cpu->sve_vl), which is also the value assigned to cg.vl
         * above. Native builds never define RJ_WASM. */
        true_vl = (size_t)cpu->sve_vl;
#else
        __asm__ volatile(".arch armv8-a+sve\n\tcntb %0" : "=r"(true_vl));
#endif
        cg.vl = (int)true_vl;
    }
    cg.U = cand->U;
    cg.group = cg.U * cg.vl;
    cg.nt = cand->nt;
    /* PRFM imm12 x 8: encodable distances are 0..32760, multiple of 8.
     * Fail the twin build cleanly on anything else (same rule as the
     * RVV Zicbop backend) -- never silently truncate the immediate
     * field. Checked on the size_t before narrowing to unsigned. */
    if (cand->prefetch_dist > 32760 || (cand->prefetch_dist & 7u) != 0) {
        snprintf(err, errlen,
                 "aarch64 backend: prefetch distance %zu not encodable "
                 "(PRFM immediate: 0..32760, multiple of 8)",
                 cand->prefetch_dist);
        return -1;
    }
    cg.prefetch_dist = cand->prefetch_dist;
    cg.notail = cand->notail;
    cg.predtail = cand->predtail;
    /* lmul is an RVV-only axis: reject cleanly rather than silently
     * ignoring it. Only 0 (means 1) and 1 are valid here; the explicit
     * != form also rejects negative values (defense in depth). */
    if (cand->lmul != 0 && cand->lmul != 1) {
        snprintf(err, errlen,
                 "lmul=%d requires RVV (not implemented for AArch64)",
                 cand->lmul);
        return -1;
    }
    /* notail and predtail are disjoint twins: the first omits the tail,
     * the second replaces it with a predicated vector tail. */
    assert(!(cg.notail && cg.predtail));
    if (cg.predtail && !cg.is_sve) {
        snprintf(err, errlen,
                 "predtail twin requires SVE (not implemented for %s)",
                 cg.is_sve ? "sve" : "this ISA");
        return -1;
    }
    /* pipe+predtail is not implemented: the predicated tail reuses the
     * non-pipelined gen_body, whose register layout (cg->t live) is
     * dead in pipe mode (cg->t = -1; the D/Dn datasets replace it).
     * Reject cleanly instead of tripping the V() register assert.
     * (Found 2026-10-07: I8's pipe twin can win heuristic_select, and
     * api.c builds the predtail twin from the winner unconditionally.)
     */
    if (cg.predtail && cg.cand.pipe) {
        snprintf(err, errlen,
                 "predtail twin not implemented for the pipe "
                 "(software-pipelined) body");
        return -1;
    }
    if (cg.is_sve && cg.vl <= 0) {
        snprintf(err, errlen, "SVE selected but no vector length");
        return -1;
    }
    memset(v, 0, sizeof(*v));
    v->plan = pl; /* borrowed: bench validates against the owning plan */
    v->notail = cand->notail;
    v->predtail = cand->predtail;
    v->group_bytes = (size_t)cg.group;

    gf_build_logexp(cg.logtab, cg.exptab);
    /* need_logexp doubles as the "constant pool is needed" signal for
     * HORNER/DELTA (need_pool below): the vector body loads nibble tables
     * via the pool base X16. It must stay set even for a notail twin --
     * gating it on !notail broke the pool setup (the log/exp tables
     * themselves are allocated lazily and never emitted when gen_tail
     * doesn't run, so keeping the flag costs nothing). */
    if (pl->kind == P_MULACC || pl->kind == P_DELTA ||
        pl->kind == P_HORNER)
        cg.need_logexp = 1;
    if (pl->kind == P_MULACC || pl->kind == P_DELTA)
        /* Single source of truth: plan_any_nontrivial (see plan.h). The
         * old per-column (any && neednib) scan is exactly this predicate:
         * neednib implies any, so it reduces to "some coefficient outside
         * {0,1}". For P_DELTA the scan covers the n_out x nchanged delta
         * columns, not n_out x n_in. */
        cg.need_mask = plan_any_nontrivial(pl);

    /* SVE2 EOR3 codegen (see docs/sve2_audit.md): on CPUs reporting SVE2
     * via hwcap, fold the nibble-multiply reduction into one EOR3 per
     * multiply (HORNER nontrivial (j,u): drops MOV + 1 EOR; MULACC/DELTA
     * accumulate: 2 EOR -> 1 EOR3). The gate is exact: use_eor3 is set
     * iff the backend will emit at least one EOR3, so the "+eor3" desc
     * tag means "this build contains EOR3 instructions". In particular
     * a nontrivial coefficient on its row's *first* nonzero column takes
     * the copy+xor path (destination dead, not zero) and does not count.
     * The baseline SVE path is untouched for non-SVE2 machines. */
    {
        int nontriv = 0;
        if (pl->kind == P_HORNER) {
            /* EOR3 sites live in the i>0 loop, so n_in >= 2 is required
             * (n_in <= 1 HORNER is unreachable from the planner; this is
             * defensive). */
            if (pl->n_in >= 2) {
                for (int j = 0; j < pl->n_out; j++) {
                    uint8_t c = pl->horner_c[j];
                    if (c != 0 && c != 1) {
                        nontriv = 1;
                        break;
                    }
                }
            }
        } else if (pl->kind == P_MULACC) {
            /* EOR3 is emitted only on the accumulate path: a nontrivial
             * coefficient at a column past its row's first nonzero
             * column (mirrors the first_col logic consumed by
             * mulacc_column). */
            for (int j = 0; j < pl->n_out && !nontriv; j++) {
                int first_nz = -1;
                for (int i = 0; i < pl->n_in; i++) {
                    if (pl->mat[(size_t)j * (size_t)pl->n_in + (size_t)i]) {
                        first_nz = i;
                        break;
                    }
                }
                for (int i = 0; i < pl->n_in; i++) {
                    uint8_t c =
                        pl->mat[(size_t)j * (size_t)pl->n_in + (size_t)i];
                    if (c != 0 && c != 1 && i != first_nz) {
                        nontriv = 1;
                        break;
                    }
                }
            }
        } else if (pl->kind == P_DELTA) {
            /* P_DELTA passes first_col=NULL (accumulators are seeded from
             * old parity), so every nontrivial coefficient takes the
             * accumulate path: need_mask is already exact. */
            nontriv = cg.need_mask;
        } else {
            nontriv = 0; /* P_XOR and any future kind: no EOR3 */
        }
        cg.use_eor3 = cg.is_sve && cpu->sve2 && nontriv;
    }

    /* rec_cf twin: the body above is the generic MULACC gate over the
     * baked R matrix, but gen_body_rec_cf emits the syndrome+combine
     * form, whose EOR3 sites differ. Recompute the gate from the cf
     * coefficients so it stays exact: a syndrome takes the accumulate
     * (EOR3-eligible) path for a nontrivial coefficient past its first
     * contributing column (mirrors cf_syndrome_column's first logic);
     * the e=2 combine's b-term (t ^= b*q, t live) is also eligible.
     * Dead-dst multiplies (final cinv, d/c/a terms, detinv) never use
     * EOR3. */
    if (cand->rec_cf) {
        const struct rec_cf_data *cfd = pl->rec_cf;
        int nontriv = 0;
        if (cfd->nf == 1) {
            int first = -1;
            for (int s = 0; s < cfd->ns && first < 0; s++)
                if (cfd->ecoef[s])
                    first = s;
            for (int s = 0; s < cfd->ns && !nontriv; s++)
                if (s != first && cfd->ecoef[s] != 0 &&
                    cfd->ecoef[s] != 1)
                    nontriv = 1;
        } else {
            int first1 = -1, first2 = -1;
            for (int s = 0; s < cfd->ns && first1 < 0; s++)
                if (cfd->ecoef1[s])
                    first1 = s;
            for (int s = 0; s < cfd->ns && first2 < 0; s++)
                if (cfd->ecoef2[s])
                    first2 = s;
            for (int s = 0; s < cfd->ns && !nontriv; s++) {
                if (s != first1 && cfd->ecoef1[s] != 0 &&
                    cfd->ecoef1[s] != 1)
                    nontriv = 1;
                if (s != first2 && cfd->ecoef2[s] != 0 &&
                    cfd->ecoef2[s] != 1)
                    nontriv = 1;
            }
            if (cfd->b != 0 && cfd->b != 1)
                nontriv = 1;
        }
        cg.use_eor3 = cg.is_sve && cpu->sve2 && nontriv;
    }

    /* Over-pressure must fail cleanly (return -1), not abort. The
     * planner's filter normally prevents this, but force-built
     * candidates (tests) must also be safe. The xorbit twin has its
     * own (smaller) register footprint; checking it against the naive
     * plan_need_regs would wrongly reject valid xorbit candidates
     * (e.g. P_HORNER, whose naive body needs hregs the xorbit body
     * never touches). */
    {
        int need = cand->xorbit
            ? xorbit_need_regs(cg.is_sve ? ISA_SVE : ISA_NEON, pl, cand->U)
            : plan_need_regs(cg.is_sve ? ISA_SVE : ISA_NEON,
                             MUL_NIBBLE, pl, cand->U, cand->remat,
                             cand->pipe, 0, cand->crc_fold);
        int nregs = isa_nregs(cg.is_sve ? ISA_SVE : ISA_NEON);
        if (need > nregs) {
            snprintf(err, errlen,
                     "aarch64 backend: register pressure (needs %d regs, "
                     "isa has %d)", need, nregs);
            free(cg.hreg);
            return -1;
        }
    }

    compute_regs(&cg);

    uint8_t *buf = malloc(1 << 20);
    if (!buf)
        goto oom;
    em_init(&cg.em, buf, 1 << 20);
    cg.pool_lab = em_label(&cg.em);
    cg.pool_cap = 4096;
    cg.pool = malloc(cg.pool_cap);
    if (!cg.pool) {
        free(buf);
        goto oom;
    }

    gen_function(&cg);

    /* constant pool after the code, 64-byte aligned. The dead space
     * between ret and the pool is trap-filled (never NOPs). */
    em_trap_align(&cg.em, 6);
    em_bind(&cg.em, cg.pool_lab);
    size_t code_size = cg.em.len;
    for (size_t i = 0; i < cg.pool_len; i++)
        em_byte(&cg.em, cg.pool[i]);
    em_resolve(&cg.em);
    if (cg.em.err) {
        snprintf(err, errlen, "codegen error (branch/pool out of range)");
        free(buf);
        goto fail;
    }
    size_t total = (cg.em.len + 4095) & ~(size_t)4095;
#ifdef RJ_WASM
    /* Compile-only (WebAssembly, tools/wasm/SPEC.md §S6.2): no
     * mmap/mprotect in the browser and no I-cache maintenance to
     * perform. The emitters are position-independent, so the bytes are
     * the contract; keep exactly cg.em.len bytes in a malloc'd block.
     * Native builds never define RJ_WASM. */
    void *mem = malloc(cg.em.len ? cg.em.len : 1);
    if (!mem) {
        free(buf);
        snprintf(err, errlen, "malloc failed");
        goto fail;
    }
    total = cg.em.len;
    memcpy(mem, cg.em.buf, cg.em.len);
#else
    void *mem = mmap(NULL, total, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) {
        snprintf(err, errlen, "mmap failed");
        free(buf);
        goto fail;
    }
    memcpy(mem, cg.em.buf, cg.em.len);
    /* AArch64 needs I-cache maintenance before executing new code */
    __builtin___clear_cache((char *)mem, (char *)mem + total);
    if (mprotect(mem, total, PROT_READ | PROT_EXEC) != 0) {
        munmap(mem, total);
        snprintf(err, errlen, "mprotect failed");
        goto oom;
    }
#endif

    v->cand = *cand;
    v->kind = pl->kind;
    v->mem = mem;
    v->memsz = total;
    v->fn = (rj_fn)mem;
    v->gbps = -1;
    v->vl = cg.vl;
    v->code_size = code_size;
    v->const_size = cg.pool_len;
    v->uops_per_byte = cg.uops / (double)cg.group;
    v->mem_per_byte = cg.mem / (double)cg.group;

    const char *isa = cg.is_sve ? "sve" : "neon";
    if (pl->kind == P_XOR)
        snprintf(v->desc, sizeof(v->desc), "%.65s %s xor U=%d%s%s%s%s%s",
                 pl->name, isa, cg.U,
                 cg.cand.xorsched ? "+xsched" : "",
                 cg.nt ? "+nt" : "",
                 cg.prefetch_dist ? "+pf" : "",
                 cand->notail ? " notail" : "",
                 cand->predtail ? " predtail" : "");
    else
        snprintf(v->desc, sizeof(v->desc), "%.65s %s %s+nibble U=%d%s%s%s%s%s%s%s%s",
                 pl->name, isa,
                 pl->kind == P_DELTA ? "delta" :
                 pl->kind == P_HORNER ? "horner" : "mulacc",
                 cg.U, cg.nt ? "+nt" : "", cg.prefetch_dist ? "+pf" : "",
                 cg.cand.remat ? "+remat" : "",
                 cg.use_eor3 ? "+eor3" : "",
                 cand->pipe ? "+pipe" : "",
                 cand->rec_cf ? "+reccf" : "",
                 cand->notail ? " notail" : "",
                 cand->predtail ? " predtail" : "");

    em_free(&cg.em);
    free(buf);
    free(cg.pool);
    free(cg.cent);
    free(cg.hreg);
    return 0;

oom:
    snprintf(err, errlen, "out of memory");
fail:
    em_free(&cg.em);
    free(cg.pool);
    free(cg.cent);
    free(cg.hreg);
    return -1;
}

void jit_free_version(version_t *v)
{
    if (v->mem) {
#ifdef RJ_WASM
        /* Paired with the malloc pivot in jit_build_version (§S6.2). */
        free(v->mem);
#else
        munmap(v->mem, v->memsz);
#endif
    }
    v->mem = NULL;
    v->fn = NULL;
}
