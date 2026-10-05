/* raidjit AArch64 code generator (NEON + SVE/SVE2).
 *
 * Implements jit_build_version() from src/jit.h. Mirrors the x86-64
 * backend's algorithms (P_XOR / P_MULACC / P_DELTA / P_HORNER, constant
 * pool, scalar tail) with AArch64 instruction selection:
 *
 *  - NEON: 128-bit vectors, TBL (nibble method) for GF multiplies.
 *  - SVE:  VL-agnostic, z16-z31 only (logical r -> z(16+r)), p0 only.
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
    int t, tlo, thi, r1, r2, tbl, mask; /* logical vector regs */
    int t2; /* XOR-schedule 2nd temp (P_XOR xorsched twin only) */
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

/* logical vector reg -> physical: NEON identity, SVE z(16+r) */
static int V(const cg_t *cg, int r)
{
    return cg->is_sve ? 16 + r : r;
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

/* load lane u of the vector at ptr into logical reg r (ptr already holds
 * base+offset). SVE bumps ptr by the runtime VL (x12); the compile-time
 * cg->vl must NOT be used for SVE address arithmetic. In predicated-tail
 * mode (cg->pred) the load runs under p1 and the bump is skipped: the
 * chunk loop recomputes every pointer from x11 each iteration. */
static void c_ld_to(cg_t *cg, int r, int u, int ptr)
{
    em_t *e = &cg->em;
    if (cg->is_sve) {
        em_sve_ld1b(e, V(cg, r), cg->pred ? 1 : 0, ptr);
        if (!cg->pred)
            em_add_rr(e, ptr, ptr, X12);   /* x12 = runtime VL bytes */
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
        if (cg->nt)
            em_sve_stnt1b(e, V(cg, r), cg->pred ? 1 : 0, ptr);
        else
            em_sve_st1b(e, V(cg, r), cg->pred ? 1 : 0, ptr);
        if (!cg->pred)
            em_add_rr(e, ptr, ptr, X12);   /* x12 = runtime VL bytes */
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
    if ((pl->kind == P_MULACC || pl->kind == P_DELTA) && cg->need_mask) {
        if (cg->is_sve)
            em_sve_dup(e, V(cg, cg->mask), 15);
        else
            em_neon_movi(e, cg->mask, 15);
        cg->uops += 1;
    } else if (pl->kind == P_HORNER) {
        if (cg->is_sve)
            em_sve_dup(e, V(cg, cg->mask), 15);
        else
            em_neon_movi(e, cg->mask, 15);
        cg->uops += 1;
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
 * cg->t holds the per-lane multiplicand (tlo/thi already split when the
 * nibble path is needed); mat is n_out x ncols row-major.
 * first_col[j] is the first column with a nonzero entry in row j, or
 * NULL when accumulators are not provably zero (P_DELTA seeds from
 * oldpar). At the first contributing column the accumulator is known
 * zero, so the update is a move instead of an xor-with-zero. */
static void mulacc_column(cg_t *cg, int col, int ncols, int u,
                          const int *first_col)
{
    const plan_t *pl = cg->pl;
    int n_out = pl->n_out;
    for (int j = 0; j < n_out; j++) {
        uint8_t c = pl->mat[(size_t)j * (size_t)ncols + (size_t)col];
        if (!c)
            continue;
        int acc = j * cg->U + u;
        /* HORNER never reaches this function: it uses the inline i-loop
         * in gen_body, not mulacc_column. */
        assert(pl->kind != P_HORNER);
        int first = first_col && col == first_col[j];
        if (c == 1) {
            if (first)
                c_copy(cg, acc, cg->t); /* acc was 0 */
            else
                c_xor(cg, acc, cg->t);
        } else {
            c_ldpool(cg, cg->tbl, CK_NIBLO, c);
            c_tbl_into(cg, cg->r1, cg->tbl, cg->tlo);
            c_ldpool(cg, cg->tbl, CK_NIBHI, c);
            c_tbl_into(cg, cg->r2, cg->tbl, cg->thi);
            if (first) {
                /* acc is dead on this path (first contributing column;
                 * the accumulator is overwritten before its first read),
                 * so acc = r1^r2 is a single unpredicated EOR on SVE --
                 * no mov needed, and the unpredicated form never reads
                 * the dead acc. EOR3 is NOT usable here: it computes
                 * Zdn^Zm^Za, i.e. it folds acc's current value in, but
                 * acc is merely dead (uninitialized), not zero. NEON
                 * keeps mov+eor (its EOR is already 3-operand but the
                 * c_xor helper is destructive). */
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

static void gen_body(cg_t *cg)
{
    em_t *e = &cg->em;
    const plan_t *pl = cg->pl;
    int n_in = pl->n_in, n_out = pl->n_out, U = cg->U;
    /* Predicated-tail mode processes one lane per chunk (the u=0 slice);
     * the chunk loop re-enters gen_body per VL-sized chunk. */
    int ulanes = cg->pred ? 1 : U;

    /* Accumulator seeding. Every live accumulator is overwritten before
     * its first read (see x86 backend for the argument); dead rows are
     * zeroed here. */
    if (pl->kind == P_HORNER && n_in == 0) {
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
            for (int j = 0; j < n_out; j++)
                if (pl->mat[(size_t)j * (size_t)n_in + (size_t)i])
                    nu++;
            if (!nu)
                continue;
            c_inptr(cg, X10, i);   /* X10 = in[i] + x11 */
            for (int u = 0; u < ulanes; u++) {
                /* Load once per (i,u) into t; each row then either
                 * initializes its accumulator from t (first contributing
                 * column) or XORs t in. The old code loaded per (j,u)
                 * (2x input traffic for multi-row P_XOR). t = n_out*U
                 * never aliases an acc and is dead across the j-loop
                 * except as the shared source. The load's uops/mem
                 * accounting moves from per-(j,u) to per-(i,u); the
                 * per-row mov/xor account themselves. */
                if (cg->is_sve)
                    em_sve_ld1b(e, V(cg, cg->t), cg->pred ? 1 : 0, X10);
                else
                    em_neon_ldr(e, cg->t, X10, (unsigned)u * 16);
                cg->uops += 1;
                cg->mem += cg->vl;
                for (int j = 0; j < n_out; j++) {
                    if (!pl->mat[(size_t)j * (size_t)n_in + (size_t)i])
                        continue;
                    int acc = j * U + u;
                    if (i == first_i[j])
                        c_copy(cg, acc, cg->t);
                    else
                        c_xor(cg, acc, cg->t);
                }
                if (cg->is_sve && !cg->pred)
                    em_add_rr(e, X10, X10, X12);  /* bump by runtime VL */
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
            for (int u = 0; u < ulanes; u++) {
                if (u == 0) {
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
                }
                c_ld_to(cg, cg->t, u, X10);
                c_ld_to(cg, cg->r1, u, X3);
                c_xor(cg, cg->t, cg->r1);
                if (neednib) {
                    c_and3(cg, cg->tlo, cg->t);
                    c_lsr4(cg, cg->thi, cg->t);
                }
                mulacc_column(cg, i, nc, u, NULL);
            }
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
            for (int u = 0; u < ulanes; u++) {
                if (u == 0)
                    c_inptr(cg, X10, i);
                if (pl->kind == P_HORNER && i > 0) {
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
                                 * tbl scratch: every (i,j,u) needs both
                                 * tables, and tbl is shared across rows, so
                                 * the 2 loads per use are inherent -- the
                                 * table cannot be hoisted out of the u-loop
                                 * (same tradeoff as the x86 backend: a
                                 * second scratch would change the proven
                                 * n_out*U+6 pressure formula). */
                                c_ldpool(cg, cg->tbl, CK_NIBLO, c);
                                c_tbl_into(cg, lo_dst, cg->tbl, cg->tlo);
                                c_ldpool(cg, cg->tbl, CK_NIBHI, c);
                                c_tbl_into(cg, cg->r2, cg->tbl, cg->thi);
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
                    c_ld_to(cg, cg->t, u, X10);
                    if (pl->kind == P_HORNER && i == 0) {
                        /* Horner i==0: acc = d0 (no multiply) */
                        for (int j = 0; j < n_out; j++) {
                            int acc = j * U + u;
                            c_copy(cg, acc, cg->t);
                        }
                    } else {
                        if (neednib) {
                            c_and3(cg, cg->tlo, cg->t);
                            c_lsr4(cg, cg->thi, cg->t);
                        }
                        mulacc_column(cg, i, n_in, u, first_col);
                    }
                }
            }
        }
        free(first_col);
    }

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
            em_mov_wi(e, 12, 0);
            em_bind(e, lj);
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
            /* x13 = x12 * U = group bytes. U is a compile-time constant
             * but not necessarily a power of two, so use mul. */
            em_movz(e, X13, (unsigned)cg->U);
            em_mul(e, X13, X12, X13);
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
        em_bind(e, loop);
        {
            int al = o->loop_align ? o->loop_align : 16;
            int lg = 0;
            while ((1 << lg) < al)
                lg++;
            em_align(e, (unsigned)lg);
        }
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
    if (pl->kind == P_XOR) {
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
             * pool into tbl on each use.
             * Pressure proof: n_out*U accumulators (0..n_out*U-1) plus
             * t/r1 (n_out*U), tlo (+1), thi (+2), r2 (+3), mask (+4),
             * tbl (+5): exactly n_out*U+6 vector regs, matching the
             * plan_need_regs P_HORNER remat branch (asserted below). */
            for (int j = 0; j < n_out; j++)
                cg->hreg[2 * j] = cg->hreg[2 * j + 1] = -1;
            cg->tbl = n_out * U + 5;
            cg->vmax = cg->tbl;
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
            cg->tlo = n_out * U + 1;
            cg->thi = n_out * U + 2;
            cg->r1 = n_out * U + 3;
            cg->r2 = n_out * U + 4;
            cg->tbl = n_out * U + 5;
            cg->mask = n_out * U + 6;
            cg->vmax = cg->mask;
        } else {
            /* All coefficients in {0,1} (only P_DELTA reaches here: the
             * planner guarantees a nontrivial coefficient for
             * P_MULACC). The nibble split, its tables and the mask never
             * execute, so tlo/thi/r2/tbl/mask are dead and not allocated.
             * r1 survives as the delta temp: the body forms old^new into
             * r1 for every column on both NEON and SVE. */
            if (pl->kind == P_DELTA) {
                cg->r1 = n_out * U + 1;
                cg->vmax = cg->r1;
            } else {
                cg->vmax = cg->t;
            }
        }
    }
    assert(cg->vmax < 16 || cg->is_sve);
    /* SVE maps logical r -> z(16+r); p0-only keeps 16 regs live */
    if (cg->is_sve)
        assert(cg->vmax < 16);
    /* Single source of truth: the planner's filter must agree exactly
     * with this assignment, or candidates could be wrongly dropped or
     * (worse) admitted past the regfile. The xorsched twin has its own
     * pressure formula (extra temp). */
    int want_regs = cg->cand.xorsched
                        ? xorsched_need_regs(cg->is_sve ? ISA_SVE : ISA_NEON,
                                             pl, U)
                        : plan_need_regs(cg->is_sve ? ISA_SVE : ISA_NEON,
                                         MUL_NIBBLE, pl, U, cg->cand.remat);
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
     * the backend only implements remat for P_HORNER. */
    assert(!(cand->remat && pl->kind != P_HORNER));
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
        __asm__ volatile(".arch armv8-a+sve\n\tcntb %0" : "=r"(true_vl));
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
        snprintf(v->desc, sizeof(v->desc), "%.65s %s %s+nibble U=%d%s%s%s%s%s%s",
                 pl->name, isa,
                 pl->kind == P_DELTA ? "delta" :
                 pl->kind == P_HORNER ? "horner" : "mulacc",
                 cg.U, cg.nt ? "+nt" : "", cg.prefetch_dist ? "+pf" : "",
                 cg.cand.remat ? "+remat" : "",
                 cg.use_eor3 ? "+eor3" : "",
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
    if (v->mem)
        munmap(v->mem, v->memsz);
    v->mem = NULL;
    v->fn = NULL;
}
