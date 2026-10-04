/* tests/test_matrix_search.c — tests for the plan-time generator-matrix
 * search (src/matrix_search.c).
 *
 *  1. test_mds_proof: every searched matrix in the envelope is PROVEN MDS
 *     by exact minor enumeration (all C(ndata+npar, npar) minors), and an
 *     independent path (solve_recovery over all erasure subsets) agrees.
 *  2. test_mds_rejects: the MDS check rejects known-bad matrices
 *     (beyond-envelope default, duplicated rows, zero entries) and bad
 *     args -- the check is demonstrably able to fail. Plus the
 *     single-singular-minor corpus: two matrices with EXACTLY ONE
 *     singular minor (at enumeration indices 1 and 44 of 45), which
 *     catch count-preserving skipped-minor mutations that evade the
 *     rest of the corpus.
 *  3. test_cost_le_old: searched cost <= historical-default cost on a
 *     corpus (the old generator's formula is replicated here as a
 *     static copy for comparison; the canonical implementation
 *     rs_default_matrix() remains non-static in src/gf.c).
 *  4. test_cost_model_crosscheck (x86-64): the transcribed static cost
 *     model matches real jit_build_version uops/mem per byte with exact
 *     equality on a corpus of matrices -- pins the transcription
 *     against backend accounting drift.
 *  5. test_search_deterministic: repeated calls return identical bytes
 *     (exercises the cache path); dense envelope tripwire: search
 *     output is byte-identical to the historical default on EVERY
 *     in-envelope shape (npar=2,3 x ndata=1..51, npar=1 sampled) --
 *     the empirical guard behind the fixed-point conclusion.
 *  6. test_end_to_end: ISA x topology x op x length combination test --
 *     rj_compile (public API, searched matrices) vs an independent
 *     scalar reference. Runs natively and under QEMU (aarch64/riscv64).
 *  7. test_envelope: constructors still return NULL outside the envelope.
 *
 * Mutation evidence (done by hand during development, recorded in
 * docs/matrix_search.md):
 *  - A: matrix_search.c Horner nontrivial entry cost 7 -> 6:
 *    cost_model_crosscheck FAILS
 *    (default(8,2) horner: uops/byte model=0.9590 codegen=1.0684).
 *  - B: matrix_search.c winner forced to the Cauchy matrix (still MDS):
 *    cost_le_old FAILS on all multi-parity shapes
 *    (e.g. (51,3): searched 17.6211 > old 12.6992) and
 *    search_deterministic FAILS on the format-change tripwire.
 *  - C: matrix_search.c singular-minor rejection neutered (&& 0):
 *    mds_rejects FAILS on all 5 bad matrices, mds_proof FAILS via the
 *    independent solver check, and the search itself wanders to a
 *    non-default matrix (the proof gate matters).
 *  - D (T1): matrix_search.c count-preserving skipped-minor mutation
 *    (`if (n != SKIP && !minor_nonsingular(...))`, n still counts every
 *    minor): with SKIP=44 the pre-hardening suite PASSES SILENTLY
 *    (full evasion -- every old bad matrix has singular minors at
 *    other positions); the hardened suite FAILS on the
 *    single-singular-minor corpus ("matrix B accepted as MDS").
 *    With SKIP=1 the old suite fails only incidentally (the blind
 *    search wanders for (3,2), caught by the solver check); the
 *    hardened suite fails directly on matrix A. Mutation reverted.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "../include/raidjit.h"
#include "../src/matrix_search.h"
#include "../src/plan.h"
#include "../src/gf.h"
#include "../src/ref.h"
#if defined(__x86_64__)
#include "../src/jit.h"
#endif

static int failures = 0;
#define CHECK(c, ...) do {                                             \
    if (!(c)) {                                                        \
        printf("FAIL "); printf(__VA_ARGS__); printf("\n"); failures++; \
    }                                                                  \
} while (0)
#define TEST_BEGIN int _f0 = failures
#define TEST_OK(name) do {                                             \
    if (failures == _f0)                                                \
        printf("ok " name "\n");                                      \
} while (0)

/* ---- the historical default generator, test-only copy ----
 * M[j][i] = 2^(j*(ndata-1-i)). This is the pre-search generator,
 * kept here solely for old-vs-new cost comparison; it is not used by
 * the shipped library. */
static void old_default_matrix(int ndata, int npar, uint8_t *m)
{
    for (int j = 0; j < npar; j++)
        for (int i = 0; i < ndata; i++)
            m[(size_t)j * (size_t)ndata + (size_t)i] =
                gf_pow(2, (unsigned)(j * (ndata - 1 - i)));
}

/* Cauchy matrix for cost comparison: MDS by proof, not Vandermonde. */
static void cauchy_matrix(int ndata, int npar, uint8_t *m)
{
    for (int j = 0; j < npar; j++)
        for (int i = 0; i < ndata; i++)
            m[(size_t)j * (size_t)ndata + (size_t)i] =
                gf_inv((uint8_t)((i + 1) ^ (128 + j)));
}

static long binom2(int n, int k)
{
    long r = 1;
    for (int i = 0; i < k; i++)
        r = r * (n - i) / (i + 1);
    return r;
}

/* ---- 1. MDS proof + independent agreement ---- */

/* independent MDS verification via the library's own recovery solver:
 * every erasure subset of size <= npar must be recoverable. This is a
 * different code path from minor enumeration (Gauss-Jordan on the
 * unknown block vs closed-form determinants). */
static int all_subsets_recoverable(const uint8_t *M, int ndata, int npar,
                                   int cap)
{
    int N = ndata + npar, checked = 0;
    int sub[8];
    for (int k = 1; k <= npar; k++) {
        for (int i = 0; i < k; i++)
            sub[i] = i;
        for (;;) {
            int S[256], ns = 0;
            for (int d = 0, fi = 0; d < N; d++)
                if (fi < k && sub[fi] == d)
                    fi++;
                else
                    S[ns++] = d;
            uint8_t R[8 * 256];
            if (solve_recovery(M, ndata, npar, sub, k, S, ns, R) != 0)
                return 0;
            if (++checked >= cap)
                return 1;
            int i = k - 1;
            while (i >= 0 && sub[i] == N - k + i)
                i--;
            if (i < 0)
                break;
            sub[i]++;
            for (int j = i + 1; j < k; j++)
                sub[j] = sub[j - 1] + 1;
        }
    }
    return 1;
}

/* Number of singular npar x npar minors via the independent solver:
 * an erasure set of size npar fails recovery iff the corresponding
 * minor of H is singular. Used to prove the single-singular-minor
 * corpus below has EXACTLY ONE singular minor (not merely "at least
 * one"): the construction argument alone is not trusted. */
static int count_singular_minors(const uint8_t *M, int ndata, int npar)
{
    int N = ndata + npar, bad = 0;
    int sub[RS_MDS_MAX_NPAR];
    for (int i = 0; i < npar; i++)
        sub[i] = i;
    for (;;) {
        int S[256], ns = 0;
        for (int d = 0, fi = 0; d < N; d++)
            if (fi < npar && sub[fi] == d)
                fi++;
            else
                S[ns++] = d;
        uint8_t R[8 * 256];
        if (solve_recovery(M, ndata, npar, sub, npar, S, ns, R) != 0)
            bad++;
        int i = npar - 1;
        while (i >= 0 && sub[i] == N - npar + i)
            i--;
        if (i < 0)
            break;
        sub[i]++;
        for (int j = i + 1; j < npar; j++)
            sub[j] = sub[j - 1] + 1;
    }
    return bad;
}

static void test_mds_proof(void)
{
    TEST_BEGIN;
    /* (ndata, npar, solver_subset_cap): full proof everywhere; the
     * independent solver check is exhaustive on small shapes and
     * sampled on the envelope corners. */
    static const struct {
        int nd, np, cap;
    } shapes[] = { { 3, 2, 100000 }, { 8, 2, 100000 }, { 51, 2, 3000 },
                   { 4, 3, 100000 }, { 10, 3, 100000 }, { 51, 3, 3000 },
                   { 200, 1, 100000 }, { 255, 1, 100000 } };
    for (unsigned s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++) {
        int nd = shapes[s].nd, np = shapes[s].np;
        uint8_t m[3 * 256];
        CHECK(matrix_search_default(nd, np, m) == 0,
              "search failed for (%d,%d)", nd, np);
        if (failures)
            return;
        long nminors = -1;
        int r = matrix_is_mds(m, nd, np, 0, &nminors);
        CHECK(r == 1, "(%d,%d): not MDS per full enumeration", nd, np);
        CHECK(nminors == binom2(nd + np, np),
              "(%d,%d): examined %ld minors, want %ld", nd, np, nminors,
              binom2(nd + np, np));
        CHECK(all_subsets_recoverable(m, nd, np, shapes[s].cap),
              "(%d,%d): solver disagrees with minor enumeration", nd, np);
        /* the named constructor must accept the searched shape */
        rj_topology *t = rj_topo_rs(nd, np);
        CHECK(t != NULL, "rj_topo_rs(%d,%d) returned NULL", nd, np);
        rj_topo_free(t);
    }
    TEST_OK("mds_proof");
}

/* ---- 2. the MDS check rejects ---- */

static void test_mds_rejects(void)
{
    TEST_BEGIN;
    uint8_t m[3 * 256];
    long nm;

    /* beyond-envelope historical default: rs(52,2) is NOT MDS
     * (fails {0,51}; docs/limitations.md). npar<=3 so the check runs. */
    old_default_matrix(52, 2, m);
    CHECK(matrix_is_mds(m, 52, 2, 0, &nm) == 0,
          "rs(52,2)-style default accepted as MDS");

    /* duplicated data columns -> singular 2x2 minor */
    old_default_matrix(8, 2, m);
    for (int j = 0; j < 2; j++)
        m[j * 8 + 3] = m[j * 8 + 5];
    CHECK(matrix_is_mds(m, 8, 2, 0, &nm) == 0,
          "duplicated-column matrix accepted as MDS");

    /* zero entry -> singular 1x1 minor (via a parity column) */
    old_default_matrix(8, 3, m);
    m[1 * 8 + 2] = 0;
    CHECK(matrix_is_mds(m, 8, 3, 0, &nm) == 0,
          "zero-entry matrix accepted as MDS");

    /* all-zero matrix */
    memset(m, 0, sizeof(m));
    CHECK(matrix_is_mds(m, 8, 2, 0, &nm) == 0,
          "all-zero matrix accepted as MDS");

    /* bad args */
    CHECK(matrix_is_mds(m, 8, 4, 0, &nm) == -1, "npar=4 not rejected");
    CHECK(matrix_is_mds(NULL, 8, 2, 0, &nm) == -1, "NULL not rejected");
    CHECK(matrix_is_mds(m, 0, 2, 0, &nm) == -1, "ndata=0 not rejected");

    /* a corrupted searched matrix must fail: flip one entry of a
     * searched matrix to a value that provably kills a minor */
    CHECK(matrix_search_default(10, 3, m) == 0, "search (10,3) failed");
    m[2 * 10 + 4] = 0; /* zero entry */
    CHECK(matrix_is_mds(m, 10, 3, 0, &nm) == 0,
          "corrupted searched matrix accepted as MDS");

    /* single-singular-minor corpus: enumeration-completeness tripwire.
     * Each matrix below has EXACTLY ONE singular minor (verified by
     * the independent solver above, not just by construction). A
     * count-preserving skipped-minor mutation in matrix_is_mds -- one
     * that still reports the full minor count but skips the
     * nonsingularity test at some enumeration index -- evades every
     * other bad matrix here (they all have singular minors at several
     * positions, so skipping one index still rejects them) and is
     * caught only by the corpus member whose singular minor sits at
     * the skipped index:
     *  - matrix A: column 0 = 2 * column 1 of the (8,2) default. The
     *    only dependent column pair is {0,1}: enumeration index 1 of
     *    45 (catches a skip-first mutation).
     *  - matrix B: (8,2) default with P[0][7] = 0. The only singular
     *    minor pairs data column 7 with the second identity column:
     *    {7,9}, enumeration index 44 of 45 (catches a skip-near-last
     *    mutation; {8,9} itself is the identity block, always
     *    nonsingular). */
    old_default_matrix(8, 2, m);
    for (int j = 0; j < 2; j++)
        m[j * 8 + 0] = gf_mul(2, m[j * 8 + 1]);
    CHECK(matrix_is_mds(m, 8, 2, 1, &nm) == 0,
          "single-singular-minor matrix A accepted as MDS");
    CHECK(nm == 1, "matrix A: first singular minor at %ld, want 1", nm);
    CHECK(count_singular_minors(m, 8, 2) == 1,
          "matrix A: solver finds != 1 singular minor");

    old_default_matrix(8, 2, m);
    m[0 * 8 + 7] = 0;
    CHECK(matrix_is_mds(m, 8, 2, 1, &nm) == 0,
          "single-singular-minor matrix B accepted as MDS");
    CHECK(nm == 44, "matrix B: first singular minor at %ld, want 44",
          nm);
    CHECK(count_singular_minors(m, 8, 2) == 1,
          "matrix B: solver finds != 1 singular minor");
    TEST_OK("mds_rejects");
}

/* ---- 3. searched cost <= old default cost ---- */

static void test_cost_le_old(void)
{
    TEST_BEGIN;
    static const struct {
        int nd, np;
    } shapes[] = { { 2, 1 }, { 255, 1 }, { 3, 2 }, { 8, 2 },   { 51, 2 },
                   { 4, 3 }, { 10, 3 },  { 51, 3 } };
    printf("%-10s %12s %12s %12s %12s\n", "shape", "old_uops/B",
           "new_uops/B", "old_mem/B", "new_mem/B");
    for (unsigned s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++) {
        int nd = shapes[s].nd, np = shapes[s].np;
        uint8_t om[3 * 256], nm[3 * 256];
        old_default_matrix(nd, np, om);
        CHECK(matrix_search_default(nd, np, nm) == 0,
              "search failed for (%d,%d)", nd, np);
        double ou, omm, nu, nmm;
        CHECK(matrix_search_cost(om, nd, np, &ou, &omm) == 0,
              "old cost failed (%d,%d)", nd, np);
        CHECK(matrix_search_cost(nm, nd, np, &nu, &nmm) == 0,
              "new cost failed (%d,%d)", nd, np);
        printf("(%3d,%d)    %12.4f %12.4f %12.4f %12.4f\n", nd, np, ou, nu,
               omm, nmm);
        CHECK(nu <= ou + 1e-12,
              "(%d,%d): searched cost %.4f > old %.4f", nd, np, nu, ou);
    }
    /* the cost function distinguishes matrices: Cauchy (MULACC) must be
     * strictly pricier than the default (Horner) -- the cost is not
     * a constant. */
    {
        uint8_t cm[2 * 16], dm[2 * 16];
        cauchy_matrix(16, 2, cm);
        old_default_matrix(16, 2, dm);
        double cu, du, cmm, dmm;
        CHECK(matrix_search_cost(cm, 16, 2, &cu, &cmm) == 0, "cauchy cost");
        CHECK(matrix_search_cost(dm, 16, 2, &du, &dmm) == 0, "default cost");
        CHECK(cu > du + 1e-9,
              "cost does not distinguish Cauchy (%.4f) from default (%.4f)",
              cu, du);
    }
    TEST_OK("cost_le_old");
}

/* ---- 4. cost model vs real codegen (x86-64) ---- */

#if defined(__x86_64__)
static void crosscheck_one(const uint8_t *P, int ndata, int npar,
                           const char *mname)
{
    char err[256];
    topo_int_t ti;
    ti.ndata = ndata;
    ti.npar = npar;
    ti.mat = malloc((size_t)npar * (size_t)ndata);
    CHECK(ti.mat != NULL, "oom");
    if (!ti.mat)
        return;
    memcpy(ti.mat, P, (size_t)npar * (size_t)ndata);
    rj_op op = rj_op_encode();
    plan_t pl;
    CHECK(plan_build(&ti, &op, mname, &pl, err, sizeof(err)) == 0,
          "plan_build %s: %s", mname, err);
    if (failures) {
        topo_int_free(&ti);
        return;
    }
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    cand_t cands[128];
    int nc = plan_candidates(&pl, &cpu, cands, 128, 0, 0);
    /* first plain AVX-512 nibble candidate (vl=64, VEX accounting) */
    cand_t *cp = NULL;
    for (int i = 0; i < nc; i++)
        if (cands[i].isa == ISA_AVX512 && cands[i].mul == MUL_NIBBLE &&
            !cands[i].nt && !cands[i].prefetch_dist && !cands[i].ternlog &&
            !cands[i].remat) {
            cp = &cands[i];
            break;
        }
    CHECK(cp != NULL, "%s: no plain avx512 nibble candidate", mname);
    if (!cp) {
        plan_free(&pl);
        rj_op_free(&op);
        topo_int_free(&ti);
        return;
    }
    rj_options opts;
    rj_options_default(&opts);
    version_t v;
    memset(&v, 0, sizeof(v));
    CHECK(jit_build_version(&pl, cp, &cpu, &opts, &v, err, sizeof(err)) == 0,
          "jit_build_version %s: %s", mname, err);
    if (!failures) {
        double uu, mu, ug, mg;
        CHECK(matrix_search_cost_parts(P, ndata, npar, &uu, &mu, &ug,
                                      &mg) == 0,
              "cost_parts %s", mname);
        double grp = (double)cp->U * isa_vl(cp->isa);
        double eu = (cp->U * uu + ug) / grp;
        double em = (cp->U * mu + mg) / grp;
        CHECK(fabs(v.uops_per_byte - eu) < 1e-9,
              "%s: uops/byte model=%.12f codegen=%.12f (U=%d)", mname, eu,
              v.uops_per_byte, cp->U);
        CHECK(fabs(v.mem_per_byte - em) < 1e-9,
              "%s: mem/byte model=%.12f codegen=%.12f (U=%d)", mname, em,
              v.mem_per_byte, cp->U);
    }
    jit_free_version(&v);
    plan_free(&pl);
    rj_op_free(&op);
    topo_int_free(&ti);
}

static void test_cost_model_crosscheck(void)
{
    TEST_BEGIN;
    /* corpus: Horner (old default), MULACC (Cauchy), XOR (npar=1),
     * and a non-Vandermonde MDS matrix; ndata both <=5 (all pointers
     * hoisted) and >5 (reload terms), exercising every branch of the
     * transcription. The ndata=1 shapes pin the ptr_hoist_count corner:
     * with a single input nothing is hoisted, so in[0]'s pointer is
     * reloaded every group iteration (1 uop + 8 mem bytes). */
    uint8_t m[3 * 256];
    old_default_matrix(8, 2, m);
    crosscheck_one(m, 8, 2, "default(8,2) horner");
    old_default_matrix(16, 2, m);
    crosscheck_one(m, 16, 2, "default(16,2) horner reloads");
    old_default_matrix(8, 3, m);
    crosscheck_one(m, 8, 3, "default(8,3) horner");
    cauchy_matrix(8, 2, m);
    crosscheck_one(m, 8, 2, "cauchy(8,2) mulacc");
    cauchy_matrix(12, 3, m);
    crosscheck_one(m, 12, 3, "cauchy(12,3) mulacc reloads");
    old_default_matrix(8, 1, m);
    crosscheck_one(m, 8, 1, "default(8,1) xor");
    old_default_matrix(1, 1, m);
    crosscheck_one(m, 1, 1, "default(1,1) xor hoist-corner");
    old_default_matrix(1, 2, m);
    crosscheck_one(m, 1, 2, "default(1,2) hoist-corner");
    /* hand-built sparse-but-MDS matrix: exercises first_col != 0 rows?
     * (MDS forbids zero columns; use a permuted Cauchy instead) */
    cauchy_matrix(6, 2, m);
    /* permute columns deterministically: still MDS, different bytes */
    {
        uint8_t t[2 * 6];
        for (int j = 0; j < 2; j++)
            for (int i = 0; i < 6; i++)
                t[j * 6 + i] = m[j * 6 + (i * 5 + 1) % 6];
        crosscheck_one(t, 6, 2, "cauchy-permuted(6,2) mulacc");
    }
    TEST_OK("cost_model_crosscheck");
}
#else
static void test_cost_model_crosscheck(void)
{
    TEST_BEGIN;
    printf("skip cost_model_crosscheck (needs x86-64 backend)\n");
    TEST_OK("cost_model_crosscheck (skipped)");
}
#endif

/* ---- 5. determinism / cache ---- */

static void test_search_deterministic(void)
{
    TEST_BEGIN;
    uint8_t a[3 * 256], b[3 * 256];
    static const struct {
        int nd, np;
    } shapes[] = { { 8, 2 }, { 51, 3 }, { 200, 1 } };
    for (unsigned s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++) {
        int nd = shapes[s].nd, np = shapes[s].np;
        CHECK(matrix_search_default(nd, np, a) == 0, "search a (%d,%d)",
              nd, np);
        CHECK(matrix_search_default(nd, np, b) == 0, "search b (%d,%d)",
              nd, np);
        CHECK(memcmp(a, b, (size_t)np * (size_t)nd) == 0,
              "search not deterministic for (%d,%d)", nd, np);
    }
    /* stability vs the historical default: the search's fixed point is
     * the old bytes (no gratuitous parity-format churn). If the search
     * ever finds something strictly better, this CHECK is the tripwire
     * that forces the format change to be deliberate and documented. */
    {
        uint8_t o[3 * 256];
        old_default_matrix(51, 3, o);
        CHECK(matrix_search_default(51, 3, a) == 0, "search (51,3)");
        CHECK(memcmp(a, o, 3 * 51) == 0,
              "search output differs from historical default: "
              "parity format change must be deliberate");
    }
    /* dense envelope tripwire (T2): byte-identity on EVERY in-envelope
     * shape -- npar=2,3 x ndata=1..51, npar=1 sampled. This is the
     * empirical guard that carries the fixed-point conclusion in
     * docs/matrix_search.md: the analytical note bounds the one-count
     * and explains the mechanism, but the claim "the search emits the
     * historical default" is proven here, by exhaustion, not by the
     * analysis. ~20 s on this VM; the cache makes each shape pay the
     * search exactly once. */
    for (int np = 2; np <= 3; np++)
        for (int nd = 1; nd <= 51; nd++) {
            uint8_t o[3 * 256], s[3 * 256];
            old_default_matrix(nd, np, o);
            CHECK(matrix_search_default(nd, np, s) == 0,
                  "search failed for (%d,%d)", nd, np);
            if (failures)
                return;
            CHECK(memcmp(s, o, (size_t)np * (size_t)nd) == 0,
                  "search output != historical default for (%d,%d): "
                  "parity format change must be deliberate and documented",
                  nd, np);
        }
    { /* npar=1: the search short-circuits to the all-ones row, which is
       * exactly the historical default -- sample the ndata range. */
        static const int nds[] = { 1, 2, 3, 7, 51, 128, 200, 255 };
        for (unsigned k = 0; k < sizeof(nds) / sizeof(nds[0]); k++) {
            int nd = nds[k];
            uint8_t o[256], s[256];
            old_default_matrix(nd, 1, o);
            CHECK(matrix_search_default(nd, 1, s) == 0,
                  "search failed for (%d,1)", nd);
            if (failures)
                return;
            CHECK(memcmp(s, o, (size_t)nd) == 0,
                  "search output != historical default for (%d,1)", nd);
        }
    }
    TEST_OK("search_deterministic");
}

/* ---- 6. end-to-end combination test ---- */

static uint64_t t_rng = 0x123456789abcdefULL;
static uint8_t t_rbyte(void)
{
    t_rng ^= t_rng << 13;
    t_rng ^= t_rng >> 7;
    t_rng ^= t_rng << 17;
    return (uint8_t)(t_rng >> 56);
}

static void *t_abuf(size_t n)
{
    void *p = NULL;
    if (posix_memalign(&p, 64, n ? n : 64) != 0)
        return NULL;
    return p;
}

/* independent scalar encode from the topology's matrix bytes */
static void scalar_encode(const uint8_t *M, int nd, int np, uint8_t **in,
                          uint8_t **out, size_t len)
{
    for (int j = 0; j < np; j++)
        for (size_t k = 0; k < len; k++) {
            uint8_t acc = 0;
            for (int i = 0; i < nd; i++)
                acc ^= gf_mul(M[(size_t)j * (size_t)nd + (size_t)i],
                              in[i][k]);
            out[j][k] = acc;
        }
}

static void e2e_encode(rj_topology *t, const char *tname, size_t len,
                       int bench)
{
    char err[256];
    topo_int_t ti;
    CHECK(topo_int_build(&ti, t, err, sizeof(err)) == 0, "%s topo: %s",
          tname, err);
    if (failures)
        return;
    int nd = ti.ndata, np = ti.npar;
    uint8_t **in = malloc(sizeof(*in) * (size_t)nd);
    uint8_t **out = malloc(sizeof(*out) * (size_t)np);
    uint8_t **ref = malloc(sizeof(*ref) * (size_t)np);
    CHECK(in && out && ref, "oom");
    if (!in || !out || !ref) {
        topo_int_free(&ti);
        return;
    }
    for (int i = 0; i < nd; i++) {
        in[i] = t_abuf(len);
        for (size_t k = 0; k < len; k++)
            in[i][k] = t_rbyte();
    }
    for (int j = 0; j < np; j++) {
        out[j] = t_abuf(len);
        ref[j] = t_abuf(len);
        memset(out[j], 0xA5, len);
    }
    scalar_encode(ti.mat, nd, np, in, ref, len);
    rj_op op = rj_op_encode();
    rj_options o;
    rj_options_default(&o);
    o.benchmark = bench;
    rj_jit *j = rj_compile(t, &op, &o, err, sizeof(err));
    CHECK(j != NULL, "%s encode len=%zu bench=%d compile: %s", tname, len,
          bench, err);
    if (j) {
        rj_get_fn(j)((const void **)in, (void **)out, len);
        for (int jj = 0; jj < np && !failures; jj++)
            CHECK(memcmp(out[jj], ref[jj], len) == 0,
                  "%s encode len=%zu par%d mismatch (bench=%d)", tname,
                  len, jj, bench);
        rj_free(j);
    }
    rj_op_free(&op);
    for (int i = 0; i < nd; i++)
        free(in[i]);
    for (int j = 0; j < np; j++) {
        free(out[j]);
        free(ref[j]);
    }
    free(in);
    free(out);
    free(ref);
    topo_int_free(&ti);
}

static void e2e_recover(rj_topology *t, const char *tname, const int *failed,
                        int nfailed, size_t len)
{
    char err[256];
    topo_int_t ti;
    CHECK(topo_int_build(&ti, t, err, sizeof(err)) == 0, "%s topo: %s",
          tname, err);
    if (failures)
        return;
    int nd = ti.ndata, np = ti.npar, N = nd + np;
    int isf[RJ_MAX_DISKS] = { 0 };
    for (int i = 0; i < nfailed; i++)
        isf[failed[i]] = 1;
    /* all disks' original contents */
    uint8_t **disk = malloc(sizeof(*disk) * (size_t)N);
    uint8_t **par = malloc(sizeof(*par) * (size_t)np);
    CHECK(disk && par, "oom");
    if (!disk || !par) {
        topo_int_free(&ti);
        return;
    }
    for (int i = 0; i < nd; i++) {
        disk[i] = t_abuf(len);
        for (size_t k = 0; k < len; k++)
            disk[i][k] = t_rbyte();
    }
    for (int j = 0; j < np; j++)
        disk[nd + j] = t_abuf(len);
    for (int j = 0; j < np; j++)
        par[j] = disk[nd + j];
    scalar_encode(ti.mat, nd, np, disk, par, len);
    /* survivors in increasing order */
    int ns = N - nfailed;
    const void **sin = malloc(sizeof(*sin) * (size_t)ns);
    void **sout = malloc(sizeof(*sout) * (size_t)nfailed);
    CHECK(sin && sout, "oom");
    if (sin && sout) {
        int si = 0, fi = 0;
        /* map failed-disk order -> output slot (increasing failed) */
        int ford[RJ_MAX_DISKS];
        for (int d = 0; d < N; d++) {
            if (isf[d])
                ford[fi++] = d;
            else
                sin[si++] = disk[d];
        }
        for (int i = 0; i < nfailed; i++) {
            sout[i] = t_abuf(len);
            memset(sout[i], 0x5A, len);
        }
        rj_op op = rj_op_recover(failed, nfailed);
        rj_options o;
        rj_options_default(&o);
        o.benchmark = 0;
        rj_jit *j = rj_compile(t, &op, &o, err, sizeof(err));
        CHECK(j != NULL, "%s recover len=%zu compile: %s", tname, len,
              err);
        if (j) {
            rj_get_fn(j)(sin, sout, len);
            for (int i = 0; i < nfailed && !failures; i++)
                CHECK(memcmp(sout[i], disk[ford[i]], len) == 0,
                      "%s recover len=%zu disk%d mismatch", tname, len,
                      ford[i]);
            rj_free(j);
        }
        rj_op_free(&op);
        for (int i = 0; i < nfailed; i++)
            free(sout[i]);
    }
    free(sin);
    free(sout);
    for (int d = 0; d < N; d++)
        free(disk[d]);
    free(disk);
    free(par);
    topo_int_free(&ti);
}

static void test_end_to_end(void)
{
    TEST_BEGIN;
    /* topology x op x length. benchmark=1 on one config to exercise
     * the benchmark path; the rest use benchmark=0 for speed. */
    static const size_t lens[] = { 1, 7, 63, 64, 65, 1000, 4096 };
    struct {
        rj_topology *t;
        const char *name;
    } topos[4];
    topos[0].t = rj_topo_raid6(8);
    topos[0].name = "raid6(8)";
    topos[1].t = rj_topo_rs(6, 3);
    topos[1].name = "rs(6,3)";
    topos[2].t = rj_topo_rs(10, 2);
    topos[2].name = "rs(10,2)";
    topos[3].t = rj_topo_raid5(4);
    topos[3].name = "raid5(4)";
    for (int ti = 0; ti < 4; ti++)
        CHECK(topos[ti].t != NULL, "topo %s NULL", topos[ti].name);
    if (failures)
        return;
    for (int ti = 0; ti < 4 && !failures; ti++)
        for (unsigned li = 0; li < sizeof(lens) / sizeof(lens[0]); li++)
            e2e_encode(topos[ti].t, topos[ti].name, lens[li],
                       (ti == 0 && li == 3) ? 1 : 0);
    /* recover combos: raid6(8): {0},{7},{0,1},{8,9(data+par)}; rs(6,3):
     * {0},{5},{0,1,2},{6,7,8}; rs(10,2): {3},{9},{0,9}; raid5(4): {2} */
    {
        int f[][3] = { { 0, -1, -1 }, { 7, -1, -1 }, { 0, 1, -1 },
                       { 8, 9, -1 } };
        int nf[] = { 1, 1, 2, 2 };
        for (int k = 0; k < 4 && !failures; k++)
            e2e_recover(topos[0].t, topos[0].name, f[k], nf[k], 1000);
    }
    {
        int f[][3] = { { 0, -1, -1 }, { 5, -1, -1 }, { 0, 1, 2 },
                       { 6, 7, 8 } };
        int nf[] = { 1, 1, 3, 3 };
        for (int k = 0; k < 4 && !failures; k++)
            e2e_recover(topos[1].t, topos[1].name, f[k], nf[k], 1000);
    }
    {
        int f[][3] = { { 3, -1, -1 }, { 9, -1, -1 }, { 0, 9, -1 } };
        int nf[] = { 1, 1, 2 };
        for (int k = 0; k < 3 && !failures; k++)
            e2e_recover(topos[2].t, topos[2].name, f[k], nf[k], 500);
    }
    {
        int f[] = { 2 };
        e2e_recover(topos[3].t, topos[3].name, f, 1, 1000);
    }
    for (int ti = 0; ti < 4; ti++)
        rj_topo_free(topos[ti].t);
    TEST_OK("end_to_end");
}

/* ---- 7. envelope ---- */

static void test_envelope(void)
{
    TEST_BEGIN;
    rj_topology *t;
    CHECK(rj_topo_rs(52, 2) == NULL, "rs(52,2) should be NULL");
    CHECK(rj_topo_rs(20, 4) == NULL, "rs(20,4) should be NULL");
    t = rj_topo_rs(51, 2);
    CHECK(t != NULL, "rs(51,2) should work");
    rj_topo_free(t);
    t = rj_topo_rs(51, 3);
    CHECK(t != NULL, "rs(51,3) should work");
    rj_topo_free(t);
    t = rj_topo_rs(255, 1);
    CHECK(t != NULL, "rs(255,1) should work");
    rj_topo_free(t);
    CHECK(rj_topo_raid6(2) == NULL, "raid6(2) should be NULL");
    CHECK(rj_topo_raid6(52) == NULL, "raid6(52) should be NULL");
    TEST_OK("envelope");
}

int main(void)
{
    test_mds_proof();
    test_mds_rejects();
    test_cost_le_old();
    test_cost_model_crosscheck();
    test_search_deterministic();
    test_end_to_end();
    test_envelope();
    if (failures) {
        printf("%d FAILURES\n", failures);
        return 1;
    }
    printf("test_matrix_search: all ok\n");
    return 0;
}
