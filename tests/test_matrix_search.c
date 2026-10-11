/* tests/test_matrix_search.c — tests for the plan-time generator-matrix
 * search (src/matrix_search.c), matrix_search 2.0 (bitmatrix-weight
 * objective).
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
 *  3. test_bitweight_table: the weight table is pinned against
 *     hand-derived constants (wt(0)=0, wt(1)=8, wt(2)=11, wt(3)=19,
 *     wt(4)=14, wt(128)=30); derivations in the comments below -- not
 *     computed by the table code), and 1 is pinned as the UNIQUE
 *     minimizer over nonzero values (the npar==1 short-circuit relies
 *     on it). Bad args to matrix_search_bitweight return -1.
 *  4. test_bitweight_beats_old: on a fixed corpus the searched matrix
 *     has weight <= the historical default's weight everywhere, and
 *     STRICTLY less on every multi-parity shape. (The 1.0 search's
 *     fixed point was byte-identical to the historical default on the
 *     whole envelope -- proven by the 1.0 dense tripwire -- so the
 *     historical default below *is* the old objective's selection.)
 *  5. test_search_deterministic: repeated calls return identical bytes
 *     (exercises the cache path); byte pins for (8,2) and (6,3) --
 *     deliberate-change detectors: any future search change that moves
 *     these bytes fails loudly so the parity-format change is
 *     deliberate and documented. Moderate envelope sweep: the search
 *     succeeds everywhere sampled and never exceeds the old default's
 *     weight.
 *  6. test_end_to_end: ISA x topology x op x length combination test --
 *     rj_compile (public API, searched matrices) vs an independent
 *     scalar reference. Runs natively and under QEMU (aarch64/riscv64).
 *  7. test_envelope: constructors still return NULL outside the envelope.
 *
 * Mutation evidence (done by hand during development, recorded in
 * docs/matrix_search.md):
 *  - E: matrix_search.c bitweight table forced constant (bitw[c] = 8
 *    for all c): the greedy degenerates (no strict improvement
 *    possible), the winner is seed 0 = the historical default, and
 *    test_bitweight_beats_old FAILS on the strict-< checks
 *    ("(8,2): new weight 215 not < old weight 215"). Mutation reverted.
 *  - F: matrix_search.c weight-table column computation perturbed
 *    (column k built from (1u << ((k+1) % 8)) instead of (1u << k)):
 *    test_bitweight_table FAILS on the pinned constants
 *    (wt(2) = 14, want 11). Mutation reverted.
 *  - G: matrix_search.c forbidden_values neutered to "nothing
 *    forbidden": the greedy 1-izes every entry into a non-MDS matrix;
 *    the full-enumeration proof gate catches it and falls back to the
 *    verified seed (mds_proof still passes -- defense in depth holds),
 *    but the suite FAILS via bitweight_beats_old (no strict improvement)
 *    and the search_deterministic byte pins. Mutation reverted.
 *  (1.0 mutation evidence A-D, for the superseded uops objective, is
 *  archived in docs/matrix_search.md.)
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
 * M[j][i] = 2^(j*(ndata-1-i)). This was the pre-2.0 generator AND the
 * 1.0 search's proven fixed point (byte-identical on the envelope), so
 * it is used here solely as "the old objective's selection" for the
 * weight comparison; it is not used by the shipped library. */
static void old_default_matrix(int ndata, int npar, uint8_t *m)
{
    for (int j = 0; j < npar; j++)
        for (int i = 0; i < ndata; i++)
            m[(size_t)j * (size_t)ndata + (size_t)i] =
                gf_pow(2, (unsigned)(j * (ndata - 1 - i)));
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

/* ---- 3. the weight table ---- */

/* Hand-derived weight pins (independent of the table code: each column
 * k of the 8x8 matrix of c is the bits of gf_mul(c, 1<<k) over
 * GF(2^8)/0x11B, weight = popcount summed over k):
 *  - c=0: zero matrix -> 0.
 *  - c=1: identity -> 8.
 *  - c=2: cols 0..6 are single bits (2,4,...,128), col 7 = 0x1B
 *    (x^8 = x^4+x^3+x+1, popcount 4) -> 7+4 = 11.
 *  - c=3: cols 0..6 are 3,6,12,24,48,96,192 (popcount 2 each),
 *    col 7 = 3*128 = 384 = 0x180 -> 0x1B^0x80 = 0x9B (popcount 5)
 *    -> 14+5 = 19.
 *  - c=4: cols 0..5 single bits, col 6 = 0x1B (4), col 7 = 2*0x1B =
 *    0x36 (popcount 4) -> 6+4+4 = 14.
 *  - c=128: cols = 0x80,0x1B,0x36,0x6C,0xD8,0xAB,0x4D,0x9A with
 *    popcounts 1,4,4,4,4,5,4,4 -> 30. */
static void test_bitweight_table(void)
{
    TEST_BEGIN;
    uint8_t m[1];
    uint64_t w;
    static const struct {
        uint8_t c;
        uint64_t want;
    } pins[] = { { 0, 0 }, { 1, 8 }, { 2, 11 },
                 { 3, 19 }, { 4, 14 }, { 128, 30 } };
    for (unsigned i = 0; i < sizeof(pins) / sizeof(pins[0]); i++) {
        m[0] = pins[i].c;
        CHECK(matrix_search_bitweight(m, 1, 1, &w) == 0,
              "bitweight failed for c=%u", pins[i].c);
        CHECK(w == pins[i].want, "wt(%u) = %llu, want %llu", pins[i].c,
              (unsigned long long)w, (unsigned long long)pins[i].want);
    }
    /* 1 is the UNIQUE minimizer over nonzero values: every other
     * nonzero c has a nonsingular 8x8 bitmatrix, hence >= 1 one per
     * row (>= 8 total), with equality only for a permutation matrix,
     * i.e. c = 1 (any higher power of x wraps through the reduction
     * polynomial; any other c has a multi-bit column 0). The npar==1
     * all-ones short-circuit in the search relies on this. */
    for (int c = 2; c < 256; c++) {
        m[0] = (uint8_t)c;
        CHECK(matrix_search_bitweight(m, 1, 1, &w) == 0,
              "bitweight failed for c=%d", c);
        if (w <= 8)
            CHECK(0, "wt(%d) = %llu <= 8: 1 is not the unique minimizer",
                  c, (unsigned long long)w);
    }
    /* bad args */
    m[0] = 1;
    CHECK(matrix_search_bitweight(NULL, 1, 1, &w) == -1,
          "NULL matrix not rejected");
    CHECK(matrix_search_bitweight(m, 1, 1, NULL) == -1,
          "NULL out not rejected");
    CHECK(matrix_search_bitweight(m, 0, 1, &w) == -1,
          "ndata=0 not rejected");
    CHECK(matrix_search_bitweight(m, 1, 4, &w) == -1,
          "npar=4 not rejected");
    CHECK(matrix_search_default(0, 2, m) == -1,
          "search ndata=0 not rejected");
    CHECK(matrix_search_default(8, 2, NULL) == -1,
          "search NULL out not rejected");
    TEST_OK("bitweight_table");
}

/* ---- 4. searched weight vs the old objective's selection ---- */

static void test_bitweight_beats_old(void)
{
    TEST_BEGIN;
    /* (ndata, npar, expect_strict): the multi-parity shapes must be
     * STRICTLY lighter than the historical default (the 1.0 objective's
     * proven fixed point); npar=1 ties (all-ones is the unique
     * bitweight optimum too). */
    static const struct {
        int nd, np, strict;
    } shapes[] = { { 2, 1, 0 }, { 255, 1, 0 }, { 3, 2, 1 }, { 8, 2, 1 },
                   { 51, 2, 1 }, { 4, 3, 1 }, { 10, 3, 1 }, { 51, 3, 1 } };
    printf("%-10s %10s %10s\n", "shape", "old_w", "new_w");
    for (unsigned s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++) {
        int nd = shapes[s].nd, np = shapes[s].np;
        uint8_t om[3 * 256], nm[3 * 256];
        old_default_matrix(nd, np, om);
        CHECK(matrix_search_default(nd, np, nm) == 0,
              "search failed for (%d,%d)", nd, np);
        uint64_t ow, nw;
        CHECK(matrix_search_bitweight(om, nd, np, &ow) == 0,
              "old weight failed (%d,%d)", nd, np);
        CHECK(matrix_search_bitweight(nm, nd, np, &nw) == 0,
              "new weight failed (%d,%d)", nd, np);
        printf("(%3d,%d)    %10llu %10llu\n", nd, np,
               (unsigned long long)ow, (unsigned long long)nw);
        CHECK(nw <= ow, "(%d,%d): searched weight %llu > old %llu", nd,
              np, (unsigned long long)nw, (unsigned long long)ow);
        if (shapes[s].strict)
            CHECK(nw < ow,
                  "(%d,%d): new weight %llu not < old weight %llu", nd,
                  np, (unsigned long long)nw, (unsigned long long)ow);
        else
            CHECK(nw == ow, "(%d,%d): npar=1 weight %llu != %llu", nd,
                  np, (unsigned long long)nw, (unsigned long long)ow);
    }
    TEST_OK("bitweight_beats_old");
}

/* ---- 5. determinism / cache / byte pins ---- */

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
    /* Byte pins: deliberate-change detectors. The search is fully
     * deterministic (fixed seeds, fixed candidate order,
     * strict-improvement acceptance), so these bytes are stable; any
     * future search change that moves them fails loudly so the
     * parity-format change is deliberate and documented. */
    {
        static const uint8_t want82[2][8] = {
            { 1, 1, 1, 1, 1, 1, 1, 1 },
            { 141, 203, 140, 70, 8, 4, 2, 1 },
        };
        CHECK(matrix_search_default(8, 2, a) == 0, "search (8,2)");
        CHECK(memcmp(a, want82, sizeof(want82)) == 0,
              "(8,2) bytes changed: parity-format change must be "
              "deliberate and documented");
    }
    {
        static const uint8_t want63[3][6] = {
            { 1, 1, 1, 1, 1, 1 },
            { 141, 203, 8, 4, 2, 1 },
            { 203, 8, 2, 141, 4, 1 },
        };
        CHECK(matrix_search_default(6, 3, a) == 0, "search (6,3)");
        CHECK(memcmp(a, want63, sizeof(want63)) == 0,
              "(6,3) bytes changed: parity-format change must be "
              "deliberate and documented");
    }
    /* Moderate envelope sweep: the search succeeds on every sampled
     * shape and never exceeds the historical default's weight (it
     * starts from seed 0 and accepts only strict improvements). */
    {
        static const int nds[] = { 1, 2, 5, 13, 25, 37, 51 };
        for (int np = 2; np <= 3; np++)
            for (unsigned k = 0; k < sizeof(nds) / sizeof(nds[0]); k++) {
                int nd = nds[k];
                uint8_t o[3 * 256], s[3 * 256];
                old_default_matrix(nd, np, o);
                CHECK(matrix_search_default(nd, np, s) == 0,
                      "search failed for (%d,%d)", nd, np);
                if (failures)
                    return;
                uint64_t ow, nw;
                CHECK(matrix_search_bitweight(o, nd, np, &ow) == 0,
                      "old weight failed (%d,%d)", nd, np);
                CHECK(matrix_search_bitweight(s, nd, np, &nw) == 0,
                      "new weight failed (%d,%d)", nd, np);
                CHECK(nw <= ow,
                      "(%d,%d): searched weight %llu > old %llu", nd, np,
                      (unsigned long long)nw, (unsigned long long)ow);
                CHECK(matrix_is_mds(s, nd, np, 1, NULL) == 1,
                      "(%d,%d): searched matrix not MDS", nd, np);
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
    test_bitweight_table();
    test_bitweight_beats_old();
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
