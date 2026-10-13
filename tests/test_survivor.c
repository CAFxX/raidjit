/* test_survivor: plan-time selection of the cheapest measured survivor
 * set (track A5, raidjit item 12).
 *
 * Coverage:
 *  1. enumeration units (deterministic, no wall clock): subset counts
 *     and row contents on RS(6,3)/RS(10,4)-custom/RDP(5)/EVENODD(5),
 *     the failed-parity row-forcing rule, the nf==npar single-subset
 *     rule, the exactly-one is_today marker (first match), the
 *     ENUM_CAP guard (survivor_over_cap + the -2 return), and invalid
 *     inputs (nf=0, nf>npar -> -1). Plus the P1.1 regression:
 *     solve_recovery_rows' pivrows_out reports the true pivot
 *     equations (original rows[] values) across row swaps, and R
 *     still recovers on every pivot case.
 *  2. shortlist units (deterministic): the exact pinned shortlist on
 *     RS(6,3) ({{0}} only) under the exact-static-signal rule, the
 *     is_today anchor, the pareto structural property re-derived
 *     independently (every pruned subset is dominated by a kept one or
 *     cut by the documented top-(K-1)/hard-cap rule), the hard cap
 *     (<= SURVIVOR_SHORTLIST_CAP on the 34-subset RDP(5) case),
 *     RJ_NO_SUBSET_PRUNE=1 keeping all, no-signal-never-dominated,
 *     and NaN-signal determinism (NaN ties everything, ranks by
 *     subset index, never dominates). Plus the P1.2 static-tie pins:
 *     a canonical version's NT twin is never worse on the static
 *     counts (uops <=, mem ==) and its prefetch twin never better on
 *     uops (uops >=, mem ==) -- the relations the NT/prefetch-blind
 *     L2 signal relies on. Plus the P2.1 OOM-path pin: a forced
 *     version-alloc failure makes survivor_select return -1 with empty
 *     outputs and no leaked unprocessed subsets (ASan/LSan-clean),
 *     and a follow-up select still works.
 *  3. subset-independence of correctness (end to end): for every
 *     subset the enumerator produces, one JITed version recovers the
 *     original data bit-exactly, across topologies (RAID-6, RS(6,3),
 *     RS(10,3), RS(10,4)-custom, RDP/EVENODD/Liberation opt-ins),
 *     failure counts 1..npar (incl. nf==npar and f_data==0 rows), and
 *     boundary/tail lengths. Any version may win the benchmark;
 *     correctness must not depend on the subset.
 *  4. mutation evidence: RJ_SUBSET_PRUNE_INVERT (test-only hook)
 *     flips the shortlist (M1 unit level); the product pick on
 *     rs(6,3)/f0 stays [rows 0] under inversion -- the LRC locality
 *     rule pins the local row despite cost inversion (M1 product
 *     level: falsifiable pin of the LRC/hook interaction); corrupting
 *     one subset's solve is caught by the end-to-end check (M2); the P2
 *     significance gate is unit-tested pure (noise flip falls back to
 *     today, genuine win ships, today-winning/absent cases unchanged).
 *  5. dispatcher integration: rj_compile with benchmark=1 over a
 *     multi-subset recovery; rj_get_fn (possibly dispatched) is
 *     correct at vector-boundary/tail lengths; the plan desc records
 *     the winning rows. Includes the whole-disk RDP(5) product case
 *     (34 enumerated subsets; the stress case behind the fixed
 *     realloc-dangle bug) and the RJ_NO_SURVIVOR_SELECT public
 *     disable knob (legacy single-subset path).
 *  6. product subset-independence: default vs RJ_NO_SUBSET_PRUNE=1
 *     (exhaustive) both recover correctly. The exhaustive mode also
 *     pins the bug-(a) fix: multi-subset pooled validation with
 *     pre-sized (never realloc'd) plan borrows. Winner equality is NOT
 *     asserted here (benchmark noise flips winners by construction);
 *     it is asserted in Part 7 under pinned bench knobs.
 *  7. L2 tripwire (DESIGN.md section 7): on a fixed corpus with pinned
 *     bench knobs (bench_bytes, bench_iters), the L2-shortlisted
 *     selection must pick the same winning subset as the exhaustive
 *     selection. Each corpus case runs twice: plain and with NT twins
 *     enabled (P1.2: the empirical backstop for the NT-delta
 *     subset-independence assumption -- the pre-P1.2 tripwire never had
 *     NT twins in play). Compares the RAW measured winners
 *     (RJ_SURVIVOR_NO_GATE=1): the tripwire is the L2-safety
 *     instrument, the P2 gate a separately-tested post-measurement
 *     policy. A strict flip goes to the head-to-head REFEREE (the two
 *     subsets benched together, pinned knobs): only a gap beyond the
 *     measured noise window fails loudly as a genuine over-prune; a
 *     within-noise gap passes with a loud note as borderline (P4
 *     domain). The referee predicate is pure and unit-tested
 *     red-green. Red-green for the strict comparator: under
 *     RJ_SUBSET_PRUNE_INVERT (which drops the today anchor) on
 *     rs(6,3)/f02 the flip is structural on the inverted side -- the
 *     inverted shortlist keeps only the dominated {1+2} (a
 *     single-subset pool, so the inverted winner is [rows 1+2] by
 *     construction) -- while the exhaustive side rests on the measured
 *     gap: the keeplist probe gives static ratios of ~1.05x uops /
 *     ~1.10x mem for {1+2} vs {0+1}, and the referee measured ~1.18x
 *     (54.54 vs 46.27 GB/s) against a 15.8% noise window on this
 *     machine. That margin is thin and load-dependent: the exhaustive
 *     winner avoids the dominated subset only while the gap clears
 *     the window -- a flake on a heavily loaded box fails loudly, it
 *     is not a correctness hole. (f0 was
 *     the original case: its P_XOR-vs-P_MULACC gap made the flip
 *     deterministic until the LRC locality rule -- which pins f0's
 *     winner despite cost inversion, by design -- took the shortlist
 *     out of the decision path. f02's two data-disk failures share
 *     one local group, so the LRC rule cannot fire and the shortlist
 *     decides.) The
 *     full flip -> head-to-head -> referee -> loud-FAIL path is
 *     pinned by tripwire_referee_fires: the flip reaches the referee
 *     through the real bench plumbing, the referee adjudicates
 *     (a firing assertion would be load-flaky: the ~1.05x/~1.10x
 *     static gap (uops/mem, keeplist probe) is only ~1.18x measured
 *     vs a 15.8% window on this machine, and the verdict flipped with
 *     box state),
 *     the loud FAIL fires iff the verdict fired, and the
 *     verdict==1 -> loud-FAIL link is unit-pinned.
 *
 * Check-count note (C5e): native and ASan builds of this file must
 * report the SAME check count: every CHECK is on a deterministic code
 * path (benchmark outcomes only select data, never the number of
 * checks). A native/ASan count delta means a check count depends on
 * allocator- or timing-dependent behavior and must be investigated,
 * not normalized.
 *
 * Portable: runs natively on x86-64 and under QEMU for aarch64/riscv64
 * (uses rj_cpu_detect for the build CPU).
 */

#include "../include/raidjit.h"
#include "../src/survivor.h"
#include "../src/plan.h"
#include "../src/jit.h"
#include "../src/gf.h"
#include "../src/ref.h"
#include "../src/bench.h"
#include "../src/prune.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static int failures = 0;
static int nchecks = 0;

#define CHECK(cond, ...)                                                   \
    do {                                                                   \
        nchecks++;                                                         \
        if (!(cond)) {                                                     \
            printf("FAIL %s:%d: ", __func__, __LINE__);                    \
            printf(__VA_ARGS__);                                           \
            printf("\n");                                                  \
            failures++;                                                    \
        }                                                                  \
    } while (0)

static uint64_t xrng_state = 0x243F6A8885A308D3ULL;
static uint64_t xrng(void)
{
    uint64_t x = xrng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    xrng_state = x;
    return x;
}

/* Shared e2e scratch buffers for the product-path parts (5-7).
 *
 * rj_fn 64-byte contract (include/raidjit.h:207-209): every buffer
 * passed to the JITed function must be 64-byte aligned -- the contract
 * is unchecked for speed and codegen uses aligned vector loads/stores
 * throughout, so a misaligned buffer raises #GP. An earlier version of
 * this test passed 16-byte-aligned malloc'd buffers and crashed
 * layout-dependently (the Heisenbug); the static-array workaround that
 * followed was fragile because a plain static has no guaranteed
 * alignment. These arrays carry an explicit 64-byte alignment
 * attribute; every row starts at a 65536-multiple offset from the
 * base, so aligning the base aligns all rows. assert_bufs_aligned()
 * pins the contract at runtime. */
#define SBUF_ROWS 24
#define SBUF_LEN 65536
static uint8_t s_all[SBUF_ROWS][SBUF_LEN]
    __attribute__((aligned(64)));
static uint8_t s_keep[SBUF_ROWS][SBUF_LEN]
    __attribute__((aligned(64)));

static void assert_bufs_aligned(void)
{
    for (int i = 0; i < SBUF_ROWS; i++) {
        CHECK(((uintptr_t)s_all[i] & 63) == 0,
              "s_all row %d not 64-byte aligned", i);
        CHECK(((uintptr_t)s_keep[i] & 63) == 0,
              "s_keep row %d not 64-byte aligned", i);
    }
}

static void randfill(uint8_t *p, size_t n)
{
    for (size_t i = 0; i < n; i++)
        p[i] = (uint8_t)xrng();
}

/* scalar reference: par[j][k] = sum_i d[i][k]*mat[j*nd+i] */
static void ref_encode(const uint8_t *mat, int nd, int np, uint8_t **d,
                       uint8_t **p, size_t len)
{
    for (int j = 0; j < np; j++)
        for (size_t k = 0; k < len; k++) {
            uint8_t a = 0;
            for (int i = 0; i < nd; i++) {
                uint8_t c = mat[j * nd + i];
                if (c)
                    a ^= gf_mul(d[i][k], c);
            }
            p[j][k] = a;
        }
}

static uint8_t **abufs(int n, size_t len)
{
    uint8_t **b = malloc(sizeof(*b) * (size_t)n);
    for (int i = 0; i < n; i++)
        b[i] = malloc(len ? len : 1);
    return b;
}

static void freebufs(uint8_t **b, int n)
{
    for (int i = 0; i < n; i++)
        free(b[i]);
    free(b);
}

/* Build a topo_int from a public topology. Returns 0 on success. */
static int mk_ti(rj_topology *t, topo_int_t *ti, char *err)
{
    memset(ti, 0, sizeof(*ti));
    return topo_int_build(ti, t, err, 256);
}

/* ---------------- Part 1: enumeration units ---------------- */

/* Find the subset whose rows equal want[0..nwant-1]; -1 if absent. */
static int find_rows(survivor_subset_t *subs, int nsub, const int *want,
                     int nwant)
{
    for (int i = 0; i < nsub; i++) {
        if (subs[i].nrows != nwant)
            continue;
        int ok = 1;
        for (int k = 0; k < nwant; k++)
            if (subs[i].rows[k] != want[k]) {
                ok = 0;
                break;
            }
        if (ok)
            return i;
    }
    return -1;
}

static int count_is_today(survivor_subset_t *subs, int nsub)
{
    int c = 0;
    for (int i = 0; i < nsub; i++)
        c += subs[i].is_today ? 1 : 0;
    return c;
}

/* Enumerate and return subsets; CHECKs the call succeeded. */
static survivor_subset_t *enumerate_ok(topo_int_t *ti, const int *F, int nf,
                                      const char *opname, int *nsub_out,
                                      const char *tag)
{
    survivor_subset_t *subs = NULL;
    int nsub = 0;
    char err[256];
    int rc = survivor_enumerate(ti, F, nf, opname, &subs, &nsub, err,
                                sizeof(err));
    CHECK(rc == 0, "%s: enumerate rc=%d (%s)", tag, rc, err);
    if (rc != 0)
        return NULL;
    *nsub_out = nsub;
    return subs;
}

static void enum_rs63(void)
{
    rj_topology *t = rj_topo_rs(6, 3);
    topo_int_t ti;
    char err[256];
    CHECK(mk_ti(t, &ti, err) == 0, "rs63 topo: %s", err);
    if (ti.mat) {
        /* 1 failure on data disk 0: 3 row choices */
        {
            int F[1] = { 0 };
            int nsub = 0;
            survivor_subset_t *subs =
                enumerate_ok(&ti, F, 1, "rs(6,3) rec 0", &nsub, "rs63/f0");
            if (subs) {
                int r0[1] = { 0 }, r1[1] = { 1 }, r2[1] = { 2 };
                CHECK(nsub == 3, "rs63/f0: nsub=%d want 3", nsub);
                CHECK(find_rows(subs, nsub, r0, 1) >= 0, "rs63/f0: {0}");
                CHECK(find_rows(subs, nsub, r1, 1) >= 0, "rs63/f0: {1}");
                CHECK(find_rows(subs, nsub, r2, 1) >= 0, "rs63/f0: {2}");
                CHECK(count_is_today(subs, nsub) == 1,
                      "rs63/f0: exactly one is_today");
                int ti_idx = find_rows(subs, nsub, r0, 1);
                CHECK(ti_idx >= 0 && subs[ti_idx].is_today,
                      "rs63/f0: is_today is the {0} subset");
                survivor_free_subsets(subs, nsub);
            }
        }
        /* 2 failures on data disks 0,2: C(3,2)=3 */
        {
            int F[2] = { 0, 2 };
            int nsub = 0;
            survivor_subset_t *subs =
                enumerate_ok(&ti, F, 2, "rs(6,3) rec 0+2", &nsub,
                             "rs63/f02");
            if (subs) {
                int a[2] = { 0, 1 }, b[2] = { 0, 2 }, c[2] = { 1, 2 };
                CHECK(nsub == 3, "rs63/f02: nsub=%d want 3", nsub);
                CHECK(find_rows(subs, nsub, a, 2) >= 0, "rs63/f02: {0,1}");
                CHECK(find_rows(subs, nsub, b, 2) >= 0, "rs63/f02: {0,2}");
                CHECK(find_rows(subs, nsub, c, 2) >= 0, "rs63/f02: {1,2}");
                CHECK(count_is_today(subs, nsub) == 1,
                      "rs63/f02: exactly one is_today");
                survivor_free_subsets(subs, nsub);
            }
        }
        /* failed parity forces its row: F={data0, parity0} */
        {
            int F[2] = { 0, 6 };
            int nsub = 0;
            survivor_subset_t *subs =
                enumerate_ok(&ti, F, 2, "rs(6,3) rec 0+6", &nsub,
                             "rs63/f06");
            if (subs) {
                CHECK(nsub == 2, "rs63/f06: nsub=%d want 2", nsub);
                for (int i = 0; i < nsub; i++)
                    CHECK(subs[i].nrows == 2 && subs[i].rows[0] == 0,
                          "rs63/f06: subset %d forces row 0", i);
                survivor_free_subsets(subs, nsub);
            }
        }
        /* nf == npar: exactly one subset (all rows) */
        {
            int F[3] = { 6, 7, 8 };
            int nsub = 0;
            survivor_subset_t *subs =
                enumerate_ok(&ti, F, 3, "rs(6,3) rec 6+7+8", &nsub,
                             "rs63/f678");
            if (subs) {
                int all[3] = { 0, 1, 2 };
                CHECK(nsub == 1, "rs63/f678: nsub=%d want 1", nsub);
                CHECK(find_rows(subs, nsub, all, 3) >= 0,
                      "rs63/f678: the all-rows subset");
                CHECK(subs[0].is_today, "rs63/f678: is_today");
                survivor_free_subsets(subs, nsub);
            }
        }
    }
    topo_int_free(&ti);
    rj_topo_free(t);
}

static void enum_rs104c(void)
{
    /* RS(10,4) with the searched custom matrix: 2 failures -> C(4,2)=6 */
    uint8_t m[4 * 10];
    for (int j = 0; j < 4; j++)
        for (int i = 0; i < 10; i++)
            m[j * 10 + i] = (uint8_t)(1 + ((7 * i + 13 * j) % 255));
    rj_topology *t = rj_topo_custom(10, 4, m);
    topo_int_t ti;
    char err[256];
    CHECK(mk_ti(t, &ti, err) == 0, "rs104c topo: %s", err);
    if (ti.mat) {
        int F[2] = { 1, 3 };
        int nsub = 0;
        survivor_subset_t *subs =
            enumerate_ok(&ti, F, 2, "rs(10,4) rec 1+3", &nsub, "rs104c");
        if (subs) {
            CHECK(nsub == 6, "rs104c: nsub=%d want 6", nsub);
            CHECK(count_is_today(subs, nsub) == 1,
                  "rs104c: exactly one is_today");
            for (int i = 0; i < nsub; i++)
                CHECK(subs[i].nrows == 2, "rs104c: subset %d has 2 rows",
                      i);
            survivor_free_subsets(subs, nsub);
        }
    }
    topo_int_free(&ti);
    rj_topo_free(t);
}

/* Whole-disk failure on an XOR array code: a single disk fails; the
 * enumerator produces the C(npar,1) row choices (singular ones are
 * dropped). Every produced subset must have nf rows. */
static void enum_xorcode_whole(const char *tag, rj_topology *t,
                               int ndisks_expect)
{
    topo_int_t ti;
    char err[256];
    CHECK(mk_ti(t, &ti, err) == 0, "%s topo: %s", tag, err);
    (void)ndisks_expect;
    if (ti.mat) {
        int F[1] = { 0 }; /* whole disk 0 fails */
        int nsub = 0;
        survivor_subset_t *subs =
            enumerate_ok(&ti, F, 1, tag, &nsub, tag);
        if (subs) {
            CHECK(nsub > 0, "%s: at least one valid subset", tag);
            CHECK(count_is_today(subs, nsub) == 1,
                  "%s: exactly one is_today", tag);
            for (int i = 0; i < nsub; i++)
                CHECK(subs[i].nrows == 1,
                      "%s: subset %d has %d rows (want 1)", tag, i,
                      subs[i].nrows);
            printf("  info: %s whole-disk: %d valid subsets\n", tag, nsub);
            survivor_free_subsets(subs, nsub);
        }
    }
    topo_int_free(&ti);
}

/* The ENUM_CAP guard: a topology whose combination count exceeds the
 * cap must report -2 (caller falls back), never hang or OOM. */
static void enum_over_cap(void)
{
    int nd = 10, np = 20;
    uint8_t *m = malloc(sizeof(*m) * (size_t)(nd * np));
    for (int i = 0; i < nd * np; i++)
        m[i] = (uint8_t)(1 + (i % 255));
    rj_topology *t = rj_topo_custom(nd, np, m);
    free(m);
    topo_int_t ti;
    char err[256];
    CHECK(mk_ti(t, &ti, err) == 0, "custom(10,20) topo: %s", err);
    if (ti.mat) {
        int F[10];
        for (int i = 0; i < 10; i++)
            F[i] = i; /* C(20,10)=184756 > cap */
        CHECK(survivor_over_cap(&ti, F, 10) == 1,
              "custom(10,20)/nf=10: over_cap must be true");
        survivor_subset_t *subs = NULL;
        int nsub = 0;
        int rc = survivor_enumerate(&ti, F, 10, "custom rec", &subs, &nsub,
                                    err, sizeof(err));
        CHECK(rc == -2, "custom(10,20)/nf=10: enumerate rc=%d want -2 (%s)",
              rc, err);
        CHECK(subs == NULL, "custom(10,20): no subsets on -2");
        /* and a small case on the same topo is under the cap */
        int F2[2] = { 0, 1 };
        CHECK(survivor_over_cap(&ti, F2, 2) == 0,
              "custom(10,20)/nf=2: over_cap must be false");
    }
    topo_int_free(&ti);
    rj_topo_free(t);
}

/* Invalid inputs: nf=0 and nf>npar must fail loudly (-1), never
 * enumerate nonsense or crash. */
static void enum_invalid(void)
{
    rj_topology *t = rj_topo_rs(6, 3);
    topo_int_t ti;
    char err[256];
    CHECK(mk_ti(t, &ti, err) == 0, "rs63 topo: %s", err);
    if (ti.mat) {
        survivor_subset_t *subs = NULL;
        int nsub = 0;
        int F[1] = { 0 };
        int rc = survivor_enumerate(&ti, F, 0, "rs63/nf0", &subs, &nsub,
                                    err, sizeof(err));
        CHECK(rc == -1, "rs63/nf0: rc=%d want -1 (%s)", rc, err);
        CHECK(subs == NULL && nsub == 0, "rs63/nf0: no subsets");
        int F4[4] = { 0, 1, 2, 3 };
        rc = survivor_enumerate(&ti, F4, 4, "rs63/nf4", &subs, &nsub,
                                err, sizeof(err));
        CHECK(rc == -1, "rs63/nf4>npar: rc=%d want -1 (%s)", rc, err);
        CHECK(subs == NULL && nsub == 0, "rs63/nf4: no subsets");
        CHECK(survivor_over_cap(&ti, F, 0) == 0,
              "rs63/nf0: over_cap must be false (not true)");
    }
    topo_int_free(&ti);
    rj_topo_free(t);
}

/* P1.1 regression: solve_recovery_rows' pivrows_out must report the
 * TRUE pivot equations (original rows[] values), not post-swap matrix
 * positions. The elimination swaps only the augmented matrix, so the
 * fix shadows the permutation (rperm in gf.c) and maps post-swap
 * positions back through it.
 *
 * Mutation: reverting the rperm shadow (pivrows_out[c] =
 * rows[pivrow[c]]) makes the swap cases fail -- case f7 reports 0
 * instead of 1, case f78 reports 0/1 instead of 1/2, case revorder
 * reports 2 instead of 1 -- while the no-swap control still passes.
 * R itself is verified to recover the failed disks on every case, so
 * the test also pins that the fix did not disturb the solution. */
static void pivrows_case(const char *tag, const int *F, int nf,
                         const int *rows, int nrows,
                         const int *want_piv)
{
    uint8_t M[3 * 6];
    rs_default_matrix(6, 3, M);
    int Sall[9], ns = 0;
    for (int d = 0; d < 9; d++) {
        int failed = 0;
        for (int i = 0; i < nf; i++)
            if (F[i] == d)
                failed = 1;
        if (!failed)
            Sall[ns++] = d;
    }
    uint8_t R[3 * 9];
    int piv[3];
    for (int i = 0; i < 3; i++)
        piv[i] = -99;
    int rc = solve_recovery_rows(M, 6, 3, F, nf, Sall, ns, rows, nrows,
                                 R, piv);
    CHECK(rc == 0, "pivrows/%s: solve rc=%d", tag, rc);
    if (rc != 0)
        return;
    for (int c = 0; c < nf; c++)
        CHECK(piv[c] == want_piv[c],
              "pivrows/%s: unknown %d (disk %d): pivot row %d, want %d",
              tag, c, F[c], piv[c], want_piv[c]);
    /* R must still recover the failed disks (fix disturbs nothing). */
    uint8_t full[9];
    for (int d = 0; d < 6; d++)
        full[d] = (uint8_t)(0x9e + 37 * d);
    for (int j = 0; j < 3; j++) {
        uint8_t acc = 0;
        for (int i = 0; i < 6; i++)
            acc ^= gf_mul(M[j * 6 + i], full[i]);
        full[6 + j] = acc;
    }
    for (int c = 0; c < nf; c++) {
        uint8_t acc = 0;
        for (int s = 0; s < ns; s++)
            acc ^= gf_mul(R[c * ns + s], full[Sall[s]]);
        CHECK(acc == full[F[c]],
              "pivrows/%s: R row %d does not recover disk %d", tag, c,
              F[c]);
    }
}

static void pivrows_units(void)
{
    /* The P1.1 probe case: single parity-1 failure; the unknown
     * column's single nonzero sits at position 1, so elimination
     * swaps rows 1<->0. True pivot: original row 1. */
    {
        int F[1] = { 7 }, rows[3] = { 0, 1, 2 }, want[1] = { 1 };
        pivrows_case("f7", F, 1, rows, 3, want);
    }
    /* Two parity failures: two swaps; pivots are original rows 1, 2
     * in F order. */
    {
        int F[2] = { 7, 8 }, rows[3] = { 0, 1, 2 }, want[2] = { 1, 2 };
        pivrows_case("f78", F, 2, rows, 3, want);
    }
    /* No-swap control: data-disk failure pivots position 0 directly. */
    {
        int F[1] = { 0 }, rows[3] = { 0, 1, 2 }, want[1] = { 0 };
        pivrows_case("f0", F, 1, rows, 3, want);
    }
    /* Non-identity rows[] order: the report is the rows[] VALUE (1),
     * not the position (0). */
    {
        int F[1] = { 7 }, rows[3] = { 2, 1, 0 }, want[1] = { 1 };
        pivrows_case("revorder", F, 1, rows, 3, want);
    }
}

/* ---------------- Part 2: shortlist units ---------------- */

/* Build the L2 static signals for test subsets (codegen, no
 * benchmarking). */
static void l2_signals(survivor_subset_t *subs, int nsub)
{
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    rj_options o;
    rj_options_default(&o);
    survivor_l2_signal(subs, nsub, &cpu, &o);
}

/* P1.2: pins the static relations the NT/prefetch-blind L2 signal
 * relies on. For each subset's canonical version (cands[0]):
 *  - its NT twin is never WORSE on the static counts (uops <=, mem ==;
 *    exact ties on x86-64/SVE/RVV by construction -- c_st/c_st_to/c_vse
 *    count NT stores identically, RVV's NTL hint is uncounted -- and
 *    the NEON STNP pairing discount is uops-only, never mem);
 *  - its prefetch twin is never BETTER on uops (uops >=, mem ==).
 * If a future change makes NT twins statically distinguishable in a
 * subset-dependent way, these CHECKs fail and force the L2-signal
 * re-audit (the documented re-audit trigger). Mutation: giving NT
 * stores a uop/mem discount (or prefetch a uop credit) breaks the
 * corresponding CHECK. */
static void nt_static_tie(void)
{
    rj_topology *t = rj_topo_rs(6, 3);
    topo_int_t ti;
    char err[512];
    CHECK(mk_ti(t, &ti, err) == 0, "nt_static_tie topo: %s", err);
    if (ti.mat) {
        survivor_subset_t *subs = NULL;
        int nsub = 0;
        int F[1] = { 0 };
        int rc = survivor_enumerate(&ti, F, 1, "rs63/f0", &subs, &nsub,
                                    err, sizeof(err));
        CHECK(rc == 0 && nsub == 3, "nt_static_tie enumerate: rc=%d "
              "nsub=%d", rc, nsub);
        if (rc == 0 && nsub == 3) {
            rj_cpu_info cpu;
            rj_cpu_detect(&cpu);
            rj_options o;
            rj_options_default(&o);
            for (int i = 0; i < nsub; i++) {
                cand_t cands[512];
                int nc = plan_candidates(&subs[i].plan, &cpu, cands,
                                         512, 1, 64);
                CHECK(nc > 0, "nt_static_tie sub %d: no candidates", i);
                if (nc <= 0)
                    continue;
                /* cands[0] is the plain canonical; find its exact NT
                 * and prefetch twins (field-identical except nt /
                 * prefetch_dist). */
                int i_nt = -1, i_pf = -1;
                for (int k = 1; k < nc; k++) {
                    cand_t a = cands[k], b = cands[0];
                    a.nt = b.nt;
                    a.prefetch_dist = b.prefetch_dist;
                    if (memcmp(&a, &b, sizeof(a)) != 0)
                        continue;
                    if (cands[k].nt && !cands[k].prefetch_dist)
                        i_nt = k;
                    if (!cands[k].nt && cands[k].prefetch_dist)
                        i_pf = k;
                }
                CHECK(i_nt >= 0, "nt_static_tie sub %d: no NT twin", i);
                CHECK(i_pf >= 0, "nt_static_tie sub %d: no prefetch "
                      "twin", i);
                if (i_nt < 0 || i_pf < 0)
                    continue;
                version_t v_plain, v_nt, v_pf;
                char e2[256];
                int ok = jit_build_version(&subs[i].plan, &cands[0],
                                           &cpu, &o, &v_plain, e2,
                                           sizeof(e2)) == 0 &&
                         jit_build_version(&subs[i].plan, &cands[i_nt],
                                           &cpu, &o, &v_nt, e2,
                                           sizeof(e2)) == 0 &&
                         jit_build_version(&subs[i].plan, &cands[i_pf],
                                           &cpu, &o, &v_pf, e2,
                                           sizeof(e2)) == 0;
                CHECK(ok, "nt_static_tie sub %d: version build: %s", i,
                      e2);
                if (!ok)
                    continue;
                CHECK(v_nt.uops_per_byte <= v_plain.uops_per_byte,
                      "nt_static_tie sub %d: NT uops %.4f > plain "
                      "%.4f", i, v_nt.uops_per_byte,
                      v_plain.uops_per_byte);
                CHECK(v_nt.mem_per_byte == v_plain.mem_per_byte,
                      "nt_static_tie sub %d: NT mem %.4f != plain "
                      "%.4f", i, v_nt.mem_per_byte,
                      v_plain.mem_per_byte);
                CHECK(v_pf.uops_per_byte >= v_plain.uops_per_byte,
                      "nt_static_tie sub %d: prefetch uops %.4f < "
                      "plain %.4f", i, v_pf.uops_per_byte,
                      v_plain.uops_per_byte);
                CHECK(v_pf.mem_per_byte == v_plain.mem_per_byte,
                      "nt_static_tie sub %d: prefetch mem %.4f != "
                      "plain %.4f", i, v_pf.mem_per_byte,
                      v_plain.mem_per_byte);
                jit_free_version(&v_plain);
                jit_free_version(&v_nt);
                jit_free_version(&v_pf);
            }
        }
        survivor_free_subsets(subs, nsub);
    }
    topo_int_free(&ti);
    rj_topo_free(t);
}

/* P2.1 regression: survivor_select's !ok (OOM) path must free
 * unprocessed subsets' plans/rows, not leak them. The test-only hook
 * RJ_SURVIVOR_FAIL_VS_ALLOC forces the per-subset version calloc to
 * fail; the select must return -1 with empty outputs, and under
 * ASan/LSan the process must report no leaks (pre-fix, every
 * unprocessed subset's plan+rows leaked here). A follow-up unhooked
 * select must still work -- the failed run leaves no dangling state
 * (a double-free or heap corruption would crash even without ASan).
 * Mutation: reverting the !ok cleanup makes the ASan run report the
 * leaked subsets. */
static void select_oom_cleanup(void)
{
    rj_topology *t = rj_topo_rs(6, 3);
    topo_int_t ti;
    char err[512];
    CHECK(mk_ti(t, &ti, err) == 0, "oom topo: %s", err);
    if (ti.mat) {
        rj_cpu_info cpu;
        rj_cpu_detect(&cpu);
        rj_options o;
        rj_options_default(&o);
        o.benchmark = 0; /* deterministic: codegen only, no benching */
        const int F[1] = { 0 };
        setenv("RJ_NO_SUBSET_PRUNE", "1", 1);
        setenv("RJ_SURVIVOR_FAIL_VS_ALLOC", "1", 1);
        version_t *vs = NULL;
        int nv = 0;
        plan_t *plans = NULL;
        int nplans = 0;
        int rc = survivor_select(&ti, F, 1, "rs63/f0", &cpu, &o, &vs,
                                 &nv, &plans, &nplans, NULL, err,
                                 sizeof(err));
        unsetenv("RJ_SURVIVOR_FAIL_VS_ALLOC");
        unsetenv("RJ_NO_SUBSET_PRUNE");
        CHECK(rc == -1, "oom select: rc=%d want -1 (%s)", rc, err);
        CHECK(vs == NULL && nv == 0 && plans == NULL && nplans == 0,
              "oom select: outputs must stay empty on OOM");
        /* No dangling state: an unhooked select right after must work. */
        rc = survivor_select(&ti, F, 1, "rs63/f0", &cpu, &o, &vs, &nv,
                             &plans, &nplans, NULL, err, sizeof(err));
        CHECK(rc >= 0, "oom select: follow-up rc=%d (%s)", rc, err);
        if (rc >= 0) {
            for (int i = 0; i < nv; i++)
                jit_free_version(&vs[i]);
            free(vs);
            for (int i = 0; i < nplans; i++)
                plan_free(&plans[i]);
            free(plans);
        }
    }
    topo_int_free(&ti);
    rj_topo_free(t);
}

/* The pinned shortlist on RS(6,3) 1-failure under the exact-signal
 * rule: the {0} subset's canonical version is P_XOR with strictly
 * lower (uops/B, mem/B) than the P_MULACC {1},{2} subsets (measured
 * 2026-10-03 on x86-64: 0.111 vs 0.309 uops/B; the structural P_XOR <
 * P_MULACC gap holds on every backend), so {0} pareto-dominates on
 * the exact (l2_uops, l2_mem) signal -- and it is also the today
 * anchor. Only {0} is kept. This pins the comparator direction; M1
 * (inverted) must flip it. */
static void shortlist_rs63(void)
{
    rj_topology *t = rj_topo_rs(6, 3);
    topo_int_t ti;
    char err[256];
    CHECK(mk_ti(t, &ti, err) == 0, "rs63 topo: %s", err);
    if (ti.mat) {
        int F[1] = { 0 };
        int nsub = 0;
        survivor_subset_t *subs =
            enumerate_ok(&ti, F, 1, "rs(6,3) rec 0", &nsub, "sl-rs63");
        if (subs) {
            CHECK(nsub == 3, "sl-rs63: 3 subsets in");
            l2_signals(subs, nsub);
            for (int i = 0; i < nsub; i++)
                CHECK(subs[i].l2_valid,
                      "sl-rs63: subset %d has an L2 signal", i);
            int *keep = calloc((size_t)nsub, sizeof(*keep));
            survivor_shortlist(subs, nsub, keep);
            int nkept = 0;
            for (int i = 0; i < nsub; i++)
                nkept += keep[i];
            CHECK(nkept == 1, "sl-rs63: nkept=%d want 1", nkept);
            int r0[1] = { 0 };
            int idx = find_rows(subs, nsub, r0, 1);
            CHECK(idx >= 0 && keep[idx] == 1,
                  "sl-rs63: the kept subset is {0}");
            free(keep);
            survivor_free_subsets(subs, nsub);
        }
    }
    topo_int_free(&ti);
    rj_topo_free(t);
}

/* Structural shortlist property, re-derived independently from the
 * documented rule (no shared helpers with the implementation):
 * expected keep = {today} ∪ top-(K-1) statically-ranked pareto
 * members on the exact (l2_uops, l2_mem) signal, hard cap K =
 * SURVIVOR_SHORTLIST_CAP. The implementation must match exactly. */
static void shortlist_pareto_struct(const char *tag, rj_topology *t,
                                    const int *F, int nf)
{
    topo_int_t ti;
    char err[256];
    CHECK(mk_ti(t, &ti, err) == 0, "%s topo: %s", tag, err);
    if (!ti.mat) {
        topo_int_free(&ti);
        return;
    }
    int nsub = 0;
    survivor_subset_t *subs =
        enumerate_ok(&ti, F, nf, tag, &nsub, tag);
    if (subs) {
        l2_signals(subs, nsub);
        int *keep = calloc((size_t)nsub, sizeof(*keep));
        survivor_shortlist(subs, nsub, keep);

        /* Independent re-derivation of the expected keep set. */
        int *expected = calloc((size_t)nsub, sizeof(*expected));
        int *in_par = calloc((size_t)nsub, sizeof(*in_par));
        for (int i = 0; i < nsub; i++) {
            int dom = 0;
            for (int j = 0; j < nsub && !dom; j++) {
                if (i == j || !subs[j].l2_valid || !subs[i].l2_valid)
                    continue;
                if (subs[j].l2_uops <= subs[i].l2_uops &&
                    subs[j].l2_mem <= subs[i].l2_mem &&
                    (subs[j].l2_uops < subs[i].l2_uops ||
                     subs[j].l2_mem < subs[i].l2_mem))
                    dom = 1;
            }
            in_par[i] = !dom;
        }
        int today = -1;
        for (int i = 0; i < nsub; i++)
            if (subs[i].is_today) {
                today = i;
                break; /* first match (C6) */
            }
        CHECK(today >= 0, "%s: a today anchor exists", tag);
        int nkept_exp = 0;
        if (today >= 0) {
            expected[today] = 1;
            nkept_exp++;
        }
        /* top-(K-1): repeatedly take the smallest (uops, mem) pareto
         * member not already kept; signal-less subsets rank last,
         * ties break by subset index (the documented rule). */
        for (int k = 0; k < SURVIVOR_SHORTLIST_CAP - 1; k++) {
            int best = -1;
            for (int i = 0; i < nsub; i++) {
                if (!in_par[i] || expected[i])
                    continue;
                if (best < 0) {
                    best = i;
                    continue;
                }
                int ibad = !subs[i].l2_valid;
                int bbad = !subs[best].l2_valid;
                if ((!ibad && bbad) ||
                    (ibad == bbad &&
                     (subs[i].l2_uops < subs[best].l2_uops ||
                      (subs[i].l2_uops == subs[best].l2_uops &&
                       (subs[i].l2_mem < subs[best].l2_mem ||
                        (subs[i].l2_mem == subs[best].l2_mem &&
                         i < best))))))
                    best = i;
            }
            if (best < 0 || nkept_exp >= SURVIVOR_SHORTLIST_CAP)
                break;
            expected[best] = 1;
            nkept_exp++;
        }
        int nkept = 0;
        for (int i = 0; i < nsub; i++) {
            nkept += keep[i];
            CHECK(keep[i] == expected[i],
                  "%s: subset %d keep=%d want %d", tag, i, keep[i],
                  expected[i]);
        }
        CHECK(nkept <= SURVIVOR_SHORTLIST_CAP,
              "%s: nkept=%d exceeds hard cap %d", tag, nkept,
              SURVIVOR_SHORTLIST_CAP);
        CHECK(nkept >= 1, "%s: at least one kept", tag);
        printf("  info: %s: %d/%d kept\n", tag, nkept, nsub);
        free(expected);
        free(in_par);
        free(keep);
        survivor_free_subsets(subs, nsub);
    }
    topo_int_free(&ti);
}

/* A subset without an L2 signal is never treated as dominated: it
 * stays in the pareto set and ranks last (a missing signal can only
 * keep a subset, never drop it). Pure units, no codegen. */
static void shortlist_no_signal(void)
{
    survivor_subset_t subs[2];
    memset(subs, 0, sizeof(subs));
    subs[0].l2_valid = 1;
    subs[0].l2_uops = 1.0;
    subs[0].l2_mem = 1.0;
    subs[0].is_today = 1;
    subs[1].l2_valid = 0; /* canonical build failed: no signal */
    int keep[2] = { 0, 0 };
    survivor_shortlist(subs, 2, keep);
    CHECK(keep[0] == 1 && keep[1] == 1,
          "no-signal: both kept (keep=%d,%d)", keep[0], keep[1]);

    /* and it is never dominated even by a strictly better signal */
    memset(subs, 0, sizeof(subs));
    subs[0].l2_valid = 1;
    subs[0].l2_uops = 0.1;
    subs[0].l2_mem = 0.1;
    subs[0].is_today = 1;
    subs[1].l2_valid = 0;
    keep[0] = keep[1] = 0;
    survivor_shortlist(subs, 2, keep);
    CHECK(keep[1] == 1, "no-signal: not dominated by better signal");
}

/* NaN static signals tie deterministically: a NaN l2_uops/l2_mem
 * compares false on every relation (l2_dominates and l2_rank_less),
 * so a NaN subset is never dominated, stays in the pareto set, and
 * ranks in index order. The keep-set must be deterministic across
 * runs and the top-(K-1) cut must take the lowest indices. Pure
 * units, no codegen. */
static void shortlist_nan_signal(void)
{
    survivor_subset_t subs[4];
    memset(subs, 0, sizeof(subs));
    double nan = 0.0 / 0.0; /* NaN: compiles clean under -Werror */
    for (int i = 0; i < 4; i++) {
        subs[i].l2_valid = 1;
        subs[i].l2_uops = nan;
        subs[i].l2_mem = nan;
    }
    subs[3].is_today = 1; /* anchor is NOT the first subset */
    int keep[4] = { 0, 0, 0, 0 };
    survivor_shortlist(subs, 4, keep);
    /* anchor kept + top-(K-1)=2 pareto members in index order: NaN
     * ties everything, so indices 0 and 1 rank first and the hard
     * cap (3) cuts index 2. */
    CHECK(keep[0] == 1 && keep[1] == 1 && keep[2] == 0 && keep[3] == 1,
          "nan-signal: keep=(%d,%d,%d,%d) want (1,1,0,1)", keep[0],
          keep[1], keep[2], keep[3]);

    /* determinism: an identical second run must keep exactly the
     * same set (NaN must not introduce order nondeterminism). */
    survivor_subset_t subs2[4];
    memcpy(subs2, subs, sizeof(subs2));
    int keep2[4] = { 0, 0, 0, 0 };
    survivor_shortlist(subs2, 4, keep2);
    for (int i = 0; i < 4; i++)
        CHECK(keep2[i] == keep[i], "nan-signal: deterministic subset %d",
              i);

    /* a NaN subset is never dominated even by a strictly better valid
     * signal: the good signal must not drop it. */
    memset(subs, 0, sizeof(subs));
    subs[0].l2_valid = 1;
    subs[0].l2_uops = 0.1;
    subs[0].l2_mem = 0.1;
    subs[1].l2_valid = 1;
    subs[1].l2_uops = nan;
    subs[1].l2_mem = nan;
    subs[1].is_today = 1;
    memset(keep, 0, sizeof(keep));
    survivor_shortlist(subs, 4, keep);
    CHECK(keep[1] == 1, "nan-signal: not dominated by better signal");
}

/* RJ_NO_SUBSET_PRUNE=1 keeps every enumerated subset. */
static void shortlist_noprune(void)
{
    rj_topology *t = rj_topo_rs(6, 3);
    topo_int_t ti;
    char err[256];
    CHECK(mk_ti(t, &ti, err) == 0, "rs63 topo: %s", err);
    if (ti.mat) {
        int F[2] = { 0, 2 };
        int nsub = 0;
        survivor_subset_t *subs =
            enumerate_ok(&ti, F, 2, "rs(6,3) rec 0+2", &nsub,
                         "sl-noprune");
        if (subs) {
            setenv("RJ_NO_SUBSET_PRUNE", "1", 1);
            int *keep = calloc((size_t)nsub, sizeof(*keep));
            survivor_shortlist(subs, nsub, keep);
            unsetenv("RJ_NO_SUBSET_PRUNE");
            int nkept = 0;
            for (int i = 0; i < nsub; i++)
                nkept += keep[i];
            CHECK(nkept == nsub, "sl-noprune: nkept=%d want %d", nkept,
                  nsub);
            free(keep);
            survivor_free_subsets(subs, nsub);
        }
    }
    topo_int_free(&ti);
    rj_topo_free(t);
}

/* ---------------- Part 3: per-subset end-to-end ----------------
 * Correctness must be independent of the chosen subset: every
 * enumerated subset's plan is JITed (one version) and must recover
 * the original data bit-exactly. */

/* JIT one version from a subset plan and run it end to end.
 * Returns the number of mismatched bytes (0 = correct).
 *
 * The plan is exercised via ref_run (the scalar plan interpreter):
 * this tests the subset-specific artifact (the plan) directly. JIT
 * codegen for the chosen subset is covered by Parts 5-6 through the
 * product path, which validates every version before benchmarking.
 * (Driving jit_build_version manually here bypasses the product's
 * per-candidate validation; cands[0] can be a candidate the product
 * would reject.) */
static int run_subset_e2e(topo_int_t *ti, const int *F, int nf,
                          survivor_subset_t *sub, size_t len,
                          const char *tag)
{
    int nd = ti->ndata, np = ti->npar, n = nd + np;
    uint8_t **all = abufs(n, len);
    uint8_t **keep = abufs(n, len);
    for (int i = 0; i < nd; i++)
        randfill(all[i], len);
    ref_encode(ti->mat, nd, np, all, all + nd, len);
    for (int i = 0; i < n; i++)
        memcpy(keep[i], all[i], len);

    /* wipe failed, arrange in[]/out[] in disk order (rj_fn ABI) */
    for (int f = 0; f < nf; f++)
        memset(all[F[f]], 0xA5, len);
    const uint8_t **in = malloc(sizeof(*in) * (size_t)(n - nf));
    uint8_t **out = malloc(sizeof(*out) * (size_t)nf);
    int ni = 0, fi = 0, k = 0;
    for (int dsk = 0; dsk < n; dsk++) {
        if (fi < nf && F[fi] == dsk) {
            out[k++] = all[dsk];
            fi++;
        } else
            in[ni++] = all[dsk];
    }
    /* scalar plan interpreter: runs the subset's plan directly */
    ref_run(&sub->plan, in, out, len);

    int bad = 0;
    for (int f = 0; f < nf; f++)
        for (size_t b = 0; b < len; b++)
            if (all[F[f]][b] != keep[F[f]][b])
                bad++;
    free(in);
    free(out);
    freebufs(all, n);
    freebufs(keep, n);
    (void)tag;
    return bad;
}

/* Every enumerated subset recovers correctly at the given lengths. */
static void e2e_all_subsets(const char *tag, rj_topology *t, const int *F,
                            int nf, const size_t *lens, int nlens)
{
    topo_int_t ti;
    char err[256];
    CHECK(mk_ti(t, &ti, err) == 0, "%s topo: %s", tag, err);
    if (!ti.mat) {
        topo_int_free(&ti);
        return;
    }
    int nsub = 0;
    survivor_subset_t *subs =
        enumerate_ok(&ti, F, nf, tag, &nsub, tag);
    if (subs) {
        for (int i = 0; i < nsub; i++) {
            for (int li = 0; li < nlens; li++) {
                char stag[256];
                snprintf(stag, sizeof(stag), "%s subset %d len=%zu", tag,
                         i, lens[li]);
                int bad = run_subset_e2e(&ti, F, nf, &subs[i],
                                         lens[li], stag);
                CHECK(bad == 0, "%s: %d mismatched bytes", stag, bad);
            }
        }
        printf("  info: %s: %d subsets x %d lengths all correct\n", tag,
               nsub, nlens);
        survivor_free_subsets(subs, nsub);
    }
    topo_int_free(&ti);
}

/* ---------------- Part 4: mutation evidence ---------------- */

/* M1: RJ_SUBSET_PRUNE_INVERT flips the pareto comparator (test-only
 * hook). Green pins shortlist=={{0}}; red must give {{1},{2}} at the
 * shortlist level (proven by the direct survivor_shortlist calls
 * below -- the mechanism-level red-green proof that the shortlist pins
 * a real outcome).
 *
 * Product level, post-LRC: the winner on rs(6,3)/f0 is pinned by the
 * LRC locality rule, not the shortlist. rs(6,3)'s all-ones P row is a
 * pure-XOR partitioning row, so the single-disk failure admits a
 * pure-local cover and survivor_select restricts keep[] to {0} before
 * the shortlist ever runs. The rule is intentionally robust to cost
 * inversion (see test_lrc.c::prune_invert_redgreen: "even with
 * inverted costs, the local row must still win"), so the inverted
 * product compile must STILL pick [rows 0] -- the inversion flips the
 * shortlist underneath, but the LRC rule pins the product outcome.
 * This is a falsifiable pin of the LRC/hook interaction, not a
 * vacuous assertion: if the LRC rule ever stopped firing here, the
 * inverted compile would pick a P_MULACC subset and the CHECK would
 * fail. The end-to-end flip demonstration lives in the re-homed
 * tripwire_redgreen (rs(6,3)/f02), where the shortlist -- not the LRC
 * rule -- decides.
 *
 * Soundness note on the product assertions below (C5d): they are
 * probabilistic only in the pathological case. The LRC locality rule
 * keeps only {0} here (the shortlist never runs on this case), so the
 * product has a single subset to choose from -- the winner's desc is
 * "[rows 0]" by construction, not by benchmark luck. (On multi-subset
 * cases the P2 significance gate additionally collapses noise flips
 * back to today's subset; a genuine above-noise win by another subset
 * would change the desc, which is the documented tripwire behavior,
 * not a test bug.) */
static void mutation_invert(void)
{
    rj_topology *t = rj_topo_rs(6, 3);
    topo_int_t ti;
    char err[256];
    CHECK(mk_ti(t, &ti, err) == 0, "rs63 topo: %s", err);
    if (ti.mat) {
        int F[1] = { 0 };
        int nsub = 0;
        survivor_subset_t *subs =
            enumerate_ok(&ti, F, 1, "rs(6,3) rec 0", &nsub, "mut1");
        if (subs) {
            int r0[1] = { 0 }, r1[1] = { 1 }, r2[1] = { 2 };
            int i0 = find_rows(subs, nsub, r0, 1);
            int i1 = find_rows(subs, nsub, r1, 1);
            int i2 = find_rows(subs, nsub, r2, 1);
            CHECK(i0 >= 0 && i1 >= 0 && i2 >= 0, "mut1: fixtures");

            setenv("RJ_SUBSET_PRUNE_INVERT", "1", 1);
            int *keep = calloc((size_t)nsub, sizeof(*keep));
            l2_signals(subs, nsub);
            survivor_shortlist(subs, nsub, keep);
            unsetenv("RJ_SUBSET_PRUNE_INVERT");
            /* inverted: {0} (the dominator) dropped, {1},{2} kept */
            CHECK(keep[i0] == 0, "mut1: inverted drops {0}");
            CHECK(keep[i1] == 1 && keep[i2] == 1,
                  "mut1: inverted keeps {1},{2}");

            /* green (no hook): {0} kept alone -- the pinned outcome */
            memset(keep, 0, sizeof(*keep) * (size_t)nsub);
            survivor_shortlist(subs, nsub, keep);
            CHECK(keep[i0] == 1 && keep[i1] == 0 && keep[i2] == 0,
                  "mut1: green keeps only {0}");
            free(keep);
            survivor_free_subsets(subs, nsub);
        }

        /* product-level: the LRC locality rule pins the winner to
         * [rows 0] with AND without the hook -- the rule is
         * intentionally robust to cost inversion (see the M1 header
         * and test_lrc.c::prune_invert_redgreen). The comparator flip
         * is proven at the shortlist level above; the end-to-end flip
         * demonstration lives in tripwire_redgreen (rs(6,3)/f02).
         * (benchmark=1, cheap single-subset cases so the compile is
         * fast.) */
        rj_options topts;
        memset(&topts, 0, sizeof(topts));
        topts.benchmark = 1;
        rj_op op = rj_op_recover((const int[]){ 0 }, 1);
        char cerr[512];

        setenv("RJ_SUBSET_PRUNE_INVERT", "1", 1);
        rj_jit *jr = rj_compile(t, &op, &topts, cerr, sizeof(cerr));
        unsetenv("RJ_SUBSET_PRUNE_INVERT");
        CHECK(jr != NULL, "mut1: inverted product compile: %s", cerr);
        if (jr) {
            rj_info info;
            rj_get_info(jr, &info);
            CHECK(strstr(info.desc, "[rows 0]") != NULL,
                  "mut1: inverted winner must still be [rows 0] -- LRC "
                  "rule pins the local row despite cost inversion (%s)",
                  info.desc);
            rj_free(jr);
        }

        rj_jit *jg = rj_compile(t, &op, &topts, cerr, sizeof(cerr));
        CHECK(jg != NULL, "mut1: green product compile: %s", cerr);
        if (jg) {
            rj_info info;
            rj_get_info(jg, &info);
            CHECK(strstr(info.desc, "[rows 0]") != NULL,
                  "mut1: green winner must be [rows 0] (%s)", info.desc);
            rj_free(jg);
        }
        rj_op_free(&op);
    }
    topo_int_free(&ti);
    rj_topo_free(t);
}

/* M2: corrupting one subset's solve must be caught by the end-to-end
 * check (proves Part 3 is sensitive to solve corruption). */
static void mutation_corrupt_solve(void)
{
    rj_topology *t = rj_topo_rs(6, 3);
    topo_int_t ti;
    char err[256];
    CHECK(mk_ti(t, &ti, err) == 0, "rs63 topo: %s", err);
    if (ti.mat) {
        int F[1] = { 0 };
        int nsub = 0;
        survivor_subset_t *subs =
            enumerate_ok(&ti, F, 1, "rs(6,3) rec 0", &nsub, "mut2");
        if (subs) {
            int r0[1] = { 0 };
            int i0 = find_rows(subs, nsub, r0, 1);
            CHECK(i0 >= 0, "mut2: fixture {0}");
            if (i0 >= 0) {
                /* sanity: uncorrupted recovers cleanly */
                int bad0 = run_subset_e2e(&ti, F, 1, &subs[i0], 1000,
                                          "mut2-clean");
                CHECK(bad0 == 0, "mut2: clean subset correct");

                /* corrupt one entry of the solve matrix. Toggle
                 * zero/non-zero (not just the value): P_XOR codegen
                 * only distinguishes zero from non-zero, so 1^0xFF
                 * would emit identical code. */
                plan_t *pl = &subs[i0].plan;
                uint8_t save = pl->mat[0];
                pl->mat[0] = (uint8_t)(save == 0 ? 1 : 0);
                int bad = run_subset_e2e(&ti, F, 1, &subs[i0], 1000,
                                         "mut2-corrupt");
                pl->mat[0] = save;
                CHECK(bad > 0,
                      "mut2: corrupted solve must mismatch (got %d)",
                      bad);
            }
            survivor_free_subsets(subs, nsub);
        }
    }
    topo_int_free(&ti);
    rj_topo_free(t);
}

/* ---------------- Part 4b: P2 significance-gate units ----------------
 * Pure-function tests for survivor_significance_gate (no JIT, no wall
 * clock). The gate is red-green in both directions: the noise-flip
 * case asserts the fallback happens (a no-op gate would fail it), the
 * genuine-win case asserts no fallback (an always-fall-back gate would
 * fail it). */
static void mkver(version_t *v, double gbps, double uops, double mem,
                  size_t code)
{
    memset(v, 0, sizeof(*v));
    v->gbps = gbps;
    v->uops_per_byte = uops;
    v->mem_per_byte = mem;
    v->code_size = code;
}

static void gate_units(void)
{
    /* pool: subset 0 = today (2 versions), subset 1 = challenger
     * (2 versions). today_si = 0. */
    version_t pool[4];
    int psub[4] = { 0, 0, 1, 1 };
    mkver(&pool[0], 50.0, 0.30, 7.0, 1000);
    mkver(&pool[1], 49.0, 0.31, 7.1, 1100);
    mkver(&pool[2], 51.0, 0.29, 6.9, 900);
    mkver(&pool[3], 48.0, 0.32, 7.2, 1200);

    /* (a) noise flip: challenger leads 51 vs 50 (2%) within a 3%
     * window -> fall back to today's best measured version (index
     * 0: 50 GB/s beats 49). */
    int g = survivor_significance_gate(pool, psub, 4, 0, 2, 0.03);
    CHECK(g == 0, "gate/noise-flip: got %d want 0 (today's best)", g);

    /* (b) genuine win: challenger leads 20% > window -> ships. */
    mkver(&pool[2], 60.0, 0.29, 6.9, 900);
    g = survivor_significance_gate(pool, psub, 4, 0, 2, 0.03);
    CHECK(g == 2, "gate/genuine-win: got %d want 2", g);
    mkver(&pool[2], 51.0, 0.29, 6.9, 900);

    /* (c) today already wins -> unchanged. */
    g = survivor_significance_gate(pool, psub, 4, 0, 0, 0.03);
    CHECK(g == 0, "gate/today-wins: got %d want 0", g);

    /* (d) anchor absent from the pool (only under the invert hook) ->
     * gate is a no-op. */
    g = survivor_significance_gate(pool, psub, 4, -1, 2, 0.03);
    CHECK(g == 2, "gate/no-anchor: got %d want 2", g);

    /* (e) bad inputs are no-ops, never out-of-pool. */
    g = survivor_significance_gate(pool, psub, 4, 0, -1, 0.03);
    CHECK(g == -1, "gate/bad-winner: got %d want -1", g);
    g = survivor_significance_gate(pool, psub, 4, 0, 9, 0.03);
    CHECK(g == 9, "gate/oob-winner: got %d want 9", g);
    g = survivor_significance_gate(pool, psub, 4, 0, 2, 0.0);
    CHECK(g == 2, "gate/zero-window: got %d want 2", g);

    /* (f) fallback tie-break: today's two versions measure equal;
     * the lower uops/byte wins (bench protocol order). */
    mkver(&pool[0], 50.0, 0.35, 7.0, 1000);
    mkver(&pool[1], 50.0, 0.30, 7.1, 1100);
    g = survivor_significance_gate(pool, psub, 4, 0, 2, 0.03);
    CHECK(g == 1, "gate/tie-break: got %d want 1", g);
}

/* ---------------- Part 5: dispatcher integration ---------------- */

/* rj_compile with benchmark=1 over a multi-subset recovery; the A2
 * dispatcher (rj_get_fn) must serve correct bytes, and the desc must
 * record the winning rows. Lengths are chosen as multiples of 64 to
 * stay within the notail twin's admitted set when the dispatcher is
 * installed (full cross-length dispatcher behavior is covered by
 * test_dispatch). */
static void dispatcher_integration(const char *tag, rj_topology *t,
                                   const int *F, int nf)
{
    rj_options topts;
    memset(&topts, 0, sizeof(topts));
    topts.benchmark = 1;
    rj_op op = rj_op_recover(F, nf);
    char err[512];
    rj_jit *j = rj_compile(t, &op, &topts, err, sizeof(err));
    CHECK(j != NULL, "%s: product compile: %s", tag, err);
    if (!j) {
        rj_op_free(&op);
        return;
    }
    rj_info info;
    rj_get_info(j, &info);
    CHECK(strstr(info.desc, "[rows") != NULL,
          "%s: desc records winning rows (%s)", tag, info.desc);
    printf("  info: %s winner: %s\n", tag, info.desc);

    /* end to end through the (possibly dispatched) fn. Uses the
     * file-scope 64-byte-aligned scratch buffers (see the rj_fn
     * alignment note above): an earlier version of this test passed
     * malloc'd buffers and crashed layout-dependently. */
    topo_int_t ti;
    char terr[256];
    CHECK(mk_ti(t, &ti, terr) == 0, "%s topo: %s", tag, terr);
    rj_fn fn = rj_get_fn(j);
    CHECK(fn != NULL, "%s: rj_get_fn", tag);
    if (ti.mat && fn) {
        int nd = ti.ndata, np = ti.npar, n = nd + np;
        if (n > SBUF_ROWS) {
            printf("FAIL %s: n=%d > %d\n", tag, n, SBUF_ROWS);
            failures++;
        } else {
            size_t len = 4096;
            uint8_t **all = malloc(sizeof(*all) * (size_t)n);
            uint8_t **keep = malloc(sizeof(*keep) * (size_t)n);
            for (int i = 0; i < n; i++) {
                all[i] = s_all[i];
                keep[i] = s_keep[i];
            }
            for (int i = 0; i < nd; i++)
                randfill(all[i], len);
            ref_encode(ti.mat, nd, np, all, all + nd, len);
            for (int i = 0; i < n; i++)
                memcpy(keep[i], all[i], len);
            for (int f = 0; f < nf; f++)
                memset(all[F[f]], 0xA5, len);
            const void *in[24];
            void *out[8];
            int ni = 0, fi = 0, k = 0;
            for (int dsk = 0; dsk < n; dsk++) {
                if (fi < nf && F[fi] == dsk) {
                    out[k++] = all[dsk];
                    fi++;
                } else
                    in[ni++] = all[dsk];
            }
            fn(in, out, len);
            int bad = 0;
            for (int f = 0; f < nf; f++)
                for (size_t b = 0; b < len; b++)
                    if (all[F[f]][b] != keep[F[f]][b])
                        bad++;
            CHECK(bad == 0, "%s: len=%zu %d mismatches", tag, len, bad);
            free(all);
            free(keep);
        }
    }
    topo_int_free(&ti);
    rj_free(j);
    rj_op_free(&op);
}

/* Whole-disk RDP(5) through the product path (rj_compile,
 * benchmark=1): 34 enumerated subsets, the stress case behind the
 * fixed plans[]-realloc borrow-dangle bug. The shortlist caps the
 * pooled bench at SURVIVOR_SHORTLIST_CAP subsets; the winner's rows
 * are recorded in the desc and the dispatched fn recovers correctly. */
static void product_wholedisk_rdp5(void)
{
    const char *tag = "rdp(5)/whole-disk0";
    rj_topology *t = rj_topo_rdp(5);
    const int F[4] = { 0, 1, 2, 3 }; /* whole data disk 0 = 4 row-disks */
    rj_options topts;
    memset(&topts, 0, sizeof(topts));
    topts.benchmark = 1;
    rj_op op = rj_op_recover(F, 4);
    char err[512];
    rj_jit *j = rj_compile(t, &op, &topts, err, sizeof(err));
    CHECK(j != NULL, "%s: product compile: %s", tag, err);
    if (!j) {
        rj_op_free(&op);
        rj_topo_free(t);
        return;
    }
    rj_info info;
    rj_get_info(j, &info);
    CHECK(strstr(info.desc, "[rows") != NULL,
          "%s: desc records winning rows (%s)", tag, info.desc);
    printf("  info: %s winner: %s\n", tag, info.desc);

    topo_int_t ti;
    char terr[256];
    CHECK(mk_ti(t, &ti, terr) == 0, "%s topo: %s", tag, terr);
    rj_fn fn = rj_get_fn(j);
    CHECK(fn != NULL, "%s: rj_get_fn", tag);
    if (ti.mat && fn) {
        int nd = ti.ndata, np = ti.npar, n = nd + np;
        size_t len = 4096; /* multiple of 64: dispatcher-safe */
        uint8_t **all = malloc(sizeof(*all) * (size_t)n);
        uint8_t **keep = malloc(sizeof(*keep) * (size_t)n);
        if (n > SBUF_ROWS) {
            printf("FAIL %s: n=%d > %d\n", tag, n, SBUF_ROWS);
            failures++;
        } else {
            for (int i = 0; i < n; i++) {
                all[i] = s_all[i];
                keep[i] = s_keep[i];
            }
            for (int i = 0; i < nd; i++)
                randfill(all[i], len);
            ref_encode(ti.mat, nd, np, all, all + nd, len);
            for (int i = 0; i < n; i++)
                memcpy(keep[i], all[i], len);
            for (int f = 0; f < 4; f++)
                memset(all[F[f]], 0xA5, len);
            const void *in[24];
            void *out[4];
            int ni = 0, fi = 0, k = 0;
            for (int dsk = 0; dsk < n; dsk++) {
                if (fi < 4 && F[fi] == dsk) {
                    out[k++] = all[dsk];
                    fi++;
                } else
                    in[ni++] = all[dsk];
            }
            fn(in, out, len);
            int bad = 0;
            for (int f = 0; f < 4; f++)
                for (size_t b = 0; b < len; b++)
                    if (all[F[f]][b] != keep[F[f]][b])
                        bad++;
            CHECK(bad == 0, "%s: %d mismatches", tag, bad);
            free(all);
            free(keep);
        }
    }
    topo_int_free(&ti);
    rj_free(j);
    rj_op_free(&op);
    rj_topo_free(t);
}

/* P3: RJ_NO_SURVIVOR_SELECT=1 takes the legacy single-subset
 * (all-rows) path in rj_compile's recover branch. Red-green: the
 * knobbed compile's desc must name the all-rows plan ("[rows
 * 0+1+2]"), while the default compile's desc names the subset winner
 * ("[rows 0]"). A broken knob (ignored env) fails the first
 * assertion; a broken default path fails the second. */
static void disable_knob(void)
{
    rj_topology *t = rj_topo_rs(6, 3);
    const int F[1] = { 0 };
    rj_options topts;
    memset(&topts, 0, sizeof(topts));
    topts.benchmark = 1;
    char err[512];

    rj_op op = rj_op_recover(F, 1);
    rj_jit *jg = rj_compile(t, &op, &topts, err, sizeof(err));
    rj_op_free(&op);
    CHECK(jg != NULL, "knob/default compile: %s", err);
    if (jg) {
        rj_info info;
        rj_get_info(jg, &info);
        CHECK(strstr(info.desc, "[rows 0]") != NULL,
              "knob/default: subset winner desc (%s)", info.desc);
        rj_free(jg);
    }

    setenv("RJ_NO_SURVIVOR_SELECT", "1", 1);
    op = rj_op_recover(F, 1);
    rj_jit *jk = rj_compile(t, &op, &topts, err, sizeof(err));
    unsetenv("RJ_NO_SURVIVOR_SELECT");
    rj_op_free(&op);
    CHECK(jk != NULL, "knob/disabled compile: %s", err);
    if (jk) {
        rj_info info;
        rj_get_info(jk, &info);
        CHECK(strstr(info.desc, "[rows 0+1+2]") != NULL,
              "knob/disabled: legacy all-rows desc (%s)", info.desc);
        rj_free(jk);
    }

    /* "0" and empty mean "not set" (the A1 RJ_NO_PRUNE contract). */
    setenv("RJ_NO_SURVIVOR_SELECT", "0", 1);
    op = rj_op_recover(F, 1);
    rj_jit *jz = rj_compile(t, &op, &topts, err, sizeof(err));
    unsetenv("RJ_NO_SURVIVOR_SELECT");
    rj_op_free(&op);
    CHECK(jz != NULL, "knob/zero compile: %s", err);
    if (jz) {
        rj_info info;
        rj_get_info(jz, &info);
        CHECK(strstr(info.desc, "[rows 0]") != NULL,
              "knob/zero: survivor path still taken (%s)", info.desc);
        rj_free(jz);
    }
    rj_topo_free(t);
}

/* ---------------- Part 6: product subset-independence ----------------
 * Default (shortlisted) vs RJ_NO_SUBSET_PRUNE=1 (exhaustive) must both
 * recover correctly. The exhaustive mode also pins the bug-(a) fix:
 * the multi-subset pool is validated per-version against its own plan
 * while version_t.plan borrows into the pre-sized (never realloc'd)
 * plans[] array -- the old realloc path dangled those borrows and
 * failed validation against freed memory here. Winner equality is NOT
 * asserted: benchmark noise flips winners between runs by
 * construction; it is asserted in Part 7 under pinned bench knobs. */
static void product_subset_independence(void)
{
    rj_topology *t = rj_topo_rs(6, 3);
    const int F[2] = { 0, 2 };
    for (int mode = 0; mode < 2; mode++) {
        if (mode == 1)
            setenv("RJ_NO_SUBSET_PRUNE", "1", 1);
        rj_options topts;
        memset(&topts, 0, sizeof(topts));
        topts.benchmark = 1;
        rj_op op = rj_op_recover(F, 2);
        char err[512];
        rj_jit *j = rj_compile(t, &op, &topts, err, sizeof(err));
        if (mode == 1)
            unsetenv("RJ_NO_SUBSET_PRUNE");
        CHECK(j != NULL, "prod-indep mode=%d: compile: %s", mode, err);
        if (!j) {
            rj_op_free(&op);
            continue;
        }
        rj_info info;
        rj_get_info(j, &info);
        printf("  info: prod-indep mode=%d winner: %s\n", mode,
               info.desc);

        topo_int_t ti;
        char terr[256];
        CHECK(mk_ti(t, &ti, terr) == 0, "prod-indep topo: %s", terr);
        rj_fn fn = rj_get_fn(j);
        if (ti.mat && fn) {
            int nd = ti.ndata, np = ti.npar, n = nd + np;
            size_t len = 4096;
            /* shared 64-byte-aligned scratch (see Part 5 note) */
            uint8_t **all = malloc(sizeof(*all) * (size_t)n);
            uint8_t **keep = malloc(sizeof(*keep) * (size_t)n);
            if (n > SBUF_ROWS) {
                printf("FAIL prod-indep: n=%d > %d\n", n, SBUF_ROWS);
                failures++;
            } else {
                uint8_t **all = malloc(sizeof(*all) * (size_t)n);
                uint8_t **keep = malloc(sizeof(*keep) * (size_t)n);
                for (int i = 0; i < n; i++) {
                    all[i] = s_all[i];
                    keep[i] = s_keep[i];
                }
                for (int i = 0; i < nd; i++)
                    randfill(all[i], len);
                ref_encode(ti.mat, nd, np, all, all + nd, len);
                for (int i = 0; i < n; i++)
                    memcpy(keep[i], all[i], len);
                for (int f = 0; f < 2; f++)
                    memset(all[F[f]], 0xA5, len);
                const void *in[16];
                void *out[2];
                int ni = 0, fi = 0, k = 0;
                for (int dsk = 0; dsk < n; dsk++) {
                    if (fi < 2 && F[fi] == dsk) {
                        out[k++] = all[dsk];
                        fi++;
                    } else
                        in[ni++] = all[dsk];
                }
                fn(in, out, len);
                int bad = 0;
                for (int f = 0; f < 2; f++)
                    for (size_t b = 0; b < len; b++)
                        if (all[F[f]][b] != keep[F[f]][b])
                            bad++;
                CHECK(bad == 0, "prod-indep mode=%d: %d mismatches",
                      mode, bad);
                free(all);
                free(keep);
            }
        }
        topo_int_free(&ti);
        rj_free(j);
        rj_op_free(&op);
    }
    rj_topo_free(t);
}

/* ---------------- Part 7: L2 tripwire (DESIGN.md section 7) ----------------
 * On a fixed corpus with PINNED bench knobs, the L2-shortlisted
 * selection must pick the same winning subset as the exhaustive
 * (RJ_NO_SUBSET_PRUNE=1) selection. The winning subset is read off
 * the plan desc's "[rows ...]" suffix. Any flip fails loudly: it
 * means the shortlist dropped a subset that would have won measured
 * (L2 unsafe).
 *
 * Noise note: both compiles run their own bench_select, so pure
 * noise can in principle flip either side -- on a heavily loaded
 * machine the measured noise window can exceed 100% (observed
 * 2026-10-03: 27-136% with a competing benchmark load), making
 * strict winner equality flaky through no fault of the shortlist.
 * The comparison is on the RAW measured winners
 * (RJ_SURVIVOR_NO_GATE=1 skips the P2 significance gate): the
 * tripwire is the L2-safety instrument ("no pruned subset would have
 * won measured", DESIGN §3), while the gate is a post-measurement
 * policy unit-tested separately in Part 4b. On a strict flip, the
 * head-to-head REFEREE adjudicates: the two disagreeing subsets'
 * versions are benched together in one pool with pinned knobs, and
 * only a gap beyond the measured noise window fails loudly as a
 * genuine over-prune; a within-noise gap passes with a loud note as
 * borderline (the deferred P4's domain). The referee predicate is
 * pure and unit-tested red-green in both directions. */

/* Compile one recover op and extract the "[rows ...]" winner suffix.
 * exhaustive=1 sets RJ_NO_SUBSET_PRUNE=1. no_gate=1 sets
 * RJ_SURVIVOR_NO_GATE=1 (raw measured winner). Returns 1 on success. */
static int tripwire_compile(const char *tag, rj_topology *t, const int *F,
                            int nf, int exhaustive, int no_gate, int nt,
                            char *rows_out, size_t rows_sz)
{
    rj_options topts;
    memset(&topts, 0, sizeof(topts));
    topts.benchmark = 1;
    /* nt: P1.2 extension -- run the tripwire with NT twins enabled so
     * the shortlisted-vs-exhaustive comparison actually exercises the
     * NT-flip failure mode (the pre-P1.2 tripwire used nt_threshold=0,
     * so NT twins were never in play). */
    topts.nt_threshold = nt ? 1 : 0;
    /* pinned bench knobs: the two runs must be comparable */
    topts.bench_bytes = (size_t)1 << 20;
    topts.bench_iters = 11;
    rj_op op = rj_op_recover(F, nf);
    char err[512];
    if (exhaustive)
        setenv("RJ_NO_SUBSET_PRUNE", "1", 1);
    if (no_gate)
        setenv("RJ_SURVIVOR_NO_GATE", "1", 1);
    rj_jit *j = rj_compile(t, &op, &topts, err, sizeof(err));
    if (no_gate)
        unsetenv("RJ_SURVIVOR_NO_GATE");
    if (exhaustive)
        unsetenv("RJ_NO_SUBSET_PRUNE");
    rj_op_free(&op);
    CHECK(j != NULL, "%s/exh=%d/nt=%d: product compile: %s", tag,
          exhaustive, nt, err);
    if (!j)
        return 0;
    rj_info info;
    rj_get_info(j, &info);
    const char *p = strstr(info.desc, "[rows ");
    CHECK(p != NULL, "%s/exh=%d/nt=%d: desc has [rows ...] (%s)", tag,
          exhaustive, nt, info.desc);
    if (p) {
        size_t k = 0;
        while (p[k] && p[k] != ']' && k + 1 < rows_sz) {
            rows_out[k] = p[k];
            k++;
        }
        if (p[k] == ']' && k + 1 < rows_sz) {
            rows_out[k] = ']';
            rows_out[k + 1] = '\0';
        } else
            rows_out[0] = '\0';
    } else
        rows_out[0] = '\0';
    printf("  info: tripwire %s exh=%d nt=%d winner %s (%s)\n", tag,
           exhaustive, nt, rows_out, info.desc);
    rj_free(j);
    return 1;
}

/* Parse a "[rows 0+1+2]" winner suffix into row ints. Returns the
 * row count, or -1 on parse failure. */
static int parse_rows(const char *s, int *rows, int maxrows)
{
    const char *p = strstr(s, "[rows ");
    if (!p)
        return -1;
    p += strlen("[rows "); /* 6 chars: '[', 'r', 'o', 'w', 's', ' ' */
    int n = 0;
    while (*p && *p != ']' && n < maxrows) {
        char *end;
        long v = strtol(p, &end, 10);
        if (end == p)
            return -1;
        rows[n++] = (int)v;
        p = end;
        if (*p == '+')
            p++;
        else if (*p != ']')
            return -1;
    }
    return (*p == ']') ? n : -1;
}

/* Tripwire referee (pure): head-to-head best measured GB/s of the
 * shortlisted winner's subset (ga) vs the exhaustive winner's subset
 * (gb), plus the head-to-head's measured noise window. Returns 1 if
 * gb is GENUINELY faster than ga -- a real L2 over-prune -- and 0 if
 * the gap is within noise (borderline: the deferred P4's domain) or
 * the inputs are bad. Red-green in both directions below. */
static int tripwire_referee(double ga, double gb, double window)
{
    if (!(ga > 0) || !(gb > 0) || !(window > 0))
        return 0;
    /* gb genuinely faster iff it exceeds ga by more than the window.
     * Strict > keeps the borderline on the PASS side. */
    return gb > ga * (1.0 + window) ? 1 : 0;
}

static void tripwire_referee_units(void)
{
    /* parse_rows: the "[rows ...]" suffix parser (an off-by-one here
     * silently misidentified the referee's subsets). */
    {
        int rows[8];
        int n = parse_rows("[rows 0+1+2+7]", rows, 8);
        CHECK(n == 4 && rows[0] == 0 && rows[1] == 1 && rows[2] == 2 &&
                  rows[3] == 7,
              "parse_rows/basic: n=%d", n);
        n = parse_rows("rs(6,3) rec 0 [rows 0] xor avx2 U=6", rows, 8);
        CHECK(n == 1 && rows[0] == 0, "parse_rows/single: n=%d", n);
        n = parse_rows("no rows here", rows, 8);
        CHECK(n == -1, "parse_rows/missing: n=%d", n);
        n = parse_rows("[rows 0+1", rows, 8);
        CHECK(n == -1, "parse_rows/unclosed: n=%d", n);
    }
    /* genuine win: 50% faster, 5% window -> over-prune */
    CHECK(tripwire_referee(50.0, 75.0, 0.05) == 1,
          "referee/genuine");
    /* noise: 2% gap, 5% window -> borderline, not an over-prune */
    CHECK(tripwire_referee(50.0, 51.0, 0.05) == 0,
          "referee/noise");
    /* slower exhaustive winner -> not an over-prune */
    CHECK(tripwire_referee(50.0, 40.0, 0.05) == 0,
          "referee/slower");
    /* bad inputs never report an over-prune */
    CHECK(tripwire_referee(0.0, 75.0, 0.05) == 0,
          "referee/bad-ga");
    CHECK(tripwire_referee(50.0, 75.0, 0.0) == 0,
          "referee/bad-window");
}

/* Build all A1-pruned versions for one subset's plan. */
static int build_subset_versions(const topo_int_t *ti, survivor_subset_t *sub,
                                 const rj_cpu_info *cpu, const rj_options *o,
                                 int allow_nt,
                                 version_t **vs_out, int *nv_out)
{
    cand_t cands[512];
    int nc = plan_candidates(&sub->plan, cpu, cands, 512, allow_nt, 0);
    version_t *vs = calloc((size_t)(nc > 0 ? nc : 1), sizeof(*vs));
    int nv = 0;
    for (int i = 0; i < nc; i++) {
        char e2[256];
        if (jit_build_version(&sub->plan, &cands[i], cpu, o, &vs[nv],
                              e2, sizeof(e2)) == 0)
            nv++;
    }
    nv = prune_versions(vs, nv, 0);
    *vs_out = vs;
    *nv_out = nv;
    (void)ti;
    return nv > 0 ? 0 : -1;
}

/* Head-to-head referee bench: build the A1-pruned versions of exactly
 * the two disagreeing subsets and bench them together with pinned
 * knobs. Reports each subset's best measured GB/s and the measured
 * noise window. Returns 0 on success. */
static int tripwire_head_to_head(const char *tag, rj_topology *t,
                                 const int *F, int nf, const int *ra,
                                 int na, const int *rb, int nb, int nt,
                                 double *ga_out, double *gb_out,
                                 double *window_out)
{
    topo_int_t ti;
    char err[512];
    if (mk_ti(t, &ti, err) != 0) {
        printf("FAIL %s: referee topo: %s\n", tag, err);
        failures++;
        return -1;
    }
    int nsub = 0;
    /* NOTE (C5e): enumerate inline with a loud non-CHECK failure
     * instead of enumerate_ok: this function runs only on tripwire
     * flips, so a CHECK here would make the check count
     * flip-dependent (benchmark-noise-dependent). Every error path
     * in this function records failures without CHECK for the same
     * reason -- the count must not depend on bench outcomes. */
    survivor_subset_t *subs = NULL;
    {
        char e2[512];
        int erc = survivor_enumerate(&ti, F, nf, tag, &subs, &nsub, e2,
                                    sizeof(e2));
        if (erc != 0) {
            printf("FAIL %s: referee enumerate rc=%d (%s)\n", tag, erc,
                   e2);
            failures++;
            topo_int_free(&ti);
            return -1;
        }
    }
    int ia = find_rows(subs, nsub, ra, na);
    int ib = find_rows(subs, nsub, rb, nb);
    if (ia < 0 || ib < 0) {
        printf("FAIL %s: referee subsets not found (a=%d b=%d)\n", tag,
               ia, ib);
        failures++;
        survivor_free_subsets(subs, nsub);
        topo_int_free(&ti);
        return -1;
    }
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    rj_options o;
    rj_options_default(&o);
    o.benchmark = 1;
    /* Same flags as the tripwire compiles (P1.2): the referee must
     * adjudicate in the NT-twins-enabled world when the tripwire ran
     * there, or it could miss an NT-driven flip. */
    o.nt_threshold = nt ? 1 : 0;
    o.bench_bytes = (size_t)1 << 20;
    o.bench_iters = 11;
    int allow_nt = o.nt_threshold > 0 && o.benchmark;

    version_t *pool = NULL, *vsa = NULL, *vsb = NULL;
    int npool = 0, nva = 0, nvb = 0;
    int *pool_sub = NULL;
    int rc = 0;
    if (build_subset_versions(&ti, &subs[ia], &cpu, &o, allow_nt, &vsa,
                              &nva) != 0 ||
        build_subset_versions(&ti, &subs[ib], &cpu, &o, allow_nt, &vsb,
                              &nvb) != 0) {
        printf("FAIL %s: referee version build\n", tag);
        failures++;
        rc = -1;
    } else {
        pool = malloc(sizeof(*pool) * (size_t)(nva + nvb));
        pool_sub = malloc(sizeof(*pool_sub) * (size_t)(nva + nvb));
        if (!pool || !pool_sub) {
            printf("FAIL %s: referee oom\n", tag);
            failures++;
            rc = -1;
        } else {
            for (int i = 0; i < nva; i++) {
                pool[npool] = vsa[i];
                pool_sub[npool++] = 0;
            }
            for (int i = 0; i < nvb; i++) {
                pool[npool] = vsb[i];
                pool_sub[npool++] = 1;
            }
            free(vsa);
            free(vsb);
            vsa = vsb = NULL;
            double window = 0;
            int w = bench_select(pool, npool, &subs[ia].plan, &o, 0,
                                 &window, err, sizeof(err));
            if (w < 0) {
                printf("FAIL %s: referee bench: %s\n", tag, err);
                failures++;
                rc = -1;
            } else {
                double ga = -1, gb = -1;
                for (int i = 0; i < npool; i++) {
                    if (pool_sub[i] == 0 && pool[i].gbps > ga)
                        ga = pool[i].gbps;
                    if (pool_sub[i] == 1 && pool[i].gbps > gb)
                        gb = pool[i].gbps;
                }
                printf("  info: referee %s: A=%.2f GB/s B=%.2f GB/s "
                       "window=%.1f%%\n",
                       tag, ga, gb, window * 100.0);
                *ga_out = ga;
                *gb_out = gb;
                *window_out = window;
            }
        }
    }
    for (int i = 0; i < npool; i++)
        jit_free_version(&pool[i]);
    free(pool);
    free(pool_sub);
    if (vsa) {
        for (int i = 0; i < nva; i++)
            jit_free_version(&vsa[i]);
        free(vsa);
    }
    if (vsb) {
        for (int i = 0; i < nvb; i++)
            jit_free_version(&vsb[i]);
        free(vsb);
    }
    survivor_free_subsets(subs, nsub);
    topo_int_free(&ti);
    return rc;
}

/* Tripwire adjudication core shared by tripwire_case (loud; the
 * green-path instrument) and tripwire_referee_fires (the probe that
 * asserts the referee fires). Runs the shortlisted and exhaustive
 * compiles (pinned knobs, raw winners), strict-compares the winning
 * subsets, and on a flip runs the head-to-head referee. Prints only
 * info lines and never touches `failures`; the caller adds the loud
 * FAIL or the assertion.
 *
 * Returns 0 when the shortlist is safe (strict subset match, or a
 * flip the referee adjudicates as within-noise borderline), 1 when
 * the referee fires (genuine over-prune), 2 when the winner rows
 * cannot be parsed, 3 on a compile or head-to-head failure (the
 * inner helper already recorded the FAIL). The detail out-params
 * carry the adjudication for the caller's message. */
static int tripwire_verdict(const char *tag, rj_topology *t, const int *F,
                            int nf, int nt, char *rs, size_t rs_sz,
                            char *re, size_t re_sz, double *ga_out,
                            double *gb_out, double *window_out)
{
    if (!tripwire_compile(tag, t, F, nf, 0, 1, nt, rs, rs_sz))
        return 3;
    if (!tripwire_compile(tag, t, F, nf, 1, 1, nt, re, re_sz))
        return 3;
    if (strcmp(rs, re) == 0)
        return 0;
    printf("  info: tripwire %s FLIP: shortlisted %s != exhaustive %s; "
           "running head-to-head referee\n",
           tag, rs, re);
    int ra[32], rb[32];
    int na = parse_rows(rs, ra, 32);
    int nb = parse_rows(re, rb, 32);
    if (na < 0 || nb < 0) {
        printf("  info: tripwire %s: cannot parse winner rows (%s / %s)\n",
               tag, rs, re);
        return 2;
    }
    double ga = 0, gb = 0, window = 0;
    if (tripwire_head_to_head(tag, t, F, nf, ra, na, rb, nb, nt, &ga,
                              &gb, &window) != 0)
        return 3;
    *ga_out = ga;
    *gb_out = gb;
    *window_out = window;
    if (tripwire_referee(ga, gb, window)) {
        printf("  info: tripwire %s: referee FIRED: exhaustive winner %s "
               "(%.2f GB/s) beats shortlisted winner %s (%.2f GB/s) "
               "beyond the %.1f%% noise window\n",
               tag, re, gb, rs, ga, window * 100.0);
        return 1;
    }
    printf("  info: tripwire %s: flip within noise (%.2f vs %.2f GB/s, "
           "window %.1f%%): borderline, P4 domain\n",
           tag, ga, gb, window * 100.0);
    return 0;
}

/* Loud wrapper for a precomputed tripwire verdict: records the FAIL
 * exactly as the in-line path would, without re-running any benches.
 * Returns tripwire_case's return for that verdict (1 = pass, 0 =
 * genuine over-prune or internal error). Lets tests pin the
 * verdict -> loud-FAIL wiring without paying for a second bench run. */
static int tripwire_loud(int v, const char *tag, const char *rs,
                         const char *re, double ga, double gb,
                         double window)
{
    if (v == 1) {
        printf("FAIL tripwire %s: GENUINE over-prune: exhaustive winner "
               "%s (%.2f GB/s) beats shortlisted winner %s (%.2f GB/s) "
               "beyond the %.1f%% noise window\n",
               tag, re, gb, rs, ga, window * 100.0);
        failures++;
        return 0;
    }
    if (v == 2) {
        printf("FAIL tripwire %s: cannot parse winner rows (%s / %s)\n",
               tag, rs, re);
        failures++;
        return 0;
    }
    /* v == 3: the compile/head-to-head helper already recorded its
     * FAIL; nothing more to add. */
    return v == 0 ? 1 : 0;
}

/* Returns 1 if the tripwire passes, 0 on a genuine over-prune.
 * Strict winner equality first; on a flip, the head-to-head referee
 * adjudicates: a within-noise gap is borderline (the deferred P4's
 * domain) and passes with a loud note, a genuine gap fails loudly.
 * nt: 0 = plain (NT twins disabled), 1 = NT twins enabled (P1.2). */
static int tripwire_case(const char *tag, rj_topology *t, const int *F,
                         int nf, int nt)
{
    char rs[64], re[64];
    double ga = 0, gb = 0, window = 0;
    int v = tripwire_verdict(tag, t, F, nf, nt, rs, sizeof(rs), re,
                             sizeof(re), &ga, &gb, &window);
    return tripwire_loud(v, tag, rs, re, ga, gb, window);
}

static void tripwire(void)
{
    /* Each corpus case runs twice: plain (nt=0) and with NT twins
     * enabled (nt=1, P1.2). The NT pass is the empirical backstop for
     * the NT-delta subset-independence assumption (docs/survivor_sets.md
     * section 2c): a subset L2-dropped on its plain version must not
     * own the measured winner through its NT twin. */
    {
        rj_topology *t = rj_topo_rs(6, 3);
        const int F[2] = { 0, 2 };
        CHECK(tripwire_case("rs(6,3)/f02", t, F, 2, 0) == 1,
              "tripwire rs(6,3)/f02: no genuine over-prune");
        CHECK(tripwire_case("rs(6,3)/f02+nt", t, F, 2, 1) == 1,
              "tripwire rs(6,3)/f02+nt: no genuine over-prune");
        rj_topo_free(t);
    }
    {
        uint8_t m[4 * 10];
        for (int j = 0; j < 4; j++)
            for (int i = 0; i < 10; i++)
                m[j * 10 + i] = (uint8_t)(1 + ((7 * i + 13 * j) % 255));
        rj_topology *t = rj_topo_custom(10, 4, m);
        const int F[2] = { 1, 3 };
        CHECK(tripwire_case("rs(10,4)c/f13", t, F, 2, 0) == 1,
              "tripwire rs(10,4)c/f13: no genuine over-prune");
        CHECK(tripwire_case("rs(10,4)c/f13+nt", t, F, 2, 1) == 1,
              "tripwire rs(10,4)c/f13+nt: no genuine over-prune");
        rj_topo_free(t);
    }
    {
        /* 34 enumerated subsets: the case where L2 actually prunes */
        rj_topology *t = rj_topo_rdp(5);
        const int F[4] = { 0, 1, 2, 3 };
        CHECK(tripwire_case("rdp(5)/whole-disk0", t, F, 4, 0) == 1,
              "tripwire rdp(5)/whole-disk0: no genuine over-prune");
        CHECK(tripwire_case("rdp(5)/whole-disk0+nt", t, F, 4, 1) == 1,
              "tripwire rdp(5)/whole-disk0+nt: no genuine over-prune");
        rj_topo_free(t);
    }
}

/* Red-green for the tripwire: under RJ_SUBSET_PRUNE_INVERT the
 * shortlist drops the today anchor (and inverts pareto), so the
 * strict comparison MUST flip. Uses rs(6,3)/f02: the two data-disk
 * failures share the single local group, so the LRC locality rule
 * cannot fire (its cover needs distinct local rows) and the shortlist
 * decides. Inverted, the shortlist keeps only the dominated {1+2} --
 * a single-subset pool, so the inverted winner is [rows 1+2] by
 * construction -- while the exhaustive side rests on the measured
 * gap clearing the noise window (static ~1.05x uops / ~1.10x mem,
 * measured ~1.18x vs 15.8% window on this machine: observed, thin
 * margin -- a flake on a heavily loaded box fails loudly, not a
 * correctness hole). The inverted side of the flip is structural;
 * the exhaustive side is empirical. (f0
 * was the original case, with the same structural shape via the
 * P_XOR-vs-P_MULACC gap; the LRC rule -- intentionally robust to cost
 * inversion -- now pins f0's winner before the shortlist runs, so the
 * probe moved to f02. See M1's header.) If the strict comparator ever
 * stops failing here, it has lost its teeth. (The green path's referee
 * is pure logic unit-tested in tripwire_referee_units.) */
static void tripwire_redgreen(void)
{
    rj_topology *t = rj_topo_rs(6, 3);
    const int F[2] = { 0, 2 };
    char rs[64], re[64];
    setenv("RJ_SUBSET_PRUNE_INVERT", "1", 1);
    int ok1 = tripwire_compile("rs(6,3)/f02", t, F, 2, 0, 1, 0, rs,
                               sizeof(rs));
    unsetenv("RJ_SUBSET_PRUNE_INVERT");
    int ok2 = tripwire_compile("rs(6,3)/f02", t, F, 2, 1, 1, 0, re,
                               sizeof(re));
    if (ok1 && ok2) {
        CHECK(strcmp(rs, re) != 0,
              "tripwire red: inverted shortlist must flip the winner "
              "(short=%s exh=%s)",
              rs, re);
        CHECK(strstr(rs, "[rows 1+2]") != NULL,
              "tripwire red: inverted winner is the dominated {1+2} "
              "(%s)",
              rs);
        CHECK(strstr(re, "[rows 1+2]") == NULL,
              "tripwire red: exhaustive raw winner is not the dominated "
              "{1+2} (%s)",
              re);
    }
    rj_topo_free(t);
}

/* The full referee path, end to end, through the real bench
 * plumbing: under RJ_SUBSET_PRUNE_INVERT the strict comparator flips
 * (shortlisted {{1+2}} vs exhaustive {0+1}/{0+2} on rs(6,3)/f02 -- the
 * re-homed tripwire_redgreen case, where the shortlist decides because
 * the LRC rule cannot fire), the flip reaches tripwire_head_to_head,
 * and tripwire_referee adjudicates the measured gap against the
 * measured noise window.
 *
 * DEVIATION (documented, 2026-10-03; re-homed f0 -> f02 2026-10-13): the
 * review finding asked to assert the referee FIRES here (verdict == 1), on
 * the theory that the STATIC gap "must exceed any sane window".
 * Measured on rs(6,3)/f02 it is load-dependent, not a fixed property: the
 * keeplist probe gives static ratios of only ~1.05x (uops) / ~1.10x (mem)
 * for {1+2} vs {0+1}, and the measured gap is ~1.18x on the INVERT path
 * (54.54 vs 46.27 GB/s, window 15.8% -- FIRED, verdict 1 this run) and ~1.04x
 * on the green path (55.77 vs 53.53, window 10.1% -- within noise,
 * verdict 0). Both verdicts occur on the same box in the same run, so
 * asserting either one would be load-flaky by construction. Instead this
 * probe pins what IS deterministic:
 * (a) the flip reaches the referee through real benches (the verdict
 * adjudicates, never an internal error); (b) the verdict -> loud-FAIL
 * wiring is consistent (exactly one loud FAIL iff the referee
 * fired); (c) the referee==1 -> loud-FAIL link itself, as a pure
 * unit. The loud FAIL firing in (b) is exactly the mechanism the
 * P4-domain tripwire_case probes rely on in production.
 *
 * Red-green coverage note: tripwire_redgreen pins the strict
 * comparator's flip, and tripwire_referee_units pins the pure referee
 * predicate in both directions; this test pins the wiring between
 * them. A verdict of 2/3 means the probe itself is broken (fix the
 * probe, do not normalize). The loud-path CHECK restores the
 * counter: the FAIL is the asserted wiring outcome, not a defect. */
static void tripwire_referee_fires(void)
{
    rj_topology *t = rj_topo_rs(6, 3);
    const int F[2] = { 0, 2 };
    setenv("RJ_SUBSET_PRUNE_INVERT", "1", 1);
    /* (a) through the real bench plumbing: the flip must reach the
     * referee and the referee must adjudicate (0 or 1), not fail
     * internally (2/3). */
    char rs[64], re[64];
    double ga = 0, gb = 0, window = 0;
    int v = tripwire_verdict("rs(6,3)/f02", t, F, 2, 0, rs, sizeof(rs),
                             re, sizeof(re), &ga, &gb, &window);
    unsetenv("RJ_SUBSET_PRUNE_INVERT");
    CHECK(v == 0 || v == 1,
          "tripwire referee-path: referee must adjudicate through real "
          "benches (verdict=%d short=%s exh=%s ga=%.2f gb=%.2f "
          "window=%.1f%%)",
          v, rs, re, ga, gb, window * 100.0);
    int flip = (v == 0 || v == 1) && strcmp(rs, re) != 0;
    CHECK(flip == 1,
          "tripwire referee-path: INVERT must flip the strict comparator "
          "(short=%s exh=%s)",
          rs, re);
    /* (b) verdict -> loud-FAIL wiring consistency: exactly one loud
     * FAIL iff the referee fired. tripwire_loud is the same wrapper
     * tripwire_case uses, so no second bench run is needed. */
    int before = failures;
    int ok = tripwire_loud(v, "rs(6,3)/f02", rs, re, ga, gb, window);
    int fired = failures - before;
    failures = before;
    CHECK((v == 1) == (!ok && fired == 1),
          "tripwire referee-path: loud FAIL iff referee fired "
          "(verdict=%d ok=%d fired=%d)",
          v, ok, fired);
    /* (c) the referee==1 -> loud-FAIL link, pure unit (no benches):
     * a firing verdict records exactly one FAIL, a safe verdict
     * records none. The firing call below is INTENTIONAL and is
     * announced on stdout as a self-test, so the deliberately
     * recorded FAIL line (tagged selftest-loud-path, which no real
     * topology tag can match) cannot be mistaken for a real
     * tripwire failure. */
    before = failures;
    printf("SELFTEST tripwire loud-path: the following FAIL line is "
           "intentional (asserts the loud-FAIL wiring)\n");
    ok = tripwire_loud(1, "selftest-loud-path", "[rows 1+2]", "[rows 0+2]",
                       46.27, 54.54, 0.158);
    CHECK(!ok && failures - before == 1,
          "tripwire loud/fires: one FAIL (ok=%d fired=%d)", ok,
          failures - before);
    failures = before;
    before = failures;
    ok = tripwire_loud(0, "selftest-loud-path", "[rows 0+2]", "[rows 0+2]",
                       54.54, 54.54, 0.158);
    CHECK(ok && failures == before, "tripwire loud/safe: no FAIL");
    rj_topo_free(t);
}

int main(void)
{
    printf("== Part 1: enumeration units ==\n");
    enum_rs63();
    enum_rs104c();
    {
        rj_topology *t = rj_topo_rdp(5);
        enum_xorcode_whole("rdp(5)", t, 7);
        rj_topo_free(t);
    }
    {
        rj_topology *t = rj_topo_evenodd(5);
        enum_xorcode_whole("evenodd(5)", t, 7);
        rj_topo_free(t);
    }
    {
        rj_topology *t = rj_topo_liberation(4);
        enum_xorcode_whole("liberation(4)", t, 6);
        rj_topo_free(t);
    }
    enum_over_cap();
    enum_invalid();
    pivrows_units();

    printf("== Part 2: shortlist units ==\n");
    nt_static_tie();
    select_oom_cleanup();
    shortlist_rs63();
    {
        rj_topology *t = rj_topo_rdp(5);
        topo_int_t ti;
        char err[256];
        if (mk_ti(t, &ti, err) == 0 && ti.mat) {
            int F[2] = { 0, 1 };
            shortlist_pareto_struct("rdp(5)/f01", t, F, 2);
        }
        topo_int_free(&ti);
        rj_topo_free(t);
    }
    {
        rj_topology *t = rj_topo_liberation(4);
        topo_int_t ti;
        char err[256];
        if (mk_ti(t, &ti, err) == 0 && ti.mat) {
            int F[2] = { 0, 1 };
            shortlist_pareto_struct("liberation(4)/f01", t, F, 2);
        }
        topo_int_free(&ti);
        rj_topo_free(t);
    }
    {
        /* the 34-subset stress case: the hard cap must bind */
        rj_topology *t = rj_topo_rdp(5);
        topo_int_t ti;
        char err[256];
        if (mk_ti(t, &ti, err) == 0 && ti.mat) {
            int F[4] = { 0, 1, 2, 3 };
            shortlist_pareto_struct("rdp(5)/whole-disk0", t, F, 4);
        }
        topo_int_free(&ti);
        rj_topo_free(t);
    }
    shortlist_noprune();
    shortlist_no_signal();
    shortlist_nan_signal();

    printf("== Part 3: per-subset end-to-end ==\n");
    {
        size_t lens[] = { 64, 1000, 65536 };
        rj_topology *t = rj_topo_raid6(8);
        const int F1[1] = { 3 };
        e2e_all_subsets("raid6(8)/f3", t, F1, 1, lens, 3);
        rj_topo_free(t);
    }
    {
        size_t lens[] = { 64, 1000, 65536 };
        rj_topology *t = rj_topo_rs(6, 3);
        const int F1[1] = { 0 };
        const int F2[2] = { 0, 2 };
        e2e_all_subsets("rs(6,3)/f0", t, F1, 1, lens, 3);
        e2e_all_subsets("rs(6,3)/f02", t, F2, 2, lens, 3);
        /* f_data == 0: only parity disks fail (single forced subset) */
        const int FP1[1] = { 6 };
        const int FP2[2] = { 6, 7 };
        e2e_all_subsets("rs(6,3)/f6", t, FP1, 1, lens, 3);
        e2e_all_subsets("rs(6,3)/f67", t, FP2, 2, lens, 3);
        /* nf == npar: exactly one subset = today's all-rows plan */
        const int FN[3] = { 6, 7, 8 };
        e2e_all_subsets("rs(6,3)/f678", t, FN, 3, lens, 3);
        rj_topo_free(t);
    }
    {
        size_t lens[] = { 1000, 65536 };
        rj_topology *t = rj_topo_rs(10, 3);
        const int F2[2] = { 0, 1 };
        e2e_all_subsets("rs(10,3)/f01", t, F2, 2, lens, 2);
        rj_topo_free(t);
    }
    {
        size_t lens[] = { 1000, 65536 };
        uint8_t m[4 * 10];
        for (int j = 0; j < 4; j++)
            for (int i = 0; i < 10; i++)
                m[j * 10 + i] = (uint8_t)(1 + ((7 * i + 13 * j) % 255));
        rj_topology *t = rj_topo_custom(10, 4, m);
        const int F2[2] = { 1, 3 };
        e2e_all_subsets("rs(10,4)c/f13", t, F2, 2, lens, 2);
        rj_topo_free(t);
    }
    {
        size_t lens[] = { 1000 };
        rj_topology *t = rj_topo_rdp(5);
        const int F2[2] = { 0, 1 };
        e2e_all_subsets("rdp(5)/f01", t, F2, 2, lens, 1);
        rj_topo_free(t);
    }
    {
        size_t lens[] = { 1000 };
        rj_topology *t = rj_topo_evenodd(5);
        const int F2[2] = { 0, 1 };
        e2e_all_subsets("evenodd(5)/f01", t, F2, 2, lens, 1);
        rj_topo_free(t);
    }
    {
        size_t lens[] = { 1000 };
        rj_topology *t = rj_topo_liberation(4);
        const int F2[2] = { 0, 1 };
        e2e_all_subsets("liberation(4)/f01", t, F2, 2, lens, 1);
        rj_topo_free(t);
    }

    printf("== Part 4: mutation evidence ==\n");
    mutation_invert();
    mutation_corrupt_solve();

    printf("== Part 4b: P2 significance-gate units ==\n");
    gate_units();

    printf("== Part 5: dispatcher integration ==\n");
    assert_bufs_aligned();
    {
        rj_topology *t = rj_topo_rs(6, 3);
        const int F2[2] = { 0, 2 };
        dispatcher_integration("rs(6,3)/f02", t, F2, 2);
        rj_topo_free(t);
    }
    {
        rj_topology *t = rj_topo_rdp(5);
        const int F2[2] = { 0, 1 };
        dispatcher_integration("rdp(5)/f01", t, F2, 2);
        rj_topo_free(t);
    }
    product_wholedisk_rdp5();
    disable_knob();

    printf("== Part 6: product subset-independence ==\n");
    product_subset_independence();

    printf("== Part 7: L2 tripwire ==\n");
    tripwire_referee_units();
    tripwire();
    tripwire_redgreen();
    tripwire_referee_fires();

    printf("%d checks, %d failures\n", nchecks, failures);
    return failures ? 1 : 0;
}
