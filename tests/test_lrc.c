/* test_lrc: LRC topology constructors (A6 step 1).
 *
 * Coverage:
 *  1. constructor validation: bad parameters -> NULL (k<2, h<1, r<2,
 *     k>51, rj_topo_lrc_optimal with r not dividing k); good parameters
 *     -> the documented (ndata, npar, name); matrix structure (global
 *     rows = Vandermonde exponents 1..h recomputed independently;
 *     local rows = disjoint XOR supports partitioning the data disks,
 *     each of size <= r).
 *  2. encode bit-exact vs the independent scalar reference (azure,
 *     lrc(8,2,4), lrc(10,2,6) -- the last exercises the non-divisible
 *     general case), at boundary/tail lengths.
 *  3. local-recovery selection (the Step-1 verification): for a single
 *     failure inside a group, the survivor machinery's exact static
 *     signal strictly prefers the local-parity subset over every
 *     global-parity subset (fewer streams, pure XOR); the shortlist
 *     keeps it; survivor_select on the static (benchmark-off) path
 *     reports win_rows == the local row; the rj_compile plan ID names
 *     the local row. Same story for one failure per group (both local
 *     rows win) and for 2 failures in one group (the local row is in
 *     the winning set). Bit-exact recovery for each pattern.
 *  4. multi-failure correctness incl. beyond-capability: h+1 failures
 *     in one group recover bit-exactly (local+global rows); h+2 in one
 *     group is not recoverable (compile fails loudly).
 *  5. distance-optimality, verified exhaustively per instance: every
 *     (h+1)-erasure pattern is recoverable (survivor_enumerate finds a
 *     subset), and a concrete (h+2)-pattern is not -- i.e. d = h+2,
 *     the Singleton-type bound. Instances: (12,2,6), (8,2,4), (9,3,3).
 *  6. static cost table (the Step-1 measurement): per-subset (rows,
 *     kind, ncols, l2_uops, l2_mem) printed for the single-failure
 *     case on azure and lrc(20,2,4).
 *  7. fallback behavior (also red-green evidence): a tampered topology
 *     with the local row zeroed no longer picks the local path -- the
 *     winner is a global row -- while recovery stays bit-exact. A
 *     corrupted encode matrix is caught by the bit-exact check.
 *
 * No benchmarks anywhere: static cost model only, per the A6 brief
 * (benchmarks postponed). Every CHECK is on a deterministic path, so
 * native and ASan builds must report the same check count (C5e).
 *
 * Portable: runs natively on x86-64 and under QEMU for aarch64/riscv64
 * (uses rj_cpu_detect for the build CPU).
 */

#include "../include/raidjit.h"
#include "../src/plan.h"
#include "../src/gf.h"
#include "../src/ref.h"
#include "../src/survivor.h"
#include "../src/jit.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
static long nchecks = 0;
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

static rj_options topts; /* benchmark off: correctness + static model only */

static const size_t lens[] = { 0, 1, 7, 16, 17, 64, 100, 1000 };
static const int nlens = sizeof(lens) / sizeof(lens[0]);

static uint64_t rng_state = 0x51ab3c9d77f2e105ULL;
static uint64_t xrng(void)
{
    uint64_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return rng_state = x;
}

static void *abuf(size_t n)
{
    void *p = NULL;
    if (posix_memalign(&p, 64, n ? n : 64)) {
        fprintf(stderr, "oom\n");
        exit(1);
    }
    return p;
}

static void randfill(void *p, size_t n)
{
    uint8_t *b = p;
    for (size_t i = 0; i < n; i++)
        b[i] = (uint8_t)xrng();
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

static int mk_ti(rj_topology *t, topo_int_t *ti, char *err)
{
    return topo_int_build(ti, t, err, 256);
}

/* ---------------- 1. constructor validation ---------------- */

static void ctor_validation(void)
{
    /* bad parameters -> NULL */
    CHECK(rj_topo_lrc(1, 2, 6) == NULL, "lrc k=1 accepted");
    CHECK(rj_topo_lrc(12, 0, 6) == NULL, "lrc h=0 accepted");
    CHECK(rj_topo_lrc(12, 2, 1) == NULL, "lrc r=1 accepted");
    CHECK(rj_topo_lrc(12, 2, 0) == NULL, "lrc r=0 accepted");
    CHECK(rj_topo_lrc(52, 2, 6) == NULL, "lrc k=52 accepted (Vandermonde "
          "nodes 2^i repeat past order 51)");
    CHECK(rj_topo_lrc_optimal(10, 2, 6) == NULL, "lrc_optimal r∤k "
          "accepted");
    CHECK(rj_topo_lrc_optimal(12, 2, 0) == NULL, "lrc_optimal r=0 "
          "accepted");
    CHECK(rj_topo_lrc_optimal(12, 2, 1) == NULL, "lrc_optimal r=1 "
          "accepted");

    /* azure: the canonical instance */
    {
        rj_topology *t = rj_topo_lrc_azure();
        CHECK(t != NULL, "azure ctor NULL");
        if (!t)
            return;
        CHECK(rj_topo_ndata(t) == 12, "azure ndata=%d",
              rj_topo_ndata(t));
        CHECK(rj_topo_npar(t) == 4, "azure npar=%d", rj_topo_npar(t));
        CHECK(strcmp(topo_internal_name(t), "lrc-azure(12,2,2)") == 0,
              "azure name=%s", topo_internal_name(t));
        rj_topo_free(t);
    }
    /* general + optimal naming and dimensions */
    {
        rj_topology *t = rj_topo_lrc(10, 2, 6);
        CHECK(t != NULL, "lrc(10,2,6) ctor NULL");
        if (t) {
            /* g = ceil(10/6) = 2 groups -> npar = 2 + 2 */
            CHECK(rj_topo_ndata(t) == 10 && rj_topo_npar(t) == 4,
                  "lrc(10,2,6) dims %d+%d", rj_topo_ndata(t),
                  rj_topo_npar(t));
            CHECK(strcmp(topo_internal_name(t), "lrc(10,2,6)") == 0,
                  "lrc name=%s", topo_internal_name(t));
            rj_topo_free(t);
        }
        t = rj_topo_lrc_optimal(20, 2, 4);
        CHECK(t != NULL, "lrc_optimal(20,2,4) ctor NULL");
        if (t) {
            CHECK(rj_topo_ndata(t) == 20 && rj_topo_npar(t) == 7,
                  "lrc-opt(20,2,4) dims %d+%d", rj_topo_ndata(t),
                  rj_topo_npar(t));
            CHECK(strcmp(topo_internal_name(t), "lrc-opt(20,2,4)") == 0,
                  "lrc-opt name=%s", topo_internal_name(t));
            rj_topo_free(t);
        }
    }
}

/* matrix structure, recomputed independently of the constructor */
static void matrix_structure(int k, int h, int r, rj_topology *(*ctor)(void),
                             const char *tag)
{
    (void)k;
    (void)h;
    (void)r;
    rj_topology *t = ctor();
    CHECK(t != NULL, "%s ctor NULL", tag);
    if (!t)
        return;
    topo_int_t ti;
    char err[256];
    CHECK(mk_ti(t, &ti, err) == 0, "%s topo: %s", tag, err);
    if (ti.mat) {
        int kk = ti.ndata, np = ti.npar;
        int g = np - h;
        CHECK(g >= 1 && h >= 1, "%s bad split h=%d g=%d", tag, h, g);
        /* global rows: Vandermonde exponents 1..h, nodes 2^i */
        for (int j = 0; j < h; j++) {
            uint8_t step = gf_pow(2, (unsigned)(j + 1)), v = 1;
            for (int i = 0; i < kk; i++) {
                CHECK(ti.mat[(size_t)j * kk + i] == v,
                      "%s global row %d col %d: %02x != %02x", tag, j, i,
                      ti.mat[(size_t)j * kk + i], v);
                v = gf_mul(v, step);
            }
        }
        /* local rows: 0/1, disjoint supports, partition of the data
         * disks, each of size <= r and >= 1 */
        int *owner = calloc((size_t)kk, sizeof(int));
        CHECK(owner != NULL, "%s oom", tag);
        if (owner) {
            for (int gi = 0; gi < g; gi++) {
                const uint8_t *row = ti.mat + (size_t)(h + gi) * kk;
                int sz = 0;
                for (int i = 0; i < kk; i++) {
                    uint8_t c = row[i];
                    CHECK(c == 0 || c == 1, "%s local row %d col %d "
                          "nonbinary %02x", tag, gi, i, c);
                    if (c) {
                        CHECK(owner[i] == 0, "%s col %d in two groups",
                              tag, i);
                        owner[i] = gi + 1;
                        sz++;
                    }
                }
                CHECK(sz >= 1 && sz <= r, "%s group %d size %d (r=%d)",
                      tag, gi, sz, r);
            }
            for (int i = 0; i < kk; i++)
                CHECK(owner[i] != 0, "%s col %d in no group", tag, i);
            free(owner);
        }
        topo_int_free(&ti);
    }
    rj_topo_free(t);
}

static rj_topology *mk_azure(void) { return rj_topo_lrc_azure(); }
static rj_topology *mk_842(void) { return rj_topo_lrc(8, 2, 4); }

/* ---------------- 2. encode bit-exact ---------------- */

static void test_encode(rj_topology *t, const char *name)
{
    topo_int_t ti;
    char err[256];
    CHECK(mk_ti(t, &ti, err) == 0, "%s topo: %s", name, err);
    if (!ti.mat && ti.npar > 0)
        return;
    int nd = ti.ndata, np = ti.npar;
    for (int li = 0; li < nlens; li++) {
        size_t len = lens[li];
        uint8_t **d = malloc(sizeof(*d) * (size_t)nd);
        uint8_t **p = malloc(sizeof(*p) * (size_t)np);
        uint8_t **pr = malloc(sizeof(*pr) * (size_t)np);
        for (int i = 0; i < nd; i++) {
            d[i] = abuf(len);
            randfill(d[i], len);
        }
        for (int j = 0; j < np; j++) {
            p[j] = abuf(len);
            pr[j] = abuf(len);
        }
        ref_encode(ti.mat, nd, np, d, pr, len);
        rj_op op = rj_op_encode();
        rj_jit *j = rj_compile(t, &op, &topts, err, sizeof(err));
        CHECK(j != NULL, "%s encode len=%zu compile: %s", name, len, err);
        if (j) {
            const void **in = malloc(sizeof(*in) * (size_t)nd);
            void **out = malloc(sizeof(*out) * (size_t)np);
            for (int i = 0; i < nd; i++)
                in[i] = d[i];
            for (int j2 = 0; j2 < np; j2++)
                out[j2] = p[j2];
            rj_get_fn(j)(in, out, len);
            for (int j2 = 0; j2 < np; j2++)
                CHECK(memcmp(p[j2], pr[j2], len) == 0,
                      "%s encode len=%zu par%d mismatch", name, len, j2);
            rj_free(j);
            free(in);
            free(out);
        }
        rj_op_free(&op);
        for (int i = 0; i < nd; i++)
            free(d[i]);
        for (int j2 = 0; j2 < np; j2++) {
            free(p[j2]);
            free(pr[j2]);
        }
        free(d);
        free(p);
        free(pr);
    }
    topo_int_free(&ti);
    printf("ok encode %s\n", name);
}

/* ---------------- recover helpers ---------------- */

/* bit-exact recovery of `failed[0..nfailed)` via rj_compile; returns
 * the unit's plan id in idbuf ("" when compile failed and !expect_ok).
 * When expect_ok, the recovered buffers must match the pre-wipe keep
 * copies, at several boundary/tail lengths (one compile per pattern;
 * len varies only at run time). */
static const size_t rlens[] = { 1, 17, 64, 1000 };
static const int nrlens = sizeof(rlens) / sizeof(rlens[0]);

static void test_recover(rj_topology *t, const char *name, const int *failed,
                         int nfailed, int expect_ok, char *idbuf,
                         size_t idlen)
{
    if (idbuf && idlen)
        idbuf[0] = '\0';
    topo_int_t ti;
    char err[256];
    CHECK(mk_ti(t, &ti, err) == 0, "%s topo: %s", name, err);
    if (!ti.mat && ti.npar > 0)
        return;
    int nd = ti.ndata, np = ti.npar, N = nd + np;

    int fs[RJ_MAX_DISKS];
    memcpy(fs, failed, sizeof(int) * (size_t)nfailed);
    /* insertion sort (nfailed tiny) */
    for (int i = 1; i < nfailed; i++) {
        int v = fs[i], j = i - 1;
        while (j >= 0 && fs[j] > v) {
            fs[j + 1] = fs[j];
            j--;
        }
        fs[j + 1] = v;
    }

    rj_op op = rj_op_recover(failed, nfailed);
    rj_jit *j = rj_compile(t, &op, &topts, err, sizeof(err));
    if (!expect_ok) {
        CHECK(j == NULL, "%s recover(%d failed): expected failure, "
              "got plan %s", name, nfailed,
              j ? rj_plan_id(j) : "?");
        if (j)
            rj_free(j);
        rj_op_free(&op);
        topo_int_free(&ti);
        return;
    }
    CHECK(j != NULL, "%s recover(%d failed) compile: %s", name,
          nfailed, err);
    if (!j) {
        rj_op_free(&op);
        topo_int_free(&ti);
        return;
    }
    if (idbuf && idlen) {
        const char *id = rj_plan_id(j);
        snprintf(idbuf, idlen, "%s", id ? id : "?");
    }
    for (int li = 0; li < nrlens; li++) {
        size_t len = rlens[li];
        uint8_t **all = malloc(sizeof(*all) * (size_t)N);
        uint8_t **keep = malloc(sizeof(*keep) * (size_t)N);
        for (int i = 0; i < N; i++) {
            all[i] = abuf(len);
            keep[i] = abuf(len);
            if (i < nd)
                randfill(all[i], len);
        }
        ref_encode(ti.mat, nd, np, all, all + nd, len);
        for (int i = 0; i < N; i++)
            memcpy(keep[i], all[i], len);
        for (int f = 0; f < nfailed; f++)
            memset(all[fs[f]], 0xA5, len);
        const void **in = malloc(sizeof(*in) * (size_t)(N - nfailed));
        void **out = malloc(sizeof(*out) * (size_t)nfailed);
        int ni = 0, fi = 0, nf = 0;
        for (int dsk = 0; dsk < N; dsk++) {
            if (fi < nfailed && fs[fi] == dsk) {
                out[nf++] = all[dsk];
                fi++;
            } else {
                in[ni++] = all[dsk];
            }
        }
        rj_get_fn(j)(in, out, len);
        for (int f = 0; f < nfailed; f++)
            CHECK(memcmp(all[fs[f]], keep[fs[f]], len) == 0,
                  "%s recover disk %d len=%zu mismatch (id %s)", name,
                  fs[f], len, rj_plan_id(j));
        free(in);
        free(out);
        for (int i = 0; i < N; i++) {
            free(all[i]);
            free(keep[i]);
        }
        free(all);
        free(keep);
    }
    rj_free(j);
    rj_op_free(&op);
    topo_int_free(&ti);
}

/* survivor_select on the static path; returns win_rows[0..nf). */
static void static_win_rows(rj_topology *t, const char *tag, const int *F,
                            int nf, int *win_rows)
{
    topo_int_t ti;
    char err[512];
    CHECK(mk_ti(t, &ti, err) == 0, "%s topo: %s", tag, err);
    if (!ti.mat && ti.npar > 0)
        return;
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    version_t *vs = NULL;
    int nv = 0;
    plan_t *plans = NULL;
    int nplans = 0;
    int rows[RJ_MAX_DISKS];
    int w = survivor_select(&ti, F, nf, tag, &cpu, &topts, &vs, &nv,
                            &plans, &nplans, rows, err, sizeof(err));
    CHECK(w >= 0 && w < nv, "%s survivor_select: w=%d nv=%d %s", tag, w,
          nv, err);
    if (w >= 0) {
        memcpy(win_rows, rows, sizeof(int) * (size_t)nf);
        for (int i = 0; i < nv; i++)
            jit_free_version(&vs[i]);
        free(vs);
        for (int i = 0; i < nplans; i++)
            plan_free(&plans[i]);
        free(plans);
    }
    topo_int_free(&ti);
}

/* ---------------- 3. local-recovery selection ---------------- */

/* the exact static signal per subset + the dominance assertions */
static void local_pick_static(rj_topology *t, const char *tag, int k, int h,
                              int r, int fail_disk, int local_row)
{
    (void)r;
    topo_int_t ti;
    char err[512];
    CHECK(mk_ti(t, &ti, err) == 0, "%s topo: %s", tag, err);
    if (!ti.mat)
        return;
    (void)k;
    int F[1] = { fail_disk };
    survivor_subset_t *subs = NULL;
    int nsub = 0;
    int rc = survivor_enumerate(&ti, F, 1, tag, &subs, &nsub, err,
                                sizeof(err));
    /* subsets: one per surviving parity row with a nonzero coefficient
     * on the failed disk: the h globals + the one local row of the
     * failed disk's group (other groups' locals are singular). */
    CHECK(rc == 0 && nsub == h + 1, "%s enumerate: rc=%d nsub=%d "
          "want %d", tag, rc, nsub, h + 1);
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    survivor_l2_signal(subs, nsub, &cpu, &topts);

    int ilocal = -1;
    for (int i = 0; i < nsub; i++) {
        CHECK(subs[i].nrows == 1, "%s sub %d nrows=%d", tag, i,
              subs[i].nrows);
        if (subs[i].nrows == 1 && subs[i].rows[0] == local_row)
            ilocal = i;
    }
    CHECK(ilocal >= 0, "%s: local row %d not among subsets", tag,
          local_row);
    if (ilocal >= 0) {
        /* the local subset's plan is pure XOR over the group */
        CHECK(subs[ilocal].plan.kind == P_XOR, "%s local subset kind=%d "
              "want P_XOR", tag, (int)subs[ilocal].plan.kind);
        CHECK(subs[ilocal].l2_valid, "%s local subset has no L2 signal",
              tag);
        /* strict static dominance over every global subset: fewer
         * streams AND cheaper arithmetic. This is the mechanism by
         * which the existing cost model prefers local recovery with
         * no LRC-specific logic. */
        for (int i = 0; i < nsub; i++) {
            if (i == ilocal)
                continue;
            if (!subs[i].l2_valid)
                continue; /* never-dominated; can only keep, not drop */
            CHECK(subs[ilocal].l2_uops < subs[i].l2_uops,
                  "%s local uops %.4f not < global sub{%d} %.4f", tag,
                  subs[ilocal].l2_uops, subs[i].rows[0],
                  subs[i].l2_uops);
            CHECK(subs[ilocal].l2_mem < subs[i].l2_mem,
                  "%s local mem %.4f not < global sub{%d} %.4f", tag,
                  subs[ilocal].l2_mem, subs[i].rows[0],
                  subs[i].l2_mem);
        }
        int *keep = calloc((size_t)nsub, sizeof(int));
        CHECK(keep != NULL, "%s oom", tag);
        if (keep) {
            survivor_shortlist(subs, nsub, keep);
            CHECK(keep[ilocal], "%s: shortlist dropped the local subset",
                  tag);
            free(keep);
        }
    }
    printf("%-28s fail=%2d local_row=%d kind(local)=%d\n", tag, fail_disk,
           local_row,
           ilocal >= 0 ? (int)subs[ilocal].plan.kind : -1);
    survivor_free_subsets(subs, nsub);
    topo_int_free(&ti);
}

/* ---------------- 6. static cost table ---------------- */

static const char *kindname(plan_kind_t k)
{
    switch (k) {
    case P_XOR: return "xor";
    case P_MULACC: return "mulacc";
    case P_HORNER: return "horner";
    case P_DELTA: return "delta";
    }
    return "?";
}

static void cost_table(rj_topology *t, const char *tag, int fail_disk)
{
    topo_int_t ti;
    char err[512];
    if (mk_ti(t, &ti, err) != 0) {
        printf("cost_table %s: %s\n", tag, err);
        return;
    }
    int F[1] = { fail_disk };
    survivor_subset_t *subs = NULL;
    int nsub = 0;
    if (survivor_enumerate(&ti, F, 1, tag, &subs, &nsub, err,
                           sizeof(err)) != 0) {
        printf("cost_table %s: %s\n", tag, err);
        topo_int_free(&ti);
        return;
    }
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    survivor_l2_signal(subs, nsub, &cpu, &topts);
    printf("static cost, %s, single failure of disk %d "
           "(uops/B, mem B/B of the canonical version):\n", tag,
           fail_disk);
    printf("  rows  kind     ncols  nnz   uops_per_byte  mem_per_byte\n");
    for (int i = 0; i < nsub; i++) {
        printf("  {%-3d} %-7s %5d %5ld %13.4f %12.4f%s\n",
               subs[i].rows[0], kindname(subs[i].plan.kind),
               subs[i].ncols, subs[i].nnz, subs[i].l2_uops,
               subs[i].l2_mem, subs[i].l2_valid ? "" : "  (no signal)");
    }
    survivor_free_subsets(subs, nsub);
    topo_int_free(&ti);
}

/* ---------------- 5. distance-optimality ----------------
 * Exhaustive: every (h+1)-erasure pattern must be recoverable
 * (survivor_enumerate returns a subset); the witness (h+2)-pattern
 * must not be. */

/* next lexicographic combination of m out of n (comb[0..m)); 0 when
 * exhausted. */
static int next_comb(int *comb, int m, int n)
{
    int i = m - 1;
    while (i >= 0 && comb[i] == n - m + i)
        i--;
    if (i < 0)
        return 0;
    comb[i]++;
    for (int j = i + 1; j < m; j++)
        comb[j] = comb[j - 1] + 1;
    return 1;
}

static void distance_optimal(rj_topology *t, const char *tag, int k, int h,
                             const int *witness, int nwit)
{
    (void)k;
    topo_int_t ti;
    char err[512];
    CHECK(mk_ti(t, &ti, err) == 0, "%s topo: %s", tag, err);
    if (!ti.mat)
        return;
    int n = ti.ndata + ti.npar;
    int m = h + 1;
    int *comb = malloc(sizeof(int) * (size_t)m);
    CHECK(comb != NULL, "%s oom", tag);
    long npat = 0;
    if (comb) {
        for (int i = 0; i < m; i++)
            comb[i] = i;
        do {
            survivor_subset_t *subs = NULL;
            int nsub = 0;
            int rc = survivor_enumerate(&ti, comb, m, tag, &subs, &nsub,
                                        err, sizeof(err));
            /* print only failures; the count is pinned below */
            if (!(rc == 0 && nsub >= 1)) {
                printf("FAIL %s: pattern (", tag);
                for (int i = 0; i < m; i++)
                    printf("%d%s", comb[i], i + 1 < m ? "," : "");
                printf(") not recoverable: rc=%d %s\n", rc, err);
                failures++;
            }
            nchecks++;
            npat++;
            survivor_free_subsets(subs, nsub);
        } while (next_comb(comb, m, n));
        free(comb);
    }
    /* the witness (h+2)-pattern is NOT recoverable -> d <= h+2;
     * together with the exhaustive pass, d = h+2 = the bound. */
    {
        survivor_subset_t *subs = NULL;
        int nsub = 0;
        int rc = survivor_enumerate(&ti, witness, nwit, tag, &subs,
                                    &nsub, err, sizeof(err));
        CHECK(rc != 0, "%s: witness pattern recoverable (rc=%d) -- "
              "distance exceeds h+2?", tag, rc);
        survivor_free_subsets(subs, nsub);
    }
    printf("ok distance %s: %ld %d-erasure patterns recoverable, "
           "witness not\n", tag, npat, m);
    topo_int_free(&ti);
}

/* ---------------- main ---------------- */

int main(void)
{
    rj_options_default(&topts);
    topts.benchmark = 0; /* static model only; benchmarks postponed */

    /* 1. constructors */
    ctor_validation();
    matrix_structure(12, 2, 6, mk_azure, "azure");
    matrix_structure(8, 2, 4, mk_842, "lrc(8,2,4)");

    /* 2. encode bit-exact */
    {
        rj_topology *t = rj_topo_lrc_azure();
        test_encode(t, "lrc-azure(12,2,2)");
        rj_topo_free(t);
        t = rj_topo_lrc(8, 2, 4);
        test_encode(t, "lrc(8,2,4)");
        rj_topo_free(t);
        t = rj_topo_lrc(10, 2, 6);
        test_encode(t, "lrc(10,2,6)");
        rj_topo_free(t);
    }
    /* corrupted-matrix red-green: the bit-exact check must catch it */
    {
        rj_topology *t = rj_topo_lrc_azure();
        topo_int_t ti;
        char err[256];
        if (t && mk_ti(t, &ti, err) == 0 && ti.mat) {
            /* corrupt the CUSTOM topo's matrix; the reference keeps
             * the original, so a missed mismatch is a live CHECK */
            uint8_t *bad = malloc(12u * 4u);
            if (bad) {
                memcpy(bad, ti.mat, 12u * 4u);
                bad[0] ^= 0xFF; /* corrupt one global coefficient */
                rj_topology *tc = rj_topo_custom(12, 4, bad);
                if (tc) {
                size_t len = 64;
                uint8_t **d = malloc(sizeof(*d) * 12);
                uint8_t **p = malloc(sizeof(*p) * 4);
                uint8_t **pr = malloc(sizeof(*pr) * 4);
                for (int i = 0; i < 12; i++) {
                    d[i] = abuf(len);
                    randfill(d[i], len);
                }
                for (int j = 0; j < 4; j++) {
                    p[j] = abuf(len);
                    pr[j] = abuf(len);
                }
                ref_encode(ti.mat, 12, 4, d, pr, len);
                rj_op op = rj_op_encode();
                rj_jit *j = rj_compile(tc, &op, &topts, err, sizeof(err));
                int differ = 0;
                if (j) {
                    const void **in = malloc(sizeof(*in) * 12);
                    void **out = malloc(sizeof(*out) * 4);
                    for (int i = 0; i < 12; i++)
                        in[i] = d[i];
                    for (int j2 = 0; j2 < 4; j2++)
                        out[j2] = p[j2];
                    rj_get_fn(j)(in, out, len);
                    for (int j2 = 0; j2 < 4; j2++)
                        if (memcmp(p[j2], pr[j2], len) != 0)
                            differ = 1;
                    rj_free(j);
                    free(in);
                    free(out);
                } else {
                    differ = 1; /* failed loudly: also acceptable */
                }
                CHECK(differ, "corrupted-matrix encode not caught");
                rj_op_free(&op);
                for (int i = 0; i < 12; i++)
                    free(d[i]);
                for (int j2 = 0; j2 < 4; j2++) {
                    free(p[j2]);
                    free(pr[j2]);
                }
                free(d);
                free(p);
                free(pr);
                rj_topo_free(tc);
                }
                free(bad);
            }
            topo_int_free(&ti);
        }
        rj_topo_free(t);
    }

    /* 3. local-recovery selection. Azure layout: data 0..11, globals
     * rows 0..1, locals row 2 (group 0: disks 0..5) and row 3 (group
     * 1: disks 6..11). */
    {
        rj_topology *t = rj_topo_lrc_azure();
        /* static signal prefers the local subset, strictly, both axes */
        local_pick_static(t, "azure/f0", 12, 2, 6, 0, 2);
        local_pick_static(t, "azure/f6", 12, 2, 6, 6, 3);
        local_pick_static(t, "azure/f11", 12, 2, 6, 11, 3);
        /* the static path's winner is the local row */
        int wr[RJ_MAX_DISKS];
        static_win_rows(t, "azure/f0", (int[]){ 0 }, 1, wr);
        CHECK(wr[0] == 2, "azure/f0 static winner rows={%d} want {2}",
              wr[0]);
        static_win_rows(t, "azure/f11", (int[]){ 11 }, 1, wr);
        CHECK(wr[0] == 3, "azure/f11 static winner rows={%d} want {3}",
              wr[0]);
        /* one failure per group: both local rows win */
        static_win_rows(t, "azure/f0+f6", (int[]){ 0, 6 }, 2, wr);
        CHECK(wr[0] == 2 && wr[1] == 3,
              "azure/f0+f6 static winner rows={%d,%d} want {2,3}",
              wr[0], wr[1]);
        /* two failures in one group: the local row is in the set */
        static_win_rows(t, "azure/f0+f1", (int[]){ 0, 1 }, 2, wr);
        CHECK(wr[0] == 2 || wr[1] == 2,
              "azure/f0+f1 static winner rows={%d,%d} lacks local 2",
              wr[0], wr[1]);
        /* plan-ID level: rj_compile names the winning local row */
        char id[4096];
        test_recover(t, "azure", (int[]){ 0 }, 1, 1, id, sizeof(id));
        CHECK(strstr(id, "rec0.rows2:") != NULL,
              "azure plan id lacks rec0.rows2: %s", id);
        test_recover(t, "azure", (int[]){ 11 }, 1, 1, id, sizeof(id));
        CHECK(strstr(id, "rec11.rows3:") != NULL,
              "azure plan id lacks rec11.rows3: %s", id);
        test_recover(t, "azure", (int[]){ 0, 6 }, 2, 1, id, sizeof(id));
        CHECK(strstr(id, "rec0+6.rows2+3:") != NULL,
              "azure plan id lacks rec0+6.rows2+3: %s", id);
        /* bit-exact recovery, more patterns */
        test_recover(t, "azure", (int[]){ 5 }, 1, 1, NULL, 0);
        test_recover(t, "azure", (int[]){ 0, 1 }, 2, 1, NULL, 0);
        test_recover(t, "azure", (int[]){ 0, 1, 2 }, 3, 1, NULL, 0);
        test_recover(t, "azure", (int[]){ 3, 9 }, 2, 1, NULL, 0);
        test_recover(t, "azure", (int[]){ 12 }, 1, 1, NULL,
                     0); /* failed global parity: recompute */
        test_recover(t, "azure", (int[]){ 14 }, 1, 1, NULL,
                     0); /* failed local parity: recompute */
        test_recover(t, "azure", (int[]){ 0, 12 }, 2, 1, NULL,
                     0); /* data + global parity */
        /* beyond capability: 4 data failures in one group */
        test_recover(t, "azure", (int[]){ 0, 1, 2, 3 }, 4, 0, NULL, 0);
        rj_topo_free(t);
    }
    /* second family instance: lrc(20,2,4), locality 4 */
    {
        rj_topology *t = rj_topo_lrc_optimal(20, 2, 4);
        /* groups of 4: row 2 = local of disks 0..3 */
        local_pick_static(t, "lrc-opt(20,2,4)/f0", 20, 2, 4, 0, 2);
        int wr[RJ_MAX_DISKS];
        static_win_rows(t, "lrc-opt(20,2,4)/f19", (int[]){ 19 }, 1, wr);
        /* disk 19 in group 4 (disks 16..19): local row 2+4 = 6 */
        CHECK(wr[0] == 6, "lrc-opt(20,2,4)/f19 winner rows={%d} want "
              "{6}", wr[0]);
        test_recover(t, "lrc-opt(20,2,4)", (int[]){ 19 }, 1, 1, NULL, 0);
        test_recover(t, "lrc-opt(20,2,4)", (int[]){ 0, 1, 2 }, 3, 1,
                     NULL, 0);
        rj_topo_free(t);
    }

    /* 5. distance-optimality, exhaustive per instance */
    {
        rj_topology *t = rj_topo_lrc_azure();
        distance_optimal(t, "lrc-azure(12,2,2)", 12, 2,
                         (int[]){ 0, 1, 2, 3 }, 4);
        rj_topo_free(t);
        t = rj_topo_lrc(8, 2, 4);
        distance_optimal(t, "lrc(8,2,4)", 8, 2, (int[]){ 0, 1, 2, 3 },
                         4);
        rj_topo_free(t);
        t = rj_topo_lrc_optimal(9, 3, 3);
        distance_optimal(t, "lrc-opt(9,3,3)", 9, 3,
                         (int[]){ 0, 1, 2, 9, 10 }, 5);
        rj_topo_free(t);
    }

    /* 7. fallback: zeroed local row -> the local path is gone, a
     * global row wins, recovery stays bit-exact */
    {
        rj_topology *t = rj_topo_lrc_azure();
        topo_int_t ti;
        char err[256];
        if (t && mk_ti(t, &ti, err) == 0 && ti.mat) {
            memset(ti.mat + 2 * 12, 0, 12); /* kill local row 2 */
            rj_topology *tc = rj_topo_custom(12, 4, ti.mat);
            if (tc) {
                int wr[RJ_MAX_DISKS];
                static_win_rows(tc, "tampered/f0", (int[]){ 0 }, 1, wr);
                CHECK(wr[0] == 0 || wr[0] == 1,
                      "tampered/f0 winner rows={%d}: local path must "
                      "be gone", wr[0]);
                test_recover(tc, "tampered", (int[]){ 0 }, 1, 1, NULL,
                             0);
                rj_topo_free(tc);
            }
            topo_int_free(&ti);
        }
        rj_topo_free(t);
    }

    /* 6. static cost tables (the Step-1 measurement) */
    {
        rj_topology *t = rj_topo_lrc_azure();
        cost_table(t, "lrc-azure(12,2,2)", 0);
        rj_topo_free(t);
        t = rj_topo_lrc_optimal(20, 2, 4);
        cost_table(t, "lrc-opt(20,2,4)", 0);
        rj_topo_free(t);
    }

    printf("lrc: %ld checks, %d failures\n", nchecks, failures);
    return failures != 0;
}
