/* A4 phase 1: closed-form small-erasure recovery (planner only).
 *
 * 1. Unit differential: plan_recover_closed_form() vs
 *    solve_recovery_rows() -- bit-identical R whenever the closed form
 *    applies (e=1/e=2, invertible pivot); returns 0 (fallback) for
 *    nf >= 3. Across RS/RAID-5/RAID-6/custom/XOR (RDP, EvenOdd)
 *    topologies, data+parity and parity+parity combos, single-row and
 *    row-pair subsets, all-rows, and an unsorted row order.
 * 2. Integration: plan_build_recover_rows() with
 *    RJ_RECOVER_NO_CLOSEDFORM on/off builds bit-identical plans (the
 *    closed form is what the general path computes).
 * 3. End-to-end: rj_compile() recover for e=1/e=2 combos; recovered
 *    bytes must equal the pre-failure bytes bit-exactly (differential
 *    vs the scalar ground truth), plus an e=3 fallback case.
 */
#include "../include/raidjit.h"
#include "../src/plan.h"
#include "../src/gf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, ...)                                                   \
    do {                                                                   \
        if (!(cond)) {                                                     \
            printf("FAIL %s:%d: ", __func__, __LINE__);                    \
            printf(__VA_ARGS__);                                           \
            printf("\n");                                                  \
            failures++;                                                    \
        }                                                                  \
    } while (0)

static uint64_t rng_state = 0x51ab3f7d2c9e0415ULL;
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

/* ---------------- 1. unit differential ---------------- */

static long n_cf_applied = 0, n_cf_fallback = 0, n_cf_total = 0;

/* One (topo, F, rows) differential case. */
static void diff_case(topo_int_t *ti, const char *tname, const int *F, int nf,
                      const int *rows, int nrows)
{
    int nd = ti->ndata, np = ti->npar;
    char err[256];
    (void)err;
    /* survivors, increasing */
    int ns = nd + np - nf;
    int *S = malloc((size_t)(ns > 0 ? ns : 1) * sizeof(int));
    int fi = 0, si = 0;
    for (int d = 0; d < nd + np; d++) {
        if (fi < nf && F[fi] == d)
            fi++;
        else
            S[si++] = d;
    }
    uint8_t *Rcf = malloc((size_t)(nf > 0 ? nf : 1) * (size_t)(ns > 0 ? ns : 1));
    uint8_t *Rgen = malloc((size_t)(nf > 0 ? nf : 1) * (size_t)(ns > 0 ? ns : 1));
    memset(Rcf, 0xA5, (size_t)nf * (size_t)(ns > 0 ? ns : 1));
    int cf = plan_recover_closed_form(ti, F, nf, S, ns, rows, nrows, Rcf);
    int gen =
        solve_recovery_rows(ti->mat, nd, np, F, nf, S, ns, rows, nrows,
                            Rgen, NULL);
    n_cf_total++;
    if (cf)
        n_cf_applied++;
    else
        n_cf_fallback++;
    char fb[48] = "", rb[48] = "";
    size_t o = 0;
    for (int i = 0; i < nf && o < sizeof(fb) - 6; i++)
        o += (size_t)snprintf(fb + o, sizeof(fb) - o, "%s%d", i ? "+" : "",
                              F[i]);
    o = 0;
    for (int i = 0; i < nrows && o < sizeof(rb) - 6; i++)
        o += (size_t)snprintf(rb + o, sizeof(rb) - o, "%s%d", i ? "+" : "",
                              rows[i]);
    /* The closed form must agree with the general solve on
     * applicability (for nf <= 2), and produce bit-identical rows when
     * it applies. */
    if (nf <= 2) {
        CHECK((cf != 0) == (gen == 0),
              "%s F={%s} rows={%s}: closed_form=%d general=%d (disagree)",
              tname, fb, rb, cf, gen);
        if (cf && gen == 0)
            CHECK(memcmp(Rcf, Rgen, (size_t)nf * (size_t)ns) == 0,
                  "%s F={%s} rows={%s}: R mismatch", tname, fb, rb);
    }
    /* nf >= 3 must always fall back (closed form is e<=2 only). */
    if (nf >= 3)
        CHECK(cf == 0, "%s F={%s}: closed form must not apply for nf=%d",
              tname, fb, nf);
    free(S);
    free(Rcf);
    free(Rgen);
}

static void add_uniq(int *arr, int *n, int v, int lo, int hi)
{
    if (v < lo || v >= hi)
        return;
    for (int i = 0; i < *n; i++)
        if (arr[i] == v)
            return;
    arr[(*n)++] = v;
}

/* e=1/e=2 differential sweep for one topology. */
static void diff_topo(rj_topology *t, const char *tname)
{
    topo_int_t ti;
    char err[256];
    CHECK(topo_int_build(&ti, t, err, sizeof(err)) == 0, "%s: %s", tname,
          err);
    if (failures)
        return;
    int nd = ti.ndata, np = ti.npar, N = nd + np;
    int allrows[256];
    for (int r = 0; r < np; r++)
        allrows[r] = r;

    /* e=1: first/last data disk, first/last parity disk */
    int f1[4], nf1 = 0;
    add_uniq(f1, &nf1, 0, 0, N);
    add_uniq(f1, &nf1, nd - 1, 0, N);
    add_uniq(f1, &nf1, nd, 0, N);
    add_uniq(f1, &nf1, nd + np - 1, 0, N);
    for (int i = 0; i < nf1; i++) {
        int F[1] = { f1[i] };
        diff_case(&ti, tname, F, 1, allrows, np);
        for (int r = 0; r < np; r++) {
            int rows[1] = { r };
            diff_case(&ti, tname, F, 1, rows, 1);
        }
    }

    /* e=2: data+data, data+parity, parity+parity */
    if (np >= 2) {
        int pairs[][2] = { {0, 1}, {0, nd - 1}, {1, nd - 1},
                           {0, nd}, {1, nd},   {nd - 1, nd},
                           {nd, nd + 1}, {0, nd + np - 1} };
        for (size_t pi = 0; pi < sizeof(pairs) / sizeof(pairs[0]); pi++) {
            int a = pairs[pi][0], b = pairs[pi][1];
            if (a < 0 || b < 0 || a >= N || b >= N || a == b)
                continue;
            int F[2] = { a < b ? a : b, a < b ? b : a };
            diff_case(&ti, tname, F, 2, allrows, np);
            for (int r1 = 0; r1 < np; r1++)
                for (int r2 = r1 + 1; r2 < np; r2++) {
                    int rows[2] = { r1, r2 };
                    diff_case(&ti, tname, F, 2, rows, 2);
                }
            /* unsorted row order: pivot replication must not depend on
             * rows[] being sorted */
            if (np >= 2) {
                int rows[2] = { 1, 0 };
                diff_case(&ti, tname, F, 2, rows, 2);
            }
        }
    }

    /* e=3: closed form must fall back (general path still solves it
     * when recoverable) */
    if (np >= 3) {
        int F[3] = { 0, 1, nd };
        diff_case(&ti, tname, F, 3, allrows, np);
    }
    topo_int_free(&ti);
}

/* ---------------- 2. integration: hook on/off ---------------- */

static void plan_cmp(const char *tname, const int *F, int nf, const int *rows,
                     int nrows)
{
    char err[256];
    rj_topology *t = NULL;
    if (!strcmp(tname, "rs(6,3)"))
        t = rj_topo_rs(6, 3);
    else
        t = rj_topo_raid6(8);
    topo_int_t ti;
    CHECK(topo_int_build(&ti, t, err, sizeof(err)) == 0, "%s", err);
    plan_t p_cf, p_gen;
    memset(&p_cf, 0, sizeof(p_cf));
    memset(&p_gen, 0, sizeof(p_gen));
    unsetenv("RJ_RECOVER_NO_CLOSEDFORM");
    int rc1 = plan_build_recover_rows(&ti, F, nf, rows, nrows, tname, &p_cf,
                                      err, sizeof(err));
    setenv("RJ_RECOVER_NO_CLOSEDFORM", "1", 1);
    int rc2 = plan_build_recover_rows(&ti, F, nf, rows, nrows, tname, &p_gen,
                                      err, sizeof(err));
    unsetenv("RJ_RECOVER_NO_CLOSEDFORM");
    CHECK(rc1 == 0 && rc2 == 0, "%s: build rc %d/%d: %s", tname, rc1, rc2,
          err);
    if (rc1 == 0 && rc2 == 0) {
        CHECK(p_cf.kind == p_gen.kind, "%s: kind %d vs %d", tname,
              p_cf.kind, p_gen.kind);
        CHECK(p_cf.n_in == p_gen.n_in && p_cf.n_out == p_gen.n_out,
              "%s: shape %dx%d vs %dx%d", tname, p_cf.n_in, p_cf.n_out,
              p_gen.n_in, p_gen.n_out);
        CHECK(memcmp(p_cf.mat, p_gen.mat,
                     (size_t)p_cf.n_in * (size_t)p_cf.n_out) == 0,
              "%s: plan matrix differs with hook on/off", tname);
        CHECK(!strcmp(p_cf.name, p_gen.name), "%s: name differs", tname);
    }
    plan_free(&p_cf);
    plan_free(&p_gen);
    topo_int_free(&ti);
    rj_topo_free(t);
}

/* ---------------- 3. end-to-end vs ground truth ---------------- */

static rj_options eopts;

static void e2e_recover(rj_topology *t, const char *tname, const int *failed,
                        int nfailed, const size_t *lens, int nlens)
{
    topo_int_t ti;
    char err[256];
    CHECK(topo_int_build(&ti, t, err, sizeof(err)) == 0, "%s topo: %s",
          tname, err);
    if (failures)
        return;
    int nd = ti.ndata, np = ti.npar, N = nd + np;
    int fs[256];
    memcpy(fs, failed, sizeof(int) * (size_t)nfailed);
    /* sort (insertion; nfailed is tiny) */
    for (int i = 1; i < nfailed; i++) {
        int k = fs[i], j = i - 1;
        while (j >= 0 && fs[j] > k) {
            fs[j + 1] = fs[j];
            j--;
        }
        fs[j + 1] = k;
    }
    for (int li = 0; li < nlens; li++) {
        size_t len = lens[li];
        uint8_t **all = malloc(sizeof(*all) * (size_t)N);
        uint8_t **keep = malloc(sizeof(*keep) * (size_t)N);
        for (int i = 0; i < N; i++) {
            all[i] = abuf(len);
            keep[i] = abuf(len);
            if (i < nd)
                randfill(all[i], len);
        }
        /* scalar encode = ground truth for the parities */
        for (int j = 0; j < np; j++)
            for (size_t k = 0; k < len; k++) {
                uint8_t a = 0;
                for (int i = 0; i < nd; i++) {
                    uint8_t c = ti.mat[(size_t)j * (size_t)nd + (size_t)i];
                    if (c)
                        a ^= gf_mul(all[i][k], c);
                }
                all[nd + j][k] = a;
            }
        for (int i = 0; i < N; i++)
            memcpy(keep[i], all[i], len);
        rj_op op = rj_op_recover(failed, nfailed);
        rj_jit *j = rj_compile(t, &op, &eopts, err, sizeof(err));
        CHECK(j != NULL, "%s recover len=%zu compile: %s", tname, len,
              err);
        if (!j) {
            rj_op_free(&op);
            goto cleanup;
        }
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
                  "%s recover disk%d len=%zu mismatch", tname, fs[f], len);
        rj_free(j);
        free(in);
        free(out);
        rj_op_free(&op);
    cleanup:
        for (int i = 0; i < N; i++) {
            free(all[i]);
            free(keep[i]);
        }
        free(all);
        free(keep);
    }
    topo_int_free(&ti);
    if (!failures)
        printf("ok e2e %s failed={", tname);
    else
        printf("FAIL e2e %s failed={", tname);
    for (int f = 0; f < nfailed; f++)
        printf("%s%d", f ? "," : "", failed[f]);
    printf("}\n");
}

/* ---------------- 4. randomized differential fuzz ---------------- */

/* Attacks the pivot-replication logic: random small matrices (often
 * singular), random failure sets, random row subsets in random order.
 * The closed form must agree with the general solve on applicability
 * and produce bit-identical rows whenever it applies; nf=3 must never
 * apply. */
static uint64_t frnd_state = 0x12345678abcdefULL;
static unsigned frnd(unsigned n)
{
    uint64_t s = frnd_state;
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    frnd_state = s;
    return (unsigned)(s % n);
}

static void fuzz_differential(void)
{
    long total = 0, applied = 0;
    for (int iter = 0; iter < 5000; iter++) {
        int nd = 2 + (int)frnd(5); /* 2..6 */
        int np = 1 + (int)frnd(3); /* 1..3 */
        int N = nd + np;
        uint8_t mat[3 * 6];
        for (int i = 0; i < np * nd; i++)
            mat[i] = (uint8_t)frnd(256);
        topo_int_t ti = { .ndata = nd, .npar = np, .mat = mat };
        int nf = 1 + (int)frnd(2); /* 1..2 */
        if (nf > np)
            continue;
        int F[2];
        F[0] = (int)frnd((unsigned)N);
        do {
            F[1] = (int)frnd((unsigned)N);
        } while (nf > 1 && F[1] == F[0]);
        if (nf == 2 && F[0] > F[1]) {
            int t = F[0];
            F[0] = F[1];
            F[1] = t;
        }
        int ns = N - nf;
        int S[8], fi = 0, si = 0;
        for (int d = 0; d < N; d++) {
            if (fi < nf && F[fi] == d)
                fi++;
            else
                S[si++] = d;
        }
        /* random row subset (nf..np rows) in random order */
        int perm[3];
        for (int i = 0; i < np; i++)
            perm[i] = i;
        for (int i = np - 1; i > 0; i--) {
            int j = (int)frnd((unsigned)(i + 1));
            int t = perm[i];
            perm[i] = perm[j];
            perm[j] = t;
        }
        int nrows = nf + (int)frnd((unsigned)(np - nf + 1));
        uint8_t Rcf[2 * 8], Rgen[2 * 8];
        memset(Rcf, 0xA5, sizeof(Rcf));
        int cf =
            plan_recover_closed_form(&ti, F, nf, S, ns, perm, nrows, Rcf);
        int gen = solve_recovery_rows(mat, nd, np, F, nf, S, ns, perm,
                                      nrows, Rgen, NULL);
        total++;
        CHECK((cf != 0) == (gen == 0),
              "fuzz iter %d: applicability disagree (cf=%d gen=%d)", iter,
              cf, gen);
        if (cf) {
            applied++;
            CHECK(memcmp(Rcf, Rgen, (size_t)nf * (size_t)ns) == 0,
                  "fuzz iter %d: R mismatch", iter);
        }
        if (failures)
            return;
        /* nf=3 must never apply */
        if (np >= 3) {
            int F3[3] = { 0, 1, 2 }, S3[8], f3 = 0, s3 = 0;
            int ns3 = N - 3;
            for (int d = 0; d < N; d++) {
                if (f3 < 3 && F3[f3] == d)
                    f3++;
                else
                    S3[s3++] = d;
            }
            uint8_t R3[3 * 8];
            CHECK(plan_recover_closed_form(&ti, F3, 3, S3, ns3, perm, np,
                                           R3) == 0,
                  "fuzz iter %d: nf=3 must fall back", iter);
            if (failures)
                return;
        }
    }
    printf("fuzz differential: %ld cases, %ld applied, all agree\n", total,
           applied);
    CHECK(applied > 0, "fuzz never applied (vacuous)");
}

int main(void)
{
    rj_options_default(&eopts);
    eopts.benchmark = 0;

    /* 1. unit differential across topologies */
    {
        rj_topology *t = rj_topo_raid5(4);
        diff_topo(t, "raid5(4)");
        rj_topo_free(t);
    }
    {
        rj_topology *t = rj_topo_raid6(8);
        diff_topo(t, "raid6(8)");
        rj_topo_free(t);
    }
    {
        rj_topology *t = rj_topo_rs(6, 3);
        diff_topo(t, "rs(6,3)");
        rj_topo_free(t);
    }
    {
        uint8_t cmat[8] = { 1, 1, 1, 1, 1, 2, 4, 8 };
        rj_topology *t = rj_topo_custom(4, 2, cmat);
        diff_topo(t, "custom(4,2)");
        rj_topo_free(t);
    }
    /* non-MDS-shaped custom matrix: exercises singular-pair fallback
     * inside the differential (both paths must agree on failure) */
    {
        uint8_t cmat[8] = { 1, 2, 3, 4, 2, 4, 6, 8 };
        rj_topology *t = rj_topo_custom(4, 2, cmat);
        diff_topo(t, "custom2(4,2)");
        rj_topo_free(t);
    }
    {
        rj_topology *t = rj_topo_rdp(5);
        diff_topo(t, "rdp(5)");
        rj_topo_free(t);
    }
    {
        rj_topology *t = rj_topo_evenodd(5);
        diff_topo(t, "evenodd(5)");
        rj_topo_free(t);
    }
    printf("closed-form differential: %ld cases, %ld applied, %ld fallback\n",
           n_cf_total, n_cf_applied, n_cf_fallback);
    CHECK(n_cf_applied > 0, "closed form never applied (vacuous test)");
    CHECK(n_cf_fallback > 0, "fallback never exercised (vacuous test)");

    /* 1b. randomized differential fuzz */
    fuzz_differential();

    /* 2. integration: hook on/off bit-identity */
    {
        int all3[3] = { 0, 1, 2 };
        int f25[2] = { 2, 5 };
        plan_cmp("rs(6,3)", f25, 2, all3, 3);
        int f7[1] = { 7 };
        plan_cmp("rs(6,3)", f7, 1, all3, 3);
        int all2[2] = { 0, 1 };
        int f9[1] = { 9 };
        plan_cmp("raid6(8)", f9, 1, all2, 2);
        int f07[2] = { 0, 7 };
        plan_cmp("raid6(8)", f07, 2, all2, 2);
        printf("ok integration hook on/off\n");
    }

    /* 3. end-to-end vs ground truth */
    static const size_t lens[] = { 0, 1, 63, 64, 65, 1000 };
    int nlens = (int)(sizeof(lens) / sizeof(lens[0]));
    {
        rj_topology *t = rj_topo_raid6(8);
        int f3[] = { 3 }, f8[] = { 8 }, f9[] = { 9 };
        int f15[] = { 1, 5 }, f28[] = { 2, 8 }, f49[] = { 4, 9 },
            f89[] = { 8, 9 };
        e2e_recover(t, "raid6(8)", f3, 1, lens, nlens);
        e2e_recover(t, "raid6(8)", f8, 1, lens, nlens);
        e2e_recover(t, "raid6(8)", f9, 1, lens, nlens);
        e2e_recover(t, "raid6(8)", f15, 2, lens, nlens);
        e2e_recover(t, "raid6(8)", f28, 2, lens, nlens);
        e2e_recover(t, "raid6(8)", f49, 2, lens, nlens);
        e2e_recover(t, "raid6(8)", f89, 2, lens, nlens);
        rj_topo_free(t);
    }
    {
        rj_topology *t = rj_topo_rs(6, 3);
        int f2[] = { 2 }, f7[] = { 7 };
        int f04[] = { 0, 4 }, f36[] = { 3, 6 }, f68[] = { 6, 8 };
        int f128[] = { 1, 2, 8 }; /* e=3: general-path fallback */
        e2e_recover(t, "rs(6,3)", f2, 1, lens, nlens);
        e2e_recover(t, "rs(6,3)", f7, 1, lens, nlens);
        e2e_recover(t, "rs(6,3)", f04, 2, lens, nlens);
        e2e_recover(t, "rs(6,3)", f36, 2, lens, nlens);
        e2e_recover(t, "rs(6,3)", f68, 2, lens, nlens);
        e2e_recover(t, "rs(6,3)", f128, 3, lens, nlens);
        rj_topo_free(t);
    }
    {
        rj_topology *t = rj_topo_raid5(4);
        int f1[] = { 1 }, f4[] = { 4 };
        e2e_recover(t, "raid5(4)", f1, 1, lens, nlens);
        e2e_recover(t, "raid5(4)", f4, 1, lens, nlens);
        rj_topo_free(t);
    }
    {
        uint8_t cmat[8] = { 1, 1, 1, 1, 1, 2, 4, 8 };
        rj_topology *t = rj_topo_custom(4, 2, cmat);
        int f0[] = { 0 }, f4[] = { 4 };
        int f13[] = { 1, 3 }, f05[] = { 0, 5 };
        e2e_recover(t, "custom(4,2)", f0, 1, lens, nlens);
        e2e_recover(t, "custom(4,2)", f4, 1, lens, nlens);
        e2e_recover(t, "custom(4,2)", f13, 2, lens, nlens);
        e2e_recover(t, "custom(4,2)", f05, 2, lens, nlens);
        rj_topo_free(t);
    }

    if (failures) {
        printf("%d FAILURES\n", failures);
        return 1;
    }
    printf("all tests passed\n");
    return 0;
}
