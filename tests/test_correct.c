/* Exhaustive correctness tests: every topology x op x length, checked
 * against a scalar GF(2^8) reference implementation. */
#include "../include/raidjit.h"
#include "../src/plan.h"
#include "../src/gf.h"
#include "../src/ref.h"

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

static rj_options topts; /* benchmark off: correctness only */

static const size_t lens[] = { 0, 1, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65,
                               100, 255, 256, 257, 1000, 4096, 65537 };
static const int nlens = sizeof(lens) / sizeof(lens[0]);

static uint64_t rng_state = 0x9e3779b97f4a7c15ULL;
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

static int topo_mat(rj_topology *t, topo_int_t *ti, char *err)
{
    return topo_int_build(ti, t, err, 256);
}

static void test_encode(rj_topology *t, const char *name)
{
    topo_int_t ti;
    char err[256];
    CHECK(topo_mat(t, &ti, err) == 0, "%s topo: %s", name, err);
    if (failures)
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
        /* cross-check the library's scalar oracle (used by benchmark
         * validation) against the independent reference above */
        {
            plan_t pl;
            if (plan_build(&ti, &op, name, &pl, err, sizeof(err)) == 0) {
                uint8_t **pr2 = malloc(sizeof(*pr2) * (size_t)np);
                for (int j = 0; j < np; j++)
                    pr2[j] = abuf(len);
                ref_run(&pl, (const uint8_t **)d, pr2, len);
                for (int j = 0; j < np; j++)
                    CHECK(memcmp(pr[j], pr2[j], len) == 0,
                          "%s encode len=%zu ref_run diverges (par%d)", name,
                          len, j);
                for (int j = 0; j < np; j++)
                    free(pr2[j]);
                free(pr2);
                plan_free(&pl);
            } else {
                CHECK(0, "%s encode len=%zu plan: %s", name, len, err);
            }
        }
        rj_jit *j = rj_compile(t, &op, &topts, err, sizeof(err));
        CHECK(j != NULL, "%s encode len=%zu compile: %s", name, len, err);
        if (j) {
            rj_info info;
            rj_get_info(j, &info);
            const void **in = malloc(sizeof(*in) * (size_t)nd);
            void **out = malloc(sizeof(*out) * (size_t)np);
            for (int i = 0; i < nd; i++)
                in[i] = d[i];
            for (int j2 = 0; j2 < np; j2++)
                out[j2] = p[j2];
            rj_get_fn(j)(in, out, len);
            for (int j2 = 0; j2 < np; j2++)
                CHECK(memcmp(p[j2], pr[j2], len) == 0,
                      "%s encode len=%zu par%d mismatch (%s)", name, len, j2,
                      info.desc);
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

static int int_cmp(const void *a, const void *b)
{
    return *(const int *)a - *(const int *)b;
}

static void test_recover(rj_topology *t, const char *name, const int *failed,
                         int nfailed, int expect_ok)
{
    topo_int_t ti;
    char err[256];
    CHECK(topo_mat(t, &ti, err) == 0, "%s topo: %s", name, err);
    if (failures)
        return;
    int nd = ti.ndata, np = ti.npar, N = nd + np;

    int fs[RJ_MAX_DISKS];
    memcpy(fs, failed, sizeof(int) * (size_t)nfailed);
    qsort(fs, (size_t)nfailed, sizeof(int), int_cmp);

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
        ref_encode(ti.mat, nd, np, all, all + nd, len);
        for (int i = 0; i < N; i++)
            memcpy(keep[i], all[i], len);

        rj_op op = rj_op_recover(failed, nfailed);
        rj_jit *j = rj_compile(t, &op, &topts, err, sizeof(err));
        if (!expect_ok) {
            CHECK(j == NULL, "%s recover(%d failed) len=%zu: expected failure",
                  name, nfailed, len);
            rj_op_free(&op);
            for (int i = 0; i < N; i++) {
                free(all[i]);
                free(keep[i]);
            }
            free(all);
            free(keep);
            continue;
        }
        CHECK(j != NULL, "%s recover len=%zu compile: %s", name, len, err);
        if (!j) {
            rj_op_free(&op);
            for (int i = 0; i < N; i++) {
                free(all[i]);
                free(keep[i]);
            }
            free(all);
            free(keep);
            continue;
        }
        /* wipe failed disks */
        for (int f = 0; f < nfailed; f++)
            memset(all[fs[f]], 0xA5, len);
        /* survivors in increasing disk order */
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
        rj_info info;
        rj_get_info(j, &info);
        rj_get_fn(j)(in, out, len);
        for (int f = 0; f < nfailed; f++)
            CHECK(memcmp(all[fs[f]], keep[fs[f]], len) == 0,
                  "%s recover disk%d len=%zu mismatch (%s)", name, fs[f], len,
                  info.desc);
        rj_free(j);
        free(in);
        free(out);
        rj_op_free(&op);
        for (int i = 0; i < N; i++) {
            free(all[i]);
            free(keep[i]);
        }
        free(all);
        free(keep);
    }
    topo_int_free(&ti);
    printf("ok recover %s failed={", name);
    for (int f = 0; f < nfailed; f++)
        printf("%s%d", f ? "," : "", failed[f]);
    printf("}\n");
}

int main(void)
{
    rj_cpu_info ci;
    rj_cpu_detect(&ci);
    printf("cpu: %s\n", ci.brand);
    rj_options_default(&topts);
    topts.benchmark = 0;
    topts.verbose = 0;

    /* RAID-0 (encode = identity copy) */
    rj_topology *r0 = rj_topo_raid0(4);
    test_encode(r0, "raid0(4)");

    /* RAID-1 */
    rj_topology *r1 = rj_topo_raid1();
    test_encode(r1, "raid1");
    int f0[] = { 0 };
    test_recover(r1, "raid1", f0, 1, 1);
    int f1[] = { 1 };
    test_recover(r1, "raid1", f1, 1, 1);

    /* RAID-5 */
    rj_topology *r5 = rj_topo_raid5(4);
    test_encode(r5, "raid5(4)");
    int f2[] = { 2 };
    test_recover(r5, "raid5(4)", f2, 1, 1);
    int f4[] = { 4 };
    test_recover(r5, "raid5(4)", f4, 1, 1); /* parity disk */
    int f24[] = { 2, 4 };
    test_recover(r5, "raid5(4)", f24, 2, 0); /* unrecoverable */

    /* RAID-6 */
    rj_topology *r6 = rj_topo_raid6(8);
    test_encode(r6, "raid6(8)");
    int fa[] = { 0 };
    test_recover(r6, "raid6(8)", fa, 1, 1);
    int fb[] = { 8 };
    test_recover(r6, "raid6(8)", fb, 1, 1);
    int fc[] = { 9 };
    test_recover(r6, "raid6(8)", fc, 1, 1);
    int fd[] = { 7, 0 }; /* unsorted input exercises sorting */
    test_recover(r6, "raid6(8)", fd, 2, 1);
    int fe[] = { 3, 8 };
    test_recover(r6, "raid6(8)", fe, 2, 1);
    int ff[] = { 8, 9 };
    test_recover(r6, "raid6(8)", ff, 2, 1);
    int fg[] = { 0, 1, 2 };
    test_recover(r6, "raid6(8)", fg, 3, 0); /* unrecoverable */

    /* small RAID-6, boundary n=3 */
    rj_topology *r6s = rj_topo_raid6(3);
    test_encode(r6s, "raid6(3)");
    int fh[] = { 1, 2 };
    test_recover(r6s, "raid6(3)", fh, 2, 1);

    /* Reed-Solomon 6+3 */
    rj_topology *rs = rj_topo_rs(6, 3);
    test_encode(rs, "rs(6,3)");
    int fi3[] = { 0, 1, 2 };
    test_recover(rs, "rs(6,3)", fi3, 3, 1);
    int fj3[] = { 6, 7, 8 };
    test_recover(rs, "rs(6,3)", fj3, 3, 1);
    int fk3[] = { 2, 5, 7 };
    test_recover(rs, "rs(6,3)", fk3, 3, 1);
    int fl4[] = { 0, 1, 2, 3 };
    test_recover(rs, "rs(6,3)", fl4, 4, 0); /* unrecoverable */

    /* custom non-Horner matrix -> generic mulacc path */
    uint8_t cmat[8] = { 1, 1, 1, 1, 1, 2, 4, 8 };
    rj_topology *rc = rj_topo_custom(4, 2, cmat);
    test_encode(rc, "custom(4,2)");
    int fm[] = { 0, 3 };
    test_recover(rc, "custom(4,2)", fm, 2, 1);
    int fn2[] = { 4, 5 };
    test_recover(rc, "custom(4,2)", fn2, 2, 1);

    /* benchmark-selection path (one case, short bench) */
    {
        rj_options bo;
        rj_options_default(&bo);
        bo.benchmark = 1;
        bo.bench_bytes = 65536;
        bo.bench_iters = 3;
        bo.verbose = 1;
        char err[256];
        rj_op op = rj_op_encode();
        rj_jit *j = rj_compile(r6, &op, &bo, err, sizeof(err));
        CHECK(j != NULL, "benchmark compile: %s", err);
        if (j) {
            rj_info info;
            rj_get_info(j, &info);
            printf("benchmark picked: %s (%.2f GB/s)\n", info.desc,
                   info.gbps);
            uint8_t *d[8], *p[2];
            for (int i = 0; i < 8; i++) {
                d[i] = abuf(1000);
                randfill(d[i], 1000);
            }
            for (int k = 0; k < 2; k++)
                p[k] = abuf(1000);
            const void *in[8];
            void *out[2];
            for (int i = 0; i < 8; i++)
                in[i] = d[i];
            for (int k = 0; k < 2; k++)
                out[k] = p[k];
            rj_get_fn(j)(in, out, 1000);
            uint8_t *pr[2];
            for (int k = 0; k < 2; k++)
                pr[k] = abuf(1000);
            topo_int_t ti;
            CHECK(topo_mat(r6, &ti, err) == 0, "%s", err);
            ref_encode(ti.mat, 8, 2, d, pr, 1000);
            for (int k = 0; k < 2; k++)
                CHECK(memcmp(p[k], pr[k], 1000) == 0,
                      "benchmark version encode mismatch");
            topo_int_free(&ti);
            for (int i = 0; i < 8; i++)
                free(d[i]);
            for (int k = 0; k < 2; k++) {
                free(p[k]);
                free(pr[k]);
            }
            rj_free(j);
        }
    }

    rj_topo_free(r0);
    rj_topo_free(r1);
    rj_topo_free(r5);
    rj_topo_free(r6);
    rj_topo_free(r6s);
    rj_topo_free(rs);
    rj_topo_free(rc);

    /* MDS envelope of the default RS matrix (peer-review pin): g=2 has
     * order 51, and the {0,3} row-exponent minor vanishes for npar>=4,
     * so the named constructors must refuse anything outside
     * npar<=3 / ndata<=51 (npar<=1 needs no distinct bases). */
    CHECK(rj_topo_rs(52, 2) == NULL, "rs(52,2) must be rejected");
    CHECK(rj_topo_rs(20, 4) == NULL, "rs(20,4) must be rejected");
    CHECK(rj_topo_rs(8, 4) == NULL, "rs(8,4) must be rejected");
    CHECK(rj_topo_raid6(52) == NULL, "raid6(52) must be rejected");
    {
        rj_topology *ok1 = rj_topo_rs(51, 3);
        rj_topology *ok2 = rj_topo_raid6(51);
        rj_topology *ok3 = rj_topo_rs(60, 1); /* npar=1: no basis limit */
        CHECK(ok1 != NULL, "rs(51,3) must be accepted");
        CHECK(ok2 != NULL, "raid6(51) must be accepted");
        CHECK(ok3 != NULL, "rs(60,1) must be accepted");
        rj_topo_free(ok1);
        rj_topo_free(ok2);
        rj_topo_free(ok3);
    }

    if (failures) {
        printf("%d FAILURES\n", failures);
        return 1;
    }
    printf("all tests passed\n");
    return 0;
}
