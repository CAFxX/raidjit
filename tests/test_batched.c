/* Batched Tier-1 API tests (queue item #9).
 *
 * The oracle is the single-stripe path: batched output must be
 * bit-exact vs S individual rj_fn calls (test_correct already proves
 * the single calls correct against the scalar reference). Coverage:
 *   - encode / recover / parity-update across RS and XOR-code topologies
 *   - S = 1 edge case, small S grid, odd lengths (incl. non-multiples
 *     of 64), per-stripe-distinct data (catches stride/offset bugs)
 *   - large S past the internal chunk boundary (Sc_stage = 585 for
 *     10+4 encode): S = 585 / 586 / 1200
 *   - raid0 encode (npar == 0): NULL out, no-op fn
 *   - 64-byte alignment asserted on every buffer handed to the driver
 *   - rj_get_fn_batched(NULL) == NULL; repeated fetch is the same entry
 *   - one benchmark-path compile (dispatcher composition)
 *
 * Non-tautological: verified by mutation -- see the commit notes (a
 * stride/off-by-one mutation in src/api_batched.c must fail this
 * suite).
 */
#include "../include/raidjit.h"

#include <stdint.h>
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

static uint64_t rng_state = 0x243f6a8885a308d3ULL;
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

static void check_aligned(const void *p, const char *what)
{
    CHECK(((uintptr_t)p & 63) == 0, "%s not 64-byte aligned: %p", what, p);
}

/* ---------------- encode ---------------- */

static void test_encode_one(rj_topology *t, const char *name, size_t S,
                            size_t len)
{
    int nd = rj_topo_ndata(t), np = rj_topo_npar(t);
    char err[256];
    rj_op op = rj_op_encode();
    rj_jit *j = rj_compile(t, &op, &topts, err, sizeof(err));
    CHECK(j != NULL, "%s encode S=%zu len=%zu compile: %s", name, S, len,
          err);
    if (!j) {
        rj_op_free(&op);
        return;
    }
    rj_fn fn = rj_get_fn(j);
    rj_fn_batched bfn = rj_get_fn_batched(j);
    CHECK(bfn != NULL, "%s encode: rj_get_fn_batched returned NULL", name);
    CHECK(rj_get_fn_batched(j) == bfn,
          "%s encode: repeated fetch must return the same entry", name);
    if (!bfn) {
        rj_free(j);
        rj_op_free(&op);
        return;
    }

    const void **in_flat = malloc(sizeof(*in_flat) * S * (size_t)nd);
    void **outB = malloc(sizeof(*outB) * S * (size_t)(np ? np : 1));
    void **outR = malloc(sizeof(*outR) * S * (size_t)(np ? np : 1));
    CHECK(in_flat && outB && outR, "oom");
    if (!in_flat || !outB || !outR) {
        rj_free(j);
        rj_op_free(&op);
        return;
    }
    for (size_t s = 0; s < S; s++) {
        for (int i = 0; i < nd; i++) {
            uint8_t *b = abuf(len);
            randfill(b, len); /* distinct data per stripe */
            check_aligned(b, "encode in");
            in_flat[s * (size_t)nd + (size_t)i] = b;
        }
        for (int p = 0; p < np; p++) {
            uint8_t *b = abuf(len);
            uint8_t *r = abuf(len);
            check_aligned(b, "encode outB");
            check_aligned(r, "encode outR");
            memset(b, 0xA5, len);
            memset(r, 0xA5, len);
            outB[s * (size_t)np + (size_t)p] = b;
            outR[s * (size_t)np + (size_t)p] = r;
        }
    }

    /* Batched call. npar==0 (raid0): out may be NULL, fn is a no-op. */
    bfn(in_flat, np ? outB : NULL, S, len);
    for (size_t s = 0; s < S; s++)
        fn(&in_flat[s * (size_t)nd], &outR[s * (size_t)np], len);
    for (size_t s = 0; s < S; s++)
        for (int p = 0; p < np; p++)
            CHECK(memcmp(outB[s * (size_t)np + (size_t)p],
                         outR[s * (size_t)np + (size_t)p], len) == 0,
                  "%s encode S=%zu len=%zu stripe %zu par %d mismatch",
                  name, S, len, s, p);

    for (size_t s = 0; s < S; s++) {
        for (int i = 0; i < nd; i++)
            free((void *)in_flat[s * (size_t)nd + (size_t)i]);
        for (int p = 0; p < np; p++) {
            free(outB[s * (size_t)np + (size_t)p]);
            free(outR[s * (size_t)np + (size_t)p]);
        }
    }
    free(in_flat);
    free(outB);
    free(outR);
    rj_free(j);
    rj_op_free(&op);
}

static void test_encode(rj_topology *t, const char *name)
{
    static const size_t Ss[] = { 1, 2, 3, 7 };
    static const size_t lens[] = { 1, 17, 100, 1000, 4096 };
    for (size_t si = 0; si < sizeof(Ss) / sizeof(Ss[0]); si++)
        for (size_t li = 0; li < sizeof(lens) / sizeof(lens[0]); li++)
            test_encode_one(t, name, Ss[si], lens[li]);
    printf("ok encode %s\n", name);
}

/* ---------------- recover ---------------- */

static void test_recover_one(rj_topology *t, const char *name,
                             const int *failed, int nfailed, size_t S,
                             size_t len)
{
    int nd = rj_topo_ndata(t), np = rj_topo_npar(t), N = nd + np;
    int ns = N - nfailed;
    char err[256];
    rj_op op = rj_op_recover(failed, nfailed);
    rj_jit *j = rj_compile(t, &op, &topts, err, sizeof(err));
    CHECK(j != NULL, "%s recover S=%zu len=%zu compile: %s", name, S, len,
          err);
    if (!j) {
        rj_op_free(&op);
        return;
    }
    rj_fn fn = rj_get_fn(j);
    rj_fn_batched bfn = rj_get_fn_batched(j);
    CHECK(bfn != NULL, "%s recover: rj_get_fn_batched returned NULL", name);
    if (!bfn) {
        rj_free(j);
        rj_op_free(&op);
        return;
    }

    const void **in_flat = malloc(sizeof(*in_flat) * S * (size_t)ns);
    void **outB = malloc(sizeof(*outB) * S * (size_t)nfailed);
    void **outR = malloc(sizeof(*outR) * S * (size_t)nfailed);
    /* per-stripe full disk images (kept for the singles oracle only via
     * outR; batched and singles share the same survivor inputs) */
    uint8_t ***disk = malloc(sizeof(*disk) * S);
    CHECK(in_flat && outB && outR && disk, "oom");
    if (!in_flat || !outB || !outR || !disk) {
        rj_free(j);
        rj_op_free(&op);
        return;
    }
    for (size_t s = 0; s < S; s++) {
        disk[s] = malloc(sizeof(**disk) * (size_t)N);
        for (int d = 0; d < N; d++) {
            disk[s][d] = abuf(len);
            randfill(disk[s][d], len);
            check_aligned(disk[s][d], "recover disk");
        }
        /* wipe the failed disks, like test_correct */
        for (int f = 0; f < nfailed; f++)
            memset(disk[s][failed[f]], 0xA5, len);
        /* survivors in increasing disk order -> in_flat; failed -> out */
        int ni = 0, oi = 0;
        for (int d = 0; d < N; d++) {
            int is_failed = 0;
            for (int f = 0; f < nfailed; f++)
                if (failed[f] == d)
                    is_failed = 1;
            if (is_failed) {
                uint8_t *b = abuf(len), *r = abuf(len);
                check_aligned(b, "recover outB");
                check_aligned(r, "recover outR");
                outB[s * (size_t)nfailed + (size_t)oi] = b;
                outR[s * (size_t)nfailed + (size_t)oi] = r;
                oi++;
            } else {
                in_flat[s * (size_t)ns + (size_t)ni] = disk[s][d];
                ni++;
            }
        }
        CHECK(ni == ns && oi == nfailed, "recover layout botch");
    }

    bfn(in_flat, outB, S, len);
    for (size_t s = 0; s < S; s++)
        fn(&in_flat[s * (size_t)ns], &outR[s * (size_t)nfailed], len);
    for (size_t s = 0; s < S; s++)
        for (int f = 0; f < nfailed; f++)
            CHECK(memcmp(outB[s * (size_t)nfailed + (size_t)f],
                         outR[s * (size_t)nfailed + (size_t)f], len) == 0,
                  "%s recover S=%zu len=%zu stripe %zu failed[%d] mismatch",
                  name, S, len, s, f);

    for (size_t s = 0; s < S; s++) {
        for (int d = 0; d < N; d++)
            free(disk[s][d]);
        free(disk[s]);
        for (int f = 0; f < nfailed; f++) {
            free(outB[s * (size_t)nfailed + (size_t)f]);
            free(outR[s * (size_t)nfailed + (size_t)f]);
        }
    }
    free(disk);
    free(in_flat);
    free(outB);
    free(outR);
    rj_free(j);
    rj_op_free(&op);
}

static void test_recover(rj_topology *t, const char *name, const int *failed,
                         int nfailed)
{
    static const size_t Ss[] = { 1, 3, 8 };
    static const size_t lens[] = { 64, 1000 };
    for (size_t si = 0; si < sizeof(Ss) / sizeof(Ss[0]); si++)
        for (size_t li = 0; li < sizeof(lens) / sizeof(lens[0]); li++)
            test_recover_one(t, name, failed, nfailed, Ss[si], lens[li]);
    printf("ok recover %s\n", name);
}

/* ---------------- parity update ---------------- */

static void test_pupdate_one(rj_topology *t, const char *name,
                             const int *changed, int nchanged, size_t S,
                             size_t len)
{
    int np = rj_topo_npar(t);
    int ni = 2 * nchanged + np;
    char err[256];
    rj_op op = rj_op_parity_update(changed, nchanged);
    rj_jit *j = rj_compile(t, &op, &topts, err, sizeof(err));
    CHECK(j != NULL, "%s pupdate S=%zu len=%zu compile: %s", name, S, len,
          err);
    if (!j) {
        rj_op_free(&op);
        return;
    }
    rj_fn fn = rj_get_fn(j);
    rj_fn_batched bfn = rj_get_fn_batched(j);
    CHECK(bfn != NULL, "%s pupdate: rj_get_fn_batched returned NULL", name);
    if (!bfn) {
        rj_free(j);
        rj_op_free(&op);
        return;
    }

    const void **in_flat = malloc(sizeof(*in_flat) * S * (size_t)ni);
    void **outB = malloc(sizeof(*outB) * S * (size_t)np);
    void **outR = malloc(sizeof(*outR) * S * (size_t)np);
    CHECK(in_flat && outB && outR, "oom");
    if (!in_flat || !outB || !outR) {
        rj_free(j);
        rj_op_free(&op);
        return;
    }
    /* in[] = [old_0..old_{nc-1}, new_0..new_{nc-1}, oldpar_0..oldpar_{np-1}].
     * The op sorts its copy of changed[], so feed it an unsorted list
     * ({3,1}) to exercise the sort path at compile time; batched and
     * singles see identical in[] either way. */
    for (size_t s = 0; s < S; s++) {
        size_t base = s * (size_t)ni;
        for (int k = 0; k < 2 * nchanged + np; k++) {
            uint8_t *b = abuf(len);
            randfill(b, len);
            check_aligned(b, "pupdate in");
            in_flat[base + (size_t)k] = b;
        }
        for (int p = 0; p < np; p++) {
            uint8_t *b = abuf(len), *r = abuf(len);
            check_aligned(b, "pupdate outB");
            check_aligned(r, "pupdate outR");
            outB[s * (size_t)np + (size_t)p] = b;
            outR[s * (size_t)np + (size_t)p] = r;
        }
    }

    bfn(in_flat, outB, S, len);
    for (size_t s = 0; s < S; s++)
        fn(&in_flat[s * (size_t)ni], &outR[s * (size_t)np], len);
    for (size_t s = 0; s < S; s++)
        for (int p = 0; p < np; p++)
            CHECK(memcmp(outB[s * (size_t)np + (size_t)p],
                         outR[s * (size_t)np + (size_t)p], len) == 0,
                  "%s pupdate S=%zu len=%zu stripe %zu par %d mismatch",
                  name, S, len, s, p);

    for (size_t s = 0; s < S; s++) {
        for (int k = 0; k < ni; k++)
            free((void *)in_flat[s * (size_t)ni + (size_t)k]);
        for (int p = 0; p < np; p++) {
            free(outB[s * (size_t)np + (size_t)p]);
            free(outR[s * (size_t)np + (size_t)p]);
        }
    }
    free(in_flat);
    free(outB);
    free(outR);
    rj_free(j);
    rj_op_free(&op);
}

static void test_pupdate(rj_topology *t, const char *name)
{
    static const size_t Ss[] = { 1, 4 };
    static const size_t lens[] = { 128, 2048 };
    int ch[2] = { 3, 1 }; /* deliberately unsorted; op sorts it */
    for (size_t si = 0; si < sizeof(Ss) / sizeof(Ss[0]); si++)
        for (size_t li = 0; li < sizeof(lens) / sizeof(lens[0]); li++)
            test_pupdate_one(t, name, ch, 2, Ss[si], lens[li]);
    printf("ok pupdate %s\n", name);
}

/* ---------------- large S: internal chunking ----------------
 * 12+3 encode: Sc_stage = 65536/(15*8) = 546 stripes per chunk.
 * S = 546 / 547 / 1200 exercises the exact boundary and multi-chunk. */
static void test_large_s(void)
{
    static const size_t Ss[] = { 546, 547, 1200 };
    rj_topology *t = rj_topo_rs(12, 3);
    CHECK(t != NULL, "topo_rs(12,3)");
    if (!t)
        return;
    for (size_t si = 0; si < sizeof(Ss) / sizeof(Ss[0]); si++)
        test_encode_one(t, "rs(12,3)", Ss[si], 256);
    rj_topo_free(t);
    printf("ok large-S chunking\n");
}

/* ---------------- misc API surface ---------------- */

static void test_api_surface(void)
{
    CHECK(rj_get_fn_batched(NULL) == NULL,
          "rj_get_fn_batched(NULL) must be NULL");
    printf("ok api surface\n");
}

/* ---------------- benchmark path (dispatcher composition) ----------------
 * Default options (benchmark=1): if a tail twin wins its gate the
 * per-stripe fn is the length dispatcher; the batched driver must
 * compose with it bit-exactly. */
static void test_benchmark_path(void)
{
    rj_topology *t = rj_topo_raid5(4);
    CHECK(t != NULL, "topo_raid5(4)");
    if (!t)
        return;
    char err[256];
    rj_options o;
    rj_options_default(&o); /* benchmark on */
    rj_op op = rj_op_encode();
    rj_jit *j = rj_compile(t, &op, &o, err, sizeof(err));
    CHECK(j != NULL, "benchmark-path compile: %s", err);
    if (!j) {
        rj_op_free(&op);
        rj_topo_free(t);
        return;
    }
    rj_info info;
    rj_get_info(j, &info);
    rj_fn fn = rj_get_fn(j);
    rj_fn_batched bfn = rj_get_fn_batched(j);
    CHECK(bfn != NULL, "benchmark path: batched entry NULL");
    if (bfn) {
        static const size_t lens[] = { 1000, 4096, 100000 };
        for (size_t li = 0; li < sizeof(lens) / sizeof(lens[0]); li++) {
            size_t len = lens[li], S = 5;
            const void **in_flat = malloc(sizeof(*in_flat) * S * 4);
            void **outB = malloc(sizeof(*outB) * S);
            void **outR = malloc(sizeof(*outR) * S);
            for (size_t s = 0; s < S; s++) {
                for (int i = 0; i < 4; i++) {
                    uint8_t *b = abuf(len);
                    randfill(b, len);
                    in_flat[s * 4 + (size_t)i] = b;
                }
                outB[s] = abuf(len);
                outR[s] = abuf(len);
            }
            bfn(in_flat, outB, S, len);
            for (size_t s = 0; s < S; s++)
                fn(&in_flat[s * 4], &outR[s], len);
            for (size_t s = 0; s < S; s++)
                CHECK(memcmp(outB[s], outR[s], len) == 0,
                      "benchmark path len=%zu stripe %zu mismatch (%s)",
                      len, s, info.desc);
            for (size_t s = 0; s < S; s++) {
                for (int i = 0; i < 4; i++)
                    free((void *)in_flat[s * 4 + (size_t)i]);
                free(outB[s]);
                free(outR[s]);
            }
            free(in_flat);
            free(outB);
            free(outR);
        }
    }
    rj_free(j);
    rj_op_free(&op);
    rj_topo_free(t);
    printf("ok benchmark path\n");
}

int main(void)
{
    rj_cpu_info ci;
    rj_cpu_detect(&ci);
    printf("cpu: %s\n", ci.brand);
    rj_options_default(&topts);
    topts.benchmark = 0;
    topts.verbose = 0;

    test_api_surface();

    rj_topology *t;
    t = rj_topo_raid5(4);
    CHECK(t != NULL, "topo_raid5(4)");
    if (t) {
        test_encode(t, "raid5(4)");
        rj_topo_free(t);
    }
    t = rj_topo_raid6_rs(6);
    CHECK(t != NULL, "topo_raid6_rs(6)");
    if (t) {
        test_encode(t, "raid6_rs(6)");
        int f2[2] = { 1, 4 };
        test_recover(t, "raid6_rs(6)", f2, 2);
        test_pupdate(t, "raid6_rs(6)");
        rj_topo_free(t);
    }
    t = rj_topo_raid5(4);
    CHECK(t != NULL, "topo_raid5(4)");
    if (t) {
        int f1[1] = { 2 };
        test_recover(t, "raid5(4)", f1, 1);
        rj_topo_free(t);
    }
    t = rj_topo_rdp(5);
    CHECK(t != NULL, "topo_rdp(5)");
    if (t) {
        test_encode(t, "rdp(5)");
        rj_topo_free(t);
    }
    t = rj_topo_evenodd(5);
    CHECK(t != NULL, "topo_evenodd(5)");
    if (t) {
        test_encode(t, "evenodd(5)");
        rj_topo_free(t);
    }
    t = rj_topo_raid0(4);
    CHECK(t != NULL, "topo_raid0(4)");
    if (t) {
        test_encode(t, "raid0(4)"); /* npar==0: NULL out, no-op fn */
        rj_topo_free(t);
    }

    test_large_s();
    test_benchmark_path();

    if (failures) {
        printf("test_batched: %d FAILURES\n", failures);
        return 1;
    }
    printf("test_batched: all OK\n");
    return 0;
}
