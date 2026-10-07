/* Cache-blocked striping tests (queue item #10).
 *
 * The tiled driver must be bit-exact vs the untiled rj_fn (same
 * planner, same kernel; tiles are disjoint), across ops (encode,
 * recover, parity-update), topologies, and lengths (tile boundaries,
 * odd sizes, multi-tile stripes). rj_tile_select is pinned by unit
 * checks (monotonicity, budget fit, exact values at a few stream
 * counts).
 *
 * Non-tautological: tiled-vs-untiled compares two independently
 * compiled pipelines; the tile-size pins would fail if the heuristic
 * changed shape.
 */
#include "../include/raidjit_fused.h"
#include "../src/striping.h"

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

static uint64_t rng_state = 0xfedcba987654321ULL;
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

/* ---------------- rj_tile_select unit checks ---------------- */

static void test_tile_select(void)
{
    /* Exact pins at a few stream counts (LLC/2 = 8MiB budget). */
    CHECK(rj_tile_select(1) == 1048576, "1 stream -> 1MiB, got %zu",
          rj_tile_select(1));
    CHECK(rj_tile_select(9) == 524288, "9 streams -> 512KiB, got %zu",
          rj_tile_select(9));
    CHECK(rj_tile_select(100000) == 4096, "huge -> 4KiB floor, got %zu",
          rj_tile_select(100000));
    CHECK(rj_tile_select(0) == 1048576, "0 streams clamps like 1");
    /* Monotonic non-increasing. */
    size_t prev = rj_tile_select(1);
    for (size_t s = 2; s <= 300; s++) {
        size_t cur = rj_tile_select(s);
        CHECK(cur <= prev, "monotonic at %zu: %zu > %zu", s, cur,
              prev);
        prev = cur;
    }
    /* Budget fit: nstreams*tile <= 8MiB for a sweep. */
    for (size_t s = 1; s <= 1000; s++) {
        size_t tile = rj_tile_select(s);
        CHECK(s * tile <= 8 * 1024 * 1024, "budget at %zu", s);
    }
    /* Power of two, in [4KiB, 1MiB]. */
    for (size_t s = 1; s <= 500; s++) {
        size_t tile = rj_tile_select(s);
        CHECK(tile >= 4096 && tile <= 1048576, "range at %zu", s);
        CHECK((tile & (tile - 1)) == 0, "pow2 at %zu", s);
    }
    printf("ok tile_select\n");
}

/* ---------------- tiled vs untiled: encode ---------------- */

static void run_encode_case(rj_topology *t, const char *name, int nd,
                            int np, size_t len)
{
    char err[256];
    rj_op op = rj_op_encode();
    rj_tiled *tt = rj_compile_tiled(t, &op, &topts, err, sizeof(err));
    CHECK(tt != NULL, "%s len=%zu: tiled compile: %s", name, len, err);
    rj_jit *u = rj_compile(t, &op, &topts, err, sizeof(err));
    rj_op_free(&op);
    CHECK(u != NULL, "%s len=%zu: plain compile: %s", name, len, err);
    rj_fn tfn = tt ? rj_get_fn_tiled(tt) : NULL;
    rj_fn ufn = u ? rj_get_fn(u) : NULL;
    CHECK(tfn != NULL && ufn != NULL, "%s: fn NULL", name);

    /* Tile size is immutable across calls and matches the heuristic. */
    size_t tile0 = rj_tiled_tile_bytes(tt);
    CHECK(tile0 == rj_tile_select((size_t)nd + (size_t)np),
          "%s: tile %zu != select(%d)", name, tile0, nd + np);

    const void **in = malloc(sizeof(*in) * (size_t)nd);
    void **oT = malloc(sizeof(*oT) * (size_t)np);
    void **oU = malloc(sizeof(*oU) * (size_t)np);
    for (int i = 0; i < nd; i++) {
        uint8_t *b = abuf(len);
        randfill(b, len);
        in[i] = b;
    }
    for (int p = 0; p < np; p++) {
        oT[p] = abuf(len);
        oU[p] = abuf(len);
    }
    if (tfn && ufn) {
        tfn(in, oT, len);
        ufn(in, oU, len);
        for (int p = 0; p < np; p++)
            CHECK(memcmp(oT[p], oU[p], len) == 0,
                  "%s len=%zu: tiled != untiled (p=%d)", name, len, p);
        CHECK(rj_tiled_tile_bytes(tt) == tile0, "%s: tile changed",
              name);
    }
    for (int i = 0; i < nd; i++)
        free((void *)in[i]);
    for (int p = 0; p < np; p++) {
        free(oT[p]);
        free(oU[p]);
    }
    free(in);
    free(oT);
    free(oU);
    rj_free_tiled(tt);
    rj_free(u);
}

static void test_tiled_encode(void)
{
    static const size_t lens[] = { 1, 63, 64, 65, 1000, 4096,
                                   524287, 524288, 524289, 1048699,
                                   2097275 };
    struct {
        rj_topology *t;
        const char *name;
    } ts[] = {
        { rj_topo_raid5(4), "raid5(4)" },
        { rj_topo_raid6_rs(6), "raid6_rs(6)" },
        { rj_topo_rs(6, 3), "rs(6,3)" },
        { rj_topo_rdp(5), "rdp(5)" },
        { rj_topo_evenodd(5), "evenodd(5)" },
    };
    for (size_t ti = 0; ti < 5; ti++) {
        int nd = rj_topo_ndata(ts[ti].t);
        int np = rj_topo_npar(ts[ti].t);
        for (size_t li = 0; li < 11; li++)
            run_encode_case(ts[ti].t, ts[ti].name, nd, np, lens[li]);
    }
    for (size_t ti = 0; ti < 5; ti++)
        rj_topo_free(ts[ti].t);
    printf("ok tiled_encode\n");
}

/* ---------------- tiled vs untiled: recover ---------------- */

static void test_tiled_recover(void)
{
    /* raid6_rs(6), fail disks {1,4}: 4 survivors in, 2 failed out. */
    rj_topology *t = rj_topo_raid6_rs(6);
    CHECK(t != NULL, "topo");
    if (!t)
        return;
    int failed[2] = { 1, 4 };
    char err[256];
    rj_op op = rj_op_recover(failed, 2);
    rj_tiled *tt = rj_compile_tiled(t, &op, &topts, err, sizeof(err));
    CHECK(tt != NULL, "tiled recover compile: %s", err);
    rj_jit *u = rj_compile(t, &op, &topts, err, sizeof(err));
    rj_op_free(&op);
    CHECK(u != NULL, "plain recover compile: %s", err);
    rj_fn tfn = tt ? rj_get_fn_tiled(tt) : NULL;
    rj_fn ufn = u ? rj_get_fn(u) : NULL;

    static const size_t lens[] = { 64, 4096, 100000 };
    for (size_t li = 0; li < 3; li++) {
        size_t len = lens[li];
        /* Build a full stripe, then present survivors. */
        uint8_t *disk[8];
        const void *ein[6];
        void *epar[2];
        for (int i = 0; i < 8; i++)
            disk[i] = abuf(len);
        for (int i = 0; i < 6; i++) {
            randfill(disk[i], len);
            ein[i] = disk[i];
        }
        epar[0] = disk[6];
        epar[1] = disk[7];
        {
            rj_op eop = rj_op_encode();
            rj_jit *e = rj_compile(t, &eop, &topts, err, sizeof(err));
            rj_op_free(&eop);
            if (e) {
                rj_get_fn(e)(ein, epar, len);
                rj_free(e);
            }
        }
        const void *rin[6];
        void *rT[2], *rU[2];
        int s = 0;
        for (int i = 0; i < 8; i++) {
            if (i == 1 || i == 4)
                continue;
            rin[s++] = disk[i];
        }
        for (int k = 0; k < 2; k++) {
            rT[k] = abuf(len);
            rU[k] = abuf(len);
        }
        if (tfn && ufn) {
            tfn(rin, rT, len);
            ufn(rin, rU, len);
            for (int k = 0; k < 2; k++)
                CHECK(memcmp(rT[k], rU[k], len) == 0,
                      "recover len=%zu tiled != untiled (k=%d)", len,
                      k);
            /* And both equal the original failed disks. */
            CHECK(memcmp(rT[0], disk[1], len) == 0,
                  "recover len=%zu wrong disk1", len);
            CHECK(memcmp(rT[1], disk[4], len) == 0,
                  "recover len=%zu wrong disk4", len);
        }
        for (int i = 0; i < 8; i++)
            free(disk[i]);
        for (int k = 0; k < 2; k++) {
            free(rT[k]);
            free(rU[k]);
        }
    }
    rj_free_tiled(tt);
    rj_free(u);
    rj_topo_free(t);
    printf("ok tiled_recover\n");
}

/* ---------------- tiled vs untiled: parity update ---------------- */

static void test_tiled_pupdate(void)
{
    rj_topology *t = rj_topo_raid5(4);
    CHECK(t != NULL, "topo");
    if (!t)
        return;
    int ch[1] = { 2 };
    char err[256];
    rj_op op = rj_op_parity_update(ch, 1);
    rj_tiled *tt = rj_compile_tiled(t, &op, &topts, err, sizeof(err));
    CHECK(tt != NULL, "tiled pupd compile: %s", err);
    rj_jit *u = rj_compile(t, &op, &topts, err, sizeof(err));
    rj_op_free(&op);
    CHECK(u != NULL, "plain pupd compile: %s", err);
    rj_fn tfn = tt ? rj_get_fn_tiled(tt) : NULL;
    rj_fn ufn = u ? rj_get_fn(u) : NULL;

    static const size_t lens[] = { 64, 50000 };
    for (size_t li = 0; li < 2; li++) {
        size_t len = lens[li];
        /* in: old, new, oldpar; out aliases oldpar (in place). */
        uint8_t *old = abuf(len), *newd = abuf(len);
        uint8_t *parT = abuf(len), *parU = abuf(len);
        randfill(old, len);
        randfill(newd, len);
        randfill(parT, len);
        memcpy(parU, parT, len);
        const void *inT[3] = { old, newd, parT };
        const void *inU[3] = { old, newd, parU };
        void *outT[1] = { parT };
        void *outU[1] = { parU };
        if (tfn && ufn) {
            tfn(inT, outT, len);
            ufn(inU, outU, len);
            CHECK(memcmp(parT, parU, len) == 0,
                  "pupd len=%zu tiled != untiled", len);
        }
        free(old);
        free(newd);
        free(parT);
        free(parU);
    }
    rj_free_tiled(tt);
    rj_free(u);
    rj_topo_free(t);
    printf("ok tiled_pupdate\n");
}

/* ---------------- API surface ---------------- */

static void test_api_surface(void)
{
    char err[256];
    rj_topology *t = rj_topo_raid5(4);
    rj_op op = rj_op_encode();
    CHECK(rj_compile_tiled(NULL, &op, &topts, err, sizeof(err)) ==
              NULL,
          "null topo must fail");
    CHECK(rj_compile_tiled(t, NULL, &topts, err, sizeof(err)) == NULL,
          "null op must fail");
    CHECK(rj_get_fn_tiled(NULL) == NULL, "get_fn(NULL)");
    CHECK(rj_tiled_tile_bytes(NULL) == 0, "tile(NULL)");
    {
        rj_info info;
        rj_get_info_tiled(NULL, &info); /* must not crash */
    }
    rj_free_tiled(NULL); /* must not crash */
    {
        /* Empty changed list fails closed. */
        rj_op bad = rj_op_parity_update(NULL, 0);
        CHECK(rj_compile_tiled(t, &bad, &topts, err, sizeof(err)) ==
                  NULL,
              "empty parity update must fail");
        rj_op_free(&bad);
    }
    rj_op_free(&op);
    rj_topo_free(t);
    printf("ok api_surface\n");
}

int main(void)
{
    rj_options_default(&topts);
    topts.benchmark = 0;
    test_tile_select();
    test_tiled_encode();
    test_tiled_recover();
    test_tiled_pupdate();
    test_api_surface();
    if (failures) {
        printf("test_striping: %d FAILURES\n", failures);
        return 1;
    }
    printf("test_striping: all OK\n");
    return 0;
}
