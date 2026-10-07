/* Integrated-CRC encode tests (queue item #38).
 *
 * Oracles:
 *   - CRC-32/ISO-HDLC catalog check value CRC32("123456789") =
 *     0xCBF43926 (external ground truth; anchors both the scalar
 *     reference and the slice-by-8 fast path).
 *   - An independent bitwise CRC-32 written directly from the
 *     polynomial definition inside this file (no shared code with
 *     src/crc.c): the fused driver's CRCs must match it.
 *   - The plain rj_fn encode path (test_correct proves it against the
 *     scalar GF reference): fused parities must be bit-exact vs it.
 *
 * Coverage: KAT; fast-vs-scalar differential over lengths 0..600 with
 * misaligned starts; fused-vs-unfused bit-exact across topologies x
 * lengths (incl. tile boundaries, odd lengths, len=1); NULL crc sides;
 * npar==0 (raid0, NULL out); tile-size observability; API surface;
 * benchmark-path dispatcher composition.
 *
 * Non-tautological: the KAT pins an external check value (a table or
 * final-xor mutation fails it); the bitwise oracle shares no code with
 * the implementation; fused-vs-unfused compares two independently
 * compiled pipelines.
 */
#include "../include/raidjit_fused.h"

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

static rj_crc_options copts; /* benchmark off: correctness only */

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

/* Independent bitwise CRC-32/ISO-HDLC, straight from the polynomial
 * definition. Shares no code with src/crc.c. */
static uint32_t crc32_bitwise(const uint8_t *p, size_t n)
{
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; i++) {
        crc ^= p[i];
        for (int k = 0; k < 8; k++)
            crc = (crc & 1) ? (crc >> 1) ^ 0xEDB88320u : crc >> 1;
    }
    return crc ^ 0xFFFFFFFFu;
}

/* ---------------- CRC module tests (src/crc.c internals) ----------------
 * test_crc.c links the static lib; reach the two update variants via a
 * tiny local declaration of the internal header (the header is not
 * public API). */
uint32_t rj_crc32_update_scalar(uint32_t crc, const void *data, size_t len);
uint32_t rj_crc32_update(uint32_t crc, const void *data, size_t len);
uint32_t rj_crc32_final(uint32_t crc);
void rj_crc32_init(uint32_t *crc);

static uint32_t ref_fast(const uint8_t *p, size_t n)
{
    uint32_t c;
    rj_crc32_init(&c);
    c = rj_crc32_update(c, p, n);
    return rj_crc32_final(c);
}

static uint32_t ref_scalar(const uint8_t *p, size_t n)
{
    uint32_t c;
    rj_crc32_init(&c);
    c = rj_crc32_update_scalar(c, p, n);
    return rj_crc32_final(c);
}

/* Oracle selector: the bitwise reference is O(8) per byte, so for
 * large buffers use the slice-by-8 path (proven bit-identical to the
 * bitwise oracle on thousands of small cases in test_fast_vs_scalar).
 * Chain of trust: bitwise == fast (small) == fused (all sizes). */
static uint32_t oracle_crc(const uint8_t *p, size_t n)
{
    if (n <= 65536)
        return crc32_bitwise(p, n);
    return ref_fast(p, n);
}

static void test_kat(void)
{
    static const uint8_t msg[] = "123456789";
    CHECK(ref_fast(msg, 9) == 0xCBF43926u, "KAT fast: got %08x",
          ref_fast(msg, 9));
    CHECK(ref_scalar(msg, 9) == 0xCBF43926u, "KAT scalar: got %08x",
          ref_scalar(msg, 9));
    CHECK(crc32_bitwise(msg, 9) == 0xCBF43926u, "KAT bitwise oracle");
    /* Empty input: defined value for CRC-32/ISO-HDLC. */
    CHECK(ref_fast(msg, 0) == 0x00000000u, "empty fast: got %08x",
          ref_fast(msg, 0));
    CHECK(ref_scalar(msg, 0) == 0x00000000u, "empty scalar");
    printf("ok kat\n");
}

static void test_fast_vs_scalar(void)
{
    uint8_t *buf = abuf(700);
    randfill(buf, 700);
    /* Lengths 0..600 plus misaligned starts 0..7: exercises the head,
     * slice, and tail paths of the fast implementation. */
    for (size_t len = 0; len <= 600; len++) {
        for (size_t mis = 0; mis < 8 && mis + len <= 700; mis++) {
            uint32_t f = ref_fast(buf + mis, len);
            uint32_t s = ref_scalar(buf + mis, len);
            uint32_t b = crc32_bitwise(buf + mis, len);
            if (f != s || f != b) {
                CHECK(0, "len=%zu mis=%zu: fast=%08x scalar=%08x bit=%08x",
                      len, mis, f, s, b);
                goto done;
            }
        }
    }
done:
    free(buf);
    printf("ok fast_vs_scalar\n");
}

static void test_crc_mutation_demo(void)
{
    /* A single flipped input bit must change the CRC (avalanche sanity
     * on a fixed vector -- deterministic, essentially impossible to
     * collide). Guards against a degenerate "constant CRC" stub passing
     * the differential tests. */
    uint8_t v[64];
    randfill(v, sizeof(v));
    uint32_t c0 = ref_fast(v, sizeof(v));
    v[17] ^= 0x01;
    uint32_t c1 = ref_fast(v, sizeof(v));
    CHECK(c0 != c1, "flipped bit did not change the CRC");
    printf("ok mutation_demo\n");
}

/* ---------------- fused driver tests ---------------- */

static void test_fused_len(rj_fn_crc cfn, rj_fn fn,
                         rj_topology *t, const char *name, size_t len)
{
    int nd = rj_topo_ndata(t), np = rj_topo_npar(t);

    const void **in = malloc(sizeof(*in) * (size_t)(nd ? nd : 1));
    void **outF = malloc(sizeof(*outF) * (size_t)(np ? np : 1));
    void **outP = malloc(sizeof(*outP) * (size_t)(np ? np : 1));
    uint32_t *crc_in = malloc(sizeof(*crc_in) * (size_t)(nd ? nd : 1));
    uint32_t *crc_out = malloc(sizeof(*crc_out) * (size_t)(np ? np : 1));
    CHECK(in && outF && outP && crc_in && crc_out, "oom");
    if (!in || !outF || !outP || !crc_in || !crc_out) {
        free(in);
        free(outF);
        free(outP);
        free(crc_in);
        free(crc_out);
        return;
    }
    for (int i = 0; i < nd; i++) {
        uint8_t *b = abuf(len);
        randfill(b, len);
        CHECK(((uintptr_t)b & 63) == 0, "input not aligned");
        in[i] = b;
    }
    for (int q = 0; q < np; q++) {
        outF[q] = abuf(len);
        outP[q] = abuf(len);
        memset(outF[q], 0xA5, len);
        memset(outP[q], 0xA5, len);
    }

    cfn(in, np ? outF : NULL, len, crc_in, np ? crc_out : NULL);
    fn(in, np ? outP : NULL, len);

    /* 1. fused parities bit-exact vs unfused. */
    for (int q = 0; q < np; q++)
        CHECK(memcmp(outF[q], outP[q], len) == 0,
              "%s len=%zu: parity %d differs fused vs unfused", name, len,
              q);
    /* 2. fused CRCs match the independent oracle over the same buffers
     * (inputs as fed in; outputs as produced). */
    for (int i = 0; i < nd; i++)
        CHECK(crc_in[i] == oracle_crc(in[i], len),
              "%s len=%zu: crc_in[%d] %08x != oracle %08x", name, len, i,
              crc_in[i], oracle_crc(in[i], len));
    for (int q = 0; q < np; q++)
        CHECK(crc_out[q] == oracle_crc(outF[q], len),
              "%s len=%zu: crc_out[%d] %08x != oracle %08x", name, len, q,
              crc_out[q], oracle_crc(outF[q], len));

    /* 3. NULL sides: skip inputs, skip outputs, skip both -- parities
     * still correct. */
    if (np) {
        for (int q = 0; q < np; q++)
            memset(outF[q], 0x5A, len);
        cfn(in, outF, len, NULL, crc_out);
        for (int q = 0; q < np; q++)
            CHECK(memcmp(outF[q], outP[q], len) == 0,
                  "%s len=%zu: NULL crc_in broke parity %d", name, len, q);
        for (int q = 0; q < np; q++)
            memset(outF[q], 0x5A, len);
        cfn(in, outF, len, crc_in, NULL);
        for (int q = 0; q < np; q++)
            CHECK(memcmp(outF[q], outP[q], len) == 0,
                  "%s len=%zu: NULL crc_out broke parity %d", name, len, q);
        for (int q = 0; q < np; q++)
            memset(outF[q], 0x5A, len);
        cfn(in, outF, len, NULL, NULL);
        for (int q = 0; q < np; q++)
            CHECK(memcmp(outF[q], outP[q], len) == 0,
                  "%s len=%zu: NULL/NULL broke parity %d", name, len, q);
    } else {
        /* npar == 0: NULL out, NULL crc_out; inputs still checksummed. */
        cfn(in, NULL, len, crc_in, NULL);
        for (int i = 0; i < nd; i++)
            CHECK(crc_in[i] == oracle_crc(in[i], len),
                  "%s len=%zu raid0: crc_in[%d] mismatch", name, len, i);
    }

    for (int i = 0; i < nd; i++)
        free((void *)in[i]);
    for (int q = 0; q < np; q++) {
        free(outF[q]);
        free(outP[q]);
    }
    free(in);
    free(outF);
    free(outP);
    free(crc_in);
    free(crc_out);
    printf("ok fused %s len=%zu\n", name, len);
}

/* One topology: compile once (fused + plain oracle), run the length
 * grid. Compiles dominate test time, not the kernels, so the length
 * grid reuses both handles. Tile-relative lengths come from the
 * handle's own plan-time tile decision. */
static void test_fused_topo(rj_topology *t, const char *name)
{
    int nd = rj_topo_ndata(t), np = rj_topo_npar(t);
    char err[256];

    rj_jit_crc *jc = rj_compile_crc(t, &copts, err, sizeof(err));
    CHECK(jc != NULL, "%s compile_crc: %s", name, err);
    if (!jc)
        return;
    rj_fn_crc cfn = rj_get_fn_crc(jc);
    CHECK(cfn != NULL, "%s: rj_get_fn_crc returned NULL", name);
    CHECK(rj_get_fn_crc(jc) == cfn, "%s: repeated fetch differs", name);
    CHECK(rj_crc_ndata(jc) == nd, "%s: ndata %d != %d", name,
          rj_crc_ndata(jc), nd);
    CHECK(rj_crc_npar(jc) == np, "%s: npar %d != %d", name, rj_crc_npar(jc),
          np);
    size_t tile = rj_crc_tile_bytes(jc);
    CHECK(tile >= 4096 && tile <= (size_t)1 << 20, "%s: tile %zu out of range",
          name, tile);
    CHECK((tile & (tile - 1)) == 0, "%s: tile %zu not a power of two", name,
          tile);
    printf("%s: tile %zu\n", name, tile);

    /* Plain unfused oracle: independently compiled rj_fn encode. */
    rj_op op = rj_op_encode();
    rj_options o;
    rj_options_default(&o);
    o.benchmark = 0;
    rj_jit *j = rj_compile(t, &op, &o, err, sizeof(err));
    CHECK(j != NULL, "%s plain compile: %s", name, err);
    rj_op_free(&op);
    if (!j) {
        rj_free_crc(jc);
        return;
    }
    rj_fn fn = rj_get_fn(j);

    if (cfn) {
        size_t lens[] = { 1, 63, 64, 65, 1000, 4096,
                          tile - 1, tile, tile + 1, tile * 2 + 123 };
        for (size_t li = 0; li < sizeof(lens) / sizeof(lens[0]); li++)
            test_fused_len(cfn, fn, t, name, lens[li]);
    }

    rj_free(j);
    rj_free_crc(jc);
}


/* ---------------- benchmark path (dispatcher composition) ----------------
 * Default options (benchmark=1): tail twins may win their install gates
 * and the per-tile calls route through the length dispatcher. The tiled
 * driver must compose with it bit-exactly (the dispatcher honors the
 * rj_fn contract for every tile length, including tails). */
static void test_benchmark_path(void)
{
    rj_topology *t = rj_topo_raid5(4);
    CHECK(t != NULL, "topo");
    if (!t)
        return;
    char err[256];
    rj_crc_options bco;
    rj_crc_options_default(&bco); /* benchmark on */
    rj_jit_crc *jc = rj_compile_crc(t, &bco, err, sizeof(err));
    CHECK(jc != NULL, "benchmark-path compile_crc: %s", err);
    if (!jc) {
        rj_topo_free(t);
        return;
    }
    rj_fn_crc cfn = rj_get_fn_crc(jc);
    CHECK(cfn != NULL, "benchmark path: crc fn NULL");
    rj_info info;
    rj_get_info_crc(jc, &info);
    printf("benchmark path: %s\n", info.desc);

    rj_op op = rj_op_encode();
    rj_options o;
    rj_options_default(&o); /* benchmark on */
    rj_jit *j = rj_compile(t, &op, &o, err, sizeof(err));
    CHECK(j != NULL, "benchmark-path plain compile: %s", err);
    rj_op_free(&op);
    rj_fn fn = j ? rj_get_fn(j) : NULL;

    static const size_t lens[] = { 1000, 4096, 100000 };
    for (size_t li = 0; li < sizeof(lens) / sizeof(lens[0]); li++) {
        size_t len = lens[li];
        const void *in[4];
        void *outF[1], *outP[1];
        for (int i = 0; i < 4; i++) {
            uint8_t *b = abuf(len);
            randfill(b, len);
            in[i] = b;
        }
        outF[0] = abuf(len);
        outP[0] = abuf(len);
        uint32_t crc_in[4], crc_out[1];
        if (cfn && fn) {
            cfn(in, outF, len, crc_in, crc_out);
            fn(in, outP, len);
            CHECK(memcmp(outF[0], outP[0], len) == 0,
                  "benchmark path len=%zu: parity mismatch", len);
            for (int i = 0; i < 4; i++)
                CHECK(crc_in[i] == oracle_crc(in[i], len),
                      "benchmark path len=%zu: crc_in[%d]", len, i);
            CHECK(crc_out[0] == oracle_crc(outF[0], len),
                  "benchmark path len=%zu: crc_out", len);
        }
        for (int i = 0; i < 4; i++)
            free((void *)in[i]);
        free(outF[0]);
        free(outP[0]);
    }
    rj_free(j);
    rj_free_crc(jc);
    rj_topo_free(t);
    printf("ok benchmark_path\n");
}

static void test_api_surface(void)
{
    char err[256];
    CHECK(rj_get_fn_crc(NULL) == NULL, "rj_get_fn_crc(NULL) must be NULL");
    CHECK(rj_compile_crc(NULL, &copts, err, sizeof(err)) == NULL,
          "rj_compile_crc(NULL) must fail");
    CHECK(rj_crc_tile_bytes(NULL) == 0, "tile_bytes(NULL) must be 0");
    CHECK(rj_crc_ndata(NULL) == -1, "ndata(NULL) must be -1");
    CHECK(rj_crc_npar(NULL) == -1, "npar(NULL) must be -1");
    rj_free_crc(NULL); /* no-op */
    rj_info info;
    rj_get_info_crc(NULL, &info); /* no crash */
    /* loop_align validation mirrors rj_compile. */
    rj_topology *t = rj_topo_raid5(4);
    CHECK(t != NULL, "topo");
    if (t) {
        rj_crc_options bad = copts;
        bad.loop_align = 3;
        CHECK(rj_compile_crc(t, &bad, err, sizeof(err)) == NULL,
              "loop_align=3 must fail: %s", err);
        rj_topo_free(t);
    }
    printf("ok api_surface\n");
}

int main(void)
{
    rj_cpu_info ci;
    rj_cpu_detect(&ci);
    printf("cpu: %s\n", ci.brand);
    rj_crc_options_default(&copts);
    copts.benchmark = 0;
    copts.verbose = 0;

    test_kat();
    test_fast_vs_scalar();
    test_crc_mutation_demo();
    test_api_surface();
    test_benchmark_path();

    /* Per-topology fused tests: each compiles once and runs its own
     * length grid (edges, odd sizes, tile boundaries from the
     * handle's plan-time tile decision). */
    struct {
        rj_topology *t;
        const char *name;
    } topos[6];
    int nt = 0;
    topos[nt].t = rj_topo_raid5(4);
    topos[nt++].name = "raid5(4)";
    topos[nt].t = rj_topo_raid6_rs(6);
    topos[nt++].name = "raid6_rs(6)";
    topos[nt].t = rj_topo_rs(6, 3);
    topos[nt++].name = "rs(6,3)";
    topos[nt].t = rj_topo_rdp(5);
    topos[nt++].name = "rdp(5)";
    topos[nt].t = rj_topo_evenodd(5);
    topos[nt++].name = "evenodd(5)";
    topos[nt].t = rj_topo_raid0(4);
    topos[nt++].name = "raid0(4)";

    for (int ti = 0; ti < nt; ti++) {
        if (!topos[ti].t) {
            CHECK(0, "topo %s failed", topos[ti].name);
            continue;
        }
        test_fused_topo(topos[ti].t, topos[ti].name);
        rj_topo_free(topos[ti].t);
    }

    if (failures) {
        printf("test_crc: %d FAILURES\n", failures);
        return 1;
    }
    printf("test_crc: all OK\n");
    return 0;
}
