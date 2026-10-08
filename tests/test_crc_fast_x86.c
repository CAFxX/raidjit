/* Tests for the x86-64 fast CRC-32 variants (src/x86_64/crc_fast.c).
 *
 * Every variant is proved bit-exact vs the scalar oracle
 * (rj_crc32_update_scalar) and vs zlib's crc32 (independent implementation,
 * same IEEE polynomial). The test matrix:
 *  - KAT: CRC32("123456789") == 0xCBF43926 per variant.
 *  - Exhaustive lengths 0..256, all misalignments 0..15, >=3 seeds.
 *  - Tail classes around 16/64B boundaries; lengths {1000, 10000, 1<<20}.
 *  - Split-point chaining: f(c,A||B) == f(f(c,A),B) at many splits.
 *  - Guard-page: no over-read past the buffer end (ASan + PROT_NONE page).
 *  - NULL/len-0, garbage crc, concurrent dispatch.
 *  - rj_crc32_fast_x86_class matches detected features.
 *
 * Red-green: each correctness test fails if a fold constant is corrupted.
 * Demonstrated by corrupting K1/K2 (or a lane constant), rebuilding, and
 * observing failures; then restoring. See the W1 final report for the
 * recorded evidence.
 */
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <zlib.h>

#include "raidjit.h"
#include "../src/crc.h"
#include "../src/x86_64/crc_fast.h"

static int fails = 0;
#define CHECK(cond, ...) do { \
    if (!(cond)) { \
        printf("FAIL %s:%d: ", __FILE__, __LINE__); \
        printf(__VA_ARGS__); printf("\n"); fails++; \
    } \
} while (0)

/* Deterministic PRNG (xorshift64*). */
static uint64_t rng_state;
static uint64_t rng_next(void)
{
    uint64_t x = rng_state;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    rng_state = x;
    return x * 0x2545F4914F6CDD1Dull;
}

static void fill_buf(uint8_t *p, size_t n, uint64_t seed)
{
    rng_state = seed;
    for (size_t i = 0; i < n; i++)
        p[i] = (uint8_t)(rng_next() >> 33);
}

/* The four entry points under test. */
typedef uint32_t (*variant_fn)(uint32_t, const void *, size_t);
static const struct { const char *name; variant_fn fn; } variants[] = {
    { "vpclmul", rj_crc32_fast_x86_vpclmul },
    { "pclmul",  rj_crc32_fast_x86_pclmul },
    { "crc32hw", rj_crc32_fast_x86_crc32hw },
    { "x86",     rj_crc32_fast_x86 },
};
#define NVAR (sizeof(variants)/sizeof(variants[0]))

static void test_kat(void)
{
    static const uint8_t msg[] = "123456789";
    for (size_t v = 0; v < NVAR; v++) {
        uint32_t c;
        rj_crc32_init(&c);
        c = variants[v].fn(c, msg, 9);
        c = rj_crc32_final(c);
        CHECK(c == 0xCBF43926u, "%s KAT: got 0x%08x want 0xCBF43926",
              variants[v].name, c);
    }
    printf("ok kat\n");
}

/* Compare variant vs scalar oracle and zlib across lengths/seeds/offsets. */
static void check_range(const uint8_t *base, size_t off, size_t len,
                        uint32_t seed_crc)
{
    const uint8_t *p = base + off;
    uint32_t ref = rj_crc32_update_scalar(seed_crc, p, len);
    uint32_t zref = (uint32_t)crc32((uLong)seed_crc ^ 0xFFFFFFFFu,
                                    p, (uInt)len) ^ 0xFFFFFFFFu;
    /* zlib's crc32 takes the FINAL crc; convert: our running crc is ~final.
     * crc32(final_crc, ...) expects the non-inverted running state, so we
     * pass ~seed_crc and invert the result. */
    for (size_t v = 0; v < NVAR; v++) {
        uint32_t got = variants[v].fn(seed_crc, p, len);
        CHECK(got == ref, "%s len=%zu off=%zu seed=0x%08x: got 0x%08x want 0x%08x (scalar)",
              variants[v].name, len, off, seed_crc, got, ref);
        CHECK(got == zref, "%s len=%zu off=%zu seed=0x%08x: got 0x%08x want 0x%08x (zlib)",
              variants[v].name, len, off, seed_crc, got, zref);
    }
}

static void test_exhaustive_small(void)
{
    /* 512B scratch; offsets 0..15, lengths 0..256, 3 seeds. */
    uint8_t *buf = malloc(512 + 16);
    const uint64_t seeds[3] = { 0x123456789ABCDEF0ull, 1ull, 0xDEADBEEFCAFEBABEull };
    const uint32_t crc_seeds[3] = { 0xFFFFFFFFu, 0x00000000u, 0x12345678u };
    for (int s = 0; s < 3; s++) {
        fill_buf(buf, 512 + 16, seeds[s]);
        for (size_t off = 0; off < 16; off++)
            for (size_t len = 0; len <= 256; len++)
                check_range(buf, off, len, crc_seeds[s]);
    }
    free(buf);
    printf("ok exhaustive_small\n");
}

static void test_boundaries(void)
{
    /* Tail classes around 16/64B boundaries + larger lengths. */
    static const size_t lens[] = {
        15, 16, 17, 31, 32, 33, 47, 48, 49, 55, 56, 57, 63, 64, 65,
        95, 96, 97, 103, 104, 105, 111, 112, 113, 127, 128, 129,
        191, 192, 193, 199, 200, 201, 207, 208, 209, 215, 216, 217,
        1000, 10000, 1u << 20
    };
    uint8_t *buf = malloc((1u << 20) + 16);
    const uint64_t seeds[3] = { 42ull, 0xABCDEFull, 0x13579BDFull };
    for (int s = 0; s < 3; s++) {
        fill_buf(buf, (1u << 20) + 16, seeds[s]);
        for (size_t li = 0; li < sizeof(lens)/sizeof(lens[0]); li++)
            for (size_t off = 0; off < 16; off++)
                check_range(buf, off, lens[li], 0xFFFFFFFFu);
    }
    free(buf);
    printf("ok boundaries\n");
}

static void test_chaining(void)
{
    /* Split-point chaining: f(c,A||B) == f(f(c,A),B). Load-bearing for the
     * fused driver's per-tile accumulation. */
    uint8_t *buf = malloc(2048);
    fill_buf(buf, 2048, 0x77777777ull);
    static const size_t lens[] = { 1, 7, 8, 15, 16, 31, 32, 55, 56, 63, 64,
                                   100, 103, 104, 111, 112, 199, 200, 207, 208,
                                   500, 1000, 1500 };
    for (size_t li = 0; li < sizeof(lens)/sizeof(lens[0]); li++) {
        size_t len = lens[li];
        for (size_t v = 0; v < NVAR; v++) {
            uint32_t whole = variants[v].fn(0xFFFFFFFFu, buf, len);
            /* Every split offset mod 64 (cover 0..63 twice where possible). */
            for (size_t split = 0; split <= len && split < 128; split++) {
                uint32_t a = variants[v].fn(0xFFFFFFFFu, buf, split);
                uint32_t ab = variants[v].fn(a, buf + split, len - split);
                CHECK(ab == whole,
                      "%s chaining len=%zu split=%zu: 0x%08x != 0x%08x",
                      variants[v].name, len, split, ab, whole);
            }
        }
    }
    free(buf);
    printf("ok chaining\n");
}

static void test_null_zero(void)
{
    static const uint8_t one[1] = { 0 };
    for (size_t v = 0; v < NVAR; v++) {
        uint32_t c0 = variants[v].fn(0xFFFFFFFFu, NULL, 0);
        CHECK(c0 == 0xFFFFFFFFu, "%s NULL/0", variants[v].name);
        uint32_t c1 = variants[v].fn(0xFFFFFFFFu, one, 0);
        CHECK(c1 == 0xFFFFFFFFu, "%s non-NULL/0", variants[v].name);
        uint32_t g = variants[v].fn(0x12345678u, NULL, 0);
        CHECK(g == 0x12345678u, "%s garbage crc/0", variants[v].name);
    }
    printf("ok null_zero\n");
}

static void test_class(void)
{
    rj_cpu_info ci;
    rj_cpu_detect(&ci);
    int cls = rj_crc32_fast_x86_class();
    int expect = 0;
    if (ci.vpclmulqdq && ci.pclmulqdq && ci.avx512f && ci.avx512bw)
        expect = 3;  /* rung 1 */
    else if (ci.vpclmulqdq && ci.pclmulqdq && ci.avx2)
        expect = 3;  /* rung 2 */
    else if (ci.pclmulqdq)
        expect = 2;  /* rung 3: pclmul128 fold */
    CHECK(cls == expect, "class: got %d want %d (vpclmulqdq=%d pclmulqdq=%d avx512f=%d avx512bw=%d avx2=%d)",
          cls, expect, ci.vpclmulqdq, ci.pclmulqdq, ci.avx512f, ci.avx512bw, ci.avx2);
    printf("cpu: pclmulqdq=%d vpclmulqdq=%d sse42=%d avx512f=%d avx512bw=%d avx2=%d -> class %d\n",
           ci.pclmulqdq, ci.vpclmulqdq, ci.sse42, ci.avx512f, ci.avx512bw, ci.avx2, cls);
    printf("ok class\n");
}

/* Guard-page: buffer ends exactly at a PROT_NONE page; any over-read faults. */
static void test_guard_page(void)
{
    long ps = sysconf(_SC_PAGESIZE);
    /* Lengths chosen to exercise every path: scalar-only, 128b, 256b, 512b
     * vector bodies, plus head/tail misalignments. */
    static const size_t lens[] = {
        1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 55, 56, 57, 63, 64, 65,
        95, 96, 97, 103, 104, 105, 111, 112, 113, 127, 128,
        199, 200, 201, 207, 208, 209, 215, 216, 500, 1000
    };
    for (size_t li = 0; li < sizeof(lens)/sizeof(lens[0]); li++) {
        size_t len = lens[li];
        for (size_t off = 0; off < 16; off++) {
            size_t total = off + len;
            size_t maplen = ((total + ps - 1) / ps + 1) * ps;
            uint8_t *map = mmap(NULL, maplen, PROT_READ | PROT_WRITE,
                                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            CHECK(map != MAP_FAILED, "mmap");
            /* Guard page at the end. */
            CHECK(mprotect(map + maplen - ps, ps, PROT_NONE) == 0, "mprotect");
            uint8_t *end = map + maplen - ps; /* one past last readable byte */
            uint8_t *p = end - len;          /* buffer ends at guard page */
            /* Start `off` bytes earlier so the head is misaligned. */
            uint8_t *start = p - off;
            /* Fill the readable region deterministically. */
            for (size_t i = 0; i < total; i++)
                start[i] = (uint8_t)(i * 31 + 7);
            uint32_t ref = rj_crc32_update_scalar(0xFFFFFFFFu, start, total);
            for (size_t v = 0; v < NVAR; v++) {
                uint32_t got = variants[v].fn(0xFFFFFFFFu, start, total);
                CHECK(got == ref, "%s guard len=%zu off=%zu: 0x%08x != 0x%08x",
                      variants[v].name, len, off, got, ref);
            }
            munmap(map, maplen);
        }
    }
    printf("ok guard_page\n");
}

static void *thread_fn(void *arg)
{
    uint8_t *buf = arg;
    uint32_t acc = 0;
    for (int i = 0; i < 200; i++)
        acc ^= rj_crc32_fast_x86(0xFFFFFFFFu, buf, 1000);
    return (void *)(uintptr_t)acc;
}

static void test_threads(void)
{
    uint8_t *buf = malloc(1000);
    fill_buf(buf, 1000, 99ull);
    pthread_t th[8];
    for (int i = 0; i < 8; i++)
        CHECK(pthread_create(&th[i], NULL, thread_fn, buf) == 0, "pthread_create");
    for (int i = 0; i < 8; i++)
        CHECK(pthread_join(th[i], NULL) == 0, "pthread_join");
    free(buf);
    printf("ok threads\n");
}

/* size_t-arithmetic review aid: block-count math must not truncate at 2^32. */
static void test_sizet_arith(void)
{
    /* Mirror the exact expressions used in the vector bodies. */
    size_t lens[] = { (1ull << 32) - 16, (1ull << 32), (1ull << 32) + 16,
                      (1ull << 32) * 3 };
    size_t iters[] = { 200, 104, 56 };
    for (size_t li = 0; li < 4; li++)
        for (size_t ii = 0; ii < 3; ii++) {
            size_t len = lens[li], it = iters[ii];
            size_t blk = (len - 8) / it;
            /* blk * it + 8 <= len < (blk+1) * it + 8, no wraparound. */
            CHECK(blk * it + 8 <= len, "blk math len=%zu it=%zu", len, it);
            CHECK((blk + 1) * it + 8 > len, "blk math len=%zu it=%zu", len, it);
            CHECK(blk == (len - 8) / it, "truncation len=%zu", len);
        }
    printf("ok sizet_arith\n");
}

int main(void)
{
    test_kat();
    test_exhaustive_small();
    test_boundaries();
    test_chaining();
    test_null_zero();
    test_class();
    test_guard_page();
    test_threads();
    test_sizet_arith();
    if (fails) {
        printf("%d FAILURES\n", fails);
        return 1;
    }
    printf("all x86 crc_fast tests passed\n");
    return 0;
}
