/* Standalone fast-CRC API tests (Option C, Phase 1; spec R4/R5).
 *
 * What this proves, per section:
 *   kat            CRC32("123456789") == 0xCBF43926 through the public
 *                  API and every pinned backend entry point.
 *   bitexact       every variant bit-exact vs the independent bitwise
 *                  oracle: lengths 0..256 exhaustive x misalignments
 *                  0..15 x 3 seeds x varied running CRC. (The 0..256
 *                  sweep subsumes the 16B/64B tail classes: every
 *                  residue mod 16/64 is covered.) Scalar-vs-bitwise and
 *                  zlib-vs-bitwise agreement is checked on strides, so
 *                  the oracle chain is anchored, not circular.
 *   large          {1000, 10000, 1<<20} x misalignments x seeds.
 *   chaining       f(c,A||B) == f(f(c,A),B) at every split offset mod 64
 *                  (spec review R1-3: catches variants that are right
 *                  on whole buffers but wrong across chunk boundaries).
 *   fuzz_zero      len 0 with NULL / non-NULL data and garbage CRCs.
 *   guard          data ending 1 byte before (and starting 1 byte after)
 *                  a PROT_NONE page: any vector over-read faults loudly
 *                  (ASan build turns it into a sanitizer report).
 *   huge           2^32-16 / 2^32 / 2^32+16 differential vs slice-by-8
 *                  (spec review R1-9: 32-bit block-count truncation).
 *                  Uses MAP_NORESERVE zero pages: reads, no 4GB RAM.
 *                  Skipped with RJ_CRC_SKIP_HUGE=1 (QEMU/ASan runs).
 *   threads        8 threads x 200 iterations concurrent dispatch.
 *   dispatch       backend class() vs independent hardware detection;
 *                  best-entry == forced variant for the class. While the
 *                  backend TU is not linked (W1/W2/W3 in flight) these
 *                  go through the DEFERRED path (loud NOTE, counted);
 *                  RJ_CRC_STRICT=1 turns deferrals into failures.
 *   oneshot        rj_crc32() routes through rj_crc32_fast (spec R1-5).
 *
 * Oracles (independent of the code under test):
 *   - the catalog check value 0xCBF43926 (external ground truth);
 *   - crc32_bitwise below: straight from the polynomial, shares no code
 *     with src/crc.c;
 *   - zlib crc32() when linked (RJ_CRC_HAVE_ZLIB, see Makefile);
 *   - rj_crc32_update_scalar (independent stepping from slice-by-8).
 *
 * Non-tautological: tests/mutate_crc.sh flips a constant in the CRC
 * sources and expects this binary to FAIL (red-green evidence).
 */
#define _GNU_SOURCE /* mmap, getauxval prototypes */
#include "../include/raidjit.h"

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#if defined(__aarch64__) || defined(__riscv)
#include <sys/auxv.h>
#endif
#ifdef RJ_CRC_HAVE_ZLIB
#include <zlib.h>
#endif

/* Pinned backend entry points (spec R2.5). Declared here -- not via the
 * backend private headers -- so this test builds before W1/W2/W3 land;
 * the declarations match the spec contract exactly. */
typedef uint32_t (*crc_fn_t)(uint32_t, const void *, size_t);
#if defined(__x86_64__)
uint32_t rj_crc32_fast_x86(uint32_t crc, const void *d, size_t n);
uint32_t rj_crc32_fast_x86_vpclmul(uint32_t crc, const void *d, size_t n);
uint32_t rj_crc32_fast_x86_pclmul(uint32_t crc, const void *d, size_t n);
uint32_t rj_crc32_fast_x86_crc32hw(uint32_t crc, const void *d, size_t n);
int rj_crc32_fast_x86_class(void);
#elif defined(__aarch64__)
uint32_t rj_crc32_fast_arm(uint32_t crc, const void *d, size_t n);
uint32_t rj_crc32_fast_arm_fusion(uint32_t crc, const void *d, size_t n);
uint32_t rj_crc32_fast_arm_crc32hw(uint32_t crc, const void *d, size_t n);
uint32_t rj_crc32_fast_arm_pmull(uint32_t crc, const void *d, size_t n);
int rj_crc32_fast_arm_class(void);
#elif defined(__riscv)
uint32_t rj_crc32_fast_riscv(uint32_t crc, const void *d, size_t n);
uint32_t rj_crc32_fast_riscv_zvbc(uint32_t crc, const void *d, size_t n);
uint32_t rj_crc32_fast_riscv_zbc(uint32_t crc, const void *d, size_t n);
int rj_crc32_fast_riscv_class(void);
#endif
/* Test-only scaffolding from src/crc_fast_fallback.c: 1 once a backend
 * TU's strong definitions are linked, 0 while the weak fallback holds. */
int rj_crc32_fast_test_backend_present(void);

static int failures = 0;
static int deferred = 0;
#define CHECK(cond, ...)                                                   \
    do {                                                                   \
        if (!(cond)) {                                                     \
            printf("FAIL %s:%d: ", __func__, __LINE__);                    \
            printf(__VA_ARGS__);                                           \
            printf("\n");                                                  \
            failures++;                                                    \
        }                                                                  \
    } while (0)
/* NOTE: a check that cannot run yet (backend TU not linked). Loud,
 * counted, and fatal under RJ_CRC_STRICT=1 -- never a silent pass. */
#define NOTE(...)                                                          \
    do {                                                                   \
        printf("NOTE %s:%d: ", __func__, __LINE__);                        \
        printf(__VA_ARGS__);                                               \
        printf("\n");                                                      \
        deferred++;                                                        \
    } while (0)

struct crc_variant {
    const char *name;
    crc_fn_t fn;
};

#if defined(__x86_64__)
static const struct crc_variant variants[] = {
    { "public", rj_crc32_fast },
    { "x86/best", rj_crc32_fast_x86 },
    { "x86/vpclmul", rj_crc32_fast_x86_vpclmul },
    { "x86/pclmul", rj_crc32_fast_x86_pclmul },
    { "x86/crc32hw", rj_crc32_fast_x86_crc32hw },
};
#elif defined(__aarch64__)
static const struct crc_variant variants[] = {
    { "public", rj_crc32_fast },
    { "arm/best", rj_crc32_fast_arm },
    { "arm/fusion", rj_crc32_fast_arm_fusion },
    { "arm/crc32hw", rj_crc32_fast_arm_crc32hw },
    { "arm/pmull", rj_crc32_fast_arm_pmull },
};
#elif defined(__riscv)
static const struct crc_variant variants[] = {
    { "public", rj_crc32_fast },
    { "riscv/best", rj_crc32_fast_riscv },
    { "riscv/zvbc", rj_crc32_fast_riscv_zvbc },
    { "riscv/zbc", rj_crc32_fast_riscv_zbc },
};
#else
static const struct crc_variant variants[] = {
    { "public", rj_crc32_fast },
};
#endif
#define NVAR (sizeof(variants) / sizeof(variants[0]))

static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;
static void rng_seed(uint64_t s)
{
    rng_state = s ? s : 0x9E3779B97F4A7C15ULL;
}
static uint64_t xrng(void)
{
    uint64_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return rng_state = x;
}
/* Thread-local variant: the worker threads must not share rng_state. */
static uint64_t xrng_step(uint64_t *s)
{
    uint64_t x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return *s = x;
}

static uint8_t *tbuf;
static size_t tbuf_n;
static void tbuf_ensure(size_t n)
{
    if (n <= tbuf_n)
        return;
    free(tbuf);
    tbuf = NULL;
    tbuf_n = 0;
    /* Over-allocate: callers index tbuf[0..n+64). */
    if (posix_memalign((void **)&tbuf, 64, n + 64)) {
        fprintf(stderr, "oom\n");
        exit(1);
    }
    tbuf_n = n + 64;
}
static void tbuf_fill(uint64_t seed, size_t n)
{
    tbuf_ensure(n);
    rng_seed(seed);
    for (size_t i = 0; i < n; i++)
        tbuf[i] = (uint8_t)xrng();
}

/* Independent bitwise CRC-32/ISO-HDLC, straight from the polynomial
 * definition. Shares no code with src/crc.c. Takes the running CRC. */
static uint32_t crc32_bitwise(uint32_t crc, const uint8_t *p, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        crc ^= p[i];
        for (int k = 0; k < 8; k++)
            crc = (crc & 1) ? (crc >> 1) ^ 0xEDB88320u : crc >> 1;
    }
    return crc;
}

static void test_kat(void)
{
    static const uint8_t msg[] = "123456789";
    for (size_t v = 0; v < NVAR; v++) {
        uint32_t c = variants[v].fn(0xFFFFFFFFu, msg, 9);
        CHECK(c == 0xCBF43926u, "KAT variant %s: got 0x%08X, want 0xCBF43926",
              variants[v].name, c);
    }
    /* The bitwise oracle agrees with the catalog value (anchors the
     * oracle chain; a broken oracle cannot hide behind the KAT). */
    CHECK(crc32_bitwise(0xFFFFFFFFu, msg, 9) == 0xCBF43926u,
          "bitwise oracle disagrees with catalog KAT");
}

static void test_bitexact(void)
{
    const uint64_t seeds[] = {
        0x123456789ABCDEFULL, 0xDEADBEEFCAFEBABEULL, 0x9E3779B97F4A7C15ULL
    };
    for (int si = 0; si < 3; si++) {
        for (size_t len = 0; len <= 256; len++) {
            tbuf_fill(seeds[si] ^ (len * 0x9E3779B9u), len + 16);
            for (int m = 0; m < 16; m++) {
                const uint8_t *p = tbuf + m;
                uint32_t crc0 =
                    (len & 1) ? 0xFFFFFFFFu : 0x12345678u;
                uint32_t exp = crc32_bitwise(crc0, p, len);
                /* Oracle-agreement strides (not every combo: bitwise
                 * is the reference; these prove the other oracles
                 * agree with it). */
                if (((len + (size_t)m + (size_t)si) % 7) == 0)
                    CHECK(rj_crc32_update_scalar(crc0, p, len) == exp,
                          "scalar vs bitwise disagree len=%zu m=%d",
                          len, m);
#ifdef RJ_CRC_HAVE_ZLIB
                if (((len + (size_t)m) % 11) == 0)
                    CHECK(crc32(crc0, p, (uInt)len) == exp,
                          "zlib vs bitwise disagree len=%zu m=%d",
                          len, m);
#endif
                for (size_t v = 0; v < NVAR; v++)
                    CHECK(variants[v].fn(crc0, p, len) == exp,
                          "variant %s len=%zu mis=%d seed=%d: "
                          "got 0x%08X want 0x%08X",
                          variants[v].name, len, m, si,
                          variants[v].fn(crc0, p, len), exp);
            }
        }
    }
    /* len 0: NULL and non-NULL data, garbage running CRCs. */
    static const uint32_t crcs[] = { 0xFFFFFFFFu, 0x00000000u, 0xDEADBEEFu };
    for (size_t ci = 0; ci < 3; ci++)
        for (size_t v = 0; v < NVAR; v++) {
            CHECK(variants[v].fn(crcs[ci], NULL, 0) == crcs[ci],
                  "variant %s len=0 NULL changed crc", variants[v].name);
            CHECK(variants[v].fn(crcs[ci], tbuf, 0) == crcs[ci],
                  "variant %s len=0 non-NULL changed crc",
                  variants[v].name);
        }
}

static void test_large(void)
{
    const size_t lens[] = { 1000, 10000, 1 << 20 };
    const int mis[] = { 0, 1, 7, 15 };
    const uint64_t seeds[] = { 0xABCDEF0112345678ULL, 0x13579BDF2468ACE0ULL };
    for (size_t li = 0; li < 3; li++)
        for (size_t mi = 0; mi < 4; mi++)
            for (int si = 0; si < 2; si++) {
                size_t len = lens[li];
                tbuf_fill(seeds[si] ^ (len * 0x51ED), len + 16);
                const uint8_t *p = tbuf + mis[mi];
                uint32_t exp = crc32_bitwise(0xFFFFFFFFu, p, len);
#ifdef RJ_CRC_HAVE_ZLIB
                CHECK(crc32(0xFFFFFFFFu, p, (uInt)len) == exp,
                      "zlib vs bitwise disagree len=%zu", len);
#endif
                for (size_t v = 0; v < NVAR; v++)
                    CHECK(variants[v].fn(0xFFFFFFFFu, p, len) == exp,
                          "variant %s len=%zu mis=%d: got 0x%08X want 0x%08X",
                          variants[v].name, len, mis[mi],
                          variants[v].fn(0xFFFFFFFFu, p, len), exp);
            }
}

/* Split-point chaining (spec review R1-3): a variant that is right on
 * whole buffers but wrong across chunk boundaries must fail here. */
static void test_chaining(void)
{
    static const size_t clens[] = {
        0, 1, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 256, 1000, 4096
    };
    const uint64_t seeds[] = { 0xF00D1234ABCD9876ULL, 0x1122334455667788ULL };
    for (size_t li = 0; li < sizeof(clens) / sizeof(clens[0]); li++)
        for (int si = 0; si < 2; si++) {
            size_t L = clens[li];
            tbuf_fill(seeds[si] ^ (L * 0x2545F491u), L + 64);
            const uint8_t *p = tbuf;
            uint32_t crc0 = si ? 0xFFFFFFFFu : 0xA5A5A5A5u;
            for (int s = 0; s < 64; s++) {
                size_t split = L == 0 ? 0 : (size_t)s % (L + 1);
                for (size_t v = 0; v < NVAR; v++) {
                    uint32_t whole = variants[v].fn(crc0, p, L);
                    uint32_t part =
                        variants[v].fn(variants[v].fn(crc0, p, split),
                                       p + split, L - split);
                    CHECK(whole == part,
                          "variant %s chaining L=%zu split=%zu: "
                          "whole=0x%08X part=0x%08X",
                          variants[v].name, L, split, whole, part);
                }
            }
        }
}

/* Guard pages: any vector over-read past the end (or before the start)
 * of the buffer faults loudly instead of silently succeeding. */
static void test_guard_pages(void)
{
    long ps = sysconf(_SC_PAGESIZE);
    if (ps < 64) {
        NOTE("page size %ld too small; guard test skipped", ps);
        return;
    }
    size_t P = (size_t)ps;
    uint8_t *base =
        mmap(NULL, 3 * P, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED) {
        NOTE("mmap failed; guard test skipped");
        return;
    }
    if (mprotect(base, P, PROT_NONE) || mprotect(base + 2 * P, P, PROT_NONE)) {
        NOTE("mprotect failed; guard test skipped");
        munmap(base, 3 * P);
        return;
    }
    uint8_t copy[128];
    /* Tail guard: data ends exactly where the PROT_NONE page begins. */
    for (size_t len = 1; len <= 64; len++) {
        uint8_t *p = base + 2 * P - len;
        for (size_t i = 0; i < len; i++)
            p[i] = (uint8_t)(0xA5 + i * 3);
        memcpy(copy, p, len);
        uint32_t exp = crc32_bitwise(0xFFFFFFFFu, copy, len);
        for (size_t v = 0; v < NVAR; v++)
            CHECK(variants[v].fn(0xFFFFFFFFu, p, len) == exp,
                  "variant %s tail-guard len=%zu (over-read?)",
                  variants[v].name, len);
    }
    /* Head guard: data starts 1 byte after the PROT_NONE page. */
    for (size_t len = 1; len <= 64; len++) {
        uint8_t *p = base + P + 1;
        for (size_t i = 0; i < len; i++)
            p[i] = (uint8_t)(0x5A + i * 7);
        memcpy(copy, p, len);
        uint32_t exp = crc32_bitwise(0xFFFFFFFFu, copy, len);
        for (size_t v = 0; v < NVAR; v++)
            CHECK(variants[v].fn(0xFFFFFFFFu, p, len) == exp,
                  "variant %s head-guard len=%zu (under-read?)",
                  variants[v].name, len);
    }
    munmap(base, 3 * P);
}

/* 2^32 block-count truncation (spec review R1-9): a variant that counts
 * blocks in 32 bits computes the CRC of the truncated length instead.
 * Zero pages via MAP_NORESERVE: the reads are real, the RAM is not. */
static void test_huge(void)
{
    if (getenv("RJ_CRC_SKIP_HUGE")) {
        printf("SKIP huge (RJ_CRC_SKIP_HUGE set)\n");
        return;
    }
    static const unsigned long long sizes[] = {
        (1ULL << 32) - 16, (1ULL << 32), (1ULL << 32) + 16
    };
    for (size_t si = 0; si < 3; si++) {
        size_t n = (size_t)sizes[si];
        uint8_t *z =
            mmap(NULL, n + 64, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (z == MAP_FAILED) {
            NOTE("huge mmap failed for %llu bytes; skipped", sizes[si]);
            continue;
        }
        /* Prove the mapping is real (first/last page); the rest stays
         * on the shared zero page -- reads, not RAM. */
        z[0] = 0;
        z[n - 1] = 0;
        uint32_t exp = rj_crc32_update(0xFFFFFFFFu, z, n);
        for (size_t v = 0; v < NVAR; v++)
            CHECK(variants[v].fn(0xFFFFFFFFu, z, n) == exp,
                  "variant %s size 2^32%+lld: got 0x%08X want 0x%08X "
                  "(32-bit truncation?)",
                  variants[v].name,
                  (long long)sizes[si] - (long long)(1ULL << 32),
                  variants[v].fn(0xFFFFFFFFu, z, n), exp);
        munmap(z, n + 64);
    }
}

#define T_NTHREAD 8
#define T_ITERS 200
static uint8_t t_data[1 << 20];
static uint32_t t_expected[T_NTHREAD * T_ITERS];
static pthread_mutex_t t_mu = PTHREAD_MUTEX_INITIALIZER;

static void *t_worker(void *arg)
{
    long id = (long)arg;
    uint64_t st = 0x1000 + (uint64_t)id; /* per-thread stream */
    for (int i = 0; i < T_ITERS; i++) {
        size_t off = (size_t)(xrng_step(&st) % (sizeof(t_data) - 8192));
        size_t len = 1 + (size_t)(xrng_step(&st) % 8192);
        uint32_t crc = rj_crc32_fast(0x9E3779B9u, t_data + off, len);
        if (crc != t_expected[id * T_ITERS + i]) {
            pthread_mutex_lock(&t_mu);
            printf("FAIL t_worker:%d: thread %ld iter %d: "
                   "got 0x%08X want 0x%08X\n",
                   __LINE__, id, i, crc,
                   t_expected[id * T_ITERS + i]);
            failures++;
            pthread_mutex_unlock(&t_mu);
        }
    }
    return NULL;
}

static void test_threads(void)
{
    rng_seed(0x5EED);
    for (size_t i = 0; i < sizeof(t_data); i++)
        t_data[i] = (uint8_t)xrng();
    /* Precompute single-threaded with the same per-thread streams. */
    for (long id = 0; id < T_NTHREAD; id++) {
        uint64_t st = 0x1000 + (uint64_t)id;
        for (int i = 0; i < T_ITERS; i++) {
            size_t off = (size_t)(xrng_step(&st) % (sizeof(t_data) - 8192));
            size_t len = 1 + (size_t)(xrng_step(&st) % 8192);
            t_expected[id * T_ITERS + i] =
                crc32_bitwise(0x9E3779B9u, t_data + off, len);
        }
    }
    pthread_t th[T_NTHREAD];
    for (long id = 0; id < T_NTHREAD; id++)
        if (pthread_create(&th[id], NULL, t_worker, (void *)id)) {
            NOTE("pthread_create failed; thread test skipped");
            return;
        }
    for (long id = 0; id < T_NTHREAD; id++)
        pthread_join(th[id], NULL);
}

/* Dispatch: backend class() vs INDEPENDENT hardware detection (not the
 * library's rj_cpu_detect -- the selection must match the machine even
 * if detection wiring moves). While the backend TU is not linked these
 * checks are DEFERRED (loud NOTE); they go strict automatically once
 * the backend lands, and RJ_CRC_STRICT=1 makes deferral fatal now. */
#if defined(__x86_64__)
static void x86_cpuid(unsigned leaf, unsigned sub, unsigned *a, unsigned *b,
                      unsigned *c, unsigned *d)
{
    __asm__ volatile("cpuid"
                     : "=a"(*a), "=b"(*b), "=c"(*c), "=d"(*d)
                     : "a"(leaf), "c"(sub));
}

static void test_dispatch(void)
{
    unsigned a, b, c, d;
    x86_cpuid(1, 0, &a, &b, &c, &d);
    int hw_pclmul = (c >> 1) & 1;
    int hw_sse42 = (c >> 20) & 1;
    int hw_vpclmul = 0;
    x86_cpuid(0, 0, &a, &b, &c, &d);
    if (a >= 7) {
        x86_cpuid(7, 0, &a, &b, &c, &d);
        hw_vpclmul = (c >> 10) & 1;
    }
    /* Spec R2 chain, best first (Finding 13: class 3 covers BOTH vpclmul
     * fusion rungs -- vpclmulqdq&&avx512f and vpclmulqdq&&!avx512f --
     * the rj_crc32_fast_x86_vpclmul entry point selects the rung
     * internally, so the class test does not distinguish them). */
    int expected = hw_vpclmul ? 3 : hw_pclmul ? 2 : hw_sse42 ? 1 : 0;
    int present = rj_crc32_fast_test_backend_present();
    int strict = getenv("RJ_CRC_STRICT") != NULL;
    if (!present) {
        NOTE("x86 backend TU not linked (sibling W1 in flight): "
             "class/best-variant assertions DEFERRED; re-run with "
             "RJ_CRC_STRICT=1 to fail here");
        deferred++; /* second deferred check, counted explicitly */
        if (strict) {
            printf("FAIL %s:%d: RJ_CRC_STRICT=1 with no backend linked\n",
                   __func__, __LINE__);
            failures++;
        }
        return;
    }
    int cls = rj_crc32_fast_x86_class();
    CHECK(cls == expected,
          "x86 class %d != expected %d "
          "(hw: vpclmul=%d pclmul=%d sse42=%d)",
          cls, expected, hw_vpclmul, hw_pclmul, hw_sse42);
    /* Informational only: the library's own rj_cpu_detect() may not
     * report the new flags yet (their wiring is backend-worker scope);
     * selection is verified against hardware CPUID directly above, so
     * a mismatch here is a NOTE, never a failure. */
    {
        rj_cpu_info ci;
        rj_cpu_detect(&ci);
        if (ci.pclmulqdq != hw_pclmul || ci.vpclmulqdq != hw_vpclmul ||
            ci.sse42 != hw_sse42)
            NOTE("rj_cpu_detect flags differ from CPUID "
                 "(pclmul %d/%d vpclmul %d/%d sse42 %d/%d): "
                 "detection wiring pending",
                 ci.pclmulqdq, hw_pclmul, ci.vpclmulqdq, hw_vpclmul,
                 ci.sse42, hw_sse42);
    }
}
#elif defined(__aarch64__)
#ifndef HWCAP_CRC32
#define HWCAP_CRC32 (1UL << 7)
#endif
#ifndef HWCAP_PMULL
#define HWCAP_PMULL (1UL << 4)
#endif
#ifndef HWCAP_SHA3
#define HWCAP_SHA3 (1UL << 17)
#endif

static void test_dispatch(void)
{
    unsigned long hwcap = getauxval(AT_HWCAP);
    int hw_crc32 = (hwcap & HWCAP_CRC32) != 0;
    int hw_pmull = (hwcap & HWCAP_PMULL) != 0;
    int hw_sha3 = (hwcap & HWCAP_SHA3) != 0;
    /* Spec R2 chain: fusion, then crc32hw BEFORE pmull (the documented
     * N1 inversion: pure PMULL fold is slower than scalar crc32x there). */
    int expected =
        (hw_pmull && hw_sha3) ? 3 : hw_crc32 ? 1 : hw_pmull ? 2 : 0;
    int present = rj_crc32_fast_test_backend_present();
    int strict = getenv("RJ_CRC_STRICT") != NULL;
    if (!present) {
        NOTE("arm backend TU not linked (sibling W2 in flight): "
             "class/best-variant assertions DEFERRED");
        deferred++;
        if (strict) {
            printf("FAIL %s:%d: RJ_CRC_STRICT=1 with no backend linked\n",
                   __func__, __LINE__);
            failures++;
        }
        return;
    }
    int cls = rj_crc32_fast_arm_class();
    CHECK(cls == expected,
          "arm class %d != expected %d "
          "(hw: pmull=%d sha3=%d crc32=%d)",
          cls, expected, hw_pmull, hw_sha3, hw_crc32);
}
#elif defined(__riscv)
static int cpuinfo_has_isa_token(const char *tok)
{
    /* Mirrors the detection pattern in src/riscv/cpu.c: QEMU user-mode
     * synthesizes the guest isa line from its -cpu flags. */
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (!f)
        return -1;
    char line[1024];
    int found = 0;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "isa", 3) != 0)
            continue;
        char *p = strchr(line, ':');
        if (!p)
            continue;
        for (char *t = strtok(p + 1, " \t\n"); t;
             t = strtok(NULL, " \t\n")) {
            if (strcmp(t, tok) == 0) {
                found = 1;
                break;
            }
        }
    }
    fclose(f);
    return found;
}

static void test_dispatch(void)
{
    int hw_zvbc = cpuinfo_has_isa_token("zvbc");
    int hw_zbc = cpuinfo_has_isa_token("zbc");
    int present = rj_crc32_fast_test_backend_present();
    int strict = getenv("RJ_CRC_STRICT") != NULL;
    if (hw_zvbc < 0 || hw_zbc < 0) {
        NOTE("no /proc/cpuinfo isa line; riscv dispatch check skipped");
        return;
    }
    int expected = hw_zvbc ? 2 : hw_zbc ? 1 : 0;
    if (!present) {
        NOTE("riscv backend TU not linked (sibling W3 in flight): "
             "class/best-variant assertions DEFERRED");
        deferred++;
        if (strict) {
            printf("FAIL %s:%d: RJ_CRC_STRICT=1 with no backend linked\n",
                   __func__, __LINE__);
            failures++;
        }
        return;
    }
    int cls = rj_crc32_fast_riscv_class();
    CHECK(cls == expected,
          "riscv class %d != expected %d (hw: zvbc=%d zbc=%d)",
          cls, expected, hw_zvbc, hw_zbc);
}
#else
static void test_dispatch(void)
{
    /* Unknown ISA: only the slice-by-8 fallback exists; nothing to
     * select. Correctness is covered by the matrix above. */
    printf("SKIP dispatch (no backend contract on this ISA)\n");
}
#endif

/* best-entry == forced variant for the selected class (x86 shape shown;
 * arm/riscv mirror it). Only meaningful with a backend linked. */
#if defined(__x86_64__)
static void test_best_matches_class(void)
{
    if (!rj_crc32_fast_test_backend_present()) {
        NOTE("x86 backend TU not linked: best==forced check DEFERRED");
        if (getenv("RJ_CRC_STRICT")) {
            printf("FAIL %s:%d: RJ_CRC_STRICT=1 with no backend linked\n",
                   __func__, __LINE__);
            failures++;
        }
        return;
    }
    int cls = rj_crc32_fast_x86_class();
    crc_fn_t forced = cls == 3   ? rj_crc32_fast_x86_vpclmul
                      : cls == 2 ? rj_crc32_fast_x86_pclmul
                      : cls == 1 ? rj_crc32_fast_x86_crc32hw
                                 : NULL;
    tbuf_fill(0xBE57C1A55ULL, 1 << 16);
    const uint8_t *p = tbuf + 3; /* misaligned on purpose */
    size_t n = 100003;
    uint32_t crc0 = 0xFFFFFFFFu;
    if (forced)
        CHECK(rj_crc32_fast_x86(crc0, p, n) == forced(crc0, p, n),
              "x86 best != forced variant for class %d", cls);
    else
        CHECK(rj_crc32_fast_x86(crc0, p, n) == rj_crc32_update(crc0, p, n),
              "x86 class-0 best != slice-by-8");
    /* The public entry rides the same dispatch. */
    CHECK(rj_crc32_fast(crc0, p, n) == rj_crc32_fast_x86(crc0, p, n),
          "public rj_crc32_fast != x86 best entry");
}
#elif defined(__aarch64__)
static void test_best_matches_class(void)
{
    if (!rj_crc32_fast_test_backend_present()) {
        NOTE("arm backend TU not linked: best==forced check DEFERRED");
        if (getenv("RJ_CRC_STRICT")) {
            printf("FAIL %s:%d: RJ_CRC_STRICT=1 with no backend linked\n",
                   __func__, __LINE__);
            failures++;
        }
        return;
    }
    int cls = rj_crc32_fast_arm_class();
    crc_fn_t forced = cls == 3   ? rj_crc32_fast_arm_fusion
                      : cls == 1 ? rj_crc32_fast_arm_crc32hw
                      : cls == 2 ? rj_crc32_fast_arm_pmull
                                 : NULL;
    tbuf_fill(0xBE57C1A55ULL, 1 << 16);
    const uint8_t *p = tbuf + 3;
    size_t n = 100003;
    uint32_t crc0 = 0xFFFFFFFFu;
    if (forced)
        CHECK(rj_crc32_fast_arm(crc0, p, n) == forced(crc0, p, n),
              "arm best != forced variant for class %d", cls);
    else
        CHECK(rj_crc32_fast_arm(crc0, p, n) == rj_crc32_update(crc0, p, n),
              "arm class-0 best != slice-by-8");
    CHECK(rj_crc32_fast(crc0, p, n) == rj_crc32_fast_arm(crc0, p, n),
          "public rj_crc32_fast != arm best entry");
}
#elif defined(__riscv)
static void test_best_matches_class(void)
{
    if (!rj_crc32_fast_test_backend_present()) {
        NOTE("riscv backend TU not linked: best==forced check DEFERRED");
        if (getenv("RJ_CRC_STRICT")) {
            printf("FAIL %s:%d: RJ_CRC_STRICT=1 with no backend linked\n",
                   __func__, __LINE__);
            failures++;
        }
        return;
    }
    int cls = rj_crc32_fast_riscv_class();
    crc_fn_t forced = cls == 2   ? rj_crc32_fast_riscv_zvbc
                      : cls == 1 ? rj_crc32_fast_riscv_zbc
                                 : NULL;
    tbuf_fill(0xBE57C1A55ULL, 1 << 16);
    const uint8_t *p = tbuf + 3;
    size_t n = 100003;
    uint32_t crc0 = 0xFFFFFFFFu;
    if (forced)
        CHECK(rj_crc32_fast_riscv(crc0, p, n) == forced(crc0, p, n),
              "riscv best != forced variant for class %d", cls);
    else
        CHECK(rj_crc32_fast_riscv(crc0, p, n) == rj_crc32_update(crc0, p, n),
              "riscv class-0 best != slice-by-8");
    CHECK(rj_crc32_fast(crc0, p, n) == rj_crc32_fast_riscv(crc0, p, n),
          "public rj_crc32_fast != riscv best entry");
}
#else
static void test_best_matches_class(void)
{
    printf("SKIP best==forced (no backend contract on this ISA)\n");
}
#endif

/* rj_crc32() one-shot routes through rj_crc32_fast (spec R1-5). */
static void test_oneshot(void)
{
    CHECK(rj_crc32("123456789", 9) == 0xCBF43926u,
          "rj_crc32 KAT: got 0x%08X", rj_crc32("123456789", 9));
    tbuf_fill(0x0E57E07ULL, 100000);
    /* init + fast + final composes to the one-shot. */
    uint32_t c;
    rj_crc32_init(&c);
    c = rj_crc32_fast(c, tbuf, 100000);
    CHECK(rj_crc32(tbuf, 100000) == rj_crc32_final(c),
          "rj_crc32 != init/fast/final composition");
}

int main(void)
{
    test_kat();
    test_bitexact();
    test_large();
    test_chaining();
    test_guard_pages();
    test_huge();
    test_threads();
    test_dispatch();
    test_best_matches_class();
    test_oneshot();
    free(tbuf);
    printf("test_crc_fast: %s (failures=%d deferred=%d)%s\n",
           failures ? "FAIL" : "PASS", failures, deferred,
           deferred && !failures ? " -- dispatch checks deferred, see NOTEs"
                                 : "");
    return failures != 0;
}
