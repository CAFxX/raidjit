/* W3 worker verification: RISC-V fast-CRC variants vs oracles.
 * Portable C; compiles natively and under riscv64/QEMU.
 * (Kept in hidden_files as worker verification evidence, not part of
 * the shipped test suite -- W4 owns tests/test_crc_fast.c.)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "riscv/crc_fast.h"
#include "crc.h"

static int failures = 0;
#define CHECK(cond, ...) do { \
    if (!(cond)) { printf("FAIL %s:%d: ", __func__, __LINE__); \
        printf(__VA_ARGS__); printf("\n"); failures++; } \
} while (0)

typedef uint32_t (*fn_t)(uint32_t, const void *, size_t);

static uint64_t rng_state;
static uint64_t rng_next(void) {
    uint64_t x = rng_state;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    rng_state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static void fill(uint8_t *p, size_t n, uint64_t seed) {
    rng_state = seed ? seed : 0x9E3779B97F4A7C15ULL;
    for (size_t i = 0; i < n; i++) p[i] = (uint8_t)rng_next();
}

static void test_variant(const char *name, fn_t f)
{
    { /* KAT: final(init+update) == catalog check value */
        uint32_t c = 0xFFFFFFFFu;
        c = f(c, "123456789", 9);
        c ^= 0xFFFFFFFFu;
        CHECK(c == 0xCBF43926u, "%s KAT: got 0x%08x", name, c);
    }
    for (size_t n = 0; n <= 256; n++) {
        for (int s = 0; s < 3; s++) {
            uint8_t buf[288];
            fill(buf + 7, n + 16, 0x1234 + n * 131 + s * 7919);
            const uint8_t *p = buf + 7;
            uint32_t crc0 =
                (uint32_t)(0xDEADBEEFu ^ (n * 2654435761u) ^ (s * 40503u));
            uint32_t got = f(crc0, p, n);
            CHECK(got == rj_crc32_update(crc0, p, n),
                  "%s len %zu seed %d", name, n, s);
            CHECK(got == rj_crc32_update_scalar(crc0, p, n),
                  "%s len %zu seed %d vs scalar", name, n, s);
        }
    }
    {
        size_t lens[] = {15,16,17,31,32,33,47,48,49,63,64,65,127,128,129,
                         1000,10000};
        for (size_t li = 0; li < sizeof(lens)/sizeof(lens[0]); li++) {
            size_t n = lens[li];
            uint8_t *buf = malloc(n + 32);
            for (int mis = 0; mis < 16; mis++) {
                fill(buf + mis, n, 0x777 + n + mis);
                const uint8_t *p = buf + mis;
                uint32_t crc0 = 0x12345678u ^ (uint32_t)n;
                CHECK(f(crc0, p, n) == rj_crc32_update(crc0, p, n),
                      "%s len %zu mis %d", name, n, mis);
            }
            free(buf);
        }
    }
    { /* split-point chaining at every offset */
        for (size_t n = 0; n <= 128; n++) {
            uint8_t buf[160];
            fill(buf, n, 0xABCD + n);
            uint32_t crc0 = 0x0F0F0F0Fu ^ (uint32_t)n;
            uint32_t whole = f(crc0, buf, n);
            for (size_t sp = 0; sp <= n; sp++) {
                uint32_t mid = f(crc0, buf, sp);
                CHECK(f(mid, buf + sp, n - sp) == whole,
                      "%s chain len %zu split %zu", name, n, sp);
            }
        }
    }
    { /* len 0: NULL and non-NULL, garbage crc -> identity */
        uint8_t dummy = 0xAA;
        CHECK(f(0x12345678u, NULL, 0) == 0x12345678u, "%s len0 NULL", name);
        CHECK(f(0x12345678u, &dummy, 0) == 0x12345678u, "%s len0", name);
        CHECK(f(0xFFFFFFFFu, NULL, 0) == 0xFFFFFFFFu, "%s len0 init", name);
    }
    printf("ok %s\n", name);
}

static void test_dispatch(void)
{
    rj_cpu_info info;
    memset(&info, 0, sizeof(info));
    CHECK(rj_crc32_fast_riscv_class_for(&info) == 0, "class neither");
    CHECK(rj_crc32_fast_riscv_fn_for(&info) == rj_crc32_update, "fn neither");
    info.zbc = 1;
    CHECK(rj_crc32_fast_riscv_class_for(&info) == 1, "class zbc");
    CHECK(rj_crc32_fast_riscv_fn_for(&info) == rj_crc32_fast_riscv_zbc, "fn zbc");
    info.rvv_zvbc = 1;
    CHECK(rj_crc32_fast_riscv_class_for(&info) == 2, "class zvbc");
    CHECK(rj_crc32_fast_riscv_fn_for(&info) == rj_crc32_fast_riscv_zvbc,
          "fn zvbc");
    memset(&info, 0, sizeof(info));
    info.rvv_zvbc = 1;
    CHECK(rj_crc32_fast_riscv_class_for(&info) == 2, "class zvbc alone");
    printf("ok dispatch\n");
}

int main(void)
{
    test_variant("zvbc", rj_crc32_fast_riscv_zvbc);
    test_variant("zbc", rj_crc32_fast_riscv_zbc);
    test_dispatch();
    {
        uint8_t buf[100];
        fill(buf, 100, 42);
        int cls = rj_crc32_fast_riscv_class();
        fn_t forced = cls == 2 ? rj_crc32_fast_riscv_zvbc
                    : cls == 1 ? rj_crc32_fast_riscv_zbc
                    : rj_crc32_update;
        for (size_t n = 0; n < 100; n++)
            CHECK(rj_crc32_fast_riscv(0xFFFFFFFFu, buf, n) ==
                  forced(0xFFFFFFFFu, buf, n),
                  "best != class %d at len %zu", cls, n);
        printf("ok best==class (class=%d)\n", cls);
    }
    if (failures) { printf("%d FAILURES\n", failures); return 1; }
    printf("ALL GREEN\n");
    return 0;
}
