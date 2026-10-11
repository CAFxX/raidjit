/* AArch64 fast CRC-32 variants (Option C, Phase 1).
 *
 * Three accelerated variants plus internal runtime selection, all sharing
 * rj_crc32_update's chaining contract (running CRC in, running CRC out).
 * Bit-exactness against rj_crc32_update / zlib is proven by the test
 * matrix (tests/test_crc_fast.c, owned by the Phase-1 integrator), not
 * by construction: every constant and every lane-combination below is
 * exercised by lengths 0..256 exhaustive, tail classes, {1000, 10000},
 * misalignments 0..15, multiple seeds, KAT, and split-point chaining.
 *
 * Dispatch order (best first): fusion (pmull && sha3) -> scalar crc32x
 * 3-way -> pure PMULL fold -> slice-by-8. Rationale: on Neoverse N1 the
 * pure PMULL fold is 7.3 B/cycle while scalar crc32x is 7.9 B/cycle (the
 * N1 inversion, FINAL_SYNTHESIS.md); scalar therefore precedes the pure
 * fold, and the fusion exists to use both units at once. Apple cores
 * always have pmull+sha3+crc32 and take tier 1.
 *
 * Variant notes:
 *
 * - rj_crc32_fast_arm_fusion: Cawley 2025 fusion as shipped in
 *   zlib-ng arch/arm/crc32_armv8_pmull_eor3.c (9 PMULL vector lanes at a
 *   144-byte stride + 3 scalar crc32 lanes, EOR3-combined, 192 B/iter),
 *   adapted from the zlib one-shot convention to raidjit's running-CRC
 *   convention (the vendored code complements at entry/exit; here the
 *   entry value is already the running state, so the complements are
 *   dropped) and with the copy path removed. Fold constants
 *   {0x26b70c3d,0x3f41287a} (144 B stride), tree-reduction constants
 *   {0xae689191,0xccaa009e}, {0xf1da05aa,0x81256527},
 *   {0x8f352d95,0x1d9513d7} are Cawley's, used verbatim.
 *
 * - rj_crc32_fast_arm_crc32hw: scalar crc32x 3-way interleave, 24
 *   B/iter, same lane geometry as Cawley's scalar section. The lane
 *   combine needs a CRC "shift" (advance the register by n zero bytes).
 *   Cawley computes it with PMULL; this tier must also work when PMULL
 *   is absent (crc32=1, pmull=0), so the shift is scalarized instead:
 *   crc_shift_scalar(c, n) = low64(clmul32(c, xnmodp_scalar(8n-33))),
 *   mirroring Cawley's formula with a scalar 32x32->64 carryless
 *   multiply and a scalar xnmodp (the __crc32w/__crc32d/vmull_p8 steps
 *   replaced by rj_crc32_update / scalar clmul8; all O(log n)). The
 *   combine itself is Cawley's verbatim, so the scalarization is the
 *   only novel math and is differentially pinned against the PMULL
 *   shift. One code path regardless of PMULL: the shift runs once per
 *   call (microseconds), not worth a second variant.
 *
 * - rj_crc32_fast_arm_pmull: pure PMULL fold, dougallj structure
 *   (thirdparty/fast-crc32/third_party/dougallj.cpp): 4 chains x 16 B,
 *   fold constants k1=x^(4*128+32), k2=x^(4*128-32) mod P =
 *   {0x154442bd4, 0x1c6e41596}, fold-down k3=x^160, k4=x^96 mod P =
 *   {0x1751997d0, 0x0ccaa009e} (all four match zlib-ng's PCLMULQDQ
 *   fold constants, an independent cross-check). The running state is
 *   XORed into the low 32 bits of the first chunk; the 128-bit folded
 *   state is finalized through rj_crc32_update from 0 (16 bytes on the
 *   stack) instead of dougallj's Barrett reduction -- identical
 *   result, far less novel code. Only reached when the CRC32
 *   instructions are absent, so no __crc32* is used here.
 *
 * No over-reads: heads are consumed with rj_crc32_update (exact for any
 * length); vector loads are 16-byte aligned (head aligns to 16); the
 * 8-byte scalar loads are ordinary (alignment-safe) loads. Guard-page
 * tests cover 1 byte before an unmapped page.
 *
 * Compile story: the TU builds on any host. All AArch64 code is inside
 * #ifdef __aarch64__; every intrinsic-using function carries a
 * function-level target attribute for exactly the extensions it needs
 * (the Makefile builds aarch64 with the baseline -march, so without
 * these the PMULL/CRC32/EOR3 intrinsics would not compile). Never raw
 * asm: <arm_neon.h> / <arm_acle.h> intrinsics only.
 */
#include "crc_fast.h"

#ifdef __aarch64__

#include <arm_acle.h>
#include <arm_neon.h>
#include <assert.h>
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "cpu.h"   /* rj_cpu_detect */
#include "crc.h"   /* rj_crc32_update (scalar fallback + head/tail reuse) */

#define RJ_TGT_CRC    __attribute__((target("arch=armv8.2-a+crc")))
#define RJ_TGT_PMULL  __attribute__((target("arch=armv8.2-a+crypto")))
#define RJ_TGT_FUSION __attribute__((target("arch=armv8.2-a+crc+crypto+sha3")))

/* ------------------------------------------------------------------ */
/* Small helpers                                                      */
/* ------------------------------------------------------------------ */

/* 16-byte load that is strict-aliasing clean; compiles to one ldp. */
static inline uint64x2_t vload16(const uint8_t *p)
{
    uint64x2_t v;
    memcpy(&v, p, 16);
    return v;
}

static inline uint64_t load64(const uint8_t *p)
{
    uint64_t v;
    memcpy(&v, p, 8);
    return v;
}

/* Carryless multiply of the low 64-bit lanes: a[0] * b[0] -> 128 bits. */
RJ_TGT_PMULL
static inline uint64x2_t clmul_lo(uint64x2_t a, uint64x2_t b)
{
    return vreinterpretq_u64_p128(vmull_p64(
        vget_lane_p64(vreinterpret_p64_u64(vget_low_u64(a)), 0),
        vget_lane_p64(vreinterpret_p64_u64(vget_low_u64(b)), 0)));
}

/* Carryless multiply of the high 64-bit lanes: a[1] * b[1] -> 128 bits. */
RJ_TGT_PMULL
static inline uint64x2_t clmul_hi(uint64x2_t a, uint64x2_t b)
{
    return vreinterpretq_u64_p128(
        vmull_high_p64(vreinterpretq_p64_u64(a), vreinterpretq_p64_u64(b)));
}

/* Scalar 32x32 -> 64 carryless multiply (for the PMULL-free shift). */
static uint64_t clmul32(uint32_t a, uint32_t b)
{
    uint64_t r = 0, aa = a;
    while (b) {
        if (b & 1)
            r ^= aa;
        aa <<= 1;
        b >>= 1;
    }
    return r;
}

/* Scalar 8x8 -> 16 carryless multiply (scalar vmull_p8). */
static uint16_t clmul8(uint8_t a, uint8_t b)
{
    uint16_t r = 0, aa = a;
    while (b) {
        if (b & 1)
            r ^= aa;
        aa <<= 1;
        b >>= 1;
    }
    return r;
}

/* ------------------------------------------------------------------ */
/* CRC shift without PMULL (crc32x tier's lane combine)               */
/*                                                                    */
/* shift(c, n) advances the CRC register by n zero bytes. Cawley's     */
/* PMULL form is low64(clmul(c, x^(8n-33) mod P)); this is the same    */
/* formula with scalar primitives, so the combine code stays          */
/* identical to the reference. O(log n); exact for every n.           */
/* ------------------------------------------------------------------ */

static uint32_t crc32w_from0_step(uint32_t acc)
{
    /* One __crc32w(acc, 0): advance by 4 zero bytes. */
    static const uint8_t z[4] = { 0, 0, 0, 0 };
    return rj_crc32_update(acc, z, 4);
}

static uint32_t crc32d_from0(uint64_t w)
{
    /* __crc32d(0, w). */
    return rj_crc32_update(0, &w, 8);
}

/* x^n mod P in the reflected domain, scalar version of Cawley's
 * xnmodp (which uses __crc32w/__crc32d/vmull_p8). */
static uint32_t xnmodp_scalar(uint64_t n)
{
    uint64_t stack = ~(uint64_t)1;
    uint32_t acc, low;
    for (; n > 191; n = (n >> 1) - 16)
        stack = (stack << 1) + (n & 1);
    stack = ~stack;
    acc = ((uint32_t)0x80000000) >> (n & 31);
    for (n >>= 5; n; --n)
        acc = crc32w_from0_step(acc);
    while ((low = (uint32_t)(stack & 1)), (stack >>= 1) != 0) {
        /* y = four 16-bit carryless squares, one per byte of acc
         * (lanes 0..3 of vmull_p8); only acc's low 32 bits are live. */
        uint64_t y = 0;
        for (int i = 0; i < 4; i++) {
            uint8_t bi = (uint8_t)(acc >> (8 * i));
            y |= (uint64_t)clmul8(bi, bi) << (16 * i);
        }
        acc = crc32d_from0(y << low);
    }
    return acc;
}

/* Advance the CRC register by nbytes zero bytes, PMULL-free.
 * Differentially pinned against the PMULL crc_shift (fusion tier). */
static uint64_t crc_shift_scalar(uint32_t crc, size_t nbytes)
{
    assert(nbytes >= 5);
    return clmul32(crc, xnmodp_scalar((uint64_t)nbytes * 8 - 33));
}

/* ------------------------------------------------------------------ */
/* PMULL crc_shift (fusion tier), verbatim Cawley                      */
/* ------------------------------------------------------------------ */

/* 32x32 -> 128 carryless multiply (both 32-bit lanes in the low lane). */
RJ_TGT_FUSION
static inline uint64x2_t clmul_scalar(uint32_t a, uint32_t b)
{
    return vreinterpretq_u64_p128(vmull_p64((poly64_t)a, (poly64_t)b));
}

/* x^n mod P, Cawley's xnmodp verbatim (needs __crc32w + vmull_p8). */
RJ_TGT_FUSION
static uint32_t xnmodp(uint64_t n)
{
    uint64_t stack = ~(uint64_t)1;
    uint32_t acc, low;
    for (; n > 191; n = (n >> 1) - 16)
        stack = (stack << 1) + (n & 1);
    stack = ~stack;
    acc = ((uint32_t)0x80000000) >> (n & 31);
    for (n >>= 5; n; --n)
        acc = __crc32w(acc, 0);
    while ((low = (uint32_t)(stack & 1)), (stack >>= 1) != 0) {
        poly8x8_t x = vreinterpret_p8_u64(vmov_n_u64(acc));
        uint64_t y = vgetq_lane_u64(vreinterpretq_u64_p16(vmull_p8(x, x)), 0);
        acc = __crc32d(0, y << low);
    }
    return acc;
}

/* Advance the CRC register by nbytes zero bytes (128-bit vector; the
 * callers use lane 0). Requires nbytes >= 5 (Cawley's precondition). */
RJ_TGT_FUSION
static inline uint64x2_t crc_shift(uint32_t crc, size_t nbytes)
{
    assert(nbytes >= 5);
    return clmul_scalar(crc, xnmodp((uint64_t)nbytes * 8 - 33));
}

/* ------------------------------------------------------------------ */
/* Tail: 8/4/2/1-byte hardware CRC steps (needs +crc only)             */
/* ------------------------------------------------------------------ */

RJ_TGT_CRC
static uint32_t crc_tail_hw(uint32_t crc, const uint8_t *src, size_t len)
{
    while (len >= 8) {
        crc = __crc32d(crc, load64(src));
        src += 8;
        len -= 8;
    }
    if (len & 4) {
        uint32_t v;
        memcpy(&v, src, 4);
        crc = __crc32w(crc, v);
        src += 4;
    }
    if (len & 2) {
        uint16_t v;
        memcpy(&v, src, 2);
        crc = __crc32h(crc, v);
        src += 2;
    }
    if (len & 1)
        crc = __crc32b(crc, *src);
    return crc;
}

/* ------------------------------------------------------------------ */
/* Tier 2: scalar crc32x 3-way interleave                              */
/* ------------------------------------------------------------------ */

RJ_TGT_CRC
uint32_t rj_crc32_fast_arm_crc32hw(uint32_t crc, const void *data, size_t len)
{
    uint32_t crc0 = crc;
    const uint8_t *src = (const uint8_t *)data;

    /* Head: reach 8-byte alignment with the scalar path. */
    {
        size_t h = (8 - ((uintptr_t)src & 7)) & 7;
        if (h > len)
            h = len;
        if (h) {
            crc0 = rj_crc32_update(crc0, src, h);
            src += h;
            len -= h;
        }
    }

    if (len >= 80) {
        /* 3-way interleave, 24 bytes/iteration, Cawley's lane geometry:
         * lane i covers [src + i*klen, src + i*klen + klen). */
        size_t klen = ((len - 8) / 24) * 8;
        const uint8_t *buf0 = src;
        const uint8_t *buf1 = src + klen;
        const uint8_t *buf2 = src + klen * 2;
        uint32_t crc1 = 0, crc2 = 0;
        size_t rem = len;

        do {
            uint64_t v0 = load64(buf0);
            uint64_t v1 = load64(buf1);
            uint64_t v2 = load64(buf2);
            crc0 = __crc32d(crc0, v0);
            crc1 = __crc32d(crc1, v1);
            crc2 = __crc32d(crc2, v2);
            buf0 += 8;
            buf1 += 8;
            buf2 += 8;
            rem -= 24;
        } while (rem >= 32);
        /* Each lane consumed exactly klen bytes (8 * iters == klen);
         * rem = len - 3*klen is in [8, 32). */

        /* Combine (Cawley's, with the scalar shift): shift each lane's
         * CRC past the bytes that follow its lane, XOR the shifted
         * lanes into the next 8 data bytes, finish with the tail. */
        {
            uint64_t vc0 = crc_shift_scalar(crc0, klen * 2 + 8);
            uint64_t vc1 = crc_shift_scalar(crc1, klen + 8);
            uint64_t vc = vc0 ^ vc1;
            uint64_t vf = load64(buf2);
            crc0 = __crc32d(crc2, vf ^ vc);
            src = buf2 + 8;
            len = rem - 8;
        }
    }

    return crc_tail_hw(crc0, src, len);
}

/* ------------------------------------------------------------------ */
/* Tier 3: pure PMULL fold (dougallj structure, 4 chains)              */
/* ------------------------------------------------------------------ */

/* One 16-byte fold step: acc = clmul_lo(acc,k) ^ clmul_hi(acc,k) ^ data. */
RJ_TGT_PMULL
static inline uint64x2_t fold16(uint64x2_t acc, uint64x2_t data, uint64x2_t k)
{
    uint64x2_t lo = clmul_lo(acc, k);
    uint64x2_t hi = clmul_hi(acc, k);
    return veorq_u64(veorq_u64(lo, hi), data);
}

RJ_TGT_PMULL
uint32_t rj_crc32_fast_arm_pmull(uint32_t crc, const void *data, size_t len)
{
    uint32_t st = crc;
    const uint8_t *p = (const uint8_t *)data;

    /* Align to 16 bytes with the scalar path (no over-read). */
    {
        size_t h = (16 - ((uintptr_t)p & 15)) & 15;
        if (h > len)
            h = len;
        if (h) {
            st = rj_crc32_update(st, p, h);
            p += h;
            len -= h;
        }
    }

    if (len >= 64) {
        /* 4 chains fold a 64-byte stride per iteration.
         * k1 = x^(4*128+32), k2 = x^(4*128-32) mod P (reflected). */
        static const uint64_t k1k2[2] __attribute__((aligned(16))) = {
            0x154442bd4ULL, 0x1c6e41596ULL
        };
        /* Fold-down: k3 = x^160, k4 = x^96 mod P (reflected). */
        static const uint64_t k3k4[2] __attribute__((aligned(16))) = {
            0x1751997d0ULL, 0x0ccaa009eULL
        };
        uint64x2_t k = vld1q_u64(k1k2);
        uint64x2_t v0 = vload16(p);
        uint64x2_t v1 = vload16(p + 16);
        uint64x2_t v2 = vload16(p + 32);
        uint64x2_t v3 = vload16(p + 48);
        uint64x2_t x;

        /* Fold the running state into the low 32 bits of chain 0. */
        v0 = veorq_u64(v0, vcombine_u64(vcreate_u64(st), vcreate_u64(0)));

        p += 64;
        len -= 64;
        {
            const uint8_t *fast_end = p + (len / 64) * 64;
            while (p < fast_end) {
                v0 = fold16(v0, vload16(p), k);
                p += 16;
                v1 = fold16(v1, vload16(p), k);
                p += 16;
                v2 = fold16(v2, vload16(p), k);
                p += 16;
                v3 = fold16(v3, vload16(p), k);
                p += 16;
                len -= 64;
            }
        }

        /* Fold the 4 chains down to one 128-bit state. */
        {
            uint64x2_t kd = vld1q_u64(k3k4);
            x = fold16(v0, v1, kd);
            x = fold16(x, v2, kd);
            x = fold16(x, v3, kd);
            while (len >= 16) {
                x = fold16(x, vload16(p), kd);
                p += 16;
                len -= 16;
            }
        }

        /* Finalize: the folded 128-bit state through the CRC unit from
         * 0 (16 bytes on the stack). Bit-identical to dougallj's
         * Barrett reduction, without the extra novel code. */
        {
            uint8_t fin[16];
            vst1q_u8(fin, vreinterpretq_u8_u64(x));
            st = rj_crc32_update(0, fin, 16);
        }
    }

    return rj_crc32_update(st, p, len);
}

/* ------------------------------------------------------------------ */
/* Tier 1: Cawley fusion (9 PMULL lanes + 3 scalar crc32 lanes, EOR3)  */
/* (M1 sub-variant: v9s3x2e_s3)                                       */
/* ------------------------------------------------------------------ */

RJ_TGT_FUSION
uint32_t rj_crc32_fast_arm_fusion(uint32_t crc, const void *data, size_t len)
{
    uint32_t crc0 = crc;
    const uint8_t *src = (const uint8_t *)data;

    if (len == 1) {
        crc0 = __crc32b(crc0, *src);
        return crc0;
    }

    /* Align to 16 bytes for the vector path (scalar head). */
    {
        size_t h = (16 - ((uintptr_t)src & 15)) & 15;
        if (h > len)
            h = len;
        if (h) {
            crc0 = rj_crc32_update(crc0, src, h);
            src += h;
            len -= h;
        }
    }

    if (len >= 192) {
        size_t blk = len / 192;
        size_t klen = blk * 16;
        const uint8_t *end = src + len;
        const uint8_t *src0 = src;
        const uint8_t *src1 = src + klen;
        const uint8_t *src2 = src + klen * 2;
        const uint8_t *srcv = src + klen * 3;
        uint32_t crc1 = 0, crc2 = 0;
        uint64x2_t vc0, vc1, vc2;
        uint64_t vc;
        uint64x2_t x0, x1, x2, x3, x4, x5, x6, x7, x8;
        uint64x2_t y0, y1, y2, y3, y4, y5, y6, y7, y8;
        uint64x2_t k;
        /* k = {x^144, x^144+64} mod P for the 144-byte stride fold. */
        static const uint64_t k144[2] __attribute__((aligned(16))) = {
            0x26b70c3dULL, 0x3f41287aULL
        };

        /* Load the first 9 vector chunks (144 bytes). */
        x0 = vload16(srcv);
        x1 = vload16(srcv + 16);
        x2 = vload16(srcv + 32);
        x3 = vload16(srcv + 48);
        x4 = vload16(srcv + 64);
        x5 = vload16(srcv + 80);
        x6 = vload16(srcv + 96);
        x7 = vload16(srcv + 112);
        x8 = vload16(srcv + 128);
        k = vld1q_u64(k144);
        srcv += 144;

        /* Fold 9 vectors + 3-way parallel scalar CRC. */
        if (blk > 1) {
            const uint8_t *limit = src0 + klen - 32;
            while (src0 <= limit) {
                y0 = clmul_lo(x0, k);
                x0 = clmul_hi(x0, k);
                y1 = clmul_lo(x1, k);
                x1 = clmul_hi(x1, k);
                y2 = clmul_lo(x2, k);
                x2 = clmul_hi(x2, k);
                y3 = clmul_lo(x3, k);
                x3 = clmul_hi(x3, k);
                y4 = clmul_lo(x4, k);
                x4 = clmul_hi(x4, k);
                y5 = clmul_lo(x5, k);
                x5 = clmul_hi(x5, k);
                y6 = clmul_lo(x6, k);
                x6 = clmul_hi(x6, k);
                y7 = clmul_lo(x7, k);
                x7 = clmul_hi(x7, k);
                y8 = clmul_lo(x8, k);
                x8 = clmul_hi(x8, k);

                /* EOR3: fold, data XOR in one instruction per lane. */
                {
                    uint64x2_t d0 = vload16(srcv);
                    uint64x2_t d1 = vload16(srcv + 16);
                    uint64x2_t d2 = vload16(srcv + 32);
                    uint64x2_t d3 = vload16(srcv + 48);
                    uint64x2_t d4 = vload16(srcv + 64);
                    uint64x2_t d5 = vload16(srcv + 80);
                    uint64x2_t d6 = vload16(srcv + 96);
                    uint64x2_t d7 = vload16(srcv + 112);
                    uint64x2_t d8 = vload16(srcv + 128);
                    x0 = veor3q_u64(x0, y0, d0);
                    x1 = veor3q_u64(x1, y1, d1);
                    x2 = veor3q_u64(x2, y2, d2);
                    x3 = veor3q_u64(x3, y3, d3);
                    x4 = veor3q_u64(x4, y4, d4);
                    x5 = veor3q_u64(x5, y5, d5);
                    x6 = veor3q_u64(x6, y6, d6);
                    x7 = veor3q_u64(x7, y7, d7);
                    x8 = veor3q_u64(x8, y8, d8);
                }

                /* 3-way parallel scalar CRC (16 bytes per lane). */
                {
                    uint64_t s0a = load64(src0);
                    uint64_t s0b = load64(src0 + 8);
                    uint64_t s1a = load64(src1);
                    uint64_t s1b = load64(src1 + 8);
                    uint64_t s2a = load64(src2);
                    uint64_t s2b = load64(src2 + 8);
                    crc0 = __crc32d(crc0, s0a);
                    crc0 = __crc32d(crc0, s0b);
                    crc1 = __crc32d(crc1, s1a);
                    crc1 = __crc32d(crc1, s1b);
                    crc2 = __crc32d(crc2, s2a);
                    crc2 = __crc32d(crc2, s2b);
                }
                src0 += 16;
                src1 += 16;
                src2 += 16;
                srcv += 144;
            }
        }

        /* Reduce the 9 vectors to 1 with a tree reduction. */
        {
            static const uint64_t kA[2] __attribute__((aligned(16))) = {
                0xae689191ULL, 0xccaa009eULL
            };
            static const uint64_t kB[2] __attribute__((aligned(16))) = {
                0xf1da05aaULL, 0x81256527ULL
            };
            static const uint64_t kC[2] __attribute__((aligned(16))) = {
                0x8f352d95ULL, 0x1d9513d7ULL
            };
            k = vld1q_u64(kA);
            y0 = clmul_lo(x0, k);
            x0 = clmul_hi(x0, k);
            x0 = veor3q_u64(x0, y0, x1);
            x1 = x2;
            x2 = x3;
            x3 = x4;
            x4 = x5;
            x5 = x6;
            x6 = x7;
            x7 = x8;

            y0 = clmul_lo(x0, k);
            x0 = clmul_hi(x0, k);
            y2 = clmul_lo(x2, k);
            x2 = clmul_hi(x2, k);
            y4 = clmul_lo(x4, k);
            x4 = clmul_hi(x4, k);
            y6 = clmul_lo(x6, k);
            x6 = clmul_hi(x6, k);
            x0 = veor3q_u64(x0, y0, x1);
            x2 = veor3q_u64(x2, y2, x3);
            x4 = veor3q_u64(x4, y4, x5);
            x6 = veor3q_u64(x6, y6, x7);

            k = vld1q_u64(kB);
            y0 = clmul_lo(x0, k);
            x0 = clmul_hi(x0, k);
            y4 = clmul_lo(x4, k);
            x4 = clmul_hi(x4, k);
            x0 = veor3q_u64(x0, y0, x2);
            x4 = veor3q_u64(x4, y4, x6);

            k = vld1q_u64(kC);
            y0 = clmul_lo(x0, k);
            x0 = clmul_hi(x0, k);
            x0 = veor3q_u64(x0, y0, x4);
        }

        /* Final scalar chunk for the 3 scalar lanes. */
        {
            uint64_t s0a = load64(src0);
            uint64_t s0b = load64(src0 + 8);
            uint64_t s1a = load64(src1);
            uint64_t s1b = load64(src1 + 8);
            uint64_t s2a = load64(src2);
            uint64_t s2b = load64(src2 + 8);
            crc0 = __crc32d(crc0, s0a);
            crc0 = __crc32d(crc0, s0b);
            crc1 = __crc32d(crc1, s1a);
            crc1 = __crc32d(crc1, s1b);
            crc2 = __crc32d(crc2, s2a);
            crc2 = __crc32d(crc2, s2b);
        }

        /* Shift and combine the 3 scalar CRCs with the vector state. */
        vc0 = crc_shift(crc0, klen * 2 + blk * 144);
        vc1 = crc_shift(crc1, klen + blk * 144);
        vc2 = crc_shift(crc2, blk * 144);
        vc = vgetq_lane_u64(veor3q_u64(vc0, vc1, vc2), 0);

        /* Fold the 128-bit vector state + combined scalars through the
         * scalar CRC unit. */
        crc0 = __crc32d(0, vgetq_lane_u64(x0, 0));
        crc0 = __crc32d(crc0, vc ^ vgetq_lane_u64(x0, 1));
        src = srcv;
        len = (size_t)(end - srcv);
    }

    if (len >= 80) {
        /* 3-way scalar CRC (24 bytes/iter), Cawley's geometry. */
        size_t klen = ((len - 8) / 24) * 8;
        const uint8_t *buf0 = src;
        const uint8_t *buf1 = src + klen;
        const uint8_t *buf2 = src + klen * 2;
        uint32_t crc1 = 0, crc2 = 0;
        uint64x2_t vc0, vc1;
        uint64_t vc;
        size_t rem = len;

        do {
            uint64_t v0 = load64(buf0);
            uint64_t v1 = load64(buf1);
            uint64_t v2 = load64(buf2);
            crc0 = __crc32d(crc0, v0);
            crc1 = __crc32d(crc1, v1);
            crc2 = __crc32d(crc2, v2);
            buf0 += 8;
            buf1 += 8;
            buf2 += 8;
            rem -= 24;
        } while (rem >= 32);

        /* Combine the 3 CRCs. */
        vc0 = crc_shift(crc0, klen * 2 + 8);
        vc1 = crc_shift(crc1, klen + 8);
        vc = vgetq_lane_u64(veorq_u64(vc0, vc1), 0);

        /* Process the final 8 bytes with the combined CRC. */
        crc0 = crc2;
        crc0 = __crc32d(crc0, load64(buf2) ^ vc);
        src = buf2 + 8;
        len = rem - 8;
    }

    return crc_tail_hw(crc0, src, len);
}

/* ------------------------------------------------------------------ */
/* Tier 1 (N1 sub-variant): Cawley fusion v3s4x2e_v2                  */
/*                                                                    */
/* 3 PMULL vector lanes (48-byte stride) + 4 scalar crc32 lanes, 112  */
/* bytes/iteration. No EOR3 (the generated sample does not use it).   */
/* The lane balance suits cores with PMULL throughput 1/cycle (N1):   */
/* per 112 B: 6 PMULL (6 cycles) + 8 crc32d (8 cycles) -> the CRC      */
/* unit is the bottleneck at ~14 B/c, vs the M1 variant's 9 lanes     */
/* which would saturate N1's single PMULL pipe (~10.7 B/c analytic).  */
/*                                                                    */
/* Structure: Cawley's sample_neon_crc32_v3s4x2e_v2.c (generated by   */
/* fast-crc32), raw asm replaced by intrinsics (fold16 is              */
/* mathematically identical to the sample's clmul_lo_e/hi_e pair),    */
/* zlib one-shot convention adapted to the running-CRC convention,    */
/* head alignment via rj_crc32_update. Constants: 48 B stride         */
/* {0x3db1ecdc, 0xaf449247}, tree/secondary {0xae689191, 0xccaa009e}    */
/* and {0xf1da05aa, 0x81256527}, all Cawley's verbatim.               */
/* ------------------------------------------------------------------ */

RJ_TGT_FUSION
uint32_t rj_crc32_fast_arm_fusion_n1(uint32_t crc, const void *data, size_t len)
{
    uint32_t crc0 = crc;
    const uint8_t *buf = (const uint8_t *)data;

    /* Align to 16 bytes for the vector path (scalar head). */
    {
        size_t h = (16 - ((uintptr_t)buf & 15)) & 15;
        if (h > len)
            h = len;
        if (h) {
            crc0 = rj_crc32_update(crc0, buf, h);
            buf += h;
            len -= h;
        }
    }

    if (len >= 112) {
        const uint8_t *end = buf + len;
        size_t blk = len / 112;
        size_t klen = blk * 16;
        const uint8_t *buf2 = buf + klen * 4;
        const uint8_t *limit = buf + klen - 32;
        uint32_t crc1 = 0, crc2 = 0, crc3 = 0;
        uint64x2_t vc0, vc1, vc2, vc3;
        uint64_t vc;
        uint64x2_t x0, x1, x2, k;
        /* k = {x^48, x^48+64} mod P for the 48-byte stride fold. */
        static const uint64_t k48[2] __attribute__((aligned(16))) = {
            0x3db1ecdcULL, 0xaf449247ULL
        };

        /* Load the first 3 vector chunks (48 bytes). */
        x0 = vload16(buf2);
        x1 = vload16(buf2 + 16);
        x2 = vload16(buf2 + 32);
        k = vld1q_u64(k48);
        buf2 += 48;

        /* Fold 3 vectors + 4-way parallel scalar CRC (16 B per lane). */
        while (buf <= limit) {
            x0 = fold16(x0, vload16(buf2), k);
            x1 = fold16(x1, vload16(buf2 + 16), k);
            x2 = fold16(x2, vload16(buf2 + 32), k);
            crc0 = __crc32d(crc0, load64(buf));
            crc1 = __crc32d(crc1, load64(buf + klen));
            crc2 = __crc32d(crc2, load64(buf + klen * 2));
            crc3 = __crc32d(crc3, load64(buf + klen * 3));
            crc0 = __crc32d(crc0, load64(buf + 8));
            crc1 = __crc32d(crc1, load64(buf + klen + 8));
            crc2 = __crc32d(crc2, load64(buf + klen * 2 + 8));
            crc3 = __crc32d(crc3, load64(buf + klen * 3 + 8));
            buf += 16;
            buf2 += 48;
        }

        /* Reduce the 3 vectors to 1. */
        {
            static const uint64_t kA[2] __attribute__((aligned(16))) = {
                0xae689191ULL, 0xccaa009eULL
            };
            k = vld1q_u64(kA);
            x0 = fold16(x0, x1, k);
            x0 = fold16(x0, x2, k);
        }

        /* Final scalar chunk (16 bytes per lane). */
        crc0 = __crc32d(crc0, load64(buf));
        crc1 = __crc32d(crc1, load64(buf + klen));
        crc2 = __crc32d(crc2, load64(buf + klen * 2));
        crc3 = __crc32d(crc3, load64(buf + klen * 3));
        crc0 = __crc32d(crc0, load64(buf + 8));
        crc1 = __crc32d(crc1, load64(buf + klen + 8));
        crc2 = __crc32d(crc2, load64(buf + klen * 2 + 8));
        crc3 = __crc32d(crc3, load64(buf + klen * 3 + 8));

        /* Shift and combine the 4 scalar CRCs with the vector state. */
        vc0 = crc_shift(crc0, klen * 3 + blk * 48);
        vc1 = crc_shift(crc1, klen * 2 + blk * 48);
        vc2 = crc_shift(crc2, klen + blk * 48);
        vc3 = crc_shift(crc3, blk * 48);
        vc = vgetq_lane_u64(veorq_u64(veorq_u64(vc0, vc1), veorq_u64(vc2, vc3)), 0);

        /* Fold the 128-bit vector state + combined scalars through the
         * scalar CRC unit. */
        crc0 = __crc32d(0, vgetq_lane_u64(x0, 0));
        crc0 = __crc32d(crc0, vc ^ vgetq_lane_u64(x0, 1));
        buf = buf2;
        len = (size_t)(end - buf2);
    }

    if (len >= 32) {
        /* Two-vector PMULL fold for the mid tail, running crc folded
         * into the low 32 bits of x0 (dougallj's init trick). */
        static const uint64_t k32[2] __attribute__((aligned(16))) = {
            0xf1da05aaULL, 0x81256527ULL
        };
        static const uint64_t kA[2] __attribute__((aligned(16))) = {
            0xae689191ULL, 0xccaa009eULL
        };
        uint64x2_t x0 = vload16(buf);
        uint64x2_t x1 = vload16(buf + 16);
        uint64x2_t k = vld1q_u64(k32);
        x0 = veorq_u64(x0, vcombine_u64(vcreate_u64(crc0), vcreate_u64(0)));
        buf += 32;
        len -= 32;
        while (len >= 32) {
            x0 = fold16(x0, vload16(buf), k);
            x1 = fold16(x1, vload16(buf + 16), k);
            buf += 32;
            len -= 32;
        }
        k = vld1q_u64(kA);
        x0 = fold16(x0, x1, k);
        crc0 = __crc32d(0, vgetq_lane_u64(x0, 0));
        crc0 = __crc32d(crc0, vgetq_lane_u64(x0, 1));
    }

    return crc_tail_hw(crc0, buf, len);
}

/* ------------------------------------------------------------------ */
/* Internal runtime selection                                         */
/* ------------------------------------------------------------------ */

/* Parse "CPU implementer" from /proc/cpuinfo (same line-scan pattern
 * as the brand reader in cpu.c). Returns the implementer, or -1 when
 * the field is absent or unparseable. */
static int read_implementer(void)
{
    FILE *f = fopen("/proc/cpuinfo", "r");
    int impl = -1;
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "CPU implementer", 15) == 0) {
                char *c = strchr(line, ':');
                if (c) {
                    unsigned v;
                    if (sscanf(c + 1, " %x", &v) == 1)
                        impl = (int)v;
                }
                break;
            }
        }
        fclose(f);
    }
    return impl;
}

/* Fusion sub-variant: 0 = M1 v9s3x2e_s3, 1 = N1 v3s4x2e_v2.
 * Pure function of the MIDR implementer (test-visible). */
int rj_crc32_fast_arm_pick_fusion(int implementer)
{
    return (implementer == 0x61) ? 0 : 1;
}

/* Pure dispatch-chain decision (test-visible; all flag combos).
 *
 * Tier 1 needs crc32 as well as pmull+sha3: both fusion sub-variants
 * execute the scalar crc32 instructions, so without HWCAP_CRC32 they
 * would SIGILL. The spec's "pmull && sha3" gate is read as implying
 * the co-occurring crc32 extension (true on every Apple core and all
 * documented crypto+sha3 cores); making it explicit keeps the
 * "never SIGILL, always degrade" contract. */
int rj_crc32_fast_arm_pick(int crc32, int pmull, int sha3)
{
    if (pmull && sha3 && crc32)
        return 3; /* fusion (sub-variant by MIDR) */
    if (crc32)
        return 1; /* scalar crc32x */
    if (pmull)
        return 2; /* pure PMULL fold */
    return 0;     /* slice-by-8 */
}

static pthread_once_t arm_once = PTHREAD_ONCE_INIT;
static uint32_t (*arm_best)(uint32_t, const void *, size_t) = NULL;
static int arm_class_value = 0;
static int arm_fusion_sub = 0;

static void arm_select(void)
{
    rj_cpu_info ci;
    static uint32_t (*const table[4])(uint32_t, const void *, size_t) = {
        rj_crc32_update,            /* 0: scalar slice-by-8 (reused) */
        rj_crc32_fast_arm_crc32hw,  /* 1 */
        rj_crc32_fast_arm_pmull,    /* 2 */
        NULL,                       /* 3: fusion, sub-variant below */
    };
    int cls;

    rj_cpu_detect(&ci);
    cls = rj_crc32_fast_arm_pick(ci.crc32, ci.pmull, ci.sha3);
    arm_class_value = cls;
    if (cls == 3) {
        arm_fusion_sub = rj_crc32_fast_arm_pick_fusion(read_implementer());
        arm_best = arm_fusion_sub ? rj_crc32_fast_arm_fusion_n1
                                  : rj_crc32_fast_arm_fusion;
    } else {
        arm_best = table[cls];
    }
}

uint32_t rj_crc32_fast_arm(uint32_t crc, const void *data, size_t len)
{
    pthread_once(&arm_once, arm_select);
    return arm_best(crc, data, len);
}

int rj_crc32_fast_arm_class(void)
{
    pthread_once(&arm_once, arm_select);
    return arm_class_value;
}

/* Which fusion sub-variant this machine selected: 0 = M1 v9s3x2e_s3,
 * 1 = N1 v3s4x2e_v2. Test-visible; only meaningful when
 * rj_crc32_fast_arm_class() == 3. */
int rj_crc32_fast_arm_fusion_sub(void)
{
    pthread_once(&arm_once, arm_select);
    return arm_fusion_sub;
}

#endif /* __aarch64__ */
