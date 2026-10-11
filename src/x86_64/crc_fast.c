/* x86-64 fast CRC-32 (IEEE 0x04C11DB7) variants -- Phase 1, Option C.
 *
 * Three vector variants plus the scalar fallback, all bit-exact vs
 * rj_crc32_update / zlib crc32 and sharing its chaining contract (take a
 * running CRC, return the updated running CRC).
 *
 * ---------------------------------------------------------------------------
 * Variant structure (Cawley-style scalar+vector fusion)
 * ---------------------------------------------------------------------------
 * Each vector variant splits the input into a vector region and a scalar
 * region. Vector lanes fold their stream with carryless multiply while a
 * scalar lane runs in parallel; at the end the vector lanes are reduced to
 * one 128-bit value, shifted past the scalar region, and XORed into the
 * scalar lane. Structure and constants follow Cawley's fast-crc32 generator
 * (thirdparty/fast-crc32) run as `-p crc32` (IEEE) -- e.g.
 *   ./generate -i avx512_vpclmulqdq -p crc32 -a v3s1_s3
 *   ./generate -i sse -p crc32 -a v3s1_s3
 * whose output is bit-exact vs zlib. Fold constants come from the
 * generator's own xnmodp: k1=xnmodp(S+31), k2=xnmodp(S-33) for an S-bit
 * shift, verified against every constant the generator emits
 * (see the W1 report).
 *
 * ---------------------------------------------------------------------------
 * IEEE vs Castagnoli -- why there is no `crc32`-instruction tier on x86
 * ---------------------------------------------------------------------------
 * The x86 `crc32` instruction (SSE4.2) implements CRC-32C (Castagnoli,
 * 0x1EDC6F41), NOT the IEEE CRC-32 (0x04C11DB7) this library computes.
 * It cannot be used here -- not for a "scalar lane", not for an N-way
 * interleave. Cawley's generator confirms: for `-p crc32` on x86 it emits
 * PCLMUL-based scalar lanes (`crc_u64` below), reserving `_mm_crc32_*` for
 * `-p crc32c` only. Consequently:
 *  - the "scalar lanes" in the fusion are PCLMUL-based, and
 *  - rj_crc32_fast_x86_crc32hw has no hardware to use; it delegates to the
 *    slice-by-8 fallback (documented in crc_fast.h), and the internal
 *    selector skips it. An sse42-only CPU therefore reports class 0.
 * (ARM differs: ARMv8 has distinct IEEE `crc32b/h/w/x` and Castagnoli
 * `crc32cb/ch/cw/cx` instructions, so the ARM backend can use hw crc32.)
 *
 * ---------------------------------------------------------------------------
 * Dispatch gates (spec R2 as amended by Finding 13)
 * ---------------------------------------------------------------------------
 *  rung 1 (vpclmul, 512-bit fusion): vpclmulqdq && pclmulqdq && avx512f
 *      && avx512bw.
 *      avx512bw joins per spec review R1-6: the reference 512-bit fusion
 *      uses _mm512_broadcast_i32x4 (VPBROADCASTI32X4) for the fold constants
 *      and _mm512_extracti32x4_epi32 (VEXTRACTI32X4) in the 512->128 lane
 *      reduction -- both are AVX512BW instructions. (On every shipping CPU,
 *      vpclmulqdq+avx512f implies avx512bw -- Intel ICL/SPR, AMD Zen4/5 --
 *      but the gate is explicit so the code is correct by construction.)
 *      pclmulqdq joins because the scalar lane and crc_shift use
 *      _mm_clmulepi64_si128. AVX512VL is NOT needed: the single 128-bit
 *      ternarylogic in the reference is replaced by two _mm_xor_si128.
 *  rung 2 (vpclmul, 256-bit fusion): vpclmulqdq && pclmulqdq && avx2, without
 *      avx512f (Zen3 -- the synthesis's optimal formulation for that row:
 *      2.3x over the 128-bit fold). avx2 joins because GCC categorizes the
 *      256-bit integer XOR intrinsics as AVX2.
 *  rung 3 (pclmul, 128-bit fold): pclmulqdq without vpclmulqdq. Pure vector
 *      fold (v4 lanes, zlib-ng/Chromium structure) -- NOT fusion, per the
 *      Finding-13 amendment (the "34% over vector-only" fusion citation
 *      belonged to rung 1 and was miscited).
 *  rungs 4/5: see crc_fast.h (crc32hw delegates to scalar; no IEEE hw tier).
 *
 * rj_crc32_fast_x86_vpclmul covers rungs 1+2 (selects internally);
 * rj_crc32_fast_x86_pclmul is the rung-3 128-bit fold.
 *
 * Every public entry point re-checks its gate and degrades to a slower
 * CORRECT variant when the CPU lacks the feature (never wrong, never
 * SIGILL), so direct calls from tests are safe on any x86-64 host.
 * Function `target` attributes let this TU compile on any host; dispatch
 * guarantees a variant never executes where unsupported.
 *
 * No over-reads: all vector loads are within [buf, buf+len); the head
 * aligns the body and the tail is handled scalar-first. Guard-page tests
 * (tests/test_crc_fast_x86.c) prove no byte past the end is touched.
 */

#include "crc_fast.h"

#include <immintrin.h>
#include <pthread.h>
#include <stdlib.h> /* getenv (test-only override) */
#include <wmmintrin.h>

#include "../cpu.h" /* rj_cpu_detect */
#include "../crc.h" /* rj_crc32_update, rj_crc32_update_scalar */
#include "raidjit.h" /* rj_cpu_info */

/* ---------------------------------------------------------------------------
 * Scalar helpers (PCLMUL-based, IEEE-correct).
 * Adapted from Cawley's fast-crc32 generator output for `-p crc32` on x86.
 * crc_u64 reduces 8 input bytes into the 32-bit CRC state with two
 * carryless multiplies (Barrett-style); crc_shift multiplies a CRC state by
 * x^(nbytes*8) via xnmodp. The byte-at-a-time head/tail uses
 * rj_crc32_update_scalar (shared table, not duplicated).
 * ------------------------------------------------------------------------- */

static uint32_t xnmodp_u64(uint64_t n)
{
    /* Verbatim from fast-crc32 generate.c (g_poly = 0xEDB88320, IEEE
     * reflected). Computes x^n mod P(x). */
    static const uint32_t g_poly = 0xEDB88320u;
    uint64_t stack = ~(uint64_t)1, r;
    uint32_t i;
    for (; n > 31; n >>= 1)
        stack = (stack << 1) + (n & 1);
    stack = ~stack;
    r = ((uint32_t)0x80000000) >> n; /* r = x^n (n <= 31) */
    while ((i = (uint32_t)(stack & 1)), (stack >>= 1)) {
        /* r = r^2 * x^i, expanding 32 -> 64 bits (squaring in GF(2) is
         * bit-interleaving in the reflected domain). */
        r ^= r << 16; r &= 0x0000ffff0000ffffull;
        r ^= r << 8;  r &= 0x00ff00ff00ff00ffull;
        r ^= r << 4;  r &= 0x0f0f0f0f0f0f0f0full;
        r ^= r << 2;  r &= 0x3333333333333333ull;
        r ^= r << 1;  r &= 0x5555555555555555ull;
        r <<= i;
        /* r = r mod P, narrowed back to 32 bits. */
        for (i = 0; i < 32; ++i)
            r = (r >> 1) ^ ((r & 1) * g_poly);
    }
    return (uint32_t)r;
}

/* 32x32 -> 64 carryless multiply of two CRC states (PCLMULQDQ). */
__attribute__((target("pclmul")))
static __m128i clmul_scalar32(uint32_t a, uint32_t b)
{
    return _mm_clmulepi64_si128(_mm_cvtsi32_si128((int)a),
                                _mm_cvtsi32_si128((int)b), 0x00);
}

/* Fold 8 input bytes into the CRC state (PCLMUL-based, IEEE). */
__attribute__((target("pclmul,sse4.1")))
static uint32_t crc_u64_step(uint32_t crc, uint64_t val)
{
    /* k = Barrett-style reduction constants for 64 input bits, from the
     * generator (need_crc_scalar, size=8, x86 path). */
    __m128i k = _mm_setr_epi32(0xf7011641, 0xb4e5b025, 0xdb710641, 1);
    __m128i a = _mm_cvtsi64_si128((long long)(crc ^ val));
    __m128i b = _mm_clmulepi64_si128(a, k, 0x00);
    __m128i c = _mm_clmulepi64_si128(b, k, 0x10);
    return (uint32_t)_mm_extract_epi32(c, 2);
}

/* Multiply a CRC state by x^(nbytes*8): shift past nbytes of input. */
__attribute__((target("pclmul")))
static uint64_t crc_shift_state(uint32_t crc, size_t nbytes)
{
    __m128i p = clmul_scalar32(crc, xnmodp_u64(nbytes * 8 - 33));
    return (uint64_t)_mm_cvtsi128_si64(p);
}

/* Byte-at-a-time step reusing the shared slice-by-8 table (no duplication). */
static uint32_t crc_u8_step(uint32_t crc, const uint8_t *p)
{
    return rj_crc32_update_scalar(crc, p, 1);
}

/* ---------------------------------------------------------------------------
 * Tier 3: 512-bit VPCLMULQDQ fusion (v3s1).
 *
 * Three 64-byte vector lanes + one PCLMUL scalar lane; 200 bytes per
 * iteration (192 vector + 8 scalar). Structure and constants are Cawley's
 * `-i avx512_vpclmulqdq -p crc32 -a v3s1_s3` output, with:
 *  - the single _mm_ternarylogic_epi64 (needs AVX512VL) replaced by two
 *    _mm_xor_si128, so AVX512VL is not required;
 *  - byte head/tail via crc_u8_step (shared table);
 *  - the 3-way scalar interleave for 32..199B replaced by the plain scalar
 *    tail (same result, less code);
 *  - the ~crc0/~return removed: this keeps the rj_crc32_update chaining
 *    contract (running CRC in, running CRC out).
 * Constants (generator's xnmodp, IEEE):
 *  main-loop fold (shift 192B): k1=0x596c8d81 k2=0xf5e48c85
 *  lane reduction (shift 64B):  k1=0x8f352d95 k2=0x1d9513d7
 *  512->128 sub-lane shifts: 48B:(0x3db1ecdc,0xaf449247)
 *    32B:(0xf1da05aa,0x81256527) 16B:(0xae689191,0xccaa009e) 0B:(0,0)
 * ------------------------------------------------------------------------- */
__attribute__((target("avx512f,avx512bw,vpclmulqdq,pclmul,sse4.1")))
static uint32_t vpclmul_512_body(uint32_t crc0, const uint8_t *buf, size_t len)
{
#define CRC512_clmul_lo(a, b) (_mm512_clmulepi64_epi128((a), (b), 0x00))
#define CRC512_clmul_hi(a, b) (_mm512_clmulepi64_epi128((a), (b), 0x11))

    for (; len && ((uintptr_t)buf & 7); --len)
        crc0 = crc_u8_step(crc0, buf++);
    while (((uintptr_t)buf & 56) && len >= 8) {
        crc0 = crc_u64_step(crc0, *(const uint64_t *)buf);
        buf += 8;
        len -= 8;
    }
    if (len >= 208) {
        size_t blk = (len - 8) / 200;
        size_t klen = blk * 8;
        const uint8_t *buf2 = buf;
        uint64_t vc;
        __m128i z0;
        /* First vector chunk. */
        __m512i x0 = _mm512_loadu_si512((const void *)buf2), y0;
        __m512i x1 = _mm512_loadu_si512((const void *)(buf2 + 64)), y1;
        __m512i x2 = _mm512_loadu_si512((const void *)(buf2 + 128)), y2;
        __m512i k;
        k = _mm512_broadcast_i32x4(_mm_setr_epi32(0x596c8d81, 0, 0xf5e48c85, 0));
        x0 = _mm512_xor_si512(_mm512_castsi128_si512(_mm_cvtsi32_si128((int)crc0)), x0);
        crc0 = 0;
        buf2 += 192;
        len -= 200;
        buf += blk * 192;
        /* Main loop. */
        while (len >= 208) {
            y0 = CRC512_clmul_lo(x0, k); x0 = CRC512_clmul_hi(x0, k);
            y1 = CRC512_clmul_lo(x1, k); x1 = CRC512_clmul_hi(x1, k);
            y2 = CRC512_clmul_lo(x2, k); x2 = CRC512_clmul_hi(x2, k);
            x0 = _mm512_ternarylogic_epi64(x0, y0, _mm512_loadu_si512((const void *)buf2), 0x96);
            x1 = _mm512_ternarylogic_epi64(x1, y1, _mm512_loadu_si512((const void *)(buf2 + 64)), 0x96);
            x2 = _mm512_ternarylogic_epi64(x2, y2, _mm512_loadu_si512((const void *)(buf2 + 128)), 0x96);
            crc0 = crc_u64_step(crc0, *(const uint64_t *)buf);
            buf += 8;
            buf2 += 192;
            len -= 200;
        }
        /* Reduce x0 ... x2 to just x0. */
        k = _mm512_broadcast_i32x4(_mm_setr_epi32(0x8f352d95, 0, 0x1d9513d7, 0));
        y0 = CRC512_clmul_lo(x0, k); x0 = CRC512_clmul_hi(x0, k);
        x0 = _mm512_ternarylogic_epi64(x0, y0, x1, 0x96);
        x1 = x2;
        y0 = CRC512_clmul_lo(x0, k); x0 = CRC512_clmul_hi(x0, k);
        x0 = _mm512_ternarylogic_epi64(x0, y0, x1, 0x96);
        /* Final scalar chunk. */
        crc0 = crc_u64_step(crc0, *(const uint64_t *)buf);
        buf += 8;
        vc = 0;
        /* Reduce 512 bits to 128 bits. */
        k = _mm512_setr_epi32(0x3db1ecdc, 0, 0xaf449247, 0,
                              0xf1da05aa, 0, 0x81256527, 0,
                              0xae689191, 0, 0xccaa009e, 0,
                              0, 0, 0, 0);
        y0 = CRC512_clmul_lo(x0, k);
        {
            __m512i t = CRC512_clmul_hi(x0, k);
            y0 = _mm512_xor_si512(y0, t);
        }
        /* 128-bit ternarylogic replaced by SSE2 XORs (no AVX512VL needed). */
        z0 = _mm_xor_si128(_mm_xor_si128(_mm512_castsi512_si128(y0),
                                        _mm512_extracti32x4_epi32(y0, 1)),
                           _mm512_extracti32x4_epi32(y0, 2));
        z0 = _mm_xor_si128(z0, _mm512_extracti32x4_epi32(x0, 3));
        /* Reduce 128 bits to 32 bits, and multiply by x^32. */
        vc ^= crc_shift_state(crc_u64_step(crc_u64_step(0, (uint64_t)_mm_extract_epi64(z0, 0)),
                                                         (uint64_t)_mm_extract_epi64(z0, 1)),
                                         klen + 8);
        /* Final 8 bytes. */
        crc0 = crc_u64_step(crc0, *(const uint64_t *)buf ^ vc);
        buf += 8;
        len -= 8;
    }
    for (; len >= 8; buf += 8, len -= 8)
        crc0 = crc_u64_step(crc0, *(const uint64_t *)buf);
    for (; len; --len)
        crc0 = crc_u8_step(crc0, buf++);
    return crc0;

#undef CRC512_clmul_lo
#undef CRC512_clmul_hi
}
/* ---------------------------------------------------------------------------
 * Rung 3: 128-bit PCLMULQDQ fold (v4, pure vector -- NOT fusion).
 *
 * Four 16-byte vector lanes, 64 bytes per iteration; no scalar lane. This is
 * Cawley's `-i sse -p crc32 -a v4` output (the zlib-ng/Chromium 128-bit fold
 * structure), adapted as above (byte head/tail via shared table, chaining
 * contract without ~). Per the spec amendment (Finding 13), rung 3 is the
 * pure fold; the "34% over vector-only" fusion citation belonged to rung 1.
 * Constants (IEEE):
 *  main-loop fold (shift 64B): k1=0x8f352d95 k2=0x1d9513d7
 *  lane reduction: 16B (0xae689191,0xccaa009e), then 32B (0xf1da05aa,0x81256527)
 * ------------------------------------------------------------------------- */
__attribute__((target("pclmul,sse4.1")))
static uint32_t pclmul_128_fold_body(uint32_t crc0, const uint8_t *buf, size_t len)
{
#define CRC128F_clmul_lo(a, b) (_mm_clmulepi64_si128((a), (b), 0x00))
#define CRC128F_clmul_hi(a, b) (_mm_clmulepi64_si128((a), (b), 0x11))

    for (; len && ((uintptr_t)buf & 7); --len)
        crc0 = crc_u8_step(crc0, buf++);
    if (((uintptr_t)buf & 8) && len >= 8) {
        crc0 = crc_u64_step(crc0, *(const uint64_t *)buf);
        buf += 8;
        len -= 8;
    }
    if (len >= 64) {
        /* First vector chunk: 4x128-bit lanes. */
        __m128i x0 = _mm_loadu_si128((const __m128i *)buf), y0;
        __m128i x1 = _mm_loadu_si128((const __m128i *)(buf + 16)), y1;
        __m128i x2 = _mm_loadu_si128((const __m128i *)(buf + 32)), y2;
        __m128i x3 = _mm_loadu_si128((const __m128i *)(buf + 48)), y3;
        __m128i k = _mm_setr_epi32(0x8f352d95, 0, 0x1d9513d7, 0);
        x0 = _mm_xor_si128(_mm_cvtsi32_si128((int)crc0), x0);
        crc0 = 0;
        buf += 64;
        len -= 64;
        /* Main loop: fold each lane by 64B, XOR 64B new data. */
        while (len >= 64) {
            y0 = CRC128F_clmul_lo(x0, k); x0 = CRC128F_clmul_hi(x0, k);
            y1 = CRC128F_clmul_lo(x1, k); x1 = CRC128F_clmul_hi(x1, k);
            y2 = CRC128F_clmul_lo(x2, k); x2 = CRC128F_clmul_hi(x2, k);
            y3 = CRC128F_clmul_lo(x3, k); x3 = CRC128F_clmul_hi(x3, k);
            y0 = _mm_xor_si128(y0, _mm_loadu_si128((const __m128i *)buf));
            x0 = _mm_xor_si128(x0, y0);
            y1 = _mm_xor_si128(y1, _mm_loadu_si128((const __m128i *)(buf + 16)));
            x1 = _mm_xor_si128(x1, y1);
            y2 = _mm_xor_si128(y2, _mm_loadu_si128((const __m128i *)(buf + 32)));
            x2 = _mm_xor_si128(x2, y2);
            y3 = _mm_xor_si128(y3, _mm_loadu_si128((const __m128i *)(buf + 48)));
            x3 = _mm_xor_si128(x3, y3);
            buf += 64;
            len -= 64;
        }
        /* Reduce x0..x3 to just x0: pairwise 16B shifts, then a 32B shift. */
        k = _mm_setr_epi32(0xae689191, 0, 0xccaa009e, 0);
        y0 = CRC128F_clmul_lo(x0, k); x0 = CRC128F_clmul_hi(x0, k);
        y2 = CRC128F_clmul_lo(x2, k); x2 = CRC128F_clmul_hi(x2, k);
        y0 = _mm_xor_si128(y0, x1); x0 = _mm_xor_si128(x0, y0);
        y2 = _mm_xor_si128(y2, x3); x2 = _mm_xor_si128(x2, y2);
        k = _mm_setr_epi32(0xf1da05aa, 0, 0x81256527, 0);
        y0 = CRC128F_clmul_lo(x0, k); x0 = CRC128F_clmul_hi(x0, k);
        y0 = _mm_xor_si128(y0, x2); x0 = _mm_xor_si128(x0, y0);
        /* Reduce 128 bits to 32 bits. */
        crc0 = crc_u64_step(0, (uint64_t)_mm_extract_epi64(x0, 0));
        crc0 = crc_u64_step(crc0, (uint64_t)_mm_extract_epi64(x0, 1));
    }
    for (; len >= 8; buf += 8, len -= 8)
        crc0 = crc_u64_step(crc0, *(const uint64_t *)buf);
    for (; len; --len)
        crc0 = crc_u8_step(crc0, buf++);
    return crc0;

#undef CRC128F_clmul_lo
#undef CRC128F_clmul_hi
}


/* ---------------------------------------------------------------------------
 * Tier 2a: 256-bit VPCLMULQDQ fusion (v6s1) -- vpclmulqdq without avx512f.
 *
 * Three 32-byte vector registers holding six independent 16-byte sub-lanes
 * (VPCLMULQDQ works per 128-bit lane, so each 256-bit register folds two
 * sub-lanes in one instruction) plus one PCLMUL scalar lane; 104 bytes per
 * iteration (96 vector + 8 scalar). This is the Cawley v-lane structure with
 * the lane width following the best available clmul (spec review R1-2):
 * on Zen3 (VPCLMULQDQ, no AVX-512) the 256-bit form halves the vector clmul
 * instruction count vs the 128-bit form.
 *
 * Constants (generator's xnmodp, IEEE):
 *  main-loop fold (shift 96B): k1=0xdf068dc2 k2=0x57c54819
 *  sub-lane pairwise reduction (shift 16B): k1=0xae689191 k2=0xccaa009e
 *  final combine: m0<<64B (0x8f352d95,0x1d9513d7), m1<<32B (0xf1da05aa,0x81256527)
 * The six sub-lanes sit at input offsets 0,16,32,48,64,80; the reduction
 * shifts lanes 0..4 forward to align with lane 5, then XORs.
 * ------------------------------------------------------------------------- */
__attribute__((target("avx2,vpclmulqdq,pclmul,sse4.1")))
static uint32_t pclmul_256_body(uint32_t crc0, const uint8_t *buf, size_t len)
{
#define CRC256_clmul_lo(a, b) (_mm256_clmulepi64_epi128((a), (b), 0x00))
#define CRC256_clmul_hi(a, b) (_mm256_clmulepi64_epi128((a), (b), 0x11))

    for (; len && ((uintptr_t)buf & 7); --len)
        crc0 = crc_u8_step(crc0, buf++);
    while (((uintptr_t)buf & 24) && len >= 8) {
        crc0 = crc_u64_step(crc0, *(const uint64_t *)buf);
        buf += 8;
        len -= 8;
    }
    if (len >= 112) {
        size_t blk = (len - 8) / 104;
        size_t klen = blk * 8;
        const uint8_t *buf2 = buf;
        uint64_t vc;
        /* First vector chunk: 3x256-bit = 6x128-bit sub-lanes. */
        __m256i r0 = _mm256_loadu_si256((const __m256i *)buf2), y0;
        __m256i r1 = _mm256_loadu_si256((const __m256i *)(buf2 + 32)), y1;
        __m256i r2 = _mm256_loadu_si256((const __m256i *)(buf2 + 64)), y2;
        __m256i k;
        /* 96B shift per sub-lane. */
        k = _mm256_setr_epi32(0xdf068dc2, 0, 0x57c54819, 0,
                              0xdf068dc2, 0, 0x57c54819, 0);
        r0 = _mm256_xor_si256(r0,
                              _mm256_castsi128_si256(_mm_cvtsi32_si128((int)crc0)));
        crc0 = 0;
        buf2 += 96;
        len -= 104;
        buf += blk * 96;
        /* Main loop. */
        while (len >= 112) {
            y0 = CRC256_clmul_lo(r0, k); r0 = CRC256_clmul_hi(r0, k);
            y1 = CRC256_clmul_lo(r1, k); r1 = CRC256_clmul_hi(r1, k);
            y2 = CRC256_clmul_lo(r2, k); r2 = CRC256_clmul_hi(r2, k);
            r0 = _mm256_xor_si256(_mm256_xor_si256(r0, y0),
                                 _mm256_loadu_si256((const __m256i *)buf2));
            r1 = _mm256_xor_si256(_mm256_xor_si256(r1, y1),
                                 _mm256_loadu_si256((const __m256i *)(buf2 + 32)));
            r2 = _mm256_xor_si256(_mm256_xor_si256(r2, y2),
                                 _mm256_loadu_si256((const __m256i *)(buf2 + 64)));
            crc0 = crc_u64_step(crc0, *(const uint64_t *)buf);
            buf += 8;
            buf2 += 96;
            len -= 104;
        }
        /* Reduce 6 sub-lanes to one 128-bit value.
         * Step 1: pairwise (l0<<16B)^l1 -> m0, (l2<<16B)^l3 -> m1,
         * (l4<<16B)^l5 -> m2. The high 128-bit lane of ks16 is zero, so the
         * odd sub-lane passes through the fold untouched and is XORed back
         * from the original register. */
        {
            __m256i ks16 = _mm256_setr_epi32(0xae689191, 0, 0xccaa009e, 0,
                                            0, 0, 0, 0);
            __m128i m0, m1, m2, acc;
            __m256i t;
            y0 = CRC256_clmul_lo(r0, ks16);
            t = CRC256_clmul_hi(r0, ks16);
            y0 = _mm256_xor_si256(y0, t);
            m0 = _mm_xor_si128(_mm256_castsi256_si128(y0),
                               _mm256_extracti128_si256(r0, 1));
            y1 = CRC256_clmul_lo(r1, ks16);
            t = CRC256_clmul_hi(r1, ks16);
            y1 = _mm256_xor_si256(y1, t);
            m1 = _mm_xor_si128(_mm256_castsi256_si128(y1),
                               _mm256_extracti128_si256(r1, 1));
            y2 = CRC256_clmul_lo(r2, ks16);
            t = CRC256_clmul_hi(r2, ks16);
            y2 = _mm256_xor_si256(y2, t);
            m2 = _mm_xor_si128(_mm256_castsi256_si128(y2),
                               _mm256_extracti128_si256(r2, 1));
            /* Step 2: (m0<<64B) ^ (m1<<32B) ^ m2. */
            {
                __m128i k64 = _mm_setr_epi32(0x8f352d95, 0, 0x1d9513d7, 0);
                __m128i a0 = _mm_clmulepi64_si128(m0, k64, 0x00);
                __m128i b0 = _mm_clmulepi64_si128(m0, k64, 0x11);
                m0 = _mm_xor_si128(a0, b0);
            }
            {
                __m128i k32 = _mm_setr_epi32(0xf1da05aa, 0, 0x81256527, 0);
                __m128i a1 = _mm_clmulepi64_si128(m1, k32, 0x00);
                __m128i b1 = _mm_clmulepi64_si128(m1, k32, 0x11);
                m1 = _mm_xor_si128(a1, b1);
            }
            acc = _mm_xor_si128(_mm_xor_si128(m0, m1), m2);
            /* Final scalar chunk. */
            crc0 = crc_u64_step(crc0, *(const uint64_t *)buf);
            buf += 8;
            vc = 0;
            /* Reduce 128 bits to 32 bits, and multiply by x^32. */
            vc ^= crc_shift_state(crc_u64_step(crc_u64_step(0,
                                            (uint64_t)_mm_extract_epi64(acc, 0)),
                                            (uint64_t)_mm_extract_epi64(acc, 1)),
                                             klen + 8);
            /* Final 8 bytes. */
            crc0 = crc_u64_step(crc0, *(const uint64_t *)buf ^ vc);
            buf += 8;
            len -= 8;
        }
    }
    for (; len >= 8; buf += 8, len -= 8)
        crc0 = crc_u64_step(crc0, *(const uint64_t *)buf);
    for (; len; --len)
        crc0 = crc_u8_step(crc0, buf++);
    return crc0;

#undef CRC256_clmul_lo
#undef CRC256_clmul_hi
}

/* ---------------------------------------------------------------------------
 * Public entry points (spec R2.5, exact names). Each re-checks its gate so a
 * direct call on an unsupported CPU degrades to a slower CORRECT variant --
 * never SIGILL. rj_crc32_fast_x86 caches the selection with pthread_once,
 * matching src/crc.c style.
 * ------------------------------------------------------------------------- */

static int x86_has_pclmul(void)
{
    rj_cpu_info ci;
    rj_cpu_detect(&ci);
    return ci.pclmulqdq;
}

/* Rung selection for rj_crc32_fast_x86_vpclmul as a pure function of the
 * detected features and the test-only override flag. Returns 1 (512-bit
 * VPCLMULQDQ fusion), 2 (256-bit VPCLMULQDQ fusion, Zen3), or 0 (degrade
 * to the pclmul entry, rung 3+).
 *
 * Rung 2 needs avx2 because GCC categorizes the 256-bit integer XOR
 * intrinsics as AVX2 (all VPCLMULQDQ CPUs have AVX2).
 *
 * Test-only override: force256 (RJ_CRC_X86_FORCE256=1) selects rung 2 on
 * AVX-512-capable CPUs so tests can execute the Zen3 path there. It is
 * honored BEFORE the AVX-512 tier -- previously the tier check ran first
 * and shadowed the override, making it dead on exactly the CPUs it
 * targets (peer review MAJOR-1). The hardware requirement
 * (vpclmulqdq && pclmulqdq && avx2) is never bypassed. */
static int x86_vpclmul_rung(const rj_cpu_info *ci, int force256)
{
    if (ci->vpclmulqdq && ci->pclmulqdq && ci->avx2 &&
        (force256 || !ci->avx512f))
        return 2;
    if (ci->vpclmulqdq && ci->pclmulqdq && ci->avx512f && ci->avx512bw)
        return 1;
    return 0;
}

/* Test-only: which rung rj_crc32_fast_x86_vpclmul would take on this
 * machine with the given force256 override (1/2/0 as above). Lets the
 * test prove the override engages before trusting forced-rung coverage:
 * without this, a forced matrix could silently re-exercise rung 1. */
int rj_crc32_fast_x86_test_rung(int force256)
{
    rj_cpu_info ci;
    rj_cpu_detect(&ci);
    return x86_vpclmul_rung(&ci, force256);
}

uint32_t rj_crc32_fast_x86_vpclmul(uint32_t crc, const void *d, size_t n)
{
    /* Rungs 1+2 (spec R2.5): select internally. */
    if (n == 0)
        return crc;
    rj_cpu_info ci;
    rj_cpu_detect(&ci);
    switch (x86_vpclmul_rung(&ci, getenv("RJ_CRC_X86_FORCE256") != NULL)) {
    case 1:
        return vpclmul_512_body(crc, (const uint8_t *)d, n);
    case 2:
        return pclmul_256_body(crc, (const uint8_t *)d, n);
    default:
        return rj_crc32_fast_x86_pclmul(crc, d, n);
    }
}

uint32_t rj_crc32_fast_x86_pclmul(uint32_t crc, const void *d, size_t n)
{
    /* Rung 3: 128-bit PCLMULQDQ fold (pure vector, not fusion). */
    if (n == 0)
        return crc;
    if (!x86_has_pclmul())
        return rj_crc32_update(crc, d, n);
    return pclmul_128_fold_body(crc, (const uint8_t *)d, n);
}

uint32_t rj_crc32_fast_x86_crc32hw(uint32_t crc, const void *d, size_t n)
{
    /* No distinct hardware tier: the x86 crc32 instruction is CRC-32C
     * (Castagnoli)-only and cannot produce IEEE CRC-32. Delegates to the
     * shared slice-by-8 fallback. The internal selector never picks this
     * as a separate tier (see rj_crc32_fast_x86_class). */
    return rj_crc32_update(crc, d, n);
}

/* Cached selection for rj_crc32_fast_x86. */
typedef uint32_t (*crc_fast_fn)(uint32_t, const void *, size_t);
static crc_fast_fn x86_selected_fn = NULL;
static pthread_once_t x86_select_once = PTHREAD_ONCE_INIT;

static void x86_select_init(void)
{
    /* Spec R2 chain as amended by Finding 13. The avx512f-without-avx512bw
     * row is hypothetical (no shipping CPU); it is routed to rung 3
     * directly so class() stays truthful -- the vpclmul entry would
     * degrade there anyway (x86_vpclmul_rung returns 0). */
    rj_cpu_info ci;
    rj_cpu_detect(&ci);
    if (ci.vpclmulqdq && ci.pclmulqdq && ci.avx512f && ci.avx512bw)
        x86_selected_fn = rj_crc32_fast_x86_vpclmul;  /* rung 1: 512-bit fusion */
    else if (ci.vpclmulqdq && ci.pclmulqdq && ci.avx2 && !ci.avx512f)
        x86_selected_fn = rj_crc32_fast_x86_vpclmul;  /* rung 2: 256-bit fusion */
    else if (ci.pclmulqdq)
        x86_selected_fn = rj_crc32_fast_x86_pclmul;   /* rung 3: 128-bit fold */
    else
        x86_selected_fn = rj_crc32_update;            /* rungs 4+5: scalar */
}

uint32_t rj_crc32_fast_x86(uint32_t crc, const void *d, size_t n)
{
    pthread_once(&x86_select_once, x86_select_init);
    return x86_selected_fn(crc, d, n);
}

int rj_crc32_fast_x86_class(void)
{
    /* Matches the selection in x86_select_init exactly.
     * Class codes (spec R2.5 as amended): 0=scalar, 1=crc32hw, 2=pclmul128,
     * 3=vpclmul (rungs 1+2). Note class 1 is never returned: see the
     * IEEE-vs-Castagnoli note above. */
    rj_cpu_info ci;
    rj_cpu_detect(&ci);
    if (ci.vpclmulqdq && ci.pclmulqdq && ci.avx512f && ci.avx512bw)
        return 3;
    if (ci.vpclmulqdq && ci.pclmulqdq && ci.avx2 && !ci.avx512f)
        return 3;
    if (ci.pclmulqdq)
        return 2;
    return 0;
}
