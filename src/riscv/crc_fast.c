/* RISC-V fast CRC-32 variants (Phase 1, Option C standalone API).
 *
 * Spec: hidden_files/crc_impl_SPEC.md R2 (RISC-V chain), R2.5 (pinned
 * entry-point names), R3 (zbc detection, in src/riscv/cpu.c).
 *
 * ---------------------------------------------------------------------------
 * Variant 1: rj_crc32_fast_riscv_zvbc -- Zvbc vector fold, SEW=64
 * ---------------------------------------------------------------------------
 * The 128-bit fold formulation verified bit-exact in
 * hidden_files/design_crc_fold.md §11 (re-derived from scratch in
 * hidden_files/crc_optimum_research/fold_verify.py). Per 16-byte chunk,
 * with S the 128-bit state (S0 = G(crc): the incoming running CRC in the
 * low 32 bits -- the chaining derivation is in crc_fast.h):
 *
 *     W   = S XOR C            (128-bit; C = 16-byte chunk, LE)
 *     S'  = CLMUL(W[63:0], K1) XOR CLMUL(W[127:64], K2)
 *     K1  = x^{-128} mod QREF = 0xae689191
 *     K2  = x^{-64}  mod QREF = 0xccaa009e
 *     QREF = 0x1DB710641 (reflected CRC-32 polynomial, degree 32)
 *
 * Tail (r bytes, 1<=r<=15): W = S XOR D (D in the low 8r bits);
 *     S' = CLMUL(W[63:0], x^{-8r}) XOR CLMUL(W[127:64], x^{64-8r}).
 * Final: the running CRC is red_QREF(S) (32 bits; no xor -- the xorout
 * lives in rj_crc32_final, and returning the un-xored reduction is what
 * makes len==0 the identity and chaining exact).
 *
 * This is the SEW=64 element dataflow a vclmul.vv/vclmulh.vv
 * implementation would execute (state = 2 x u64 elements, per-element
 * carryless multiply against the (K1,K2) constant vector); the
 * carryless multiply itself is the portable rj_clmul128 below (see the
 * toolchain note in crc_fast.h).
 *
 * ---------------------------------------------------------------------------
 * Variant 2: rj_crc32_fast_riscv_zbc -- scalar clmul fold
 * ---------------------------------------------------------------------------
 * The two-accumulator structure of zlib-ng's arch/riscv/crc32_zbc.c
 * (ByteDance / Yin Tong, 2025), constants reproduced verbatim:
 *     R3 = 0x1751997d0, R4 = 0x0ccaa009e, R5 = 0x163cd6124,
 *     RU = 0x1F7011641, POLY = 0x1DB710641.
 * Chaining adaptation (verified against zlib in Python before writing
 * this C): zlib-ng seeds its core with ~crc and complements the result
 * because zlib's running convention complements at both ends; rj's
 * rj_crc32_update convention does NOT complement, so the two inversions
 * cancel -- this variant seeds the core with crc directly and returns
 * its result directly. Formally, with Z(c,d) = zlib-ng's outer function
 * (= zlib crc32) and I(p,d) its core:
 *     rj_update(c,d) = Z(c^0xFFFFFFFF,d)^0xFFFFFFFF
 *                    = ~I(~(c^0xFFFFFFFF),d)^0xFFFFFFFF = I(c,d).
 * Short buffers (len < 16) and the head (len % 16) go through rj's
 * slice-by-8 (rj_crc32_update), standing in for zlib-ng's braid, exactly
 * as their outer function peels the unaligned head before the clmul
 * bulk. The core never over-reads: it runs only on len >= 16 with
 * len % 16 == 0 (their `goto finish_fold` len<16 path, which reads a
 * full 16 bytes, is never taken here).
 *
 * ---------------------------------------------------------------------------
 * Dispatch
 * ---------------------------------------------------------------------------
 * rj_crc32_fast_riscv: first-call pthread_once selection over
 * rj_cpu_detect(), cached function pointer (spec R2). Gated on
 * rvv_zvbc (2) then zbc (1), else the slice-by-8 (0). The pure
 * rj_crc32_fast_riscv_{class,fn}_for() selection is exposed for tests
 * with synthetic rj_cpu_info.
 */
#include "crc_fast.h"

#include <pthread.h>
#include <stddef.h>
#include <stdint.h>

#include "crc.h" /* rj_crc32_update (slice-by-8 fallback / short-head) */

/* ------------------------------------------------------------------ */
/* Portable 64x64 -> 128-bit carryless multiply (GF(2)[x]).            */
/* ------------------------------------------------------------------ */

typedef struct {
    uint64_t lo;
    uint64_t hi;
} rj_clmul128_t;

/* result = XOR over i with b_i set of ((uint128_t)a << i).
 * MSB-first Russian peasant: acc = (acc << 1) ^ (b_i ? a : 0).
 * 64 iterations, no data-dependent branches, no UB (all shifts < 64). */
static rj_clmul128_t rj_clmul128(uint64_t a, uint64_t b)
{
    uint64_t lo = 0, hi = 0;
    for (int i = 63; i >= 0; i--) {
        hi = (hi << 1) | (lo >> 63);
        lo <<= 1;
        if ((b >> (unsigned)i) & 1u)
            lo ^= a;
    }
    rj_clmul128_t r = { lo, hi };
    return r;
}

static uint64_t rj_load_le64(const uint8_t *p)
{
    return (uint64_t)p[0] | ((uint64_t)p[1] << 8) |
           ((uint64_t)p[2] << 16) | ((uint64_t)p[3] << 24) |
           ((uint64_t)p[4] << 32) | ((uint64_t)p[5] << 40) |
           ((uint64_t)p[6] << 48) | ((uint64_t)p[7] << 56);
}

/* ------------------------------------------------------------------ */
/* Variant 1: Zvbc vector fold (§11 formulation).                       */
/* ------------------------------------------------------------------ */

#define CRC_FOLD_K1 0xae689191u /* x^{-128} mod QREF */
#define CRC_FOLD_K2 0xccaa009eu /* x^{-64} mod QREF */
#define CRC_FOLD_QREF 0x1DB710641ULL /* reflected CRC-32 polynomial */

/* Tail constants: TAIL_K_LO[r] = x^{-8r} mod QREF,
 * TAIL_K_HI[r] = x^{64-8r} mod QREF, 1 <= r <= 15. Derived from scratch
 * in hidden_files/crc_optimum_research/fold_verify.py (printed by
 * `python3 -c "from fold_verify import TAIL_K; ..."`); spot checks:
 * r=8 gives (K2, 1), r=4 gives hi = 0xdb710641 = x^{32} mod QREF. */
static const uint32_t tail_k_lo[16] = {
    0x00000000u, 0x77073096u, 0x191b3141u, 0x01c26a37u, 0xb8bc6765u,
    0x3d6029b0u, 0xcb5cd3a5u, 0xa6770bb4u, 0xccaa009eu, 0x177b1443u,
    0xefc26b3eu, 0xc18edfc0u, 0x9ba54c6fu, 0xdd96d985u, 0x9d0fe176u,
    0xb9fbdbe8u
};
static const uint32_t tail_k_hi[16] = {
    0x00000000u, 0x3f036dc2u, 0x7555a0f1u, 0xcacf972au, 0xdb710641u,
    0x01000000u, 0x00010000u, 0x00000100u, 0x00000001u, 0x77073096u,
    0x191b3141u, 0x01c26a37u, 0xb8bc6765u, 0x3d6029b0u, 0xcb5cd3a5u,
    0xa6770bb4u
};

/* One 16-byte fold step: S ^= C (via the XOR below), then
 * S = CLMUL(W_lo,K1) ^ CLMUL(W_hi,K2). */
static void fold16_step(uint64_t *s_lo, uint64_t *s_hi, uint64_t c_lo,
                        uint64_t c_hi)
{
    uint64_t w_lo = *s_lo ^ c_lo;
    uint64_t w_hi = *s_hi ^ c_hi;
    rj_clmul128_t p0 = rj_clmul128(w_lo, CRC_FOLD_K1);
    rj_clmul128_t p1 = rj_clmul128(w_hi, CRC_FOLD_K2);
    *s_lo = p0.lo ^ p1.lo;
    *s_hi = p0.hi ^ p1.hi;
}

/* Tail step for 1 <= r <= 15 bytes: D occupies the low 8r bits of the
 * 128-bit (d_hi:d_lo). All shifts are < 64 (no UB). */
static void fold_tail_step(uint64_t *s_lo, uint64_t *s_hi, const uint8_t *p,
                           size_t r)
{
    uint64_t d_lo = 0, d_hi = 0;
    for (size_t i = 0; i < r; i++) {
        if (i < 8)
            d_lo |= (uint64_t)p[i] << (8u * (unsigned)i);
        else
            d_hi |= (uint64_t)p[i] << (8u * (unsigned)(i - 8));
    }
    uint64_t w_lo = *s_lo ^ d_lo;
    uint64_t w_hi = *s_hi ^ d_hi;
    rj_clmul128_t p0 = rj_clmul128(w_lo, tail_k_lo[r]);
    rj_clmul128_t p1 = rj_clmul128(w_hi, tail_k_hi[r]);
    *s_lo = p0.lo ^ p1.lo;
    *s_hi = p0.hi ^ p1.hi;
}

/* Reduce the 128-bit (hi:lo) modulo QREF by binary long division.
 * Result < 2^32. Once per call; clarity over speed. */
static uint32_t fold_reduce_qref(uint64_t hi, uint64_t lo)
{
    for (int i = 127; i >= 32; i--) {
        uint64_t bit = (i >= 64) ? (hi >> (i - 64)) : (lo >> (unsigned)i);
        if (bit & 1u) {
            /* (hi:lo) ^= QREF << (i - 32). */
            int s = i - 32; /* 0..95 */
            uint64_t q_lo, q_hi;
            if (s >= 64) {
                q_lo = 0;
                q_hi = CRC_FOLD_QREF << (s - 64); /* shift 0..31 */
            } else if (s == 0) {
                q_lo = CRC_FOLD_QREF;
                q_hi = 0;
            } else {
                q_lo = CRC_FOLD_QREF << s; /* shift 1..63 */
                q_hi = CRC_FOLD_QREF >> (64 - s); /* shift 1..63 */
            }
            hi ^= q_hi;
            lo ^= q_lo;
        }
    }
    return (uint32_t)lo;
}

uint32_t rj_crc32_fast_riscv_zvbc(uint32_t crc, const void *data, size_t len)
{
    const uint8_t *p = (const uint8_t *)data;
    /* 128-bit fold state; S0 = G(crc) per the chaining derivation. */
    uint64_t s_lo = crc, s_hi = 0;

    while (len >= 16) {
        fold16_step(&s_lo, &s_hi, rj_load_le64(p), rj_load_le64(p + 8));
        p += 16;
        len -= 16;
    }
    if (len)
        fold_tail_step(&s_lo, &s_hi, p, len);

    /* Running CRC = red_QREF(S). len==0 falls through with S = G(crc),
     * and the reduction is the identity below 2^32: the chaining
     * identity f(c, empty) = c holds with no special case. NULL data
     * with len==0 never dereferences. */
    return fold_reduce_qref(s_hi, s_lo);
}

/* ------------------------------------------------------------------ */
/* Variant 2: Zbc scalar clmul fold (zlib-ng crc32_zbc.c structure).    */
/* ------------------------------------------------------------------ */

#define ZBC_R3 0x1751997d0ULL
#define ZBC_R4 0x0ccaa009eULL
#define ZBC_R5 0x163cd6124ULL
#define ZBC_RU 0x1F7011641ULL
#define ZBC_POLY 0x1DB710641ULL
#define ZBC_MASK32 0xFFFFFFFFULL

/* The clmul bulk. len >= 16 && len % 16 == 0. See the file-header note
 * for the chaining adaptation (seed = crc, return as-is). */
static uint32_t zbc_fold_bulk(uint32_t crc, const uint8_t *p, size_t len)
{
    uint64_t low = rj_load_le64(p) ^ crc;
    uint64_t high = rj_load_le64(p + 8);
    p += 16;
    len -= 16;

    while (len >= 16) {
        /* clmul/clmulh (low/high 64 bits of the RISC-V insns) are the
         * .lo/.hi halves of one rj_clmul128 each. */
        rj_clmul128_t a = rj_clmul128(ZBC_R4, high);
        rj_clmul128_t b = rj_clmul128(ZBC_R3, low);
        low = b.lo ^ a.lo;
        high = b.hi ^ a.hi;
        low ^= rj_load_le64(p);
        high ^= rj_load_le64(p + 8);
        p += 16;
        len -= 16;
    }

    /* finish_fold: fold the 128-bit (low,high) into 64 bits. */
    rj_clmul128_t f = rj_clmul128(low, ZBC_R4);
    low = high ^ f.lo;
    high = f.hi;

    /* Barrett reduction to 32 bits (their uint64_t truncation preserved:
     * every clmul result below is the low 64 bits, via .lo). */
    uint64_t combined = (low >> 32) | ((high & ZBC_MASK32) << 32);
    uint64_t reduced_low =
        rj_clmul128(low & ZBC_MASK32, ZBC_R5).lo ^ combined;
    uint64_t barrett =
        rj_clmul128(reduced_low & ZBC_MASK32, ZBC_RU).lo & ZBC_MASK32;
    barrett = rj_clmul128(barrett, ZBC_POLY).lo;
    uint64_t final = barrett ^ reduced_low;
    return (uint32_t)(final >> 32);
}

uint32_t rj_crc32_fast_riscv_zbc(uint32_t crc, const void *data, size_t len)
{
    const uint8_t *p = (const uint8_t *)data;
    if (len < 16)
        return rj_crc32_update(crc, data, len);
    /* zlib-ng structure: scalar head to a 16-byte multiple, then the
     * clmul bulk. rj's slice-by-8 stands in for their braid. */
    size_t head = len % 16;
    if (head) {
        crc = rj_crc32_update(crc, p, head);
        p += head;
        len -= head;
    }
    return zbc_fold_bulk(crc, p, len);
}

/* ------------------------------------------------------------------ */
/* Internal runtime selection.                                         */
/* ------------------------------------------------------------------ */

int rj_crc32_fast_riscv_class_for(const rj_cpu_info *info)
{
    /* info is required non-NULL (test hook; production always passes
     * the rj_cpu_detect() output). */
    if (info->rvv_zvbc)
        return 2;
    if (info->zbc)
        return 1;
    return 0;
}

rj_crc_variant_fn rj_crc32_fast_riscv_fn_for(const rj_cpu_info *info)
{
    switch (rj_crc32_fast_riscv_class_for(info)) {
    case 2:
        return rj_crc32_fast_riscv_zvbc;
    case 1:
        return rj_crc32_fast_riscv_zbc;
    default:
        return rj_crc32_update;
    }
}

static pthread_once_t riscv_crc_once = PTHREAD_ONCE_INIT;
static rj_crc_variant_fn riscv_crc_selected = NULL;
static int riscv_crc_selected_class = 0;

static void riscv_crc_select_once(void)
{
    rj_cpu_info info;
    rj_cpu_detect(&info);
    riscv_crc_selected_class = rj_crc32_fast_riscv_class_for(&info);
    riscv_crc_selected = rj_crc32_fast_riscv_fn_for(&info);
}

uint32_t rj_crc32_fast_riscv(uint32_t crc, const void *d, size_t n)
{
    pthread_once(&riscv_crc_once, riscv_crc_select_once);
    return riscv_crc_selected(crc, d, n);
}

int rj_crc32_fast_riscv_class(void)
{
    pthread_once(&riscv_crc_once, riscv_crc_select_once);
    return riscv_crc_selected_class;
}
