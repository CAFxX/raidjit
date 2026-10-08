/* x86-64 fast CRC-32 variants (Phase 1, Option C standalone API).
 *
 * Private header: the pinned cross-worker entry points (spec R2.5). All
 * variants implement the IDENTICAL chaining contract to rj_crc32_update:
 * they take a running CRC and return the updated running CRC; init/final
 * via rj_crc32_init/rj_crc32_final. Bit-exact vs zlib crc32 (IEEE
 * 0x04C11DB7) on all inputs; see tests/test_crc_fast_x86.c.
 *
 * Dispatch order (best first), selected internally by rj_crc32_fast_x86
 * (spec R2 as amended by Finding 13):
 *   3. vpclmul  -- rungs 1+2, selected internally by
 *                   rj_crc32_fast_x86_vpclmul. Rung 1: Cawley-style fusion,
 *                   512-bit VPCLMULQDQ vector lanes + PCLMUL-based scalar
 *                   lane (vpclmulqdq && pclmulqdq && avx512f && avx512bw).
 *                   Rung 2: scalar+vector fusion with 256-bit VPCLMULQDQ
 *                   lanes (vpclmulqdq && !avx512f, e.g. Zen3).
 *   2. pclmul   -- rung 3: 128-bit PCLMULQDQ FOLD (pure vector, v4 lanes;
 *                   zlib-ng/Chromium structure), NOT fusion. Gated on
 *                   pclmulqdq without vpclmulqdq.
 *   1. crc32hw  -- No distinct hardware path exists for IEEE CRC-32 on
 *                   x86-64 (the crc32 instruction is CRC-32C/Castagnoli-only);
 *                   this entry point delegates to the slice-by-8 fallback.
 *                   The internal selector never picks it as a distinct tier.
 *   0. scalar   -- existing slice-by-8 rj_crc32_update (reused, not copied).
 *
 * Every entry point is safe to call on any x86-64 CPU: a variant whose
 * ISA requirements are not met degrades to a slower CORRECT variant
 * (never wrong, never SIGILL).
 *
 * IMPORTANT (IEEE vs Castagnoli): the x86 `crc32` instruction (SSE4.2)
 * implements CRC-32C (Castagnoli polynomial 0x1EDC6F41), NOT the IEEE
 * CRC-32 (0x04C11DB7) this library computes. It cannot be used here.
 * (Cawley's fast-crc32 documents the same: "x86_64 only has scalar
 * acceleration for -p crc32c"; its -p crc32 x86 output uses PCLMUL-based
 * scalar lanes, which is what this file implements.) ARM is different:
 * ARMv8 has separate IEEE (`crc32b/h/w/x`) and Castagnoli (`crc32cb/ch/cw/cx`)
 * instructions, so the ARM backend CAN use its hw crc32 for IEEE.
 */
#pragma once
#include <stddef.h>
#include <stdint.h>

uint32_t rj_crc32_fast_x86(uint32_t crc, const void *d, size_t n);
/* Best available x86 variant (internal runtime selection per the chain
 * above). */

uint32_t rj_crc32_fast_x86_vpclmul(uint32_t crc, const void *d, size_t n);
uint32_t rj_crc32_fast_x86_pclmul(uint32_t crc, const void *d, size_t n);
uint32_t rj_crc32_fast_x86_crc32hw(uint32_t crc, const void *d, size_t n);

int rj_crc32_fast_x86_class(void);
/* Which variant the internal selection picks on this machine:
 *   3 = vpclmul (rungs 1+2), 2 = pclmul128 (rung 3), 0 = scalar.
 * (1 = crc32hw is never returned: no distinct IEEE hw tier exists on x86-64;
 * an sse42-only CPU reports 0.) */
