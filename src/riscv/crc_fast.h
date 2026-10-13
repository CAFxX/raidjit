/* RISC-V fast CRC-32 variants (Phase 1, Option C standalone API).
 *
 * Private header: the pinned cross-worker entry points (spec R2.5). All
 * variants implement the IDENTICAL chaining contract to rj_crc32_update:
 * they take a running CRC and return the updated running CRC; init/final
 * via rj_crc32_init/rj_crc32_final. Bit-exact vs zlib crc32 (IEEE
 * 0x04C11DB7) on all inputs; see tests/test_crc_fast.c (R4/R5 matrix, W4).
 *
 * Dispatch order (best first), selected internally by rj_crc32_fast_riscv:
 *   2. zvbc -- Zvbc vector fold, SEW=64 (vclmul.vv/vclmulh.vv), streams
 *              packed as elements. Gated on rvv_zvbc.
 *   1. zbc  -- scalar clmul fold (zlib-ng crc32_zbc.c structure, ByteDance /
 *              Yin Tong 2025). Gated on zbc.
 *   0. scalar -- existing slice-by-8 rj_crc32_update (reused, not copied).
 *
 * Every entry point is safe to call on any RISC-V CPU (and on non-RISC-V
 * hosts: this TU compiles unconditionally): a variant whose ISA
 * requirements are not met degrades to a slower CORRECT variant (never
 * wrong, never SIGILL).
 *
 * ---------------------------------------------------------------------------
 * Toolchain note (why this is portable C, not intrinsics or asm)
 * ---------------------------------------------------------------------------
 * The pinned toolchain (riscv64-linux-gnu-gcc 13.3) provides NO vector
 * intrinsics for vclmul.vv/vclmulh.vv (absent from riscv_vector.h;
 * verified by grep) and NO scalar __builtin_riscv_clmul; the task forbids
 * raw asm. QEMU 8.2.2 cannot execute vclmul.vv/vclmulh.vv anyway (SIGILL
 * even when the isa string advertises Zvbc; see src/riscv/cpu.c). Both
 * variants are therefore implemented in portable C with an inlined
 * 64x64->128-bit carryless-multiply emulation (rj_clmul128). The
 * ALGORITHMIC structures are faithful to the spec:
 *   - zvbc: the §11 single-128-bit-state fold (K1/K2 + tail constants),
 *     whose dataflow is exactly what SEW=64 vclmul.vv/vclmulh.vv would
 *     execute (state = 2 x u64 elements, per-element clmul against the
 *     (K1,K2) constant vector);
 *   - zbc: the zlib-ng two-accumulator scalar fold (R3/R4/R5/RU Barrett).
 * Bit-exactness -- not the instruction selection -- is the gate, and it
 * is proven by the R4 matrix. A future toolchain with vclmul intrinsics
 * can replace rj_clmul128's body with __riscv_vclmul_vv_u64m1 with zero
 * algorithmic change.
 *
 * Chaining math (verified in
 * hidden_files/crc_optimum_research/fold_verify.py and re-verified by the
 * R4 split-point tests): the reflected CRC register is tracked by the
 * reduced 128-bit fold state t = red(S) mod QREF, which evolves
 * self-contained (t' = (t ^ red(C)) * x^{-128}); seeding the 128-bit
 * state S0 = G(crc) (incoming running CRC in the low 32 bits) and
 * returning red(S) therefore chains exactly, with len==0 the identity.
 */
#pragma once
#include <stddef.h>
#include <stdint.h>

#include "raidjit.h" /* rj_cpu_info, for the test-only selection hooks */

uint32_t rj_crc32_fast_riscv(uint32_t crc, const void *d, size_t n);
/* Best available RISC-V variant (internal runtime selection per the chain
 * above; first-call selection, thread-safe). */

uint32_t rj_crc32_fast_riscv_zvbc(uint32_t crc, const void *d, size_t n);
/* Zvbc vector fold (SEW=64 element dataflow). */

uint32_t rj_crc32_fast_riscv_zbc(uint32_t crc, const void *d, size_t n);
/* Zbc scalar clmul fold (zlib-ng structure). */

int rj_crc32_fast_riscv_class(void);
/* Which variant the internal selection picks on this machine:
 *   2 = zvbc, 1 = zbc, 0 = scalar. */

/* Test-only selection hooks (pure functions of synthetic cpu info; the
 * production path caches rj_cpu_detect() via pthread_once). Tests use
 * these to prove dispatch correctness without depending on host
 * hardware. */
typedef uint32_t (*rj_crc_variant_fn)(uint32_t crc, const void *d, size_t n);
int rj_crc32_fast_riscv_class_for(const rj_cpu_info *info);
rj_crc_variant_fn rj_crc32_fast_riscv_fn_for(const rj_cpu_info *info);
