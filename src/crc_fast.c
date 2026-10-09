/* Top-level dispatch for the standalone fast-CRC API (Option C, Phase 1;
 * spec: hidden_files/crc_impl_SPEC.md R1/R2/R2.5).
 *
 * rj_crc32_fast(crc, data, len) has a chaining contract IDENTICAL to
 * rj_crc32_update: it takes the running CRC and returns the updated
 * running CRC, bit-exact (rj_crc32_fast(c,d,n) == rj_crc32_update(c,d,n)
 * for all inputs; tests/test_crc_fast.c proves it).
 *
 * Dispatch is by COMPILE-TIME ISA: on x86-64 the call goes to the x86
 * backend's best-variant entry (which does its own runtime feature
 * selection internally), on AArch64 to the ARM backend, on RISC-V to the
 * RISC-V backend, and everywhere else (WASM/unknown) to the portable
 * slice-by-8. The selected entry is bound in a static initializer: no
 * mutable state, no first-call init race, trivially thread-safe.
 *
 * The backend entry points are declared here as plain externs. Their
 * definitions live in the per-ISA backend TUs (src/<isa>/crc_fast.c,
 * spec R2.5), which are unconditionally part of ARCH_SRCS. The reference
 * from this TU to the backend entry is deliberately a strong-undefined
 * one: that is what forces the linker to extract the real backend TU
 * from the archive for every consumer, not just the test binaries.
 * tests/test_crc_fast.c distinguishes backend-present from
 * slice-by-8-fallback via rj_crc32_fast_test_backend_present().
 */
#include "crc.h"

#include <stddef.h>
#include <stdint.h>

#if defined(__x86_64__)
uint32_t rj_crc32_fast_x86(uint32_t crc, const void *d, size_t n);
#elif defined(__aarch64__)
uint32_t rj_crc32_fast_arm(uint32_t crc, const void *d, size_t n);
#elif defined(__riscv)
uint32_t rj_crc32_fast_riscv(uint32_t crc, const void *d, size_t n);
#endif

typedef uint32_t (*rj_crc_fast_fn)(uint32_t, const void *, size_t);

/* Address constant: valid static initializer, resolved at link time to
 * whichever definition (backend or weak fallback) the link selected. */
#if defined(__x86_64__)
static rj_crc_fast_fn g_crc_fast_fn = rj_crc32_fast_x86;
#elif defined(__aarch64__)
static rj_crc_fast_fn g_crc_fast_fn = rj_crc32_fast_arm;
#elif defined(__riscv)
static rj_crc_fast_fn g_crc_fast_fn = rj_crc32_fast_riscv;
#else
/* WASM (__EMSCRIPTEN__) and unknown ISAs (spec Finding 17c): portable
 * slice-by-8, no backend entry. */
static rj_crc_fast_fn g_crc_fast_fn = rj_crc32_update;
#endif

uint32_t rj_crc32_fast(uint32_t crc, const void *data, size_t len)
{
    return g_crc_fast_fn(crc, data, len);
}
