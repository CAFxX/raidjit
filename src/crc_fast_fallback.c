/* Transitional WEAK definitions of the pinned fast-CRC backend entry
 * points (spec R2.5: rj_crc32_fast_<isa>[_variant] / _class).
 *
 * DELETE THIS FILE (and its Makefile stanza) once src/x86_64/crc_fast.c,
 * src/aarch64/crc_fast.c and src/riscv/crc_fast.c have all landed. Until
 * then it keeps the tree building and the R4/R5 matrix meaningful: every
 * entry is bit-exact (it funnels into the slice-by-8 path), just
 * unaccelerated.
 *
 * Link mechanics (verified empirically, see docs/crc.md): the entries
 * are weak ALIASES of one static body. A strong-undefined reference
 * (from src/crc_fast.c's dispatcher, or from a test's direct call)
 * extracts the first archive member defining the symbol -- so this TU
 * sorts AFTER the backend TUs in libraidjit.a, and a real backend's
 * strong definitions override these aliases wherever the backend TU is
 * linked. rj_crc32_fast_test_backend_present() tells the two states
 * apart exactly (address comparison: alias-active means the symbol IS
 * the fallback body), so the test suite's dispatch assertions go strict
 * automatically once a backend lands -- no heuristic, no silent pass.
 */
#include "crc.h"

#include <stddef.h>
#include <stdint.h>

static uint32_t crc_fb_body(uint32_t crc, const void *d, size_t n)
{
    return rj_crc32_update(crc, d, n);
}

/* 0 = scalar, per the R2.5 class enum. */
static int crc_fb_class(void)
{
    return 0;
}

__attribute__((weak, alias("crc_fb_body")))
uint32_t rj_crc32_fast_x86(uint32_t crc, const void *d, size_t n);
__attribute__((weak, alias("crc_fb_body")))
uint32_t rj_crc32_fast_x86_vpclmul(uint32_t crc, const void *d, size_t n);
__attribute__((weak, alias("crc_fb_body")))
uint32_t rj_crc32_fast_x86_pclmul(uint32_t crc, const void *d, size_t n);
__attribute__((weak, alias("crc_fb_body")))
uint32_t rj_crc32_fast_x86_crc32hw(uint32_t crc, const void *d, size_t n);
__attribute__((weak, alias("crc_fb_class")))
int rj_crc32_fast_x86_class(void);

__attribute__((weak, alias("crc_fb_body")))
uint32_t rj_crc32_fast_arm(uint32_t crc, const void *d, size_t n);
__attribute__((weak, alias("crc_fb_body")))
uint32_t rj_crc32_fast_arm_fusion(uint32_t crc, const void *d, size_t n);
__attribute__((weak, alias("crc_fb_body")))
uint32_t rj_crc32_fast_arm_crc32hw(uint32_t crc, const void *d, size_t n);
__attribute__((weak, alias("crc_fb_body")))
uint32_t rj_crc32_fast_arm_pmull(uint32_t crc, const void *d, size_t n);
__attribute__((weak, alias("crc_fb_class")))
int rj_crc32_fast_arm_class(void);

__attribute__((weak, alias("crc_fb_body")))
uint32_t rj_crc32_fast_riscv(uint32_t crc, const void *d, size_t n);
__attribute__((weak, alias("crc_fb_body")))
uint32_t rj_crc32_fast_riscv_zvbc(uint32_t crc, const void *d, size_t n);
__attribute__((weak, alias("crc_fb_body")))
uint32_t rj_crc32_fast_riscv_zbc(uint32_t crc, const void *d, size_t n);
__attribute__((weak, alias("crc_fb_class")))
int rj_crc32_fast_riscv_class(void);

/* Test-only scaffolding (goes away with this file): 1 when a backend
 * TU's strong definitions are linked for this ISA, 0 while the weak
 * fallback above is in effect. */
int rj_crc32_fast_test_backend_present(void)
{
#if defined(__x86_64__)
    return rj_crc32_fast_x86 != crc_fb_body;
#elif defined(__aarch64__)
    return rj_crc32_fast_arm != crc_fb_body;
#elif defined(__riscv)
    return rj_crc32_fast_riscv != crc_fb_body;
#else
    return 0;
#endif
}
