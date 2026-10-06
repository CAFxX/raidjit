#pragma once
#include "../include/raidjit.h"

/* Fills rj_cpu_info. Never claims an extension the OS hasn't enabled
 * (checks XCR0 via XGETBV for AVX/AVX512 state). */
void rj_cpu_detect(rj_cpu_info *out);

/* Microarchitecture tag for the uarch-aware static cost model (I3).
 * Derived from CPUID on x86-64 (vendor + family/model); RJ_UARCH_UNKNOWN
 * on other architectures or when the CPUID leaf is unavailable. */
typedef enum {
    RJ_UARCH_UNKNOWN = 0,
    RJ_UARCH_INTEL,                 /* Intel, family not distinguished */
    RJ_UARCH_AMD,                   /* AMD, family not distinguished */
    RJ_UARCH_INTEL_SKYLAKEX,        /* Skylake-X / Cascade Lake (fam 6, 0x55) */
    RJ_UARCH_INTEL_ICELAKE,         /* Ice Lake client + server
                                     * (fam 6, 0x6A/0x6C/0x7D/0x7E) */
    RJ_UARCH_INTEL_SAPPHIRERAPIDS,  /* Sapphire Rapids / Emerald Rapids
                                     * (fam 6, 0x8F/0xCF) */
    RJ_UARCH_AMD_ZEN3,              /* fam 0x19, models outside Zen4 ranges */
    RJ_UARCH_AMD_ZEN4,              /* fam 0x19, models 0x10-0x1F (Genoa) and
                                     * 0xA0-0xAF (Bergamo/Zen4c) */
    RJ_UARCH_AMD_ZEN5,              /* fam 0x1A */
    RJ_UARCH_COUNT
} rj_uarch_t;

/* Detect the host microarchitecture from CPUID. Pure function of the
 * machine; the result is cached after the first call. */
rj_uarch_t rj_cpu_uarch(void);

/* Effective uarch for the cost model: the test override when set, else
 * rj_cpu_uarch(). */
rj_uarch_t rj_uarch_current(void);

/* Test-only override for the uarch tag, so tests can pin the uarch
 * deterministically instead of depending on the build host. NOT
 * thread-safe; production code never calls this. There is no separate
 * "clear": pass rj_cpu_uarch() to restore host detection. */
void rj_uarch_override(rj_uarch_t u);
