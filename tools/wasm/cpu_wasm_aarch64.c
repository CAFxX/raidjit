/* cpu_wasm_aarch64.c — rj_cpu_detect stub for the AArch64 WASM module.
 * Copies the synthetic bundle installed by rjw_api.c; the native
 * src/aarch64/cpu.c (getauxval//proc/cpuinfo/prctl) is NOT compiled. */
#include "rjw_wasm.h"

void rj_cpu_detect(rj_cpu_info *out)
{
    *out = *rjw_cpu_current();
}
