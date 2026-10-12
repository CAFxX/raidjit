/* cpu_wasm_x86.c — rj_cpu_detect stub for the x86-64 WASM module.
 * Copies the synthetic bundle installed by rjw_api.c; the native
 * src/x86_64/cpu.c (cpuid/xgetbv asm) is NOT compiled into the module. */
#include "rjw_wasm.h"

void rj_cpu_detect(rj_cpu_info *out)
{
    *out = *rjw_cpu_current();
}
