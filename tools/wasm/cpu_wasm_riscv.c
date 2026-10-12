/* cpu_wasm_riscv.c — rj_cpu_detect stub for the RISC-V WASM module.
 * Copies the synthetic bundle installed by rjw_api.c; the native
 * src/riscv/cpu.c (riscv_hwprobe ecall//proc/cpuinfo) is NOT compiled. */
#include "rjw_wasm.h"

void rj_cpu_detect(rj_cpu_info *out)
{
    *out = *rjw_cpu_current();
}
