/* rjw_wasm.h — shared internals for the WASM compile-only API.
 *
 * Each per-ISA module (tools/wasm/SPEC.md §S3) compiles rjw_api.c plus one
 * cpu_wasm_<arch>.c, selected by exactly one of -DRJW_X86 / -DRJW_AARCH64 /
 * -DRJW_RISCV (see tools/wasm/Makefile.wasm).
 */
#pragma once
#include "../include/raidjit.h"

/* The synthetic CPU bundle for this compile, installed by rjw_api.c from
 * the "target" preset before any compile work. The per-arch cpu_wasm_*.c
 * rj_cpu_detect stub copies it out (replacing the native detectors). */
const rj_cpu_info *rjw_cpu_current(void);
