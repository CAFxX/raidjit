/* Generalized fusion planner (queue item #8): internal header.
 *
 * rj_fuse_compile takes a short sequence of ops over one topology,
 * compiles each data op through the public rj_compile planner, checks
 * fusion compatibility up front, and drives the sequence from one
 * cache-blocked tile loop (src/striping.h). See
 * include/raidjit_fused.h for the public contract and docs/fused_crc.md
 * for the design.
 */
#pragma once
#include "../include/raidjit_fused.h"

/* Maximum ops per fused sequence; longer sequences are rejected at
 * compile time (fail closed). The motivating fusions need at most 3. */
#define RJ_FUSED_MAX_OPS 4
/* Maximum total flat buffers (sum of per-op in+out counts) staged per
 * tile on the stack; longer layouts are rejected at compile time. */
#define RJ_FUSED_MAX_FLAT 2048

/* Sub-unit info for wrappers (e.g. api_crc.c's rj_get_info_crc):
 * idx is the sequence index of a data op. */
void rj_fused_sub_info(const rj_fused *f, int idx, rj_info *info);
int rj_fused_sub_ndata(const rj_fused *f, int idx); /* -1 unless ENCODE */
int rj_fused_sub_npar(const rj_fused *f, int idx);  /* -1 unless ENCODE */

/* Direct driver call for internal composition (e.g. api_crc.c's
 * adapter): runs the fused tile loop without going through a
 * trampoline slot. Same contract as rj_fn_fused. */
void rj_fused_call(const rj_fused *f, const void *const *in, void *const *out,
                   size_t len, uint32_t *crc);

/* Phase 2 (interleaved CRC fold) microarchitecture gate: 1 if in-kernel
 * CRC folding is allowed on this CPU, 0 for scalar fallback. Conservative:
 * unknown/new µarchs always return 0. See
 * hidden_files/crc_fold_phase2_SPEC.md §2. */
int fused_crc_interleave_ok(const rj_cpu_info *cpu);
