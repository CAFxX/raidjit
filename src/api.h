#pragma once
#include "jit.h"
#include "bench.h"

/* Internal stage-2 wiring of rj_compile (src/api.c), exposed for unit
 * tests (tests/test_tailduel.c). Not part of the public API.
 *
 * One tail twin's install gate (SVE predtail twin, track B1sve; AVX-512
 * masked-tail twin, track B1x86).
 *
 * Consumes `tw` -- a twin of the shootout winner built by the caller --
 * and runs the tail duel against the general version: tail-heavy lengths
 * (where the twin must earn its place) plus a large-size no-regression
 * gate. Returns:
 *   >= 0: install -- the twin was adopted into twins[(*ntwins)++]; the
 *           return value is its index;
 *   -1:     drop -- the twin lost the duel; it is freed, not installed
 *           (tail traffic falls back to the scalar tail);
 *   -2:     fatal -- the duel's validation or benchmark failed; the twin
 *           is freed and the caller must fail compilation (wrong code
 *           must never ship quietly).
 *
 * `duel` is the duel runner: bench_tail_duel in production, a mock in
 * tests. The gate is ISA-agnostic -- it only needs the twin's fn,
 * group_bytes and plan. The caller must have room in twins[] (the gate
 * appends at most one entry).
 */
int api_tail_duel_gate(version_t *tw, version_t *general, const plan_t *wpl,
                       const rj_options *o, int verbose, version_t *twins,
                       int *ntwins, tail_duel_fn duel, const char *name,
                       char *err, size_t errlen);

/* Track B2 (SOTA item 15): stage-2 permi2b duel eligibility (src/api.c).
 * Exposed for unit tests (tests/test_permi2b.c).
 *
 * 1 = the stage-1 `winner` may be duelled against its permi2b twin:
 * AVX-512, MUL_NIBBLE, P_MULACC/P_DELTA plan, twin not already
 * installed, host reports VBMI, and the plan has at least one
 * nontrivial coefficient (else the twin is byte-identical to plain and
 * the duel is vacuous). 0 = skip the duel.
 *
 * Only reads winner->cand, wpl and cpu (no build, no benchmark).
 */
int api_permi2b_duel_eligible(const version_t *winner, const plan_t *wpl,
                              const rj_cpu_info *cpu);

/* Phase 2 (interleaved CRC fold): query whether a compiled version has
 * in-kernel CRC folding active. 1 = the kernel expects a fold_state
 * argument (nstreams*16 bytes, 16-byte aligned, nstreams known to the
 * driver from the topology); 0 = plain encode kernel (scalar fallback
 * path). Internal only (fused.c driver); NOT public. */
int rj_jit_crc_fold(const version_t *v);
