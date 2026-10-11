/* Calibrated prefetch distance (I4, SOTA reframe of rejected item 18).
 *
 * Item 18's per-kernel prefetch auto-tuning was rejected with evidence
 * (no stable argmin across reps; see docs/limitations.md section 11):
 * benchmarking a distance ladder per kernel costs search time for no
 * consistent win. This module is the reframe: measure the MACHINE once
 * (a ~10 ms pointer-chase + memset probe at first rj_init, or lazily at
 * the first plan that needs an analytical distance), cache the result
 * in-process, and compute each kernel's prefetch distance ANALYTICALLY
 * at plan time from its stride (U x vector-bytes x input streams) and
 * the calibrated latency/bandwidth. Zero runtime cost: the distance is
 * a compile-time constant baked into the emitted prefetch instructions.
 *
 * The analytical distance is a HEURISTIC, not a measured optimum --
 * Carlo's invariant stands: the compile-time benchmark remains the
 * final arbiter, and the distance-0 (no-prefetch) twin is always
 * enumerated alongside the prefetch twin, so a machine whose core
 * ignores or misbehaves on software prefetch still has its candidate.
 * Never present the analytical distance as measured-optimal.
 *
 * Persistence: deliberately in-process only. A persistent machine-profile
 * cache is item #29's database design; when it lands, this module's
 * probe becomes the "miss" path behind it. Do not bolt ad-hoc file
 * caching on here.
 *
 * Heterogeneous-core migration (documented limitation, NOT implemented):
 * the cached calibration describes the core the probe ran on. If the
 * process migrates between heterogeneous cores (big.LITTLE, or a VM
 * vCPU re-pinned across sockets), the cached numbers go stale. The
 * re-calibration trigger (re-run the probe on observed migration) is
 * future work; migration detection is explicitly out of scope for I4.
 */
#pragma once

#include <stddef.h>

#include "plan.h" /* isa_t, RJ_PREFETCH_AUTO */

/* Machine memory characterization from the one-time probe. */
typedef struct {
    double latency_ns;    /* pointer-chase load latency, ns (> 0) */
    double bandwidth_gbs; /* streaming memset bandwidth, GB/s (> 0) */
    int valid;            /* 0 = probe failed/unavailable; callers must
                           * treat an invalid calibration as "no data"
                           * and decline the prefetch twin (fail closed). */
} prefetch_calib_t;

/* Run the probe once per process (idempotent, thread-safe) and return
 * the cached machine calibration. Never returns NULL; check ->valid.
 * Cost: ~10 ms on the first call, ~0 afterwards.
 *
 * Virtualization caveat: on a VM the "DRAM" the probe sees may be the
 * host's last-level cache (the guest cannot size the host LLC), so
 * latency_ns can read as LLC latency and bandwidth_gbs as LLC
 * bandwidth. The product is still the load-use cost the CPU actually
 * experiences for large-footprint access, which is what the distance
 * formula needs -- but treat the numbers as hierarchy
 * characterization, not DRAM datasheet values. The iters clamp and
 * the benchmark arbitration keep the heuristic robust to this. */
const prefetch_calib_t *prefetch_calib_ensure(void);

/* Test-only override (mirrors rj_uarch_override in cpu.h): inject a
 * fake calibration so unit tests can pin latency/bandwidth
 * deterministically instead of depending on the build host. Serves
 * exactly what is injected, including valid=0 (which exercises the
 * fail-closed path). NOT thread-safe; production code never calls
 * this. Sticky for process lifetime -- there is no restore: tests must
 * run real-probe checks before injecting. */
void prefetch_calib_inject(double latency_ns, double bandwidth_gbs,
                           int valid);

/* Analytical prefetch distance for one candidate, in bytes ahead per
 * input stream (the unit of cand_t.prefetch_dist). Returns 0 when no
 * prefetch twin should be emitted: calibration invalid, degenerate
 * geometry (U/vl/streams <= 0), or the computed distance falls below
 * the ISA's minimum encodable distance (RVV prefetch.r cannot encode
 * < 32 bytes -- below one cache line a prefetch is noise anyway).
 *
 * Formula (Little's-law heuristic):
 *   in_flight = latency_ns * bandwidth_gbs      (GB/s == bytes/ns)
 *   stride    = U * vl_bytes * n_streams        (bytes/iteration,
 *                                                all input streams)
 *   iters     = clamp(ceil(in_flight / stride), 1, 64)
 *   dist      = iters * U * vl_bytes, then ISA-range-adjusted:
 *     AArch64 (PRFM imm12*8): clamp to 32760, round down to 8;
 *     RVV (prefetch.r):       32..2016, round down to 32 (< 32 -> 0);
 *     x86 (PREFETCHT0):       any byte distance encodable.
 * The backends independently reject unencodable distances at build
 * time; the rounding here only avoids emitting doomed twins.
 */
size_t prefetch_calib_distance_for(const plan_t *pl, const cand_t *cand,
                                   const rj_cpu_info *cpu);

/* Resolve the plan_candidates() prefetch argument from user options:
 * benchmarking off -> 0 (no prefetch twins, as before);
 * manual override  -> o->prefetch_distance (unchanged behavior);
 * otherwise        -> RJ_PREFETCH_AUTO (per-candidate analytical default).
 * The manual override keeps working exactly as before; the analytical
 * value is only the default when the option is unset. */
size_t prefetch_calib_dist_arg(const rj_options *o);
