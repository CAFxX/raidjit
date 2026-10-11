#pragma once
#include "jit.h"
#include "plan.h"

/* Benchmark every version, pick the winner. Returns winner index (or -1 on
 * error). Sets gbps for every version. Protocol: (1) every candidate's
 * output is validated against the scalar reference before timing -- a wrong
 * candidate fails compilation; (2) timing uses warm-up, interleaved rounds,
 * medians over several working-set sizes; (3) the noise rule: any version
 * within max(1.5%, measured noise) of the best total time is treated as
 * tied; among ties the lowest uops/byte (then lowest mem/byte, then
 * smallest code) wins. Performance claims are therefore limited to: best
 * among the generated legal candidates under this protocol on this CPU.
 *
 * Length-dispatch twins (version_t.notail, see docs/length_dispatch.md)
 * are validated only at group-multiple lengths and timed at group-rounded
 * sizes; each version's GB/s uses its own byte total.
 *
 * noise_out (nullable): if non-NULL, receives the measured eligibility
 * window (max(1.5%, pooled noise)) used for the selection, so callers
 * doing multi-stage selection (track A5's significance gate) can apply
 * the same noise criterion. */
int bench_select(version_t *vs, int nv, const plan_t *pl,
                 const rj_options *opts, int verbose, double *noise_out,
                 char *err, size_t errlen);

/* Static heuristic pick (no benchmarking). Returns winner index. */
int heuristic_select(version_t *vs, int nv, const plan_t *pl);
int validate_versions(version_t *vs, int nv, const plan_t *pl,
                      char *err, size_t errlen);

/* Pure selection step, factored out for unit testing: given measured median
 * seconds/call in secs[] and a relative eligibility window, pick the winner
 * index. Only candidates within `window` of the best time are eligible;
 * ties break on uops/byte, then mem/byte, then code size. bench_select
 * passes window = max(1.5%, measured noise). */
int select_by_time(const version_t *vs, int nv, const double *secs,
                   double window);

/* Pure noise-window computation, factored out for unit testing. csn is
 * nv*nsizes row-major relative IQRs; returns max(1.5%, pooled noise).
 * pcand_out (nullable) receives per-candidate noise in candidate order. */
double noise_window(const double *csn, int nv, int nsizes, double *pcand_out);

/* Tail duel (track B1x86): benchmark a masked/predicated tail twin against
 * the general (scalar-tail) version at tail-heavy lengths, and gate its
 * installation.
 *
 * The standard bench_select runs at large sizes where the tail is a
 * negligible fraction -- a masked twin can never win there, and must not
 * be judged there. The duel instead measures both versions at
 * tail-dominated lengths (small lens, group-relative tails) plus a few
 * large multiple-plus-tail lengths as a no-regression gate.
 *
 * Returns 1 if the twin wins its install gate: its tail-dominated score
 * (sum of per-length medians) beats general STRICTLY outside the noise
 * window (a tie keeps the status-quo general -- installing a dispatcher
 * has a cost), AND it is no slower than general on the SUM of the large
 * lengths (tolerance = max(window, 5%)). Returns 0 if the twin loses the duel (correct but
 * not faster: the caller drops it). Returns -1 on error: like
 * bench_select, a twin that fails validation fails compilation (wrong
 * code must never ship), so the caller must treat -1 as fatal.
 *
 * Both versions are validated against the scalar reference at every duel
 * length before timing. On a 1 return, twin->gbps is set from the duel
 * measurements (tail-length GB/s, for dispatcher ranking); the general
 * version is untouched.
 *
 * The install gate is ISA-agnostic (any TAIL_MASKED twin); the
 * length-aware dispatcher (dispatch.h) then routes tail lengths to the
 * installed twin deterministically. */
int bench_tail_duel(version_t *twin, version_t *general, const plan_t *pl,
                    const rj_options *opts, int verbose,
                    char *err, size_t errlen);

/* Duel-runner signature (bench_tail_duel's), injectable so api.c's
 * tail-twin install gate is unit-testable with a mock duel. */
typedef int (*tail_duel_fn)(version_t *twin, version_t *general,
                            const plan_t *pl, const rj_options *opts,
                            int verbose, char *err, size_t errlen);

/* Pure duel decision, factored out for unit testing. twin_meds /
 * general_meds hold per-length median seconds over the ndecisive
 * tail-dominated lengths; twin_large / general_large over the nlarge
 * no-regression lengths; window is the relative noise window
 * (>= 1.5%, e.g. from noise_window). Returns 1 (install) iff the twin's
 * tail score wins strictly outside the window AND the SUM of the large
 * lengths shows no regression (tolerance = max(window, 5%)); 0
 * otherwise. */
int tail_duel_decide(const double *twin_meds, const double *general_meds,
                     int ndecisive, const double *twin_large,
                     const double *general_large, int nlarge,
                     double window);
