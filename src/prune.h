/* Static-dominance pruning of the benchmark candidate list.
 *
 * prune_versions() removes statically-dominated candidates BEFORE
 * wall-clock benchmarking, shrinking the bench_select working set.
 * ("Statically" dominated: dominated on the static cost model. Safety
 * rests on the precondition that no pruned version would be the
 * argmin-time candidate; tests/test_prune.c Part 3 asserts this as a
 * tripwire. See docs/candidate_pruning.md.)
 *
 * The dominance rule is deliberately narrow: only the PLAIN member of a
 * +ternlog twin pair is pruned, and only when the twin (same plan kind,
 * isa, mul, U, nt, prefetch distance, remat, notail, xorsched;
 * ternlog=1 vs 0) pareto-dominates it on the static (uops_per_byte,
 * mem_per_byte, code_size) triple, the twin's 2xVPXOR->VPTERNLOGD
 * substitution is not port-hostile on the host uarch (I3 leg), and the
 * twin's roofline-aware cost does not exceed the plain's on the host
 * uarch (#31 leg) -- and only for plan kinds P_MULACC /
 * P_DELTA, where the twin direction is validated (enforced by a
 * plan-kind guard in prune.c, not by the triple values). Twin pairs are
 * the one place where codegen is structurally identical except for the
 * multiply instruction, so the triple faithfully captures the
 * difference. In particular the rule NEVER prunes across U, across NT,
 * across ISA/mul, across remat, across notail/xorsched, across plan
 * kinds, or a +ternlog twin in favor of plain (the P_XOR twin is
 * empirically competitive -- see docs/candidate_pruning.md).
 *
 * Signature (consumed by the later survivor-set enumeration track):
 *
 *   int prune_versions(version_t *vs, int nv, unsigned flags);
 *
 * Contract:
 * - vs[0..nv-1] are built versions; survivors are compacted to
 *   vs[0..ret-1] preserving their original relative order.
 * - Returns the new version count (0 <= ret <= nv). nv == 0 returns 0.
 * - Pruned versions are finalized inside this call (jit_free_version:
 *   their executable mappings are released); the caller owns only the
 *   returned survivors and must not touch the pruned ones again.
 * - flags & PRUNE_DISABLE, or the RJ_NO_PRUNE environment variable set
 *   to any non-empty value other than exactly "0", disables pruning:
 *   returns nv with vs untouched. The contract is precisely
 *   "non-empty AND != "0"": "1", "false", "00", "0 " (trailing space)
 *   all DISABLE pruning; only "" (empty/unset) and "0" leave it
 *   enabled.
 * - Only the plain (ternlog=0) member of a +ternlog twin pair is ever
 *   pruned, and only for plan kinds P_MULACC / P_DELTA, where the twin
 *   direction is validated (enforced by code in prune.c, not by the
 *   triple values). Safe precondition: no pruned version would be the
 *   argmin-time candidate; tests/test_prune.c Part 3 asserts this as a
 *   tripwire on the unpruned run.
 */
#ifndef RAIDJIT_PRUNE_H
#define RAIDJIT_PRUNE_H

#include "jit.h" /* version_t */

/* prune_versions() flag: disable pruning, return nv unchanged. */
#define PRUNE_DISABLE 1u

/* prune_versions() flag: log kind-guard refusals to stderr. When the
 * twin relation and the triple gate both hold but the plan kind is not
 * one of the validated ones (P_MULACC, P_DELTA), pruning is refused;
 * with this flag the refusal is reported instead of being silent. */
#define PRUNE_VERBOSE 2u

/* Epsilon for floating-point triple comparison: pure float-comparison
 * dust (1e-9) so near-ties from model arithmetic do not count as
 * strict dominance. This is NOT the select_by_time 1.5% measurement
 * noise floor -- the triple gate is a dominance test on the static
 * cost model, not a comparison of measured times. */
#define PRUNE_EPS 1e-9

int prune_versions(version_t *vs, int nv, unsigned flags);

/* Uarch-aware leg of the twin dominance gate (I3, prune-soundness P1),
 * exposed for tests. Returns 1 when the +ternlog twin's 2xVPXOR ->
 * VPTERNLOGD substitution is not port-hostile under the given
 * per-uarch weights (i.e. pruning the plain may proceed); 0 when the
 * substitution costs cycles on that uarch (the plain must be kept).
 * See src/prune.c for the derivation. */
int prune_twin_uarch_ok(const version_t *twin, const version_t *plain,
                        const rj_uarch_weights_t *w);

/* Roofline-aware leg of the twin dominance gate (#31), exposed for
 * tests. Returns 1 when the twin's roofline-aware cost does not exceed
 * the plain's on the given uarch (i.e. pruning the plain may
 * proceed); 0 when it does (the plain must be kept -- e.g. a
 * bandwidth-bound regime where a uop saving does not pay). Veto-only.
 * For the AVX-512 nibble twin (xorbit == 0) the comparison is exact
 * (the 2xVPXOR->VPTERNLOGD substitution economics against an
 * identical bandwidth leg), so the leg agrees with the triple+I3
 * verdict on every uarch; other twin shapes use the averaged
 * plan_uarch_cost comparison. See src/prune.c for the derivation. */
int prune_twin_roofline_ok(const version_t *twin, const version_t *plain,
                           rj_uarch_t uarch);

#endif
