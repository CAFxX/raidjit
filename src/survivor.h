/* Cheapest measured survivor set for recovery (track A5).
 *
 * For a recover op with nf < npar, the nf parity-row equations used by
 * the recovery solve can be chosen: every failed parity disk j forces
 * its row into the set, and the f_data failed data disks need f_data
 * further rows from the surviving parity rows. Each candidate row
 * subset yields a different recovery plan (different coefficients,
 * different plan kind, different parity disks actually read); the
 * benchmark picks the measured cheapest.
 *
 * All of this happens at plan time (Carlo's invariant): the chosen set
 * bakes into the plan matrix, and the runtime (A2 dispatcher) sees an
 * ordinary plan.
 */
#pragma once
#include "plan.h"
#include "jit.h"
#include "../include/raidjit.h"

/* One candidate parity-row equation set. */
typedef struct {
    int *rows;     /* [nrows] parity-row indices, sorted ascending;
                    * nrows == nf */
    int nrows;
    plan_t plan;   /* built plan: solve over rows[] + strategy select */
    int kind_rank; /* P_XOR=0, P_HORNER=1, P_MULACC=2, P_DELTA=3
                    * (informational; the L2 shortlist keys on the
                    * l2_* exact static signal instead) */
    long nnz;      /* nonzero entries in plan.mat (informational) */
    int ncols;     /* nonzero columns in plan.mat (informational) */
    int is_today;  /* 1 = the all-rows greedy pivot set (status quo):
                    * always shortlisted, the no-regression anchor.
                    * Identified by matrix equality vs the all-rows
                    * solve; only the FIRST match is marked (two
                    * distinct subsets can in principle yield
                    * bit-identical R -- either is a valid anchor). */
    double l2_uops; /* exact static signal for the L2 shortlist: the
                     * canonical version's (plan_candidates[0]: best
                     * ISA, highest fitting U) real uops_per_byte from
                     * jit_build_version -- the twin-model lockstep rule
                     * (AGENTS.md) forbids a second, duplicated model,
                     * so the signal is read off the real version.
                     * Deliberately NT/prefetch-blind: always the plain
                     * temporal canonical (P1.2 decision). */
    double l2_mem;  /* canonical version's real mem_per_byte (same
                     * NT/prefetch-blindness as l2_uops) */
    int l2_valid;   /* 1 if l2_uops/l2_mem hold a built version's values */
} survivor_subset_t;

/* Enumerate every valid (non-singular) parity-row subset for a recover
 * op. F[0..nf-1] is the validated sorted/deduped failed list
 * (1 <= nf <= npar). On success *out holds nsub subsets (caller frees
 * with survivor_free_subsets). Returns 0 on success, -1 on error/OOM,
 * -2 when the combination count exceeds SURVIVOR_ENUM_CAP (err set; the
 * caller falls back to the single all-rows plan). Singular subsets are
 * silently dropped (common for XOR array codes). */
#define SURVIVOR_ENUM_CAP 8192
int survivor_enumerate(const topo_int_t *ti, const int *F, int nf,
                       const char *opname, survivor_subset_t **out,
                       int *nsub_out, char *err, size_t errlen);
void survivor_free_subsets(survivor_subset_t *subs, int nsub);
/* 1 if the subset enumeration for (ti,F,nf) would exceed
 * SURVIVOR_ENUM_CAP (the caller then falls back to the single all-rows
 * plan instead of failing). F is the validated sorted/deduped failed
 * list. */
int survivor_over_cap(const topo_int_t *ti, const int *F, int nf);

/* L2 shortlist (A1-style static prune across subsets). For each valid
 * subset the exact static signal is the canonical version's real
 * (uops_per_byte, mem_per_byte) from jit_build_version (see
 * survivor_l2_signal; no duplicated cost model). The shortlist keeps:
 *   - the pareto frontier on (l2_uops, l2_mem), then
 *   - the is_today subset (the no-regression anchor) unconditionally,
 *   - at most the statically top-(K-1) pareto members,
 * with a HARD CAP of SURVIVOR_SHORTLIST_CAP subsets total: keep =
 * {today} ∪ top-(K-1) statically-ranked pareto members, |keep| <= K.
 * Subsets whose canonical version failed to build (l2_valid == 0) are
 * never treated as dominated -- they stay in the pareto set and rank
 * last statically, so a missing signal can only keep a subset, never
 * drop it.
 * RJ_NO_SUBSET_PRUNE (non-empty and != "0", same contract as
 * RJ_NO_PRUNE) disables the shortlist: every subset is kept.
 * RJ_SUBSET_PRUNE_INVERT is a test-only hook (inverts the pareto
 * decision, skips the is_today anchor) for red-green mutation tests;
 * never set in production. */
#define SURVIVOR_SHORTLIST_CAP 3
void survivor_shortlist(survivor_subset_t *subs, int nsub, int *keep);

/* Build the L2 static signal for every subset: exactly ONE version per
 * subset (plan_candidates[0]: best ISA, highest fitting U) via
 * jit_build_version, then read its real (uops_per_byte, mem_per_byte).
 * The version is freed immediately; only the signal is kept. Subsets
 * whose canonical version cannot be built get l2_valid = 0 (the
 * shortlist treats them as never-dominated). Deterministic given
 * (cpu, o): the shortlist, and hence the C1 tripwire, is reproducible.
 * NT/prefetch candidate flags follow the same rule as the full build
 * in survivor_select.
 *
 * The signal is deliberately NT/prefetch-blind (P1.2 decision): a
 * static extension to the twin family is vacuous -- NT twins tie
 * their temporal original on the static counts by construction and
 * prefetch twins are statically dominated by theirs (see
 * docs/survivor_sets.md section 2c). The no-flip property rests on
 * the NT-delta subset-independence assumption documented there, with
 * the Part-7 tripwire (which runs the corpus with NT twins enabled)
 * as the empirical backstop. Re-audit trigger: a future twin family
 * with subset-dependent benefit must re-audit this signal. */
void survivor_l2_signal(survivor_subset_t *subs, int nsub,
                        const rj_cpu_info *cpu, const rj_options *o);

/* Significance gate against the no-regression anchor (pure; unit
 * tested). pool_sub[i] is the subset index (0..nsub-1) of pool entry i;
 * today_si is the subset index of the is_today anchor (-1 if the anchor
 * is not in the pool); winner is bench_select's winner index; window is
 * bench_select's measured noise window (max(1.5%, pooled noise)).
 *
 * If the winning subset is not today's AND its lead over today's best
 * measured GB/s is within the noise window, the win is a noise flip:
 * the fallback is select_by_time restricted to today's pool entries
 * with the measured window -- the bench protocol's own selector, with
 * 1/gbps as the relative-time proxy (exact: vtot is identical for
 * every pooled version). Within-window ties break by uops/byte, then
 * mem/byte, then code size, exactly as bench_select does. Genuine wins
 * (lead > window) are returned unchanged. For in-range inputs the
 * return is always a valid pool index; a bad winner (out of range),
 * a missing anchor (today_si < 0), or a non-positive window is passed
 * through unchanged -- the gate only adjudicates well-formed inputs.
 *
 * No-regression reasoning: the fallback fires only when the winner's
 * lead over today's best is within the window, so every candidate the
 * selector can return is measurement-indistinguishable from the
 * winner; the returned version is never measurably worse than today's
 * measured best.
 */
int survivor_significance_gate(const version_t *pool, const int *pool_sub,
                               int npool, int today_si, int winner,
                               double window);

/* Full plan-time selection for a recover op: enumerate, L2 static
 * signal + shortlist (at most SURVIVOR_SHORTLIST_CAP subsets), build +
 * A1-prune versions per kept subset, benchmark the pooled versions with
 * the standard compile-time benchmark, apply the significance gate
 * against today's anchor (a noise-flip win falls back to today's best
 * measured version), return the winner. F[0..nf-1] is the validated
 * sorted/deduped failed list.
 *
 * Returns the winner index 0 <= ret < *nv_out, -1 on error (err
 * set), or -2 when the enumeration exceeds SURVIVOR_ENUM_CAP (err set;
 * the caller falls back to the single all-rows plan). The caller owns
 * *vs_out (jit_free_version every entry but the
 * winner) and *plans_out (plan_free each entry); the winner's
 * version_t.plan borrows the corresponding *plans_out entry and must
 * stay alive while the winner is used (through the A2 twin stage).
 *
 * The significance gate is skipped when benchmarking is off (no
 * measured noise window exists) or when the anchor is absent from the
 * pool (only under the RJ_SUBSET_PRUNE_INVERT test hook). The
 * test-only RJ_SURVIVOR_NO_GATE hook (any value) also skips the gate,
 * returning bench_select's raw winner; the L2 tripwire uses it to
 * compare raw measured winners.
 *
 * win_rows_out (may be NULL): when non-NULL, receives the winning
 * plan's equation row set (nf sorted ints) -- the subset dimension of
 * the winner, needed for the winner's stable plan ID (#29). */
int survivor_select(const topo_int_t *ti, const int *F, int nf,
                    const char *opname, const rj_cpu_info *cpu,
                    const rj_options *o, version_t **vs_out, int *nv_out,
                    plan_t **plans_out, int *nplans_out, int *win_rows_out,
                    char *err, size_t errlen);
