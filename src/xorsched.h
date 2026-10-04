/* XOR-schedule / cross-row CSE (track A3).
 *
 * Plan-time XOR DAG IR + greedy common-subexpression elimination over the
 * P_XOR parity equations. Produces an optional straight-line program
 * (xorsched_t) attached to the plan; schedule-driven P_XOR emitters in the
 * three backends consume it. All CSE/scheduling happens here, at plan
 * time -- the emitters just walk the program.
 *
 * Algorithm (greedy flat pair-CSE):
 *  1. Deduplicate identical matrix rows (duplicates become vector moves).
 *  2. Repeatedly extract the most co-occurring unordered INPUT pair
 *     (a,b) across unique rows into a temp t_k = in_a ^ in_b, replacing
 *     the pair in every row containing both. Stops when the best pair
 *     occurs in < 2 rows or 1024 temps are reached. Temps are flat pairs
 *     of inputs only -- never nested (keeps the live temp set at 2 and
 *     the register-pressure formula exact).
 *  3. If no temp was extracted, there is no schedule (NULL).
 *
 * Each extracted temp provably replaces >= 1 XOR uop per user row, but
 * extraction is only a heuristic: the twin is benchmark-gated, and
 * plan_candidates only enumerates it when xorsched_version_uops predicts
 * a STRICT static win under the emitter's own counting convention.
 */
#pragma once
#include "plan.h"

#ifdef __cplusplus
extern "C" {
#endif

/* One extracted temp: t_k = in_a ^ in_b (a < b, input indices). */
typedef struct {
    int a, b;
    int nusers;
    int *users; /* [nusers] unique-row indices, ascending */
} xorsched_temp_t;

/* Straight-line XOR program for a P_XOR plan.
 * Atom ids: temp k -> k (0 <= k < ntemps); input i -> ntemps + i. */
typedef struct xorsched {
    int n_in, n_out;
    int ntemps;
    xorsched_temp_t *temps; /* [ntemps], formation order */
    int nunique;            /* deduplicated row count */
    int *urow_of;   /* [n_out] unique-row index for each plan row */
    int *first_row; /* [nunique] canonical (first) plan row */
    int *mult;      /* [nunique] row multiplicity (>= 1) */
    int *row_natoms;  /* [nunique] */
    int **row_atoms;  /* [nunique][k]: atom ids, temps (formation order)
                       * then inputs (ascending) */
    int *row_first; /* [nunique]: first atom id, -1 if the row is empty */
    int *in_nusers; /* [n_in] unique rows where input i survives directly */
    int **in_users; /* [n_in][k] */
    int ndead_plan; /* plan rows with no atoms (zeroed by prologue) */
    int ndup_live;  /* live duplicate plan rows (emitted as moves) */
} xorsched_t;

/* Build the schedule for a P_XOR plan. Returns NULL when there is nothing
 * to share (not P_XOR, < 2 rows, no pair co-occurs in >= 2 rows), when
 * the plan exceeds the JIT-time guards (n_in*n_out > 16384, n_in > 256),
 * or on allocation failure (fail-stop: any partial state is unwound, no
 * corrupt schedule is ever returned). The caller owns the result
 * (xorsched_free). */
xorsched_t *xorsched_build(const plan_t *pl);
void xorsched_free(xorsched_t *s);

/* Test hook: fail the Nth allocation inside xorsched_build (counting from
 * 0), to exercise the OOM unwind path. -1 (default) disables; inert
 * otherwise. Not thread-safe: tests only. */
void xorsched_test_oom_at(int n);

/* Predicted version-level core vector uops for the naive (sched=0) or
 * scheduled (sched=1) P_XOR body, under each backend's exact counting
 * convention (x86/AArch64: body + stores per group; RVV: the triple
 * gen_body emission + stores + 3 vsetvli, matching the emitter).
 * Prefetch/pointer/loop overhead excluded identically on both sides.
 * Model/emitter exactness holds at prefetch_distance=0 only: the model
 * excludes prefetch uops while the emitter counts them, so a version
 * with prefetch enabled reports more uops than the model predicts. The
 * gate is unaffected: every contributing input is prefetched exactly
 * once in both bodies, so prefetch is a wash between the twins.
 * The nt flag selects the non-temporal store accounting: on NEON, nt
 * stores pair adjacent lanes with STNP (one uop per pair, trailing lane
 * via STR); on every other ISA nt stores cost one uop per vector just
 * like temporal stores (RVV nt is an uncounted NTL.ALL hint). The gate
 * always passes nt=0 (base twin); nt=1 exists so tests can check the
 * +nt combo versions against the model exactly.
 * The notail flag selects the length-dispatch no-tail twin's accounting:
 * on RVV the notail twin emits only the single nu=U body (plus the one
 * prologue vsetvli), not the triple emission; on x86/AArch64 the notail
 * twin omits only the uncounted scalar tail, so the prediction is
 * unchanged. The plan_candidates gate always passes notail=0 (it
 * enumerates full versions); notail=1 exists so tests can check the
 * notail combo versions against the model exactly.
 * Used by the plan_candidates gate (strict <) and by tests to check the
 * emitters against the model. */
long xorsched_version_uops(const xorsched_t *s, const plan_t *pl,
                           isa_t isa, int U, int sched, int nt, int notail);

/* Register need for the scheduled twin (P_XOR only): n_out*U + 1 on x86
 * AVX (folded memory XOR forms the temp with one live reg), +1 more on
 * x86-SSE / AArch64 / RVV (two temps: t and t2/vd1). */
int xorsched_need_regs(isa_t isa, const plan_t *pl, int U);

#ifdef __cplusplus
}
#endif
