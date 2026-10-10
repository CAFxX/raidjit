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
 * convention (x86/AArch64: body + stores per group; RVV: the double
 * gen_body emission (counted loop + remainder) + stores + 2 vsetvli,
 * matching the emitter).
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
 * prologue vsetvli), not the double emission; on x86/AArch64 the notail
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

/* ---- bit-level (gate-level) XOR scheduling for GF-multiply rows ----
 * (track A1).
 *
 * The byte-level xorsched above only applies to P_XOR rows. For
 * GF-multiply rows (P_MULACC / P_DELTA / P_HORNER) each coefficient c
 * expands to its 8x8 bitmatrix (column k = bits of gf_mul(c, 1<<k)),
 * turning the whole plan into a GF(2)-linear circuit: every output bit
 * is an XOR of input bits (bit k of input byte i). The scheduler runs
 * greedy pair-CSE at bit granularity over these incidence sets --
 * flat pairs of INPUT BITS only (never nested; A3 is the follow-on) --
 * deduplicating identical output-bit rows, and emits a straight-line
 * bit program (xorbit_t) with cross-row AND cross-coefficient sharing:
 * a pair of input bits shared between different coefficients'
 * bitmatrices (e.g. the xtime(x) subexpression inside gf_mul(3,x) =
 * xtime(x)^x) is computed once. This subsumes strength reduction.
 *
 * Bit-program format:
 *   input bit id: q = i*8+k  (0 <= i < n_in_sched, 0 <= k < 8), where
 *     n_in_sched = n_in for P_MULACC/P_HORNER and 2*nchanged for P_DELTA
 *     (the scheduler sees the raw old/new inputs; old^new delta pairs
 *     are discovered by the pair-CSE itself, and oldpar bits never
 *     enter the program -- P_DELTA accumulators are seeded from oldpar
 *     by the emitter).
 *   temp m: t_m = in[a_m] ^ in[b_m]  (input-bit ids, a_m < b_m).
 *   atom ids: temp m -> m (0 <= m < ntemps); input bit q -> ntemps + q.
 *   output bit (j,b) (0 <= j < n_out, 0 <= b < 8) = XOR of the atoms in
 *     bit_atoms[ubit_of[j*8+b]].
 *
 * Backend contract (all ISAs): the emitter materializes each input bit
 * as a bit-0-normalized byte vector ((in_i >> k) & 1), evaluates each
 * temp with vector XORs, and assembles output bytes as
 * out_j = XOR_b (P(j,b) << b) where P(j,b) is the output bit's value.
 * Accumulators start zero (P_MULACC/P_HORNER: dead bytes zeroed by the
 * prologue, live bytes written on first touch, never XORed with zero;
 * P_DELTA: seeded from oldpar, always XORed). All CSE/scheduling is
 * plan-time; the emitter walks the program with no runtime decisions.
 *
 * Build returns NULL when there is nothing to share (wrong kind, no
 * temps extracted), when the plan exceeds the guards
 * (8*n_in_sched*n_out > 131072 bit-rows... see XB_MAX_DIMS, or
 * n_in_sched > 256), or on allocation failure (fail-stop, like
 * xorsched_build). The caller owns the result (xorbit_free). */

typedef struct {
    int a, b;      /* input-bit ids (a < b) */
    int nusers;
    int *users;    /* [nusers] output-bit ids (j*8+b), sorted by (b, j) so
                    * each bit-position group is consecutive for the
                    * emitter's fan-out */
} xorbit_temp_t;

typedef struct xorbit {
    int n_in_sched;  /* scheduler input bytes (see above) */
    int n_out;       /* output bytes */
    int n_in;        /* plan n_in (emitter input count; == n_in_sched
                      * except P_DELTA, where the trailing oldpar inputs
                      * are seed-only) */
    int ntemps;
    xorbit_temp_t *temps; /* [ntemps], formation order */
    int nunique;          /* deduplicated output-bit count */
    int *ubit_of;   /* [8*n_out] unique-bit index for each output bit */
    int *first_bit; /* [nunique] canonical (first) output bit id = j*8+b */
    int *mult;      /* [nunique] multiplicity (>= 1) */
    int *bit_natoms;  /* [nunique] */
    int **bit_atoms;  /* [nunique][k]: atom ids, temps (formation order)
                       * then input bits (ascending) */
    int *bit_first; /* [nunique]: first atom id, -1 if the bit is empty */
    int *inbit_nusers; /* [8*n_in_sched] unique bits where input bit q
                        * survives directly */
    int **inbit_users; /* [8*n_in_sched][k]: output-bit ids (j*8+b) */
    int ndead_bytes; /* output bytes with no atoms (zeroed by prologue) */
    plan_kind_t kind; /* plan kind (P_DELTA seeds accs from oldpar) */
} xorbit_t;

xorbit_t *xorbit_build(const plan_t *pl);
void xorbit_free(xorbit_t *s);

/* Test hook: fail the Nth allocation inside xorbit_build (counting from
 * 0), to exercise the OOM unwind path. -1 (default) disables; inert
 * otherwise. Not thread-safe: tests only. */
void xorbit_test_oom_at(int n);

/* Static cost model for the bit-program twin (plan_candidates gate).
 * sched=0 models the naive nibble body, sched=1 the bit-program body,
 * under the emitter's counting convention (body + stores per
 * group; prologue const setup uncounted; prefetch excluded -- the gate
 * always passes prefetch_dist=0). nt selects non-temporal store
 * accounting (x86: one uop per vector either way). x86 ISAs mirror the
 * x86_64 emitter exactly; NEON/SVE mirror the AArch64 emitter
 * (src/aarch64/jit.c); RVV mirrors the RISC-V emitter (src/riscv/jit.c).
 * For SVE the model assumes SVE2 EOR3 is available, making it a LOWER
 * BOUND on the true naive cost (on non-SVE2 hardware the true naive is
 * costlier); the gate (u_sched < u_naive) therefore stays sound -- it
 * may miss wins on non-SVE2, but never admits a losing twin.
 * Used by the plan_candidates gate (strict <) and by tests to check
 * the emitter against the model. */
long xorbit_version_uops(const xorbit_t *s, const plan_t *pl,
                         isa_t isa, int U, int sched, int nt);

/* Register need for the bit-program twin: n_out*U accumulators + 4
 * (input-byte reg, two bit temps, 0x01 mask). The nibble tables are
 * dead on this path. */
int xorbit_need_regs(isa_t isa, const plan_t *pl, int U);

#ifdef __cplusplus
}
#endif
