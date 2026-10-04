/* Plan-time search for low-cost MDS generator matrices over GF(2^8).
 *
 * rj_topo_rs()/rj_topo_raid6() no longer hardcode the historical
 * Vandermonde default (M[j][i] = 2^(j*(ndata-1-i))). Instead they call
 * matrix_search_default(), which runs a deterministic, bounded,
 * host-independent search for the lowest-cost MDS systematic generator
 * matrix and caches the result per (ndata, npar).
 *
 * Design summary (full note: docs/matrix_search.md):
 *  - Cost = the planner's own static cost model: x86-64 nibble-backend
 *    vector uops / mem-bytes per stripe byte, transcribed 1:1 from
 *    src/x86_64/jit.c and cross-checked against real codegen by
 *    tests/test_matrix_search.c (exact-equality, so any backend
 *    accounting drift breaks the test instead of silently diverging).
 *    Strategy selection (P_XOR/P_HORNER/P_MULACC) is the planner's real
 *    plan_build, not a reimplementation.
 *  - MDS = exact minor enumeration: every npar x npar minor of the
 *    parity-check matrix H=[-P|I] is checked nonsingular over GF(2^8).
 *    For the supported envelope (npar <= 3, ndata <= 51) that is at most
 *    C(54,3) = 24804 tiny determinants -- a real proof per emitted
 *    matrix, not a randomized check.
 *  - Search = greedy single-entry hill climbing from several fixed
 *    seeds (historical default, Cauchy, fixed-seed random MDS),
 *    accepting only strict cost improvements. The historical default is
 *    seed 0 and wins all ties, so the output is byte-identical to the
 *    old default unless a strictly cheaper MDS matrix is found (none
 *    is: the default is the search's fixed point -- see the design note
 *    for the one-count bound + placement-constraint argument and the
 *    empirical tripwire that carries the conclusion).
 */
#pragma once
#include <stdint.h>

/* Exact MDS check for a systematic generator matrix P (npar x ndata,
 * row-major). Returns 1 iff every choice of npar columns of the
 * parity-check matrix H = [-P | I_npar] is linearly independent over
 * GF(2^8) -- equivalently, every erasure set of size <= npar is
 * recoverable. This enumerates all C(ndata+npar, npar) npar x npar
 * minors exactly (1x1/2x2/3x3 closed forms); npar > 3 is rejected.
 * early_exit != 0 stops at the first singular minor (fast reject for
 * search); early_exit == 0 checks every minor (the proof path).
 * nminors_out (may be NULL) receives the number of minors examined. */
int matrix_is_mds(const uint8_t *P, int ndata, int npar, int early_exit,
                  long *nminors_out);

/* Static encode cost of a generator matrix under the planner's own cost
 * model. Builds the real encode plan with plan_build (so the
 * P_XOR/P_HORNER/P_MULACC strategy choice is the planner's actual
 * code), then evaluates the transcribed x86-64 nibble-backend static
 * model (vector uops and mem-bytes per stripe byte) at the reference
 * shape (U=8, vl=64, 5 hoisted input pointers, VEX encoding).
 * Returns 0 on success, -1 on bad args/OOM. */
int matrix_search_cost(const uint8_t *P, int ndata, int npar,
                       double *uops_per_byte, double *mem_per_byte);

/* Lower-level cost components behind matrix_search_cost, exposed for the
 * exact cross-check test: per-u-lane (uops_u/mem_u) and
 * per-group-iteration (uops_g/mem_g) vector uops and mem-bytes of the
 * encode body at the reference shape. The test combines them with the
 * real candidate's (U, vl) and compares against real codegen. */
int matrix_search_cost_parts(const uint8_t *P, int ndata, int npar,
                             double *uops_u, double *mem_u,
                             double *uops_g, double *mem_g);

/* Deterministic bounded search for the lowest-cost MDS generator matrix
 * for (ndata, npar) inside the supported envelope (npar <= 3;
 * ndata <= 51 for npar >= 2; ndata <= 255 for npar == 1).
 * Fills out[] (npar*ndata bytes) and returns 0 on success, -1 on bad
 * args or OOM. Results are cached per (ndata, npar) for the process
 * lifetime (thread-safe); the search runs at most once per shape.
 * The output is deterministic and host-independent: fixed seeds, fixed
 * move order, strict-improvement acceptance with the historical
 * default as seed 0 (so ties keep the old bytes -- no gratuitous
 * parity-format churn). */
int matrix_search_default(int ndata, int npar, uint8_t *out /*npar*ndata*/);
