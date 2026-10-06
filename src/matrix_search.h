/* Plan-time search for low-bitweight MDS generator matrices over GF(2^8).
 *
 * rj_topo_rs()/rj_topo_raid6() do not hardcode a generator matrix.
 * Instead they call matrix_search_default(), which runs a deterministic,
 * bounded, host-independent search for the lowest-bitweight MDS
 * systematic generator matrix and caches the result per (ndata, npar).
 *
 * Design summary (full note: docs/matrix_search.md):
 *  - Cost = total bitmatrix weight: each GF(2^8) entry c expands to the
 *    8x8 binary matrix of the GF-linear map x -> c*x (column k = bits
 *    of c*2^k); the objective is the total number of 1s over all
 *    entries. Each 1 is one input bit feeding one parity bit, so the
 *    weight is (up to a per-output-bit constant) the XOR-gate count of
 *    a straightforward bit-level schedule -- the quantity a bit-level
 *    XOR scheduler (track A1) minimizes. Static and deterministic;
 *    never presented as a measurement.
 *  - MDS = exact minor enumeration: every npar x npar minor of the
 *    parity-check matrix H=[-P|I] is checked nonsingular over GF(2^8).
 *    For the supported envelope (npar <= 3, ndata <= 51) that is at most
 *    C(54,3) = 24804 tiny determinants -- a real proof per emitted
 *    matrix, not a randomized check.
 *  - Search = greedy single-entry hill climbing from several fixed
 *    seeds (historical default, Cauchy, fixed-seed random MDS),
 *    accepting only strict weight improvements. The objective is
 *    separable per entry, so the fixpoint has no single-entry strict
 *    improvement left (multi-entry coordinated moves are outside the
 *    neighborhood). The historical default is seed 0 and wins all
 *    ties, so the output changes only where the weight objective
 *    demands it.
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

/* Bitmatrix-weight objective of a generator matrix: each GF(2^8) entry c
 * expands to the 8x8 binary matrix of the GF-linear map x -> c*x
 * (column k = bits of c*2^k); *w_out receives the total number of 1s
 * over all npar*ndata entries. Returns 0 on success, -1 on bad args. */
int matrix_search_bitweight(const uint8_t *P, int ndata, int npar,
                            uint64_t *w_out);

/* Deterministic bounded search for the lowest-bitweight MDS generator
 * matrix for (ndata, npar) inside the supported envelope (npar <= 3;
 * ndata <= 51 for npar >= 2; ndata <= 255 for npar == 1).
 * Fills out[] (npar*ndata bytes) and returns 0 on success, -1 on bad
 * args or OOM. Results are cached per (ndata, npar) for the process
 * lifetime (thread-safe); the search runs at most once per shape.
 * The output is deterministic and host-independent: fixed seeds, fixed
 * move order, strict-improvement acceptance with the historical
 * default as seed 0 (so ties keep the old bytes).
 *
 * NOTE (matrix_search 2.0): the objective changed from the 1.0 static
 * uops model to the bitmatrix weight, so the emitted bytes deliberately
 * differ from the historical default wherever the weight objective
 * finds something lighter. There is no backward compatibility: the
 * parity format is whatever this search emits. */
int matrix_search_default(int ndata, int npar, uint8_t *out /*npar*ndata*/);
