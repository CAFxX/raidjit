/* Scalar GF(2^8) reference for a plan: out[j][k] = XOR_i in[i][k]*mat[j,i].
 * Used to validate JIT candidate output before benchmarking (a wrong
 * candidate must never win on speed). No-op when n_out == 0. */
#pragma once
#include "plan.h"

void ref_run(const plan_t *pl, const uint8_t **in, uint8_t **out, size_t len);
