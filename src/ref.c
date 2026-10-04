/* Scalar GF(2^8) reference for a plan. */
#include "ref.h"
#include "gf.h"

void ref_run(const plan_t *pl, const uint8_t **in, uint8_t **out, size_t len)
{
    int n_in = pl->n_in, n_out = pl->n_out;
    if (n_out == 0 || !pl->mat)
        return;
    if (pl->kind == P_DELTA) {
        /* new_p[j] = old_p[j] ^ Σ_i ((old[i] ^ new[i]) * M[j][changed[i]]) */
        int nc = pl->nchanged;
        for (size_t k = 0; k < len; k++) {
            for (int j = 0; j < n_out; j++) {
                uint8_t acc = in[2 * nc + j][k];
                const uint8_t *row = pl->mat + (size_t)j * (size_t)nc;
                for (int i = 0; i < nc; i++) {
                    uint8_t c = row[i];
                    if (c)
                        acc ^=
                            gf_mul(c, (uint8_t)(in[i][k] ^ in[nc + i][k]));
                }
                out[j][k] = acc;
            }
        }
        return;
    }
    /* Loop order k,j,i matches the mathematical definition; this is a
     * correctness oracle, not a performance path. */
    for (size_t k = 0; k < len; k++) {
        for (int j = 0; j < n_out; j++) {
            uint8_t acc = 0;
            const uint8_t *row = pl->mat + (size_t)j * (size_t)n_in;
            for (int i = 0; i < n_in; i++) {
                uint8_t c = row[i];
                if (c)
                    acc ^= gf_mul(c, in[i][k]);
            }
            out[j][k] = acc;
        }
    }
}
