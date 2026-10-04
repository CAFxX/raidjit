#include "gf.h"
#include <stdlib.h>
#include <string.h>

uint8_t gf_mul(uint8_t a, uint8_t b)
{
    uint8_t p = 0;
    while (b) {
        if (b & 1)
            p ^= a;
        uint8_t hi = a & 0x80;
        a <<= 1;
        if (hi)
            a ^= 0x1B; /* 0x11B without the x^8 term */
        b >>= 1;
    }
    return p;
}

uint8_t gf_pow(uint8_t a, unsigned e)
{
    uint8_t r = 1;
    while (e) {
        if (e & 1)
            r = gf_mul(r, a);
        a = gf_mul(a, a);
        e >>= 1;
    }
    return r;
}

uint8_t gf_inv(uint8_t a)
{
    /* a^254, a != 0 */
    return gf_pow(a, 254);
}

void gf_build_logexp(uint8_t logt[256], uint8_t expt[512])
{
    /* 3 is primitive in GF(2^8)/0x11B (order 255); 2 has order 51. */
    uint8_t x = 1;
    for (int i = 0; i < 255; i++) {
        expt[i] = x;
        logt[x] = (uint8_t)i;
        x = gf_mul(x, 3);
    }
    for (int i = 255; i < 512; i++)
        expt[i] = expt[i - 255];
    logt[0] = 0;
}

void gf_build_nibble(uint8_t c, uint8_t lo[16], uint8_t hi[16])
{
    for (int t = 0; t < 16; t++) {
        lo[t] = gf_mul(c, (uint8_t)t);
        hi[t] = gf_mul(c, (uint8_t)(t << 4));
    }
}

uint64_t gf_affine_matrix(uint8_t c)
{
    /* VGF2P8AFFINEQB semantics (empirically validated): with data byte d in
     * vvvv and matrix qword M in r/m, each output byte is
     *   out = bitrev8(M x d_colvec),  (out_bit[7-k] = XOR_j d[j]*M[k][j])
     * so to compute out = gf_mul(c, d) we need M[k][j] = bit(7-k) of (c*2^j):
     * M byte k, bit j = bit (7-k) of gf_mul(c, 1<<j). */
    uint64_t m = 0;
    for (int k = 0; k < 8; k++) {
        uint64_t byte = 0;
        for (int j = 0; j < 8; j++)
            if ((gf_mul(c, (uint8_t)(1u << j)) >> (7 - k)) & 1)
                byte |= 1ULL << j;
        m |= byte << (8 * k);
    }
    return m;
}

void rs_default_matrix(int ndata, int npar, uint8_t *m)
{
    for (int j = 0; j < npar; j++)
        for (int i = 0; i < ndata; i++)
            m[j * ndata + i] = gf_pow(2, (unsigned)(j * (ndata - 1 - i)));
}

int mat_is_horner_vandermonde(const uint8_t *mat, int nr, int nc,
                              uint8_t *c_out)
{
    if (nc < 2)
        return 0;
    for (int j = 0; j < nr; j++) {
        if (mat[j * nc + nc - 1] != 1)
            return 0; /* c^0 must be 1 */
        uint8_t c = mat[j * nc + nc - 2];
        for (int i = 0; i < nc; i++) {
            if (mat[j * nc + i] != gf_pow(c, (unsigned)(nc - 1 - i)))
                return 0;
        }
        c_out[j] = c;
    }
    return 1;
}

int solve_recovery_rows(const uint8_t *M, int ndata, int npar,
                        const int *F, int nf, const int *S, int ns,
                        const int *rows, int nrows, uint8_t *R,
                        int *pivrows_out)
{
    if (nf > nrows)
        return -1;
    for (int r = 0; r < nrows; r++)
        if (rows[r] < 0 || rows[r] >= npar)
            return -1;
    int ndisk = ndata + npar;
    int ncols = nf + ns;
    /* augmented matrix: nrows rows, (nf + ns) cols.
     * col c<nf  : E[row][ F[c] ]   (unknowns: failed disks)
     * col nf+s  : E[row][ S[s] ]   (knowns: survivors)
     * E[row][i] = M[rows[row]][i] for data cols, 1 on the parity
     * diagonal (row rows[row], parity disk rows[row]-ndata). */
    uint8_t *a = calloc((size_t)nrows * ncols, 1);
    if (!a)
        return -1;
    for (int r = 0; r < nrows; r++) {
        for (int c = 0; c < nf; c++) {
            int d = F[c];
            a[r * ncols + c] =
                (d < ndata) ? M[(size_t)rows[r] * ndata + d]
                            : (uint8_t)(rows[r] == d - ndata);
        }
        for (int s = 0; s < ns; s++) {
            int d = S[s];
            a[r * ncols + nf + s] =
                (d < ndata) ? M[(size_t)rows[r] * ndata + d]
                            : (uint8_t)(rows[r] == d - ndata);
        }
    }
    /* Gauss-Jordan to reduced row echelon, pivoting on unknown columns.
     * rperm shadows the row permutation: matrix row r currently holds
     * the equation that started at original position rperm[r], so the
     * pivrows_out report below can map post-swap positions back to the
     * caller's rows[]. (Without this, pivrow[c] -- a post-swap
     * position -- would index the unpermuted rows[], reporting the row
     * that used to be at that position instead of the pivot
     * equation.) */
    int *pivrow = malloc((size_t)(nf > 0 ? nf : 1) * sizeof(int));
    int *rperm = malloc((size_t)(nrows > 0 ? nrows : 1) * sizeof(int));
    if (!pivrow || !rperm) {
        free(pivrow);
        free(rperm);
        free(a);
        return -1;
    }
    for (int c = 0; c < nf; c++)
        pivrow[c] = -1;
    for (int r = 0; r < nrows; r++)
        rperm[r] = r;
    int row = 0, ok = 1;
    for (int c = 0; c < nf && ok; c++) {
        int p = -1;
        for (int r = row; r < nrows; r++) {
            if (a[r * ncols + c]) {
                p = r;
                break;
            }
        }
        if (p < 0) {
            ok = 0; /* singular: failure set not recoverable from these
                     * rows */
            break;
        }
        if (p != row) {
            for (int k = 0; k < ncols; k++) {
                uint8_t t = a[p * ncols + k];
                a[p * ncols + k] = a[row * ncols + k];
                a[row * ncols + k] = t;
            }
            /* shadow the permutation alongside the matrix */
            int t = rperm[p];
            rperm[p] = rperm[row];
            rperm[row] = t;
        }
        uint8_t inv = gf_inv(a[row * ncols + c]);
        if (inv != 1)
            for (int k = c; k < ncols; k++)
                a[row * ncols + k] = gf_mul(a[row * ncols + k], inv);
        for (int r = 0; r < nrows; r++) {
            if (r == row)
                continue;
            uint8_t f = a[r * ncols + c];
            if (f)
                for (int k = c; k < ncols; k++)
                    a[r * ncols + k] ^= gf_mul(f, a[row * ncols + k]);
        }
        pivrow[c] = row;
        row++;
    }
    if (ok) {
        for (int c = 0; c < nf; c++) {
            memcpy(R + (size_t)c * ns, a + (size_t)pivrow[c] * ncols + nf,
                   (size_t)ns);
            if (pivrows_out)
                /* pivrow[c] is a post-swap position; rperm maps it
                 * back to the original position, rows[] to the
                 * caller's parity-row index. */
                pivrows_out[c] = rows[rperm[pivrow[c]]];
        }
    }
    free(pivrow);
    free(rperm);
    free(a);
    (void)ndisk;
    return ok ? 0 : -1;
}

int solve_recovery(const uint8_t *M, int ndata, int npar,
                   const int *F, int nf, const int *S, int ns,
                   uint8_t *R)
{
    /* all-rows special case of solve_recovery_rows */
    int *rows = malloc((size_t)(npar > 0 ? npar : 1) * sizeof(int));
    if (!rows)
        return -1;
    for (int r = 0; r < npar; r++)
        rows[r] = r;
    int rc = solve_recovery_rows(M, ndata, npar, F, nf, S, ns, rows,
                                 npar, R, NULL);
    free(rows);
    return rc;
}
