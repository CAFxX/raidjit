/* GF(2^8) arithmetic (poly x^8+x^4+x^3+x+1 = 0x11B), Reed-Solomon helpers,
 * and the general erasure-recovery solver used to lower "recover" ops into
 * plain linear combinations of the surviving disks. */
#pragma once
#include <stdint.h>

uint8_t gf_mul(uint8_t a, uint8_t b);
uint8_t gf_pow(uint8_t a, unsigned e);
uint8_t gf_inv(uint8_t a);              /* a != 0 */

/* log/exp tables for primitive element g=3 (exp has 512 entries to skip
 * mod; 2 has order 51 in GF(2^8)/0x11B and is not primitive) */
void gf_build_logexp(uint8_t logt[256], uint8_t expt[512]);

/* nibble tables for multiply-by-c via pshufb:
 *   out = lo[x & 0xF] ^ hi[x >> 4]  ==  gf_mul(c, x)            */
void gf_build_nibble(uint8_t c, uint8_t lo[16], uint8_t hi[16]);

/* 8x8 bit matrix for VGF2P8AFFINEQB implementing multiply-by-c:
 * matrix byte k, bit j = bit (7-k) of gf_mul(c, x^j); imm8 b = 0.
 * (VGF2P8AFFINEQB with data in vvvv, matrix in r/m computes
 * out = bitrev8(matrix x data_colvec).) */
uint64_t gf_affine_matrix(uint8_t c);

/* MDS envelope of the default RS matrix (g=2). Column bases are
 * 2^(ndata-1-i); since ord(2)=51 in GF(2^8)/0x11B, ndata<=51 keeps them
 * distinct. Recovery solves square systems over arbitrary subsets of
 * parity rows; the 2x2 minor on row-exponents {0,3} vanishes when two
 * bases differ by a factor of order 3 (2^17), so npar<=3 is required.
 * Verified numerically: inside this envelope every <=npar erasure set is
 * recoverable; outside it, some are singular. rj_topo_raid6/rj_topo_rs
 * return NULL beyond the envelope; wider MDS topologies (e.g. Cauchy)
 * are available via rj_topo_custom. */
#define RS_MDS_MAX_NDATA 51
#define RS_MDS_MAX_NPAR  3

/* Default RS matrix in "Horner order": M[j][i] = g^(j*(ndata-1-i)).
 * Row 0 is all ones (XOR parity). */
void rs_default_matrix(int ndata, int npar, uint8_t *m /*npar*ndata*/);

/* Detect whether mat (nr x nc) is Vandermonde in Horner order, i.e. each
 * row j equals [ c_j^(nc-1), ..., c_j^1, c_j^0 ] for some c_j.
 * Returns 1 and fills c_out[j] on success. */
int mat_is_horner_vandermonde(const uint8_t *mat, int nr, int nc,
                              uint8_t *c_out);

/* Solve for the recovery coefficients of a "recover" op.
 * Disks are numbered 0..ndata-1 (data), ndata..ndata+npar-1 (parity).
 * F[0..nf-1] failed (sorted), S[0..ns-1] survivors (sorted).
 * Finds R (nf x ns) with disk[F[r]] = XOR_s R[r][s] * disk[S[s]].
 * Returns 0 on success, -1 if the failure set is not recoverable. */
int solve_recovery(const uint8_t *M, int ndata, int npar,
                   const int *F, int nf, const int *S, int ns,
                   uint8_t *R /*nf*ns*/);

/* Row-restricted solve: use only the parity rows listed in
 * rows[0..nrows-1] (each 0 <= rows[r] < npar) as equations. nrows >= nf
 * is required. A failed parity disk j forces rows[r] == j for some r
 * (its unknown appears only in its own equation); otherwise the caller
 * chooses. Returns 0 on success, -1 when the failure set is not
 * recoverable from exactly these rows (singular subsystem) or the row
 * list is invalid. pivrows_out (nullable, [nf]) receives, for each
 * unknown in F order, the parity-row index (a value from rows[]) whose
 * equation was pivoted for that unknown during elimination -- the rows
 * R's rows are read from. (The survivor-set track identifies its
 * no-regression anchor by solution-matrix equality, not by
 * pivrows_out; see survivor.c.) solve_recovery() is the all-rows
 * special case.
 * (Track A5: the survivor-set enumeration solves each candidate row
 * subset separately and benchmarks the resulting plans.) */
int solve_recovery_rows(const uint8_t *M, int ndata, int npar,
                        const int *F, int nf, const int *S, int ns,
                        const int *rows, int nrows, uint8_t *R /*nf*ns*/,
                        int *pivrows_out /*nullable [nf]*/);
