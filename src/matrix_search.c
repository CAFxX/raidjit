/* Plan-time search for low-bitweight MDS generator matrices over GF(2^8).
 * See src/matrix_search.h and docs/matrix_search.md. */
#include "matrix_search.h"
#include "plan.h"
#include "gf.h"

#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

/* ---------------- exact MDS check ---------------- */

/* Entry (row, col) of the parity-check matrix H = [-P | I_npar]:
 * data columns carry P's column, parity columns are unit vectors.
 * (The sign of -P is irrelevant for nonsingularity.) */
static uint8_t hcol(const uint8_t *P, int ndata, int col, int row)
{
    if (col < ndata)
        return P[(size_t)row * (size_t)ndata + (size_t)col];
    return (col - ndata == row) ? 1 : 0;
}

/* 1 iff the npar x npar minor of H on columns cols[] is nonsingular.
 * Closed forms for npar <= 3 (the supported envelope). */
static int minor_nonsingular(const uint8_t *P, int ndata, int npar,
                             const int *cols)
{
    if (npar == 1)
        return hcol(P, ndata, cols[0], 0) != 0;
    if (npar == 2) {
        uint8_t a = hcol(P, ndata, cols[0], 0);
        uint8_t b = hcol(P, ndata, cols[1], 0);
        uint8_t c = hcol(P, ndata, cols[0], 1);
        uint8_t d = hcol(P, ndata, cols[1], 1);
        /* det = ad-bc; in GF(2^8) subtraction is XOR */
        return gf_mul(a, d) != gf_mul(b, c);
    }
    /* npar == 3: Sarrus' rule; det != 0.
     * det = a(ei^fh) ^ b(di^fg) ^ c(dh^eg). */
    uint8_t m[3][3];
    for (int r = 0; r < 3; r++)
        for (int s = 0; s < 3; s++)
            m[r][s] = hcol(P, ndata, cols[s], r);
    uint8_t det =
        gf_mul(m[0][0], gf_mul(m[1][1], m[2][2]) ^ gf_mul(m[1][2], m[2][1])) ^
        gf_mul(m[0][1], gf_mul(m[1][0], m[2][2]) ^ gf_mul(m[1][2], m[2][0])) ^
        gf_mul(m[0][2], gf_mul(m[1][0], m[2][1]) ^ gf_mul(m[1][1], m[2][0]));
    return det != 0;
}

int matrix_is_mds(const uint8_t *P, int ndata, int npar, int early_exit,
                  long *nminors_out)
{
    if (!P || ndata < 1 || npar < 1 || npar > RS_MDS_MAX_NPAR)
        return -1;
    int ncols = ndata + npar;
    int cols[RS_MDS_MAX_NPAR];
    for (int s = 0; s < npar; s++)
        cols[s] = s;
    long n = 0;
    int ok = 1;
    for (;;) {
        n++;
        if (!minor_nonsingular(P, ndata, npar, cols)) {
            ok = 0;
            if (early_exit) {
                /* fast reject for the search: stop at the first
                 * singular minor. */
                if (nminors_out)
                    *nminors_out = n;
                return 0;
            }
            /* else: proof path -- keep enumerating so that a positive
             * verdict always implies full enumeration, and nminors_out
             * always reports the full minor count. */
        }
        /* lexicographic next combination */
        int i = npar - 1;
        while (i >= 0 && cols[i] == ncols - npar + i)
            i--;
        if (i < 0)
            break;
        cols[i]++;
        for (int s = i + 1; s < npar; s++)
            cols[s] = cols[s - 1] + 1;
    }
    if (nminors_out)
        *nminors_out = n;
    return ok;
}

/* Value (not just nonzero/zero) of the npar x npar minor on cols[].
 * Closed forms for npar <= 3. */
static uint8_t minor_det(const uint8_t *P, int ndata, int npar,
                         const int *cols)
{
    uint8_t m[3][3];
    for (int r = 0; r < npar; r++)
        for (int s = 0; s < npar; s++)
            m[r][s] = hcol(P, ndata, cols[s], r);
    if (npar == 1)
        return m[0][0];
    if (npar == 2)
        return gf_mul(m[0][0], m[1][1]) ^ gf_mul(m[0][1], m[1][0]);
    return gf_mul(m[0][0], gf_mul(m[1][1], m[2][2]) ^ gf_mul(m[1][2], m[2][1])) ^
           gf_mul(m[0][1], gf_mul(m[1][0], m[2][2]) ^ gf_mul(m[1][2], m[2][0])) ^
           gf_mul(m[0][2], gf_mul(m[1][0], m[2][1]) ^ gf_mul(m[1][1], m[2][0]));
}

/* Cofactor of minor entry (j0, t): determinant of the submatrix with row
 * j0 and column t removed (signs are irrelevant in characteristic 2). */
static uint8_t minor_cofactor(const uint8_t *P, int ndata, int npar,
                              const int *cols, int j0, int t)
{
    uint8_t s[2][2] = { { 0, 0 }, { 0, 0 } };
    int r2 = 0;
    for (int r = 0; r < npar; r++) {
        if (r == j0)
            continue;
        int s2 = 0;
        for (int c = 0; c < npar; c++) {
            if (c == t)
                continue;
            s[r2][s2++] = hcol(P, ndata, cols[c], r);
        }
        r2++;
    }
    if (npar == 1)
        return 1;
    if (npar == 2)
        return s[0][0];
    return gf_mul(s[0][0], s[1][1]) ^ gf_mul(s[0][1], s[1][0]);
}

/* Values v for P[j0][i0] that would make some npar x npar minor of H
 * singular: forbidden[v] = 1. Precondition: P is MDS.
 *
 * Only minors containing data column i0 can change status when
 * P[j0][i0] changes (every other minor is byte-identical, hence still
 * nonsingular), so enumerating those -- C(ndata+npar-1, npar-1) instead
 * of C(ndata+npar, npar) -- is exact. Each minor's determinant is
 * affine in the changed entry: det(v) = A*v ^ B with A the cofactor of
 * the entry's position, so a minor forbids at most one value
 * (v = B/A when A != 0; when A == 0 the determinant is v-independent
 * and -- P being MDS -- nonzero for all v). The complement of
 * forbidden[] is exactly the set of MDS-preserving values; in
 * particular the current value is never forbidden. */
static void forbidden_values(const uint8_t *P, int ndata, int npar,
                             int j0, int i0, uint8_t *forbidden)
{
    memset(forbidden, 0, 256);
    int ncols = ndata + npar;
    /* pool: every column except i0; choose npar-1 of them */
    int pool[256], np = 0;
    for (int c = 0; c < ncols; c++)
        if (c != i0)
            pool[np++] = c;
    int nm = npar - 1;
    int sub[RS_MDS_MAX_NPAR];
    for (int s = 0; s < nm; s++)
        sub[s] = s;
    int cols[RS_MDS_MAX_NPAR];
    uint8_t v_cur = P[(size_t)j0 * (size_t)ndata + (size_t)i0];
    for (;;) {
        /* merge pool[sub[]] with i0, keeping sorted order */
        int a = 0, o = 0, placed = 0;
        while (a < nm || !placed) {
            int pc = (a < nm) ? pool[sub[a]] : ncols; /* sentinel */
            if (!placed && i0 < pc) {
                cols[o++] = i0;
                placed = 1;
            } else {
                cols[o++] = pc;
                a++;
            }
        }
        int t = 0;
        while (cols[t] != i0)
            t++;
        uint8_t det = minor_det(P, ndata, npar, cols);
        /* det != 0 by the MDS precondition */
        uint8_t A = minor_cofactor(P, ndata, npar, cols, j0, t);
        if (A != 0)
            forbidden[gf_mul(det, gf_inv(A)) ^ v_cur] = 1;
        /* lexicographic next (npar-1)-combination of pool */
        int i = nm - 1;
        while (i >= 0 && sub[i] == np - nm + i)
            i--;
        if (i < 0)
            break;
        sub[i]++;
        for (int s = i + 1; s < nm; s++)
            sub[s] = sub[s - 1] + 1;
    }
}

/* ---------------- bitmatrix-weight objective ---------------- */

/* The search objective (matrix_search 2.0). Each GF(2^8) entry c expands
 * to the 8x8 binary matrix of the GF-linear map x -> c*x: column k holds
 * the bits of gf_mul(c, 1<<k), i.e. bit r of column k is 1 iff input bit
 * k feeds output bit r. The weight of c is the number of 1s in that
 * matrix; the matrix objective is the sum over all entries.
 *
 * Rationale: in a bit-level (gate-level) XOR schedule each 1 is one
 * input bit XORed into one parity bit, so the total weight is -- up to
 * one constant per nonzero output bit -- the XOR-gate count of the
 * straightforward schedule. Minimizing the weight minimizes the thing
 * a bit-level scheduler (track A1) has to work with; cross-coefficient
 * sharing can only do better from a lighter starting point. This is a
 * static, deterministic ranking criterion, never presented as a
 * measurement.
 *
 * Elementary facts used below (all pinned by tests/test_matrix_search.c):
 *  - bitw[0] = 0, bitw[1] = 8 (the identity), and bitw[c] > 8 for every
 *    other nonzero c: a nonzero GF-multiplication bitmatrix has at
 *    least one 1 per row (nonsingular), hence >= 8 ones, with equality
 *    only for a permutation matrix, i.e. c = 1 (any higher power of x
 *    wraps through the reduction polynomial and gains ones; any other
 *    c already has a multi-bit column 0).
 *  - the objective is separable per entry: changing entry e from a to b
 *    changes the total by exactly bitw[b] - bitw[a]. */

/* bitw[c]: number of 1s in the 8x8 binary matrix of x -> c*x. */
static uint8_t bitw[256];
/* cand_order: the 255 nonzero values sorted by (bitw, value). */
static uint8_t cand_order[255];
static atomic_flag tables_lock = ATOMIC_FLAG_INIT;
static int tables_done;

static void search_tables_ensure(void)
{
    while (atomic_flag_test_and_set(&tables_lock))
        ;
    if (!tables_done) {
        for (int c = 0; c < 256; c++) {
            unsigned w = 0;
            for (int k = 0; k < 8; k++)
                w += (unsigned)__builtin_popcount(
                    gf_mul((uint8_t)c, (uint8_t)(1u << k)));
            bitw[c] = (uint8_t)w;
        }
        for (int i = 0; i < 255; i++)
            cand_order[i] = (uint8_t)(i + 1);
        /* insertion sort by (bitw, value); 255 elements, once */
        for (int i = 1; i < 255; i++) {
            uint8_t v = cand_order[i];
            int j = i - 1;
            while (j >= 0 && (bitw[cand_order[j]] > bitw[v] ||
                              (bitw[cand_order[j]] == bitw[v] &&
                               cand_order[j] > v))) {
                cand_order[j + 1] = cand_order[j];
                j--;
            }
            cand_order[j + 1] = v;
        }
        tables_done = 1;
    }
    atomic_flag_clear(&tables_lock);
}

int matrix_search_bitweight(const uint8_t *P, int ndata, int npar,
                            uint64_t *w_out)
{
    if (!P || !w_out || ndata < 1 || npar < 1 || npar > RS_MDS_MAX_NPAR ||
        ndata + npar > RJ_MAX_DISKS)
        return -1;
    search_tables_ensure();
    uint64_t w = 0;
    for (int e = 0; e < npar * ndata; e++)
        w += bitw[P[e]];
    *w_out = w;
    return 0;
}

/* ---------------- the search ---------------- */

/* Seed 0: the historical default, M[j][i] = 2^(j*(ndata-1-i)). */
static void seed_default(int ndata, int npar, uint8_t *m)
{
    for (int j = 0; j < npar; j++)
        for (int i = 0; i < ndata; i++)
            m[(size_t)j * (size_t)ndata + (size_t)i] =
                gf_pow(2, (unsigned)(j * (ndata - 1 - i)));
}

/* Seed 1: Cauchy matrix P[j][i] = inv(x_i ^ y_j), x_i = i+1, y_j = 128+j.
 * MDS by proof for ndata+npar <= 255 (distinct nonzero x, distinct y,
 * x^y != 0). Not Vandermonde: exercises a different basin. */
static void seed_cauchy(int ndata, int npar, uint8_t *m)
{
    for (int j = 0; j < npar; j++)
        for (int i = 0; i < ndata; i++)
            m[(size_t)j * (size_t)ndata + (size_t)i] =
                gf_inv((uint8_t)((i + 1) ^ (128 + j)));
}

static uint64_t rng_next(uint64_t *s)
{
    uint64_t x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return *s = x;
}

/* Seed 2: random Vandermonde (c[0] = 1, other bases distinct nonzero),
 * rejection-sampled to MDS with a fixed RNG seed (deterministic).
 * Same cost class as the default; checks the search explores. */
static int seed_randvander(int ndata, int npar, uint8_t *m)
{
    uint64_t rng = 0x9e3779b97f4a7c15ULL;
    for (int tries = 0; tries < 1000; tries++) {
        uint8_t c[RS_MDS_MAX_NPAR];
        c[0] = 1;
        for (int j = 1; j < npar; j++) {
            for (;;) {
                uint8_t v = (uint8_t)(2 + rng_next(&rng) % 254);
                int dup = 0;
                for (int k = 0; k < j; k++)
                    if (c[k] == v)
                        dup = 1;
                if (!dup) {
                    c[j] = v;
                    break;
                }
            }
        }
        for (int j = 0; j < npar; j++)
            for (int i = 0; i < ndata; i++)
                m[(size_t)j * (size_t)ndata + (size_t)i] =
                    gf_pow(c[j], (unsigned)(ndata - 1 - i));
        if (matrix_is_mds(m, ndata, npar, 1, NULL) == 1)
            return 0;
    }
    return -1;
}

/* Greedy single-entry weight minimization to a fixpoint (sweep cap for
 * the budget). For each entry in fixed row-major order, compute the set
 * of MDS-forbidden values (forbidden_values) and jump straight to the
 * lightest allowed one.
 *
 * This is complete for strict improvements *by single-entry moves*,
 * not a heuristic: the objective is separable per entry, so a strict
 * single-entry improvement at e must replace m[e] with a lighter
 * allowed value, and the loop moves to the lightest allowed value.
 * A fixpoint therefore has no single-entry strict improvement left.
 * NOT in the neighborhood: multi-entry coordinated moves (changing two
 * entries at once can preserve MDS where neither change alone does).
 * `0` is always forbidden: for npar == 1 the 1x1 minors of H are the
 * entries themselves; for npar >= 2 the 2x2 minors pairing each data
 * column with an identity column force every entry nonzero. */
#define GREEDY_MAX_SWEEPS 8
static void greedy_bitweight(uint8_t *m, int ndata, int npar,
                             uint64_t *cost)
{
    uint8_t forbidden[256];
    for (int sweep = 0; sweep < GREEDY_MAX_SWEEPS; sweep++) {
        int improved = 0;
        for (int e = 0; e < npar * ndata; e++) {
            int j0 = e / ndata, i0 = e % ndata;
            uint8_t cur = m[e];
            if (cur == 1)
                continue; /* bitw[1] = 8 is the global minimum over
                           * nonzero values (pinned by the test), so no
                           * strictly lighter value exists */
            forbidden_values(m, ndata, npar, j0, i0, forbidden);
            /* lightest allowed value strictly lighter than cur
             * (cand_order is sorted by weight, so the scan stops at
             * the first candidate of weight >= bitw[cur]) */
            for (int k = 0; k < 255; k++) {
                uint8_t v = cand_order[k];
                if (bitw[v] >= bitw[cur])
                    break;
                if (!forbidden[v]) {
                    m[e] = v;
                    *cost -= (uint64_t)(bitw[cur] - bitw[v]);
                    improved = 1;
                    break;
                }
            }
        }
        if (!improved)
            break;
    }
}

/* ---------------- result cache ---------------- */

#define MC_CACHE_N 16
#define MC_CACHE_MAX (RS_MDS_MAX_NPAR * RJ_MAX_DISKS) /* 3*256 */
static uint8_t mc_mat[MC_CACHE_N][MC_CACHE_MAX];
static int mc_ndata[MC_CACHE_N], mc_npar[MC_CACHE_N], mc_valid[MC_CACHE_N];
static int mc_next;
static atomic_flag mc_lock = ATOMIC_FLAG_INIT;

static int cache_lookup(int ndata, int npar, uint8_t *out)
{
    int hit = -1;
    while (atomic_flag_test_and_set(&mc_lock))
        ;
    for (int i = 0; i < MC_CACHE_N; i++)
        if (mc_valid[i] && mc_ndata[i] == ndata && mc_npar[i] == npar)
            hit = i;
    if (hit >= 0)
        memcpy(out, mc_mat[hit], (size_t)npar * (size_t)ndata);
    atomic_flag_clear(&mc_lock);
    return hit >= 0;
}

static void cache_insert(int ndata, int npar, const uint8_t *m)
{
    while (atomic_flag_test_and_set(&mc_lock))
        ;
    int i = mc_next;
    mc_next = (mc_next + 1) % MC_CACHE_N;
    mc_ndata[i] = ndata;
    mc_npar[i] = npar;
    memcpy(mc_mat[i], m, (size_t)npar * (size_t)ndata);
    mc_valid[i] = 1;
    atomic_flag_clear(&mc_lock);
}

int matrix_search_default(int ndata, int npar, uint8_t *out)
{
    if (!out || ndata < 1 || npar < 1 || npar > RS_MDS_MAX_NPAR ||
        (npar >= 2 && ndata > RS_MDS_MAX_NDATA) ||
        ndata + npar > RJ_MAX_DISKS)
        return -1;

    /* npar == 1: all-ones is the unique optimum (MDS needs every entry
     * nonzero; 1 is the unique minimizer of the bitweight over nonzero
     * values, so any deviation strictly increases the weight). */
    if (npar == 1) {
        memset(out, 1, (size_t)ndata);
        cache_insert(ndata, npar, out);
        return 0;
    }

    if (cache_lookup(ndata, npar, out))
        return 0;

    /* The search runs without holding the cache lock (it is
     * deterministic, so concurrent inserts compute identical bytes). */
    search_tables_ensure();
    size_t msz = (size_t)npar * (size_t)ndata;
    uint8_t *best = malloc(msz);
    uint8_t *cand = malloc(msz);
    if (!best || !cand) {
        free(best);
        free(cand);
        return -1;
    }
    uint64_t best_w = 0;

    /* Seed 0: historical default. It is MDS inside the envelope (the
     * envelope proof); verify anyway -- a failure here is a bug. */
    seed_default(ndata, npar, best);
    if (matrix_is_mds(best, ndata, npar, 1, NULL) != 1 ||
        matrix_search_bitweight(best, ndata, npar, &best_w) != 0) {
        free(best);
        free(cand);
        return -1;
    }
    greedy_bitweight(best, ndata, npar, &best_w);
    /* refresh the weight after the greedy (it maintains the delta, but
     * a full recompute is bulletproof and cheap) */
    matrix_search_bitweight(best, ndata, npar, &best_w);

    /* Seeds 1..2: genuinely different MDS matrices, same treatment.
     * Strict weight improvement wins; full ties keep seed 0 (no
     * gratuitous parity-format churn beyond what the objective
     * demands). */
    for (int seed = 1; seed <= 2; seed++) {
        int ok;
        if (seed == 1) {
            seed_cauchy(ndata, npar, cand);
            ok = 1;
        } else {
            ok = seed_randvander(ndata, npar, cand) == 0;
        }
        if (!ok)
            continue;
        if (matrix_is_mds(cand, ndata, npar, 1, NULL) != 1)
            continue; /* checked seed failed: skip, never emit unproven */
        uint64_t w = 0;
        if (matrix_search_bitweight(cand, ndata, npar, &w) != 0)
            continue;
        greedy_bitweight(cand, ndata, npar, &w);
        if (matrix_search_bitweight(cand, ndata, npar, &w) != 0)
            continue;
        if (w < best_w) {
            memcpy(best, cand, msz);
            best_w = w;
        }
    }
    free(cand);

    /* The proof: full minor enumeration (no early exit) on the winner. */
    long nminors = 0;
    if (matrix_is_mds(best, ndata, npar, 0, &nminors) != 1) {
        /* Cannot happen: every accepted matrix passed the check.
         * Fall back to the verified seed rather than emitting
         * an unproven matrix. */
        seed_default(ndata, npar, best);
    }
    (void)nminors;

    memcpy(out, best, msz);
    cache_insert(ndata, npar, best);
    free(best);
    return 0;
}
