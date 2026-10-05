/* Plan-time search for low-cost MDS generator matrices over GF(2^8).
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

/* ---------------- static encode cost (planner's own model) ---------------- */

/* Reference shape for the transcribed static model: the x86-64
 * nibble-backend accounting (VEX/EVEX encoding) at U=8, vl=64, with the
 * first 5 input pointers hoisted (PTR_NHOIST in src/x86_64/jit.c;
 * nothing hoisted when n_in < 2, mirroring ptr_hoist_count exactly).
 * The comparison the search needs is between matrices of the same
 * (ndata, npar): the U/vl-dependent terms (pointer reloads for i>=5)
 * are identical for all MDS matrices of one shape (no zero columns),
 * so the ranking is U- and vl-independent. */
#define REF_U 8
#define REF_VL 64
#define REF_NHOIST 5

/* Per-u-lane (uops_u/mem_u) and per-group-iteration (uops_g/mem_g) vector
 * uops and mem-bytes for the encode body, transcribed 1:1 from the
 * x86-64 nibble codegen in src/x86_64/jit.c:
 *  - c_ld/c_st/c_ldpool: 1 uop + vl mem; raw_ldpool (prologue): uncounted
 *  - c_vpxor/c_vmov/c_shuf/c_and/c_srlw: 1 uop (VEX: no extra moves)
 *  - body_inptr: 0 when hoisted, else 1 uop + 8 mem (once per group iter)
 *  - first-touch accumulators emit vmov, never xor-with-zero
 * tests/test_matrix_search.c pins this against real jit_build_version
 * output with exact equality, so backend accounting drift breaks the
 * test instead of silently diverging. */
static void body_cost(const plan_t *pl, double *uops_u, double *mem_u,
                      double *uops_g, double *mem_g)
{
    int n_in = pl->n_in, n_out = pl->n_out;
    double u = 0.0, m = 0.0, ug = 0.0, mg = 0.0;
    if (pl->kind == P_HORNER) {
        /* i == 0: acc = 0*c + t = t: load directly into acc[0] plus
         * (n_out-1) moves (not load-to-temp + n_out moves). */
        u += n_out;
        m += REF_VL;
        for (int i = 1; i < n_in; i++) {
            u += 1; /* c_ld t (shared across rows) */
            m += REF_VL;
            for (int j = 0; j < n_out; j++) {
                uint8_t c = pl->horner_c[j];
                /* c in {0,1}: pure move/xor; else nibble multiply of
                 * the accumulator: 3 (split) + 2 (shuf) + 2 (xor) */
                u += (c == 0 || c == 1) ? 1 : 7;
            }
        }
    } else if (pl->kind == P_MULACC) {
        int first_col[RJ_MAX_DISKS];
        for (int j = 0; j < n_out; j++) {
            first_col[j] = -1;
            for (int i = 0; i < n_in; i++)
                if (pl->mat[(size_t)j * (size_t)n_in + (size_t)i]) {
                    first_col[j] = i;
                    break;
                }
        }
        for (int i = 0; i < n_in; i++) {
            int any = 0, neednib = 0;
            for (int j = 0; j < n_out; j++) {
                uint8_t c = pl->mat[(size_t)j * (size_t)n_in + (size_t)i];
                if (c) {
                    any = 1;
                    if (c != 1)
                        neednib = 1;
                }
            }
            if (!any)
                continue;
            u += 1; /* c_ld t */
            m += REF_VL;
            if (neednib)
                u += 3; /* nibble split of t (shared across rows) */
            for (int j = 0; j < n_out; j++) {
                uint8_t c = pl->mat[(size_t)j * (size_t)n_in + (size_t)i];
                if (!c)
                    continue;
                if (c == 1) {
                    u += 1; /* vmov (first) or vpxor */
                    continue;
                }
                /* nibble multiply: 2 ldpool + 2 shuf, then 1 xor at
                 * the first contributing column, 2 after */
                u += 4;
                m += 2 * REF_VL;
                u += (i == first_col[j]) ? 1 : 2;
            }
        }
    } else if (pl->kind == P_XOR) {
        for (int i = 0; i < n_in; i++) {
            int nu = 0;
            for (int j = 0; j < n_out; j++)
                if (pl->mat[(size_t)j * (size_t)n_in + (size_t)i])
                    nu++;
            if (!nu)
                continue;
            /* nu == 1: c_ld acc (first) or c_vpxor_mem: 1 uop + vl mem;
             * nu > 1: c_ld t + one xor/move per user */
            u += 1;
            m += REF_VL;
            if (nu > 1)
                u += nu;
        }
    }
    /* output stores */
    u += n_out;
    m += (double)n_out * REF_VL;
    /* per-group-iteration pointer reloads for non-hoisted inputs.
     * Mirrors ptr_hoist_count() in src/x86_64/jit.c exactly, including
     * the n_in < 2 corner: with a single input nothing is hoisted, so
     * in[0]'s pointer is reloaded every group iteration (1 uop + 8 mem
     * bytes). Same-shape ranking is unaffected (a constant shift), but
     * the transcription is exact, so pin it with the cross-check. */
    int nhoist = (n_in < 2) ? 0 : (n_in < REF_NHOIST ? n_in : REF_NHOIST);
    for (int i = nhoist; i < n_in; i++) {
        ug += 1;
        mg += 8;
    }
    *uops_u = u;
    *mem_u = m;
    *uops_g = ug;
    *mem_g = mg;
}

int matrix_search_cost_parts(const uint8_t *P, int ndata, int npar,
                             double *uops_u, double *mem_u,
                             double *uops_g, double *mem_g)
{
    if (!P || !uops_u || !mem_u || !uops_g || !mem_g || ndata < 1 ||
        npar < 1 || ndata + npar > RJ_MAX_DISKS)
        return -1;
    /* The real planner builds the encode plan: strategy selection
     * (P_XOR/P_HORNER/P_MULACC) is the planner's actual code. */
    topo_int_t ti;
    ti.ndata = ndata;
    ti.npar = npar;
    ti.mat = malloc((size_t)npar * (size_t)ndata);
    if (!ti.mat)
        return -1;
    memcpy(ti.mat, P, (size_t)npar * (size_t)ndata);
    rj_op op = rj_op_encode();
    plan_t pl;
    char err[256];
    int rc = plan_build(&ti, &op, "msearch", &pl, err, sizeof(err));
    free(ti.mat);
    rj_op_free(&op);
    if (rc != 0)
        return -1;
    if (pl.kind == P_DELTA || pl.noop) {
        plan_free(&pl);
        return -1; /* unreachable for an encode op; fail loudly */
    }
    body_cost(&pl, uops_u, mem_u, uops_g, mem_g);
    plan_free(&pl);
    return 0;
}

int matrix_search_cost(const uint8_t *P, int ndata, int npar,
                       double *uops_per_byte, double *mem_per_byte)
{
    if (!uops_per_byte || !mem_per_byte)
        return -1;
    double uu, mu, ug, mg;
    if (matrix_search_cost_parts(P, ndata, npar, &uu, &mu, &ug, &mg) != 0)
        return -1;
    double grp = (double)REF_U * REF_VL;
    *uops_per_byte = (REF_U * uu + ug) / grp;
    *mem_per_byte = (REF_U * mu + mg) / grp;
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
 * x^y != 0). Not Vandermonde: exercises the P_MULACC basin. */
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

/* Greedy 1-ization: for each entry (fixed row-major order) try setting
 * it to 1; keep it iff MDS is preserved and the static cost strictly
 * drops. Repeats to a fixpoint (sweep cap for the budget).
 *
 * The move set is complete for strict improvements *within one planner
 * strategy*: under the static model an entry's cost depends only on its
 * class ({0}, {1}, other) -- 0 is MDS-infeasible (for npar == 1 the 1x1
 * minors of H are the entries themselves; for npar >= 2 the 2x2 minors
 * pairing each data column with an identity column force every entry
 * nonzero), and all "other" values cost identically -- so a strict
 * improvement at fixed strategy must 1-ize at least one entry.
 * NOT in the neighborhood: single-entry other->other' moves that change
 * the planner's strategy choice (e.g. completing a Vandermonde
 * structure to flip P_MULACC to P_HORNER). Latent only: any P_HORNER-
 * coded matrix costs at least the default (Horner floor = default, see
 * docs/matrix_search.md), so the gap cannot hide a strict improvement
 * over the fixed point. (Multi-entry coordinated moves are likewise
 * outside the neighborhood; the fixed-point conclusion rests on the
 * analytical bound plus the empirical tripwire in the design note.) */
#define GREEDY_MAX_SWEEPS 8
static void greedy_ones(uint8_t *m, int ndata, int npar, double *cost)
{
    for (int sweep = 0; sweep < GREEDY_MAX_SWEEPS; sweep++) {
        int improved = 0;
        for (int e = 0; e < npar * ndata; e++) {
            if (m[e] == 1)
                continue;
            uint8_t old = m[e];
            m[e] = 1;
            if (matrix_is_mds(m, ndata, npar, 1, NULL) == 1) {
                double u, mm;
                if (matrix_search_cost(m, ndata, npar, &u, &mm) == 0 &&
                    u < *cost - 1e-12) {
                    *cost = u;
                    improved = 1;
                    continue;
                }
            }
            m[e] = old;
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
     * nonzero; P_XOR needs every entry in {0,1}). */
    if (npar == 1) {
        memset(out, 1, (size_t)ndata);
        cache_insert(ndata, npar, out);
        return 0;
    }

    if (cache_lookup(ndata, npar, out))
        return 0;

    /* The search runs without holding the cache lock (it is
     * deterministic, so concurrent inserts compute identical bytes). */
    size_t msz = (size_t)npar * (size_t)ndata;
    uint8_t *best = malloc(msz);
    uint8_t *cand = malloc(msz);
    if (!best || !cand) {
        free(best);
        free(cand);
        return -1;
    }
    double best_u = 0.0, best_m = 0.0;

    /* Seed 0: historical default. It is MDS inside the envelope (the
     * envelope proof); verify anyway -- a failure here is a bug. */
    seed_default(ndata, npar, best);
    if (matrix_is_mds(best, ndata, npar, 1, NULL) != 1 ||
        matrix_search_cost(best, ndata, npar, &best_u, &best_m) != 0) {
        free(best);
        free(cand);
        return -1;
    }
    greedy_ones(best, ndata, npar, &best_u);
    /* refresh the mem tiebreak after the greedy */
    matrix_search_cost(best, ndata, npar, &best_u, &best_m);

    /* Seeds 1..2: genuinely different MDS matrices, same treatment.
     * Strict uops improvement, or uops-tie plus mem win; full ties keep
     * seed 0 (no format churn). */
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
        double u, mm;
        if (matrix_search_cost(cand, ndata, npar, &u, &mm) != 0)
            continue;
        greedy_ones(cand, ndata, npar, &u);
        if (matrix_search_cost(cand, ndata, npar, &u, &mm) != 0)
            continue;
        if (u < best_u - 1e-12 ||
            (u <= best_u + 1e-12 && mm < best_m - 1e-12)) {
            memcpy(best, cand, msz);
            best_u = u;
            best_m = mm;
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
