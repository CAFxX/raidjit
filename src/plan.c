#include "plan.h"
#include "gf.h"
#include "prefetch_calib.h"
#include "xorsched.h"
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

const char *isa_name(isa_t isa)
{
    switch (isa) {
    case ISA_SSE: return "sse";
    case ISA_AVX2: return "avx2";
    case ISA_AVX512: return "avx512";
    case ISA_NEON: return "neon";
    case ISA_SVE: return "sve";
    case ISA_RVV: return "rvv";
    case ISA_PEXT: return "pext";
    }
    return "?";
}

int isa_nregs(isa_t isa)
{
    switch (isa) {
    case ISA_AVX512: return 32;
    case ISA_RVV: return 32;
    case ISA_PEXT: return 32;
    /* [aarch64-remediation #3, 2026-10-06] SVE uses 24 vector registers:
     * z0-z7 plus z16-z31 (z8-z15 are callee-saved per AAPCS64 and never
     * touched; src/aarch64/jit.c V() maps logical r<16 to z(16+r) and
     * r in [16,24) to z(r-16)). The backend asserts vmax < 24, in
     * lockstep with this bound. NEON stays at 16 (v0-v15). */
    case ISA_SVE: return 24;
    /* NEON backend uses 16 logical vector registers (v0-v15). */
    default: return 16;
    }
}

/* Count Horner rows whose coefficient needs a table register (outside
 * {0,1}); those rows are pure XOR and need no table. */
static int horner_ntriv(const plan_t *pl)
{
    int n = 0;
    for (int j = 0; j < pl->n_out; j++) {
        uint8_t c = pl->horner_c[j];
        if (c != 0 && c != 1)
            n++;
    }
    return n;
}

/* Single source of truth for "some coefficient outside {0,1}". P_MULACC
 * scans the full matrix, P_DELTA its n_out x nchanged delta columns
 * (not n_out x n_in), P_HORNER the row coefficients. */
int plan_any_nontrivial(const plan_t *pl)
{
    if (pl->kind == P_HORNER) {
        if (!pl->horner_c)
            return 1; /* unproven: stay conservative */
        for (int j = 0; j < pl->n_out; j++) {
            uint8_t c = pl->horner_c[j];
            if (c != 0 && c != 1)
                return 1;
        }
        return 0;
    }
    if (pl->kind == P_MULACC || pl->kind == P_DELTA) {
        if (!pl->mat)
            return 1; /* unproven: stay conservative */
        int ncols = pl->kind == P_DELTA ? pl->nchanged : pl->n_in;
        for (int k = 0; k < ncols * pl->n_out; k++)
            if (pl->mat[k] != 0 && pl->mat[k] != 1)
                return 1;
        return 0;
    }
    return 0; /* P_XOR: all-{0,1} by construction */
}

int plan_need_regs(isa_t isa, mul_t mul, const plan_t *pl, int U, int remat)
{
    int n_out = pl->n_out;
    if (isa == ISA_PEXT) {
        /* No PEXT backend exists (RISC-V P-extension work is deferred),
         * so no pressure model is proven. Fail loudly instead of
         * silently falling through to the RVV formula below. */
        assert(!"plan_need_regs: no PEXT backend");
        return INT_MAX;
    }
    if (pl->kind == P_XOR)
        /* accumulators + one load temp (cg->t / vd0) */
        return n_out * U + 1;
    int x86 = (isa == ISA_SSE || isa == ISA_AVX2 || isa == ISA_AVX512);
    int a64 = (isa == ISA_NEON || isa == ISA_SVE);
    int gfni = (mul == MUL_GFNI);
    /* Coefficient-aware allocation: when every coefficient is in {0,1}
     * the nibble path -- mask, tables, split temps -- is dead and not
     * allocated. Only P_DELTA can reach the all-trivial case from the
     * planner (parity_update forces the kind even for an all-{0,1}
     * delta matrix, e.g. RAID-5): P_MULACC provably has a nontrivial
     * coefficient (else plan_build picks P_XOR) and P_HORNER too (an
     * all-{0,1} Vandermonde is all-{0,1}, caught by the P_XOR check
     * first). The formulas still key off plan_any_nontrivial so the
     * model and the emitters share one source of truth. */
    int nontriv = plan_any_nontrivial(pl);
    if (x86) {
        /* src/x86_64/jit.c: t holds the input; nibble adds tlo,thi,r1,r2,tbl,mask
         * (t..t+6); PDELTA on AVX512 adds tbl2 for NIBHI pinning (t..t+7,
         * item 7). GFNI adds t2,tbl (t..t+2). HORNER adds 2 (nibble) or 1
         * (GFNI) table regs per nontrivial row, unless remat (tables
         * reloaded from pool into tbl; no hreg). */
        if (pl->kind == P_HORNER) {
            int ntriv = horner_ntriv(pl);
            if (remat)
                return n_out * U + (gfni ? 3 : 7);
            return n_out * U + (gfni ? 2 : 6) + (gfni ? ntriv : 2 * ntriv);
        }
        /* P_MULACC / P_DELTA (DELTA reuses r1/t2 as its delta temp) */
        if (gfni) {
            /* t2/tbl are dead under all-trivial coefficients: the affine
             * table and temp are only used for c outside {0,1}, and the
             * delta temp dt (= t2) is only read on the SSE path, which
             * GFNI never takes. */
            return n_out * U + (nontriv ? 3 : 1);
        }
        if (nontriv) {
            /* PDELTA on AVX512 pins the NIBHI table in tbl2 (item 7):
             * one more vector register than the ping-pong form. */
            if (pl->kind == P_DELTA && isa == ISA_AVX512)
                return n_out * U + 8;
            return n_out * U + 7;
        }
        /* All-trivial: the nibble split never runs, so tlo/thi/r2/tbl/
         * mask are dead. r1 survives on SSE as the delta temp (P_DELTA's
         * two-operand old^new needs it); on AVX the delta folds into a
         * memory XOR and only t remains. (P_MULACC all-trivial is
         * unreachable from the planner; its r1 would be dead too.) */
        return n_out * U +
               ((pl->kind == P_DELTA && isa == ISA_SSE) ? 2 : 1);
    }
    if (a64) {
        /* [aarch64-remediation #4, 2026-10-06] src/aarch64/jit.c:
         * t,(tlo,thi,r1,r2,tbl_lo,tbl_hi,mask). The MULACC table hoisting
         * keeps one row's NIBLO/NIBHI tables live across the lane loop,
         * so two table regs are allocated (was one). The backend asserts
         * vmax+1 == this count; keep in lockstep. HORNER compacts its 2
         * table regs to nontrivial rows only, unless remat (tables
         * reloaded into tbl_lo; no hreg). */
        if (pl->kind == P_HORNER) {
            if (remat)
                return n_out * U + 6;
            return n_out * U + 5 + 2 * horner_ntriv(pl);
        }
        /* P_MULACC / P_DELTA: all-trivial kills tlo/thi/r2/tbl_lo/tbl_hi/
         * mask, but r1 stays live as the P_DELTA delta temp on both NEON
         * and SVE (old^new is formed into r1 for every column
         * unconditionally). (P_MULACC all-trivial is unreachable; its r1
         * would be dead.) */
        return n_out * U + (nontriv ? 8 : (pl->kind == P_DELTA ? 2 : 1));
    }
    /* RVV: the backend stages U input chunks in vd0..vd0+U-1, then the
     * usual temps (tlo,thi,r1,r2,tbl_lo,tbl_hi,mask for MULACC/DELTA;
     * tlo,thi,r1,r2,mask for HORNER) plus 2 table regs per nontrivial
     * Horner row, unless remat (tables reloaded; no hreg). */
    if (mul == MUL_ZVBC) {
        /* Zvbc carryless path: r1/r2/r3 temps + kvec (0x1B broadcast),
         * plus one cvec coefficient broadcast per nontrivial Horner row
         * (MULACC/DELTA keep a single cvec shared across the columns),
         * plus tlo/thi delta-formation temps for P_DELTA. No nibble
         * tables, no mask. remat is meaningless here (no tables to
         * rematerialize); the planner never enumerates it. */
        if (pl->kind == P_HORNER)
            return (n_out + 1) * U + 4 + horner_ntriv(pl);
        /* P_MULACC / P_DELTA: all-trivial kills r1/r2/r3/cvec/kvec;
         * tlo/thi stay live as the P_DELTA delta temps (the body borrows
         * them to load old/new for every column unconditionally).
         * (P_MULACC all-trivial is unreachable; its tlo/thi would be
         * dead.) */
        if (nontriv)
            return (n_out + 1) * U + (pl->kind == P_DELTA ? 7 : 5);
        if (pl->kind == P_DELTA)
            return (n_out + 1) * U + 2;
        return (n_out + 1) * U;
    }
    if (pl->kind == P_HORNER) {
        if (remat)
            return (n_out + 1) * U + 6;
        return (n_out + 1) * U + 5 + 2 * horner_ntriv(pl);
    }
    /* P_MULACC / P_DELTA: all-trivial kills r1/r2/tbl_lo/tbl_hi/mask;
     * tlo/thi stay live as the P_DELTA delta temps (the body borrows
     * them to load old/new for every column unconditionally).
     * (P_MULACC all-trivial is unreachable; its tlo/thi would be dead.) */
    if (nontriv)
        return (n_out + 1) * U + 7;
    if (pl->kind == P_DELTA)
        return (n_out + 1) * U + 2;
    /* P_MULACC all-trivial: only the U data regs survive. Unlike x86's
     * separate t temp, RVV stages inputs into vd0..vd0+U-1, already
     * counted in (n_out+1)*U, so no extra register is needed. */
    return (n_out + 1) * U;
}

int isa_vl(isa_t isa)
{
    switch (isa) {
    case ISA_SSE: return 16;
    case ISA_AVX2: return 32;
    case ISA_AVX512: return 64;
    case ISA_NEON: return 16;
    case ISA_SVE: return 0;  /* VL-agnostic, runtime-determined */
    case ISA_RVV: return 0;  /* VL-agnostic, runtime-determined */
    case ISA_PEXT: return 8; /* 64-bit subwords, fixed */
    }
    return 16;
}

void plan_free(plan_t *pl)
{
    xorsched_free(pl->xor_sched);
    xorbit_free(pl->bit_sched);
    free(pl->mat);
    free(pl->horner_c);
    memset(pl, 0, sizeof(*pl));
}

static int cmp_int(const void *a, const void *b)
{
    return *(const int *)a - *(const int *)b;
}

/* Validate + sort/dedupe a recover op's failed-disk list. On success
 * returns the deduped count (>= 1) with *F_out the sorted list (caller
 * frees); on error returns -1 with err set. */
int plan_recover_failed_list(const rj_op *op, int nd, int np, int **F_out,
                             char *err, size_t errlen)
{
    int nf = op->nfailed;
    if (op->oom) {
        snprintf(err, errlen, "recover: out of memory (failed-disk list)");
        return -1;
    }
    if (nf < 1) {
        snprintf(err, errlen, "recover: empty failed-disk list");
        return -1;
    }
    if (!op->failed) {
        snprintf(err, errlen, "recover: NULL failed-disk list");
        return -1;
    }
    int *F = malloc((size_t)nf * sizeof(int));
    if (!F) {
        snprintf(err, errlen, "out of memory");
        return -1;
    }
    memcpy(F, op->failed, (size_t)nf * sizeof(int));
    qsort(F, (size_t)nf, sizeof(int), cmp_int);
    int w = 0;
    for (int i = 0; i < nf; i++)
        if (i == 0 || F[i] != F[i - 1])
            F[w++] = F[i];
    nf = w;
    for (int i = 0; i < nf; i++) {
        if (F[i] < 0 || F[i] >= nd + np) {
            snprintf(err, errlen, "recover: disk index %d out of range",
                     F[i]);
            free(F);
            return -1;
        }
    }
    if (np == 0) {
        snprintf(err, errlen, "recover: topology has no redundancy (npar=0)");
        free(F);
        return -1;
    }
    if (nf > np) {
        snprintf(err, errlen, "recover: %d failures exceed parity count %d",
                 nf, np);
        free(F);
        return -1;
    }
    *F_out = F;
    return nf;
}

/* Strategy selection shared by plan_build and plan_build_recover_rows:
 * P_XOR when every coefficient is 0/1, else P_HORNER when the matrix is
 * Vandermonde in Horner order, else P_MULACC. On OOM the partial plan is
 * freed (matching plan_build's old goto-oom contract: callers do not
 * plan_free after a failed build). */
static int plan_select_strategy(plan_t *pl, char *err, size_t errlen)
{
    int n_in = pl->n_in, n_out = pl->n_out;
    int all01 = 1;
    for (int k = 0; k < n_in * n_out; k++)
        if (pl->mat[k] != 0 && pl->mat[k] != 1)
            all01 = 0;
    if (all01) {
        pl->kind = P_XOR;
        /* Cross-row CSE program for the xorsched candidate twin (NULL
         * when there is nothing to share). Built once at plan time. */
        pl->xor_sched = xorsched_build(pl);
        return 0;
    }
    pl->horner_c = malloc((size_t)n_out);
    if (!pl->horner_c) {
        plan_free(pl);
        snprintf(err, errlen, "out of memory");
        return -1;
    }
    if (mat_is_horner_vandermonde(pl->mat, n_out, n_in, pl->horner_c)) {
        pl->kind = P_HORNER;
        /* Bit-level CSE program for the xorbit candidate twin (NULL
         * when there is nothing to share). Built once at plan time. */
        pl->bit_sched = xorbit_build(pl);
        return 0;
    }
    free(pl->horner_c);
    pl->horner_c = NULL;
    pl->kind = P_MULACC;
    /* Bit-level CSE program for the xorbit candidate twin (NULL when
     * there is nothing to share). Built once at plan time. */
    pl->bit_sched = xorbit_build(pl);
    return 0;
}

/* Small-erasure closed forms in recovery (track A4, phase 1: planner
 * only -- no emitter changes; the plans below are ordinary plan_t the
 * existing backends already execute).
 *
 * Single-disk failure is the dominant recovery case. Instead of the
 * general Gauss-Jordan solve (solve_recovery_rows), e=1/e=2 use the
 * closed form of the erasure equations, evaluated once at plan time:
 *
 * e=1: one failure f, one equation row r with E[r][f] = c_f != 0.
 *   f = c_f^-1 * (XOR over survivors s of E[r][s] * S_s):
 *   one XOR-reduction over the survivors fused with a SINGLE constant
 *   multiply by c_f^-1. R[s] = E[r][s] * c_f^-1 is filled directly in
 *   O(ns); c_f^-1 is a plan-time constant baked into R -- no runtime
 *   division (Carlo's invariant: no runtime computation doable at plan
 *   time).
 *
 * e=2: failures f1 < f2, equation rows r1, r2. The 2x2 system
 *     [a b] [x1]   [p]      a = E[r1][f1], b = E[r1][f2],
 *     [c d] [x2] = [q]      c = E[r2][f1], d = E[r2][f2],
 *   with syndrome terms p = XOR_s E[r1][s]*S_s and
 *   q = XOR_s E[r2][s]*S_s, inverts in closed form via
 *   adjugate/determinant: det = a*d ^ b*c,
 *     x1 = (d*p ^ b*q) * det^-1,   x2 = (c*p ^ a*q) * det^-1.
 *   R[0][s] = (d*E[r1][s] ^ b*E[r2][s]) * det^-1 and
 *   R[1][s] = (c*E[r1][s] ^ a*E[r2][s]) * det^-1 share their
 *   subexpressions at plan time: E[r1][s], E[r2][s] are read once per s
 *   and feed both output rows, det^-1 is computed once (shared det
 *   reciprocal). Phase 2 will evaluate the shared syndrome terms p, q
 *   once at runtime; phase 1 bakes the combined coefficients into the
 *   plan the existing emitters execute.
 *
 * The equation rows replicate solve_recovery_rows' pivot choice (first
 * suitable row / first suitable pair in rows[] order), so an applied
 * closed form is bit-identical to the general solve -- the phase-1 win
 * is JIT-time (O(ns) direct fill vs O(nrows*(nf+ns)) elimination) plus
 * the structural hook for phase 2. Triggers only for nf == 1/2 with an
 * invertible pivot (pair); nf >= 3 or a singular pivot falls through to
 * solve_recovery_rows, which stays as the general fallback.
 *
 * Called from plan_build_recover_rows, so closed-form plans flow
 * through the normal plan -> prune -> select pipeline (A5 survivor
 * subsets included). Test hook RJ_RECOVER_NO_CLOSEDFORM (non-empty,
 * != "0") forces the general path for differential testing. */
static uint8_t recover_eqn_coef(const topo_int_t *topo, int row, int d)
{
    /* One entry of the recovery equation system: M[row][d] for a data
     * disk, 1 on the parity diagonal (row == d-ndata), 0 elsewhere.
     * Matches solve_recovery_rows' E definition. */
    int nd = topo->ndata;
    if (d < nd)
        return topo->mat[(size_t)row * (size_t)nd + (size_t)d];
    return (uint8_t)(row == d - nd);
}

/* Closed-form recovery rows, see the A4 comment above. Fills R
 * (nf x ns, row-major) and returns 1 when the closed form applied;
 * returns 0 (R untouched) when it does not apply (nf outside 1..2,
 * bad row indices, singular pivot) -- the caller falls back to
 * solve_recovery_rows. */
int plan_recover_closed_form(const topo_int_t *topo, const int *F, int nf,
                             const int *S, int ns, const int *rows, int nrows,
                             uint8_t *R)
{
    int np = topo->npar;
    if (nf < 1 || nf > 2 || nrows < nf || ns < 0)
        return 0;
    for (int r = 0; r < nrows; r++)
        if (rows[r] < 0 || rows[r] >= np)
            return 0;
    if (nf == 1) {
        /* Pivot row: first rows[] entry with a nonzero coefficient on
         * the failed disk -- solve_recovery_rows' pivot choice, so R
         * below is exactly its normalized pivot row. */
        int pr = -1;
        uint8_t cf = 0;
        for (int r = 0; r < nrows; r++) {
            uint8_t c = recover_eqn_coef(topo, rows[r], F[0]);
            if (c) {
                pr = rows[r];
                cf = c;
                break;
            }
        }
        if (pr < 0)
            return 0; /* singular: no equation constrains f */
        uint8_t kinv = gf_inv(cf); /* plan-time constant */
        for (int s = 0; s < ns; s++)
            R[s] = gf_mul(recover_eqn_coef(topo, pr, S[s]), kinv);
        return 1;
    }
    /* nf == 2: replicate the solver's pivot pair. p0 = first row with
     * E[r][F[0]] != 0; p1 = first remaining row whose F[1] coefficient
     * survives eliminating F[0], i.e. E[r][F[1]] ^ E[r][F[0]]*k != 0
     * with k = E[p0][F[1]]/E[p0][F[0]] (the normalized pivot row's F[1]
     * entry). A found p1 guarantees det != 0: the reduced coefficient
     * is det/E[p0][F[0]]. */
    int p0 = -1, p1 = -1;
    uint8_t a = 0, b = 0, c2 = 0, d2 = 0;
    for (int r = 0; r < nrows; r++) {
        uint8_t e = recover_eqn_coef(topo, rows[r], F[0]);
        if (e) {
            p0 = rows[r];
            a = e;
            b = recover_eqn_coef(topo, p0, F[1]);
            break;
        }
    }
    if (p0 < 0)
        return 0;
    uint8_t k = gf_mul(b, gf_inv(a));
    for (int r = 0; r < nrows; r++) {
        int rr = rows[r];
        if (rr == p0)
            continue;
        uint8_t e1 = recover_eqn_coef(topo, rr, F[0]);
        uint8_t e2 = recover_eqn_coef(topo, rr, F[1]);
        if ((e2 ^ gf_mul(e1, k)) != 0) {
            p1 = rr;
            c2 = e1;
            d2 = e2;
            break;
        }
    }
    if (p1 < 0)
        return 0; /* singular pair: fall back (solver fails loudly) */
    uint8_t det = gf_mul(a, d2) ^ gf_mul(b, c2);
    if (det == 0)
        return 0; /* unreachable when p1 exists; defensive */
    uint8_t dinv = gf_inv(det); /* shared det reciprocal, plan time */
    for (int s = 0; s < ns; s++) {
        /* Shared syndrome coefficients: e1/e2 are read once per s and
         * feed both output rows (the plan-time image of the shared
         * p, q evaluation phase 2 will do at runtime). */
        uint8_t e1 = recover_eqn_coef(topo, p0, S[s]);
        uint8_t e2 = recover_eqn_coef(topo, p1, S[s]);
        R[s] = gf_mul(gf_mul(d2, e1) ^ gf_mul(b, e2), dinv);
        R[(size_t)ns + (size_t)s] =
            gf_mul(gf_mul(c2, e1) ^ gf_mul(a, e2), dinv);
    }
    return 1;
}

/* Build a recovery plan from an explicit parity-row equation set
 * (track A5). F[0..nf-1] is the validated sorted/deduped failed list
 * (1 <= nf <= npar); rows[0..nrows-1] are the parity rows used as
 * equations (nrows >= nf). The plan name records the row set, exposing
 * the survivor-set choice as a planner dimension. */
int plan_build_recover_rows(const topo_int_t *topo, const int *F, int nf,
                            const int *rows, int nrows, const char *opname,
                            plan_t *pl, char *err, size_t errlen)
{
    int nd = topo->ndata, np = topo->npar;
    memset(pl, 0, sizeof(*pl));
    if (nf < 1 || nf > np || nrows < nf) {
        snprintf(err, errlen, "recover: bad subset (%d failed, %d rows)",
                 nf, nrows);
        return -1;
    }
    /* survivors, increasing */
    int ns = nd + np - nf;
    int *S = malloc((size_t)ns * sizeof(int));
    if (!S) {
        snprintf(err, errlen, "out of memory");
        return -1;
    }
    int fi = 0, si = 0;
    for (int d = 0; d < nd + np; d++) {
        if (fi < nf && F[fi] == d)
            fi++;
        else
            S[si++] = d;
    }
    pl->mat = malloc((size_t)nf * ns);
    if (!pl->mat) {
        free(S);
        snprintf(err, errlen, "out of memory");
        return -1;
    }
    /* A4 closed forms first (planner-only phase 1); the general solve
     * stays as the fallback for nf >= 3 and singular pivots.
     * RJ_RECOVER_NO_CLOSEDFORM (non-empty, != "0") forces the general
     * path -- differential test hook. */
    int use_cf = 1;
    const char *no_cf = getenv("RJ_RECOVER_NO_CLOSEDFORM");
    if (no_cf && no_cf[0] != '\0' && strcmp(no_cf, "0") != 0)
        use_cf = 0;
    int closed = use_cf ? plan_recover_closed_form(topo, F, nf, S, ns, rows,
                                                   nrows, pl->mat)
                        : 0;
    if (!closed &&
        solve_recovery_rows(topo->mat, nd, np, F, nf, S, ns, rows, nrows,
                            pl->mat, NULL) != 0) {        snprintf(err, errlen,
                 "recover: failure set is not recoverable from these "
                 "parity rows (singular system)");
        free(S);
        free(pl->mat);
        pl->mat = NULL;
        return -1;
    }
    pl->n_in = ns;
    pl->n_out = nf;
    /* human-readable name: failed disks + the equation rows used */
    char fb[40] = "";
    size_t o = 0;
    for (int i = 0; i < nf && o < sizeof(fb) - 6; i++)
        o += (size_t)snprintf(fb + o, sizeof(fb) - o, "%s%d", i ? "+" : "",
                              F[i] < nd ? F[i] : F[i] - nd + 100);
    char rb[40] = "";
    o = 0;
    for (int i = 0; i < nrows && o < sizeof(rb) - 6; i++)
        o += (size_t)snprintf(rb + o, sizeof(rb) - o, "%s%d", i ? "+" : "",
                              rows[i]);
    snprintf(pl->name, sizeof(pl->name), "%.31s rec %s [rows %s]", opname, fb,
             rb);
    free(S);
    return plan_select_strategy(pl, err, errlen);
}

int plan_build(const topo_int_t *topo, const rj_op *op, const char *opname,
               plan_t *pl, char *err, size_t errlen)
{
    memset(pl, 0, sizeof(*pl));
    int nd = topo->ndata, np = topo->npar;

    if (op->oom) {
        snprintf(err, errlen, "%s: out of memory building operation", opname);
        return -1;
    }
    if (op->is_parity_update && op->is_recover) {
        snprintf(err, errlen,
                 "%s: malformed operation (recover + parity_update)", opname);
        return -1;
    }

    if (op->is_parity_update) {
        /* Partial-stripe parity update: validate + dedupe the changed
         * list (already sorted by the constructor), then extract the
         * delta columns M[j][changed[i]] into pl->mat. */
        int nc = op->nchanged;
        if (nc < 1) {
            snprintf(err, errlen, "parity_update: empty changed-disk list");
            return -1;
        }
        int *C = malloc((size_t)nc * sizeof(int));
        if (!C)
            goto oom;
        int w = 0;
        for (int i = 0; i < nc; i++) {
            int d = op->changed[i];
            if (d < 0 || d >= nd) {
                snprintf(err, errlen,
                         "parity_update: data-disk index %d out of range "
                         "[0,%d)",
                         d, nd);
                free(C);
                return -1;
            }
            if (w == 0 || d != C[w - 1])
                C[w++] = d;
        }
        nc = w;
        pl->nchanged = nc;
        pl->n_in = 2 * nc + np;
        pl->n_out = np;
        if (np == 0) {
            pl->noop = 1;
            snprintf(pl->name, sizeof(pl->name), "%.31s pupd", opname);
            free(C);
            return 0;
        }
        pl->mat = malloc((size_t)np * (size_t)nc);
        if (!pl->mat) {
            free(C);
            goto oom;
        }
        for (int j = 0; j < np; j++)
            for (int i = 0; i < nc; i++)
                pl->mat[(size_t)j * (size_t)nc + (size_t)i] =
                    topo->mat[(size_t)j * (size_t)nd + (size_t)C[i]];
        /* human-readable name: list changed data disks (truncated) */
        char cb[48] = "";
        size_t o = 0;
        for (int i = 0; i < nc && o < sizeof(cb) - 6; i++)
            o += (size_t)snprintf(cb + o, sizeof(cb) - o, "%s%d",
                                  i ? "+" : "", C[i]);
        snprintf(pl->name, sizeof(pl->name), "%.31s pupd%s%s", opname,
                 nc ? " " : "", cb);
        free(C);
        /* Force P_DELTA: the extracted delta columns may be all-0/1
         * (e.g. RAID-5), but P_XOR codegen cannot seed accumulators from
         * the old parities or form old^new deltas. The P_DELTA mulacc
         * path already emits plain XORs for c == 1, so nothing is lost. */
        pl->kind = P_DELTA;
        /* Bit-level CSE program for the xorbit candidate twin (NULL
         * when there is nothing to share). Built once at plan time. */
        pl->bit_sched = xorbit_build(pl);
        return 0;
    } else if (!op->is_recover) {
        pl->n_in = nd;
        pl->n_out = np;
        if (np == 0) {
            pl->noop = 1;
            snprintf(pl->name, sizeof(pl->name), "%.31s", opname);
            return 0;
        }
        pl->mat = malloc((size_t)np * nd);
        if (!pl->mat)
            goto oom;
        memcpy(pl->mat, topo->mat, (size_t)np * nd);
    } else {
        /* recover: validate the failed list, then solve over all parity
         * rows (the single-subset case of plan_build_recover_rows). */
        int *F = NULL;
        int nf = plan_recover_failed_list(op, nd, np, &F, err, errlen);
        if (nf < 0)
            return -1;
        int *rows = malloc((size_t)(np > 0 ? np : 1) * sizeof(int));
        if (!rows) {
            free(F);
            goto oom;
        }
        for (int r = 0; r < np; r++)
            rows[r] = r;
        int rc = plan_build_recover_rows(topo, F, nf, rows, np, opname,
                                         pl, err, errlen);
        free(rows);
        free(F);
        return rc;
    }

    if (!op->is_recover)
        snprintf(pl->name, sizeof(pl->name), "%.31s", opname);

    return plan_select_strategy(pl, err, errlen);

oom:
    snprintf(err, errlen, "out of memory");
    plan_free(pl);
    return -1;
}

int plan_candidates(const plan_t *pl, const rj_cpu_info *cpu,
                    cand_t *out, int maxout, int allow_nt,
                    size_t prefetch_dist)
{
    int n = 0;
#if defined(__aarch64__)
    /* AArch64: NEON baseline, SVE/SVE2 if present. Nibble method for all
     * multiplies (TBL). SVE is VL-agnostic: conservative unrolls.
     * The SVE backend maps logical vector reg r to z(16+r) for r<16 and
     * to z(r-16) for r in [16,24) -- 24 regs total (z0-z7 + z16-z31;
     * z8-z15 are callee-saved and p1-p15 unused). */
    {
        isa_t isas[2];
        int nis = 0;
        if (cpu->sve && cpu->sve_vl > 0)
            isas[nis++] = ISA_SVE;
        if (cpu->neon)
            isas[nis++] = ISA_NEON;
        if (nis == 0)
            return 0;
        static const int Us_neon[] = {8, 7, 6, 5, 4, 3, 2, 1};
        /* SVE unrolls to 8 like NEON/x86: the MUL VL lane immediate is
         * a signed 4-bit value (u in 0..7 fits), and the exact
         * register-pressure filter below rejects (n_out,U) combos
         * that don't fit the 24-reg SVE file. The 24-reg ceiling
         * ([aarch64-remediation #3]) makes U=6-8 newly feasible where
         * the old 16-reg assumption rejected them. */
        static const int Us_sve[] = {8, 7, 6, 5, 4, 3, 2, 1};
        for (int ii = 0; ii < nis && n < maxout; ii++) {
            isa_t isa = isas[ii];
            /* AArch64 register files: NEON v0-v15 (16 regs), SVE
             * z0-z7+z16-z31 (24 regs, p0 only). Use isa_nregs() rather
             * than a hardcoded constant so the filter tracks the planner's
             * source of truth. */
            int nregs = isa_nregs(isa);
            const int *Us = (isa == ISA_SVE) ? Us_sve : Us_neon;
            int nU = 8;
            for (int ui = 0; ui < nU && n < maxout; ui++) {
                int U = Us[ui];
                int need = plan_need_regs(isa, MUL_NIBBLE, pl, U, 0);
                if (need > nregs)
                    continue;
                out[n++] = (cand_t){.isa = isa, .mul = MUL_NIBBLE, .U = U};
            }
            /* Remat twins for HORNER: unlock larger U via table reload.
             * Enumerated only when the persistent variant does not fit
             * but the remat one does (dominance: at the same U the remat
             * body is the persistent body plus extra pool loads, so it
             * can never win when persistent fits). */
            if (pl->kind == P_HORNER) {
                for (int ui = 0; ui < nU && n < maxout; ui++) {
                    int U = Us[ui];
                    int need_normal =
                        plan_need_regs(isa, MUL_NIBBLE, pl, U, 0);
                    int need_remat =
                        plan_need_regs(isa, MUL_NIBBLE, pl, U, 1);
                    if (need_normal <= nregs)
                        continue;
                    if (need_remat > nregs)
                        continue;
                    out[n++] = (cand_t){.isa = isa, .mul = MUL_NIBBLE, .U = U, .remat = 1};
                }
            }
        }
    }
#elif defined(__riscv)
    /* RISC-V: RVV if present. Nibble method via vrgather, Zvbc
     * carryless twin when the CPU reports both Zvbc and Zvkg.
     * VL-agnostic. */
    {
        if (!cpu->rvv)
            return 0;
        /* RVV unrolls to 8 like x86/NEON/SVE: the exact register-pressure
         * filter below (need * lmul <= 32) rejects (n_out,U,lmul) combos
         * that don't fit the 32 physical registers. */
        static const int Us[] = {8, 7, 6, 5, 4, 3, 2, 1};
        /* LMUL is a genuine tuning axis on RVV: each vector instruction
         * processes lmul*NOMVL bytes, so wider groups divide the per-byte
         * uop/mem cost, at the price of lmul physical registers per
         * logical one (plan_need_regs counts logical vector registers;
         * the backend maps logical r to the lmul-aligned head r*lmul).
         * Every (U, lmul) pair that fits in 32 physical registers is
         * enumerated; the cost model and the empirical benchmark pick. */
        static const int lmuls[] = {1, 2, 4, 8};
        int nregs = isa_nregs(ISA_RVV);
        /* Multiply-method twins (mirrors the x86 AVX-512 GFNI pattern):
         * the Zvbc carryless formulation is gated on Zvbc32e alone.
         * vclmul.vv and vclmulh.vv are both Zvbc (Vector Crypto
         * §36.2/36.3); Zvbc defines them for SEW=64 only, and the
         * SEW=8 forms emitted here require the separate, independent
         * Zvbc32e extension (fast-track vector-crypto-extra §33) --
         * Zvbc does NOT imply Zvbc32e, and Zvbc32e is legal standalone.
         * (Zvkg defines vghsh/vgmul, not vclmulh; the old zvkg gate
         * was a misattribution and bought no safety.) P_XOR has no
         * multiplies: the mul field is irrelevant there, keep a
         * single twin. */
        mul_t muls[2];
        int nmuls = 0;
        if (pl->kind != P_XOR && cpu->rvv_zvbc32e)
            muls[nmuls++] = MUL_ZVBC;
        muls[nmuls++] = MUL_NIBBLE;
        for (int mi = 0; mi < nmuls && n < maxout; mi++) {
            for (int ui = 0; ui < 8 && n < maxout; ui++) {
                for (int li = 0; li < 4 && n < maxout; li++) {
                    int U = Us[ui];
                    int lmul = lmuls[li];
                    int need = plan_need_regs(ISA_RVV, muls[mi], pl, U, 0) *
                               lmul;
                    if (need > nregs)
                        continue;
                    out[n++] = (cand_t){.isa = ISA_RVV,
                                        .mul = muls[mi],
                                        .U = U,
                                        .lmul = lmul};
                }
            }
            /* Remat twins for HORNER: unlock larger U via table reload.
             * Enumerated only when the persistent variant does not fit
             * but the remat one does (dominance: at the same U the remat
             * body is the persistent body plus extra pool loads, so it
             * can never win when persistent fits). Nibble only: the
             * Zvbc path has no tables to rematerialize (and needs fewer
             * registers anyway). */
            if (pl->kind == P_HORNER && muls[mi] == MUL_NIBBLE) {
                for (int ui = 0; ui < 8 && n < maxout; ui++) {
                    for (int li = 0; li < 4 && n < maxout; li++) {
                        int U = Us[ui];
                        int lmul = lmuls[li];
                        int need_normal =
                            plan_need_regs(ISA_RVV, MUL_NIBBLE, pl, U, 0) *
                            lmul;
                        int need_remat =
                            plan_need_regs(ISA_RVV, MUL_NIBBLE, pl, U, 1) *
                            lmul;
                        if (need_normal <= nregs)
                            continue;
                        if (need_remat > nregs)
                            continue;
                        out[n++] = (cand_t){.isa = ISA_RVV,
                                            .mul = MUL_NIBBLE,
                                            .U = U,
                                            .lmul = lmul,
                                            .remat = 1};
                    }
                }
            }
        }
    }
#else
    /* x86-64: existing enumeration */
    /* ISA levels, best first */
    isa_t isas[3];
    int nis = 0;
    if (cpu->avx512f)
        isas[nis++] = ISA_AVX512;
    if (cpu->avx2)
        isas[nis++] = ISA_AVX2;
    /* SSE level: XOR needs only SSE2; multiplies need SSSE3 (pshufb) */
    int sse_ok = (pl->kind == P_XOR) ? cpu->sse2 : cpu->ssse3;
    if (sse_ok)
        isas[nis++] = ISA_SSE;
    if (nis == 0)
        return 0;

    static const int Us[] = {8, 7, 6, 5, 4, 3, 2, 1};
    for (int ii = 0; ii < nis && n < maxout; ii++) {
        isa_t isa = isas[ii];
        int nregs = isa_nregs(isa);
        if (pl->kind == P_XOR) {
            for (unsigned ui = 0; ui < 8 && n < maxout; ui++) {
                int U = Us[ui];
                if (plan_need_regs(isa, MUL_NIBBLE, pl, U, 0) > nregs)
                    continue;
                out[n++] = (cand_t){.isa = isa, .mul = MUL_NIBBLE, .U = U};
            }
        } else {
            /* multiply methods available at this ISA */
            mul_t muls[2];
            int nm = 0;
            if (isa == ISA_AVX512) {
                if (cpu->gfni)
                    muls[nm++] = MUL_GFNI;
                if (cpu->avx512bw)
                    muls[nm++] = MUL_NIBBLE;
            } else {
                muls[nm++] = MUL_NIBBLE;
            }
            if (nm == 0)
                continue;
            for (int mi = 0; mi < nm && n < maxout; mi++) {
                for (unsigned ui = 0; ui < 8 && n < maxout; ui++) {
                    int U = Us[ui];
                    /* Exact backend register need (accumulators n_out*U
                     * plus the backend's temp/table registers); see
                     * plan_need_regs. */
                    int need = plan_need_regs(isa, muls[mi], pl, U, 0);
                    if (need > nregs)
                        continue;
                    out[n++] = (cand_t){.isa = isa, .mul = muls[mi], .U = U};
                }
                /* Rematerialization twins for HORNER: reload tables from
                 * the constant pool into a scratch register instead of
                 * keeping hreg table registers live. This trades L1 loads
                 * for registers, unlocking larger U values that wouldn't
                 * fit otherwise. Only generate for U that needs remat to
                 * fit: at the same U the remat loop body is a strict
                 * superset of the persistent-table body (identical
                 * shuffles and xors plus extra pool loads into tbl, with
                 * no spilling mechanism that could exploit the freed
                 * registers), so a remat twin can never win when the
                 * persistent twin fits and would only waste benchmark
                 * time. The benchmark decides empirically among the
                 * admitted twins. */
                if (pl->kind == P_HORNER) {
                    for (unsigned ui = 0; ui < 8 && n < maxout; ui++) {
                        int U = Us[ui];
                        int need_normal =
                            plan_need_regs(isa, muls[mi], pl, U, 0);
                        int need_remat =
                            plan_need_regs(isa, muls[mi], pl, U, 1);
                        if (need_normal <= nregs)
                            continue; /* normal already fits; remat pointless */
                        if (need_remat > nregs)
                            continue; /* doesn't fit even with remat */
                        out[n++] = (cand_t){.isa = isa, .mul = muls[mi], .U = U, .remat = 1};
                    }
                }
            }
        }
    }
#endif /* __aarch64__ / __riscv / x86-64 */
    /* VPTERNLOG twins for AVX-512 nibble candidates: the benchmark decides
     * empirically whether the 3-input XOR wins (fewer uops but r1+r2 live
     * simultaneously, increasing register pressure). Only for P_MULACC/
     * P_DELTA: the ternlog flag is only consulted in mulacc_column(). */
    if (pl->kind == P_MULACC || pl->kind == P_DELTA) {
        int ntemp = n;
        for (int i = 0; i < ntemp && n < maxout; i++) {
            if (out[i].isa == ISA_AVX512 && out[i].mul == MUL_NIBBLE &&
                !out[i].ternlog) {
                out[n] = out[i];
                out[n].ternlog = 1;
                n++;
            }
        }
    }
    /* VPERMI2B twin (track B2): NOT enumerated here. Enumerating it in
     * stage-1 cost +50% stage-1 builds on MULACC/DELTA (192 vs 128
     * candidates on custom(4,2) encode, 2026-10-03) while bench_select
     * never picked it (0/12 duels on AMD EPYC 9D25). Like the
     * notail/maskedtail/predtail twins, api.c builds it from the
     * stage-1 winner and duels it with the standard bench_select
     * protocol, so the empirical re-decision on other
     * microarchitectures is kept at +1 build + 1 duel, only when the
     * winner is a nibble candidate. */
    /* VPTERNLOG twins for AVX-512 P_XOR candidates: the twin folds each
     * column pair's two per-row XORs into one VPTERNLOGD (in[b] as the
     * memory operand). The benchmark decides empirically; like the
     * nibble twins these are appended after their temporal originals so
     * static selection keeps the plain candidate on ties. Gated on
     * n_out >= 2: with a single row the twin saves no uops (one uop per
     * column either way) but pays n_out*U VPXOR-zero uops up front, so
     * it is statically worse; with >= 2 rows the shared in[a] load
     * amortizes and each row saves one XOR uop per column pair. */
    if (pl->kind == P_XOR && pl->n_out >= 2) {
        int ntemp = n;
        for (int i = 0; i < ntemp && n < maxout; i++) {
            if (out[i].isa == ISA_AVX512 && !out[i].ternlog) {
                out[n] = out[i];
                out[n].ternlog = 1;
                n++;
            }
        }
    }
    /* XOR-schedule twins for P_XOR: a schedule-driven body (cross-row
     * CSE) next to the naive per-row emitter. Enumerated per fitting
     * base candidate only when the static model predicts a STRICT uop
     * win under the emitter's own counting convention -- this is the
     * gate the dominance-pruning track relies on (a twin exists only if
     * it strictly beats naive, so naive can never dominate it). Never
     * combined with ternlog (a different P_XOR body). The NT/prefetch
     * twin loops below see these and twin them automatically, so the
     * full ISA x U x sched x nt x prefetch product is covered. */
    if (pl->kind == P_XOR && pl->xor_sched) {
        int ntemp = n;
        for (int i = 0; i < ntemp && n < maxout; i++) {
            if (out[i].ternlog)
                continue;
            isa_t isa = out[i].isa;
            int U = out[i].U;
            /* RVV: xorsched_need_regs counts logical registers; the
             * twin inherits out[i].lmul, so scale to physical.
             * lmul=0 (zero-initialized) means 1. */
            int lmul = 1;
            if (isa == ISA_RVV && out[i].lmul)
                lmul = out[i].lmul;
            if (xorsched_need_regs(isa, pl, U) * lmul > isa_nregs(isa))
                continue;
            long u_naive =
                xorsched_version_uops(pl->xor_sched, pl, isa, U, 0, 0, 0);
            long u_sched =
                xorsched_version_uops(pl->xor_sched, pl, isa, U, 1, 0, 0);
            if (u_sched >= u_naive)
                continue;
            out[n] = out[i];
            out[n].xorsched = 1;
            n++;
        }
    }
    /* Bit-program twins for GF-multiply rows (track A1): a bit-schedule-
     * driven body (cross-row/coefficient CSE at bit granularity) next to
     * the naive nibble emitter. Enumerated per fitting MUL_NIBBLE base
     * candidate only when the static model predicts a STRICT uop win
     * under the emitter's own counting convention -- this is the gate
     * the dominance-pruning track relies on (a twin exists only if it
     * strictly beats naive, so naive can never dominate it). Never
     * combined with ternlog/permi2b/remat (different bodies), and only
     * on x86 (the bit-program consumer is x86-64-only for now). The
     * NT/prefetch twin loops below see these and twin them automatically,
     * so the full ISA x U x xorbit x nt x prefetch product is covered.
     * NOTE: prune.c's twin_dominates must pin the xorbit axis (like
     * xorsched/permi2b); that pin is a required follow-up (track A1
     * follow-up list) -- without it an xorbit twin could form a
     * cross-axis pruning pair with a ternlog twin. */
    if ((pl->kind == P_MULACC || pl->kind == P_DELTA ||
         pl->kind == P_HORNER) &&
        pl->bit_sched) {
        int ntemp = n;
        for (int i = 0; i < ntemp && n < maxout; i++) {
            if (out[i].ternlog || out[i].permi2b || out[i].remat ||
                out[i].xorsched)
                continue;
            isa_t isa = out[i].isa;
            if (isa != ISA_SSE && isa != ISA_AVX2 && isa != ISA_AVX512)
                continue;
            if (out[i].mul != MUL_NIBBLE)
                continue;
            int U = out[i].U;
            if (xorbit_need_regs(isa, pl, U) > isa_nregs(isa))
                continue;
            long u_naive = xorbit_version_uops(pl->bit_sched, pl, isa, U,
                                              0, 0);
            long u_sched = xorbit_version_uops(pl->bit_sched, pl, isa, U,
                                              1, 0);
            if (u_naive < 0 || u_sched < 0)
                continue;
            if (u_sched >= u_naive)
                continue;
            out[n] = out[i];
            out[n].xorbit = 1;
            n++;
        }
    }
    /* VPMULTISHIFTQB high-nibble twin (track I2, SOTA item 2): the nibble
     * split's VPSRLW becomes a VPMULTISHIFTQB (AVX512_VBMI), moving the
     * shift uop from the shift port to the shuffle port. Enumerated per
     * fitting AVX-512 MUL_NIBBLE base candidate only when the static
     * model predicts a STRICT uop win -- the gate the dominance-pruning
     * track relies on (a twin exists only if it strictly beats the base,
     * so the base can never dominate it). The split is 3 uops either way
     * (AND + shift + AND: the trailing 0x0F AND stays because the
     * multishift output's bits [7:4] carry the neighbor nibble and
     * VPSHUFB zeroes its lane when the index bit 7 is set), so the gate
     * is CLOSED today: this block enumerates nothing, and the
     * port-placement hypothesis waits for quiet-box benchmarks, which
     * can re-open the gate without a code change. Gated on cpu->vbmi
     * (VPMULTISHIFTQB is AVX512_VBMI). Never combined with
     * xorbit/permi2b/remat/xorsched (different bodies); independent of
     * ternlog (split vs fold), so ternlog bases twin too. Requires a
     * nontrivial coefficient (else the split never executes and the twin
     * is vacuous). The NT/prefetch twin loops below see these and twin
     * them automatically. */
    if ((pl->kind == P_MULACC || pl->kind == P_DELTA ||
         pl->kind == P_HORNER) &&
        cpu->vbmi && plan_any_nontrivial(pl)) {
        int ntemp = n;
        for (int i = 0; i < ntemp && n < maxout; i++) {
            if (out[i].isa != ISA_AVX512 || out[i].mul != MUL_NIBBLE)
                continue;
            if (out[i].xorbit || out[i].permi2b || out[i].remat ||
                out[i].xorsched || out[i].mshift)
                continue;
            int U = out[i].U;
            /* Pressure filter: the twin needs one more live register
             * (the multishift control) than plan_need_regs models; keep
             * the +1 in lockstep with jit_build_version's want_regs. */
            if (plan_need_regs(ISA_AVX512, MUL_NIBBLE, pl, U, 0) + 1 >
                isa_nregs(ISA_AVX512))
                continue;
            /* Static uop gate (per (i,u) split): the multishift split
             * must STRICTLY beat the VPSRLW split. Both sequences are
             * exactly 3 uops (AND + shift + AND), so this never fires
             * today -- the port-placement question is for benchmarks. */
            long u_plain = 3;  /* c_and + c_srlw + c_and */
            long u_mshift = 3; /* c_and + c_mshift + c_and */
            if (u_mshift >= u_plain)
                continue;
            out[n] = out[i];
            out[n].mshift = 1;
            n++;
        }
    }
    /* A noop plan (n_out == 0) emits a trivial function with no
     * stores and no loop, so NT/prefetch twins would be
     * byte-identical duplicates of the temporal original (and a "+nt"
     * desc would claim stores that don't exist): skip both twin
     * families. This is ISA-independent. */
    if (pl->noop)
        return n;
    if (allow_nt) {
        /* NT twins after the temporal originals (see plan.h). NT stores
         * only make sense for streaming output; the benchmark decides
         * empirically whether a twin wins. AArch64/NEON with U=1 emits
         * a single temporal STR for nt (no pair to combine), i.e. a
         * byte-identical twin: skip it. RVV implements NT via the
         * Zihintntl NTL.ALL hint prefix (a HINT NOP everywhere, so no
         * gating needed): the twin is never byte-identical, enumerate
         * it. */
        int ntemp = n;
        for (int i = 0; i < ntemp && n < maxout; i++) {
            if (out[i].isa == ISA_NEON && out[i].U == 1)
                continue;
            out[n] = out[i];
            out[n].nt = 1;
            n++;
        }
    }
    if (prefetch_dist) {
        /* Prefetch twins after the NT twins: a prefetch for each input
         * stream prefetch_dist bytes ahead. The benchmark decides
         * empirically whether the extra uops pay off. RVV needs Zicbop
         * for prefetch.r: without it the backend ignores prefetch_dist,
         * so the twin would be a byte-identical duplicate: skip it.
         * RJ_PREFETCH_AUTO (I4): resolve the distance per candidate
         * from the once-per-machine calibration and the candidate's
         * stride (pure plan-time arithmetic, zero runtime cost). A 0
         * result means the heuristic declines the twin (calibration
         * unavailable or below the ISA's minimum encodable distance);
         * the distance-0 base twin above remains the candidate. */
        int ntemp = n;
        for (int i = 0; i < ntemp && n < maxout; i++) {
            if (out[i].isa == ISA_RVV && !cpu->rvv_zicbop)
                continue;
            size_t d = prefetch_dist;
            if (d == RJ_PREFETCH_AUTO) {
                d = prefetch_calib_distance_for(pl, &out[i], cpu);
                if (d == 0)
                    continue;
            }
            out[n] = out[i];
            out[n].prefetch_dist = d;
            n++;
        }
    }
    return n;
}

/* ============ I3: uarch-aware static cost model ============ */

#if defined(__x86_64__)
/* Local CPUID helper (src/x86_64/cpu.c has its own static copy; this
 * TU must not depend on backend internals). */
static void uarch_cpuid(unsigned leaf, unsigned sub,
                        unsigned *a, unsigned *b, unsigned *c, unsigned *d)
{
    __asm__ volatile("cpuid"
                     : "=a"(*a), "=b"(*b), "=c"(*c), "=d"(*d)
                     : "a"(leaf), "c"(sub));
}
#endif

/* Test-only override (see cpu.h). */
static rj_uarch_t uarch_override_val = RJ_UARCH_UNKNOWN;
static int uarch_override_on = 0;

void rj_uarch_override(rj_uarch_t u)
{
    uarch_override_val = u;
    uarch_override_on = 1;
}

rj_uarch_t rj_cpu_uarch(void)
{
#if defined(__x86_64__)
    static int done = 0;
    static rj_uarch_t cached = RJ_UARCH_UNKNOWN;
    if (done)
        return cached;
    done = 1;
    unsigned a, b, c, d;
    uarch_cpuid(0, 0, &a, &b, &c, &d);
    if (a < 1)
        return cached; /* no CPUID leaf 1: unknown */
    char vendor[13];
    memcpy(vendor + 0, &b, 4);
    memcpy(vendor + 4, &d, 4);
    memcpy(vendor + 8, &c, 4);
    vendor[12] = '\0';
    int is_intel = memcmp(vendor, "GenuineIntel", 12) == 0;
    int is_amd = memcmp(vendor, "AuthenticAMD", 12) == 0;
    uarch_cpuid(1, 0, &a, &b, &c, &d);
    /* Family/model with the extended fields (Intel SDM vol. 2A, CPUID):
     * family = extfam + basefam, model = (extmod << 4) | basemodel. */
    unsigned family = ((a >> 8) & 0xFu) + ((a >> 20) & 0xFFu);
    unsigned model = ((a >> 4) & 0xFu) | ((a >> 12) & 0xF0u);
    if (is_intel && family == 6) {
        switch (model) {
        case 0x55: /* Skylake-X / Cascade Lake */
            cached = RJ_UARCH_INTEL_SKYLAKEX;
            break;
        case 0x6A: /* Ice Lake-SP */
        case 0x6C: /* Ice Lake-SP */
        case 0x7D: /* Ice Lake client */
        case 0x7E: /* Ice Lake client */
            cached = RJ_UARCH_INTEL_ICELAKE;
            break;
        case 0x8F: /* Sapphire Rapids */
        case 0xCF: /* Emerald Rapids */
            cached = RJ_UARCH_INTEL_SAPPHIRERAPIDS;
            break;
        default:
            cached = RJ_UARCH_INTEL;
            break;
        }
    } else if (is_intel) {
        cached = RJ_UARCH_INTEL;
    } else if (is_amd) {
        if (family == 0x19) {
            if ((model >= 0x10 && model <= 0x1F) || /* Genoa */
                (model >= 0xA0 && model <= 0xAF))   /* Bergamo (Zen4c) */
                cached = RJ_UARCH_AMD_ZEN4;
            else
                cached = RJ_UARCH_AMD_ZEN3;
        } else if (family == 0x1A) {
            cached = RJ_UARCH_AMD_ZEN5;
        } else {
            cached = RJ_UARCH_AMD;
        }
    }
    return cached;
#else
    /* AArch64 / RISC-V: no per-uarch port data is modeled (I3 is an
     * x86-64 cost model); the neutral entry keeps the static model
     * uarch-independent there. */
    return RJ_UARCH_UNKNOWN;
#endif
}

rj_uarch_t rj_uarch_current(void)
{
    if (uarch_override_on)
        return uarch_override_val;
    return rj_cpu_uarch();
}

/* Per-uarch port-pressure weights. Each t_* is cycles per instruction
 * (reciprocal throughput) for the AVX-512 zmm op class, from published
 * port data -- NOT measurements. All figures below are static
 * estimates; the benchmark remains the final arbiter.
 *
 * Sources (uops.info per-microarchitecture instruction pages;
 * "computed" = from the listed port usage, "measured" = nanoBench):
 * - VPTERNLOGD zmm: https://uops.info/html-instr/VPTERNLOGD_ZMM_ZMM_ZMM_I8.html
 *     Intel (ICL/EMR/SKX): 0.50, 1 uop, ports p0/p5.
 *     Zen4: measured 0.57/0.58 (unrolled); the port-implied 0.25 is
 *       wrong for 512-bit (double-pumped 256-bit pipes: 2x512-bit/cycle
 *       = 0.50). Zen5: measured 0.28-0.40 -> 0.33.
 * - VPSHUFB zmm: https://uops.info/html-instr/VPSHUFB_ZMM_ZMM_ZMM.html
 *     Intel: 1.00, 1 uop, port p5. Zen4: measured 1.00 (AMD doc 1.00).
 *     Zen5: measured 0.50.
 * - VPXORD zmm: https://uops.info/html-instr/VPXORD_ZMM_ZMM_ZMM.html
 *     Intel: 0.50, 1 uop, ports p0/p5. Zen4: measured 0.50 (AMD doc
 *     0.50). Zen5: 0.25 (AMD doc; measured 0.25/0.37).
 * - VGF2P8MULB zmm: https://uops.info/html-instr/VGF2P8MULB_ZMM_ZMM_ZMM.html
 *     Intel: 1.00, 1 uop, port p0. Zen4: measured 1.00 (AMD doc 1.00).
 *     Zen5: measured 0.50.
 * - VGF2P8AFFINEQB zmm: https://uops.info/html-instr/VGF2P8AFFINEQB_ZMM_ZMM_ZMM_I8.html
 *     Intel: 1.00, 1 uop, port p0. Zen4: measured 1.00. Zen5: 0.50.
 * - 64-byte vector load: Intel 0.50 (two 32B/cycle load ports); Zen4
 *   1.00 (double-pumped 32B pipes -> one 64B load/cycle; AMD SOG);
 *   Zen5 0.50 (est.: native 512-bit datapath, two loads/cycle).
 * - freq512: Intel documents license-based frequency reduction under
 *   sustained 512-bit execution (Intel 64 and IA-32 Architectures
 *   Optimization Reference Manual); the numeric derates are rough
 *   estimates (est.): SKX 1.10, SPR/EMR 1.05, client ICL ~1.0, AMD 1.0
 *   (no 512-bit downclocking).
 * - vec_bw (fallback dispatch width): Intel 4, AMD 6 -- approximate,
 *   only used for candidate classes without a kernel model.
 * - bw_bpc (#31): peak single-core sustainable DRAM streaming
 *   bandwidth, bytes/cycle (est., NOT measured -- no benchmark feeds
 *   this number). Conservative static per-uarch-class figures at the
 *   low end of published single-core streaming numbers: DDR4-era
 *   classes (SKX/ICL, single-core memcpy/NT-store ~20-25 GB/s at
 *   ~3 GHz ~= 7-8 B/cyc) get 8; DDR5-era classes (SPR/Zen4/Zen5,
 *   ~30-40 GB/s at ~3 GHz ~= 10-13 B/cyc) get 12; UNKNOWN gets the
 *   conservative 8. Deliberately low: the model errs toward
 *   bandwidth-bound, so a byte-frugal candidate is never ranked worse
 *   on uop counts alone. A zero bw_bpc (only possible in hand-built
 *   test weights, never in this table) degrades the cost to the
 *   compute leg alone.
 * - SKX has no GFNI: t_gfni falls back to the Intel value (unused in
 *   practice). Zen3 has no AVX-512: it aliases the generic AMD row
 *   (the 512-bit entries are unused there).
 * - RJ_UARCH_UNKNOWN reproduces the old generic roofline
 *   (uops/3.5 + mem/64): all t_* = 1/3.5, vec_bw = 3.5, so the cost
 *   model degrades gracefully to uarch-independent behavior. */
static const rj_uarch_weights_t uarch_w_unknown = {
    1.0 / 3.5, 1.0 / 3.5, 1.0 / 3.5, 1.0 / 3.5, 1.0 / 3.5,
    3.5, 1.0 / 64.0, 1.0, 8.0 /* est. */
};
static const rj_uarch_weights_t uarch_w_icl = {
    0.50, 1.00, 0.50, 1.00, 0.50,
    4.0, 0.50 / 64.0, 1.00, 8.0 /* est. */
};
static const rj_uarch_weights_t uarch_w_skx = {
    0.50, 1.00, 0.50, 1.00, 0.50,
    4.0, 0.50 / 64.0, 1.10 /* est. */, 8.0 /* est. */
};
static const rj_uarch_weights_t uarch_w_spr = {
    0.50, 1.00, 0.50, 1.00, 0.50,
    4.0, 0.50 / 64.0, 1.05 /* est. */, 12.0 /* est. */
};
static const rj_uarch_weights_t uarch_w_zen4 = {
    0.50, 1.00, 0.50, 1.00, 1.00,
    6.0, 1.00 / 64.0, 1.00, 12.0 /* est. */
};
static const rj_uarch_weights_t uarch_w_zen5 = {
    0.33, 0.50, 0.25, 0.50, 0.50,
    6.0, 0.50 / 64.0, 1.00, 12.0 /* est. */
};

const rj_uarch_weights_t *rj_uarch_weights(rj_uarch_t u)
{
    switch (u) {
    case RJ_UARCH_INTEL_SKYLAKEX:
        return &uarch_w_skx;
    case RJ_UARCH_INTEL_ICELAKE:
    case RJ_UARCH_INTEL:
        return &uarch_w_icl;
    case RJ_UARCH_INTEL_SAPPHIRERAPIDS:
        return &uarch_w_spr;
    case RJ_UARCH_AMD_ZEN4:
    case RJ_UARCH_AMD_ZEN3: /* no AVX-512; 512-bit entries unused */
    case RJ_UARCH_AMD:
        return &uarch_w_zen4;
    case RJ_UARCH_AMD_ZEN5:
        return &uarch_w_zen5;
    case RJ_UARCH_UNKNOWN:
    default:
        return &uarch_w_unknown;
    }
}

/* Effective cycles per uop for the candidate's characteristic kernel
 * op mix (I3). Characteristic multiply-kernel op mix per candidate
 * class. Counts are per GF multiply (nibble/GFNI) or per folded xor
 * pair (P_XOR), read off the emitter sequences in src/x86_64/jit.c:
 * mulacc_column emits 2 pool loads + 2 VPSHUFB + the fold (2 VPXOR,
 * or 1 VPTERNLOGD in the +ternlog twin); the GFNI path emits
 * 1 VGF2P8AFFINEQB + 1 VGF2P8MULB + 1 VPXOR with the affine-table
 * pool load amortized across the column (hence ld 0). P_XOR mixes
 * are approximate: the body is a pure xor reduction (2 xors per
 * load in the plain form; the twin folds each pair into one
 * VPTERNLOGD). The blend covers the uarch-sensitive kernel only;
 * kernel-external uops (nibble split, loop overhead, stores) are
 * uarch-neutral to first order and folded into the same per-uop
 * rate. This is a HEURISTIC for static ordering (est.), not a
 * cycle-accurate model: the benchmark is the final arbiter. */
static double kernel_w_uop(plan_kind_t kind, const cand_t *cand,
                           const rj_uarch_weights_t *w)
{
    double n_tl = 0.0, n_shuf = 0.0, n_xor = 0.0, n_gfni = 0.0, n_ld = 0.0;
    int modeled = 0;
    if (cand->isa == ISA_AVX512) {
        if (kind == P_MULACC || kind == P_DELTA || kind == P_HORNER) {
            if (cand->mul == MUL_GFNI) {
                n_gfni = 2.0;
                n_xor = 1.0;
                modeled = 1;
            } else if (cand->mul == MUL_NIBBLE) {
                n_shuf = 2.0;
                n_ld = 2.0;
                if (cand->ternlog)
                    n_tl = 1.0;
                else
                    n_xor = 2.0;
                modeled = 1;
            }
        } else if (kind == P_XOR) {
            n_ld = 1.0;
            if (cand->ternlog)
                n_tl = 1.0;
            else
                n_xor = 2.0;
            modeled = 1;
        }
    }
    if (modeled) {
        double ntot = n_tl + n_shuf + n_xor + n_gfni + n_ld;
        return (n_tl * w->t_ternlog + n_shuf * w->t_shuffle +
                n_xor * w->t_xor + n_gfni * w->t_gfni +
                n_ld * w->t_ld64) / ntot;
    }
    /* No kernel model for this class (non-AVX512 ISA or exotic
     * mul): generic dispatch-width rate. */
    return 1.0 / w->vec_bw;
}

double plan_uarch_cost(double uops_per_byte, double mem_per_byte,
                       plan_kind_t kind, const cand_t *cand, rj_uarch_t u)
{
    const rj_uarch_weights_t *w = rj_uarch_weights(u);
    /* Effective cycles per uop: the I3 kernel-mix rate, or the generic
     * dispatch-width rate when there are no candidate axes. */
    double w_uop = cand ? kernel_w_uop(kind, cand, w) : 1.0 / w->vec_bw;
    /* #31: roofline blend of the compute leg and the bandwidth leg.
     *
     * Compute leg: t_compute = uops_per_byte * w_uop (cycles/byte at
     * this uarch's port-weighted rate for the kernel's op mix).
     * Bandwidth leg: t_mem = mem_per_byte / bw_bpc (cycles/byte at the
     * uarch's conservative static single-core streaming bandwidth;
     * est., NOT measured).
     *
     * Arithmetic intensity AI = uops_per_byte / mem_per_byte (uops per
     * byte moved); ridge point ridge = (1/w_uop) / bw_bpc (peak
     * uops/cycle over peak bytes/cycle, both from I3's weights and the
     * static bandwidth figure). t_compute / t_mem = AI / ridge, so the
     * blend keys off exactly that comparison: below the ridge
     * (bandwidth-bound) the cost is ~t_mem (bytes moved); above it
     * (compute-bound) ~t_compute (uop cost). The blend is the
     * continuous p-norm (t_compute^4 + t_mem^4)^{1/4} -- at the ridge
     * both legs contribute equally (cost = 2^{1/4} x either leg), and
     * there is no hard max() switch. The blend is monotone in both
     * legs, which the prune gate's roofline leg relies on: a twin with
     * fewer uops and equal bytes can never cost more than the plain.
     * A non-positive bw_bpc (only possible in hand-built test
     * weights) degrades gracefully to the compute leg alone.
     * HEURISTIC for static ordering (est.): the benchmark is the final
     * arbiter. */
    double t_compute = uops_per_byte * w_uop;
    double t_mem = w->bw_bpc > 0.0 ? mem_per_byte / w->bw_bpc : 0.0;
    double c2 = t_compute * t_compute;
    double m2 = t_mem * t_mem;
    double cost = sqrt(sqrt(c2 * c2 + m2 * m2));
    if (cand && cand->isa == ISA_AVX512)
        cost *= w->freq512;
    return cost;
}
