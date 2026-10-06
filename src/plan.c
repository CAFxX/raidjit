#include "plan.h"
#include "gf.h"
#include "xorsched.h"
#include <assert.h>
#include <limits.h>
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
         * (t..t+6); GFNI adds t2,tbl (t..t+2). HORNER adds 2 (nibble) or 1
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
        if (nontriv)
            return n_out * U + 7;
        /* All-trivial: the nibble split never runs, so tlo/thi/r2/tbl/
         * mask are dead. r1 survives on SSE as the delta temp (P_DELTA's
         * two-operand old^new needs it); on AVX the delta folds into a
         * memory XOR and only t remains. (P_MULACC all-trivial is
         * unreachable from the planner; its r1 would be dead too.) */
        return n_out * U +
               ((pl->kind == P_DELTA && isa == ISA_SSE) ? 2 : 1);
    }
    if (a64) {
        /* src/aarch64/jit.c: t,(tlo,thi,r1,r2,tbl,mask); HORNER compacts
         * its 2 table regs to nontrivial rows only, unless remat (tables
         * reloaded into tbl; no hreg). */
        if (pl->kind == P_HORNER) {
            if (remat)
                return n_out * U + 6;
            return n_out * U + 5 + 2 * horner_ntriv(pl);
        }
        /* P_MULACC / P_DELTA: all-trivial kills tlo/thi/r2/tbl/mask, but
         * r1 stays live as the P_DELTA delta temp on both NEON and SVE
         * (old^new is formed into r1 for every column unconditionally).
         * (P_MULACC all-trivial is unreachable; its r1 would be dead.) */
        return n_out * U + (nontriv ? 7 : (pl->kind == P_DELTA ? 2 : 1));
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
        return 0;
    }
    free(pl->horner_c);
    pl->horner_c = NULL;
    pl->kind = P_MULACC;
    return 0;
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
    if (solve_recovery_rows(topo->mat, nd, np, F, nf, S, ns, rows, nrows,
                            pl->mat, NULL) != 0) {
        snprintf(err, errlen,
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
        static const int Us_sve[] = {5, 4, 3, 2, 1};
        for (int ii = 0; ii < nis && n < maxout; ii++) {
            isa_t isa = isas[ii];
            /* AArch64 register files: NEON v0-v15 (16 regs), SVE
             * z0-z7+z16-z31 (24 regs, p0 only). Use isa_nregs() rather
             * than a hardcoded constant so the filter tracks the planner's
             * source of truth. */
            int nregs = isa_nregs(isa);
            const int *Us = (isa == ISA_SVE) ? Us_sve : Us_neon;
            int nU = (isa == ISA_SVE) ? 5 : 8;
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
        static const int Us[] = {5, 4, 3, 2, 1};
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
            for (int ui = 0; ui < 5 && n < maxout; ui++) {
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
                for (int ui = 0; ui < 5 && n < maxout; ui++) {
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
         * so the twin would be a byte-identical duplicate: skip it. */
        int ntemp = n;
        for (int i = 0; i < ntemp && n < maxout; i++) {
            if (out[i].isa == ISA_RVV && !cpu->rvv_zicbop)
                continue;
            out[n] = out[i];
            out[n].prefetch_dist = prefetch_dist;
            n++;
        }
    }
    return n;
}
