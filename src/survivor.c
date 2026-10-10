/* Cheapest measured survivor set for recovery (track A5). See
 * survivor.h and docs/survivor_sets.md. */
#include "survivor.h"
#include "bench.h"
#include "prefetch_calib.h"
#include "prune.h"
#include "gf.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

static int cmp_int_asc(const void *a, const void *b)
{
    return *(const int *)a - *(const int *)b;
}

/* C(n,k), saturating at > cap (returns cap+1). */
static long choose_cap(int n, int k, long cap)
{
    if (k < 0 || k > n)
        return 0;
    if (k > n - k)
        k = n - k;
    long r = 1;
    for (int i = 1; i <= k; i++) {
        /* r = r * (n-k+i) / i, checked against cap */
        if (r > cap / (n - k + i))
            return cap + 1;
        r = r * (n - k + i) / i;
    }
    return r;
}

static int kind_rank_of(plan_kind_t k)
{
    switch (k) {
    case P_XOR: return 0;
    case P_HORNER: return 1;
    case P_MULACC: return 2;
    case P_DELTA: return 3;
    }
    return 4;
}

void survivor_free_subsets(survivor_subset_t *subs, int nsub)
{
    if (!subs)
        return;
    for (int i = 0; i < nsub; i++) {
        free(subs[i].rows);
        plan_free(&subs[i].plan);
    }
    free(subs);
}

int survivor_over_cap(const topo_int_t *ti, const int *F, int nf)
{
    int nd = ti->ndata, np = ti->npar;
    if (nf < 1 || nf > np)
        return 0;
    int f_par = 0, nprows = 0;
    for (int j = 0; j < np; j++) {
        int failed = 0;
        for (int i = 0; i < nf; i++)
            if (F[i] == nd + j) {
                failed = 1;
                break;
            }
        if (failed)
            f_par++;
        else
            nprows++;
    }
    int f_data = nf - f_par;
    if (f_data < 0 || f_data > nprows)
        return 0;
    return choose_cap(nprows, f_data, SURVIVOR_ENUM_CAP) > SURVIVOR_ENUM_CAP;
}

int survivor_enumerate(const topo_int_t *ti, const int *F, int nf,
                       const char *opname, survivor_subset_t **out,
                       int *nsub_out, char *err, size_t errlen)
{
    *out = NULL;
    *nsub_out = 0;
    int nd = ti->ndata, np = ti->npar;
    if (nf < 1 || nf > np) {
        snprintf(err, errlen, "survivor: bad failure count %d (npar=%d)",
                 nf, np);
        return -1;
    }
    /* Fp: failed parity rows (forced). R_surv: surviving parity rows. */
    int *Fp = malloc((size_t)np * sizeof(int));
    int *Rs = malloc((size_t)np * sizeof(int));
    if (!Fp || !Rs) {
        free(Fp);
        free(Rs);
        snprintf(err, errlen, "out of memory");
        return -1;
    }
    int f_par = 0, nprows = 0;
    for (int j = 0; j < np; j++) {
        int failed = 0;
        for (int i = 0; i < nf; i++)
            if (F[i] == nd + j) {
                failed = 1;
                break;
            }
        if (failed)
            Fp[f_par++] = j;
        else
            Rs[nprows++] = j;
    }
    int f_data = nf - f_par;
    if (f_data < 0 || f_data > nprows) {
        /* Cannot happen for a validated F (nf <= npar), but fail
         * loudly rather than enumerating nonsense. */
        free(Fp);
        free(Rs);
        snprintf(err, errlen, "survivor: inconsistent failure set");
        return -1;
    }
    if (survivor_over_cap(ti, F, nf)) {
        free(Fp);
        free(Rs);
        snprintf(err, errlen,
                 "survivor: subset enumeration C(%d,%d) exceeds cap %d",
                 nprows, f_data, SURVIVOR_ENUM_CAP);
        return -2; /* cap exceeded: caller falls back to all-rows plan */
    }

    /* Survivors S (increasing), shared by all subset solves. */
    int ns = nd + np - nf;
    int *S = malloc((size_t)(ns > 0 ? ns : 1) * sizeof(int));
    int *allrows = malloc((size_t)(np > 0 ? np : 1) * sizeof(int));
    int *piv = malloc((size_t)(nf > 0 ? nf : 1) * sizeof(int));
    if (!S || !allrows || !piv) {
        free(Fp);
        free(Rs);
        free(S);
        free(allrows);
        free(piv);
        snprintf(err, errlen, "out of memory");
        return -1;
    }
    {
        int fi = 0, si = 0;
        for (int d = 0; d < nd + np; d++) {
            if (fi < nf && F[fi] == d)
                fi++;
            else
                S[si++] = d;
        }
    }
    for (int r = 0; r < np; r++)
        allrows[r] = r;
    /* Today's subset T: the all-rows greedy solve. This must succeed
     * whenever the op is recoverable at all (it is exactly what
     * plan_build does today). We keep Rt (the all-rows solution matrix)
     * to identify T below: the enumerated subset whose plan matrix is
     * bit-identical to Rt is today's solution and becomes the
     * no-regression anchor (is_today). */
    uint8_t *Rt = malloc((size_t)nf * (size_t)(ns > 0 ? ns : 1));
    if (!Rt) {
        free(Fp);
        free(Rs);
        free(S);
        free(allrows);
        free(piv);
        snprintf(err, errlen, "out of memory");
        return -1;
    }
    if (solve_recovery_rows(ti->mat, nd, np, F, nf, S, ns, allrows, np,
                            Rt, piv) != 0) {
        free(Fp);
        free(Rs);
        free(S);
        free(allrows);
        free(piv);
        free(Rt);
        snprintf(err, errlen,
                 "recover: failure set is not recoverable "
                 "(singular system)");
        return -1;
    }
    /* piv correctly reports each unknown's pivot equation (the rows R
     * is read from), but T is still identified by matrix equality
     * below, not by piv: two distinct subsets can in principle yield
     * bit-identical R, and any of them is a valid anchor (identical R
     * ==> identical plan), while pivot-row identity would couple the
     * anchor to elimination internals (partial-pivot order). Do not
     * use piv to identify T. */
    free(piv);
    piv = NULL;

    /* Enumerate C(nprows, f_data) lexicographically. */
    survivor_subset_t *subs = NULL;
    int nsub = 0, cap = 0;
    /* today_found: the is_today anchor is the FIRST subset whose
     * solution matrix is bit-identical to the all-rows solve. Two
     * distinct subsets can in principle yield bit-identical R (e.g.
     * symmetric topologies); either is a valid anchor, so only the
     * first match is marked and the uniqueness assumption is
     * documented rather than silently double-marking (C6). */
    int today_found = 0;
    int *c = malloc((size_t)(f_data > 0 ? f_data : 1) * sizeof(int));
    if (!c) {
        free(Fp);
        free(Rs);
        free(S);
        free(allrows);
        free(piv);
        snprintf(err, errlen, "out of memory");
        return -1;
    }
    for (int i = 0; i < f_data; i++)
        c[i] = i;
    int done = 0;
    /* rows[] scratch: Fp (sorted) followed by the chosen surviving rows
     * (increasing), then sorted as a whole for the canonical form. */
    int *rows = malloc((size_t)(nf > 0 ? nf : 1) * sizeof(int));
    if (!rows) {
        free(Fp);
        free(Rs);
        free(S);
        free(allrows);
        free(piv);
        free(c);
        snprintf(err, errlen, "out of memory");
        return -1;
    }
    int rc = 0;
    while (!done) {
        for (int i = 0; i < f_par; i++)
            rows[i] = Fp[i];
        for (int i = 0; i < f_data; i++)
            rows[f_par + i] = Rs[c[i]];
        qsort(rows, (size_t)nf, sizeof(int), cmp_int_asc);
        plan_t pl;
        char e2[256];
        if (plan_build_recover_rows(ti, F, nf, rows, nf, opname, &pl, e2,
                                    sizeof(e2)) == 0) {
            if (nsub == cap) {
                int ncap = cap ? cap * 2 : 16;
                survivor_subset_t *ns2 =
                    realloc(subs, (size_t)ncap * sizeof(*ns2));
                if (!ns2) {
                    plan_free(&pl);
                    rc = -1;
                    snprintf(err, errlen, "out of memory");
                    break;
                }
                subs = ns2;
                cap = ncap;
            }
            survivor_subset_t *s = &subs[nsub];
            memset(s, 0, sizeof(*s));
            s->rows = malloc((size_t)nf * sizeof(int));
            if (!s->rows) {
                plan_free(&pl);
                rc = -1;
                snprintf(err, errlen, "out of memory");
                break;
            }
            memcpy(s->rows, rows, (size_t)nf * sizeof(int));
            s->nrows = nf;
            s->plan = pl; /* struct copy; pl now owned by s */
            s->kind_rank = kind_rank_of(pl.kind);
            long nnz = 0;
            int ncols = 0;
            for (int col = 0; col < ns; col++) {
                int any = 0;
                for (int r = 0; r < nf; r++)
                    if (pl.mat[(size_t)r * ns + col]) {
                        any = 1;
                        nnz++;
                    }
                ncols += any;
            }
            s->nnz = nnz;
            s->ncols = ncols;
            /* is_today: the subset whose solution matrix is bit-identical
             * to the all-rows solve (Rt). This is today's solution and
             * the no-regression anchor. Only the first match is marked
             * (see today_found above). */
            s->is_today =
                (!today_found && nf > 0 && ns > 0 &&
                 memcmp(pl.mat, Rt, (size_t)nf * (size_t)ns) == 0);
            if (s->is_today)
                today_found = 1;
            nsub++;
        }
        /* else: singular subset -- not a candidate, skip silently.
         * (plan_build_recover_rows returns -1 for OOM as well; OOM here
         * is treated as "skip" which leads to the loud "no valid subset"
         * error below if all subsets fail. A targeted OOM distinction
         * would require a richer error code.) */
        if (f_data == 0) {
            done = 1;
        } else {
            int i = f_data - 1;
            while (i >= 0 && c[i] == nprows - f_data + i)
                i--;
            if (i < 0) {
                done = 1;
            } else {
                c[i]++;
                for (int j = i + 1; j < f_data; j++)
                    c[j] = c[j - 1] + 1;
            }
        }
    }
    free(Fp);
    free(Rs);
    free(S);
    free(allrows);
    free(piv);
    free(Rt);
    free(c);
    free(rows);
    if (rc != 0) {
        survivor_free_subsets(subs, nsub);
        return -1;
    }
    if (nsub == 0) {
        /* The all-rows solve succeeded, so at least today's subset is
         * valid: reaching here means an internal inconsistency. */
        snprintf(err, errlen, "survivor: internal error (no valid subset)");
        return -1;
    }
    *out = subs;
    *nsub_out = nsub;
    return 0;
}

void survivor_l2_signal(survivor_subset_t *subs, int nsub,
                        const rj_cpu_info *cpu, const rj_options *o)
{
    /* NT/prefetch candidate flags follow the same rule as the full
     * build in survivor_select, so the canonical version is the same
     * candidate the full build would consider first. */
    int allow_nt = o->nt_threshold > 0 && o->benchmark;
    size_t prefetch_dist = prefetch_calib_dist_arg(o);
    for (int i = 0; i < nsub; i++) {
        subs[i].l2_valid = 0;
        subs[i].l2_uops = 0;
        subs[i].l2_mem = 0;
        /* Exactly ONE version per subset: plan_candidates[0] is the
         * best ISA at the highest fitting U (see plan_candidates).
         * Its (uops_per_byte, mem_per_byte) is the EXACT static signal
         * from the real cost model -- no duplicated model (twin-model
         * lockstep, AGENTS.md). The signal is deliberately
         * NT/prefetch-blind: always the plain temporal canonical,
         * never the twins. P1.2 decision: a static extension to the
         * twin family is vacuous (NT twins tie statically by
         * construction; prefetch twins are statically dominated by
         * their originals), so the no-flip property rests on the
         * NT-delta subset-independence assumption documented in
         * docs/survivor_sets.md section 2c, backstopped by the Part-7
         * tripwire run with NT twins enabled. */
        cand_t cands[512];
        int ncand = plan_candidates(&subs[i].plan, cpu, cands, 512,
                                    allow_nt, prefetch_dist);
        if (ncand <= 0)
            continue;
        version_t v;
        memset(&v, 0, sizeof(v));
        char e2[256];
        if (jit_build_version(&subs[i].plan, &cands[0], cpu, o, &v, e2,
                              sizeof(e2)) != 0) {
            if (o->verbose)
                fprintf(stderr,
                        "raidjit: survivor L2 signal build failed "
                        "for subset %d: %s\n",
                        i, e2);
            continue;
        }
        subs[i].l2_uops = v.uops_per_byte;
        subs[i].l2_mem = v.mem_per_byte;
        subs[i].l2_valid = 1;
        jit_free_version(&v);
    }
}

/* j dominates i on the exact static signal? Both signals must be
 * valid; a subset without a signal is never dominated (a missing
 * signal can only keep a subset, never drop it). */
static int l2_dominates(const survivor_subset_t *subs, int j, int i)
{
    if (!subs[j].l2_valid || !subs[i].l2_valid)
        return 0;
    return subs[j].l2_uops <= subs[i].l2_uops &&
           subs[j].l2_mem <= subs[i].l2_mem &&
           (subs[j].l2_uops < subs[i].l2_uops ||
            subs[j].l2_mem < subs[i].l2_mem);
}

/* Static rank for the top-(K-1) cut: lexicographic on (l2_uops,
 * l2_mem); subsets without a signal rank last; ties break by subset
 * index. Lower is better. Returns nonzero if a ranks strictly before
 * b. NaN signals compare false on every relation, so a NaN signal
 * ties everything and keeps enumeration order -- deterministic, and
 * it can only keep a subset, never wrongly dominate. */
static int l2_rank_less(const survivor_subset_t *subs, int a, int b)
{
    if (subs[a].l2_valid != subs[b].l2_valid)
        return subs[a].l2_valid > subs[b].l2_valid;
    if (subs[a].l2_uops != subs[b].l2_uops)
        return subs[a].l2_uops < subs[b].l2_uops;
    if (subs[a].l2_mem != subs[b].l2_mem)
        return subs[a].l2_mem < subs[b].l2_mem;
    return a < b;
}

void survivor_shortlist(survivor_subset_t *subs, int nsub, int *keep)
{
    const char *env = getenv("RJ_NO_SUBSET_PRUNE");
    int disabled =
        env && env[0] != '\0' && strcmp(env, "0") != 0;
    /* Test-only hook: RJ_SUBSET_PRUNE_INVERT inverts the pareto
     * decision (keeps dominated subsets, drops non-dominated ones) and
     * skips the is_today anchor. It exists so the test suite can prove
     * the shortlist pins a real outcome (red-green: the pinned
     * assertions fail under the inversion). Never set in production. */
    int invert = getenv("RJ_SUBSET_PRUNE_INVERT") != NULL;
    for (int i = 0; i < nsub; i++)
        keep[i] = disabled ? 1 : 0;
    if (disabled)
        return;
    /* Pareto frontier on the exact static signal (l2_uops, l2_mem).
     * O(nsub^2); nsub is small after the enumeration cap. */
    int *in_pareto = calloc((size_t)(nsub > 0 ? nsub : 1), sizeof(int));
    if (!in_pareto)
        return; /* OOM: keep[] stays all-zero; the caller treats an
                 * empty shortlist as "no valid subset" and fails
                 * loudly rather than benchmarking nothing. */
    for (int i = 0; i < nsub; i++) {
        int dominated = 0;
        for (int j = 0; j < nsub && !dominated; j++) {
            if (i == j)
                continue;
            if (l2_dominates(subs, j, i))
                dominated = 1;
        }
        /* keep pareto members (inverted under the test-only
         * RJ_SUBSET_PRUNE_INVERT hook: keep dominated instead) */
        in_pareto[i] = invert ? dominated : !dominated;
    }
    /* Static rank of the pareto members for the top-(K-1) cut:
     * simple O(n^2) selection sort (the pareto set is tiny; avoids
     * qsort_r portability issues). */
    int *ranked = malloc((size_t)(nsub > 0 ? nsub : 1) * sizeof(int));
    int nranked = 0;
    if (ranked) {
        for (int i = 0; i < nsub; i++)
            if (in_pareto[i])
                ranked[nranked++] = i;
        for (int i = 0; i < nranked; i++) {
            int best = i;
            for (int j = i + 1; j < nranked; j++)
                if (l2_rank_less(subs, ranked[j], ranked[best]))
                    best = j;
            int tmp = ranked[i];
            ranked[i] = ranked[best];
            ranked[best] = tmp;
        }
    }
    /* The no-regression anchor: today's subset always survives
     * (skipped under the inversion hook). Then the statically
     * top-(K-1) pareto members, hard-capped at SURVIVOR_SHORTLIST_CAP
     * total. */
    int nkept = 0;
    if (!invert) {
        for (int i = 0; i < nsub; i++) {
            if (subs[i].is_today) {
                keep[i] = 1;
                nkept++;
                break; /* first match only (C6) */
            }
        }
    }
    if (ranked) {
        for (int r = 0; r < nranked && nkept < SURVIVOR_SHORTLIST_CAP;
             r++) {
            if (!keep[ranked[r]]) {
                keep[ranked[r]] = 1;
                nkept++;
            }
        }
        free(ranked);
    } else {
        /* ranked OOM: keep the whole pareto set rather than an empty
         * shortlist. */
        for (int i = 0; i < nsub; i++)
            if (in_pareto[i])
                keep[i] = 1;
    }
    free(in_pareto);
}

int survivor_significance_gate(const version_t *pool, const int *pool_sub,
                               int npool, int today_si, int winner,
                               double window)
{
    if (winner < 0 || winner >= npool || today_si < 0 || window <= 0)
        return winner;
    int win_si = pool_sub[winner];
    if (win_si < 0 || win_si == today_si)
        return winner; /* today won outright: nothing to gate */
    /* Per-subset best measured GB/s from the already-measured pool.
     * vtot is identical for every pooled version (same n_in/n_out, no
     * notail twins in the pool), so GB/s ordering == bench time
     * ordering exactly. */
    double g_win = -1.0, g_today = -1.0;
    for (int i = 0; i < npool; i++) {
        if (pool[i].gbps <= 0)
            continue;
        if (pool_sub[i] == win_si && pool[i].gbps > g_win)
            g_win = pool[i].gbps;
        if (pool_sub[i] == today_si && pool[i].gbps > g_today)
            g_today = pool[i].gbps;
    }
    if (g_win <= 0 || g_today <= 0)
        return winner;
    /* The bench protocol's noise window is relative on time; applied
     * here on the GB/s-relative scale (equal to first order). A lead
     * within the window is a noise flip, not a genuine win. */
    double lead = (g_win - g_today) / g_today;
    if (lead > window)
        return winner; /* genuine win: ships */
    /* Noise flip: fall back to today's pool entries, resolved by the
     * bench protocol's own selector -- select_by_time restricted to
     * today's entries with the measured window. 1/gbps is an exact
     * relative-time proxy here: vtot is identical for every pooled
     * version (same n_in/n_out, no notail twins in the pool), so GB/s
     * ordering == bench time ordering exactly. Only candidates within
     * the window of today's fastest are eligible, and the
     * (uops/byte, mem/byte, code size) tie-break applies to exactly
     * that within-noise set -- identical to what bench_select would
     * pick among today's entries. (The old code applied the tie-break
     * only within 1e-9 of max GB/s; select_by_time applies it to all
     * candidates within the measured window, exactly mirroring the
     * bench protocol.)
     *
     * No-regression reasoning: this branch fires only when the
     * winner's lead over today's best measured GB/s is within
     * `window`, so every candidate the selector can return is
     * measurement-indistinguishable from the winner; whichever
     * within-noise entry wins the tie-break, the returned version is
     * never measurably worse than today's measured best. */
    int *tix = malloc(sizeof(int) * (size_t)npool);
    double *tsecs = malloc(sizeof(double) * (size_t)npool);
    version_t *tvs = malloc(sizeof(version_t) * (size_t)npool);
    int fallback = winner;
    if (tix && tsecs && tvs) {
        int nt = 0;
        for (int i = 0; i < npool; i++) {
            if (pool_sub[i] != today_si || pool[i].gbps <= 0)
                continue;
            tix[nt] = i;
            tvs[nt] = pool[i]; /* shallow copy: never freed; the caller
                                * still owns the pool */
            tsecs[nt] = 1.0 / pool[i].gbps;
            nt++;
        }
        /* nt >= 1: g_today > 0 above guarantees a measured today entry.
         * window > 0 was checked on entry. */
        if (nt > 0) {
            int k = select_by_time(tvs, nt, tsecs, window);
            if (k >= 0 && k < nt)
                fallback = tix[k];
        }
    }
    free(tix);
    free(tsecs);
    free(tvs);
    return fallback;
}

/* ---------------- LRC locality-aware recovery ----------------
 *
 * LRC topologies (docs/lrc.md) have local parity rows: pure-XOR rows
 * with disjoint supports partitioning the data disks. A failure inside
 * a local group recovers from r group-mates (pure XOR) instead of via
 * the global rows (GF-multiply over ~k streams). The A5 survivor
 * machinery already *happens* to prefer the local subset via the exact
 * static signal (it strictly dominates on both axes), but that is
 * emergent, not structural: benchmark noise or cost-model drift could
 * silently lose it. These functions make locality an explicit planner
 * rule.
 *
 * Detection is by matrix structure, not topology name: any matrix with
 * pure-XOR disjoint partitioning rows is treated as LRC-shaped. This
 * covers rj_topo_lrc, rj_topo_lrc_azure, rj_topo_lrc_optimal, and any
 * future/custom topology with the shape. */

/* Returns 1 if ti's parity matrix has identifiable local rows (pure
 * XOR, disjoint supports, partitioning the data disks). Fills
 * group_of[d] (0..ngroups-1) for each data disk d, and local_row[g]
 * (parity-row index) per group. Returns 0 if not LRC-shaped. Pure;
 * no allocation. See hidden_files/lrc_locality_SPEC.md. */
int survivor_lrc_detect(const topo_int_t *ti, int *group_of,
                        int *local_row, int *ngroups_out)
{
    int nd = ti->ndata, np = ti->npar;
    const uint8_t *mat = ti->mat;
    if (!mat || nd < 1 || np < 1)
        return 0;
    /* A row is a local-row candidate if every coefficient is 0/1 and
     * at least one is 1. */
    int is_local[RJ_MAX_DISKS];
    int ngroups = 0;
    for (int j = 0; j < np; j++) {
        int binary = 1, any = 0;
        for (int i = 0; i < nd; i++) {
            uint8_t c = mat[(size_t)j * (size_t)nd + (size_t)i];
            if (c != 0 && c != 1)
                binary = 0;
            if (c)
                any = 1;
        }
        is_local[j] = (binary && any);
        ngroups += (binary && any);
    }
    if (ngroups == 0)
        return 0;
    /* Disjoint supports + partition: each data disk in exactly one
     * local row. */
    for (int i = 0; i < nd; i++)
        group_of[i] = -1;
    int g = 0;
    for (int j = 0; j < np; j++) {
        if (!is_local[j])
            continue;
        local_row[g] = j;
        for (int i = 0; i < nd; i++) {
            if (mat[(size_t)j * (size_t)nd + (size_t)i]) {
                if (group_of[i] != -1)
                    return 0; /* column in two local rows */
                group_of[i] = g;
            }
        }
        g++;
    }
    for (int i = 0; i < nd; i++)
        if (group_of[i] == -1)
            return 0; /* column in no local row */
    *ngroups_out = ngroups;
    return 1;
}

/* Returns 1 if the failure pattern F[0..nf) admits a pure-local
 * solution: every failed disk is covered by a distinct surviving local
 * row. Fills cover_rows[0..nf) with the covering local parity-row
 * indices (sorted ascending). Returns 0 otherwise (caller uses the
 * generic path). F is the validated sorted/deduped failed list;
 * group_of/local_row come from survivor_lrc_detect. */
int survivor_lrc_local_cover(const topo_int_t *ti, const int *F, int nf,
                             const int *group_of, const int *local_row,
                             int ngroups, int *cover_rows)
{
    int nd = ti->ndata, np = ti->npar;
    (void)ngroups;
    int ncover = 0;
    for (int f = 0; f < nf; f++) {
        int d = F[f];
        int row = -1;
        int needs_survive = 0;
        if (d < nd) {
            /* Failed data disk: its group's local row, which must
             * survive (not itself failed). */
            int g = group_of[d];
            if (g < 0)
                return 0;
            row = local_row[g];
            needs_survive = 1;
        } else if (d < nd + np) {
            /* Failed parity disk: if it's a local row, recompute from
             * its group (local, pure XOR); if it's a global row,
             * recompute from data (generic, not local). The failed
             * row itself is forced into the cover (survivor_enumerate
             * forces failed parity rows); it needs no survival check
             * -- it is the recompute target, not an equation source. */
            int pj = d - nd;
            int is_local_row = 0;
            for (int gg = 0; gg < ngroups; gg++)
                if (local_row[gg] == pj)
                    is_local_row = 1;
            if (!is_local_row)
                return 0; /* global parity: generic path */
            row = pj;
            needs_survive = 0;
        } else {
            return 0; /* out of range (shouldn't happen post-validation) */
        }
        /* Distinctness: one row per failure. */
        for (int k = 0; k < ncover; k++)
            if (cover_rows[k] == row)
                return 0; /* two failures, one local row */
        /* Survival: data-failure covering rows must not be failed. */
        if (needs_survive) {
            for (int k = 0; k < nf; k++)
                if (F[k] == nd + row)
                    return 0;
        }
        cover_rows[ncover++] = row;
    }
    /* Sort ascending (insertion; ncover <= nf, tiny). */
    for (int i = 1; i < ncover; i++) {
        int v = cover_rows[i], j = i - 1;
        while (j >= 0 && cover_rows[j] > v) {
            cover_rows[j + 1] = cover_rows[j];
            j--;
        }
        cover_rows[j + 1] = v;
    }
    return 1;
}

/* 1 if every element of rows[0..nrows) is in cover[0..ncover). Both
 * sorted ascending (cover_rows from survivor_lrc_local_cover is). */
static int subset_rows_in(const int *rows, int nrows, const int *cover,
                          int ncover)
{
    for (int i = 0; i < nrows; i++) {
        int found = 0;
        for (int j = 0; j < ncover; j++)
            if (rows[i] == cover[j]) {
                found = 1;
                break;
            }
        if (!found)
            return 0;
    }
    return 1;
}

int survivor_select(const topo_int_t *ti, const int *F, int nf,
                    const char *opname, const rj_cpu_info *cpu,
                    const rj_options *o, version_t **vs_out, int *nv_out,
                    plan_t **plans_out, int *nplans_out, int *win_rows_out,
                    char *err, size_t errlen)
{
    *vs_out = NULL;
    *nv_out = 0;
    *plans_out = NULL;
    *nplans_out = 0;

    survivor_subset_t *subs = NULL;
    int nsub = 0;
    int erc = survivor_enumerate(ti, F, nf, opname, &subs, &nsub, err,
                                 errlen);
    if (erc != 0)
        return erc; /* -1 error, -2 over-cap (caller falls back) */
    int *keep = malloc((size_t)nsub * sizeof(int));
    if (!keep) {
        survivor_free_subsets(subs, nsub);
        snprintf(err, errlen, "out of memory");
        return -1;
    }
    /* LRC locality fast path (planner rule, not cost-model accident).
     * If the topology is LRC-shaped and F admits a pure-local
     * solution, restrict to the local subsets directly: the local rows
     * are structurally cheaper (fewer streams, pure XOR vs
     * GF-multiply), so no cost-model competition is needed. This is
     * robust to benchmark noise and cost-model drift. Falls back to
     * the generic path if the local subsets can't be built. See
     * hidden_files/lrc_locality_SPEC.md. */
    int lrc_group_of[RJ_MAX_DISKS], lrc_local_row[RJ_MAX_DISKS];
    int lrc_ngroups = 0, lrc_cover[RJ_MAX_DISKS];
    int use_lrc = 0;
    if (survivor_lrc_detect(ti, lrc_group_of, lrc_local_row,
                            &lrc_ngroups)
        && survivor_lrc_local_cover(ti, F, nf, lrc_group_of,
                                     lrc_local_row, lrc_ngroups,
                                     lrc_cover)) {
        /* On success lrc_cover holds exactly nf sorted row indices
         * (one distinct covering local row per failure, by the
         * duplicate check in survivor_lrc_local_cover). */
        for (int i = 0; i < nsub; i++)
            keep[i] = subset_rows_in(subs[i].rows, subs[i].nrows,
                                     lrc_cover, nf);
        int any = 0;
        for (int i = 0; i < nsub; i++)
            any |= keep[i];
        use_lrc = any;
    }
    if (use_lrc) {
        /* Still run the L2 signal to verify buildability; drop local
         * subsets that fail to build. If none remain, fall back to
         * the generic shortlist. */
        survivor_l2_signal(subs, nsub, cpu, o);
        int any_valid = 0;
        for (int i = 0; i < nsub; i++) {
            if (keep[i] && !subs[i].l2_valid)
                keep[i] = 0;
            any_valid |= keep[i];
        }
        if (!any_valid) {
            survivor_shortlist(subs, nsub, keep);
            use_lrc = 0;
        }
    } else {
        /* L2 static signal (exactly one canonical version per subset),
         * then the shortlist. Only shortlisted subsets pay the full
         * candidate build + pooled bench below. */
        survivor_l2_signal(subs, nsub, cpu, o);
        survivor_shortlist(subs, nsub, keep);
    }
    int nkeep = 0;
    int today_si = -1;
    for (int i = 0; i < nsub; i++) {
        nkeep += keep[i];
        if (keep[i] && subs[i].is_today && today_si < 0)
            today_si = i;
    }

    /* Per kept subset: candidates -> versions -> A1 prune. The pooled
     * versions share one bench_select; each version's plan travels via
     * version_t.plan (set by jit_build_version) so validation uses the
     * right scalar reference.
     *
     * INVARIANT: plans[] is pre-sized to nkeep and never realloc'd
     * below. version_t.plan borrows &plans[k]; a realloc move would
     * dangle every outstanding borrow (this bit us: the exhaustive
     * RJ_NO_SUBSET_PRUNE path realloc'd plans and validation failed
     * against freed memory). */
    version_t *pool = NULL;
    int npool = 0, poolcap = 0;
    /* pool_sub[i]: the subset index (0..nsub-1) that produced pool
     * entry i. The pool is appended in subset order; the significance
     * gate uses this to score each subset from the measured data. */
    int *pool_sub = NULL;
    plan_t *plans = nkeep > 0 ? calloc((size_t)nkeep, sizeof(*plans)) : NULL;
    int nplans = 0;
    if (nkeep > 0 && !plans) {
        survivor_free_subsets(subs, nsub);
        free(keep);
        snprintf(err, errlen, "out of memory");
        return -1;
    }
    /* #29: optionally capture each kept plan's subset row set so the
     * caller can name the winning plan's rows in its stable ID.
     * plans[] is filled in kept-subset order, so kept_rows[k*nf ..] is
     * plans[k]'s row set. nf >= 1 was validated by survivor_enumerate. */
    int *kept_rows = NULL;
    if (win_rows_out && nkeep > 0) {
        kept_rows = malloc((size_t)nkeep * (size_t)nf * sizeof(int));
        if (!kept_rows) {
            survivor_free_subsets(subs, nsub);
            free(keep);
            for (int i = 0; i < nplans; i++)
                plan_free(&plans[i]);
            free(plans);
            snprintf(err, errlen, "out of memory");
            return -1;
        }
    }
    int ok = 1;
    int si = 0; /* hoisted for the !ok cleanup below (P2.1) */
    for (si = 0; si < nsub && ok; si++) {
        if (!keep[si]) {
            plan_free(&subs[si].plan);
            free(subs[si].rows);
            subs[si].rows = NULL;
            continue;
        }
        /* move the plan into the pre-sized plans array */
        plans[nplans] = subs[si].plan; /* struct copy; owned by plans */
        memset(&subs[si].plan, 0, sizeof(plans[nplans]));
        plan_t *pl = &plans[nplans];
        nplans++;
        if (kept_rows)
            memcpy(&kept_rows[(size_t)(nplans - 1) * (size_t)nf],
                   subs[si].rows, (size_t)nf * sizeof(int));

        cand_t cands[512];
        int ncand = plan_candidates(pl, cpu, cands, 512,
                                    o->nt_threshold > 0 && o->benchmark,
                                    prefetch_calib_dist_arg(o));
        version_t *vs = NULL;
        int nv = 0;
        if (ncand > 0) {
            /* Test-only hook: RJ_SURVIVOR_FAIL_VS_ALLOC forces this
             * calloc to fail, exercising the !ok cleanup below (P2.1
             * regression test). Never set in production. */
            vs = getenv("RJ_SURVIVOR_FAIL_VS_ALLOC")
                     ? NULL
                     : calloc((size_t)ncand, sizeof(*vs));
            if (!vs) {
                ok = 0;
                break;
            }
            for (int i = 0; i < ncand; i++) {
                char e2[256];
                if (jit_build_version(pl, &cands[i], cpu, o, &vs[nv], e2,
                                      sizeof(e2)) == 0) {
                    nv++;
                } else if (o->verbose) {
                    fprintf(stderr, "raidjit: candidate build failed: %s\n",
                            e2);
                }
            }
        }
        if (nv > 0 && o->benchmark) {
            int npruned = nv;
            nv = prune_versions(vs, nv, o->verbose ? PRUNE_VERBOSE : 0);
            npruned -= nv;
            if (o->verbose && npruned > 0)
                fprintf(stderr,
                        "raidjit: prune: dropped %d of %d dominated "
                        "candidates\n",
                        npruned, npruned + nv);
        }
        /* append survivors to the pool */
        if (npool + nv > poolcap) {
            int nc = poolcap ? poolcap * 2 : 64;
            while (nc < npool + nv)
                nc *= 2;
            version_t *np2 = realloc(pool, (size_t)nc * sizeof(*np2));
            int *ns2 = realloc(pool_sub, (size_t)nc * sizeof(*ns2));
            if (!np2 || !ns2) {
                /* On realloc failure the old blocks are still valid;
                 * adopt whichever half succeeded so the !ok cleanup
                 * below frees a consistent pair. */
                if (np2)
                    pool = np2;
                if (ns2)
                    pool_sub = ns2;
                for (int i = 0; i < nv; i++)
                    jit_free_version(&vs[i]);
                free(vs);
                ok = 0;
                break;
            }
            pool = np2;
            pool_sub = ns2;
            poolcap = nc;
        }
        for (int i = 0; i < nv; i++) {
            pool[npool] = vs[i]; /* struct copy; code owned by pool */
            pool_sub[npool] = si;
            npool++;
        }
        free(vs);
        free(subs[si].rows);
        subs[si].rows = NULL;
    }
    if (!ok) {
        /* P2.1: the break left subsets [si, nsub) unprocessed. Subset
         * si was kept -- both break sites are past its plan move into
         * plans[] (its subs[] plan slot is zeroed) -- so it owns only
         * rows; subsets (si, nsub) still own both plan and rows.
         * Subsets [0, si) were already moved (plans[]) or freed. */
        for (int i = si; i < nsub; i++) {
            free(subs[i].rows);
            if (i > si)
                plan_free(&subs[i].plan);
        }
    }
    free(subs);
    free(keep);
    if (!ok) {
        for (int i = 0; i < npool; i++)
            jit_free_version(&pool[i]);
        free(pool);
        free(pool_sub);
        for (int i = 0; i < nplans; i++)
            plan_free(&plans[i]);
        free(plans);
        free(kept_rows);
        snprintf(err, errlen, "out of memory");
        return -1;
    }
    if (npool == 0) {
        free(pool_sub);
        for (int i = 0; i < nplans; i++)
            plan_free(&plans[i]);
        free(plans);
        free(kept_rows);
        snprintf(err, errlen, "no code version could be built");
        return -1;
    }
    int winner;
    if (o->benchmark) {
        /* Capture bench_select's measured noise window for the
         * significance gate below. */
        double noise_window = 0;
        winner = bench_select(pool, npool, &plans[0], o, o->verbose,
                              &noise_window, err, errlen);
        if (winner >= 0) {
            /* P2: never measurably worse than today. If the winning
             * subset is not today's and its lead over today's best
             * measured GB/s is within the noise window, fall back to
             * today's best measured version. Skipped when the anchor
             * is absent from the pool (only under the
             * RJ_SUBSET_PRUNE_INVERT test hook).
             *
             * Test-only hook: RJ_SURVIVOR_NO_GATE (any value, even
             * empty) skips the gate and returns bench_select's raw
             * winner. The L2 tripwire (test_survivor Part 7) uses it
             * to compare the raw measured winners: the gate is a
             * post-measurement policy unit-tested separately, while
             * the tripwire is the L2-safety instrument ("no pruned
             * subset would have won measured"). Never set in
             * production. */
            int no_gate = getenv("RJ_SURVIVOR_NO_GATE") != NULL;
            int gated = no_gate ? winner : survivor_significance_gate(
                pool, pool_sub, npool, today_si, winner, noise_window);
            if (gated != winner && o->verbose)
                fprintf(stderr,
                        "raidjit: survivor: subset winner within noise "
                        "(window=%.2f%%); falling back to today's subset\n",
                        noise_window * 100.0);
            winner = gated;
        }
    } else {
        /* No benchmarking: static pick over the pool. The subset
         * dimension is then decided statically (documented); the
         * measured path above is the one that matters. No measured
         * noise window exists here, so the significance gate does not
         * apply. */
        winner = heuristic_select(pool, npool, &plans[0]);
    }
    /* #29: report the winning plan's equation row set. The winner's
     * version borrows its plan from plans[]; kept_rows[k] is plans[k]'s
     * subset row set. */
    if (win_rows_out && winner >= 0) {
        int k = -1;
        for (int i = 0; i < nplans; i++)
            if (pool[winner].plan == &plans[i]) {
                k = i;
                break;
            }
        if (k < 0) {
            /* Unreachable by construction (every pooled version borrows
             * its plan from plans[]); fail loudly rather than handing
             * out a garbage row set. */
            free(pool_sub);
            for (int i = 0; i < npool; i++)
                jit_free_version(&pool[i]);
            free(pool);
            for (int i = 0; i < nplans; i++)
                plan_free(&plans[i]);
            free(plans);
            free(kept_rows);
            snprintf(err, errlen,
                     "survivor: internal error (winner plan not in "
                     "plans[])");
            return -1;
        }
        memcpy(win_rows_out, &kept_rows[(size_t)k * (size_t)nf],
               (size_t)nf * sizeof(int));
    }
    free(kept_rows);
    free(pool_sub);
    if (winner < 0) {
        for (int i = 0; i < npool; i++)
            jit_free_version(&pool[i]);
        free(pool);
        for (int i = 0; i < nplans; i++)
            plan_free(&plans[i]);
        free(plans);
        return -1;
    }
    *vs_out = pool;
    *nv_out = npool;
    *plans_out = plans;
    *nplans_out = nplans;
    return winner;
}
