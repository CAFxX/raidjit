/* Public API: rj_compile, rj_get_fn, rj_get_info, teardown. */
#include "../include/raidjit.h"
#include "plan.h"
#include "jit.h"
#include "bench.h"
#include "prune.h"
#include "cpu.h"

#include "dispatch.h"
#include "survivor.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

struct rj_jit {
    version_t v;        /* general version: always present */
    version_t twins[2]; /* tail twins (notail + predtail on SVE,
                         * notail + maskedtail on AVX-512): valid iff
                         * has_dispatch. At most 2: the ISA-specific
                         * twins are mutually exclusive. */
    int ntwins;
    int has_dispatch;
    rj_dispatcher *disp;
};

/* ---------------- options ---------------- */

void rj_options_default(rj_options *o)
{
    memset(o, 0, sizeof(*o));
    o->benchmark = 1;
    o->bench_bytes = 1 << 20;
    o->bench_iters = 7;
    o->loop_align = 32;
    o->verbose = 0;
    o->nt_threshold = 0;
    o->prefetch_distance = 0;
}

/* ---------------- compile ---------------- */

/* One tail twin's install gate (SVE predtail, track B1sve; AVX-512
 * maskedtail, track B1x86); see src/api.h. The twin is built by the
 * caller from the shootout winner; the duel pits it against the general
 * (scalar-tail) version at tail-heavy lengths. */
int api_tail_duel_gate(version_t *tw, version_t *general, const plan_t *wpl,
                       const rj_options *o, int verbose, version_t *twins,
                       int *ntwins, tail_duel_fn duel, const char *name,
                       char *err, size_t errlen)
{
    int r = duel(tw, general, wpl, o, verbose, err, errlen);
    if (r < 0) {
        /* Validation or benchmark failure: wrong code must never ship
         * quietly (same rule as the shootout). */
        jit_free_version(tw);
        return -2;
    }
    if (r == 0) {
        if (verbose)
            fprintf(stderr, "raidjit: %s twin lost its tail duel, dropped\n",
                    name);
        jit_free_version(tw);
        return -1;
    }
    int idx = *ntwins;
    twins[idx] = *tw; /* shallow; owned by twins[] */
    (*ntwins)++;
    return idx;
}

/* Track B2 (SOTA item 15): eligibility for the stage-2 permi2b duel;
 * see src/api.h. The twin is duelled only when it can actually differ
 * from the plain path:
 *  - AVX-512 winner with mul == MUL_NIBBLE on a P_MULACC/P_DELTA plan
 *    (the vpermi2b lookup branch lives in the x86-64 mulacc_column);
 *  - host reports VBMI (the emitter faults without it);
 *  - the plan has at least one nontrivial coefficient (B2 P1-2): the
 *    permi2b branch fires only for coefficients outside {0,1}, so on
 *    an all-trivial plan (reachable: P_DELTA with an all-{0,1} delta
 *    matrix) the twin is byte-identical to plain and the duel can only
 *    install noise under a misleading +permi2b desc. plan_any_nontrivial
 *    (src/plan.c) is the single source of truth for the last condition
 *    -- the same definition the backend branches on. */
int api_permi2b_duel_eligible(const version_t *winner, const plan_t *wpl,
                              const rj_cpu_info *cpu)
{
    return winner->cand.isa == ISA_AVX512 &&
           (wpl->kind == P_MULACC || wpl->kind == P_DELTA) &&
           winner->cand.mul == MUL_NIBBLE &&
           !winner->cand.permi2b && cpu->vbmi &&
           plan_any_nontrivial(wpl);
}

rj_jit *rj_compile(const rj_topology *topo, const rj_op *op,
                   const rj_options *opts, char *err, size_t errlen)
{
    if (!topo || !op) {
        if (err && errlen)
            snprintf(err, errlen, "null topology or op");
        return NULL;
    }
    rj_options o;
    if (opts)
        o = *opts;
    else
        rj_options_default(&o);

    /* loop_align feeds em_align's mask arithmetic, which requires a power
     * of two (0 = auto). Reject anything else with a clean error. */
    if (o.loop_align < 0 ||
        (o.loop_align != 0 && (o.loop_align & (o.loop_align - 1)) != 0)) {
        if (err && errlen)
            snprintf(err, errlen,
                     "loop_align must be 0 or a power of two, got %d",
                     o.loop_align);
        return NULL;
    }

    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);

    topo_int_t ti;
    if (topo_int_build(&ti, topo, err, errlen) != 0)
        return NULL;

    plan_t pl; /* single-plan path (encode / parity_update) */
    const char *opname = topo_internal_name(topo);

    /* Candidate versions and the winning plan. Recover ops go through
     * A5 survivor-set selection (the parity-row subset is a candidate
     * dimension); every other op uses the single-plan path. */
    version_t *vs = NULL;
    int nv = 0;
    int winner = -1;
    plan_t *surv_plans = NULL; /* survivor path only: owned plans */
    int n_surv_plans = 0;
    const plan_t *wpl = NULL; /* winner's plan (borrowed) */

    int *F = NULL;
    int use_survivor = 0;
    if (op->is_recover && !op->is_parity_update) {
        int nf = plan_recover_failed_list(op, ti.ndata, ti.npar, &F, err,
                                         errlen);
        if (nf < 0) {
            topo_int_free(&ti);
            return NULL;
        }
        /* RJ_NO_SURVIVOR_SELECT (non-empty and != "0", the A1
         * RJ_NO_PRUNE contract): public disable knob for A5
         * survivor-set selection. Set it to take the legacy
         * single-subset (all-rows) path below instead of enumerating
         * candidate parity-row subsets. Over-cap enumeration likewise
         * falls back to the single all-rows plan (today's behavior)
         * instead of failing the compile. */
        const char *nss = getenv("RJ_NO_SURVIVOR_SELECT");
        int no_survivor =
            nss && nss[0] != '\0' && strcmp(nss, "0") != 0;
        use_survivor = !no_survivor && !survivor_over_cap(&ti, F, nf);
        if (use_survivor) {
            winner = survivor_select(&ti, F, nf, opname, &cpu, &o, &vs,
                                     &nv, &surv_plans, &n_surv_plans, err,
                                     errlen);
            if (winner == -2)
                use_survivor = 0; /* cap raced: legacy fallback */
            else if (winner < 0) {
                free(F);
                topo_int_free(&ti);
                return NULL;
            } else
                wpl = vs[winner].plan;
        }
        free(F);
    }
    if (!use_survivor) {
        if (plan_build(&ti, op, opname, &pl, err, errlen) != 0) {
            topo_int_free(&ti);
            return NULL;
        }

        /* Candidate buffer. Worst case today: x86 P_MULACC/P_DELTA with 8
         * U-values each: AVX-512 GFNI + nibble (16), AVX2 nibble (8), SSE
         * nibble (8) = 32 base; +8 AVX-512 nibble ternlog twins = 40;
         * x2 for NT twins (80), x2 for prefetch twins (160). 512 leaves
         * ample headroom for future candidate dimensions; plan_candidates
         * still guards with maxout so a smaller buffer would truncate
         * silently. */
        cand_t cands[512];
        /* NT/prefetch twins only when benchmarking: on the static model they
         * tie with their temporal originals and lose the tie-break, so
         * generating them would only multiply compile time. */
        int ncand = plan_candidates(&pl, &cpu, cands, 512,
                                    o.nt_threshold > 0 && o.benchmark,
                                    o.benchmark ? o.prefetch_distance : 0);
        if (ncand <= 0) {
            if (err && errlen)
                snprintf(err, errlen, "no candidates for this CPU/plan");
            plan_free(&pl);
            topo_int_free(&ti);
            return NULL;
        }

        vs = calloc((size_t)ncand, sizeof(*vs));
        if (!vs) {
            if (err && errlen)
                snprintf(err, errlen, "out of memory");
            plan_free(&pl);
            topo_int_free(&ti);
            return NULL;
        }
        for (int i = 0; i < ncand; i++) {
            char e2[256];
            if (jit_build_version(&pl, &cands[i], &cpu, &o, &vs[nv], e2,
                                  sizeof(e2)) == 0) {
                nv++;
            } else if (o.verbose) {
                fprintf(stderr, "raidjit: candidate build failed: %s\n",
                        e2);
            }
        }
        if (nv == 0) {
            if (err && errlen)
                snprintf(err, errlen, "no code version could be built");
            goto fail;
        }

            if (o.benchmark) {
                /* Static-dominance pruning before wall-clock benchmarking.
                 * prune_versions() drops ONLY the plain (ternlog=0) member of
                 * a +ternlog twin pair when the twin (same plan kind, isa,
                 * mul, U, nt, prefetch_dist, remat) pareto-dominates it on the
                 * static (uops/B, mem/B, code) triple -- and only for plan
                 * kinds P_MULACC / P_DELTA, where the twin direction is
                 * validated (see prune.h and docs/candidate_pruning.md).
                 * Pruned versions are finalized inside prune_versions; only
                 * survivors reach bench_select. Disabled when the RJ_NO_PRUNE
                 * environment variable is set to any non-empty value other
                 * than exactly "0", or via the PRUNE_DISABLE flag (see
                 * prune.h). No pruning on the static (non-benchmark) path:
                 * nothing is benchmarked there, so pruning would save no wall
                 * time and could only remove candidates for no benefit. */
                int npruned = nv;
                nv = prune_versions(vs, nv, o.verbose ? PRUNE_VERBOSE : 0);
                npruned -= nv;
                if (o.verbose && npruned > 0)
                    fprintf(stderr, "raidjit: prune: dropped %d of %d dominated "
                                    "candidates\n",
                            npruned, npruned + nv);
                winner = bench_select(vs, nv, &pl, &o, o.verbose, NULL, err, errlen);
            } else {
                winner = heuristic_select(vs, nv, &pl);
            }
        if (winner < 0)
            goto fail;
        wpl = &pl;
    }
    if (winner < 0)
        goto fail;

    /* Stage 2: length-dispatch shootout (docs/length_dispatch.md). Build
     * the tail twins of the winning candidate -- the notail twin
     * (identical vector body, no tail code) on every ISA, plus the
     * predicated-tail twin (SVE whilelt tail, track B1sve) for SVE
     * winners and the masked-tail twin (AVX-512 k1 tail, track B1x86)
     * for eligible AVX-512 winners -- and run {general, notail} through
     * the same benchmark protocol. Before the tail twins, the VPERMI2B
     * twin (track B2, SOTA item 15) is duelled against the winner when
     * the winner is an AVX-512 MUL_NIBBLE MULACC/DELTA candidate on a
     * VBMI host; a strict win outside the noise window swaps it in as
     * the winner the tail twins are built from. The predicated/masked tail twins are
     * NOT judged by the shootout: at its large sizes the tail is a
     * negligible fraction, so the standard protocol can neither install
     * nor reject them meaningfully. Each is gated by its own tail duel
     * (bench_tail_duel, ISA-agnostic) instead: tail-heavy lengths, a
     * strict win outside the noise window, plus a large-size
     * no-regression gate. Under length-aware routing an installed tail
     * twin deterministically owns 100% of tail lengths, so a
     * large-size-only win is not sufficient evidence to install it. The
     * dispatcher is installed if a shootout twin wins or a tail twin
     * wins its duel; any twin build failure degrades silently to
     * general-only, but a twin that builds and then fails validation
     * fails compilation (wrong code must never ship quietly). No-op plans
     * have no tail to remove, and the heuristic (benchmark=0) path has no
     * measurement to gate on: both skip the shootout. */
    rj_dispatcher *disp = NULL;
    version_t twins[2];
    memset(twins, 0, sizeof(twins));
    int ntwins = 0;
    /* The twins are built from the WINNER's plan: on the survivor
     * path that is the winning subset's plan (wpl), so the chosen set
     * bakes into every dispatched variant. Dispatcher ABI unchanged. */
    if (o.benchmark && !wpl->noop) {
        char e3[256];
        /* VPERMI2B twin (track B2, SOTA item 15): built from the stage-1
         * winner, NOT enumerated in stage-1 (plan_candidates never sets
         * permi2b). Enumerating it in stage-1 cost +50% stage-1 builds
         * on MULACC/DELTA (192 vs 128 candidates on custom(4,2) encode,
         * 2026-10-03) while the twin never won a duel (0/12 on AMD EPYC
         * 9D25; REJECTED-WITH-EVIDENCE, host-scoped). The empirical
         * re-decision on other microarchitectures is kept: the twin is
         * duelled against the winner with the standard bench_select
         * protocol (noise window + uops/byte tie-break), so it installs
         * only on a strict measured win outside the noise window -- the
         * pre-registered B2 gate, moved from stage-1 to stage-2. Cost:
         * +1 build + 1 duel, only when the winner is a nibble candidate.
         * Eligibility (api_permi2b_duel_eligible, unit-tested): (AVX-512,
         * MUL_NIBBLE, MULACC/DELTA, VBMI) plus at least one nontrivial
         * plan coefficient -- on an all-trivial plan the twin is
         * byte-identical to plain and the duel is vacuous (B2 P1-2).
         * jit_build_version asserts the permi2b/AVX-512/MUL_NIBBLE/
         * MULACC|DELTA shape. The swap happens before the tail twins
         * below are built, so they twin the final (possibly permi2b)
         * winner -- the permi2b twin composes with the tail twins
         * exactly like the plain nibble path (track B2 §2; maskedtail
         * is P_XOR-only so it never meets permi2b). */
        if (api_permi2b_duel_eligible(&vs[winner], wpl, &cpu)) {
            cand_t p2c = vs[winner].cand;
            p2c.permi2b = 1; /* twin of the winner, VPERMI2B lookup */
            version_t ptwin;
            memset(&ptwin, 0, sizeof(ptwin));
            if (jit_build_version(wpl, &p2c, &cpu, &o, &ptwin, e3,
                                  sizeof(e3)) == 0) {
                version_t pair2[2];
                pair2[0] = vs[winner]; /* shallow; still owned by vs[] */
                pair2[1] = ptwin;      /* shallow; owned by ptwin below */
                int w3 = bench_select(pair2, 2, wpl, &o, o.verbose, NULL,
                                      err, errlen);
                if (w3 < 0) {
                    /* Validation or benchmark failure: wrong code must
                     * never ship quietly (same rule as the shootout). */
                    jit_free_version(&ptwin);
                    goto fail;
                }
                if (w3 == 1) {
                    if (o.verbose)
                        fprintf(stderr,
                                "raidjit: permi2b twin won its duel: %s\n",
                                ptwin.desc);
                    /* The twin becomes the winner: j->v below, and the
                     * tail twins are built from its candidate. Ties keep
                     * the plain path: the twin loses the uops/byte
                     * tie-break (0.6406 vs 0.5938 on custom(4,2)).
                     * bench_select measured into the local pair2 copies;
                     * copy the measurements back into the twin before
                     * installing it (B2 P1-1) -- otherwise vs[winner]
                     * keeps jit_build_version's gbps=-1 sentinel /
                     * gbps_measured=0 and rj_get_info reports the
                     * sentinel on exactly the hosts the twin is kept
                     * for. */
                    ptwin.gbps = pair2[1].gbps;
                    ptwin.gbps_measured = pair2[1].gbps_measured;
                    jit_free_version(&vs[winner]);
                    vs[winner] = ptwin;
                } else {
                    if (o.verbose)
                        fprintf(stderr,
                                "raidjit: permi2b twin lost its duel, "
                                "dropped\n");
                    jit_free_version(&ptwin);
                }
            } else if (o.verbose) {
                fprintf(stderr, "raidjit: permi2b twin build failed: %s\n",
                        e3);
            }
        }
        /* NOTE: winner indexes vs[], which skips failed builds, so it
         * must NOT index cands[]. The winner's own candidate travels in
         * vs[winner].cand. */
        cand_t twinc = vs[winner].cand;
        twinc.notail = 1; /* twin of the winner, tail path omitted */
        if (jit_build_version(wpl, &twinc, &cpu, &o, &twins[ntwins], e3,
                              sizeof(e3)) == 0) {
            ntwins++;
        } else if (o.verbose) {
            fprintf(stderr, "raidjit: notail twin build failed: %s\n", e3);
        }
        /* Tail twins with their own duel gate -- NOT the large-size
         * shootout below (at the shootout's large sizes the tail is a
         * negligible fraction, so the standard protocol can neither
         * install nor reject a tail twin meaningfully):
         *
         * - Predicated-tail twin: SVE winners only (the whilelt tail
         *   lives in the AArch64 SVE backend, track B1sve). Disjoint
         *   from the notail twin: ptwinc.notail stays 0 here;
         *   jit_build_version asserts they are never combined.
         * - Masked-tail twin: AVX-512 P_XOR winners only (track B1x86).
         *   The masked AVX-512 tail (k1, vmovdqu8) replaces the scalar
         *   tail loop. Eligibility: P_XOR, AVX-512, not ternlog/xorsched
         *   (separate bodies), not nt (masked-NT tail stores are out of
         *   scope, track Bx86 DESIGN.md §1.4), CPU has AVX-512F+BW.
         *   Disjoint from notail; never combined with predtail
         *   (different ISA).
         *
         * Each twin is built from the winner and gated by
         * api_tail_duel_gate (bench_tail_duel, ISA-agnostic): install
         * only on a strict tail win outside the noise window plus a
         * large-size no-regression gate. A duel loss drops the twin
         * (tail traffic falls back to the scalar tail); a duel error
         * fails compilation. The two twins are mutually exclusive by
         * ISA (ISA_SVE vs ISA_AVX512 on the same winner), so at most
         * one duel_idx is ever set. */
        int duel_idx = -1; /* index into twins[] of the duel-installed tail
                            * twin, or -1 when no duel was won */
        if (vs[winner].cand.isa == ISA_SVE) {
            cand_t ptwinc = vs[winner].cand;
            ptwinc.predtail = 1; /* twin of the winner, predicated tail */
            version_t ptwin;
            memset(&ptwin, 0, sizeof(ptwin));
            if (jit_build_version(wpl, &ptwinc, &cpu, &o, &ptwin, e3,
                                  sizeof(e3)) == 0) {
                int g = api_tail_duel_gate(&ptwin, &vs[winner], wpl, &o,
                                           o.verbose, twins, &ntwins,
                                           bench_tail_duel, "predtail",
                                           err, errlen);
                if (g == -2) {
                    for (int t = 0; t < ntwins; t++)
                        jit_free_version(&twins[t]);
                    goto fail;
                }
                if (g >= 0)
                    duel_idx = g;
            } else if (o.verbose) {
                fprintf(stderr, "raidjit: predtail twin build failed: %s\n",
                        e3);
            }
        }
        if (vs[winner].cand.isa == ISA_AVX512 &&
            wpl->kind == P_XOR &&
            !vs[winner].cand.ternlog &&
            !vs[winner].cand.xorsched &&
            !vs[winner].cand.nt &&
            cpu.avx512f && cpu.avx512bw) {
            cand_t mtwinc = vs[winner].cand;
            mtwinc.maskedtail = 1;
            version_t mtwin;
            memset(&mtwin, 0, sizeof(mtwin));
            if (jit_build_version(wpl, &mtwinc, &cpu, &o, &mtwin, e3,
                                  sizeof(e3)) == 0) {
                int g = api_tail_duel_gate(&mtwin, &vs[winner], wpl, &o,
                                           o.verbose, twins, &ntwins,
                                           bench_tail_duel, "maskedtail",
                                           err, errlen);
                if (g == -2) {
                    for (int t = 0; t < ntwins; t++)
                        jit_free_version(&twins[t]);
                    goto fail;
                }
                if (g >= 0)
                    duel_idx = g;
            } else if (o.verbose) {
                fprintf(stderr, "raidjit: maskedtail twin build failed: %s\n",
                        e3);
            }
        }
        if (ntwins > 0) {
            /* The large-size shootout covers {general, ...twins} EXCEPT
             * the duel-gated tail twin (predtail/maskedtail): it has its
             * own tail-duel gate above (its install decision is already
             * made). npair >= 1 always. */
            version_t pair[3];
            int npair = 0;
            pair[npair++] = vs[winner]; /* shallow copy; still owned by vs[] */
            for (int t = 0; t < ntwins; t++)
                if (t != duel_idx)
                    pair[npair++] = twins[t]; /* shallow; owned by twins[] */
            int w2 = 0; /* general "wins" the empty shootout */
            if (npair > 1) {
                w2 = bench_select(pair, npair, wpl, &o, o.verbose,
                                  NULL, err, errlen);
                if (w2 < 0) {
                    for (int t = 0; t < ntwins; t++)
                        jit_free_version(&twins[t]);
                    goto fail;
                }
            }
            if (w2 != 0 || duel_idx >= 0) {
                /* Install the dispatcher with the general variant plus
                 * every installed twin, fastest-first. A shootout twin
                 * that won (w2 != 0) earns its place at large sizes; a
                 * tail twin that won its duel (duel_idx >= 0) earns tail
                 * lengths -- it was measured at tail-heavy lengths, which
                 * are the only lengths it will serve (TAIL_MASKED admits
                 * only tail lengths; the general variant defers to it --
                 * length-aware routing, dispatch.h). Every installed twin
                 * earned its lengths by measurement, so no installed twin
                 * is slower than general at the lengths it owns -- the
                 * two-variant case behaves exactly as before. */
                rj_variant vars[3];
                memset(vars, 0, sizeof(vars));
                int nvars = 0;
                int pidx = 1; /* pair index of the next non-duel twin */
                for (int t = 0; t < ntwins; t++) {
                    vars[nvars].fn = twins[t].fn;
                    /* Twin kind -> tail policy. twins[] only ever holds
                     * the three kinds built above: notail twins
                     * physically have no tail code (TAIL_NONE: multiples
                     * only); predtail (B1sve) and maskedtail (B1x86) twins
                     * own tail lengths (TAIL_MASKED). The mapping is
                     * explicit per kind: a future twin kind that sets
                     * none of the three flags trips the assert below
                     * instead of silently inheriting TAIL_MASKED (which
                     * would hand it all tail traffic unmeasured). */
                    assert(twins[t].notail || twins[t].predtail ||
                           twins[t].maskedtail);
                    vars[nvars].tail = twins[t].notail ? TAIL_NONE
                                                      : TAIL_MASKED;
                    vars[nvars].group = twins[t].group_bytes;
                    /* bench_select measured into the pair copies; the
                     * duel-gated twin's gbps came from its tail duel. */
                    vars[nvars].gbps = (t == duel_idx)
                                           ? twins[t].gbps
                                           : pair[pidx++].gbps;
                    snprintf(vars[nvars].desc, sizeof(vars[nvars].desc),
                             "%s", twins[t].desc);
                    rj_variant_derive(&vars[nvars]);
                    nvars++;
                }
                vars[nvars].fn = vs[winner].fn;
                vars[nvars].tail = TAIL_SCALAR;
                vars[nvars].group = vs[winner].group_bytes;
                vars[nvars].gbps = pair[0].gbps;
                snprintf(vars[nvars].desc, sizeof(vars[nvars].desc), "%s",
                         vs[winner].desc);
                rj_variant_derive(&vars[nvars]);
                nvars++;
                rj_variant_sort(vars, nvars);
                disp = rj_dispatcher_create(vars, nvars);
                /* Slot exhaustion degrades to general-only: the twins are
                 * freed below via ntwins. */
            }
        }
    }

    rj_jit *j = calloc(1, sizeof(*j));
    if (!j) {
        if (err && errlen)
            snprintf(err, errlen, "out of memory");
        for (int t = 0; t < ntwins; t++)
            jit_free_version(&twins[t]);
        rj_dispatcher_destroy(disp);
        goto fail;
    }
    j->v = vs[winner];
    j->v.plan = NULL; /* drop the compile-time borrow: the plan is freed
                       * below, and nothing may dereference it after
                       * compilation (see jit.h) */
    if (disp) {
        for (int t = 0; t < ntwins; t++) {
            j->twins[t] = twins[t];
            j->twins[t].plan = NULL; /* same borrow discipline as j->v */
        }
        j->ntwins = ntwins;
        j->has_dispatch = 1;
        j->disp = disp;
        ntwins = 0; /* now owned by j->twins */
    }
    for (int t = 0; t < ntwins; t++)
        jit_free_version(&twins[t]);

    for (int i = 0; i < nv; i++)
        if (i != winner)
            jit_free_version(&vs[i]);
    free(vs);
    if (surv_plans) {
        for (int i = 0; i < n_surv_plans; i++)
            plan_free(&surv_plans[i]);
        free(surv_plans);
    } else {
        plan_free(&pl);
    }
    topo_int_free(&ti);
    return j;

fail:
    for (int i = 0; i < nv; i++)
        jit_free_version(&vs[i]);
    free(vs);
    if (surv_plans) {
        for (int i = 0; i < n_surv_plans; i++)
            plan_free(&surv_plans[i]);
        free(surv_plans);
    } else {
        plan_free(&pl);
    }
    topo_int_free(&ti);
    return NULL;
}

rj_fn rj_get_fn(const rj_jit *j)
{
    if (!j)
        return NULL;
    /* Length dispatch (docs/length_dispatch.md): when a tail twin won
     * its install gate (the large-size shootout for the notail twin,
     * the tail duel for a predicated/masked tail twin), calls route
     * through the portable C dispatcher, which has the rj_fn signature.
     * Otherwise the general version is returned exactly as before. */
    if (j->disp)
        return rj_dispatcher_fn(j->disp);
    return j->v.fn;
}

void rj_free(rj_jit *j)
{
    if (j) {
        rj_dispatcher_destroy(j->disp);
        for (int t = 0; t < j->ntwins; t++)
            jit_free_version(&j->twins[t]);
        jit_free_version(&j->v);
        free(j);
    }
}

void rj_get_info(const rj_jit *j, rj_info *info)
{
    memset(info, 0, sizeof(*info));
    if (!j)
        return;
    snprintf(info->desc, sizeof(info->desc), "%s", j->v.desc);
    if (j->has_dispatch) {
        /* The dispatcher is installed: at least one tail twin won its
         * gate. Make every installed twin observable. */
        for (int t = 0; t < j->ntwins; t++) {
            size_t n = strlen(info->desc);
            if (j->twins[t].notail)
                snprintf(info->desc + n, sizeof(info->desc) - n,
                         " +notail");
            else if (j->twins[t].predtail)
                snprintf(info->desc + n, sizeof(info->desc) - n,
                         " +predtail");
            else if (j->twins[t].maskedtail)
                snprintf(info->desc + n, sizeof(info->desc) - n,
                         " +maskedtail");
        }
    }
    snprintf(info->isa_name, sizeof(info->isa_name), "%s",
             isa_name(j->v.cand.isa));
    snprintf(info->strategy, sizeof(info->strategy), "%s",
             j->v.kind == P_XOR    ? "xor"
             : j->v.kind == P_HORNER ? "horner"
             : j->v.kind == P_DELTA  ? "delta"
                                    : "mulacc");
    info->unroll = j->v.cand.U;
    info->lmul = j->v.cand.lmul ? j->v.cand.lmul : 1;
    info->gbps_measured = j->v.gbps_measured;
    info->gbps = j->v.gbps;
    info->uops_per_byte = j->v.uops_per_byte;
    info->mem_per_byte = j->v.mem_per_byte;
    info->code_size = j->v.code_size;
    info->const_size = j->v.const_size;
}
