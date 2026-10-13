/* Public API: rj_compile, rj_get_fn, rj_get_info, teardown. */
#include "../include/raidjit.h"
#include "plan.h"
#include "jit.h"
#include "bench.h"
#include "prune.h"
#include "cpu.h"

#include "dispatch.h"
#include "prefetch_calib.h"
#include "survivor.h"
/* Item #9's batched driver (src/api_batched.c). Per the #29 fixup we do
 * NOT include api_batched.h here: the incomplete struct type plus these
 * prototypes are the whole coupling. */
struct rj_batch;
struct rj_batch *rj_batch_create(rj_fn fn, size_t nbufs_in,
                                 size_t nbufs_out);
rj_fn_batched rj_batch_fn(struct rj_batch *b);
void rj_batch_destroy(struct rj_batch *b);
#include "planid.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

struct rj_jit {
    version_t v;        /* general version: always present */
    version_t twins[2]; /* tail twins (notail + predtail on SVE,
                         * notail + maskedtail on AVX-512/RVV): valid iff
                         * has_dispatch. At most 2: the ISA-specific
                         * twins are mutually exclusive. */
    int ntwins;
    int has_dispatch;
    rj_dispatcher *disp;
    struct rj_batch *batch; /* Tier-1 batched driver state (item #9);
                             * entry point via rj_get_fn_batched */
    char plan_id[RJ_PLAN_ID_MAX]; /* stable plan ID (#29): the winning
                                  * candidate spec + the effective
                                  * stage-2 tail configuration, derived
                                  * at compile time while the
                                  * topology/op/plan are alive. */
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

/* ---------------- init ---------------- */

/* First-touch process initialization (I4): runs the prefetch
 * calibration probe (~10 ms, once per process; later calls are free).
 * Optional: the probe also runs lazily at the first rj_compile that
 * needs an analytical prefetch distance, so calling rj_init is never
 * required for correctness -- it just moves the probe cost out of the
 * first compile. */
void rj_init(void)
{
    prefetch_calib_ensure();
}

/* ---------------- compile ---------------- */

/* One tail twin's install gate (SVE predtail, track B1sve; AVX-512
 * maskedtail, track B1x86; RVV maskedtail); see src/api.h. The twin is
 * built by the caller from the shootout winner; the duel pits it
 * against the general (scalar-tail) version at tail-heavy lengths. */
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

static rj_jit *rj_compile_impl(const rj_topology *topo, const rj_op *op,
                                const rj_options *opts, char *err,
                                size_t errlen, int want_crc)
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
    int win_rows[RJ_MAX_DISKS]; /* #29: winning subset's equation rows */
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
                                     &nv, &surv_plans, &n_surv_plans,
                                     win_rows, err, errlen);
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
        pl.want_crc = want_crc;

        /* Candidate buffer. Worst case today: RVV P_HORNER with LMUL:
         * 2 muls x 5 U x 4 LMUL = 40 base; +20 remat twins = 60;
         * x2 for xorsched (P_XOR only, up to 120); x2 for NT twins (240);
         * x2 for prefetch twins (480). 512 is sufficient but at 94%
         * utilization; a future axis must increase the buffer or fail
         * closed -- silent truncation is unacceptable. plan_candidates
         * guards with maxout. */
        cand_t cands[512];
        /* NT/prefetch twins only when benchmarking: on the static model they
         * tie with their temporal originals and lose the tie-break, so
         * generating them would only multiply compile time. The prefetch
         * argument resolves the manual override vs the I4 analytical
         * default (see prefetch_calib.h). */
        int ncand = plan_candidates(&pl, &cpu, cands, 512,
                                    o.nt_threshold > 0 && o.benchmark,
                                    prefetch_calib_dist_arg(&o));
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
                /* Heuristic path: validate before selecting (bench_select
                 * does this internally; heuristic_select does not). */
                if (validate_versions(vs, nv, &pl, err, errlen) != 0)
                    goto fail;
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
    /* #29: the effective stage-2 tail configuration -- the installed
     * tail twins that actually serve traffic. A notail twin that lost
     * the shootout never serves (dead weight, freed below) and is not
     * part of the effective spec. */
    int tail_eff_notail = 0, tail_eff_predtail = 0, tail_eff_maskedtail = 0;
    /* The twins are built from the WINNER's plan: on the survivor
     * path that is the winning subset's plan (wpl), so the chosen set
     * bakes into every dispatched variant. Dispatcher ABI unchanged. */
    /* Stage 2 is skipped for REDUX crc_fold winners: fold kernels take
     * the 4-argument entry (fold_state in RCX), but an installed
     * dispatcher has the 3-argument rj_fn signature -- routing a fold
     * kernel through it would feed the kernel a garbage fold_state and
     * crash. The general fold kernel serves every length correctly
     * (its scalar tail folds whole 16B chunks per the §3.4 contract),
     * so no twin is needed. */
    if (o.benchmark && !wpl->noop && !vs[winner].cand.crc_fold) {
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
         * - Masked-tail twin: RVV winners (any plan kind / multiply
         *   method). The unrolled full-VL v0-masked tail
         *   (src/riscv/jit.c gen_tail_masked) replaces the strip-mining
         *   remainder loop. No extension gate beyond ISA_RVV (vid.v,
         *   vmsltu.vx and masked vle8.v/vse8.v are base V). Disjoint
         *   from notail/predtail (different ISA from both).
         *
         * Each twin is built from the winner and gated by
         * api_tail_duel_gate (bench_tail_duel, ISA-agnostic): install
         * only on a strict tail win outside the noise window plus a
         * large-size no-regression gate. A duel loss drops the twin
         * (tail traffic falls back to the scalar tail); a duel error
         * fails compilation. The three tail twins are mutually exclusive
         * by ISA (ISA_SVE vs ISA_AVX512 vs ISA_RVV on the same winner),
         * so at most one duel_idx is ever set. */
        int duel_idx = -1; /* index into twins[] of the duel-installed tail
                            * twin, or -1 when no duel was won */
        /* The predtail twin is not implemented for the pipe
         * (software-pipelined) body (I8): the predicated tail reuses
         * the non-pipelined gen_body whose register layout is dead in
         * pipe mode. Skip the twin when the winner is pipe; the pipe
         * twin keeps its scalar tail. */
        if (vs[winner].cand.isa == ISA_SVE && !vs[winner].cand.pipe) {
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
        /* Masked-tail twin (track B1x86): eligible for P_XOR (plain body)
         * and for GFNI multiply tails (P_MULACC/P_DELTA/P_HORNER with
         * mul == MUL_GFNI). Nibble masked tails are not implemented.
         * The tail duel (bench_tail_duel) validates correctness and
         * requires a measured win before installing. */
        if (vs[winner].cand.isa == ISA_AVX512 &&
            !vs[winner].cand.nt &&
            cpu.avx512f && cpu.avx512bw &&
            ((wpl->kind == P_XOR &&
              !vs[winner].cand.ternlog &&
              !vs[winner].cand.xorsched) ||
             (vs[winner].cand.mul == MUL_GFNI && cpu.gfni &&
              (wpl->kind == P_MULACC || wpl->kind == P_DELTA ||
               wpl->kind == P_HORNER)))) {
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
        /* Masked-tail twin (RVV): the winner's strip-mining remainder
         * loop is replaced by an unrolled full-VL masked tail (v0 mask,
         * one chunk per U; src/riscv/jit.c gen_tail_masked). Eligible
         * for every plan kind and multiply method: only data
         * loads/stores are masked, compute on masked-off lanes is
         * don't-care, and table loads stay unmasked. vid.v/vmsltu.vx and
         * masked vle8.v/vse8.v are base-V instructions, so no extension
         * gate beyond ISA_RVV is needed. The tail duel
         * (bench_tail_duel) validates correctness and requires a
         * measured win before installing, same as the x86 twin. */
        if (vs[winner].cand.isa == ISA_RVV) {
            cand_t mtwinc = vs[winner].cand;
            mtwinc.maskedtail = 1; /* twin of the winner, masked tail */
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
                     * only); predtail (B1sve) and maskedtail (B1x86, RVV)
                     * twins own tail lengths (TAIL_MASKED). The mapping
                     * is explicit per kind: a future twin kind that sets
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
                if (disp) {
                    /* #29: effective tail config. w2 != 0: the notail
                     * twin won the large-size shootout (pair[1..] hold
                     * only the non-duel twins, i.e. at most the notail
                     * twin) and serves multiples. duel_idx >= 0: the
                     * predtail/maskedtail twin won its duel and serves
                     * tails. */
                    if (w2 != 0)
                        tail_eff_notail = 1;
                    if (duel_idx >= 0) {
                        if (twins[duel_idx].predtail)
                            tail_eff_predtail = 1;
                        else if (twins[duel_idx].maskedtail)
                            tail_eff_maskedtail = 1;
                    }
                }
            }
        }
    }

    /* #29: derive the stable plan ID while the topology, op, and plan
     * are still alive. Names the winning candidate spec plus the
     * effective stage-2 tail configuration. */
    char idbuf[RJ_PLAN_ID_MAX];
    {
        planid_opdesc_t opd;
        const int *rowsp = NULL;
        int nrowsp = 0;
        int allrows[RJ_MAX_DISKS];
        int iderr = 0;
        if (op->is_recover && !op->is_parity_update) {
            if (use_survivor) {
                rowsp = win_rows; /* nf rows from survivor_select */
                /* nf: recompute below via planid_opdesc_build's
                 * canonicalization of op->failed; win_rows holds that
                 * many entries. Count them from the op. */
                int *F2 = NULL;
                int nf2 = plan_recover_failed_list(op, ti.ndata, ti.npar,
                                                  &F2, err, errlen);
                if (nf2 < 0)
                    iderr = 1;
                else
                    nrowsp = nf2;
                free(F2);
            } else {
                for (int r = 0; r < ti.npar; r++)
                    allrows[r] = r;
                rowsp = allrows;
                nrowsp = ti.npar;
            }
        }
        if (!iderr &&
            planid_opdesc_build(&opd, &ti, op, rowsp, nrowsp, err,
                                errlen) != 0)
            iderr = 1;
        planid_tail_t tail = {tail_eff_notail, tail_eff_predtail,
                              tail_eff_maskedtail};
        if (!iderr &&
            planid_format(idbuf, sizeof(idbuf), &ti,
                          topo_internal_name(topo), &opd, wpl,
                          &vs[winner].cand, &tail) != 0) {
            if (err && errlen)
                snprintf(err, errlen, "internal error: plan ID did not fit");
            iderr = 1;
        }
        if (iderr) {
            /* The fail label below does not own the stage-2 twins or
             * the dispatcher yet (they move into j after its
             * allocation): release them here. */
            for (int t = 0; t < ntwins; t++)
                jit_free_version(&twins[t]);
            rj_dispatcher_destroy(disp);
            goto fail;
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
    memcpy(j->plan_id, idbuf, sizeof(j->plan_id)); /* #29: derived above */
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
    /* Batched Tier-1 driver (item #9): inherits the dispatched
     * single-stripe winner (the same fn rj_get_fn returns), zero new
     * codegen. nbufs counts follow the rj_fn caller contract for this
     * op. ti is still alive here; it is freed below. OOM degrades to
     * single-stripe only: rj_get_fn_batched returns NULL and rj_get_fn
     * keeps working. */
    {
        size_t nbufs_in, nbufs_out;
        if (op->is_parity_update) {
            nbufs_in = (size_t)(2 * op->nchanged + ti.npar);
            nbufs_out = (size_t)ti.npar;
        } else if (op->is_recover) {
            nbufs_in = (size_t)(ti.ndata + ti.npar - op->nfailed);
            nbufs_out = (size_t)op->nfailed;
        } else {
            nbufs_in = (size_t)ti.ndata;
            nbufs_out = (size_t)ti.npar;
        }
        j->batch = rj_batch_create(j->disp ? rj_dispatcher_fn(j->disp)
                                           : j->v.fn,
                                  nbufs_in, nbufs_out);
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

rj_fn_batched rj_get_fn_batched(const rj_jit *j)
{
    if (!j)
        return NULL;
    /* NULL when the driver failed to allocate (rj_compile degrades to
     * single-stripe only); rj_batch_fn(NULL) is NULL either way. */
    return rj_batch_fn(j->batch);
}

void rj_free(rj_jit *j)
{
    if (j) {
        rj_dispatcher_destroy(j->disp);
        rj_batch_destroy(j->batch); /* NULL-safe: pinned compiles never
                                    * created one */
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

/* ---------------- stable plan IDs (#29) ---------------- */

const char *rj_plan_id(const rj_jit *j)
{
    if (!j)
        return NULL;
    return j->plan_id;
}

/* Field-by-field candidate equality for the pinned-compile membership
 * check (memcmp is unsafe: cand_t may carry padding bytes).
 * BUG-COMBO-STATIC-F04: pipe, rec_cf, crc_fold were missing, so a pinned
 * spec differing only in those axes could match an enumerated candidate
 * that lacks them (membership bypass). CLASS HUNT: any new cand_t axis
 * must be added here, to planid format/parse, to any_twin, and to
 * twin_dominates in the same change. */
static int cand_spec_equal(const cand_t *a, const cand_t *b)
{
    return a->isa == b->isa && a->mul == b->mul && a->U == b->U &&
           a->nt == b->nt && a->prefetch_dist == b->prefetch_dist &&
           a->ternlog == b->ternlog && a->remat == b->remat &&
           a->notail == b->notail && a->predtail == b->predtail &&
           a->maskedtail == b->maskedtail && a->permi2b == b->permi2b &&
           a->xorsched == b->xorsched && a->xorbit == b->xorbit &&
           a->mshift == b->mshift && a->lmul == b->lmul &&
           a->pipe == b->pipe && a->rec_cf == b->rec_cf &&
           a->crc_fold == b->crc_fold;
}

static const char *planid_kind_name(plan_kind_t k)
{
    switch (k) {
    case P_XOR: return "xor";
    case P_HORNER: return "horner";
    case P_MULACC: return "mulacc";
    case P_DELTA: return "delta";
    }
    return "?";
}

rj_jit *rj_compile_pinned(const rj_topology *topo, const rj_op *op,
                          const rj_options *opts, const char *plan_id,
                          char *err, size_t errlen)
{
    if (!topo || !op || !plan_id) {
        if (err && errlen)
            snprintf(err, errlen, "null topology, op, or plan ID");
        return NULL;
    }
    rj_options o;
    if (opts)
        o = *opts;
    else
        rj_options_default(&o);
    /* loop_align feeds em_align's mask arithmetic (same gate as
     * rj_compile). benchmark/nt_threshold/prefetch_distance are inert
     * here: the ID already pins the nt and prefetch axes, and there is
     * no benchmarking on the pinned path. */
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
    const char *opname = topo_internal_name(topo);

    planid_spec_t spec;
    if (planid_parse(plan_id, &ti, opname, op, &spec, err, errlen) != 0) {
        topo_int_free(&ti);
        return NULL;
    }

    /* Rebuild the exact plan the ID names. Recover always goes through
     * plan_build_recover_rows with the ID's equation row set: the
     * survivor-vs-single-subset decision is a property of rj_compile's
     * search, not of the pinned reproduction (deterministic by
     * construction). */
    plan_t pl;
    memset(&pl, 0, sizeof(pl));
    if (spec.opd.is_recover) {
        int *F = NULL;
        int nf = plan_recover_failed_list(op, ti.ndata, ti.npar, &F, err,
                                          errlen);
        if (nf < 0) {
            topo_int_free(&ti);
            return NULL;
        }
        /* planid_parse already validated the failed set against op. */
        int rc = plan_build_recover_rows(&ti, F, nf, spec.opd.rows,
                                        spec.opd.nrows, opname, &pl, err,
                                        errlen);
        free(F);
        if (rc != 0) {
            topo_int_free(&ti);
            return NULL;
        }
    } else {
        if (plan_build(&ti, op, opname, &pl, err, errlen) != 0) {
            topo_int_free(&ti);
            return NULL;
        }
    }

    /* The rebuilt plan's kind must match the ID's kind axis: a mismatch
     * means the planner derives a different strategy than the one the ID
     * was issued for (planner drift the version bump missed) -- fail
     * loudly, never build the wrong plan. */
    if (pl.kind != spec.kind) {
        if (err && errlen)
            snprintf(err, errlen,
                     "pinned plan ID: kind mismatch (ID names %s, this "
                     "build derives %s for this topology/op)",
                     planid_kind_name(spec.kind),
                     planid_kind_name(pl.kind));
        plan_free(&pl);
        topo_int_free(&ti);
        return NULL;
    }
    /* No-op plans have no code to twin: an ID naming any twin for one
     * is bogus. The kernel is a vacuous prologue/epilogue; reporting
     * twins as "applied" would be a phantom (BUG-COMBO-E50). */
    if (pl.noop &&
        (spec.tail.notail || spec.tail.predtail || spec.tail.maskedtail ||
         spec.cand.nt || spec.cand.ternlog || spec.cand.xorbit ||
         spec.cand.xorsched || spec.cand.permi2b || spec.cand.remat ||
         spec.cand.mshift || spec.cand.pipe || spec.cand.rec_cf ||
         spec.cand.prefetch_dist || spec.cand.crc_fold || spec.cand.lmul)) {
        if (err && errlen)
            snprintf(err, errlen,
                     "pinned plan ID: names twins for a no-op plan");
        plan_free(&pl);
        topo_int_free(&ti);
        return NULL;
    }
    /* Shape validation: every backend assert becomes a clean error (a
     * bad ID must fail loudly, never abort). */
    if (planid_cand_valid(&pl, &spec.cand, err, errlen) != 0) {
        plan_free(&pl);
        topo_int_free(&ti);
        return NULL;
    }
    /* Membership: the candidate (minus the stage-2 permi2b flag, which
     * is never enumerated) must be one this build's planner enumerates
     * for this plan on this CPU. The enumeration is deliberately a
     * superset (allow_nt=1; the ID's exact prefetch distance): the ID
     * is self-describing, so the check is "could this build construct
     * this spec", not "would rj_compile with these opts enumerate it". */
    cand_t member = spec.cand;
    member.permi2b = 0;
    cand_t cands[512];
    size_t pfarg = spec.cand.prefetch_dist ? spec.cand.prefetch_dist : 0;
    int nmem = plan_candidates(&pl, &cpu, cands, 512, 1, pfarg);
    int found = 0;
    for (int i = 0; i < nmem; i++)
        if (cand_spec_equal(&cands[i], &member)) {
            found = 1;
            break;
        }
    if (!found) {
        if (err && errlen)
            snprintf(err, errlen,
                     "unknown plan ID: candidate is not in this build's "
                     "candidate space for this plan/CPU");
        plan_free(&pl);
        topo_int_free(&ti);
        return NULL;
    }
    /* permi2b is a stage-2 twin, never enumerated: validate eligibility
     * through the single source of truth (the B2 duel gate), then set
     * the flag for the build. */
    cand_t build_cand = member;
    if (spec.cand.permi2b) {
        version_t tmpw;
        memset(&tmpw, 0, sizeof(tmpw));
        tmpw.cand = member;
        if (!api_permi2b_duel_eligible(&tmpw, &pl, &cpu)) {
            if (err && errlen)
                snprintf(err, errlen,
                         "pinned plan ID: names a permi2b twin this build "
                         "cannot construct (needs AVX-512/VBMI nibble "
                         "MULACC/DELTA with a nontrivial coefficient)");
            plan_free(&pl);
            topo_int_free(&ti);
            return NULL;
        }
        build_cand.permi2b = 1;
    }

    /* Build the general version. A clean backend error (e.g. a prefetch
     * distance the ISA cannot encode) is a loud failure, never a
     * fallback. */
    version_t v;
    memset(&v, 0, sizeof(v));
    {
        char e2[256];
        if (jit_build_version(&pl, &build_cand, &cpu, &o, &v, e2,
                              sizeof(e2)) != 0) {
            if (err && errlen)
                snprintf(err, errlen,
                         "pinned plan ID: candidate build failed: %s", e2);
            plan_free(&pl);
            topo_int_free(&ti);
            return NULL;
        }
    }
    /* Same correctness bar as rj_compile: validate against the scalar
     * reference before shipping. */
    if (validate_versions(&v, 1, &pl, err, errlen) != 0) {
        jit_free_version(&v);
        plan_free(&pl);
        topo_int_free(&ti);
        return NULL;
    }

    /* Stage-2 tail twins, exactly as named by the ID. No benchmarking:
     * the install decisions are part of the pinned spec. Any twin that
     * cannot be built is a loud error, never a silent downgrade. */
    version_t twins[2];
    memset(twins, 0, sizeof(twins));
    int ntwins = 0;
    int twin_kind[2]; /* 0=notail, 1=predtail, 2=maskedtail */
    int eff_notail = 0, eff_pred = 0, eff_masked = 0;
    rj_dispatcher *disp = NULL;
    char e3[256];
    if (spec.tail.notail) {
        cand_t tc = build_cand;
        tc.notail = 1;
        if (jit_build_version(&pl, &tc, &cpu, &o, &twins[ntwins], e3,
                              sizeof(e3)) != 0) {
            if (err && errlen)
                snprintf(err, errlen,
                         "pinned plan ID: notail twin build failed: %s",
                         e3);
            goto pinned_fail;
        }
        twin_kind[ntwins] = 0;
        ntwins++;
        eff_notail = 1;
    }
    if (spec.tail.predtail) {
        if (build_cand.isa != ISA_SVE) {
            if (err && errlen)
                snprintf(err, errlen,
                         "pinned plan ID: predtail twin needs an SVE "
                         "candidate");
            goto pinned_fail;
        }
        cand_t tc = build_cand;
        tc.predtail = 1;
        if (jit_build_version(&pl, &tc, &cpu, &o, &twins[ntwins], e3,
                              sizeof(e3)) != 0) {
            if (err && errlen)
                snprintf(err, errlen,
                         "pinned plan ID: predtail twin build failed: %s",
                         e3);
            goto pinned_fail;
        }
        twin_kind[ntwins] = 1;
        ntwins++;
        eff_pred = 1;
    }
    if (spec.tail.maskedtail) {
        /* Same eligibility as rj_compile's install gate: x86-64 AVX-512
         * (P_XOR plain or GFNI) or RVV (base V; the backend enforces
         * VLMAX <= 255). */
        int mt_ok =
            (build_cand.isa == ISA_AVX512 && !build_cand.nt &&
             cpu.avx512f && cpu.avx512bw &&
             ((pl.kind == P_XOR && !build_cand.ternlog &&
               !build_cand.xorsched) ||
              (build_cand.mul == MUL_GFNI && cpu.gfni &&
               (pl.kind == P_MULACC || pl.kind == P_DELTA ||
                pl.kind == P_HORNER)))) ||
            (build_cand.isa == ISA_RVV);
        if (!mt_ok) {
            if (err && errlen)
                snprintf(err, errlen,
                         "pinned plan ID: names a maskedtail twin this "
                         "build cannot construct");
            goto pinned_fail;
        }
        cand_t tc = build_cand;
        tc.maskedtail = 1;
        if (jit_build_version(&pl, &tc, &cpu, &o, &twins[ntwins], e3,
                              sizeof(e3)) != 0) {
            if (err && errlen)
                snprintf(err, errlen,
                         "pinned plan ID: maskedtail twin build failed: "
                         "%s",
                         e3);
            goto pinned_fail;
        }
        twin_kind[ntwins] = 2;
        ntwins++;
        eff_masked = 1;
    }
    /* Validate the twins (same bar as the general version;
     * validate_versions tests notail twins at multiples only). */
    {
        version_t aval[3];
        aval[0] = v; /* shallow; freed via v/twins[] below, never here */
        for (int t = 0; t < ntwins; t++)
            aval[1 + t] = twins[t];
        if (validate_versions(aval, 1 + ntwins, &pl, err, errlen) != 0)
            goto pinned_fail;
    }
    /* Dispatcher over exactly the named set. Deterministic
     * fastest-first order without measurements: the ID names only
     * twins that serve traffic in the original unit (a named notail
     * twin won its shootout; a named pred/masked twin won its duel),
     * so ordinal priorities reproduce the effective routing -- notail
     * (multiples) before general; the tail twin owns tails by policy.
     * rj_dispatcher_create re-sorts by these ordinals. */
    if (ntwins > 0) {
        rj_variant vars[3];
        memset(vars, 0, sizeof(vars));
        int nvars = 0;
        for (int t = 0; t < ntwins; t++) {
            vars[nvars].fn = twins[t].fn;
            vars[nvars].tail =
                twin_kind[t] == 0 ? TAIL_NONE : TAIL_MASKED;
            vars[nvars].group = twins[t].group_bytes;
            vars[nvars].gbps = twin_kind[t] == 0 ? 3.0 : 2.0;
            snprintf(vars[nvars].desc, sizeof(vars[nvars].desc), "%s",
                     twins[t].desc);
            rj_variant_derive(&vars[nvars]);
            nvars++;
        }
        vars[nvars].fn = v.fn;
        vars[nvars].tail = TAIL_SCALAR;
        vars[nvars].group = v.group_bytes;
        vars[nvars].gbps = 1.0;
        snprintf(vars[nvars].desc, sizeof(vars[nvars].desc), "%s", v.desc);
        rj_variant_derive(&vars[nvars]);
        nvars++;
        disp = rj_dispatcher_create(vars, nvars);
        /* Slot exhaustion degrades to general-only, like rj_compile;
         * the effective ID below then reports tail=scalar honestly. */
    }

    rj_jit *j = calloc(1, sizeof(*j));
    if (!j) {
        if (err && errlen)
            snprintf(err, errlen, "out of memory");
        goto pinned_fail;
    }
    j->v = v;
    j->v.plan = NULL; /* same borrow discipline as rj_compile */
    if (disp) {
        for (int t = 0; t < ntwins; t++) {
            j->twins[t] = twins[t];
            j->twins[t].plan = NULL;
        }
        j->ntwins = ntwins;
        j->has_dispatch = 1;
        j->disp = disp;
        ntwins = 0; /* now owned by j->twins */
        disp = NULL; /* now owned by j->disp */
    }
    for (int t = 0; t < ntwins; t++)
        jit_free_version(&twins[t]);
    rj_dispatcher_destroy(disp);
    /* Batched Tier-1 driver (item #9) is intentionally not created
     * here: pinned compile serves single-stripe only (j->batch stays
     * NULL via calloc, rj_get_fn_batched returns NULL). Batching is
     * rj_compile's API; the pinned path pins one exact plan and does
     * not take on the driver's trampoline slot. */
    /* The effective ID: round-trips the input ID when nothing degraded
     * (tail=scalar honestly when dispatcher creation failed). */
    {
        planid_tail_t tail = {j->disp && eff_notail, j->disp && eff_pred,
                              j->disp && eff_masked};
        if (planid_format(j->plan_id, sizeof(j->plan_id), &ti, opname,
                          &spec.opd, &pl, &build_cand, &tail) != 0) {
            /* Unreachable: the same data formatted the input ID. */
            rj_free(j);
            plan_free(&pl);
            topo_int_free(&ti);
            if (err && errlen)
                snprintf(err, errlen, "internal error: plan ID did not fit");
            return NULL;
        }
    }
    plan_free(&pl);
    topo_int_free(&ti);
    return j;

pinned_fail:
    for (int t = 0; t < ntwins; t++)
        jit_free_version(&twins[t]);
    jit_free_version(&v);
    rj_dispatcher_destroy(disp);
    plan_free(&pl);
    topo_int_free(&ti);
    return NULL;
}

/* Human-readable desc for a candidate that failed to build (no
 * version_t exists to supply one). Mirrors jit_build_version's desc
 * shape; informational only. tw is sized to its provable maximum so
 * the final snprintf stays within desc's 192 bytes under
 * -Wformat-truncation at every opt level. */
static void synth_desc(char *out, size_t outlen, const plan_t *pl,
                       const cand_t *c)
{
    const char *strat = pl->kind == P_XOR      ? "xor"
                        : pl->kind == P_HORNER ? "horner"
                        : pl->kind == P_DELTA  ? "delta"
                                               : "mulacc";
    const char *mul = c->mul == MUL_GFNI ? "+gfni"
                      : c->mul == MUL_ZVBC ? "+zvbc"
                                           : "+nibble";
    /* Max tokens: +ternlog(8)+xsched(8)+xorbit(7)+nt(3)+pf<20 digits>(23)
     * +remat(6)+permi2b(8)+mshift(7)+lmul<1 digit>(6) = 76 < 80. */
    char tw[80];
    size_t n = 0;
    tw[0] = '\0';
#define APPEND_TOK(cond, tok)                                                  \
    do {                                                                       \
        if (cond)                                                              \
            n += (size_t)snprintf(tw + n, sizeof(tw) - n, "%s", tok);           \
    } while (0)
    APPEND_TOK(c->ternlog, "+ternlog");
    APPEND_TOK(c->xorsched, "+xsched");
    APPEND_TOK(c->xorbit, "+xorbit");
    APPEND_TOK(c->nt, "+nt");
    if (c->prefetch_dist)
        n += (size_t)snprintf(tw + n, sizeof(tw) - n, "+pf%zu",
                              c->prefetch_dist);
    APPEND_TOK(c->remat, "+remat");
    APPEND_TOK(c->permi2b, "+permi2b");
    APPEND_TOK(c->mshift, "+mshift");
    if (c->lmul)
        n += (size_t)snprintf(tw + n, sizeof(tw) - n, "+lmul%d", c->lmul);
#undef APPEND_TOK
    tw[sizeof(tw) - 1] = '\0';
    (void)n;
    if (pl->kind == P_XOR)
        snprintf(out, outlen, "%.65s %s %s U=%d%s (unbuilt)", pl->name,
                 strat, isa_name(c->isa), c->U, tw);
    else
        snprintf(out, outlen, "%.65s %s %s%s U=%d%s (unbuilt)", pl->name,
                 strat, isa_name(c->isa), mul, c->U, tw);
}

/* Emit rj_list_variants entries for one plan. Builds every candidate
 * once (no benchmarking) for the exact static triple -- the same work
 * rj_compile does before pruning -- then applies the identical
 * prune_versions gate rj_compile uses (only when o->benchmark; the
 * heuristic path does not prune). When count_only, just adds the
 * candidate count to *total. Returns 0 on success, -1 on error
 * (err set). */
static int list_variants_emit(const plan_t *pl, const planid_opdesc_t *opd,
                              const topo_int_t *ti, const char *toponame,
                              const rj_cpu_info *cpu, const rj_options *o,
                              int allow_nt, size_t pfarg, int count_only,
                              rj_plan_variant *out, int nout, int *written,
                              int *total, char *err, size_t errlen)
{
    cand_t cands[512];
    int ncand = plan_candidates(pl, cpu, cands, 512, allow_nt, pfarg);
    if (count_only) {
        *total += ncand;
        return 0;
    }
    if (ncand <= 0)
        return 0;

    version_t *vs = calloc((size_t)ncand, sizeof(*vs));
    char(*berr)[256] = calloc((size_t)ncand, sizeof(*berr));
    char(*descs)[192] = calloc((size_t)ncand, sizeof(*descs));
    double *uops = calloc((size_t)ncand, sizeof(*uops));
    double *mem = calloc((size_t)ncand, sizeof(*mem));
    rj_fn *fns = calloc((size_t)ncand, sizeof(*fns));
    int *vidx = calloc((size_t)ncand, sizeof(*vidx));
    int *ispruned = calloc((size_t)ncand, sizeof(*ispruned));
    if (!vs || !berr || !descs || !uops || !mem || !fns || !vidx ||
        !ispruned) {
        free(vs);
        free(berr);
        free(descs);
        free(uops);
        free(mem);
        free(fns);
        free(vidx);
        free(ispruned);
        if (err && errlen)
            snprintf(err, errlen, "out of memory");
        return -1;
    }
    for (int i = 0; i < ncand; i++)
        vidx[i] = -1;

    int nv = 0;
    for (int i = 0; i < ncand; i++) {
        char e2[256];
        if (jit_build_version(pl, &cands[i], cpu, o, &vs[nv], e2,
                              sizeof(e2)) == 0) {
            vidx[i] = nv;
            memcpy(descs[i], vs[nv].desc, sizeof(descs[i]));
            uops[i] = vs[nv].uops_per_byte;
            mem[i] = vs[nv].mem_per_byte;
            fns[i] = vs[nv].fn;
            nv++;
        } else {
            snprintf(berr[i], sizeof(berr[i]), "%s", e2);
        }
    }

    /* The library's static assessment: exactly rj_compile's prune gate.
     * prune_versions finalizes pruned versions in place; survivors are
     * identified by their (unique) fn pointers against the pre-prune
     * snapshot. */
    int nret = nv;
    if (o->benchmark && nv > 0)
        nret = prune_versions(vs, nv, 0);
    if (o->benchmark) {
        for (int i = 0; i < ncand; i++) {
            if (vidx[i] < 0)
                continue; /* build failed: separate reason below */
            int surv = 0;
            for (int k = 0; k < nret; k++)
                if (vs[k].fn == fns[i]) {
                    surv = 1;
                    break;
                }
            ispruned[i] = !surv;
        }
    }
    for (int k = 0; k < nret; k++)
        jit_free_version(&vs[k]);
    free(vs);

    planid_tail_t no_tail = {0, 0, 0};
    int rc = 0;
    for (int i = 0; i < ncand && rc == 0; i++) {
        rj_plan_variant e;
        memset(&e, 0, sizeof(e));
        /* Stage-1 candidate: tail=scalar always (stage-2 twins are
         * built from the winner only, never enumerated). */
        if (planid_format(e.plan_id, sizeof(e.plan_id), ti, toponame, opd,
                          pl, &cands[i], &no_tail) != 0) {
            if (err && errlen)
                snprintf(err, errlen, "internal error: plan ID did not fit");
            rc = -1;
            break;
        }
        if (vidx[i] < 0) {
            synth_desc(e.desc, sizeof(e.desc), pl, &cands[i]);
            e.survivor = 0;
            snprintf(e.reason, sizeof(e.reason), "build failed: %.140s",
                     berr[i]);
        } else {
            memcpy(e.desc, descs[i], sizeof(e.desc));
            e.uops_per_byte = uops[i];
            e.mem_per_byte = mem[i];
            if (ispruned[i]) {
                e.survivor = 0;
                /* The dominating twin is the entry with +ternlog added
                 * to this entry's twins axis (prune_versions only ever
                 * prunes the plain member of a +ternlog twin pair). */
                snprintf(e.reason, sizeof(e.reason),
                         "pruned: +ternlog twin pareto-dominates on the "
                         "static (uops, mem, code) triple (see the entry "
                         "with +ternlog added)");
            } else {
                e.survivor = 1;
            }
        }
        if (*written < nout)
            out[(*written)++] = e;
        (*total)++;
    }

    free(berr);
    free(descs);
    free(uops);
    free(mem);
    free(fns);
    free(vidx);
    free(ispruned);
    return rc;
}

int rj_list_variants(const rj_topology *topo, rj_op op,
                     const rj_options *opts, rj_plan_variant *out, int nout)
{
    if (!topo)
        return -1;
    rj_options o;
    if (opts)
        o = *opts;
    else
        rj_options_default(&o);
    if (o.loop_align < 0 ||
        (o.loop_align != 0 && (o.loop_align & (o.loop_align - 1)) != 0))
        return -1;
    if (op.oom)
        return -1;

    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    char ebuf[256];
    topo_int_t ti;
    if (topo_int_build(&ti, topo, ebuf, sizeof(ebuf)) != 0)
        return -1;
    const char *opname = topo_internal_name(topo);
    /* NT/prefetch twin enumeration follows rj_compile exactly. */
    int allow_nt = o.nt_threshold > 0 && o.benchmark;
    size_t pfarg = prefetch_calib_dist_arg(&o);
    int count_only = (out == NULL || nout <= 0);

    int rc = 0, total = 0, written = 0;

    if (op.is_recover && !op.is_parity_update) {
        int *F = NULL;
        int nf = plan_recover_failed_list(&op, ti.ndata, ti.npar, &F, ebuf,
                                          sizeof(ebuf));
        if (nf < 0)
            goto lv_done;
        /* Same survivor-vs-single decision as rj_compile. */
        const char *nss = getenv("RJ_NO_SURVIVOR_SELECT");
        int no_survivor = nss && nss[0] != '\0' && strcmp(nss, "0") != 0;
        int use_survivor = !no_survivor && !survivor_over_cap(&ti, F, nf);
        if (use_survivor) {
            /* Full pre-prune space: every enumerated subset's
             * candidates. The static assessment has two layers: the
             * subset shortlist, then prune_versions per kept subset
             * (mirroring survivor_select). */
            survivor_subset_t *subs = NULL;
            int nsub = 0;
            int *keep = NULL;
            if (survivor_enumerate(&ti, F, nf, opname, &subs, &nsub, ebuf,
                                   sizeof(ebuf)) != 0) {
                free(F);
                goto lv_done;
            }
            if (nsub > 0) {
                keep = malloc((size_t)nsub * sizeof(int));
                if (!keep) {
                    survivor_free_subsets(subs, nsub);
                    free(F);
                    goto lv_done;
                }
                survivor_l2_signal(subs, nsub, &cpu, &o);
                survivor_shortlist(subs, nsub, keep);
            }
            for (int si = 0; si < nsub; si++) {
                planid_opdesc_t opd;
                if (planid_opdesc_build(&opd, &ti, &op, subs[si].rows,
                                       subs[si].nrows, ebuf,
                                       sizeof(ebuf)) != 0) {
                    rc = -1;
                    break;
                }
                if (keep && !keep[si]) {
                    /* Subset pruned by the static shortlist: tag every
                     * candidate without building (nothing was built for
                     * it in rj_compile either). */
                    cand_t cands[512];
                    int ncand = plan_candidates(&subs[si].plan, &cpu,
                                                cands, 512, allow_nt,
                                                pfarg);
                    if (count_only) {
                        total += ncand;
                        continue;
                    }
                    planid_tail_t no_tail = {0, 0, 0};
                    for (int i = 0; i < ncand && rc == 0; i++) {
                        rj_plan_variant e;
                        memset(&e, 0, sizeof(e));
                        if (planid_format(e.plan_id, sizeof(e.plan_id),
                                          &ti, opname, &opd,
                                          &subs[si].plan, &cands[i],
                                          &no_tail) != 0) {
                            rc = -1;
                            break;
                        }
                        synth_desc(e.desc, sizeof(e.desc), &subs[si].plan,
                                   &cands[i]);
                        e.survivor = 0;
                        snprintf(e.reason, sizeof(e.reason),
                                 "pruned: parity-row subset shortlisted "
                                 "out by the static L2 signal");
                        if (written < nout)
                            out[written++] = e;
                        total++;
                    }
                    if (rc != 0)
                        break;
                } else {
                    if (list_variants_emit(&subs[si].plan, &opd, &ti,
                                           opname, &cpu, &o, allow_nt,
                                           pfarg, count_only, out, nout,
                                           &written, &total, ebuf,
                                           sizeof(ebuf)) != 0) {
                        rc = -1;
                        break;
                    }
                }
            }
            free(keep);
            survivor_free_subsets(subs, nsub);
        } else {
            plan_t pl;
            memset(&pl, 0, sizeof(pl));
            if (plan_build(&ti, &op, opname, &pl, ebuf, sizeof(ebuf)) !=
                0) {
                free(F);
                goto lv_done;
            }
            int rows[RJ_MAX_DISKS];
            for (int r = 0; r < ti.npar; r++)
                rows[r] = r;
            planid_opdesc_t opd;
            if (planid_opdesc_build(&opd, &ti, &op, rows, ti.npar, ebuf,
                                    sizeof(ebuf)) != 0)
                rc = -1;
            else if (list_variants_emit(&pl, &opd, &ti, opname, &cpu, &o,
                                        allow_nt, pfarg, count_only, out,
                                        nout, &written, &total, ebuf,
                                        sizeof(ebuf)) != 0)
                rc = -1;
            plan_free(&pl);
        }
        free(F);
    } else {
        plan_t pl;
        memset(&pl, 0, sizeof(pl));
        if (plan_build(&ti, &op, opname, &pl, ebuf, sizeof(ebuf)) != 0)
            goto lv_done;
        planid_opdesc_t opd;
        if (planid_opdesc_build(&opd, &ti, &op, NULL, 0, ebuf,
                                sizeof(ebuf)) != 0)
            rc = -1;
        else if (list_variants_emit(&pl, &opd, &ti, opname, &cpu, &o,
                                    allow_nt, pfarg, count_only, out,
                                    nout, &written, &total, ebuf,
                                    sizeof(ebuf)) != 0)
            rc = -1;
        plan_free(&pl);
    }

lv_done:
    topo_int_free(&ti);
    return rc == 0 ? total : -1;
}

rj_jit *rj_compile(const rj_topology *topo, const rj_op *op,
                   const rj_options *opts, char *err, size_t errlen)
{
    return rj_compile_impl(topo, op, opts, err, errlen, 0);
}

struct rj_jit *rj_compile_want_crc(const rj_topology *t, const rj_op *op,
                                   const rj_options *o, char *err,
                                   size_t errlen)
{
    return rj_compile_impl(t, op, o, err, errlen, 1);
}

/* Phase 2: query in-kernel CRC folding. The version folds iff its
 * candidate has crc_fold=1 (set by plan_candidates when want_crc=1 and
 * the uarch gate is open). */
int rj_jit_crc_fold(const version_t *v)
{
    return v && v->cand.crc_fold;
}

/* Phase 2: query folding on a jit handle. */
int rj_jit_handle_crc_fold(const struct rj_jit *j)
{
    return j && rj_jit_crc_fold(&j->v);
}
