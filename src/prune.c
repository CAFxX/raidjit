/* Static-dominance pruning of the benchmark candidate list.
 *
 * This implements a deliberately NARROW dominance rule. An earlier,
 * broader rule (dominance on the raw (uops/B, mem/B, code) triple within
 * the same (isa, mul) class) was tried and REJECTED by the adversarial
 * test matrix (tests/test_prune.c): it pruned real benchmark winners.
 * Two failure modes were observed on 2026-10-02:
 *
 *  1. U-invariance: for P_XOR the model reports identical uops/B and
 *     mem/B for every U (unrolling benefits are invisible to it), so
 *     U=1 "dominated" U=8..U=2 on code size alone -- yet U=4/U=5 win
 *     the benchmarks. The triple is not a dominance oracle across U.
 *
 *  2. P_XOR twin competitiveness: the model ranks the plain P_XOR
 *     candidate above its +ternlog twin (0.1250/6.0 vs 0.1406/7.0),
 *     but the twin WON the 1 MiB benchmark (dense 4+2 recovery).
 *     Static uop counting does not reflect vpternlogd's true cost.
 *
 * The surviving rule therefore prunes ONLY the plain member of a
 * +ternlog twin pair, and only when the twin pareto-dominates it on the
 * triple, and only for plan kinds P_MULACC / P_DELTA. The twin pairs
 * are the one place where the codegen is structurally identical except
 * for the multiply instruction itself (nibble sequence vs VPTERNLOGD),
 * so the static triple faithfully captures the difference. The
 * direction is load-bearing: we prune plain-in-favor-of-twin (P_MULACC,
 * where the twin does strictly less work: fewer uops, smaller code,
 * same mem) and NEVER twin-in-favor-of-plain (P_XOR, where the twin is
 * empirically competitive). The plan-kind restriction is enforced by
 * CODE (the kind guard in twin_dominates), not by the current triple
 * values: if the model ever claims a P_XOR twin dominates, pruning is
 * refused (with a verbose log under PRUNE_VERBOSE). On top of the
 * triple, a uarch-aware leg (I3, prune-soundness P1) vetoes the prune
 * when the twin's 2xVPXOR->VPTERNLOGD substitution is port-hostile on
 * the host uarch (per-uarch port-pressure weights from published
 * throughputs; see prune_twin_uarch_ok). The veto is veto-only -- it
 * keeps candidates, never drops -- so the static model cannot newly
 * misrank on an unmodeled uarch. The zero-flip ship criterion in
 * tests/test_prune.c is the empirical gate for this rule, with the
 * argmin tripwire (Part 3) as the loud detector for future
 * model/codegen drift.
 *
 * Public API (also documented in prune.h for the later survivor-set
 * track):
 *   int prune_versions(version_t *vs, int nv, unsigned flags);
 */
#include "prune.h"
#include "jit.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *kind_name(plan_kind_t k)
{
    switch (k) {
    case P_XOR: return "P_XOR";
    case P_MULACC: return "P_MULACC";
    case P_HORNER: return "P_HORNER";
    case P_DELTA: return "P_DELTA";
    }
    return "?";
}

/* Uarch-aware leg of the twin dominance gate (I3, prune-soundness P1).
 *
 * The twin's uop saving over plain comes from ONE structural
 * substitution: each saved uop is a (2 VPXOR -> 1 VPTERNLOGD) fold of
 * the nibble path's acc ^= r1 ^ r2 (src/x86_64/jit.c mulacc_column;
 * the twins are otherwise structurally identical, which is also what
 * makes the triple gate sound). In port-pressure terms the
 * substitution saves du*(2*t_xor - t_ternlog) cycles/byte, with the
 * throughputs taken from this uarch's published port data
 * (rj_uarch_weights). If that quantity is negative -- the substitution
 * is port-HOSTILE here, i.e. the "saved" uop actually costs cycles --
 * the plain candidate may win the benchmark on this uarch, so pruning
 * must keep it.
 *
 * Returns 1 when the substitution is not port-hostile (prune may
 * proceed); 0 when it is (keep the plain). On current published data
 * the substitution is port-profitable on every modeled uarch, so the
 * veto never fires there; it guards future twin/codegen changes whose
 * substitution might not be. Veto-only: it can keep the plain, never
 * drop anything the triple gate kept. Exposed non-statically for
 * tests (declared in prune.h). */
int prune_twin_uarch_ok(const version_t *twin, const version_t *plain,
                        const rj_uarch_weights_t *w)
{
    /* The substitution analysis is valid only for the AVX-512 nibble
     * twin -- the only +ternlog twin the planner enumerates
     * (plan_candidates gates ternlog twins on isa == ISA_AVX512 &&
     * mul == MUL_NIBBLE). Anything else carries no uarch information:
     * no veto. */
    if (twin->cand.isa != ISA_AVX512 || twin->cand.mul != MUL_NIBBLE)
        return 1;
    double du = plain->uops_per_byte - twin->uops_per_byte;
    double margin = du * (2.0 * w->t_xor - w->t_ternlog);
    return margin >= -PRUNE_EPS;
}

/* A (the +ternlog twin) dominates B (plain) iff they are the same
 * candidate in every respect except the ternlog flag, their plan kind
 * is one where the twin direction is validated, A's triple
 * pareto-dominates B's, and the twin's uop substitution is not
 * port-hostile on this uarch (prune_twin_uarch_ok). */
static int twin_dominates(const version_t *a, const version_t *b,
                          const rj_uarch_weights_t *w, int verbose)
{
    /* Plan-kind guard (attacker 17): the twin prune direction is
     * validated only for P_MULACC and P_DELTA
     * (docs/candidate_pruning.md). The load-bearing direction is
     * enforced by CODE here, not by the current triple values: if the
     * model ever claims a P_XOR (or any other kind) twin dominates, we
     * do NOT prune. The kind check comes first so a non-validated kind
     * can never prune no matter what the triple says. */
    if (a->kind != b->kind)
        return 0;
    int kind_ok = (a->kind == P_MULACC || a->kind == P_DELTA);

    /* Twin relation: identical candidate axes except the ternlog flag.
     * Compared field-by-field, deliberately NOT memcmp: struct padding
     * bytes are not guaranteed zeroed. remat is included (reviewer1 4):
     * a HORNER remat twin is a different codegen shape (pool
     * rematerialization vs live tables), not a multiply twin, so it
     * must never satisfy the twin relation. notail/xorsched are pinned
     * too (merge hardening, C13): a future candidate axis must not
     * silently bypass the twin gate. permi2b is pinned (track B2):
     * without it a (permi2b=1,ternlog=1) version and a (permi2b=0,
     * ternlog=0) version form a pruning pair on the checked axes, and
     * only the triple gate's current values (permi2b never reduces
     * uops/mem/code) stop a cross-axis misprune -- the pin makes that
     * structural, not numeric. predtail/maskedtail are pinned for the
     * same reason (both are 0 for every in-list candidate today: never
     * set by plan_candidates, built post-selection -- but the gate must
     * not depend on that staying true). lmul is pinned too: an
     * (lmul=2,ternlog=1)/(lmul=1,ternlog=0) pair must never satisfy the
     * twin relation on the checked axes. (Unlike notail/predtail/
     * maskedtail, lmul genuinely varies in-list for RVV, so this pin
     * is load-bearing, not just hardening.) Behavior-preserving for
     * the other axes today: all in-list candidates have
     * notail=0/predtail=0/maskedtail=0, and P_XOR pairs are refused by
     * the kind guard anyway. */
    if (a->cand.ternlog != 1 || b->cand.ternlog != 0)
        return 0;
    if (a->cand.isa != b->cand.isa || a->cand.mul != b->cand.mul ||
        a->cand.U != b->cand.U || a->cand.nt != b->cand.nt ||
        a->cand.prefetch_dist != b->cand.prefetch_dist ||
        a->cand.remat != b->cand.remat ||
        a->cand.notail != b->cand.notail ||
        a->cand.predtail != b->cand.predtail ||
        a->cand.maskedtail != b->cand.maskedtail ||
        a->cand.permi2b != b->cand.permi2b ||
        a->cand.xorsched != b->cand.xorsched ||
        a->cand.lmul != b->cand.lmul)
        return 0;

    int strict = 0;
    if (a->uops_per_byte > b->uops_per_byte + PRUNE_EPS)
        return 0;
    if (a->uops_per_byte < b->uops_per_byte - PRUNE_EPS)
        strict = 1;
    if (a->mem_per_byte > b->mem_per_byte + PRUNE_EPS)
        return 0;
    if (a->mem_per_byte < b->mem_per_byte - PRUNE_EPS)
        strict = 1;
    if (a->code_size > b->code_size)
        return 0;
    if (a->code_size < b->code_size)
        strict = 1;
    if (!strict)
        return 0;

    /* The model claims twin dominance, but for a plan kind whose twin
     * direction is not validated: refuse, and say so loudly (with
     * PRUNE_VERBOSE) rather than pruning or staying silent. */
    if (!kind_ok) {
        if (verbose)
            fprintf(stderr,
                    "raidjit: prune: twin dominates on the triple for "
                    "plan kind %s, but the twin direction is validated "
                    "only for P_MULACC/P_DELTA -- NOT pruning\n",
                    kind_name(a->kind));
        return 0;
    }
    /* Uarch-aware leg (I3, prune-soundness P1): the twin's 2xVPXOR ->
     * VPTERNLOGD substitution must not be port-hostile on this uarch
     * (see prune_twin_uarch_ok). Veto-only: it keeps the plain, never
     * drops anything the triple gate kept. */
    if (!prune_twin_uarch_ok(a, b, w)) {
        if (verbose)
            fprintf(stderr,
                    "raidjit: prune: twin's 2xVPXOR->VPTERNLOGD "
                    "substitution is port-hostile on this uarch "
                    "(t_ternlog=%.2f > 2*t_xor=%.2f) -- NOT pruning\n",
                    w->t_ternlog, 2.0 * w->t_xor);
        return 0;
    }
    return 1;
}

int prune_versions(version_t *vs, int nv, unsigned flags)
{
    if (nv <= 0)
        return 0;
    if ((flags & PRUNE_DISABLE) != 0)
        return nv;
    /* RJ_NO_PRUNE disables pruning: any non-empty value other than
     * exactly "0" (checked once per call; the env var is
     * process-global). See prune.h for the exact contract. */
    const char *env = getenv("RJ_NO_PRUNE");
    if (env && env[0] != '\0' && strcmp(env, "0") != 0)
        return nv;

    /* O(nv^2) dominance check. nv is small (tens of candidates); the
     * twin relation is sparse (each plain has at most one twin). */
    char *dominated = calloc((size_t)nv, 1);
    if (!dominated)
        return nv; /* allocation failure: prune nothing (safe) */
    int verbose = (flags & PRUNE_VERBOSE) != 0;
    /* Uarch tag for the port-pressure leg of the dominance gate,
     * resolved once per call (test override honored via
     * rj_uarch_current). */
    const rj_uarch_weights_t *w =
        rj_uarch_weights(rj_uarch_current());
    for (int i = 0; i < nv; i++) {
        for (int j = 0; j < nv; j++) {
            if (i == j || dominated[j])
                continue;
            if (twin_dominates(&vs[i], &vs[j], w, verbose))
                dominated[j] = 1;
        }
    }

    /* Finalize the pruned versions, then compact the survivors in stable
     * order via a temporary buffer (so no struct is lost or double-freed
     * by the in-place move). Each plain has at most one twin, and the
     * twin itself is never pruned by this rule, so a survivor always
     * exists; the n == 0 reset is purely defensive. */
    int n = 0;
    for (int i = 0; i < nv; i++)
        if (!dominated[i])
            n++;
    if (n == 0) {
        dominated[0] = 0; /* keep the first version over an empty list */
        n = 1;
    }
    version_t *surv = malloc((size_t)n * sizeof(*surv));
    if (!surv) {
        free(dominated);
        return nv; /* allocation failure: prune nothing (safe) */
    }
    n = 0;
    for (int i = 0; i < nv; i++) {
        if (!dominated[i])
            surv[n++] = vs[i];
        else
            jit_free_version(&vs[i]);
    }
    memcpy(vs, surv, (size_t)n * sizeof(*surv));
    free(surv);
    free(dominated);
    return n;
}
