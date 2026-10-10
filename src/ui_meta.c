/* ui_meta.c — UI validation metadata, single source of truth.
 *
 * See ui_meta.h. All rules here are derived from the library's own
 * validation; the UI must query these instead of reimplementing them.
 */
#include <string.h>
#include "ui_meta.h"
#include "planid.h"

int rj_ui_twin_group(rj_twin_id_t twin)
{
    switch (twin) {
    case RJ_TWIN_XORSCHED:
    case RJ_TWIN_XORBIT:
    case RJ_TWIN_PERMI2B:
    case RJ_TWIN_MSHIFT:
    case RJ_TWIN_PIPE:
    case RJ_TWIN_REMAT:
    case RJ_TWIN_REC_CF:
        return 1; /* body: pairwise exclusive (planid.c "combines with
                   * no other body twin" checks) */
    case RJ_TWIN_NOTAIL:
    case RJ_TWIN_PREDTAIL:
    case RJ_TWIN_MASKEDTAIL:
        return 2; /* tail: pairwise exclusive (planid.c tail parsing:
                   * predtail+maskedtail mutually exclusive; notail is
                   * a separate tail axis) */
    default:
        return 0;
    }
}

const char *rj_ui_twin_name(rj_twin_id_t twin)
{
    switch (twin) {
    case RJ_TWIN_NT:         return "nt";
    case RJ_TWIN_TERNLOG:    return "ternlog";
    case RJ_TWIN_REMAT:      return "remat";
    case RJ_TWIN_NOTAIL:     return "notail";
    case RJ_TWIN_PREDTAIL:   return "predtail";
    case RJ_TWIN_MASKEDTAIL: return "maskedtail";
    case RJ_TWIN_PERMI2B:    return "permi2b";
    case RJ_TWIN_XORSCHED:   return "xorsched";
    case RJ_TWIN_XORBIT:     return "xorbit";
    case RJ_TWIN_MSHIFT:     return "mshift";
    case RJ_TWIN_PIPE:       return "pipe";
    case RJ_TWIN_REC_CF:     return "rec_cf";
    default:                 return "unknown";
    }
}

/* Set the twin flag on a zeroed candidate. Tail twins (notail/predtail/
 * maskedtail) live on the tail spec, not the candidate; they are handled
 * separately below and never reach planid_cand_valid (which rejects
 * them on cand_t by design).
 */
static void twin_set(cand_t *c, rj_twin_id_t twin)
{
    switch (twin) {
    case RJ_TWIN_NT:       c->nt = 1; break;
    case RJ_TWIN_TERNLOG:   c->ternlog = 1; break;
    case RJ_TWIN_REMAT:     c->remat = 1; break;
    case RJ_TWIN_PERMI2B:   c->permi2b = 1; break;
    case RJ_TWIN_XORSCHED:  c->xorsched = 1; break;
    case RJ_TWIN_XORBIT:    c->xorbit = 1; break;
    case RJ_TWIN_MSHIFT:    c->mshift = 1; break;
    case RJ_TWIN_PIPE:      c->pipe = 1; break;
    case RJ_TWIN_REC_CF:    c->rec_cf = 1; break;
    default: break; /* tail twins handled by the caller */
    }
}

int rj_ui_twin_valid(rj_twin_id_t twin, isa_t isa, plan_kind_t kind,
                     int n_out, int xor_sched, int is_recover)
{
    /* Tail twins: validated against api.c's tail gates (not
     * planid_cand_valid, which rejects tail flags on cand_t). */
    if (twin == RJ_TWIN_NOTAIL)
        return 1; /* no gate: always selectable */
    if (twin == RJ_TWIN_PREDTAIL)
        /* api.c pinned path: "predtail twin needs an SVE candidate". */
        return isa == ISA_SVE;
    if (twin == RJ_TWIN_MASKEDTAIL)
        /* Conservative ISA gate matching the page's historical
         * behavior. The full eligibility (api.c mt_ok: AVX-512 needs
         * !nt, avx512f/bw, kind/mul-specific conditions; RVV needs
         * VLMAX <= 255 enforced by the backend) is checked at compile
         * time. */
        return isa == ISA_AVX512 || isa == ISA_RVV;

    /* rec_cf is not covered by planid_cand_valid; its gate comes from
     * the plan.c enumeration (line ~1283): only for P_MULACC recovery
     * plans. The win conditions (cinv_cheap/cinv!=1 for e=1,
     * row_is_01 for e=2) need a full plan, so we don't check them
     * here — conservative: enabled, backend decides. */
    if (twin == RJ_TWIN_REC_CF)
        return kind == P_MULACC && is_recover;

    /* Body twins + nt/ternlog: delegate the shape checks to
     * planid_cand_valid (the same function the compile path uses),
     * then apply the enumeration gates that live outside it. */
    {
        plan_t pl;
        cand_t cand;
        char err[256];
        memset(&pl, 0, sizeof(pl));
        memset(&cand, 0, sizeof(cand));
        pl.kind = kind;
        cand.isa = isa;
        cand.mul = MUL_NIBBLE; /* nibble is the baseline; twins that
                                * need it (xorbit/mshift/permi2b/pipe)
                                * are checked against it. The page
                                * greys out on ISA+kind, not mul. */
        cand.U = 1;
        twin_set(&cand, twin);
        if (planid_cand_valid(&pl, &cand, err, sizeof(err)) != 0)
            return 0;
    }

    /* Enumeration gates from plan.c not covered by planid_cand_valid: */
    if (twin == RJ_TWIN_TERNLOG) {
        /* plan.c:1035 (P_MULACC/P_DELTA, AVX512 nibble) and plan.c:1063
         * (P_XOR, n_out >= 2, AVX512). The ISA+mul parts are already
         * covered by planid_cand_valid (AVX512-only) and the nibble
         * baseline above. */
        if (kind == P_XOR)
            return n_out >= 2;
        return kind == P_MULACC || kind == P_DELTA;
    }
    if (twin == RJ_TWIN_XORSCHED) {
        /* plan.c:1082: enumerated only when the plan has an xor
         * schedule (pl->xor_sched). */
        return xor_sched != 0;
    }
    if (twin == RJ_TWIN_XORBIT) {
        /* BUG_t1_529: xorbit is 5-8x worse than nibble on x86 where
         * vpshufb/pshufb is 1 uop. plan.c restricts enumeration to
         * NEON/SVE/RVV. */
        return isa == ISA_NEON || isa == ISA_SVE || isa == ISA_RVV;
    }

    /* nt, remat, permi2b, mshift, pipe: planid_cand_valid
     * covers their gates fully. */
    return 1;
}

int rj_ui_u_min(void) { return 0; } /* 0 = auto (api.c) */
int rj_ui_u_max(void) { return 8; } /* planid_cand_valid: U in 1..8 */

int rj_ui_lmul_valid(int lmul)
{
    /* planid_cand_valid: lmul in {1,2,4,8}, RVV-only. */
    return lmul == 1 || lmul == 2 || lmul == 4 || lmul == 8;
}

int rj_ui_prefetch_min(void) { return 0; }
