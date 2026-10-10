#pragma once
#include "plan.h"  /* isa_t, plan_kind_t */

/* ui_meta.h — single source of truth for UI validation metadata.
 *
 * The WASM explorer (and any other UI) must NOT duplicate validation
 * logic. All twin/ISA/plan-kind validity rules live here, derived from
 * the library's own validation (planid_cand_valid, plan.c enumeration
 * gates, api.c tail gates). The UI queries; it does not reimplement.
 *
 * Twin IDs. Order matches the WASM page's TWINS list; do not reorder
 * without updating the page generator.
 */
typedef enum {
    RJ_TWIN_NT = 0,
    RJ_TWIN_TERNLOG,
    RJ_TWIN_REMAT,
    RJ_TWIN_NOTAIL,
    RJ_TWIN_PREDTAIL,
    RJ_TWIN_MASKEDTAIL,
    RJ_TWIN_PERMI2B,
    RJ_TWIN_XORSCHED,
    RJ_TWIN_XORBIT,
    RJ_TWIN_MSHIFT,
    RJ_TWIN_PIPE,
    RJ_TWIN_REC_CF,
    RJ_TWIN_COUNT
} rj_twin_id_t;

/* Exclusivity group: 0 = none, 1 = body (pairwise exclusive),
 * 2 = tail (pairwise exclusive). */
int rj_ui_twin_group(rj_twin_id_t twin);

/* Short name for JSON/UI (e.g. "nt", "ternlog", "rec_cf"). */
const char *rj_ui_twin_name(rj_twin_id_t twin);

/* Returns 1 if the twin is selectable for the given context, 0 if the
 * UI should grey it out. Conservative: returns 0 only when the twin is
 * certainly invalid; when validity depends on plan details not
 * available here (e.g. rec_cf win conditions), returns 1 and lets the
 * backend decide at compile time.
 *
 * isa:      effective ISA (ISA_AVX512, ISA_RVV, ...)
 * kind:     plan kind (P_XOR, P_MULACC, P_HORNER, P_DELTA)
 * n_out:    number of output rows (for the ternlog n_out>=2 gate)
 * xor_sched: 1 if the plan has an xor schedule (for xorsched)
 * is_recover: 1 if the operation is a recovery (for rec_cf)
 */
int rj_ui_twin_valid(rj_twin_id_t twin, isa_t isa, plan_kind_t kind,
                     int n_out, int xor_sched, int is_recover);

/* Field limits for numeric inputs. */
int rj_ui_u_min(void);   /* 0 (0 = auto) */
int rj_ui_u_max(void);   /* 8 (planid_cand_valid: U in 1..8) */
int rj_ui_lmul_valid(int lmul); /* 1 if lmul in {1,2,4,8}, else 0 */
int rj_ui_prefetch_min(void);   /* 0 */
