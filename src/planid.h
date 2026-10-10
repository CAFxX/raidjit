/* Stable plan IDs (#29, design: hidden_files/design_stable_plan_ids.md).
 *
 * A plan ID is a structured, human-readable, version-prefixed string
 * naming one candidate spec exactly:
 *
 *   rj<V>:<topo>:<op>:<isa>:<kind>:<mul>:u<U>:<twins>:<tail>
 *
 * Axis order is fixed and documented; every axis is validated on parse
 * and unknown axes/values fail parsing loudly (never silently ignored).
 *
 *   rj<V>   codegen version, decimal (e.g. "rj1"). See
 *           RJ_CODEGEN_VERSION below.
 *   <topo>  "<name>#<mhash16>": topo_internal_name() (e.g. "rs(6,3)")
 *           plus 16 lowercase hex chars of FNV-1a-64 over
 *           (ndata LE32, npar LE32, matrix bytes). The name is for
 *           humans; the hash is the identity (two custom() topologies
 *           share a name but never a matrix).
 *   <op>    "enc" (encode)
 *           "pupd<d0>+<d1>..." (parity update; changed data disks,
 *             sorted, deduped, raw indices)
 *           "rec<f0>+<f1>....rows<r0>+<r1>..." (recover; failed disks
 *             sorted/deduped, then the equation row set that produced
 *             the plan -- all parity rows on the single-plan path, the
 *             winning subset on the A5 survivor path)
 *   <isa>   "sse" | "avx2" | "avx512" | "neon" | "sve" | "rvv" | "pext"
 *   <kind>  "xor" | "horner" | "mulacc" | "delta" (the plan strategy;
 *           redundant with (topo, op) but self-validating: pinned
 *           compile fails loudly if the rebuilt plan's kind differs)
 *   <mul>   "nibble" | "gfni" | "zvbc" (always present; meaningless
 *           but fixed to "nibble" for P_XOR)
 *   u<U>    unroll, decimal, e.g. "u8"
 *   <twins> "-" when no twin axes are set, else "+"-joined tokens in
 *           this fixed order: nt, pf<N>, ternlog, remat, xorsched,
 *           xorbit, mshift, permi2b, pipe, lmul<N>. pf<N> carries the
 *           prefetch distance in bytes; lmul<N> the RVV LMUL (emitted
 *           whenever nonzero, i.e. lmul=1 renders as "lmul1"). The
 *           stage-2-only flags (notail/predtail/maskedtail) are NEVER
 *           twin tokens: they live in <tail>.
 *   <tail>  the effective stage-2 tail configuration of the compiled
 *           unit: "scalar" (no dispatcher), or "+"-joined
 *           "notail" / "predtail" / "maskedtail" for the installed tail
 *           twins that actually serve traffic. A notail twin that lost
 *           its shootout never serves and is NOT named (the unit is
 *           behaviorally "scalar" for multiples). Canonical order:
 *           notail, predtail, maskedtail (predtail/maskedtail are
 *           mutually exclusive).
 *
 * Example:
 *   rj1:rs(6,3)#9e3779b97f4a7c15:enc:avx512:mulacc:gfni:u8:ternlog:scalar
 *
 * No library-side persistence, ever: the ID is a pure function of the
 * candidate spec. The caller owns storage.
 */
#pragma once

#include <stddef.h>
#include "plan.h"

/* Codegen version. Bump on ANY release whose emitter, planner,
 * candidate enumeration, twin set, prune rules, cost model, or
 * dispatcher could alter generated code. Plan IDs embed this version;
 * rj_compile_pinned rejects IDs from any other version with a
 * version-mismatch error (strict invalidation: no best-effort
 * cross-version mode, forcing callers to re-benchmark after upgrades).
 * While the project is unreleased the number moves freely; the
 * mechanism is what matters. */
#define RJ_CODEGEN_VERSION 2

/* Maximum plan-ID string length, including the NUL. Sized for the
 * pathological case (recover with ~255 failed disks: the failed and
 * row lists each approach ~1 KiB). IDs are never truncated: format
 * fails loudly instead. */
#define RJ_PLAN_ID_MAX 4096

/* Effective stage-2 tail configuration: the installed tail twins that
 * serve traffic (see the <tail> axis above). */
typedef struct {
    int notail;     /* notail twin won the large-size shootout */
    int predtail;   /* SVE predicated-tail twin won its duel */
    int maskedtail; /* AVX-512 masked-tail twin won its duel; RVV maskedtail
                     * twin won its duel */
} planid_tail_t;

/* Operation descriptor for ID formatting: the plan-shape determinants
 * beyond (topology, op-kind). Callers fill the relevant fields:
 *  - encode:          nothing (all zero)
 *  - parity update:   nchanged/changed (sorted, deduped data-disk idx)
 *  - recover:         nfailed/failed (sorted, deduped disk idx) and
 *                     nrows/rows (the equation row set, sorted)
 */
typedef struct {
    int is_recover;
    int is_parity_update;
    int nfailed;
    int failed[RJ_MAX_DISKS];
    int nchanged;
    int changed[RJ_MAX_DISKS];
    int nrows;
    int rows[RJ_MAX_DISKS];
} planid_opdesc_t;

/* A parsed plan ID, validated against (topo, op). */
typedef struct {
    cand_t cand;           /* stage-1 candidate (permi2b as parsed) */
    plan_kind_t kind;      /* expected plan kind */
    planid_tail_t tail;    /* stage-2 tail twins named by the ID */
    planid_opdesc_t opd;   /* op descriptor, validated vs caller's op */
} planid_spec_t;

/* FNV-1a-64 over (ndata LE32, npar LE32, matrix bytes) -- the topology
 * identity hash. Deterministic across processes and platforms. */
unsigned long long planid_topo_hash(const topo_int_t *ti);

/* Format the stable ID for (ti, toponame, opd, pl, cand, tail).
 * toponame is topo_internal_name(topo) ("rs(6,3)"). pl supplies the
 * kind axis. Returns 0 on success, -1 when the ID does not fit outlen
 * (never truncates). */
int planid_format(char *out, size_t outlen, const topo_int_t *ti,
                  const char *toponame, const planid_opdesc_t *opd,
                  const plan_t *pl, const cand_t *cand,
                  const planid_tail_t *tail);

/* Parse id and validate it against (ti, toponame, op):
 *  - malformed IDs, unknown axes/values -> "unknown plan ID" error
 *  - version != RJ_CODEGEN_VERSION -> version-mismatch error
 *  - topo name/hash mismatch -> topology-mismatch error
 *  - op descriptor mismatch vs the caller's op -> op-mismatch error
 * On success fills *spec. Returns 0 on success, -1 on error (err set).
 * Never aborts: every malformed input is a clean error, never an
 * assert (the ID is caller-controlled input). */
int planid_parse(const char *id, const topo_int_t *ti, const char *toponame,
                 const rj_op *op, planid_spec_t *spec, char *err,
                 size_t errlen);

/* Validate a parsed candidate against the plan shape and the planner's
 * invariants, converting every backend assert into a clean error
 * (pinned compile must fail loudly, never abort, on a bad ID):
 *  - U in 1..8; remat only for P_HORNER; xorbit/mshift/permi2b/xorsched/
 *    ternlog/lmul shape rules per the planner's enumeration contract;
 *  - the stage-2-only flags (notail/predtail/maskedtail) are never set
 *    on a parsed candidate.
 * This does NOT check CPU feature gates or candidate-space membership
 * (the caller does that via plan_candidates). Returns 0 when valid,
 * -1 with err set otherwise. */
int planid_cand_valid(const plan_t *pl, const cand_t *cand, char *err,
                      size_t errlen);

/* Build an op descriptor from (ti, op); for recover the caller supplies
 * the equation row set (all parity rows, or the winning subset). The
 * failed/changed lists are sorted and deduped (matching what the
 * planner solved). Returns 0 on success, -1 on error (err set). */
int planid_opdesc_build(planid_opdesc_t *opd, const topo_int_t *ti,
                        const rj_op *op, const int *rows, int nrows,
                        char *err, size_t errlen);
