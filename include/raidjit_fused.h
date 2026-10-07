/*
 * raidjit integrated CRC (#38): encode the stripe and accumulate
 * CRC-32 over inputs/outputs in one cache-blocked pass.
 *
 * This header will grow with the follow-ups: #8 (fusion planner over
 * op sequences) and #10 (cache-blocked striping as a first-class API).
 * The mechanism is shared: the fused unit compiles its sub-operations
 * through the full rj_compile planner, then drives them from one
 * plan-time-sized tile loop. Data is read from DRAM once per tile;
 * follow-on passes over the same tile (CRC accumulation) hit hot
 * cache.
 *
 * Unchecked contract (inherited from rj_fn): every buffer 64-byte
 * aligned, no two buffers overlapping or aliasing (except documented
 * read-only sharing across fused sub-ops), len >= 1. Violations are
 * undefined behavior; debug builds assert the cheap checks.
 *
 * NT-store fence contract: when a sub-compile's winning twin uses
 * non-temporal stores, the caller must fence once per fused call before
 * any other agent reads the outputs (SFENCE on x86, DMB ISHST on
 * AArch64; RISC-V needs no fence) -- same rule as rj_fn, once per
 * call, not per tile or per sub-op.
 *
 * Threading: concurrent calls on one handle from any threads are safe
 * (per-call state lives on the stack). The entry points share their
 * handle's lifetime -- never call after the matching rj_free_* (a
 * trampoline slot may be reused by a newer handle).
 */
#ifndef RAIDJIT_FUSED_H
#define RAIDJIT_FUSED_H

#include "raidjit.h"

/* ---------------- fused ops (#8) ----------------
 * The fusion planner takes a short sequence of ops over one topology
 * and fuses the compatible ones into a single kernel invocation (one
 * entry point, one tile loop). Compatibility is checked up front; an
 * incompatible sequence fails loudly at compile time, never silently.
 *
 * Supported op kinds:
 *   RJ_FOP_ENCODE        - full-stripe encode (in: ndata, out: npar)
 *   RJ_FOP_PARITY_UPDATE - partial-stripe parity update
 *                          (in: 2*nchanged+npar, out: npar; the rj_fn
 *                          contract's layout, in-place update allowed)
 *   RJ_FOP_CRC           - CRC-32 over one earlier data op's buffers
 *                          (target = its sequence index)
 *
 * Buffer layout is flat, in sequence order: op k owns
 * in[base_in[k] .. base_in[k]+n_in[k]) and
 * out[base_out[k] .. base_out[k]+n_out[k]), with the per-op counts
 * from the rj_fn contract. Read-only input buffers MAY be shared
 * across sub-ops (e.g. the update's new[] aliasing the encode's in[]);
 * any other aliasing is undefined behavior.
 *
 * The planner fuses a sequence when every op compiles for the topology
 * and the dataflow allows a single pass: CRC ops must target an
 * earlier ENCODE/PARITY_UPDATE op; ENCODE and PARITY_UPDATE compose as
 * a verify pair (both parities are returned). Anything else (empty
 * sequence, unknown kind, CRC with no/bad target, duplicate CRC
 * targets) is rejected at compile time.
 */
typedef enum {
    RJ_FOP_ENCODE = 0,
    RJ_FOP_PARITY_UPDATE = 1,
    RJ_FOP_CRC = 2
} rj_fop_kind_t;

typedef struct {
    rj_fop_kind_t kind;
    const int *changed; /* PARITY_UPDATE only: data-disk indices, copied
                         * at compile time; nchanged >= 1 */
    int nchanged;
    int target;         /* CRC only: sequence index of the ENCODE or
                         * PARITY_UPDATE op whose buffers are checksummed */
} rj_fop_t;

typedef struct rj_fused rj_fused;
typedef void (*rj_fn_fused)(const void *const *in, void *const *out,
                            size_t len, uint32_t *crc /* NULL unless a
                            CRC op is present; then crc[n_in[t]+n_out[t]]
                            over the target op t's buffers: inputs first */);

typedef struct {
    int benchmark;      /* default 1 */
    size_t bench_bytes; /* 0 = default */
    int bench_iters;    /* <= 0 = default */
    int verbose;
    int loop_align;     /* 0 = auto; otherwise a power of two */
    int emit_endbr;
    int emit_shadow_stack;
    size_t nt_threshold;
    size_t prefetch_distance;
} rj_fuse_options;

void rj_fuse_options_default(rj_fuse_options *o);

rj_fused *rj_fuse_compile(const rj_topology *topo, const rj_fop_t *ops,
                          int nops, const rj_fuse_options *opts,
                          char *err, size_t errlen);
rj_fn_fused rj_get_fn_fused(const rj_fused *f);
void rj_free_fused(rj_fused *f);

/* Per-op buffer counts of the fused layout (sequence order). Returns
 * -1 / 0 on a NULL handle or out-of-range index. */
int rj_fused_op_nin(const rj_fused *f, int idx);
int rj_fused_op_nout(const rj_fused *f, int idx);
int rj_fused_nops(const rj_fused *f);
size_t rj_fused_tile_bytes(const rj_fused *f);

/* ---------------- cache-blocked striping (#10) ----------------
 * The plain rj_fn entry point, but the stripe is processed in
 * plan-time-sized tiles so multi-GB stripes stay in LLC. The tile size
 * comes from the static cache heuristic (see src/striping.h), chosen
 * once at compile time -- zero runtime cost for the decision.
 * Bit-exact vs the untiled rj_fn (same kernel, disjoint tiles).
 */
typedef struct rj_tiled rj_tiled;

rj_tiled *rj_compile_tiled(const rj_topology *topo, const rj_op *op,
                           const rj_options *opts, char *err, size_t errlen);
rj_fn rj_get_fn_tiled(const rj_tiled *t);
void rj_free_tiled(rj_tiled *t);
void rj_get_info_tiled(const rj_tiled *t, rj_info *info);
size_t rj_tiled_tile_bytes(const rj_tiled *t); /* 0 for a NULL handle */

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------- integrated CRC (#38) ----------------
 * One call encodes the stripe AND accumulates CRC-32/ISO-HDLC over the
 * input data buffers and the parity output buffers. crc_in[i] receives
 * the CRC-32 of in[i] (ndata entries) and crc_out[j] the CRC-32 of
 * out[j] (npar entries); either pointer may be NULL to skip that side
 * (when npar == 0, crc_out must be NULL). The CRC is the planner-
 * selectable variant of the encode pipeline: the encode kernel is the
 * rj_compile winner (full enumeration/pruning/benchmarking); the
 * accumulation rides on the same cache-blocked tile pass, so inputs
 * are read from DRAM once.
 */
typedef struct rj_jit_crc rj_jit_crc;
typedef void (*rj_fn_crc)(const void *in[], void *out[], size_t len,
                          uint32_t *crc_in, uint32_t *crc_out);

typedef struct {
    int benchmark;      /* default 1: same meaning as rj_options */
    size_t bench_bytes; /* 0 = default */
    int bench_iters;    /* <= 0 = default */
    int verbose;
    int loop_align;     /* 0 = auto; otherwise a power of two */
    int emit_endbr;
    int emit_shadow_stack;
    size_t nt_threshold;    /* 0 = never use NT stores */
    size_t prefetch_distance; /* 0 = analytical default */
} rj_crc_options;

void rj_crc_options_default(rj_crc_options *o);

/* Compile the CRC-integrated encode pipeline for topo (encode op).
 * Returns NULL with err set on failure (same failures as rj_compile). */
rj_jit_crc *rj_compile_crc(const rj_topology *topo,
                           const rj_crc_options *opts,
                           char *err, size_t errlen);
rj_fn_crc rj_get_fn_crc(const rj_jit_crc *j);
void rj_free_crc(rj_jit_crc *j);
void rj_get_info_crc(const rj_jit_crc *j, rj_info *info);

/* Observability (plan-time decisions, fixed at compile): */
int rj_crc_ndata(const rj_jit_crc *j);
int rj_crc_npar(const rj_jit_crc *j);
size_t rj_crc_tile_bytes(const rj_jit_crc *j); /* 0 for a NULL handle */

#ifdef __cplusplus
}
#endif

#endif /* RAIDJIT_FUSED_H */
