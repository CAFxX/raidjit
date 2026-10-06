/* Plan lowering: (topology, operation) -> execution strategy + version
 * candidates.
 *
 * Strategy choice is algorithmic, not just ISA tuning:
 *   P_XOR    - every coefficient is 0/1 (RAID-5 P, single-disk rebuild via
 *              XOR parity, mirrors): pure XOR reduction, no GF multiplies.
 *   P_HORNER - the coefficient matrix is Vandermonde in Horner order
 *              (row j = [c_j^(n-1) ... c_j^0], incl. default RS matrices and
 *              RAID-6): each parity row accumulates as
 *              acc = acc*c_j ^ data, i.e. ONE GF multiply per input per row
 *              instead of a per-(input,row) table multiply.
 *   P_MULACC - general case: out[j] ^= in[i] * M[j][i] with per-constant
 *              nibble-table (pshufb) or GFNI affine multiplies.
 *   P_DELTA  - partial-stripe parity update: out[j] = oldpar[j] ^
 *              (over changed i) ((old[i] ^ new[i]) * M[j][changed[i]]).
 *              Same multiply-accumulate machinery as P_MULACC, but the
 *              per-input value is the old/new delta and accumulators are
 *              seeded from the old parities instead of zero.
 *
 * A "recover" op is first reduced to a plain linear combination of the
 * surviving disks (solve_recovery), so reconstruction is a single
 * encode-like pass -- the decode (system solve) happens once at JIT time,
 * not per stripe.
 */
#pragma once
#include <stdint.h>
#include "../include/raidjit.h"
#include "cpu.h"

#define RJ_MAX_DISKS 256

typedef enum { P_XOR, P_MULACC, P_HORNER, P_DELTA } plan_kind_t;

typedef enum {
    ISA_SSE = 0,
    ISA_AVX2 = 1,
    ISA_AVX512 = 2,
    ISA_NEON = 3,      /* AArch64 Advanced SIMD, 16-byte vectors */
    ISA_SVE = 4,       /* AArch64 SVE/SVE2, VL-agnostic */
    ISA_RVV = 5,       /* RISC-V Vector, VL-agnostic */
    ISA_PEXT = 6       /* RISC-V P-extension packed SIMD */
} isa_t;
typedef enum { MUL_NIBBLE = 0, MUL_GFNI = 1, MUL_ZVBC = 2 } mul_t;

typedef struct {
    isa_t isa;
    mul_t mul; /* P_MULACC / P_HORNER / P_DELTA only */
    int U;     /* unroll: vector-widths per group iteration */
    int nt;    /* 1 = non-temporal (streaming) output stores. Fence
                * contract is per-ISA: x86 emits no fence (caller must
                * SFENCE), AArch64 emits no DMB (caller must DMB ISHST),
                * RISC-V uses the Zihintntl NTL.ALL hint which does not
                * change memory ordering (no caller fence needed). */
    size_t prefetch_dist; /* 0 = no SW prefetch; else PREFETCHT0 each
                           * input stream this many bytes ahead */
    int ternlog; /* 1 = use VPTERNLOGD for AVX-512 XOR-reduction: for
                  * P_MULACC/P_DELTA this folds the nibble path's
                  * acc ^= r1 ^ r2 into one ternary-logic uop; for P_XOR
                  * (only when n_out >= 2, where the shared in[a] load
                  * amortizes) it folds acc ^= in[a] ^ in[b] (column
                  * pairs, in[b] as the memory operand) into one uop.
                  * The benchmark picks the twin empirically; never
                  * forced. */
    int remat;   /* 1 = HORNER: rematerialize tables from pool into a
                  * scratch register instead of keeping hreg live
                  * (trades L1 loads for registers, allowing larger U) */
    int notail;  /* 1 = length-dispatch twin: omit the tail path entirely
                  * (scalar tail on x86-64/AArch64, strip-mining remainder
                  * on RISC-V). Only valid for len % group == 0; the
                  * dispatcher (src/dispatch.c) guarantees it. Never set
                  * by plan_candidates: api.c builds the twin from the
                  * winning candidate after selection. See
                  * docs/length_dispatch.md. */
    int predtail; /* 1 = length-dispatch twin: replace the scalar tail with
                   * a fully predicated vector tail (SVE only, via whilelt;
                   * track B1sve, SOTA item 14). Serves tail lengths only
                   * (TAIL_MASKED, dispatch.h); the scalar general variant
                   * defers to it when it is installed. Disjoint from
                   * notail: a version is never both. Never set by
                   * plan_candidates: api.c builds the twin from the
                   * winning SVE candidate after selection, and the
                   * compile-time benchmark decides between tail variants
                   * empirically. */
    int maskedtail; /* 1 = length-dispatch twin: replace the scalar tail
                   * with a masked AVX-512 vector tail (x86-64 only, via
                   * opmask k1; track B1x86, SOTA item 14). Serves tail
                   * lengths only (TAIL_MASKED, dispatch.h); the scalar
                   * general variant defers to it when it is installed.
                   * Disjoint from notail/predtail: a version is never
                   * both. Never set by plan_candidates: api.c builds the
                   * twin from the winning AVX-512 candidate after
                   * selection, gated by its own tail-duel benchmark
                   * (bench_tail_duel, src/bench.c). */
    int permi2b; /* 1 = P_MULACC/P_DELTA nibble twin: look the nibbles up
                   * with VPERMI2B (AVX512_VBMI) instead of the two-load
                   * VPSHUFB ping-pong (track B2, SOTA item 15). The two
                   * 16-entry tables travel interleaved in one pool entry
                   * (CK_NIBBOTH); the +16 high-nibble index offset folds
                   * into a memory-source VPORB. Never set by
                   * plan_candidates: api.c builds the twin from the
                   * stage-1 winning candidate after selection (stage-2
                   * duel with the standard bench_select protocol, gated
                   * on cpu->vbmi), like the notail/maskedtail/predtail
                   * twins. The stage-2 benchmark picks the twin
                   * empirically, never forced. */
    int xorsched; /* 1 = P_XOR: drive the body from the plan's xor_sched
                   * program (cross-row CSE) instead of the naive per-row
                   * emitter. Enumerated only when the static model
                   * predicts a strict uop win; the benchmark picks the
                   * twin empirically, never forced. */
    int xorbit; /* 1 = P_MULACC/P_DELTA/P_HORNER: drive the body from the
                 * plan's bit_sched program (bit-level cross-row and
                 * cross-coefficient CSE, track A1) instead of the naive
                 * nibble emitter. Enumerated only when the static model
                 * predicts a strict uop win; the benchmark picks the
                 * twin empirically, never forced. Never combined with
                 * ternlog/permi2b/remat/xorsched. */
    int lmul;    /* RVV only: vector register group multiplier (LMUL).
                  * 1/2/4/8 (0 = 1, the default for zero-initialized
                  * candidates). Enumerated by plan_candidates for RVV;
                  * the backend maps logical vector register r to the
                  * lmul-aligned physical group head r*lmul, and the
                  * cost model amortizes over lmul*NOMVL bytes per
                  * instruction. Pinned in prune.c's twin_dominates like
                  * every other axis. x86/AArch64 backends reject
                  * lmul > 1. */
} cand_t;

typedef struct {
    plan_kind_t kind;
    int n_in, n_out;
    uint8_t *mat;      /* n_out * n_in, row-major; 0/1 entries for P_XOR.
                        * P_DELTA: n_out * nchanged extracted delta columns
                        * M[j][changed[i]] (see nchanged). */
    uint8_t *horner_c; /* P_HORNER: c_j per output row */
    int nchanged;      /* P_DELTA: number of changed data disks; input layout
                        * is [old_0..old_{nc-1}, new_0..new_{nc-1},
                        * oldpar_0..oldpar_{np-1}] */
    int noop;          /* n_out == 0: emit trivial function */
    char name[128];    /* e.g. "raid6 enc" / "rs(6,3) rec d0+d1";
                          * sized for the "[rows ...]" suffix; opname is
                          * t->name (32B) so %.31s below never truncates
                          * a real name */
    struct xorsched *xor_sched; /* P_XOR cross-row CSE program (NULL when
                                 * there is nothing to share); built by
                                 * plan_build, freed by plan_free. The
                                 * xorsched candidate twin reads it. */
    struct xorbit *bit_sched; /* P_MULACC/P_DELTA/P_HORNER bit-level CSE
                               * program (track A1; NULL when there is
                               * nothing to share); built by plan_build,
                               * freed by plan_free. The xorbit candidate
                               * twin reads it. */
} plan_t;

typedef struct {
    int ndata, npar;
    uint8_t *mat; /* npar*ndata or NULL when npar == 0 */
} topo_int_t;

/* Build the internal view from a public topology (copies the matrix). */
int topo_int_build(topo_int_t *ti, const rj_topology *t, char *err,
                   size_t errlen);
void topo_int_free(topo_int_t *ti);

/* Short name like "raid6(8)" / "rs(6,3)" for plan naming. */
const char *topo_internal_name(const rj_topology *t);

/* Build a plan from topology + op. Returns 0 on success, -1 on error
 * (err filled). */
int plan_build(const topo_int_t *topo, const rj_op *op, const char *opname,
               plan_t *pl, char *err, size_t errlen);
void plan_free(plan_t *pl);

/* Validate + sort/dedupe a recover op's failed-disk list (shared by
 * plan_build and the A5 survivor path). Returns the deduped count
 * (>= 1) with *F_out the sorted list (caller frees), or -1 on error
 * with err set. */
int plan_recover_failed_list(const rj_op *op, int nd, int np, int **F_out,
                             char *err, size_t errlen);

/* Build a recovery plan from an explicit parity-row equation set
 * (track A5: cheapest measured survivor set). F[0..nf-1] is the
 * validated sorted/deduped failed-disk list (1 <= nf <= npar);
 * rows[0..nrows-1] are the parity rows used as solve equations
 * (nrows >= nf; a failed parity disk j forces some rows[r] == j).
 * The plan name records the row set ("... [rows 0+2]"). Returns 0 on
 * success, -1 on error (err filled); singular row sets fail here so
 * the enumerator can skip them. plan_build's recover path is the
 * all-rows special case (rows = 0..npar-1). */
int plan_build_recover_rows(const topo_int_t *topo, const int *F, int nf,
                            const int *rows, int nrows, const char *opname,
                            plan_t *pl, char *err, size_t errlen);

/* Closed-form small-erasure recovery rows (track A4, phase 1). Fills R
 * (nf x ns, row-major) directly from the e=1/e=2 closed forms when the
 * failure pattern admits it; returns 1. Returns 0 (R untouched) when
 * the closed form does not apply (nf outside 1..2, bad row indices,
 * singular pivot) -- the caller falls back to solve_recovery_rows. The
 * pivot choice replicates solve_recovery_rows, so an applied closed
 * form is bit-identical to the general solve. All inverses (c_f^-1,
 * det^-1) are plan-time constants baked into R. */
int plan_recover_closed_form(const topo_int_t *topo, const int *F, int nf,
                             const int *S, int ns, const int *rows, int nrows,
                             uint8_t *R);

/* Enumerate version candidates for a plan on this CPU. Returns count. */
/* allow_nt: also emit a non-temporal twin of every candidate (the
 * jitted code emits no fence; on x86 the caller must SFENCE and on
 * AArch64 the caller must DMB ISHST after the call, while on RISC-V
 * the Zihintntl hint needs no caller fence -- see cand_t.nt).
 * prefetch_dist: 0 = no prefetch twins; RJ_PREFETCH_AUTO = per-candidate
 * analytical default from the once-per-machine calibration (I4, see
 * src/prefetch_calib.h); any other nonzero value = manual uniform
 * override applied to every prefetch twin. Twins are appended after
 * their temporal originals, so static (non-benchmark) selection, which
 * prefers the first of equal-cost candidates, keeps temporal behavior.
 * The distance-0 (no-prefetch) base twin is always enumerated first, so
 * it stays a candidate even when prefetch twins are added. */
#define RJ_PREFETCH_AUTO ((size_t)-1)
int plan_candidates(const plan_t *pl, const rj_cpu_info *cpu,
                    cand_t *out, int maxout, int allow_nt,
                    size_t prefetch_dist);

const char *isa_name(isa_t isa);
int isa_nregs(isa_t isa);
int isa_vl(isa_t isa); /* vector length in bytes */

/* Exact number of vector registers the backend assigns for this
 * (isa, mul, plan, U), matching the backend's register assignment
 * instruction-for-instruction (accumulators n_out*U plus the backend's
 * temp/table registers). The planner drops candidates with need >
 * isa_nregs(isa); the backends assert their assignment against this so
 * planner and backend can never drift. mul is ignored for P_XOR and for
 * ISAs with a single multiply method. For RVV the count is in LOGICAL
 * vector registers: the backend maps logical r to the lmul-aligned
 * physical group head r*lmul, so callers scale by cand->lmul before
 * comparing against isa_nregs(ISA_RVV). */
int plan_need_regs(isa_t isa, mul_t mul, const plan_t *pl, int U, int remat);

/* Per-microarchitecture port-pressure weights for the static cost
 * model (I3). Each t_* is cycles per instruction -- the reciprocal
 * throughput -- for the named AVX-512 op class on that uarch, taken
 * from published port data (uops.info per-microarchitecture pages and
 * vendor documentation; see the table in src/plan.c for per-value
 * sources). vec_bw is the generic vector-uop dispatch width
 * (uops/cycle), used as the fallback for candidate classes without a
 * characteristic kernel model. mem_cpb is cycles per memory byte
 * (load-dominated). freq512 is the sustained-frequency derate applied
 * to 512-bit-vector candidates (Intel license-based downclocking under
 * sustained AVX-512; an estimate, >= 1.0). bw_bpc is the peak
 * single-core sustainable DRAM streaming bandwidth, bytes/cycle
 * (#31): a conservative static per-uarch-class figure (est., NOT
 * measured) feeding the bandwidth leg of the roofline cost.
 *
 * These are STATIC ESTIMATES, not measurements: they feed heuristic
 * ordering and the prune-soundness gate only. The benchmark remains
 * the final arbiter (Carlo's invariant). */
typedef struct {
    double t_ternlog; /* VPTERNLOGD zmm */
    double t_shuffle; /* VPSHUFB zmm */
    double t_xor;     /* VPXORD zmm-class logic */
    double t_gfni;    /* VGF2P8MULB / VGF2P8AFFINEQB zmm */
    double t_ld64;    /* 64-byte vector load */
    double vec_bw;    /* generic vector-uop dispatch width (uops/cycle) */
    double mem_cpb;   /* cycles per memory byte */
    double freq512;   /* 512-bit sustained-frequency derate (est., >= 1) */
    double bw_bpc;    /* peak single-core DRAM streaming bandwidth,
                       * bytes/cycle (est., NOT measured; #31) */
} rj_uarch_weights_t;

/* Weights for a uarch tag (never NULL; unknown tags fall back to the
 * neutral entry). */
const rj_uarch_weights_t *rj_uarch_weights(rj_uarch_t u);

/* Uarch-aware static cost estimate (cycles/byte, est.) for a candidate:
 * a roofline blend (#31) of the I3 compute leg and a bandwidth leg.
 * The compute leg re-weights the static (uops, mem) model by the
 * candidate's characteristic op mix at this uarch's published
 * throughputs (port pressure, not raw uop counts); the bandwidth leg
 * is mem_per_byte / bw_bpc with bw_bpc the uarch's conservative
 * static single-core streaming bandwidth (bytes/cycle, est. -- NOT
 * measured). With arithmetic intensity AI = uops_per_byte /
 * mem_per_byte and ridge point ridge = (1/w_uop) / bw_bpc (peak
 * uops/cycle over peak bytes/cycle), the blend is the continuous
 * p-norm (t_compute^4 + t_mem^4)^{1/4}: below the ridge
 * (bandwidth-bound) the cost is ~bytes moved, above it
 * (compute-bound) ~uop cost, with no hard switch. Monotone in both
 * legs (load-bearing for the prune gate's roofline leg). The op mix
 * is structural per (kind, isa, mul, ternlog) class, read off the
 * emitter sequences (see src/plan.c); classes without a kernel model
 * fall back to the generic dispatch width. Intended for heuristic
 * candidate ordering (static only -- never a substitute for the
 * benchmark). */
double plan_uarch_cost(double uops_per_byte, double mem_per_byte,
                       plan_kind_t kind, const cand_t *cand, rj_uarch_t u);

/* Coefficient-triviality predicate: 1 iff some coefficient the vector
 * nibble path would multiply by lies outside {0,1} -- i.e. the nibble
 * split, its 0x0F mask register, and the table temps are really needed.
 * P_MULACC scans the full matrix, P_DELTA its n_out x nchanged delta
 * columns, P_HORNER the row coefficients; P_XOR is all-{0,1} by
 * construction (0). A NULL matrix/coefficient vector (never produced by
 * plan_build; only hand-built test plans) is treated as nontrivial:
 * nothing is proven dead, so the conservative count stands. This is the
 * single source of truth shared by plan_need_regs and every backend's
 * need_mask computation, so the cost model and the emitter cannot
 * diverge. */
int plan_any_nontrivial(const plan_t *pl);
