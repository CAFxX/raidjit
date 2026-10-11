#pragma once
#include "plan.h"
#include "../include/raidjit.h"

/* BUG-COMBO-STATIC-F16: twin effect postcondition. Each backend sets a bit
 * in its cg_t.twin_applied when the twin's code path is actually taken
 * during codegen (not merely when the flag is set on the candidate).
 * After codegen, jit_build_version rejects any requested twin whose bit
 * is not set -- a phantom twin (requested but no effect) fails cleanly
 * instead of being recorded under a false label. The bits are copied to
 * version_t.twin_applied so API layers can report effective twins.
 * CLASS HUNT: any new cand_t twin axis needs a bit here, a set-point in
 * each backend that implements it, and a postcondition entry. */
#define TWIN_APPLIED_NT         (1u << 0)
#define TWIN_APPLIED_PREFETCH   (1u << 1)
#define TWIN_APPLIED_TERNLOG    (1u << 2)
#define TWIN_APPLIED_REMAT      (1u << 3)
#define TWIN_APPLIED_XORSCHED   (1u << 4)
#define TWIN_APPLIED_XORBIT     (1u << 5)
#define TWIN_APPLIED_MSHIFT     (1u << 6)
#define TWIN_APPLIED_PERMI2B    (1u << 7)
#define TWIN_APPLIED_PIPE       (1u << 8)
#define TWIN_APPLIED_REC_CF     (1u << 9)
#define TWIN_APPLIED_CRC_FOLD   (1u << 10)

/* One JITed code version (one ISA x unroll x multiply-method point). */
typedef struct {
    cand_t cand;
    plan_kind_t kind;
    void *mem;      /* executable mapping */
    size_t memsz;
    rj_fn fn;
    double uops_per_byte; /* static model: vector uops per stripe byte */
    double mem_per_byte;  /* static model: bytes moved per stripe byte */
    size_t code_size, const_size;
    double gbps;          /* measured or heuristically estimated GB/s */
    int gbps_measured;    /* 1 if gbps came from real benchmarks */
    int vl;               /* vector length in bytes (SVE: runtime VL/8) */
    int notail;           /* 1 = no tail code emitted: only valid for
                           * len % group_bytes == 0 (length-dispatch twin) */
    uint32_t twin_applied; /* BUG-COMBO-STATIC-F16: bitmask of twins whose
                            * code path was actually taken during codegen
                            * (TWIN_APPLIED_* bits, see above). Set by
                            * jit_build_version from the backend's cg_t. */
    int predtail;         /* 1 = fully predicated vector tail (SVE whilelt
                           * twin, TAIL_MASKED): serves tail lengths only
                           * (dispatch.h). Disjoint from notail. */
    int maskedtail;       /* 1 = masked vector tail (x86-64 k1 twin,
                           * RVV v0-masked twin; TAIL_MASKED): serves tail
                           * lengths only (dispatch.h). Disjoint from
                           * notail/predtail. Set only by api.c from the
                           * winning candidate (never by plan_candidates). */
    size_t group_bytes;   /* true full-iteration group bytes (U*VL; on
                           * RISC-V U*vlenb*lmul read at build time) */
    char desc[192];
    const plan_t *plan; /* borrowed reference to the plan this version
                         * was built from, set by jit_build_version.
                         * Valid only during compilation: the plan owner
                         * (rj_compile / the test) frees it, and
                         * jit_free_version never frees it. Lets
                         * bench_select validate each version against
                         * its OWN scalar reference when the candidate
                         * pool mixes plans (track A5: one pool over
                         * several survivor-set plans). NULL for
                         * versions built before this field existed;
                         * bench_select then falls back to the shared
                         * plan. Never dereferenced after compilation. */
} version_t;

int jit_build_version(const plan_t *pl, const cand_t *cand,
                      const rj_cpu_info *cpu, const rj_options *opts,
                      version_t *v, char *err, size_t errlen);
void jit_free_version(version_t *v);
