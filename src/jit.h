#pragma once
#include "plan.h"
#include "../include/raidjit.h"

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
    int predtail;         /* 1 = fully predicated vector tail (SVE whilelt
                           * twin, TAIL_MASKED): serves tail lengths only
                           * (dispatch.h). Disjoint from notail. */
    int maskedtail;       /* 1 = masked AVX-512 vector tail (x86-64 k1 twin,
                           * TAIL_MASKED): serves tail lengths only
                           * (dispatch.h). Disjoint from notail/predtail.
                           * Set only by api.c from the winning candidate
                           * (never by plan_candidates). */
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
