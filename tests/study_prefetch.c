/* Prefetch empirical study: does PREFETCHT0 help, and at what distance?
 *
 * Builds the benchmark-winning temporal candidate for RAID-6(8), then
 * builds prefetch twins at several distances via the internal
 * plan_candidates/jit_build_version APIs, and times:
 *  1. Cold streaming: 128MB buffers (>> LLC), write once.
 *  2. In-cache: 256KB buffers, repeated.
 * Outputs are verified against the non-prefetch twin.
 */
#include "../include/raidjit.h"
#include "../src/plan.h"
#include "../src/jit.h"
#include "../src/cpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static void *alloc_aligned(size_t n)
{
    void *p = NULL;
    if (posix_memalign(&p, 64, n))
        abort();
    return p;
}

static double bench_cold(rj_fn fn, size_t len, int reps)
{
    /* Fresh buffers per rep: truly cold (>> LLC). */
    double best = 0;
    for (int r = 0; r < reps; r++) {
        uint8_t *din[8], *dout[2];
        const void *in[8];
        void *out[2];
        for (int i = 0; i < 8; i++) {
            din[i] = alloc_aligned(len);
            memset(din[i], i + r + 1, len);
            in[i] = din[i];
        }
        for (int j = 0; j < 2; j++) {
            dout[j] = alloc_aligned(len);
            memset(dout[j], 0, len); /* pre-fault: avoid page faults in timing */
            out[j] = dout[j];
        }
        uint64_t t0 = now_ns();
        fn(in, out, len);
        uint64_t t1 = now_ns();
        double gb = (double)len * 10 / 1e9;
        double s = (t1 - t0) / 1e9;
        if (gb / s > best) best = gb / s;
        for (int i = 0; i < 8; i++) free(din[i]);
        for (int j = 0; j < 2; j++) free(dout[j]);
    }
    return best;
}

int main(void)
{
    rj_topology *t = rj_topo_raid6(8);
    char err[512];
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);

    /* Build plan, get the static-model winner (temporal, no prefetch) */
    topo_int_t ti;
    if (topo_int_build(&ti, t, err, sizeof err)) {
        printf("topo failed: %s\n", err); return 1;
    }
    rj_op op = rj_op_encode();
    plan_t pl;
    if (plan_build(&ti, &op, "raid6(8)", &pl, err, sizeof err)) {
        printf("plan failed: %s\n", err); return 1;
    }
    cand_t cands[64];
    int nc = plan_candidates(&pl, &cpu, cands, 64, 0, 0);
    if (nc <= 0) { printf("no candidates\n"); return 1; }

    rj_options opts;
    rj_options_default(&opts);
    opts.benchmark = 0;

    /* base version = first candidate (static-model order, best first) */
    version_t base;
    memset(&base, 0, sizeof base);
    if (jit_build_version(&pl, &cands[0], &cpu, &opts, &base, err, sizeof err)) {
        printf("base build failed: %s\n", err); return 1;
    }
    printf("base: %s\n", base.desc);

    size_t dists[] = {0, 256, 512, 1024, 2048, 4096, 8192};
    int nd = sizeof(dists) / sizeof(dists[0]);

    size_t len_cold = 128 * 1024 * 1024;
    size_t len_cache = 256 * 1024;
    const int ndisk = 8, npar = 2;

    uint8_t *din[8], *dout[2], *ref[2];
    for (int i = 0; i < ndisk; i++) {
        din[i] = alloc_aligned(len_cold);
        memset(din[i], i + 1, len_cold);
    }
    for (int j = 0; j < npar; j++) {
        dout[j] = alloc_aligned(len_cold);
        ref[j] = alloc_aligned(len_cold);
    }
    const void *in[8];
    void *out[2], *rout[2];
    for (int i = 0; i < ndisk; i++) in[i] = din[i];
    for (int j = 0; j < npar; j++) { out[j] = dout[j]; rout[j] = ref[j]; }

    /* reference output from base */
    {
        rj_fn fb = (rj_fn)base.fn;
        fb(in, rout, len_cold);
    }

    printf("\n%-10s %12s %12s\n", "dist", "cold GB/s", "cache GB/s");
    for (int di = 0; di < nd; di++) {
        cand_t c = cands[0];
        c.prefetch_dist = dists[di];
        version_t v;
        memset(&v, 0, sizeof v);
        if (jit_build_version(&pl, &c, &cpu, &opts, &v, err, sizeof err)) {
            printf("dist %zu: build failed: %s\n", dists[di], err);
            continue;
        }
        rj_fn fn = (rj_fn)v.fn;
        /* correctness */
        fn(in, out, len_cold);
        int ok = memcmp(dout[0], ref[0], len_cold) == 0 &&
                 memcmp(dout[1], ref[1], len_cold) == 0;
        double cold = bench_cold(fn, len_cold, 5);
        /* cache: warm buffers, many reps */
        double cache;
        {
            fn(in, out, len_cache); /* warmup */
            uint64_t t0 = now_ns();
            for (int r = 0; r < 200; r++)
                fn(in, out, len_cache);
            uint64_t t1 = now_ns();
            double gb = (double)len_cache * 10 * 200 / 1e9;
            cache = gb / ((t1 - t0) / 1e9);
        }
        printf("%-10zu %12.2f %12.2f  %s\n",
               dists[di], cold, cache, ok ? "OK" : "MISMATCH");
        jit_free_version(&v);
        if (!ok) { printf("correctness failure at dist %zu\n", dists[di]); return 1; }
    }

    jit_free_version(&base);
    plan_free(&pl);
    topo_int_free(&ti);
    rj_op_free(&op);
    rj_topo_free(t);
    return 0;
}
