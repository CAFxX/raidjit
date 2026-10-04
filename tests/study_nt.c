/* NT-store empirical study: temporal vs non-temporal stores.
 *
 * Two scenarios:
 *  1. Cold streaming: buffers >> LLC (128MB), write once, no reuse.
 *     NT stores should win (avoid cache pollution, no RFO).
 *  2. Immediate readback: write then immediately read the output.
 *     Temporal should win (NT bypasses cache, forcing DRAM readback).
 *
 * This informs the nt_threshold selection policy.
 */
#include "../include/raidjit.h"

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

int main(void)
{
    /* RAID-6(8): 8 data + 2 parity */
    rj_topology *t = rj_topo_raid6(8);
    rj_op op = rj_op_encode();
    char err[512];

    /* temporal version */
    rj_options ot; rj_options_default(&ot);
    ot.benchmark = 0; ot.nt_threshold = 0;
    rj_jit *jt = rj_compile(t, &op, &ot, err, sizeof(err));
    if (!jt) { printf("temporal compile failed: %s\n", err); return 1; }

    /* NT version: force NT by setting nt_threshold and benchmark=1,
     * then pick the NT twin from candidates */
    rj_options on; rj_options_default(&on);
    on.benchmark = 1; on.nt_threshold = 1;
    rj_jit *jn = rj_compile(t, &op, &on, err, sizeof(err));
    if (!jn) { printf("NT compile failed: %s\n", err); return 1; }

    rj_info it, in;
    rj_get_info(jt, &it);
    rj_get_info(jn, &in);
    printf("temporal: %s\n", it.desc);
    printf("NT:       %s\n", in.desc);

    rj_fn ft = rj_get_fn(jt);
    rj_fn fn = rj_get_fn(jn);

    /* Scenario 1: cold streaming, 128MB per buffer (>> 32MB LLC) */
    size_t len = 128 * 1024 * 1024;
    int nd = 8, np = 2;
    uint8_t *d[8], *p_t[2], *p_n[2];
    for (int i = 0; i < nd; i++) {
        d[i] = alloc_aligned(len);
        for (size_t k = 0; k < len; k += 4096)
            d[i][k] = (uint8_t)(i + k);
    }
    for (int j = 0; j < np; j++) {
        p_t[j] = alloc_aligned(len);
        p_n[j] = alloc_aligned(len);
    }

    const void *in_d[8];
    for (int i = 0; i < nd; i++) in_d[i] = d[i];
    void *out_t[2], *out_n[2];
    for (int j = 0; j < np; j++) { out_t[j] = p_t[j]; out_n[j] = p_n[j]; }

    /* warmup */
    ft(in_d, out_t, len);
    fn(in_d, out_n, len);

    /* cold streaming: flush not needed, buffers >> LLC ensure cold */
    int reps = 5;
    uint64_t t0 = now_ns();
    for (int r = 0; r < reps; r++)
        ft(in_d, out_t, len);
    uint64_t t1 = now_ns();
    for (int r = 0; r < reps; r++)
        fn(in_d, out_n, len);
    uint64_t t2 = now_ns();

    double gb = (double)len * (nd + np) * reps / 1e9;
    double s_t = (t1 - t0) / 1e9, s_n = (t2 - t1) / 1e9;
    printf("\nCold streaming (128MB buffers, %d reps):\n", reps);
    printf("  temporal: %.2f GB/s\n", gb / s_t);
    printf("  NT:       %.2f GB/s\n", gb / s_n);
    printf("  NT speedup: %.2fx\n", (gb / s_n) / (gb / s_t));

    /* verify correctness */
    if (memcmp(p_t[0], p_n[0], len) != 0 || memcmp(p_t[1], p_n[1], len) != 0)
        printf("  WARNING: outputs differ!\n");
    else
        printf("  outputs match\n");

    /* Scenario 2: immediate readback (64KB, fits in cache) */
    size_t len2 = 64 * 1024;
    uint8_t *d2[8], *p2_t[2], *p2_n[2];
    for (int i = 0; i < nd; i++) {
        d2[i] = alloc_aligned(len2);
        memset(d2[i], i + 1, len2);
    }
    for (int j = 0; j < np; j++) {
        p2_t[j] = alloc_aligned(len2);
        p2_n[j] = alloc_aligned(len2);
    }
    const void *in2[8];
    for (int i = 0; i < nd; i++) in2[i] = d2[i];
    void *o2t[2], *o2n[2];
    for (int j = 0; j < np; j++) { o2t[j] = p2_t[j]; o2n[j] = p2_n[j]; }

    reps = 10000;
    volatile uint64_t sink = 0;
    t0 = now_ns();
    for (int r = 0; r < reps; r++) {
        ft(in2, o2t, len2);
        /* immediate readback: sum the outputs */
        for (int j = 0; j < np; j++)
            for (size_t k = 0; k < len2; k += 64)
                sink += p2_t[j][k];
    }
    t1 = now_ns();
    for (int r = 0; r < reps; r++) {
        fn(in2, o2n, len2);
        for (int j = 0; j < np; j++)
            for (size_t k = 0; k < len2; k += 64)
                sink += p2_n[j][k];
    }
    t2 = now_ns();
    (void)sink;

    gb = (double)len2 * (nd + np) * reps / 1e9;
    s_t = (t1 - t0) / 1e9; s_n = (t2 - t1) / 1e9;
    printf("\nImmediate readback (64KB buffers, %d reps, write+read):\n", reps);
    printf("  temporal: %.2f GB/s\n", gb / s_t);
    printf("  NT:       %.2f GB/s\n", gb / s_n);
    printf("  NT speedup: %.2fx ( <1 means temporal wins )\n",
           (gb / s_n) / (gb / s_t));

    rj_free(jt); rj_free(jn);
    rj_op_free(&op); rj_topo_free(t);
    return 0;
}
