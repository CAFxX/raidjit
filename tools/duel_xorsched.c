/* Quiet-system duel: naive vs +xsched twin through the real bench_select
 * harness (track A3, D3).
 *
 * Builds the sharing-heavy 20-in/4-out P_XOR config's naive and scheduled
 * versions for one (ISA, U), then runs bench_select on just that pair
 * `runs` times, collecting per-arm aggregate GB/s. Reports median and
 * IQR per arm. Run taskset-pinned on a quiet system.
 *
 * Usage: duel_xorsched <runs> <bench_bytes>
 *   bench_bytes=65536  -> sizes {16K, 64K, 256K} (64 KiB cache-fit)
 *   bench_bytes=1048576 -> sizes {256K, 1M, 4M} (1 MiB streaming)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "raidjit.h"
#include "plan.h"
#include "jit.h"
#include "bench.h"
#include "cpu.h"
#include "xorsched.h"

/* t1: sharing-heavy 4x20, 2 shared pairs -> 2 temps (row-major) */
static const uint8_t t1_mat[80] = {
    1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,
    1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,
    0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,
};

static int cmp_d(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv)
{
    int runs = argc > 1 ? atoi(argv[1]) : 12;
    size_t bench_bytes = argc > 2 ? (size_t)atol(argv[2]) : 65536;
    if (runs < 1)
        runs = 12;

    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    printf("cpu: avx512f=%d avx2=%d sse2=%d\n", cpu.avx512f, cpu.avx2,
           cpu.sse2);

    isa_t isa = ISA_AVX512;
    int U = 4;
    if (!cpu.avx512f) {
        printf("no AVX-512 on this host, cannot run the AVX-512 U=4 duel\n");
        return 2;
    }

    plan_t pl;
    memset(&pl, 0, sizeof(pl));
    pl.kind = P_XOR;
    pl.n_in = 20;
    pl.n_out = 4;
    pl.mat = malloc(80);
    memcpy(pl.mat, t1_mat, 80);
    snprintf(pl.name, sizeof(pl.name), "duel(20,4)");
    pl.xor_sched = xorsched_build(&pl);
    if (!pl.xor_sched) {
        printf("no schedule built\n");
        return 1;
    }
    printf("schedule: %d temps\n", pl.xor_sched->ntemps);

    rj_options opts;
    memset(&opts, 0, sizeof(opts));
    opts.bench_bytes = bench_bytes;
    opts.bench_iters = 11;
    char err[256];

    version_t vs[2];
    cand_t cn;
    memset(&cn, 0, sizeof(cn));
    cn.isa = isa;
    cn.mul = MUL_NIBBLE;
    cn.U = U;
    if (jit_build_version(&pl, &cn, &cpu, &opts, &vs[0], err,
                          sizeof(err)) != 0) {
        printf("naive build failed: %s\n", err);
        return 1;
    }
    cand_t cs = cn;
    cs.xorsched = 1;
    if (jit_build_version(&pl, &cs, &cpu, &opts, &vs[1], err,
                          sizeof(err)) != 0) {
        printf("sched build failed: %s\n", err);
        return 1;
    }
    printf("naive: %s uops/B=%.4f\n", vs[0].desc, vs[0].uops_per_byte);
    printf("sched: %s uops/B=%.4f\n", vs[1].desc, vs[1].uops_per_byte);

    double *gnaive = malloc((size_t)runs * sizeof(double));
    double *gsched = malloc((size_t)runs * sizeof(double));
    for (int r = 0; r < runs; r++) {
        int w = bench_select(vs, 2, &pl, &opts, 0, NULL, err, sizeof(err));
        if (w < 0) {
            printf("bench_select failed: %s\n", err);
            return 1;
        }
        gnaive[r] = vs[0].gbps;
        gsched[r] = vs[1].gbps;
        printf("run %2d: naive=%8.2f GB/s  sched=%8.2f GB/s  winner=%s\n",
               r, vs[0].gbps, vs[1].gbps, w == 0 ? "naive" : "sched");
    }

    qsort(gnaive, (size_t)runs, sizeof(double), cmp_d);
    qsort(gsched, (size_t)runs, sizeof(double), cmp_d);
    double med = (runs % 2)
                     ? gnaive[runs / 2]
                     : 0.5 * (gnaive[runs / 2 - 1] + gnaive[runs / 2]);
    double q1 = gnaive[runs / 4], q3 = gnaive[3 * runs / 4];
    double meds = (runs % 2)
                      ? gsched[runs / 2]
                      : 0.5 * (gsched[runs / 2 - 1] + gsched[runs / 2]);
    double q1s = gsched[runs / 4], q3s = gsched[3 * runs / 4];
    printf("naive: median=%.2f GB/s IQR=[%.2f, %.2f]\n", med, q1, q3);
    printf("sched: median=%.2f GB/s IQR=[%.2f, %.2f]\n", meds, q1s, q3s);

    jit_free_version(&vs[0]);
    jit_free_version(&vs[1]);
    xorsched_free(pl.xor_sched);
    free(pl.mat);
    free(gnaive);
    free(gsched);
    return 0;
}
