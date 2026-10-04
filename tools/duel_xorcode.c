/* §5c code-choice re-run: duel XOR codes (RS/RDP/EVENODD/Liberation) at
 * each ndata, interleaved A/B, per docs/xor_codes.md §4 protocol.
 *
 * For each ndata 3..13: build each applicable code's topology, compile
 * with benchmark=1, time at 512 KiB and 8 MiB total data bytes
 * (interleaved rounds across codes to share contention equally).
 * Reports median GB/s per (code, size); the caller decides flips.
 *
 * Usage: duel_xorcode [ndata_lo] [ndata_hi]
 * Defaults: 3 13.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#include "raidjit.h"
#include "plan.h"
#include "jit.h"
#include "bench.h"
#include "cpu.h"

static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static int is_prime(int n)
{
    if (n < 2) return 0;
    for (int i = 2; i * i <= n; i++)
        if (n % i == 0) return 0;
    return 1;
}

/* Build the topology for (code, ndata). Returns NULL if not applicable. */
static rj_topology *mk_code(const char *code, int ndata)
{
    if (!strcmp(code, "rs"))
        return rj_topo_raid6_rs(ndata);
    if (!strcmp(code, "rdp")) {
        if (!is_prime(ndata + 1)) return NULL;
        return rj_topo_rdp(ndata + 1);
    }
    if (!strcmp(code, "evenodd")) {
        if (!is_prime(ndata)) return NULL;
        return rj_topo_evenodd(ndata);
    }
    if (!strcmp(code, "liberation"))
        return rj_topo_liberation(ndata);
    return NULL;
}

static double median_d(double *a, int n)
{
    /* insertion sort (n small) */
    for (int i = 1; i < n; i++) {
        double k = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > k) { a[j + 1] = a[j]; j--; }
        a[j + 1] = k;
    }
    if (n % 2) return a[n / 2];
    return (a[n / 2 - 1] + a[n / 2]) / 2.0;
}

static void *abuf(size_t n)
{
    void *p = NULL;
    if (posix_memalign(&p, 64, n ? n : 64)) {
        fprintf(stderr, "oom\n");
        exit(1);
    }
    return p;
}

int main(int argc, char **argv)
{
    int lo = 3, hi = 13;
    if (argc > 1) lo = atoi(argv[1]);
    if (argc > 2) hi = atoi(argv[2]);

    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);

    const char *codes[] = { "rs", "rdp", "evenodd", "liberation" };
    const size_t sizes[] = { 512 * 1024, 8 * 1024 * 1024 };
    const char *size_names[] = { "512K", "8M" };
    const int nrounds = 5; /* interleaved A/B rounds */
    const int nwarm = 2, niter = 11;

    printf("# §5c duel host: ");
    /* print a short cpu tag */
    printf("avx512f=%d avx512bw=%d gfni=%d\n",
           cpu.avx512f, cpu.avx512bw, cpu.gfni);
    printf("# ndata code 512K_GBs 8M_GBs (median of %d interleaved rounds, %d iters each)\n",
           nrounds, niter);

    for (int ndata = lo; ndata <= hi; ndata++) {
        for (int ci = 0; ci < 4; ci++) {
            const char *code = codes[ci];
            rj_topology *t = mk_code(code, ndata);
            if (!t) {
                printf("%d %s NA NA\n", ndata, code);
                continue;
            }
            rj_op op = rj_op_encode();
            char err[256];
            rj_options opts;
            rj_options_default(&opts);
            opts.benchmark = 1;
            rj_jit *j = rj_compile(t, &op, &opts, err, sizeof(err));
            if (!j) {
                printf("%d %s COMPILE_FAIL %s\n", ndata, code, err);
                rj_topo_free(t);
                continue;
            }
            rj_fn fn = rj_get_fn(j);
            /* Build topo/plan (matches test_xorcode pattern; also gives
             * us the true ndata/npar). */
            topo_int_t ti;
            if (topo_int_build(&ti, t, err, sizeof(err))) {
                printf("%d %s TOPO_FAIL %s\n", ndata, code, err);
                rj_free(j); rj_op_free(&op); rj_topo_free(t);
                continue;
            }
            plan_t pl;
            if (plan_build(&ti, &op, "duel", &pl, err, sizeof(err))) {
                printf("%d %s PLAN_FAIL %s\n", ndata, code, err);
                rj_free(j); rj_op_free(&op); rj_topo_free(t);
                continue;
            }
            int nd = ti.ndata, np = ti.npar;
            /* allocate buffers sized for the largest measurement */
            size_t maxlen = sizes[1] / (size_t)nd + 4096;
            uint8_t **in = abuf(sizeof(*in) * (size_t)nd);
            uint8_t **out = abuf(sizeof(*out) * (size_t)np);
            for (int i = 0; i < nd; i++) {
                in[i] = abuf(maxlen);
                for (size_t k = 0; k < maxlen; k++) in[i][k] = (uint8_t)(i + k);
            }
            for (int i = 0; i < np; i++) out[i] = abuf(maxlen);

            double med[2] = { 0, 0 };
            for (int si = 0; si < 2; si++) {
                size_t total = sizes[si];
                size_t len = total / (size_t)nd;
                /* align len down to 64 for the JIT */
                len &= ~(size_t)63;
                if (len < 64) len = 64;
                double samples[5];
                for (int r = 0; r < nrounds; r++) {
                    /* warmup */
                    for (int w = 0; w < nwarm; w++)
                        fn((const void **)in, (void **)out, len);
                    uint64_t t0 = now_ns();
                    for (int it = 0; it < niter; it++)
                        fn((const void **)in, (void **)out, len);
                    uint64_t t1 = now_ns();
                    double sec = (t1 - t0) / 1e9 / niter;
                    /* data GB/s: total data bytes (in+out) per second.
                     * Use ndata*len as the data volume (parity excluded,
                     * matching §4 "total data bytes"). */
                    samples[r] = (nd * (double)len) / sec / 1e9;
                }
                med[si] = median_d(samples, nrounds);
            }
            printf("%d %s %.2f %.2f\n", ndata, code, med[0], med[1]);
            for (int i = 0; i < nd; i++) free(in[i]);
            for (int i = 0; i < np; i++) free(out[i]);
            free(in); free(out);
            rj_free(j);
            rj_op_free(&op);
            rj_topo_free(t);
        }
    }
    return 0;
}
