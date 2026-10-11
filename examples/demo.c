/* Demo: compile a RAID-6 encoder for 8+2 disks, report the chosen
 * implementation, time it, then lose two disks and rebuild them. */
#include <raidjit.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void)
{
    char err[256];
    rj_cpu_info ci;
    rj_cpu_detect(&ci);
    printf("cpu: %s\n", ci.brand);

    rj_topology *t = rj_topo_raid6(8);
    rj_options o;
    rj_options_default(&o);
    o.benchmark = 1;
    o.bench_iters = 5;

    rj_op enc = rj_op_encode();
    rj_jit *e = rj_compile(t, &enc, &o, err, sizeof(err));
    if (!e) {
        fprintf(stderr, "compile failed: %s\n", err);
        return 1;
    }
    rj_info ei;
    rj_get_info(e, &ei);
    printf("encode: %s\n", ei.desc);
    printf("        uops/byte=%.2f mem/byte=%.2f code=%zuB consts=%zuB\n",
           ei.uops_per_byte, ei.mem_per_byte, ei.code_size, ei.const_size);
    printf("        measured %.2f GB/s\n", ei.gbps);

    const size_t len = 64 << 20;
    uint8_t *d[8], *p[2];
    for (int i = 0; i < 8; i++) {
        if (posix_memalign((void **)&d[i], 64, len)) return 1;
        for (size_t k = 0; k < len; k += 4096)
            d[i][k] = (uint8_t)(k >> 12);
    }
    for (int k = 0; k < 2; k++)
        if (posix_memalign((void **)&p[k], 64, len)) return 1;

    const void *in[8];
    void *out[2];
    for (int i = 0; i < 8; i++)
        in[i] = d[i];
    for (int k = 0; k < 2; k++)
        out[k] = p[k];

    double t0 = now();
    rj_get_fn(e)(in, out, len);
    double t1 = now();
    printf("encoded %zu MiB in %.3f s (%.2f GB/s)\n", len >> 20, t1 - t0,
           (double)(10 * len) / (t1 - t0) / 1e9);

    /* lose disks 1 and 8 (data + Q parity), rebuild them */
    int failed[] = { 1, 8 };
    rj_op rec = rj_op_recover(failed, 2);
    rj_jit *r = rj_compile(t, &rec, &o, err, sizeof(err));
    if (!r) {
        fprintf(stderr, "recover compile failed: %s\n", err);
        return 1;
    }
    rj_info ri;
    rj_get_info(r, &ri);
    printf("recover: %s (%.2f GB/s)\n", ri.desc, ri.gbps);

    uint8_t *saved1, *saved8;
    if (posix_memalign((void **)&saved1, 64, len)) return 1;
    if (posix_memalign((void **)&saved8, 64, len)) return 1;
    memcpy(saved1, d[1], len);
    memcpy(saved8, p[0], len);
    memset(d[1], 0, len);
    memset(p[0], 0, len);

    /* survivors in increasing disk order: 0,2,3,4,5,6,7,9 */
    const void *rin[8] = { d[0], d[2], d[3], d[4], d[5], d[6], d[7], p[1] };
    void *rout[2] = { d[1], p[0] };
    rj_get_fn(r)(rin, rout, len);
    printf("rebuild %s\n",
           memcmp(d[1], saved1, len) == 0 && memcmp(p[0], saved8, len) == 0
               ? "OK"
               : "MISMATCH");

    rj_free(e);
    rj_free(r);
    rj_topo_free(t);
    return 0;
}
