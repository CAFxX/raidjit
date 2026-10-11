/* XOR-schedule end-to-end test (track A3, D4).
 *
 * The shipped path is rj_compile() with benchmark=1: candidates are
 * enumerated by plan_candidates, built, and measured by bench_select.
 * The unit test (test_xorsched.c) covers plan_candidates enumeration
 * only; this test drives the PUBLIC API and asserts the +xsched twin
 * is not just enumerated but actually benchmarked: with verbose=1,
 * bench_select prints one measured line per candidate, so a "+xsched"
 * line with a GB/s figure proves the twin went through the real
 * benchmark harness. Also checks the winning version is correct.
 *
 * Portable: runs natively on x86-64 and under QEMU for AArch64/RVV.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>

#include "raidjit.h"

static int fails;
#define CHECK(c, ...) do {                                              \
    if (!(c)) {                                                        \
        printf("FAIL "); printf(__VA_ARGS__); printf("\n"); fails++;   \
    }                                                                  \
} while (0)

/* Sharing-heavy 4x20 P_XOR config (same matrix as test_xorsched's t1):
 * 2 shared pairs -> the schedule extracts 2 temps and the strict-gate
 * twin is enumerated on every backend. Row-major, npar x ndata. */
static const uint8_t e2e_mat[80] = {
    1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,
    1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,
    0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,
};

static uint64_t xrng_state = 0xFEDCBA987654321ULL;
static uint64_t xrng(void)
{
    uint64_t x = xrng_state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return xrng_state = x;
}

int main(void)
{
    rj_topology *topo = rj_topo_custom(20, 4, e2e_mat);
    CHECK(topo != NULL, "rj_topo_custom failed");
    if (!topo)
        return 1;
    rj_op op = rj_op_encode();

    rj_options opts;
    rj_options_default(&opts);
    opts.benchmark = 1;
    opts.verbose = 1; /* bench_select prints one measured line/candidate */
    opts.bench_bytes = 64 << 10; /* small working set: keep the e2e fast */
    opts.bench_iters = 3;

    /* capture stderr around rj_compile: the verbose bench report */
    int pipefd[2];
    CHECK(pipe(pipefd) == 0, "pipe failed");
    int saved_err = dup(STDERR_FILENO);
    CHECK(saved_err >= 0, "dup failed");
    CHECK(dup2(pipefd[1], STDERR_FILENO) >= 0, "dup2 failed");
    close(pipefd[1]);

    char err[256];
    rj_jit *j = rj_compile(topo, &op, &opts, err, sizeof(err));

    fflush(stderr);
    CHECK(dup2(saved_err, STDERR_FILENO) >= 0, "stderr restore failed");
    close(saved_err);

    char cap[1 << 20];
    size_t ncap = 0;
    ssize_t r;
    while ((r = read(pipefd[0], cap + ncap, sizeof(cap) - ncap - 1)) > 0)
        ncap += (size_t)r;
    close(pipefd[0]);
    cap[ncap] = '\0';

    CHECK(j != NULL, "rj_compile failed: %s", err);
    if (!j) {
        printf("---- captured stderr ----\n%s\n-------------------------\n", cap);
        return 1;
    }

    /* the +xsched twin was benchmarked: its desc line carries a GB/s
     * figure in the bench_select verbose report */
    int xsched_measured = 0;
    for (char *line = strtok(cap, "\n"); line; line = strtok(NULL, "\n")) {
        if (strstr(line, "+xsched") && strstr(line, "GB/s")) {
            xsched_measured = 1;
            printf("benchmarked twin: %s\n", line);
        }
    }
    CHECK(xsched_measured,
          "+xsched twin not present in the benchmark report "
          "(enumerated but never measured?)");

    rj_info info;
    rj_get_info(j, &info);
    printf("winner: %s\n", info.desc);
    CHECK(info.gbps_measured == 1, "winner not benchmark-measured");

    /* correctness of the winning version vs a scalar reference */
    {
        size_t len = 4096;
        uint8_t *in[20], *out[4], *ref[4];
        for (int i = 0; i < 20; i++) {
            if (posix_memalign((void **)&in[i], 64, len)) {
                printf("out of memory\n");
                return 1;
            }
            for (size_t k = 0; k < len; k++)
                in[i][k] = (uint8_t)xrng();
        }
        for (int j = 0; j < 4; j++) {
            if (posix_memalign((void **)&out[j], 64, len) ||
                posix_memalign((void **)&ref[j], 64, len)) {
                printf("out of memory\n");
                return 1;
            }
            memset(ref[j], 0, len);
            for (int i = 0; i < 20; i++) {
                if (!e2e_mat[(size_t)j * 20 + (size_t)i])
                    continue;
                for (size_t k = 0; k < len; k++)
                    ref[j][k] ^= in[i][k];
            }
            memset(out[j], 0xA5, len);
        }
        rj_get_fn(j)((const void **)in, (void **)out, len);
        for (int j = 0; j < 4; j++)
            CHECK(memcmp(out[j], ref[j], len) == 0,
                  "e2e winner mismatch row %d", j);
        for (int i = 0; i < 20; i++)
            free(in[i]);
        for (int j = 0; j < 4; j++) {
            free(out[j]);
            free(ref[j]);
        }
    }

    rj_free(j);
    rj_op_free(&op);
    rj_topo_free(topo);
    printf("e2e done, %d failures\n", fails);
    return fails != 0;
}
