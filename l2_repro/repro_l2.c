/* L2 verification reproducer: const-pool alignment-gap determinism.
 *
 * Claim (L2): alignment gaps between constant-pool entries are copied
 * uninitialized into the emitted kernel image, so the image is not
 * byte-identical across heap histories.
 *
 * Method: for several buildable candidates, build the SAME candidate twice
 * with hostile heap churn (distinct fill patterns) between the builds, then
 * compare the FULL emitted image (code_size + const_size bytes at v->mem).
 * The code region must always match (existing determinism guard); any
 * difference in the pool region proves uninitialized gap bytes are copied.
 *
 * Exit 0: all images byte-identical (claim REFUTED for the tested configs).
 * Exit 1: at least one image pair differed (claim CONFIRMED).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "plan.h"
#include "jit.h"
#include "cpu.h"
#include "raidjit.h"

/* Hostile heap churn: fill freed chunks of the pool's size class (8192)
 * and neighbouring classes with a distinctive pattern, so a subsequent
 * build's fresh pool malloc reuses pattern-filled memory. */
static void churn(unsigned char pat)
{
    enum { N = 512 };
    void *b[N];
    for (int i = 0; i < N; i++) {
        b[i] = malloc(8192);
        if (b[i])
            memset(b[i], pat, 8192);
    }
    for (int i = 0; i < N; i++)
        free(b[i]);
    /* neighbouring classes, in case the build touches them first */
    for (int i = 0; i < N; i++) {
        b[i] = malloc(4096);
        if (b[i])
            memset(b[i], pat ^ 0x11, 4096);
    }
    for (int i = 0; i < N; i++)
        free(b[i]);
}

static int build_one(const plan_t *pl, const cand_t *c,
                     const rj_cpu_info *cpu, const rj_options *opts,
                     version_t *v)
{
    char err[256] = { 0 };
    memset(v, 0, sizeof(*v));
    if (jit_build_version(pl, c, cpu, opts, v, err, sizeof(err)) != 0) {
        printf("  build failed: %s\n", err);
        return 0;
    }
    return 1;
}

int main(void)
{
#if !defined(__x86_64__)
    printf("SKIP: x86-64 only\n");
    return 0;
#else
    setvbuf(stdout, NULL, _IONBF, 0);
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);

    rj_topology *topo = rj_topo_raid6_rs(6);
    topo_int_t ti;
    plan_t pl;
    char err[256];
    if (topo_int_build(&ti, topo, err, sizeof(err)) != 0) {
        printf("FAIL topo_int_build: %s\n", err);
        return 2;
    }
    rj_op op = rj_op_encode();
    if (plan_build(&ti, &op, "raid6_rs(6)", &pl, err, sizeof(err)) != 0) {
        printf("FAIL plan_build: %s\n", err);
        return 2;
    }
    cand_t cands[512];
    int n = plan_candidates(&pl, &cpu, cands, 512, 0, 0);
    printf("enumerated %d candidates\n", n);

    rj_options opts;
    rj_options_default(&opts);
    opts.benchmark = 0;

    int tested = 0, nondet = 0;
    for (int i = 0; i < n; i++) {
        version_t v0;
        if (!build_one(&pl, &cands[i], &cpu, &opts, &v0))
            continue; /* clean backend reject: skip */
        if (v0.const_size == 0) {
            jit_free_version(&v0);
            continue; /* no pool: nothing to test */
        }
        size_t total0 = v0.code_size + v0.const_size;
        jit_free_version(&v0);

        /* A/B with hostile heap history between. */
        churn(0xA5);
        version_t va, vb;
        if (!build_one(&pl, &cands[i], &cpu, &opts, &va))
            continue;
        size_t total = va.code_size + va.const_size;
        unsigned char *imga = malloc(total ? total : 1);
        memcpy(imga, va.mem, total);
        size_t code_a = va.code_size, const_a = va.const_size;
        jit_free_version(&va);

        churn(0x5A);
        if (!build_one(&pl, &cands[i], &cpu, &opts, &vb)) {
            free(imga);
            continue;
        }
        int same_len = (vb.code_size == code_a &&
                        vb.const_size == const_a);
        int diff_at = -1;
        size_t diff_region = 0; /* 0=code, 1=pool */
        if (same_len) {
            const unsigned char *imgb = (const unsigned char *)vb.mem;
            for (size_t k = 0; k < total; k++) {
                if (imga[k] != imgb[k]) {
                    diff_at = (int)k;
                    diff_region = (k >= code_a) ? 1 : 0;
                    break;
                }
            }
        }
        printf("cand %d isa=%d U=%d maskedtail=%d: code=%zu const=%zu total=%zu -> %s%s\n",
               i, cands[i].isa, cands[i].U, cands[i].maskedtail,
               code_a, const_a, total,
               same_len ? (diff_at < 0 ? "IDENTICAL" : "DIFFER") : "LENGTH-MISMATCH",
               diff_at >= 0 ? (diff_region ? " (pool region)" : " (CODE region!)") : "");
        if (diff_at >= 0) {
            /* dump a window around the first difference */
            size_t lo = diff_at > 16 ? (size_t)diff_at - 16 : 0;
            size_t hi = (size_t)diff_at + 32 < total ? (size_t)diff_at + 32 : total;
            printf("  first diff at image offset %d (0x%x):\n  A:", diff_at, diff_at);
            for (size_t k = lo; k < hi; k++)
                printf(" %02x", imga[k]);
            printf("\n  B:");
            const unsigned char *imgb = (const unsigned char *)vb.mem;
            for (size_t k = lo; k < hi; k++)
                printf(" %02x", imgb[k]);
            printf("\n");
            nondet++;
        }
        free(imga);
        jit_free_version(&vb);
        tested++;
        (void)total0;
    }
    printf("tested %d candidates with pools: %s\n", tested,
           nondet ? "NONDETERMINISTIC image(s) found" : "all byte-identical");

    plan_free(&pl);
    topo_int_free(&ti);
    rj_topo_free(topo);
    return nondet ? 1 : 0;
#endif
}
