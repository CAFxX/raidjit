/* Const-pool alignment-gap determinism (lead L2).
 *
 * The constant pool is assembled entry-by-entry at 64-byte (x86-64/AArch64)
 * or 16-byte (RISC-V) aligned offsets; the alignment gaps between entries
 * were copied uninitialized into the emitted kernel image, so two builds
 * of the same candidate could differ byte-for-byte depending on heap
 * history. The fix zero-fills every gap in pool_get at creation time.
 *
 * This test builds every buildable candidate with a non-empty pool twice,
 * with hostile heap churn (distinct fill patterns) between the builds, and
 * requires the FULL emitted image (code_size + const_size bytes) to be
 * byte-identical. Red-green: fails on the unfixed library (SSE/AVX2
 * candidates differ in the pool region, gap bytes carry the churn
 * patterns), passes with the fix.
 *
 * Portable: exercises whatever ISA the host builds for. (RISC-V pool
 * entries are always 16-byte multiples, so it is gap-free by construction
 * and passes trivially there; AArch64 has 16-byte entries at 64-byte
 * alignment and exercises the fix.)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "plan.h"
#include "jit.h"
#include "cpu.h"
#include "raidjit.h"

static int fails = 0;
#define CHECK(cond, ...)                                                  \
    do {                                                                  \
        if (!(cond)) {                                                    \
            printf("FAIL %s:%d: ", __func__, __LINE__);                    \
            printf(__VA_ARGS__);                                          \
            printf("\n");                                                 \
            fails++;                                                      \
        }                                                                 \
    } while (0)

/* Hostile heap churn: fill freed chunks of the pool buffer's size class
 * (8192) and a neighbouring class (4096) with a distinctive pattern, so a
 * subsequent build's fresh pool malloc reuses pattern-filled memory. Two
 * different patterns across the two builds make any uninitialized gap
 * byte show up as an image difference. */
static void churn(unsigned char pat)
{
    enum { N = 1024 };
    void *b[N];
    for (int i = 0; i < N; i++) {
        b[i] = malloc(8192);
        if (b[i])
            memset(b[i], pat, 8192);
    }
    for (int i = 0; i < N; i++)
        free(b[i]);
    for (int i = 0; i < N; i++) {
        b[i] = malloc(4096);
        if (b[i])
            memset(b[i], (unsigned char)(pat ^ 0x11), 4096);
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
    if (jit_build_version(pl, c, cpu, opts, v, err, sizeof(err)) != 0)
        return 0; /* clean backend reject: caller skips */
    return 1;
}

/* Build candidate c twice with hostile heap history between; require
 * byte-identical full images. Returns 1 if the A/B comparison ran,
 * 0 if the candidate was skipped (backend reject or no pool). */
static int check_candidate(const plan_t *pl, const cand_t *c, int idx,
                           const rj_cpu_info *cpu, const rj_options *opts)
{
    version_t vprobe;
    if (!build_one(pl, c, cpu, opts, &vprobe))
        return 0;
    int has_pool = vprobe.const_size > 0;
    jit_free_version(&vprobe);
    if (!has_pool)
        return 0;

    churn(0xA5);
    version_t va;
    if (!build_one(pl, c, cpu, opts, &va))
        return 0;
    size_t total = va.code_size + va.const_size;
    unsigned char *imga = malloc(total ? total : 1);
    CHECK(imga != NULL, "cand %d: out of memory", idx);
    if (!imga) {
        jit_free_version(&va);
        return 0;
    }
    memcpy(imga, va.mem, total);
    size_t code_size = va.code_size, const_size = va.const_size;
    jit_free_version(&va);

    churn(0x5A);
    version_t vb;
    if (!build_one(pl, c, cpu, opts, &vb)) {
        free(imga);
        return 0;
    }
    if (vb.code_size != code_size || vb.const_size != const_size) {
        printf("FAIL cand %d: image length changed across builds "
               "(%zu+%zu vs %zu+%zu)\n", idx, code_size, const_size,
               vb.code_size, vb.const_size);
        fails++;
    } else {
        const unsigned char *imgb = (const unsigned char *)vb.mem;
        size_t first_diff = total;
        for (size_t k = 0; k < total; k++) {
            if (imga[k] != imgb[k]) {
                first_diff = k;
                break;
            }
        }
        if (first_diff != total) {
            printf("FAIL cand %d (isa=%d U=%d): image differs at offset "
                   "%zu (%s region): 0x%02x vs 0x%02x\n", idx, c->isa,
                   c->U, first_diff,
                   first_diff >= code_size ? "pool" : "CODE",
                   imga[first_diff], imgb[first_diff]);
            fails++;
        }
    }
    free(imga);
    jit_free_version(&vb);
    return 1;
}

int main(void)
{
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

    rj_options opts;
    rj_options_default(&opts);
    opts.benchmark = 0;

    int tested = 0;
    for (int i = 0; i < n; i++)
        tested += check_candidate(&pl, &cands[i], i, &cpu, &opts);
    printf("pool-determinism: %d candidates with pools A/B-compared, %s\n",
           tested, fails ? "MISMATCHES FOUND" : "all byte-identical");
    CHECK(tested > 0, "no candidate with a const pool was tested");

    plan_free(&pl);
    topo_int_free(&ti);
    rj_topo_free(topo);
    if (fails == 0)
        printf("PASS\n");
    return fails != 0;
}
