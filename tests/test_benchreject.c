/* bench_select must reject a deliberately corrupted candidate BEFORE any
 * timing: validation against the scalar reference is the gate that keeps a
 * wrong-but-fast code version from ever winning.
 *
 * Method: build all versions for raid6(8) encode, flip every byte of one
 * version's constant pool (data, not code: no icache concerns, no crash
 * risk), then require bench_select to fail with a "wrong output" error.
 * A positive control (uncorrupted select succeeds) runs first.
 */
#include "../include/raidjit.h"
#include "../src/plan.h"
#include "../src/jit.h"
#include "../src/bench.h"
#include "../src/cpu.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

static int failures = 0;
#define CHECK(cond, ...)                                                   \
    do {                                                                   \
        if (!(cond)) {                                                     \
            printf("FAIL %s:%d: ", __func__, __LINE__);                    \
            printf(__VA_ARGS__);                                           \
            printf("\n");                                                  \
            failures++;                                                    \
        }                                                                  \
    } while (0)

int main(void)
{
    char err[256];
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);

    rj_topology *t = rj_topo_raid6(8);
    topo_int_t ti;
    if (topo_int_build(&ti, t, err, sizeof(err))) {
        printf("FAIL topo: %s\n", err);
        return 1;
    }
    rj_op op = rj_op_encode();
    plan_t pl;
    if (plan_build(&ti, &op, "raid6", &pl, err, sizeof(err))) {
        printf("FAIL plan: %s\n", err);
        return 1;
    }

    cand_t cands[32];
    int ncand = plan_candidates(&pl, &cpu, cands, 32, 0, 0);
    CHECK(ncand > 0, "plan_candidates gave none");

    rj_options o;
    rj_options_default(&o);
    o.bench_bytes = 1 << 16; /* small: positive control stays fast */
    o.bench_iters = 3;

    version_t vs[32];
    int nv = 0;
    for (int i = 0; i < ncand && nv < 32; i++) {
        char e2[256];
        if (jit_build_version(&pl, &cands[i], &cpu, &o, &vs[nv], e2,
                              sizeof(e2)) == 0)
            nv++;
    }
    CHECK(nv > 0, "no versions built");

    /* positive control: uncorrupted selection must succeed */
    int w = bench_select(vs, nv, &pl, &o, 0, NULL, err, sizeof(err));
    CHECK(w >= 0, "uncorrupted bench_select failed: %s", err);

    /* corrupt: flip the whole constant pool of the first version that
     * has one (a GF-multiply path: nibble tables or GFNI matrices) */
    int ci = -1;
    for (int i = 0; i < nv; i++)
        if (vs[i].const_size > 0) {
            ci = i;
            break;
        }
    CHECK(ci >= 0, "no candidate with a constant pool to corrupt");
    if (ci >= 0) {
        if (mprotect(vs[ci].mem, vs[ci].memsz, PROT_READ | PROT_WRITE) != 0) {
            printf("FAIL mprotect RW: %m\n");
            return 1;
        }
        uint8_t *pool = (uint8_t *)vs[ci].mem + vs[ci].code_size;
        for (size_t k = 0; k < vs[ci].const_size; k++)
            pool[k] ^= 0xFF;
        if (mprotect(vs[ci].mem, vs[ci].memsz, PROT_READ | PROT_EXEC) != 0) {
            printf("FAIL mprotect RX: %m\n");
            return 1;
        }

        w = bench_select(vs, nv, &pl, &o, 0, NULL, err, sizeof(err));
        CHECK(w == -1,
              "corrupted candidate was NOT rejected (winner=%d)", w);
        CHECK(strstr(err, "wrong output") != NULL,
              "error should name the mismatch, got: %s", err);
    }

    for (int i = 0; i < nv; i++)
        jit_free_version(&vs[i]);
    plan_free(&pl);
    rj_op_free(&op);
    topo_int_free(&ti);
    rj_topo_free(t);

    if (failures) {
        printf("BENCHREJECT TESTS FAILED (%d)\n", failures);
        return 1;
    }
    printf("benchreject: all OK\n");
    return 0;
}
