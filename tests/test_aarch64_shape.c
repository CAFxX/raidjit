/* AArch64 codegen shape test: F4 (P_XOR multi-row input-load hoisting).
 *
 * F4: for multi-row P_XOR, the input must be loaded once per (i,u) into
 * the t scratch, with per-row mov/eor — not reloaded per (j,u). This test
 * builds a synthetic 2x2 XOR-only topology (the sweep's xor22 case:
 * mat=[[1,1],[1,1]], 2 inputs, 2 parity rows) and counts vector loads
 * from the input pointer (X10) in the emitted NEON/SVE code.
 *
 * With U=1: n_in=2 input columns. Fixed code emits exactly 2 input loads
 * (one per column); the old code emitted 4 (one per row per column).
 * A revert to per-row loads fails this test.
 *
 * Runs natively on AArch64 and under QEMU (NEON always; SVE when present).
 * Prints SKIP and passes elsewhere.
 */
#include "../include/raidjit.h"
#include "../src/plan.h"
#include "../src/jit.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static int fails;
#define CHECK(c, ...) do {                                             \
    if (!(c)) {                                                        \
        printf("FAIL "); printf(__VA_ARGS__); printf("\n"); fails++;   \
    }                                                                  \
} while (0)

#if defined(__aarch64__)
/* Count NEON LDR Qt,[X10,#off] (0x3DC00000 base) and SVE LD1B
 * Zt.B,p0/z,[X10] (0xA400A000 base) in the code buffer. */
static int count_input_loads(const uint8_t *code, size_t n, int is_sve)
{
    int cnt = 0;
    const uint32_t *w = (const uint32_t *)code;
    size_t nw = n / 4;
    for (size_t k = 0; k < nw; k++) {
        uint32_t ins = w[k];
        unsigned rn = (ins >> 5) & 31u;
        if (rn != 10)
            continue;
        if (!is_sve) {
            if ((ins & 0xFFC00000u) == 0x3DC00000u)
                cnt++;
        } else {
            if ((ins & 0xFFC0FC00u) == 0xA400A000u)
                cnt++;
        }
    }
    return cnt;
}

static void test_f4_isa(const rj_cpu_info *cpu, isa_t isa, const char *nm,
                        int is_sve)
{
    /* 2x2 all-ones XOR matrix: 2 inputs, 2 parity rows, P_XOR. */
    uint8_t m[4] = {1, 1, 1, 1};
    rj_topology *topo = rj_topo_custom(2, 2, m);
    CHECK(topo != NULL, "f4 [%s]: topo_custom failed", nm);
    if (!topo)
        return;
    rj_op op = rj_op_encode();
    topo_int_t ti;
    char err[256];
    if (topo_int_build(&ti, topo, err, sizeof err)) {
        printf("FAIL f4 [%s]: topo: %s\n", nm, err);
        fails++;
        rj_op_free(&op);
        rj_topo_free(topo);
        return;
    }
    plan_t pl;
    if (plan_build(&ti, &op, "enc", &pl, err, sizeof err)) {
        printf("FAIL f4 [%s]: plan: %s\n", nm, err);
        fails++;
        topo_int_free(&ti);
        rj_op_free(&op);
        rj_topo_free(topo);
        return;
    }
    if (pl.kind != P_XOR) {
        printf("FAIL f4 [%s]: expected P_XOR, got kind %d\n", nm, pl.kind);
        fails++;
        plan_free(&pl);
        topo_int_free(&ti);
        rj_op_free(&op);
        rj_topo_free(topo);
        return;
    }

    cand_t cands[128];
    int nc = plan_candidates(&pl, cpu, cands, 128, 1, 0);
    rj_options opts;
    rj_options_default(&opts);
    opts.benchmark = 0;
    int built = 0;
    for (int i = 0; i < nc && !built; i++) {
        if (cands[i].isa != isa || cands[i].U != 1)
            continue;
        version_t v;
        memset(&v, 0, sizeof v);
        if (jit_build_version(&pl, &cands[i], cpu, &opts, &v, err,
                              sizeof err) != 0) {
            printf("FAIL f4 [%s]: build: %s\n", nm, err);
            fails++;
            continue;
        }
        /* n_in=2 columns, U=1: one load per (i,u) => exactly 2 input
         * loads. The pre-F4 code emitted 4 (one per row per column). */
        int nl = count_input_loads(v.mem, v.code_size, is_sve);
        CHECK(nl == 2, "f4 [%s]: input loads=%d, want 2", nm, nl);
        if (nl == 2)
            printf("f4 [%s]: input loads=2 OK\n", nm);
        jit_free_version(&v);
        built = 1;
    }
    if (!built) {
        printf("FAIL f4 [%s]: no U=1 candidate built\n", nm);
        fails++;
    }
    plan_free(&pl);
    topo_int_free(&ti);
    rj_op_free(&op);
    rj_topo_free(topo);
}
#endif /* __aarch64__ */

int main(void)
{
#if defined(__aarch64__)
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    if (cpu.neon)
        test_f4_isa(&cpu, ISA_NEON, "neon", 0);
    if (cpu.sve && cpu.sve_vl >= 16)
        test_f4_isa(&cpu, ISA_SVE, "sve", 1);
    if (!cpu.neon && !(cpu.sve && cpu.sve_vl >= 16))
        printf("test_aarch64_shape: SKIP (no NEON/SVE)\n");
#else
    printf("test_aarch64_shape: SKIP (not AArch64)\n");
#endif
    if (fails == 0)
        printf("test_aarch64_shape: all OK\n");
    else
        printf("test_aarch64_shape: %d FAILURES\n", fails);
    return fails != 0;
}
