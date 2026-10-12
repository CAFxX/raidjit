/* test_ternlog_gates: the planner must never enumerate a ternlog twin the
 * backend cannot take (lead L6), and must not drop legitimate ones.
 *
 * Background: the F16 postcondition (BUG-COMBO-STATIC-F16) rejects a
 * ternlog twin whose build emits no VPTERNLOGD. Two planner gates keep
 * phantom twins out of enumeration, each mirroring its backend body's
 * exact emission condition:
 *
 *  1. The P_MULACC/P_DELTA ternlog gate (src/plan.c) requires a foldable
 *     triple in the generic mulacc body: a nontrivial coefficient past
 *     the row's first contributing column (mulacc_column() only folds
 *     there; the c==1/GFNI paths and the first-column move never consult
 *     the flag).
 *  2. The rec_cf gate (src/plan.c) additionally requires a foldable
 *     triple in the closed-form syndrome when cloning rec_cf onto a
 *     ternlog base: the rec_cf body (gen_body_rec_cf) emits VPTERNLOGD
 *     only in its syndrome phase, for a nontrivial syndrome coefficient
 *     past the syndrome's first contributing column.
 *
 * The L6 instance: raid6(4) recover {0+1}/{0+2}/{1+2} -- the rec_cf
 * clones of the 8 legitimate ternlog twins were phantoms (the only
 * nontrivial syndrome coefficient sits at the first contributing
 * position), so all 24 were F16-rejected.
 *
 * Fail-capable: on the pre-fix tree this test fails 24 times (one per
 * phantom: "ternlog twin requested but no VPTERNLOGD emitted"); if
 * either gate over-fires, the pinned clean-build counts drop.
 */
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include "../src/jit.h"
#include "../src/plan.h"
#include "../src/cpu.h"
#include "../include/raidjit.h"

static int failures = 0;
#define CHECK(c, ...) do {                                             \
    if (!(c)) {                                                        \
        printf("FAIL %s:%d: ", __func__, __LINE__);                     \
        printf(__VA_ARGS__); printf("\n"); failures++;                 \
    }                                                                  \
} while (0)

/* Synthetic AVX-512 without GFNI, matching the L6 reproducer. Code is
 * only built, never executed, so the host needs no AVX-512. */
static void synth_cpu(rj_cpu_info *cpu)
{
    memset(cpu, 0, sizeof(*cpu));
    cpu->sse2 = cpu->ssse3 = cpu->avx = cpu->avx2 = 1;
    cpu->avx512f = cpu->avx512bw = cpu->avx512vl = 1;
    cpu->gfni = 0;
}

/* Build every ternlog candidate of every failure subset of size 1..2 on
 * raid6(4). Returns the number built clean; counts rejections. */
static void build_subset(const int *sub, int k, rj_topology *t,
                         const rj_cpu_info *cpu, int *n_clean,
                         int *n_rejected, int *n_rec_cf_clean)
{
    topo_int_t ti;
    char err[256];
    if (topo_int_build(&ti, t, err, sizeof(err))) {
        printf("topo: %s\n", err);
        exit(2);
    }
    rj_op op = rj_op_recover(sub, k);
    plan_t pl;
    if (plan_build(&ti, &op, "raid6(4)", &pl, err, sizeof(err))) {
        /* unrecoverable subset: nothing to enumerate */
        rj_op_free(&op);
        topo_int_free(&ti);
        return;
    }
    cand_t cands[256];
    int n = plan_candidates(&pl, cpu, cands, 256, 0, 0);
    rj_options o;
    rj_options_default(&o);
    for (int i = 0; i < n; i++) {
        if (!cands[i].ternlog)
            continue;
        version_t v;
        memset(&v, 0, sizeof(v));
        int rc = jit_build_version(&pl, &cands[i], cpu, &o, &v, err,
                                   sizeof(err));
        if (rc) {
            (*n_rejected)++;
            printf("FAIL build: subset{");
            for (int j = 0; j < k; j++)
                printf("%s%d", j ? "+" : "", sub[j]);
            printf("} U=%d rec_cf=%d: %s\n", cands[i].U, cands[i].rec_cf,
                   err);
            failures++;
        } else {
            (*n_clean)++;
            if (cands[i].rec_cf)
                (*n_rec_cf_clean)++;
            jit_free_version(&v);
        }
    }
    plan_free(&pl);
    rj_op_free(&op);
    topo_int_free(&ti);
}

static void build_all_ternlog(int *n_clean, int *n_rejected,
                              int *n_rec_cf_clean)
{
    rj_cpu_info cpu;
    synth_cpu(&cpu);
    rj_topology *t = rj_topo_raid6(4);
    for (int d = 0; d < 6; d++) {
        int sub[1] = { d };
        build_subset(sub, 1, t, &cpu, n_clean, n_rejected, n_rec_cf_clean);
    }
    for (int a = 0; a < 6; a++)
        for (int b = a + 1; b < 6; b++) {
            int sub[2] = { a, b };
            build_subset(sub, 2, t, &cpu, n_clean, n_rejected,
                         n_rec_cf_clean);
        }
    rj_topo_free(t);
}

/* The L6 subsets: no rec_cf+ternlog phantom may be enumerated, while the
 * 8 legitimate generic ternlog twins per subset must survive. */
static void l6_subsets(void)
{
    rj_cpu_info cpu;
    synth_cpu(&cpu);
    rj_topology *t = rj_topo_raid6(4);
    const int subs[3][2] = { { 0, 1 }, { 0, 2 }, { 1, 2 } };
    for (int s = 0; s < 3; s++) {
        topo_int_t ti;
        char err[256];
        if (topo_int_build(&ti, t, err, sizeof(err))) {
            printf("topo: %s\n", err);
            exit(2);
        }
        rj_op op = rj_op_recover(subs[s], 2);
        plan_t pl;
        if (plan_build(&ti, &op, "raid6(4)", &pl, err, sizeof(err))) {
            printf("plan: %s\n", err);
            exit(2);
        }
        cand_t cands[256];
        int n = plan_candidates(&pl, &cpu, cands, 256, 0, 0);
        int n_phantom = 0, n_plain = 0;
        for (int i = 0; i < n; i++) {
            if (!cands[i].ternlog)
                continue;
            if (cands[i].rec_cf)
                n_phantom++;
            else
                n_plain++;
        }
        CHECK(n_phantom == 0,
              "subset{%d+%d}: %d rec_cf+ternlog phantoms enumerated",
              subs[s][0], subs[s][1], n_phantom);
        CHECK(n_plain == 8,
              "subset{%d+%d}: %d plain ternlog twins (want 8)",
              subs[s][0], subs[s][1], n_plain);
        plan_free(&pl);
        rj_op_free(&op);
        topo_int_free(&ti);
    }
    rj_topo_free(t);
}

/* Positive case: an encode plan has foldable triples, so ternlog twins
 * must still be enumerated and build clean (the gate fires, not just
 * filters). */
static void encode_still_enumerated(void)
{
    rj_cpu_info cpu;
    synth_cpu(&cpu);
    rj_topology *t = rj_topo_raid6(4);
    topo_int_t ti;
    char err[256];
    if (topo_int_build(&ti, t, err, sizeof(err))) {
        printf("topo: %s\n", err);
        exit(2);
    }
    rj_op op = rj_op_encode();
    plan_t pl;
    if (plan_build(&ti, &op, "raid6(4)enc", &pl, err, sizeof(err))) {
        printf("plan: %s\n", err);
        exit(2);
    }
    cand_t cands[256];
    int n = plan_candidates(&pl, &cpu, cands, 256, 0, 0);
    rj_options o;
    rj_options_default(&o);
    int ntl = 0;
    for (int i = 0; i < n; i++) {
        if (!cands[i].ternlog)
            continue;
        ntl++;
        version_t v;
        memset(&v, 0, sizeof(v));
        int rc = jit_build_version(&pl, &cands[i], &cpu, &o, &v, err,
                                   sizeof(err));
        CHECK(rc == 0, "encode ternlog U=%d build failed: %s", cands[i].U,
              rc ? err : "");
        if (!rc)
            jit_free_version(&v);
    }
    CHECK(ntl == 8, "encode: %d ternlog twins enumerated (want 8)", ntl);
    plan_free(&pl);
    rj_op_free(&op);
    topo_int_free(&ti);
    rj_topo_free(t);
}

int main(void)
{
    l6_subsets();
    encode_still_enumerated();
    int n_clean = 0, n_rejected = 0, n_rec_cf_clean = 0;
    build_all_ternlog(&n_clean, &n_rejected, &n_rec_cf_clean);
    CHECK(n_rejected == 0, "%d ternlog builds rejected (phantoms)",
          n_rejected);
    /* Pinned pre-fix clean counts: 128 generic + 96 rec_cf ternlog twins.
     * Either gate over-firing drops these. */
    CHECK(n_clean == 224, "%d clean ternlog builds (want 224)", n_clean);
    CHECK(n_rec_cf_clean == 96, "%d clean rec_cf+ternlog builds (want 96)",
          n_rec_cf_clean);
    if (!failures)
        printf("PASS test_ternlog_gates\n");
    return failures ? 1 : 0;
}
