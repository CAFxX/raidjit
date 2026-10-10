/* Permanent regression test for Horner rematerialization codegen (AArch64).
 *
 * Force-builds remat candidates via jit_build_version() -- bypassing the
 * planner's twin policy -- and validates the jitted output against the
 * scalar reference (ref_run). Any divergence fails the test, so a remat
 * codegen regression cannot slip through benchmark/winner-only coverage
 * (test_correct/test_matrix only ever run the benchmark winner, which is
 * rarely a remat twin).
 *
 * AArch64 implements remat for the nibble method only (there is no GFNI
 * on this ISA). In remat mode the prologue does not preload table
 * registers (hreg[] stays -1); the u-loop reloads each needed table from
 * the constant pool into the tbl scratch on every use.
 *
 * Cases (NEON, 16 vector regs; remat need = n_out*U+6):
 *   - RS(6,3) encode, NEON nibble, U=3, remat: the headline twin
 *     (persistent needs 18 regs, remat needs 15).
 *   - RS(6,3) encode, NEON nibble, U=3, remat, all-trivial coefficients:
 *     exercises the c in {0,1} fast paths in remat mode.
 *   - RAID-6 encode, NEON nibble, U=5, remat: ntriv=1 boundary
 *     (persistent 17 regs, remat 16).
 *   - RS(6,3) encode, SVE nibble, U=3, remat (only when SVE present).
 *   - planner checks: plan_candidates() on RS(6,3) must enumerate the
 *     NEON U=3 remat twin (persistent 18 > 16 >= remat 15) and must NOT
 *     enumerate a remat twin at U=2 (persistent fits: 15 <= 16).
 *
 * Run under QEMU, e.g.
 *   QEMU_LD_PREFIX=/usr/aarch64-linux-gnu qemu-aarch64 -cpu max \
 *       ./build-aarch64/test_aarch64_remat
 * and with SVE at VL 128/256. Without NEON/SVE it prints SKIP and passes.
 */
#include "../include/raidjit.h"
#include "../src/plan.h"
#include "../src/jit.h"
#include "../src/cpu.h"
#include "../src/gf.h"
#include "../src/ref.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static const size_t lens[] = { 0, 1, 7, 15, 16, 17, 31, 32, 33, 63, 64,
                               65, 100, 255, 256, 257, 1000, 4096 };
static const int nlens = sizeof(lens) / sizeof(lens[0]);

static uint64_t rng_state = 0x9e3779b97f4a7c15ULL;
static uint64_t xrng(void)
{
    uint64_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return rng_state = x;
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

static void randfill(void *p, size_t n)
{
    uint8_t *b = p;
    for (size_t i = 0; i < n; i++)
        b[i] = (uint8_t)xrng();
}

static int count_ntriv(const plan_t *pl)
{
    int n = 0;
    if (!pl->horner_c)
        return 0;
    for (int j = 0; j < pl->n_out; j++)
        if (pl->horner_c[j] != 0 && pl->horner_c[j] != 1)
            n++;
    return n;
}

/* Force-build one remat candidate and check it against the scalar ref.
 * expect_ntriv >= 0 locks the test to a specific nontrivial-coefficient
 * count (boundary coverage); -1 disables the check. force_all_trivial
 * rewrites the plan to an all-ones Horner plan (mat and horner_c alike,
 * so ref_run stays consistent) to exercise the c in {0,1} fast paths in
 * remat mode. */
static void test_remat_case(const char *name, rj_topology *t, isa_t isa,
                            int U, int force_all_trivial, int expect_ntriv)
{
    char err[256];
    topo_int_t ti;
    if (topo_int_build(&ti, t, err, sizeof(err)) != 0) {
        CHECK(0, "%s topo: %s", name, err);
        return;
    }
    rj_op op = rj_op_encode();
    plan_t pl;
    if (plan_build(&ti, &op, name, &pl, err, sizeof(err)) != 0) {
        CHECK(0, "%s plan: %s", name, err);
        rj_op_free(&op);
        topo_int_free(&ti);
        return;
    }
    if (pl.kind != P_HORNER) {
        CHECK(0, "%s: expected P_HORNER, got kind %d", name, pl.kind);
        plan_free(&pl);
        rj_op_free(&op);
        topo_int_free(&ti);
        return;
    }

    if (force_all_trivial) {
        /* An all-ones matrix is Horner-Vandermonde with c_j = 1; force
         * both so the JIT (which reads horner_c) and ref_run (which
         * reads mat) agree. */
        for (int j = 0; j < pl.n_out; j++)
            pl.horner_c[j] = 1;
        memset(pl.mat, 1, (size_t)pl.n_out * (size_t)pl.n_in);
    }
    int ntriv = count_ntriv(&pl);
    if (expect_ntriv >= 0)
        CHECK(ntriv == expect_ntriv, "%s: ntriv=%d, expected %d", name,
              ntriv, expect_ntriv);
    int need = plan_need_regs(isa, MUL_NIBBLE, &pl, U, 1, 0, 0);
    CHECK(need <= 16, "%s: remat needs %d regs > 16", name, need);

    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    rj_options o;
    rj_options_default(&o);
    cand_t cand = { .isa = isa, .mul = MUL_NIBBLE, .U = U, .remat = 1 };
    version_t v;
    CHECK(jit_build_version(&pl, &cand, &cpu, &o, &v, err, sizeof(err)) == 0,
          "%s build: %s", name, err);
    if (failures) {
        plan_free(&pl);
        rj_op_free(&op);
        topo_int_free(&ti);
        return;
    }

    int nd = ti.ndata, np = ti.npar;
    for (int li = 0; li < nlens; li++) {
        size_t len = lens[li];
        uint8_t **d = malloc(sizeof(*d) * (size_t)nd);
        uint8_t **p = malloc(sizeof(*p) * (size_t)np);
        uint8_t **pr = malloc(sizeof(*pr) * (size_t)np);
        for (int i = 0; i < nd; i++) {
            d[i] = abuf(len);
            randfill(d[i], len);
        }
        for (int j = 0; j < np; j++) {
            p[j] = abuf(len);
            pr[j] = abuf(len);
        }
        ref_run(&pl, (const uint8_t **)d, pr, len);
        const void **in = malloc(sizeof(*in) * (size_t)nd);
        void **out = malloc(sizeof(*out) * (size_t)np);
        for (int i = 0; i < nd; i++)
            in[i] = d[i];
        for (int j = 0; j < np; j++)
            out[j] = p[j];
        v.fn(in, out, len);
        for (int j = 0; j < np; j++)
            CHECK(memcmp(p[j], pr[j], len) == 0,
                  "%s len=%zu par%d mismatch (ntriv=%d)", name, len, j,
                  ntriv);
        for (int i = 0; i < nd; i++)
            free(d[i]);
        for (int j = 0; j < np; j++) {
            free(p[j]);
            free(pr[j]);
        }
        free(d);
        free(p);
        free(pr);
        free(in);
        free(out);
        if (failures)
            break;
    }
    jit_free_version(&v);
    plan_free(&pl);
    rj_op_free(&op);
    topo_int_free(&ti);
    if (!failures)
        printf("ok remat %s (ntriv=%d)\n", name, ntriv);
}

/* The planner must enumerate the NEON U=3 remat twin for RS(6,3)
 * (persistent 18 regs > 16 >= remat 15) and must NOT enumerate a remat
 * twin at U=2 (persistent 15 regs already fits: the twin policy). */
static void test_planner_twin(void)
{
    char err[256];
    rj_topology *t = rj_topo_rs(6, 3);
    CHECK(t != NULL, "rj_topo_rs(6,3)");
    if (failures)
        return;
    topo_int_t ti;
    CHECK(topo_int_build(&ti, t, err, sizeof(err)) == 0, "topo: %s", err);
    rj_op op = rj_op_encode();
    plan_t pl;
    CHECK(plan_build(&ti, &op, "rs(6,3)", &pl, err, sizeof(err)) == 0,
          "plan: %s", err);
    CHECK(pl.kind == P_HORNER, "expected P_HORNER, got kind %d", pl.kind);

    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    cand_t cands[512];
    int n = plan_candidates(&pl, &cpu, cands, 512, 0, 0);
    CHECK(n > 0, "no candidates");
    int found_u3 = 0, found_u2 = 0;
    for (int i = 0; i < n; i++) {
        if (cands[i].remat && cands[i].mul == MUL_NIBBLE &&
            cands[i].isa == ISA_NEON) {
            if (cands[i].U == 3)
                found_u3 = 1;
            if (cands[i].U == 2)
                found_u2 = 1;
        }
    }
    CHECK(found_u3, "NEON U=3 remat twin not enumerated (n=%d)", n);
    CHECK(!found_u2, "NEON U=2 remat twin enumerated despite "
          "persistent fitting (twin policy violated)");

    plan_free(&pl);
    rj_op_free(&op);
    topo_int_free(&ti);
    rj_topo_free(t);
    if (!failures)
        printf("ok remat planner twin\n");
}

int main(void)
{
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    if (!cpu.neon && !cpu.sve) {
        printf("test_aarch64_remat: SKIP (no NEON/SVE)\n");
        return 0;
    }

    rj_topology *t63 = rj_topo_rs(6, 3);
    CHECK(t63 != NULL, "rj_topo_rs(6,3)");
    if (t63) {
        /* Headline twin: NEON U=3 needs 18 regs persistent, 15 remat. */
        test_remat_case("rs(6,3) neon U=3", t63, ISA_NEON, 3, 0, 2);
        rj_topo_free(t63);
    }

    /* All-trivial coefficients: rewrite the RS(6,3) plan to an
     * all-ones Horner plan (mat and horner_c alike). Exercises the
     * c in {0,1} fast paths in remat mode. */
    {
        rj_topology *ta = rj_topo_rs(6, 3);
        CHECK(ta != NULL, "rj_topo_rs(6,3)");
        if (ta) {
            test_remat_case("rs(6,3) all-trivial neon U=3", ta,
                            ISA_NEON, 3, 1, 0);
            rj_topo_free(ta);
        }
    }

    /* ntriv=1 boundary: RAID-6 NEON U=5 needs 17 regs persistent,
     * 16 remat. */
    {
        rj_topology *t6 = rj_topo_raid6(8);
        CHECK(t6 != NULL, "rj_topo_raid6(8)");
        if (t6) {
            test_remat_case("raid6(8) neon U=5", t6, ISA_NEON, 5, 0, 1);
            rj_topo_free(t6);
        }
    }

    if (cpu.sve) {
        rj_topology *ts = rj_topo_rs(6, 3);
        CHECK(ts != NULL, "rj_topo_rs(6,3)");
        if (ts) {
            test_remat_case("rs(6,3) sve U=3", ts, ISA_SVE, 3, 0, 2);
            rj_topo_free(ts);
        }
    } else {
        printf("SKIP: host lacks SVE\n");
    }

    test_planner_twin();

    if (failures) {
        printf("%d FAILURES\n", failures);
        return 1;
    }
    printf("aarch64_remat: all OK\n");
    return 0;
}
