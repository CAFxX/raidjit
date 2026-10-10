/* Permanent regression test for Horner rematerialization codegen (x86).
 *
 * Force-builds remat candidates via jit_build_version() -- bypassing the
 * planner's twin policy -- and validates the jitted output against the
 * scalar reference (ref_run). Any divergence fails the test, so a remat
 * codegen regression cannot slip through benchmark/winner-only coverage
 * (test_correct/test_matrix only ever run the benchmark winner, which is
 * rarely a remat twin).
 *
 * Matrix note: the RS(6,3)/RAID-6 cases below use explicit custom
 * Vandermonde matrices (the historical default via rs_default_matrix),
 * not the rj_topo_rs/rj_topo_raid6 constructors. Since A2 2.0 those
 * constructors emit searched non-Vandermonde matrices (P_MULACC), and
 * rematerialization is implemented only for P_HORNER, so the searched
 * topologies cannot exercise the remat twins at all.
 *
 * Cases:
 *   - RS(6,3) encode, AVX-512 nibble, U=8, remat: the headline twin
 *     (persistent needs 34 regs, remat needs 31).
 *   - RS(6,3) encode, AVX-512 GFNI, U=9, remat: GFNI remat path.
 *   - RAID-6 encode, AVX-512 nibble, U=12, remat: ntriv=1 boundary,
 *     exercises the c=1 fast path in remat mode.
 *   - all-ones custom topology (6 data, 3 par), AVX-512 nibble, U=8,
 *     remat: all-trivial boundary (horner_c all 1s), exercises the
 *     c in {0,1} fast paths and the dead mask-load elision.
 *   - planner check: plan_candidates() on a custom RS(6,4) must enumerate
 *     the AVX-512 GFNI U=7 remat twin (persistent 33 regs, remat 31).
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

static uint64_t rng_state = 0x51ab3d9e77f2c104ULL;
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

/* Custom Vandermonde topology via the historical default matrix
 * (rs_default_matrix): the A2 1.0 search's fixed point was byte-identical
 * to this, and the planner still picks P_HORNER for it -- unlike the
 * A2 2.0 searched matrices, which are non-Vandermonde (P_MULACC).
 * Used so the remat cases below keep testing the P_HORNER remat twins. */
static rj_topology *vandermonde_topo(int ndata, int npar)
{
    uint8_t *mat = malloc((size_t)npar * (size_t)ndata);
    if (!mat)
        return NULL;
    rs_default_matrix(ndata, npar, mat);
    rj_topology *t = rj_topo_custom(ndata, npar, mat);
    free(mat);
    return t;
}

/* Force-build one remat candidate and check it against the scalar ref.
 * expect_ntriv >= 0 locks the test to a specific nontrivial-coefficient
 * count (boundary coverage); -1 disables the check. force_all_trivial
 * rewrites the plan to an all-ones Horner plan (mat and horner_c alike,
 * so ref_run stays consistent) to exercise the c in {0,1} fast paths
 * and the dead mask-load elision in remat mode.
 *
 * Kind note: this test asserts P_HORNER deliberately. Rematerialization
 * is implemented only for P_HORNER (jit_build_version asserts, and
 * plan_candidates enumerates remat twins only under pl->kind ==
 * P_HORNER), so P_MULACC plans cannot exercise it. Since A2 2.0 the
 * searched topologies (rj_topo_rs / rj_topo_raid6) emit non-Vandermonde
 * matrices and pick P_MULACC; the callers below therefore pass explicit
 * custom Vandermonde matrices (vandermonde_topo), for which the planner
 * still takes the Horner branch. */
static void test_remat_case(const char *name, rj_topology *t, mul_t mul,
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
        /* P_HORNER is required, not incidental: remat twins exist only
         * for Horner plans, and callers pass explicit custom
         * Vandermonde matrices for exactly this reason. */
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
    int need = plan_need_regs(ISA_AVX512, mul, &pl, U, 1, 0, 0, 0);
    CHECK(need <= 32, "%s: remat needs %d regs > 32", name, need);

    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    rj_options o;
    rj_options_default(&o);
    cand_t cand = { .isa = ISA_AVX512, .mul = mul, .U = U, .remat = 1 };
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

/* The planner must enumerate the GFNI remat twin now that the
 * MUL_NIBBLE-only guard is gone: custom RS(6,4), AVX-512 GFNI U=7 needs
 * 31 regs remat vs 33 persistent. */
static void test_planner_gfni_twin(void)
{
    char err[256];
    uint8_t mat[4 * 6];
    rs_default_matrix(6, 4, mat);
    rj_topology *t = rj_topo_custom(6, 4, mat);
    CHECK(t != NULL, "custom rs(6,4) topo");
    if (failures)
        return;
    topo_int_t ti;
    CHECK(topo_int_build(&ti, t, err, sizeof(err)) == 0, "topo: %s", err);
    rj_op op = rj_op_encode();
    plan_t pl;
    CHECK(plan_build(&ti, &op, "rs(6,4)", &pl, err, sizeof(err)) == 0,
          "plan: %s", err);
    /* P_HORNER is still correct here: this is the custom historical
     * Vandermonde matrix (rs_default_matrix), not an A2 2.0 searched
     * one, so plan_select_strategy takes the Horner branch. */
    CHECK(pl.kind == P_HORNER, "expected P_HORNER, got kind %d", pl.kind);

    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    cand_t cands[512];
    int n = plan_candidates(&pl, &cpu, cands, 512, 0, 0);
    CHECK(n > 0, "no candidates");
    int found = 0;
    for (int i = 0; i < n; i++)
        if (cands[i].remat && cands[i].mul == MUL_GFNI &&
            cands[i].isa == ISA_AVX512 && cands[i].U == 7)
            found = 1;
    CHECK(found, "GFNI U=7 remat twin not enumerated (n=%d)", n);

    plan_free(&pl);
    rj_op_free(&op);
    topo_int_free(&ti);
    rj_topo_free(t);
    if (!failures)
        printf("ok remat planner GFNI twin\n");
}

int main(void)
{
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    if (!cpu.avx512f || !cpu.avx512bw) {
        printf("SKIP: host lacks AVX-512BW\n");
        return 0;
    }

    rj_topology *t63 = vandermonde_topo(6, 3);
    CHECK(t63 != NULL, "vandermonde_topo(6,3)");
    if (t63) {
        test_remat_case("rs(6,3) nibble U=8", t63, MUL_NIBBLE, 8, 0, 2);
        rj_topo_free(t63);
    }

    /* All-trivial coefficients: rewrite the RS(6,3) plan to an
     * all-ones Horner plan (mat and horner_c alike). Exercises the
     * c in {0,1} fast paths and the dead mask-load elision in remat
     * mode. */
    {
        rj_topology *ta = vandermonde_topo(6, 3);
        CHECK(ta != NULL, "vandermonde_topo(6,3)");
        if (ta) {
            test_remat_case("rs(6,3) all-trivial nibble U=8", ta,
                            MUL_NIBBLE, 8, 1, 0);
            rj_topo_free(ta);
        }
    }

    rj_topology *t6 = vandermonde_topo(8, 2);
    CHECK(t6 != NULL, "vandermonde_topo(8,2)");
    if (t6) {
        test_remat_case("raid6(8) nibble U=12", t6, MUL_NIBBLE, 12, 0, 1);
        rj_topo_free(t6);
    }

    if (cpu.gfni) {
        rj_topology *tg = vandermonde_topo(6, 3);
        CHECK(tg != NULL, "vandermonde_topo(6,3)");
        if (tg) {
            test_remat_case("rs(6,3) gfni U=9", tg, MUL_GFNI, 9, 0, 2);
            rj_topo_free(tg);
        }
        test_planner_gfni_twin();
    } else {
        printf("SKIP: host lacks GFNI\n");
    }

    if (failures) {
        printf("%d FAILURES\n", failures);
        return 1;
    }
    printf("remat: all OK\n");
    return 0;
}
