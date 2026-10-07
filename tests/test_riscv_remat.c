/* Permanent regression test for Horner rematerialization codegen (RVV).
 *
 * Force-builds remat candidates via jit_build_version() -- bypassing the
 * planner's twin policy -- and validates the jitted output against the
 * scalar reference (ref_run). Any divergence fails the test, so a remat
 * codegen regression cannot slip through benchmark/winner-only coverage
 * (test_correct/test_matrix only ever run the benchmark winner, which is
 * rarely a remat twin).
 *
 * RVV implements remat for the nibble method only (the sole multiply
 * method on this ISA). In remat mode the prologue does not preload table
 * registers (hreg[] stays -1); the body hoists both table loads per
 * (i,j) -- NIBLO into the tbl scratch, NIBHI into the mask register
 * repurposed as the second table scratch (the mask itself is dead: the
 * body uses vand.vi with the 0x0F immediate), so the u-loop runs inside
 * the tables' live scope with no extra register and the proven
 * (n_out+1)*U+6 pressure formula is unchanged.
 *
 * Cases (RVV, 32 vector regs; remat need = (n_out+1)*U+6):
 *   - RS(6,3) encode, RVV nibble, U=6, remat: the headline twin
 *     (persistent needs 33 regs, remat needs 30).
 *   - RS(6,3) encode, RVV nibble, U=5, remat, all-trivial coefficients:
 *     exercises the c==1 fast path in remat mode (no constructor
 *     produces a c==0 Horner coefficient, so the c==0 path is not
 *     covered here).
 *   - custom(6,4) Vandermonde encode, RVV nibble, U=4, remat:
 *     n_out=4, ntriv=4 (persistent 33 regs, remat 26).
 *   - planner checks: plan_candidates() on custom(6,4) must enumerate
 *     the RVV U=5 and U=4 remat twins (persistent 38/33 > 32 >= remat
 *     31/26) and must NOT enumerate a remat twin at U=3, U=2 or U=1
 *     (persistent 28/23/18 regs already fit: the twin policy); on
 *     RS(6,3) it must enumerate no remat twin at any U (persistent
 *     always fits).
 *   - over-pressure rejection: force-building an RVV remat candidate
 *     that needs more than 32 vector regs (RS(6,3) U=8, remat need 38)
 *     must fail cleanly with an error, not abort or emit code that
 *     overruns the regfile.
 *
 * Run under QEMU at several VLENs, e.g.
 *   QEMU_LD_PREFIX=/usr/riscv64-linux-gnu qemu-riscv64 -cpu max \
 *       ./build-riscv64/test_riscv_remat
 *   QEMU_LD_PREFIX=/usr/riscv64-linux-gnu qemu-riscv64 -cpu max,vlen=256 \
 *       ./build-riscv64/test_riscv_remat
 *   QEMU_LD_PREFIX=/usr/riscv64-linux-gnu qemu-riscv64 -cpu max,vlen=512 \
 *       ./build-riscv64/test_riscv_remat
 * Without RVV it prints SKIP and passes.
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

#ifdef __riscv
/* Read the VLENB CSR directly (rj_cpu_detect only reports a 16-byte
 * fallback VLEN, so this is the ground truth for the QEMU -cpu
 * max,vlen=... runs). */
static unsigned read_vlenb(void)
{
    unsigned long v;
    __asm__ volatile("csrr %0, vlenb" : "=r"(v));
    return (unsigned)v;
}
#endif

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
 * so ref_run stays consistent) to exercise the c==1 fast path in
 * remat mode. */
static void test_remat_case(const char *name, rj_topology *t, int U,
                            int force_all_trivial, int expect_ntriv)
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
    int need = plan_need_regs(ISA_RVV, MUL_NIBBLE, &pl, U, 1, 0);
    CHECK(need <= 32, "%s: remat needs %d regs > 32", name, need);
    int need_persist = plan_need_regs(ISA_RVV, MUL_NIBBLE, &pl, U, 0, 0);
    printf("info %s: persistent needs %d regs, remat needs %d regs\n",
           name, need_persist, need);

    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    rj_options o;
    rj_options_default(&o);
    cand_t cand = { .isa = ISA_RVV, .mul = MUL_NIBBLE, .U = U, .remat = 1 };
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

/* Build a custom ndata x npar Vandermonde topology in "Horner order"
 * (the same construction rj_topo_rs uses, extended past its npar <= 3
 * envelope): M[j][i] = g^((j+1)*(ndata-1-i)), g=2, so horner_c[j] =
 * 2^(j+1) and every row is nontrivial. plan_build must see P_HORNER. */
static rj_topology *topo_vandermonde4(int ndata)
{
    int npar = 4;
    uint8_t *m = malloc((size_t)npar * (size_t)ndata);
    if (!m)
        return NULL;
    for (int j = 0; j < npar; j++)
        for (int i = 0; i < ndata; i++)
            m[(size_t)j * (size_t)ndata + (size_t)i] =
                gf_pow(2, (unsigned)((j + 1) * (ndata - 1 - i)));
    rj_topology *t = rj_topo_custom(ndata, npar, m);
    free(m);
    return t;
}

/* The planner must enumerate the RVV remat twins for custom(6,4)
 * exactly where the twin policy says: for each (U, lmul), a remat twin
 * iff the persistent variant needs > 32 physical regs but remat fits.
 * (At lmul=1: U=5 and U=4 (persistent 38/33 regs); at wider lmul the
 * same rule applies per (U, lmul).) On RS(6,3) the same per-(U,lmul)
 * rule is verified. */
static void test_planner_twin(void)
{
    char err[256];
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    cand_t cands[512];

    rj_topology *tc = topo_vandermonde4(6);
    CHECK(tc != NULL, "topo_vandermonde4(6)");
    if (tc) {
        topo_int_t ti;
        CHECK(topo_int_build(&ti, tc, err, sizeof(err)) == 0, "topo: %s",
              err);
        rj_op op = rj_op_encode();
        plan_t pl;
        CHECK(plan_build(&ti, &op, "custom(6,4)", &pl, err, sizeof(err)) == 0,
              "plan: %s", err);
        CHECK(pl.kind == P_HORNER, "expected P_HORNER, got kind %d",
              pl.kind);
        CHECK(count_ntriv(&pl) == 4, "custom(6,4): ntriv=%d, want 4",
              count_ntriv(&pl));

        int n = plan_candidates(&pl, &cpu, cands, 512, 0, 0);
        CHECK(n > 0, "no candidates");
        /* Twin policy, lmul-aware: for each (U, lmul) a remat twin is
         * enumerated iff the persistent variant does not fit in 32
         * physical regs at (U, lmul) but the remat variant does. */
        static const int lmuls[] = { 1, 2, 4, 8 };
        static const int Us[] = { 5, 4, 3, 2, 1 };
        for (int ui = 0; ui < 5; ui++) {
            int U = Us[ui];
            for (int li = 0; li < 4; li++) {
                int lmul = lmuls[li];
                int need_p =
                    plan_need_regs(ISA_RVV, MUL_NIBBLE, &pl, U, 0, 0) * lmul;
                int need_r =
                    plan_need_regs(ISA_RVV, MUL_NIBBLE, &pl, U, 1, 0) * lmul;
                int want = (need_p > 32 && need_r <= 32);
                int found = 0;
                for (int i = 0; i < n; i++)
                    if (cands[i].remat && cands[i].mul == MUL_NIBBLE &&
                        cands[i].isa == ISA_RVV && cands[i].U == U &&
                        (cands[i].lmul ? cands[i].lmul : 1) == lmul)
                        found = 1;
                CHECK(found == want,
                      "custom(6,4) U=%d lmul=%d: remat twin %s, policy says %s "
                      "(persistent %d, remat %d regs)",
                      U, lmul, found ? "enumerated" : "absent",
                      want ? "enumerate" : "skip", need_p, need_r);
            }
        }

        plan_free(&pl);
        rj_op_free(&op);
        topo_int_free(&ti);
        rj_topo_free(tc);
    }

    rj_topology *t63 = rj_topo_rs(6, 3);
    CHECK(t63 != NULL, "rj_topo_rs(6,3)");
    if (t63) {
        topo_int_t ti;
        CHECK(topo_int_build(&ti, t63, err, sizeof(err)) == 0, "topo: %s",
              err);
        rj_op op = rj_op_encode();
        plan_t pl;
        CHECK(plan_build(&ti, &op, "rs(6,3)", &pl, err, sizeof(err)) == 0,
              "plan: %s", err);
        CHECK(pl.kind == P_HORNER, "expected P_HORNER, got kind %d",
              pl.kind);
        int n = plan_candidates(&pl, &cpu, cands, 512, 0, 0);
        /* Twin policy, lmul-aware: a remat twin may appear at (U, lmul)
         * only when persistent does not fit there but remat does. */
        static const int lmuls63[] = { 1, 2, 4, 8 };
        static const int Us63[] = { 5, 4, 3, 2, 1 };
        for (int ui = 0; ui < 5; ui++) {
            for (int li = 0; li < 4; li++) {
                int U = Us63[ui], lmul = lmuls63[li];
                int need_p =
                    plan_need_regs(ISA_RVV, MUL_NIBBLE, &pl, U, 0, 0) * lmul;
                int need_r =
                    plan_need_regs(ISA_RVV, MUL_NIBBLE, &pl, U, 1, 0) * lmul;
                int want = (need_p > 32 && need_r <= 32);
                int found = 0;
                for (int i = 0; i < n; i++)
                    if (cands[i].remat && cands[i].isa == ISA_RVV &&
                        cands[i].U == U &&
                        (cands[i].lmul ? cands[i].lmul : 1) == lmul)
                        found = 1;
                CHECK(found == want,
                      "RS(6,3) U=%d lmul=%d: remat twin %s, policy says %s",
                      U, lmul, found ? "enumerated" : "absent",
                      want ? "enumerate" : "skip");
            }
        }
        plan_free(&pl);
        rj_op_free(&op);
        topo_int_free(&ti);
        rj_topo_free(t63);
    }

    if (!failures)
        printf("ok remat planner twin\n");
}

/* Over-pressure remat candidates must fail cleanly (return -1 with an
 * error), not abort or emit code that overruns the regfile. RS(6,3) U=8
 * remat needs (3+1)*8+6 = 38 regs > 32. */
static void test_build_rejected(void)
{
    char err[256];
    rj_topology *t = rj_topo_rs(6, 3);
    CHECK(t != NULL, "rj_topo_rs(6,3)");
    if (!t)
        return;
    topo_int_t ti;
    if (topo_int_build(&ti, t, err, sizeof(err)) != 0) {
        CHECK(0, "topo: %s", err);
        rj_topo_free(t);
        return;
    }
    rj_op op = rj_op_encode();
    plan_t pl;
    if (plan_build(&ti, &op, "rs(6,3)", &pl, err, sizeof(err)) != 0) {
        CHECK(0, "plan: %s", err);
        topo_int_free(&ti);
        rj_topo_free(t);
        return;
    }
    CHECK(plan_need_regs(ISA_RVV, MUL_NIBBLE, &pl, 8, 1, 0) == 38,
          "remat U=8 need=%d, want 38",
          plan_need_regs(ISA_RVV, MUL_NIBBLE, &pl, 8, 1, 0));
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    rj_options o;
    rj_options_default(&o);
    cand_t cand = { .isa = ISA_RVV, .mul = MUL_NIBBLE, .U = 8, .remat = 1 };
    version_t v;
    int rc = jit_build_version(&pl, &cand, &cpu, &o, &v, err, sizeof(err));
    CHECK(rc != 0, "over-pressure remat build succeeded (want clean fail)");
    if (rc == 0)
        jit_free_version(&v);
    plan_free(&pl);
    rj_op_free(&op);
    topo_int_free(&ti);
    rj_topo_free(t);
    if (!failures)
        printf("ok remat build rejected\n");
}

/* End-to-end: rj_compile() the custom(6,4) topology, for which the
 * planner now enumerates RVV remat twins, and check the benchmark
 * winner against the scalar reference. This exercises the twin
 * through the real plan -> bench -> select -> run pipeline (candidate
 * enumeration with remat present, desc formatting, cost model). */
static void test_compile_integration(void)
{
    char err[256];
    rj_topology *tc = topo_vandermonde4(6);
    CHECK(tc != NULL, "topo_vandermonde4(6)");
    if (!tc)
        return;
    topo_int_t ti;
    if (topo_int_build(&ti, tc, err, sizeof(err)) != 0) {
        CHECK(0, "topo: %s", err);
        rj_topo_free(tc);
        return;
    }
    rj_op op = rj_op_encode();
    plan_t pl;
    if (plan_build(&ti, &op, "custom(6,4)", &pl, err, sizeof(err)) != 0) {
        CHECK(0, "plan: %s", err);
        topo_int_free(&ti);
        rj_topo_free(tc);
        return;
    }

    rj_options o;
    rj_options_default(&o);
    rj_jit *j = rj_compile(tc, &op, &o, err, sizeof(err));
    CHECK(j != NULL, "rj_compile: %s", err);
    if (j) {
        rj_fn fn = rj_get_fn(j);
        int nd = ti.ndata, np = ti.npar;
        size_t len = 4096;
        uint8_t **d = malloc(sizeof(*d) * (size_t)nd);
        uint8_t **p = malloc(sizeof(*p) * (size_t)np);
        uint8_t **pr = malloc(sizeof(*pr) * (size_t)np);
        for (int i = 0; i < nd; i++) {
            d[i] = abuf(len);
            randfill(d[i], len);
        }
        for (int k = 0; k < np; k++) {
            p[k] = abuf(len);
            pr[k] = abuf(len);
        }
        ref_run(&pl, (const uint8_t **)d, pr, len);
        const void **in = malloc(sizeof(*in) * (size_t)nd);
        void **out = malloc(sizeof(*out) * (size_t)np);
        for (int i = 0; i < nd; i++)
            in[i] = d[i];
        for (int k = 0; k < np; k++)
            out[k] = p[k];
        fn(in, out, len);
        for (int k = 0; k < np; k++)
            CHECK(memcmp(p[k], pr[k], len) == 0,
                  "compile integration par%d mismatch", k);
        for (int i = 0; i < nd; i++)
            free(d[i]);
        for (int k = 0; k < np; k++) {
            free(p[k]);
            free(pr[k]);
        }
        free(d);
        free(p);
        free(pr);
        free(in);
        free(out);
        rj_free(j);
    }
    plan_free(&pl);
    rj_op_free(&op);
    topo_int_free(&ti);
    rj_topo_free(tc);
    if (!failures)
        printf("ok remat compile integration\n");
}

int main(void)
{
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    if (!cpu.rvv) {
        printf("test_riscv_remat: SKIP (no RVV)\n");
        return 0;
    }
#ifdef __riscv
    {
        unsigned vlenb = read_vlenb();
        printf("info vlenb=%u (VLEN=%u bits)\n", vlenb, vlenb * 8);
        CHECK(vlenb >= 16, "vlenb=%u < 16", vlenb);
        CHECK((vlenb & (vlenb - 1)) == 0, "vlenb=%u not a power of two",
              vlenb);
    }
#endif

    rj_topology *t63 = rj_topo_rs(6, 3);
    CHECK(t63 != NULL, "rj_topo_rs(6,3)");
    if (t63) {
        /* Headline twin: RVV U=6 needs 33 regs persistent, 30 remat. */
        test_remat_case("rs(6,3) rvv U=6", t63, 6, 0, 2);
        rj_topo_free(t63);
    }

    /* All-trivial coefficients: rewrite the RS(6,3) plan to an
     * all-ones Horner plan (mat and horner_c alike). Exercises the
     * c==1 fast path in remat mode. */
    {
        rj_topology *ta = rj_topo_rs(6, 3);
        CHECK(ta != NULL, "rj_topo_rs(6,3)");
        if (ta) {
            test_remat_case("rs(6,3) all-trivial rvv U=5", ta, 5, 1, 0);
            rj_topo_free(ta);
        }
    }

    /* n_out=4, ntriv=4: custom Vandermonde, U=4 needs 33 regs
     * persistent, 26 remat. */
    {
        rj_topology *tc = topo_vandermonde4(6);
        CHECK(tc != NULL, "topo_vandermonde4(6)");
        if (tc) {
            test_remat_case("custom(6,4) rvv U=4", tc, 4, 0, 4);
            rj_topo_free(tc);
        }
    }

    test_planner_twin();
    test_build_rejected();
    test_compile_integration();

    if (failures) {
        printf("%d FAILURES\n", failures);
        return 1;
    }
    printf("riscv_remat: all OK\n");
    return 0;
}
