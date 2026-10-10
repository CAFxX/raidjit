/* Regression tests for exact register-pressure accounting.
 *
 * plan_need_regs() is the single source of truth for how many vector
 * registers a (isa, mul, plan, U) needs. The planner drops candidates
 * with need > regfile size; the backends assert their assignment against
 * it. This test pins the exact values and the admission boundary that
 * once (off-by-one) admitted AVX-512 candidates using vector register
 * index 32, which the emitter silently wraps to zmm0.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/plan.h"
#include "../include/raidjit.h"

static int failures = 0;
#define CHECK(cond, ...) do { \
    if (!(cond)) { \
        printf("FAIL "); printf(__VA_ARGS__); printf("\n"); \
        failures++; \
    } \
} while (0)

#if defined(__x86_64__)
/* x86-only helpers for the planner-admission sections (2, 4, 7). */
static rj_cpu_info cpu_avx512_gfni(void)
{
    rj_cpu_info c;
    memset(&c, 0, sizeof(c));
    c.sse2 = c.ssse3 = c.avx = c.avx2 = 1;
    c.avx512f = c.avx512bw = c.avx512vl = c.gfni = 1;
    snprintf(c.brand, sizeof(c.brand), "synthetic");
    return c;
}

/* Build a P_MULACC plan with n_out outputs via a custom non-Vandermonde
 * matrix (identity with one nonzero off-diagonal per row keeps every
 * coefficient nonzero so nibble temps are really needed). */
static int build_mulacc_plan(int npar, plan_t *pl, char *err, size_t errlen)
{
    int ndata = 4;
    uint8_t *mat = calloc((size_t)npar * ndata, 1);
    if (!mat)
        return -1;
    for (int j = 0; j < npar; j++) {
        mat[(size_t)j * ndata + (j % ndata)] = 1;
        mat[(size_t)j * ndata + ((j + 1) % ndata)] = 2; /* not Vandermonde */
    }
    rj_topology *t = rj_topo_custom(ndata, npar, mat);
    free(mat);
    if (!t)
        return -1;
    topo_int_t ti;
    if (topo_int_build(&ti, t, err, errlen) != 0) {
        rj_topo_free(t);
        return -1;
    }
    rj_op op = rj_op_encode();
    int rc = plan_build(&ti, &op, "custom", pl, err, errlen);
    topo_int_free(&ti);
    rj_topo_free(t);
    return rc;
}
#endif

int main(void)
{
    char err[256];
#if defined(__x86_64__)
    /* Synthetic x86 CPU for the planner-admission sections (2, 4, 7).
     * plan_candidates compiles the host arch's branch only, so these
     * sections are x86-only; the rest of the test is portable. */
    rj_cpu_info cpu = cpu_avx512_gfni();
#endif

    /* 1. Exact values: hand-computed from the backend register assignment
     *    (see plan_need_regs). n_out=3, U=8, all-nontrivial Horner. */
    {
        plan_t pl;
        memset(&pl, 0, sizeof(pl));
        pl.kind = P_HORNER;
        pl.n_out = 3;
        pl.horner_c = malloc(3);
        pl.horner_c[0] = 2; pl.horner_c[1] = 3; pl.horner_c[2] = 4;
        /* x86 nibble: 3*8 acc + t,tlo,thi,r1,r2,mask (6) + 2*3 tables */
        CHECK(plan_need_regs(ISA_AVX512, MUL_NIBBLE, &pl, 8, 0, 0, 0) == 24 + 6 + 6,
              "x86 horner nibble: got %d want 36",
              plan_need_regs(ISA_AVX512, MUL_NIBBLE, &pl, 8, 0, 0, 0));
        /* x86 GFNI: 24 + t,t2 (2) + 3 tables */
        CHECK(plan_need_regs(ISA_AVX512, MUL_GFNI, &pl, 8, 0, 0, 0) == 24 + 2 + 3,
              "x86 horner gfni: got %d want 29",
              plan_need_regs(ISA_AVX512, MUL_GFNI, &pl, 8, 0, 0, 0));
        /* RVV: (3+1)*8 staging+acc + tlo,thi,r1,r2,mask (5) + 2*3 tables */
        CHECK(plan_need_regs(ISA_RVV, MUL_NIBBLE, &pl, 8, 0, 0, 0) == 32 + 5 + 6,
              "rvv horner: got %d want 43",
              plan_need_regs(ISA_RVV, MUL_NIBBLE, &pl, 8, 0, 0, 0));
        /* RVV Zvbc: (3+1)*8 staging+acc + r1,r2,r3,kvec (4) + 3 cvec */
        CHECK(plan_need_regs(ISA_RVV, MUL_ZVBC, &pl, 8, 0, 0, 0) == 32 + 4 + 3,
              "rvv horner zvbc: got %d want 39",
              plan_need_regs(ISA_RVV, MUL_ZVBC, &pl, 8, 0, 0, 0));
        free(pl.horner_c);
    }
    {
        plan_t pl;
        memset(&pl, 0, sizeof(pl));
        pl.kind = P_MULACC;
        pl.n_out = 5;
        pl.n_in = 4;
        /* x86 nibble MULACC: 5*U acc + t,tlo,thi,r1,r2,tbl,mask (7) */
        CHECK(plan_need_regs(ISA_AVX512, MUL_NIBBLE, &pl, 6, 0, 0, 0) == 30 + 7,
              "x86 mulacc nibble U=6: got %d want 37",
              plan_need_regs(ISA_AVX512, MUL_NIBBLE, &pl, 6, 0, 0, 0));
        /* x86 GFNI MULACC: 30 + t,t2,tbl (3) = 33 > 32 */
        CHECK(plan_need_regs(ISA_AVX512, MUL_GFNI, &pl, 6, 0, 0, 0) == 33,
              "x86 mulacc gfni U=6: got %d want 33",
              plan_need_regs(ISA_AVX512, MUL_GFNI, &pl, 6, 0, 0, 0));
        CHECK(plan_need_regs(ISA_AVX512, MUL_GFNI, &pl, 5, 0, 0, 0) == 28,
              "x86 mulacc gfni U=5: got %d want 28",
              plan_need_regs(ISA_AVX512, MUL_GFNI, &pl, 5, 0, 0, 0));
        /* RVV Zvbc MULACC: 5*U acc + U data + r1,r2,r3,cvec,kvec (5) */
        CHECK(plan_need_regs(ISA_RVV, MUL_ZVBC, &pl, 6, 0, 0, 0) == 36 + 5,
              "rvv mulacc zvbc U=6: got %d want 41",
              plan_need_regs(ISA_RVV, MUL_ZVBC, &pl, 6, 0, 0, 0));
        /* XOR: acc + 1 temp on every ISA */
        pl.kind = P_XOR;
        CHECK(plan_need_regs(ISA_AVX2, MUL_NIBBLE, &pl, 8, 0, 0, 0) == 41,
              "xor: got %d want 41",
              plan_need_regs(ISA_AVX2, MUL_NIBBLE, &pl, 8, 0, 0, 0));
    }

    /* 2. Admission boundary: 5-output MULACC on AVX-512+GFNI. U=6 needs
     *    33 regs and must be rejected; U=5 needs 28 and must be offered.
     *    (The old off-by-one admitted U=6, whose backend then used
     *    vector register index 32.)
     *    x86-only: plan_candidates compiles the host arch's branch only,
     *    so the synthetic x86 CPU below yields no candidates elsewhere. */
#if defined(__x86_64__)
    {
        plan_t pl;
        if (build_mulacc_plan(5, &pl, err, sizeof(err)) != 0) {
            printf("FAIL plan: %s\n", err);
            return 1;
        }
        CHECK(pl.kind == P_MULACC, "expected P_MULACC, got kind=%d",
              pl.kind);
        cand_t cands[128];
        int nc = plan_candidates(&pl, &cpu, cands, 128, 0, 0);
        CHECK(nc > 0, "no candidates");
        int saw_gfni_u6 = 0, saw_gfni_u5 = 0, saw_nib_u4 = 0;
        int saw_gfni_u7 = 0;
        for (int i = 0; i < nc; i++) {
            /* every offered candidate must fit the regfile */
            int need = plan_need_regs(cands[i].isa, cands[i].mul, &pl,
                                      cands[i].U, cands[i].remat, 0, 0);
            CHECK(need <= isa_nregs(cands[i].isa),
                  "cand %s U=%d needs %d > %d regs",
                  isa_name(cands[i].isa), cands[i].U, need,
                  isa_nregs(cands[i].isa));
            if (cands[i].isa == ISA_AVX512 && cands[i].mul == MUL_GFNI) {
                if (cands[i].U == 7)
                    saw_gfni_u7 = 1;
                if (cands[i].U == 6)
                    saw_gfni_u6 = 1;
                if (cands[i].U == 5)
                    saw_gfni_u5 = 1;
            }
            if (cands[i].isa == ISA_AVX512 &&
                cands[i].mul == MUL_NIBBLE && cands[i].U == 4)
                saw_nib_u4 = 1;
        }
        CHECK(!saw_gfni_u7, "U=7 GFNI offered but needs 38 regs");
        CHECK(!saw_gfni_u6, "U=6 GFNI offered but needs 33 regs");
        CHECK(saw_gfni_u5, "U=5 GFNI (28 regs) should be offered");
        CHECK(saw_nib_u4, "U=4 nibble (27 regs) should be offered");
        plan_free(&pl);
    }
#endif

    /* 3. Horner 0/1-row compaction: rs(6,3)-style coefficients with one
     *    trivial row must not reserve table regs for it. */
    {
        plan_t pl;
        memset(&pl, 0, sizeof(pl));
        pl.kind = P_HORNER;
        pl.n_out = 3;
        pl.horner_c = malloc(3);
        pl.horner_c[0] = 1; /* trivial: no table regs */
        pl.horner_c[1] = 2;
        pl.horner_c[2] = 4;
        /* x86 nibble: 3*8 + 6 + 2*2 = 34 (not 36) */
        CHECK(plan_need_regs(ISA_AVX512, MUL_NIBBLE, &pl, 8, 0, 0, 0) == 34,
              "horner trivial-row: got %d want 34",
              plan_need_regs(ISA_AVX512, MUL_NIBBLE, &pl, 8, 0, 0, 0));
        free(pl.horner_c);
    }

    /* 4. U=7/U=8 admission: rs(6,3) encode is now MULACC n_out=3,
     *    nontrivial (the A2 2.0 searched matrix is non-Vandermonde, so
     *    plan_select_strategy picks P_MULACC, not P_HORNER -- the
     *    deliberate 2.0 parity-format change). x86 nibble non-remat
     *    needs n_out*U + 7 (t,tlo,thi,r1,r2,tbl,mask live): U=7 needs
     *    28, U=8 needs 31 -- both <= 32, so both are offered. Under the
     *    old P_HORNER plan U=8 needed 34 and was rejected; that verdict
     *    belonged to the old plan kind's accounting, not to the
     *    planner. x86-only, like section 2 (synthetic x86 CPU). */
#if defined(__x86_64__)
    {
        rj_topology *t = rj_topo_rs(6, 3);
        CHECK(t != NULL, "rj_topo_rs(6,3)");
        if (t) {
            topo_int_t ti;
            plan_t pl;
            rj_op op = rj_op_encode();
            CHECK(topo_int_build(&ti, t, err, sizeof(err)) == 0,
                  "topo build: %s", err);
            CHECK(plan_build(&ti, &op, "rs63", &pl, err, sizeof(err)) == 0,
                  "plan build: %s", err);
            CHECK(pl.kind == P_MULACC, "expected MULACC, got %d", pl.kind);
            /* Pin the exact P_MULACC nibble accounting (non-remat): one
             * more unroll would exceed the 32-reg file (3*9+7=34), so
             * U=8 is the boundary admission. */
            CHECK(plan_need_regs(ISA_AVX512, MUL_NIBBLE, &pl, 8, 0, 0, 0) == 31,
                  "mulacc nibble U=8: got %d want 31",
                  plan_need_regs(ISA_AVX512, MUL_NIBBLE, &pl, 8, 0, 0, 0));
            CHECK(plan_need_regs(ISA_AVX512, MUL_NIBBLE, &pl, 7, 0, 0, 0) == 28,
                  "mulacc nibble U=7: got %d want 28",
                  plan_need_regs(ISA_AVX512, MUL_NIBBLE, &pl, 7, 0, 0, 0));
            cand_t cands[128];
            int nc = plan_candidates(&pl, &cpu, cands, 128, 0, 0);
            CHECK(nc > 0, "no candidates");
            int saw_nib_u7 = 0, saw_nib_u8 = 0;
            for (int i = 0; i < nc; i++) {
                /* every offered candidate must fit the regfile */
                int need = plan_need_regs(cands[i].isa, cands[i].mul, &pl,
                                          cands[i].U, cands[i].remat, 0, 0);
                CHECK(need <= isa_nregs(cands[i].isa),
                      "cand %s U=%d needs %d > %d regs",
                      isa_name(cands[i].isa), cands[i].U, need,
                      isa_nregs(cands[i].isa));
                if (cands[i].isa == ISA_AVX512 &&
                    cands[i].mul == MUL_NIBBLE) {
                    if (cands[i].U == 7 && !cands[i].remat)
                        saw_nib_u7 = 1;
                    if (cands[i].U == 8 && !cands[i].remat)
                        saw_nib_u8 = 1;
                }
            }
            CHECK(saw_nib_u7, "U=7 nibble (28 regs) should be offered");
            CHECK(saw_nib_u8, "U=8 nibble (31 regs) should be offered");
            rj_op_free(&op);
            plan_free(&pl);
            topo_int_free(&ti);
            rj_topo_free(t);
        }
    }
#endif

    /* 5. Coefficient-aware allocation: P_DELTA with an all-{0,1} delta
     *    matrix (RAID-5 parity update) must not reserve the dead nibble
     *    temps. plan_any_nontrivial() is the single source of truth. */
    {
        int ch[1] = { 0 };
        rj_topology *t = rj_topo_raid5(4);
        CHECK(t != NULL, "rj_topo_raid5(4)");
        if (t) {
            topo_int_t ti;
            plan_t pl;
            rj_op op = rj_op_parity_update(ch, 1);
            CHECK(topo_int_build(&ti, t, err, sizeof(err)) == 0,
                  "topo build: %s", err);
            CHECK(plan_build(&ti, &op, "r5pu", &pl, err, sizeof(err)) == 0,
                  "plan build: %s", err);
            CHECK(pl.kind == P_DELTA, "expected P_DELTA, got %d", pl.kind);
            CHECK(plan_any_nontrivial(&pl) == 0,
                  "raid5 delta matrix should be all-trivial");
            /* n_out=1, U=4. x86 nibble: AVX2/AVX-512 drop
             * tlo/thi/r1/r2/tbl/mask (4+1=5); SSE keeps r1 as the delta
             * temp (4+2=6). GFNI drops t2/tbl (4+1=5). */
            CHECK(plan_need_regs(ISA_AVX2, MUL_NIBBLE, &pl, 4, 0, 0, 0) == 5,
                  "delta-trivial avx2 nibble: got %d want 5",
                  plan_need_regs(ISA_AVX2, MUL_NIBBLE, &pl, 4, 0, 0, 0));
            CHECK(plan_need_regs(ISA_SSE, MUL_NIBBLE, &pl, 4, 0, 0, 0) == 6,
                  "delta-trivial sse nibble: got %d want 6",
                  plan_need_regs(ISA_SSE, MUL_NIBBLE, &pl, 4, 0, 0, 0));
            CHECK(plan_need_regs(ISA_AVX512, MUL_NIBBLE, &pl, 4, 0, 0, 0) == 5,
                  "delta-trivial avx512 nibble: got %d want 5",
                  plan_need_regs(ISA_AVX512, MUL_NIBBLE, &pl, 4, 0, 0, 0));
            CHECK(plan_need_regs(ISA_AVX512, MUL_GFNI, &pl, 4, 0, 0, 0) == 5,
                  "delta-trivial avx512 gfni: got %d want 5",
                  plan_need_regs(ISA_AVX512, MUL_GFNI, &pl, 4, 0, 0, 0));
            /* AArch64 (NEON/SVE): r1 stays live as the delta temp
             * (4+2=6). RVV: vd0[U] + tlo/thi delta temps
             * ((1+1)*4+2=10). */
            CHECK(plan_need_regs(ISA_NEON, MUL_NIBBLE, &pl, 4, 0, 0, 0) == 6,
                  "delta-trivial neon: got %d want 6",
                  plan_need_regs(ISA_NEON, MUL_NIBBLE, &pl, 4, 0, 0, 0));
            CHECK(plan_need_regs(ISA_SVE, MUL_NIBBLE, &pl, 4, 0, 0, 0) == 6,
                  "delta-trivial sve: got %d want 6",
                  plan_need_regs(ISA_SVE, MUL_NIBBLE, &pl, 4, 0, 0, 0));
            CHECK(plan_need_regs(ISA_RVV, MUL_NIBBLE, &pl, 4, 0, 0, 0) == 10,
                  "delta-trivial rvv: got %d want 10",
                  plan_need_regs(ISA_RVV, MUL_NIBBLE, &pl, 4, 0, 0, 0));
            rj_op_free(&op);
            plan_free(&pl);
            topo_int_free(&ti);
            rj_topo_free(t);
        }
    }
    /* 6. Nontrivial P_DELTA control: RAID-6 parity update keeps the old
     *    counts exactly (n_out=2, U=4). */
    {
        int ch[1] = { 0 };
        rj_topology *t = rj_topo_raid6(4);
        CHECK(t != NULL, "rj_topo_raid6(4)");
        if (t) {
            topo_int_t ti;
            plan_t pl;
            rj_op op = rj_op_parity_update(ch, 1);
            CHECK(topo_int_build(&ti, t, err, sizeof(err)) == 0,
                  "topo build: %s", err);
            CHECK(plan_build(&ti, &op, "r6pu", &pl, err, sizeof(err)) == 0,
                  "plan build: %s", err);
            CHECK(pl.kind == P_DELTA, "expected P_DELTA, got %d", pl.kind);
            CHECK(plan_any_nontrivial(&pl) == 1,
                  "raid6 delta matrix should be nontrivial");
            CHECK(plan_need_regs(ISA_AVX2, MUL_NIBBLE, &pl, 4, 0, 0, 0) == 15,
                  "delta-nontriv avx2 nibble: got %d want 15",
                  plan_need_regs(ISA_AVX2, MUL_NIBBLE, &pl, 4, 0, 0, 0));
            CHECK(plan_need_regs(ISA_AVX512, MUL_GFNI, &pl, 4, 0, 0, 0) == 11,
                  "delta-nontriv avx512 gfni: got %d want 11",
                  plan_need_regs(ISA_AVX512, MUL_GFNI, &pl, 4, 0, 0, 0));
            CHECK(plan_need_regs(ISA_NEON, MUL_NIBBLE, &pl, 4, 0, 0, 0) == 16,
                  "delta-nontriv neon: got %d want 16",
                  plan_need_regs(ISA_NEON, MUL_NIBBLE, &pl, 4, 0, 0, 0));
            /* SVE shares the a64 logical formula (n_out*U+8 with hoisted
             * tables); the 24-reg file is enforced via isa_nregs. */
            CHECK(plan_need_regs(ISA_SVE, MUL_NIBBLE, &pl, 4, 0, 0, 0) == 16,
                  "delta-nontriv sve: got %d want 16",
                  plan_need_regs(ISA_SVE, MUL_NIBBLE, &pl, 4, 0, 0, 0));
            CHECK(plan_need_regs(ISA_RVV, MUL_NIBBLE, &pl, 4, 0, 0, 0) == 19,
                  "delta-nontriv rvv: got %d want 19",
                  plan_need_regs(ISA_RVV, MUL_NIBBLE, &pl, 4, 0, 0, 0));
            rj_op_free(&op);
            plan_free(&pl);
            topo_int_free(&ti);
            rj_topo_free(t);
        }
    }
    /* 7. Planner admission: the freed registers must unlock larger U.
     *    Hand-built all-trivial P_DELTA, n_out=2, on SSE (16 regs):
     *    U=5 needs 2*5+2=12 now (was 2*5+7=17, rejected).
     *    x86-only: plan_candidates compiles the host arch's branch only,
     *    so synthetic x86 CPUs yield no candidates elsewhere. */
#if defined(__x86_64__)
    {
        plan_t pl;
        memset(&pl, 0, sizeof(pl));
        pl.kind = P_DELTA;
        pl.n_out = 2;
        pl.nchanged = 1;
        pl.mat = calloc(2, 1);
        pl.mat[0] = pl.mat[1] = 1;
        CHECK(plan_any_nontrivial(&pl) == 0, "hand-built delta trivial");
        CHECK(plan_need_regs(ISA_SSE, MUL_NIBBLE, &pl, 5, 0, 0, 0) == 12,
              "hand-built delta sse U=5: got %d want 12",
              plan_need_regs(ISA_SSE, MUL_NIBBLE, &pl, 5, 0, 0, 0));
        rj_cpu_info cpusse;
        memset(&cpusse, 0, sizeof(cpusse));
        cpusse.sse2 = cpusse.ssse3 = 1;
        snprintf(cpusse.brand, sizeof(cpusse.brand), "synthetic-sse");
        cand_t cands[128];
        int nc = plan_candidates(&pl, &cpusse, cands, 128, 0, 0);
        int saw_u5 = 0, saw_u8 = 0;
        for (int i = 0; i < nc; i++) {
            if (cands[i].isa == ISA_SSE && cands[i].mul == MUL_NIBBLE &&
                !cands[i].remat) {
                if (cands[i].U == 5)
                    saw_u5 = 1;
                if (cands[i].U == 8)
                    saw_u8 = 1;
            }
        }
        CHECK(saw_u5, "U=5 nibble (12 regs) should now be offered on SSE");
        CHECK(!saw_u8, "U=8 nibble (18 regs) must still be rejected");
        free(pl.mat);
    }
#endif

    if (failures == 0)
        printf("pressure: all OK\n");
    return failures != 0;
}
