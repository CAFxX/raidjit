/* XOR-schedule / cross-row CSE tests (track A3).
 *
 * Differential: scheduled vs naive vs scalar reference across topologies
 * x lengths x ISAs x U. Model: the scheduled twin must strictly beat the
 * naive twin in uops_per_byte, and xorsched_version_uops must predict both
 * versions' uops exactly. Twin enumeration via plan_candidates.
 *
 * Portable: runs natively on x86-64 and under QEMU for AArch64/RVV.
 * Buffers are 64-byte aligned (the JIT uses aligned vector loads).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "jit.h"
#include "plan.h"
#include "xorsched.h"
#include "raidjit.h"

static int fails;
#define CHECK(c, ...) do {                                              \
    if (!(c)) {                                                        \
        printf("FAIL "); printf(__VA_ARGS__); printf("\n"); fails++;   \
    }                                                                  \
} while (0)

static uint64_t xrng_state = 0x123456789ABCDEFULL;
static uint64_t xrng(void)
{
    uint64_t x = xrng_state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return xrng_state = x;
}

static void *xalloc(size_t n)
{
    void *p;
    if (posix_memalign(&p, 64, n ? n : 64)) {
        printf("out of memory\n");
        exit(1);
    }
    return p;
}

/* Build a P_XOR plan from a row-major matrix, with the schedule attached
 * (as plan_build does). */
static void make_plan(plan_t *pl, int n_in, int n_out, const uint8_t *mat,
                      const char *name)
{
    memset(pl, 0, sizeof(*pl));
    pl->kind = P_XOR;
    pl->n_in = n_in;
    pl->n_out = n_out;
    pl->mat = malloc((size_t)n_in * (size_t)n_out);
    memcpy(pl->mat, mat, (size_t)n_in * (size_t)n_out);
    snprintf(pl->name, sizeof(pl->name), "%s", name);
    pl->xor_sched = xorsched_build(pl);
}

static void scalar_ref(const plan_t *pl, uint8_t **in, uint8_t **out,
                       size_t len)
{
    for (int j = 0; j < pl->n_out; j++) {
        memset(out[j], 0, len);
        for (int i = 0; i < pl->n_in; i++) {
            if (!pl->mat[(size_t)j * (size_t)pl->n_in + (size_t)i])
                continue;
            for (size_t k = 0; k < len; k++)
                out[j][k] ^= in[i][k];
        }
    }
}

/* check_one outcome buckets (D1: the headline counter used to conflate
 * these; report them separately so "checks" means iterations that
 * actually ran the differential/model assertions). */
static int n_full;    /* ran the model-exactness + differential assertions */
static int n_nosched; /* schedule NULL: nothing to compare */
static int n_nofit;   /* register fit failed: twin not emittable */

static const char *isa_str(isa_t isa)
{
    switch (isa) {
    case ISA_SSE: return "sse";
    case ISA_AVX2: return "avx2";
    case ISA_AVX512: return "avx512";
    case ISA_NEON: return "neon";
    case ISA_SVE: return "sve";
    case ISA_RVV: return "rvv";
    default: return "?";
    }
}

/* Test one (isa, U, topology, len): build naive + scheduled versions,
 * check the model, run the differential. */
static void check_one(const rj_cpu_info *cpu, isa_t isa, int U,
                      const plan_t *pl, size_t len, const char *tname)
{
    const xorsched_t *xs = pl->xor_sched;
    rj_options opts;
    memset(&opts, 0, sizeof(opts));
    char err[256];
    version_t vn, vs;
    cand_t cn;
    memset(&cn, 0, sizeof(cn));
    cn.isa = isa;
    cn.mul = MUL_NIBBLE;
    cn.U = U;

    /* register fit (mirrors plan_candidates filtering) */
    if (plan_need_regs(isa, MUL_NIBBLE, pl, U, 0, 0, 0) > isa_nregs(isa)) {
        n_nofit++;
        return;
    }
    if (jit_build_version(pl, &cn, cpu, &opts, &vn, err, sizeof(err)) != 0) {
        CHECK(0, "%s %s U=%d len=%zu: naive build failed: %s", tname,
              isa_str(isa), U, len, err);
        return;
    }
    if (!xs) {
        /* no schedule: nothing more to check for this topology */
        jit_free_version(&vn);
        n_nosched++;
        return;
    }
    if (xorsched_need_regs(isa, pl, U) > isa_nregs(isa)) {
        jit_free_version(&vn);
        n_nofit++;
        return;
    }
    cand_t cs = cn;
    cs.xorsched = 1;
    if (jit_build_version(pl, &cs, cpu, &opts, &vs, err, sizeof(err)) != 0) {
        CHECK(0, "%s %s U=%d len=%zu: sched build failed: %s", tname,
              isa_str(isa), U, len, err);
        jit_free_version(&vn);
        return;
    }

    /* model honesty: exact prediction always; strict win iff gate passes.
     * The divisor converts uops_per_byte back to total uops: it must be
     * the same byte base the backend used for uops_per_byte. For x86,
     * AArch64 (incl. SVE), that base is cg.group, reported as the
     * version's group_bytes. For RVV the backend normalizes
     * uops_per_byte to a NOMINAL 16-byte VL (src/riscv/jit.c: NOMVL),
     * while group_bytes carries the actual VLEN for the length
     * dispatcher — so RVV must use U*16 here, not group_bytes.
     * (For SVE, group_bytes is the true VL via the cntb hunk; a
     * prctl-derived VL would be 8x wrong under qemu-user, see
     * test_boundaries.c:sve_vl_bytes.) */
    long divn = (isa == ISA_RVV) ? (long)U * 16 : (long)vn.group_bytes;
    long divs = (isa == ISA_RVV) ? (long)U * 16 : (long)vs.group_bytes;
    long pn = xorsched_version_uops(xs, pl, isa, U, 0, 0, 0);
    long ps = xorsched_version_uops(xs, pl, isa, U, 1, 0, 0);
    long en = (long)(vn.uops_per_byte * (double)divn + 0.5);
    long es = (long)(vs.uops_per_byte * (double)divs + 0.5);
    CHECK(en == pn, "%s %s U=%d: naive model %ld != emitter %ld", tname,
          isa_str(isa), U, pn, en);
    CHECK(es == ps, "%s %s U=%d: sched model %ld != emitter %ld", tname,
          isa_str(isa), U, ps, es);
    int gate = (ps < pn);
    if (gate) {
        CHECK(vs.uops_per_byte < vn.uops_per_byte,
              "%s %s U=%d: sched not strictly better (%.4f vs %.4f)", tname,
              isa_str(isa), U, vs.uops_per_byte, vn.uops_per_byte);
    }

    /* differential vs scalar reference */
    uint8_t *in[256], *outn[32], *outs[32], *ref[32];
    for (int i = 0; i < pl->n_in; i++) {
        in[i] = xalloc(len);
        for (size_t k = 0; k < len; k++)
            in[i][k] = (uint8_t)xrng();
    }
    for (int j = 0; j < pl->n_out; j++) {
        outn[j] = xalloc(len);
        outs[j] = xalloc(len);
        ref[j] = xalloc(len);
        memset(outn[j], 0xA5, len);
        memset(outs[j], 0xA5, len);
    }
    scalar_ref(pl, in, ref, len);
    vn.fn((const void **)in, (void **)outn, len);
    vs.fn((const void **)in, (void **)outs, len);
    for (int j = 0; j < pl->n_out; j++) {
        if (memcmp(outn[j], ref[j], len) != 0)
            CHECK(0, "%s %s U=%d len=%zu: naive mismatch row %d", tname,
                  isa_str(isa), U, len, j);
        if (memcmp(outs[j], ref[j], len) != 0)
            CHECK(0, "%s %s U=%d len=%zu: sched mismatch row %d", tname,
                  isa_str(isa), U, len, j);
    }
    for (int i = 0; i < pl->n_in; i++)
        free(in[i]);
    for (int j = 0; j < pl->n_out; j++) {
        free(outn[j]);
        free(outs[j]);
        free(ref[j]);
    }
    n_full++;
    jit_free_version(&vn);
    jit_free_version(&vs);
}

/* ---- topologies ---- */

/* T1: sharing-heavy 4x20 (smoke topology) */
static const uint8_t t1_mat[80] = {
    1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,
    1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,
    0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,
};

/* T2: duplicate rows + sharing (3x6, rows 0 and 2 identical; pair (0,1)
 * shared across the unique rows) */
static const uint8_t t2_mat[18] = {
    1,1,1,1,0,0,
    1,1,0,0,1,1,
    1,1,1,1,0,0,
};

/* T3: dead row (3x6, row 1 all zero) */
static const uint8_t t3_mat[18] = {
    1,1,0,0,1,0,
    0,0,0,0,0,0,
    1,1,1,1,0,0,
};

/* T4: single row (no schedule expected) */
static const uint8_t t4_mat[8] = {1,1,0,1,0,1,0,0};

/* T5: no sharing (4x8, every pair appears once) */
static const uint8_t t5_mat[32] = {
    1,1,0,0,0,0,0,0,
    0,0,1,1,0,0,0,0,
    0,0,0,0,1,1,0,0,
    0,0,0,0,0,0,1,1,
};

/* T6: dense 6x12 with overlapping pairs */
static const uint8_t t6_mat[72] = {
    1,1,1,1,0,0,0,0,0,0,0,0,
    1,1,0,0,1,1,0,0,0,0,0,0,
    0,0,1,1,1,1,0,0,0,0,0,0,
    0,0,0,0,0,0,1,1,1,1,0,0,
    0,0,0,0,0,0,1,1,0,0,1,1,
    1,0,1,0,1,0,1,0,1,0,1,0,
};

static const size_t lens[] = {1, 7, 15, 16, 17, 31, 32, 33, 64, 65, 100,
                              1000, 4096};
static const int Us[] = {1, 2, 4};

int main(void)
{
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);

    isa_t isas[4];
    int nis = 0;
#if defined(__x86_64__)
    if (cpu.avx512f)
        isas[nis++] = ISA_AVX512;
    if (cpu.avx2)
        isas[nis++] = ISA_AVX2;
    if (cpu.sse2)
        isas[nis++] = ISA_SSE;
#elif defined(__aarch64__)
    if (cpu.sve && cpu.sve_vl > 0)
        isas[nis++] = ISA_SVE;
    if (cpu.neon)
        isas[nis++] = ISA_NEON;
#elif defined(__riscv)
    if (cpu.rvv)
        isas[nis++] = ISA_RVV;
#endif
    if (nis == 0) {
        printf("no ISA available, skipping\n");
        return 0;
    }

    /* X1 regression: RVV full-version model must count exactly 2 body
     * emissions + 2 vsetvli, not the stale 3x + 3. Runs first so it
     * fires even when the exact-equality checks below fail. T2 (3x6,
     * rows 0/2 identical) at RVV U=1, hand-derived:
     *   naive body: input users 3,3,2,2,1,1 (inputs 4,5 not first in
     *     their row) => bU = (1+3)+(1+3)+(1+2)+(1+2)+2+2 = 18, stU = 3.
     *     full = (18+3)+(18+3)+2 = 44, notail = (18+3)+1 = 22.
     *   sched: 1 temp (0,1) with 2 users (3 + 2 = 5); surviving inputs
     *     2,3,4,5 single-user non-first (2 each = 8); 1 live duplicate.
     *     bS = 5+8+1 = 14, stU = 3.
     *     full = (14+3)+(14+3)+2 = 36, notail = (14+3)+1 = 18.
     * The stale model reported 66/54 (exactly the 1.5x over-count seen
     * in the 312 pre-fix failures); the emitter emits 44/36. */
    {
        plan_t pl;
        memset(&pl, 0, sizeof(pl));
        pl.kind = P_XOR;
        pl.n_in = 6;
        pl.n_out = 3;
        pl.mat = malloc(sizeof(t2_mat));
        memcpy(pl.mat, t2_mat, sizeof(t2_mat));
        pl.xor_sched = xorsched_build(&pl);
        CHECK(pl.xor_sched != NULL, "X1: t2 schedule build failed");
        CHECK(pl.xor_sched->ntemps == 1 && pl.xor_sched->ndup_live == 1,
              "X1: t2 schedule shape changed (ntemps=%d ndup=%d)",
              pl.xor_sched->ntemps, pl.xor_sched->ndup_live);
        long nfull = xorsched_version_uops(pl.xor_sched, &pl, ISA_RVV,
                                           1, 0, 0, 0);
        long nnotail = xorsched_version_uops(pl.xor_sched, &pl, ISA_RVV,
                                             1, 0, 0, 1);
        long sfull = xorsched_version_uops(pl.xor_sched, &pl, ISA_RVV,
                                           1, 1, 0, 0);
        long snotail = xorsched_version_uops(pl.xor_sched, &pl, ISA_RVV,
                                             1, 1, 0, 1);
        /* RVV remainder loop is steady-state-excluded (like AArch64's
         * predicated tail): full == notail in counted uops. The notail
         * twin's benefit is code size (no remainder loop emitted), not
         * steady-state uops. */
        CHECK(nfull == 22 && nnotail == 22,
              "X1: naive RVV model %ld/%ld, want 22/22", nfull, nnotail);
        CHECK(sfull == 18 && snotail == 18,
              "X1: sched RVV model %ld/%ld, want 18/18", sfull, snotail);
        CHECK(nfull - nnotail == 0 && sfull - snotail == 0,
              "X1: full/notail delta wrong (%ld/%ld)",
              nfull - nnotail, sfull - snotail);
        xorsched_free(pl.xor_sched);
        free(pl.mat);
    }

    struct {
        const char *name;
        int n_in, n_out;
        const uint8_t *mat;
        int expect_sched; /* 1 if xorsched_build returns non-NULL */
    } topos[] = {
        {"t1-share", 20, 4, t1_mat, 1},
        {"t2-dup", 6, 3, t2_mat, 1},
        {"t3-dead", 6, 3, t3_mat, 1},
        {"t4-single", 8, 1, t4_mat, 0},
        {"t5-noshr", 8, 4, t5_mat, 0},
        {"t6-dense", 12, 6, t6_mat, 1},
    };

    int nchecks = 0;
    int ntwins = 0;
    int ntwins_nt = 0, ntwins_pf = 0;
    for (size_t ti = 0; ti < sizeof(topos) / sizeof(topos[0]); ti++) {
        plan_t pl;
        make_plan(&pl, topos[ti].n_in, topos[ti].n_out, topos[ti].mat,
                  topos[ti].name);
        CHECK((pl.xor_sched != NULL) == (topos[ti].expect_sched != 0),
              "%s: schedule %s, expected %s", topos[ti].name,
              pl.xor_sched ? "built" : "NULL",
              topos[ti].expect_sched ? "built" : "NULL");
        if (pl.xor_sched)
            printf("%s: %d temps, %d unique rows\n", topos[ti].name,
                   pl.xor_sched->ntemps, pl.xor_sched->nunique);
        for (int ii = 0; ii < nis; ii++) {
            for (size_t ui = 0; ui < sizeof(Us) / sizeof(Us[0]); ui++) {
                for (size_t li = 0;
                     li < sizeof(lens) / sizeof(lens[0]); li++) {
                    check_one(&cpu, isas[ii], Us[ui], &pl, lens[li],
                              topos[ti].name);
                    nchecks++;
                }
            }
        }
        /* twin enumeration through plan_candidates: a twin appears iff
         * the gate passes for that (isa,U); never with ternlog. */
        {
            cand_t out[64];
            int n = plan_candidates(&pl, &cpu, out, 64, 0, 0);
            int found = 0;
            for (int i = 0; i < n; i++) {
                if (out[i].xorsched) {
                    found = 1;
                    /* the twin must satisfy the gate */
                    long pn = xorsched_version_uops(pl.xor_sched, &pl,
                                                    out[i].isa, out[i].U, 0,
                                                    0, 0);
                    long ps = xorsched_version_uops(pl.xor_sched, &pl,
                                                    out[i].isa, out[i].U, 1,
                                                    0, 0);
                    CHECK(ps < pn,
                          "%s: enumerated twin fails gate (%ld >= %ld)",
                          topos[ti].name, ps, pn);
                    CHECK(!out[i].ternlog,
                          "%s: xorsched twin combined with ternlog",
                          topos[ti].name);
                }
            }
            /* every enumerated twin must pass the gate (checked above);
             * absence of a twin just means the gate passed nowhere. */
            ntwins += found;
        }
        /* m2: the NT/prefetch twin loops must twin the xsched twins too.
         * Enumerate with (nt=1, pf=1) and require >= 1 passing
         * xsched+nt and xsched+pf twin wherever a plain xsched twin
         * was enumerated. Synthetic Zicbop: the pf loop skips RVV
         * without it by design; the enumeration logic under test is
         * ISA-independent, so arm the bit like the combo section does. */
        if (pl.xor_sched) {
            rj_cpu_info pfcpu = cpu;
            pfcpu.rvv_zicbop = 1;
            cand_t out2[256];
            int n2 = plan_candidates(&pl, &pfcpu, out2, 256, 1, 512);
            CHECK(n2 < 256, "%s: candidate buffer truncated (%d)",
                  topos[ti].name, n2);
            int found_nt = 0, found_pf = 0, base_twinnable = 0;
            int nt_all_skipped = 1;
            for (int i = 0; i < n2; i++) {
                if (!out2[i].xorsched)
                    continue;
                long pn = xorsched_version_uops(pl.xor_sched, &pl,
                                                out2[i].isa, out2[i].U, 0, 0, 0);
                long ps = xorsched_version_uops(pl.xor_sched, &pl,
                                                out2[i].isa, out2[i].U, 1, 0, 0);
                CHECK(ps < pn,
                      "%s: enumerated twin fails gate (%ld >= %ld)",
                      topos[ti].name, ps, pn);
                CHECK(!out2[i].ternlog,
                      "%s: xorsched twin combined with ternlog",
                      topos[ti].name);
                if (out2[i].nt)
                    found_nt = 1;
                if (out2[i].prefetch_dist)
                    found_pf = 1;
                /* base (non-nt, non-pf) xsched twin: the pf loop twins
                 * every candidate unconditionally (under synthetic
                 * Zicbop), so each such twin implies a +pf twin; the
                 * nt loop skips only (NEON, U=1) as byte-identical. */
                if (!out2[i].nt && !out2[i].prefetch_dist) {
                    base_twinnable++;
                    if (!(out2[i].isa == ISA_NEON && out2[i].U == 1))
                        nt_all_skipped = 0;
                }
            }
            if (base_twinnable > 0) {
                CHECK(found_pf, "%s: no xsched+pf twin enumerated",
                      topos[ti].name);
                if (!nt_all_skipped)
                    CHECK(found_nt, "%s: no xsched+nt twin enumerated",
                          topos[ti].name);
            }
            ntwins_nt += found_nt;
            ntwins_pf += found_pf;
        }
        xorsched_free(pl.xor_sched);
        free(pl.mat);
    }
    CHECK(ntwins_nt > 0, "no xorsched+nt twin enumerated for any topology");
    CHECK(ntwins_pf > 0, "no xorsched+pf twin enumerated for any topology");
    CHECK(ntwins > 0, "no xorsched twin enumerated for any topology");
    printf("%d iterations (%d full differential/model checks, %d "
           "no-schedule skips, %d reg-fit skips), %d twins (%d +nt, %d "
           "+pf), %d failures\n",
           nchecks, n_full, n_nosched, n_nofit, ntwins, ntwins_nt,
           ntwins_pf, fails);
    if (fails)
        return 1;

    /* Combination: xorsched twin with NT stores and prefetch (interop). */
    {
        plan_t pl;
        make_plan(&pl, 20, 4, t1_mat, "t1-combo");
        if (!pl.xor_sched) {
            printf("FAIL t1-combo: no schedule\n");
            return 1;
        }
        rj_options opts;
        memset(&opts, 0, sizeof(opts));
        char err[256];
        /* Use AVX2 U=2 if available, else first available ISA. */
        isa_t isa = isas[0];
        int U = 2;
        if (plan_need_regs(isa, MUL_NIBBLE, &pl, U, 0, 0, 0) > isa_nregs(isa))
            U = 1;
        size_t len = 1000;
        uint8_t *in[20], *out[4], *ref[4];
        for (int i = 0; i < 20; i++) {
            in[i] = xalloc(len);
            for (size_t k = 0; k < len; k++)
                in[i][k] = (uint8_t)xrng();
        }
        for (int j = 0; j < 4; j++) {
            out[j] = xalloc(len);
            ref[j] = xalloc(len);
        }
        scalar_ref(&pl, in, ref, len);
        /* nt twin */
        {
            cand_t c;
            memset(&c, 0, sizeof(c));
            c.isa = isa; c.mul = MUL_NIBBLE; c.U = U; c.xorsched = 1; c.nt = 1;
            if (xorsched_need_regs(isa, &pl, U) > isa_nregs(isa)) {
                printf("skip nt combo (regs)\n");
            } else {
                version_t v;
                if (jit_build_version(&pl, &c, &cpu, &opts, &v, err,
                                      sizeof(err)) != 0) {
                    printf("FAIL combo nt build: %s\n", err);
                    return 1;
                }
                CHECK(strstr(v.desc, "+xsched") && strstr(v.desc, "+nt"),
                      "combo nt desc missing tags: %s", v.desc);
                /* m1: the combo must be model-exact too, not just
                 * correctly tagged: NT stores count 1 uop like temporal
                 * stores, so the emitter must match the model exactly. */
                {
                    /* Divisor must match the backend's uops_per_byte
                     * base: U*NOMVL(16) for RVV, group_bytes otherwise;
                     * see the model-honesty note in check_one. */
                    long div = (isa == ISA_RVV) ? (long)U * 16
                                                : (long)v.group_bytes;
                    long eu = (long)(v.uops_per_byte * (double)div + 0.5);
                    long pu = xorsched_version_uops(pl.xor_sched, &pl,
                                                    isa, U, 1, 1, 0);
                    CHECK(eu == pu,
                          "combo nt model %ld != emitter %ld (%s U=%d)",
                          pu, eu, isa_str(isa), U);
                }
                for (int j = 0; j < 4; j++)
                    memset(out[j], 0xA5, len);
                v.fn((const void **)in, (void **)out, len);
                for (int j = 0; j < 4; j++)
                    CHECK(memcmp(out[j], ref[j], len) == 0,
                          "combo nt mismatch row %d", j);
                jit_free_version(&v);
                printf("combo nt OK (%s)\n", isa_str(isa));
            }
        }
        /* prefetch twin */
        {
            cand_t c;
            memset(&c, 0, sizeof(c));
            c.isa = isa; c.mul = MUL_NIBBLE; c.U = U; c.xorsched = 1;
            c.prefetch_dist = 512;
            /* Synthetic Zicbop on RVV: QEMU user-mode does not report
             * multi-letter extensions (the guest sees the host's
             * non-RISC-V cpuinfo, so rj_cpu_detect reports
             * rvv_zicbop=0 and the backend degrades to temporal), but
             * qemu's -cpu max executes the prefetch hints. Mirrors the
             * synthetic-zicbop pattern in test_riscv_prefetch.c. */
            rj_cpu_info pfcpu = cpu;
            if (isa == ISA_RVV)
                pfcpu.rvv_zicbop = 1;
            version_t v;
            if (jit_build_version(&pl, &c, &pfcpu, &opts, &v, err,
                                  sizeof(err)) != 0) {
                printf("FAIL combo pf build: %s\n", err);
                return 1;
            }
            CHECK(strstr(v.desc, "+xsched") && strstr(v.desc, "+pf"),
                  "combo pf desc missing tags: %s", v.desc);
            /* m1: model-exactness on the pf combo. The model excludes
             * prefetch uops (see xorsched.h); the emitter counts 1 per
             * input stream per group emission with exactly-once dedup
             * (pf_done in all three backends; RVV emits the body 2x --
             * counted loop + remainder). */
            {
                const xorsched_t *xs = pl.xor_sched;
                int seen[256] = { 0 };
                for (int k = 0; k < xs->ntemps; k++) {
                    seen[xs->temps[k].a] = 1;
                    seen[xs->temps[k].b] = 1;
                }
                for (int i = 0; i < xs->n_in; i++)
                    if (xs->in_nusers[i])
                        seen[i] = 1;
                int pf_streams = 0;
                for (int i = 0; i < xs->n_in; i++)
                    pf_streams += seen[i];
                long pf_uops =
                    (isa == ISA_RVV) ? 1L * pf_streams : pf_streams;
                /* RVV: prefetch emitted once per stream in the counted
                 * main body; the remainder loop is steady-state-excluded
                 * (its prefetch is not counted). */
                long div = (isa == ISA_RVV) ? (long)U * 16
                                            : (long)v.group_bytes;
                long eu = (long)(v.uops_per_byte * (double)div + 0.5);
                long pu = xorsched_version_uops(xs, &pl, isa, U, 1, 0, 0);
                CHECK(eu == pu + pf_uops,
                      "combo pf model+pref %ld != emitter %ld (%s U=%d)",
                      pu + pf_uops, eu, isa_str(isa), U);
            }
            for (int j = 0; j < 4; j++)
                memset(out[j], 0xA5, len);
            v.fn((const void **)in, (void **)out, len);
            for (int j = 0; j < 4; j++)
                CHECK(memcmp(out[j], ref[j], len) == 0,
                      "combo pf mismatch row %d", j);
            jit_free_version(&v);
            printf("combo pf OK (%s)\n", isa_str(isa));
        }
        for (int i = 0; i < 20; i++)
            free(in[i]);
        for (int j = 0; j < 4; j++) {
            free(out[j]);
            free(ref[j]);
        }
        xorsched_free(pl.xor_sched);
        free(pl.mat);
    }
    printf("combo checks done, %d failures\n", fails);

    /* M1 regression: fail every xorsched_build allocation site in turn.
     * Each must unwind to NULL (never a corrupt partial schedule, never
     * a crash); the ASan run additionally leak-checks the unwind. */
    {
        plan_t pl;
        memset(&pl, 0, sizeof(pl));
        pl.kind = P_XOR;
        pl.n_in = 20;
        pl.n_out = 4;
        pl.mat = malloc(20 * 4);
        memcpy(pl.mat, t1_mat, 20 * 4);
        int n;
        for (n = 0; n < 100000; n++) {
            xorsched_test_oom_at(n);
            xorsched_t *s = xorsched_build(&pl);
            xorsched_test_oom_at(-1);
            if (s != NULL) {
                /* site n lies past the last allocation site: the
                 * build consumed fewer than n allocations. */
                xorsched_free(s);
                break;
            }
            /* NULL: the OOM unwind path ran; xorsched_free(NULL) and
             * the emitter's NULL checks make this safe downstream. */
        }
        CHECK(n < 100000, "oom sweep did not terminate");
        /* hook disarmed: the build must succeed normally again */
        {
            xorsched_t *s = xorsched_build(&pl);
            CHECK(s != NULL && s->ntemps == 2,
                  "oom sweep: normal build broken (s=%p)", (void *)s);
            xorsched_free(s);
        }
        printf("oom sweep: %d allocation sites, all unwind to NULL\n", n);
        free(pl.mat);
    }
    return fails != 0;
}
