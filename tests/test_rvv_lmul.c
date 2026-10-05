/* RVV LMUL tuning-axis tests.
 *
 * LMUL is a planner candidate axis on RVV (cand_t.lmul in {1,2,4,8}):
 * the backend maps logical vector register r to the lmul-aligned
 * physical group head r*lmul, emits vsetvli/vsetivli with the matching
 * vlmul field, and amortizes the per-byte cost model over lmul*NOMVL
 * bytes per instruction.
 *
 * Checks (each fails on the pre-LMUL codegen, which hardcodes m1):
 *  A. Enumeration: plan_candidates on a synthetic RVV CPU emits lmul in
 *     {1,2,4,8}; every RVV candidate satisfies
 *     plan_need_regs(...)*lmul <= 32; at least one lmul>1 candidate
 *     exists for P_XOR and P_HORNER-nibble plans.
 *  B. vsetvli decode: for lmul in {2,4,8}, every vsetvli/vsetivli word in
 *     the emitted code carries the matching vlmul field (m2/m4/m8), and
 *     none carries anything else.
 *  C. Group-head alignment: every vector register operand (vd/vs1/vs2/
 *     vs3) of every vector instruction in the emitted code is a multiple
 *     of lmul. A forgotten stride site produces a misaligned head and
 *     fails this.
 *  D. Execution: lmul in {1,2,4,8} kernels match the scalar reference at
 *     group-boundary lengths (P_XOR and P_HORNER nibble).
 *  E. desc marker: " lmulN" present iff lmul > 1.
 *  F. Cost model: uops_per_byte(lmul) == uops_per_byte(1)/lmul for the
 *     same plan+U (identical instruction mix, wider strip), and
 *     group_bytes == U*vlenb*lmul.
 *  G. Invalid lmul (3, 5, 6, 7) is rejected cleanly by the RVV backend.
 *  H. (x86_64/AArch64) lmul=2 is rejected cleanly by those backends.
 *
 * RVV-only for A-G (execution needs RVV; the shape scans need the
 * riscv64 build). On other hosts the binary builds and prints SKIP.
 * Run under QEMU at several VLENs, e.g.
 *   QEMU_LD_PREFIX=/usr/riscv64-linux-gnu qemu-riscv64 -cpu max \
 *       ./build-riscv64/test_rvv_lmul
 *   ... -cpu max,vlen=256 ... ; ... -cpu max,vlen=512 ...
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
#include <math.h>

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

#ifdef __riscv

static uint64_t rng_state = 0x9e3779b97f4a7c15ULL;
static uint64_t xrng(void)
{
    uint64_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return rng_state = x;
}

static uint8_t *abuf(size_t n)
{
    uint8_t *p = malloc(n ? n : 1);
    if (!p) {
        printf("FAIL: oom\n");
        exit(1);
    }
    return p;
}

static void randfill(uint8_t *p, size_t n)
{
    for (size_t i = 0; i < n; i++)
        p[i] = (uint8_t)xrng();
}

static uint32_t word_at(const void *p)
{
    uint32_t w;
    memcpy(&w, p, 4);
    return w;
}

/* vlenb CSR: VLEN in bytes (same hardware constant the backend reads). */
static size_t read_vlenb(void)
{
    size_t v;
    __asm__ volatile("csrr %0, vlenb" : "=r"(v));
    return v;
}

/* Build one RVV candidate directly, bypassing the planner's twin
 * policy (mirrors tests/test_rvv_f6.c). */
static int build_cand(const plan_t *pl, const rj_cpu_info *cpu, int U,
                      int lmul, mul_t mul, version_t *v, char *err,
                      size_t errlen)
{
    rj_options o;
    rj_options_default(&o);
    o.benchmark = 0;
    cand_t cand;
    memset(&cand, 0, sizeof(cand));
    cand.isa = ISA_RVV;
    cand.mul = mul;
    cand.U = U;
    cand.lmul = lmul;
    memset(v, 0, sizeof(*v));
    return jit_build_version(pl, &cand, cpu, &o, v, err, errlen);
}

static int is_vsetvli(uint32_t w)
{
    return (w & 0x7fu) == 0x57u && ((w >> 12) & 0x7u) == 0x7u;
}

/* B. every vsetvli/vsetivli carries the expected vlmul, and at least
 * one exists (the main-loop VLMAX pin). */
static void test_vsetvli_lmul(const plan_t *pl, const rj_cpu_info *cpu,
                              int U, int lmul, mul_t mul, const char *what)
{
    char err[256];
    version_t v;
    int want = lmul == 8 ? 3 : lmul == 4 ? 2 : lmul == 2 ? 1 : 0;
    if (build_cand(pl, cpu, U, lmul, mul, &v, err, sizeof(err)) != 0) {
        /* Register pressure may legitimately exclude wide lmul at
         * large U; lmul=1 must always build. */
        CHECK(lmul > 1, "%s U=%d lmul=%d: build failed: %s", what, U,
              lmul, err);
        return;
    }
    int n = 0;
    for (size_t i = 0; i + 4 <= v.code_size; i += 4) {
        uint32_t w = word_at((const char *)v.mem + i);
        if (!is_vsetvli(w))
            continue;
        n++;
        int vlmul = (int)((w >> 20) & 0x7u);
        CHECK(vlmul == want, "%s U=%d lmul=%d [%s]: vsetvli@%zu vlmul=%d, want %d",
              what, U, lmul, v.desc, i, vlmul, want);
    }
    CHECK(n > 0, "%s U=%d lmul=%d [%s]: no vsetvli found", what, U, lmul,
          v.desc);
    jit_free_version(&v);
}

/* C. every vector register operand is an lmul-aligned group head.
 * Decodes vd/vs1/vs2/vs3 for the vector opcodes the backend emits:
 *  0x07 loads (vd), 0x27 stores (vs3), 0x57 OPMVV/OPMVX/OPFVV/OPIVV/
 *  OPIVX (vd, vs1, vs2; OPIVI's [19:15] is an immediate, skipped). */
static void test_head_alignment(const plan_t *pl, const rj_cpu_info *cpu,
                                int U, int lmul, mul_t mul, const char *what)
{
    char err[256];
    version_t v;
    if (build_cand(pl, cpu, U, lmul, mul, &v, err, sizeof(err)) != 0) {
        CHECK(lmul > 1, "%s U=%d lmul=%d: build failed: %s", what, U,
              lmul, err);
        return;
    }
    int nvec = 0;
    for (size_t i = 0; i + 4 <= v.code_size; i += 4) {
        uint32_t w = word_at((const char *)v.mem + i);
        unsigned op = w & 0x7fu;
        int regs[4], nregs = 0;
        if (op == 0x07u && ((w >> 12) & 0x7u) == 0x0u) {
            /* vle8.v: vd only */
            regs[nregs++] = (int)((w >> 7) & 0x1fu);
        } else if (op == 0x27u) {
            /* vse8.v: vs3 only */
            regs[nregs++] = (int)((w >> 20) & 0x1fu);
        } else if (op == 0x57u) {
            unsigned f3 = (w >> 12) & 0x7u;
            if (f3 == 0x7u)
                continue; /* vsetvli/vsetivli: no vector operands */
            regs[nregs++] = (int)((w >> 7) & 0x1fu); /* vd */
            if (f3 != 0x6u)
                regs[nregs++] = (int)((w >> 15) & 0x1fu); /* vs1 */
            if (f3 <= 0x2u)
                regs[nregs++] = (int)((w >> 20) & 0x1fu); /* vs2 */
        } else {
            continue;
        }
        for (int k = 0; k < nregs; k++) {
            nvec++;
            CHECK(regs[k] % lmul == 0,
                  "%s U=%d lmul=%d [%s]: vec reg v%d at +%zu not %d-aligned",
                  what, U, lmul, v.desc, regs[k], i, lmul);
        }
    }
    CHECK(nvec > 0, "%s U=%d lmul=%d [%s]: no vector insns decoded", what,
          U, lmul, v.desc);
    jit_free_version(&v);
}

/* D. execution vs scalar reference at group-boundary lengths. */
static void test_exec(const plan_t *pl, const topo_int_t *ti,
                      const rj_cpu_info *cpu, int U, int lmul, mul_t mul,
                      const char *what)
{
    char err[256];
    version_t v;
    if (build_cand(pl, cpu, U, lmul, mul, &v, err, sizeof(err)) != 0) {
        CHECK(lmul > 1, "%s U=%d lmul=%d: build failed: %s", what, U,
              lmul, err);
        return;
    }
    /* E. desc marker iff lmul > 1. */
    char want[16];
    if (lmul > 1) {
        snprintf(want, sizeof(want), " lmul%d", lmul);
        CHECK(strstr(v.desc, want) != NULL,
              "%s U=%d lmul=%d: desc [%s] missing '%s'", what, U, lmul,
              v.desc, want);
    } else {
        CHECK(strstr(v.desc, " lmul") == NULL,
              "%s U=%d lmul=1: desc [%s] has spurious lmul marker", what,
              U, v.desc);
    }

    size_t g = v.group_bytes;
    size_t vlenb = read_vlenb();
    CHECK(g == (size_t)U * vlenb * (size_t)lmul,
          "%s U=%d lmul=%d: group_bytes=%zu != U*vlenb*lmul=%zu", what, U,
          lmul, g, (size_t)U * vlenb * (size_t)lmul);
    size_t lens[16];
    int nl = 0;
    size_t small[] = { 0, 1, 7, 15, 16, 17, 31, 100 };
    for (size_t k = 0; k < sizeof(small) / sizeof(small[0]); k++)
        lens[nl++] = small[k];
    size_t b[] = { g - 1, g, g + 1, 2 * g - 1, 2 * g, 2 * g + 1 };
    for (size_t k = 0; k < 6; k++)
        lens[nl++] = b[k];
    int nd = ti->ndata, np = ti->npar;
    for (int li = 0; li < nl && !failures; li++) {
        size_t len = lens[li];
        size_t alen = len ? len : 64;
        uint8_t *in[64], *out[64], *exp[64];
        for (int i = 0; i < nd; i++) {
            in[i] = abuf(alen);
            randfill(in[i], len);
        }
        for (int j = 0; j < np; j++) {
            out[j] = abuf(alen);
            exp[j] = abuf(alen);
            memset(out[j], 0x5A, alen);
        }
        v.fn((const void **)in, (void **)out, len);
        ref_run(pl, (const uint8_t **)in, exp, len);
        for (int j = 0; j < np; j++)
            CHECK(memcmp(out[j], exp[j], len) == 0,
                  "%s U=%d lmul=%d [%s] len=%zu parity%d mismatch", what,
                  U, lmul, v.desc, len, j);
        for (int i = 0; i < nd; i++)
            free(in[i]);
        for (int j = 0; j < np; j++) {
            free(out[j]);
            free(exp[j]);
        }
    }
    jit_free_version(&v);
}

/* F. cost model scales exactly with lmul for the same plan+U. */
static void test_cost_scaling(const plan_t *pl, const rj_cpu_info *cpu,
                              int U, mul_t mul, const char *what,
                              int exact_uops)
{
    char err[256];
    version_t v1, v2;
    CHECK(build_cand(pl, cpu, U, 1, mul, &v1, err, sizeof(err)) == 0,
          "%s U=%d lmul=1 build: %s", what, U, err);
    if (failures)
        return;
    if (build_cand(pl, cpu, U, 2, mul, &v2, err, sizeof(err)) != 0) {
        printf("note: %s U=%d lmul=2 does not fit (%s), skipping scale check\n",
               what, U, err);
        jit_free_version(&v1);
        return;
    }
    /* Identical instruction mix over a 2x-wider strip: uops_per_byte
     * halves exactly (up to fp rounding) on paths without
     * per-iteration table reloads; with reloads (mulacc/delta) the
     * vl-narrowing window adds bounded overhead, so the ratio sits
     * in (1,2]. mem_per_byte must stay ~constant: a vector
     * load/store moves lmul-x the bytes over lmul-x the stripe bytes
     * (the x86/aarch64 convention: each mem op counts its actual
     * width). Table reloads narrow vl to 16 and amortize over the
     * wider strip, so mem/byte can only drop. */
    double ratio = v1.uops_per_byte / v2.uops_per_byte;
    if (exact_uops) {
        CHECK(fabs(ratio - 2.0) < 1e-9,
              "%s U=%d [%s] vs [%s]: uops/byte ratio=%g, want 2", what, U,
              v1.desc, v2.desc, ratio);
    } else {
        CHECK(ratio > 1.0 + 1e-9 && ratio <= 2.0 + 1e-9,
              "%s U=%d [%s] vs [%s]: uops/byte ratio=%g, want in (1,2]",
              what, U, v1.desc, v2.desc, ratio);
    }
    double mratio = v1.mem_per_byte / v2.mem_per_byte;
    CHECK(mratio >= 1.0 - 1e-9 && mratio <= 2.0 + 1e-9,
          "%s U=%d [%s] vs [%s]: mem/byte ratio=%g, want in [1,2]", what,
          U, v1.desc, v2.desc, mratio);
    CHECK(v2.group_bytes == 2 * v1.group_bytes,
          "%s U=%d: group_bytes %zu vs %zu, want 2x", what, U,
          v2.group_bytes, v1.group_bytes);
    CHECK(v2.vl == 2 * v1.vl, "%s U=%d: vl %d vs %d, want 2x", what, U,
          v2.vl, v1.vl);
    jit_free_version(&v1);
    jit_free_version(&v2);
}

/* G. invalid lmul values are rejected cleanly. */
static void test_bad_lmul(const plan_t *pl, const rj_cpu_info *cpu)
{
    static const int bad[] = { 3, 5, 6, 7, 9, 16 };
    for (size_t k = 0; k < sizeof(bad) / sizeof(bad[0]); k++) {
        char err[256];
        version_t v;
        int rc = build_cand(pl, cpu, 1, bad[k], MUL_NIBBLE, &v, err,
                            sizeof(err));
        CHECK(rc != 0, "lmul=%d: build unexpectedly succeeded [%s]",
              bad[k], v.desc);
        if (rc == 0)
            jit_free_version(&v);
        else
            CHECK(strstr(err, "LMUL") != NULL,
                  "lmul=%d: error <%s> does not mention LMUL", bad[k],
                  err);
    }
}

/* A. planner enumeration over the lmul axis (synthetic RVV CPU; no
 * execution needed beyond what plan_candidates inspects). */
static void test_enumeration(void)
{
    rj_cpu_info cpu;
    memset(&cpu, 0, sizeof(cpu));
    cpu.rvv = 1;
    cpu.rvv_vlen = 128;

    /* P_XOR plan: raid6 4+1. */
    rj_topology *t = rj_topo_raid5(4);
    CHECK(t != NULL, "rj_topo_raid5(4)");
    if (!t)
        return;
    char err[256];
    topo_int_t ti;
    CHECK(topo_int_build(&ti, t, err, sizeof(err)) == 0, "topo: %s",
          err);
    if (failures) {
        rj_topo_free(t);
        return;
    }
    {
        rj_op op = rj_op_encode();
        plan_t pl;
        CHECK(plan_build(&ti, &op, "r64", &pl, err, sizeof(err)) == 0,
              "plan: %s", err);
        if (!failures) {
            cand_t out[128];
            int n = plan_candidates(&pl, &cpu, out, 128, 0, 0);
            CHECK(n > 0, "no RVV candidates enumerated");
            int seen[9];
            memset(seen, 0, sizeof(seen));
            for (int i = 0; i < n; i++) {
                CHECK(out[i].isa == ISA_RVV, "cand %d: not RVV", i);
                int lmul = out[i].lmul ? out[i].lmul : 1;
                CHECK(lmul == 1 || lmul == 2 || lmul == 4 || lmul == 8,
                      "cand %d: bad lmul=%d", i, lmul);
                if (lmul <= 8)
                    seen[lmul] = 1;
                int need = plan_need_regs(ISA_RVV, out[i].mul, &pl,
                                          out[i].U, out[i].remat) *
                           lmul;
                CHECK(need <= 32, "cand %d: U=%d lmul=%d needs %d regs",
                      i, out[i].U, lmul, need);
            }
            CHECK(seen[2] || seen[4] || seen[8],
                  "no lmul>1 candidate enumerated for P_XOR");
            printf("note: P_XOR RVV candidates: %d (lmul1=%d lmul2=%d lmul4=%d lmul8=%d)\n",
                   n, seen[1], seen[2], seen[4], seen[8]);
        }
        plan_free(&pl);
    }
    topo_int_free(&ti);
    rj_topo_free(t);

    /* P_HORNER nibble plan: rs(6,3). */
    t = rj_topo_rs(6, 3);
    CHECK(t != NULL, "rj_topo_rs(6,3)");
    if (!t)
        return;
    CHECK(topo_int_build(&ti, t, err, sizeof(err)) == 0, "topo: %s",
          err);
    if (failures) {
        rj_topo_free(t);
        return;
    }
    {
        rj_op op = rj_op_encode();
        plan_t pl;
        CHECK(plan_build(&ti, &op, "rs63", &pl, err, sizeof(err)) == 0,
              "plan: %s", err);
        if (!failures) {
            cand_t out[128];
            int n = plan_candidates(&pl, &cpu, out, 128, 0, 0);
            int seen_lmul = 0;
            for (int i = 0; i < n; i++) {
                int lmul = out[i].lmul ? out[i].lmul : 1;
                if (lmul > 1)
                    seen_lmul = 1;
                int need = plan_need_regs(ISA_RVV, out[i].mul, &pl,
                                          out[i].U, out[i].remat) *
                           lmul;
                CHECK(need <= 32, "cand %d: U=%d lmul=%d needs %d regs",
                      i, out[i].U, lmul, need);
            }
            CHECK(seen_lmul,
                  "no lmul>1 candidate enumerated for P_HORNER");
        }
        plan_free(&pl);
    }
    topo_int_free(&ti);
    rj_topo_free(t);
}

#endif /* __riscv */

#if defined(__x86_64__) || defined(__aarch64__)
/* H. lmul>1 is rejected cleanly on non-RVV backends. */
static void test_lmul_rejected_off_rvv(void)
{
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    rj_topology *t = rj_topo_raid5(4);
    if (!t) {
        printf("FAIL: rj_topo_raid5(4)\n");
        failures++;
        return;
    }
    char err[256];
    topo_int_t ti;
    if (topo_int_build(&ti, t, err, sizeof(err)) != 0) {
        printf("FAIL: topo: %s\n", err);
        failures++;
        rj_topo_free(t);
        return;
    }
    rj_op op = rj_op_encode();
    plan_t pl;
    if (plan_build(&ti, &op, "r64", &pl, err, sizeof(err)) != 0) {
        printf("FAIL: plan: %s\n", err);
        failures++;
    } else {
        rj_options o;
        rj_options_default(&o);
        o.benchmark = 0;
        cand_t cand;
        memset(&cand, 0, sizeof(cand));
#ifdef __x86_64__
        cand.isa = ISA_AVX2;
#else
        cand.isa = ISA_NEON;
#endif
        cand.U = 2;
        cand.lmul = 2;
        version_t v;
        memset(&v, 0, sizeof(v));
        int rc = jit_build_version(&pl, &cand, &cpu, &o, &v, err,
                                   sizeof(err));
        CHECK(rc != 0, "lmul=2 unexpectedly built on non-RVV backend");
        if (rc != 0)
            CHECK(strstr(err, "lmul") != NULL || strstr(err, "LMUL") != NULL,
                  "error <%s> does not mention lmul", err);
        else
            jit_free_version(&v);
        plan_free(&pl);
    }
    topo_int_free(&ti);
    rj_topo_free(t);
}
#endif

int main(void)
{
#ifdef __riscv
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    if (!cpu.rvv) {
        printf("test_rvv_lmul: SKIP (no RVV)\n");
        return failures != 0;
    }
    test_enumeration();
    if (failures)
        return 1;

    /* P_XOR plan for shape/exec: raid5 4+1. */
    rj_topology *tx = rj_topo_raid5(4);
    CHECK(tx != NULL, "rj_topo_raid5(4)");
    /* P_HORNER nibble plan: rs(6,3). */
    rj_topology *th = rj_topo_rs(6, 3);
    CHECK(th != NULL, "rj_topo_rs(6,3)");
    /* P_MULACC nibble plan: non-Vandermonde 2x4 custom matrix (row 1:
     * 2,3,4 are nontrivial, so plan_build picks P_MULACC). Covers the
     * mulacc_rows_c body, which the XOR/Horner plans never execute. */
    static const uint8_t cm[8] = { 1, 1, 1, 1, 1, 2, 3, 4 };
    rj_topology *tm = rj_topo_custom(4, 2, cm);
    CHECK(tm != NULL, "rj_topo_custom(4,2)");
    if (failures)
        return 1;
    char err[256];
    topo_int_t tix, tih, tim;
    CHECK(topo_int_build(&tix, tx, err, sizeof(err)) == 0, "topo x: %s",
          err);
    CHECK(topo_int_build(&tih, th, err, sizeof(err)) == 0, "topo h: %s",
          err);
    CHECK(topo_int_build(&tim, tm, err, sizeof(err)) == 0, "topo m: %s",
          err);
    if (failures)
        return 1;
    {
        rj_op opx = rj_op_encode();
        plan_t plx;
        CHECK(plan_build(&tix, &opx, "r64", &plx, err, sizeof(err)) == 0,
              "plan x: %s", err);
        rj_op oph = rj_op_encode();
        plan_t plh;
        CHECK(plan_build(&tih, &oph, "rs63", &plh, err, sizeof(err)) == 0,
              "plan h: %s", err);
        rj_op opm = rj_op_encode();
        plan_t plm;
        CHECK(plan_build(&tim, &opm, "cm42", &plm, err, sizeof(err)) == 0,
              "plan m: %s", err);
        CHECK(plm.kind == P_MULACC, "custom matrix: kind=%d, want P_MULACC",
              plm.kind);
        if (!failures) {
            static const int lmuls[] = { 1, 2, 4, 8 };
            for (size_t li = 0; li < 4 && !failures; li++) {
                int lmul = lmuls[li];
                /* U=1 keeps every lmul within 32 physical regs. */
                test_vsetvli_lmul(&plx, &cpu, 1, lmul, MUL_NIBBLE,
                                  "xor");
                test_head_alignment(&plx, &cpu, 1, lmul, MUL_NIBBLE,
                                    "xor");
                test_exec(&plx, &tix, &cpu, 1, lmul, MUL_NIBBLE,
                          "xor");
                test_vsetvli_lmul(&plh, &cpu, 1, lmul, MUL_NIBBLE,
                                  "horner");
                test_head_alignment(&plh, &cpu, 1, lmul, MUL_NIBBLE,
                                    "horner");
                test_exec(&plh, &tih, &cpu, 1, lmul, MUL_NIBBLE,
                          "horner");
                test_vsetvli_lmul(&plm, &cpu, 1, lmul, MUL_NIBBLE,
                                  "mulacc");
                test_head_alignment(&plm, &cpu, 1, lmul, MUL_NIBBLE,
                                    "mulacc");
                test_exec(&plm, &tim, &cpu, 1, lmul, MUL_NIBBLE,
                          "mulacc");
            }
            if (!failures) {
                test_cost_scaling(&plx, &cpu, 1, MUL_NIBBLE, "xor", 1);
                test_cost_scaling(&plh, &cpu, 1, MUL_NIBBLE, "horner", 1);
                test_cost_scaling(&plm, &cpu, 1, MUL_NIBBLE, "mulacc", 0);
            }
            if (!failures)
                test_bad_lmul(&plx, &cpu);
        }
        plan_free(&plx);
        plan_free(&plh);
        plan_free(&plm);
    }
    topo_int_free(&tix);
    topo_int_free(&tih);
    topo_int_free(&tim);
    rj_topo_free(tx);
    rj_topo_free(th);
    rj_topo_free(tm);
#endif
#if defined(__x86_64__) || defined(__aarch64__)
    test_lmul_rejected_off_rvv();
#endif
    if (failures)
        printf("RVLMUL TESTS FAILED (%d failures)\n", failures);
    else
        printf("rvlmul: all OK\n");
    return failures != 0;
}
