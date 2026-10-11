/* RVV software prefetch (Zicbop) and NT-hint (Zihintntl) codegen tests.
 *
 * The planner enumerates two new RVV twin families (see src/plan.c):
 *   - NT twins (cand.nt=1): every output vse8.v is preceded by the
 *     Zihintntl NTL.ALL hint (0x00500033, an ADD x0,x0,x5 HINT NOP).
 *     The hint does not change memory ordering, so no caller fence is
 *     needed (unlike x86 MOVNTDQ / AArch64 STNP).
 *   - prefetch twins (cand.prefetch_dist>0): one Zicbop prefetch.r per
 *     input stream per group iteration, prefetch_dist bytes ahead.
 *     Enumerated only when the CPU reports Zicbop (rj_cpu_info.
 *     rvv_zicbop); the backend emits prefetch.r only then (exact gate:
 *     "+pf" in the desc iff >= 1 prefetch is emitted).
 *
 * Checks:
 *  A. planner: RVV NT twins are enumerated with/without Zicbop; RVV
 *     prefetch twins only with Zicbop.
 *  B. force-built NT twin: desc has "+nt"; #NTL.ALL == #vse8.v >= 1 and
 *     every vse8.v is immediately preceded by NTL.ALL; temporal twin
 *     has neither; output matches the scalar reference.
 *  C. force-built prefetch twin (synthetic zicbop=1): desc has "+pf";
 *     >= 1 prefetch.r words, all with the requested offset; output
 *     matches the scalar reference.
 *  D. exact gate: synthetic zicbop=0 + prefetch_dist -> no "+pf", no
 *     prefetch.r words (temporal fallback, never wrong code).
 *  E. clean failures: prefetch_dist not a multiple of 32 (100) or
 *     above the encodable range (4096) fails the build with an error.
 *
 * RVV-only: the planner enumerates RVV candidates only in a riscv64
 * build, and the JIT parts need RVV to execute. On other hosts the
 * binary builds and prints SKIP.
 *
 * Run under QEMU at several VLENs, e.g.
 *   QEMU_LD_PREFIX=/usr/riscv64-linux-gnu qemu-riscv64 -cpu max \
 *       ./build-riscv64/test_riscv_prefetch
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
static const size_t lens[] = { 0, 1, 7, 15, 16, 17, 31, 32, 33, 63, 64,
                               65, 100, 255, 256, 257, 1000, 4096 };
static const int nlens = sizeof(lens) / sizeof(lens[0]);

static uint64_t rng_state = 0x51ed270c1c48d5a9ULL;
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

static uint32_t word_at(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

/* Zihintntl NTL.ALL (GAS-verified: `ntl.all` -> 0x00500033). */
#define W_NTL_ALL 0x00500033u

/* Zicbop prefetch.r: ORI x0,rs1,imm HINT with imm[4:0]=0b00001.
 * Mask covers opcode/funct3/rd (rd must be x0); imm[4:0]==1 selects
 * .r (not .w/.i). */
static int is_prefetch_r(uint32_t w)
{
    return (w & 0x00007f7fu) == 0x00006013u && ((w >> 20) & 0x1fu) == 1;
}
static unsigned prefetch_r_off(uint32_t w)
{
    return ((w >> 20) & 0x7ffu) & 0x7e0u; /* imm[11:5]: 32-byte units */
}

/* vse8.v unit-stride store: fixed bits 31:25=0000010, 14:12=000,
 * opcode 0100111. */
static int is_vse8(uint32_t w)
{
    return (w & 0xfe00707fu) == 0x02000027u;
}

/* ---------------- A. planner twin enumeration ---------------- */

static void test_planner_twins(void)
{
    char err[256];
    rj_topology *t = rj_topo_rs(6, 3);
    CHECK(t != NULL, "rj_topo_rs(6,3)");
    if (!t)
        return;
    topo_int_t ti;
    CHECK(topo_int_build(&ti, t, err, sizeof(err)) == 0, "topo: %s", err);
    rj_op op = rj_op_encode();
    plan_t pl;
    CHECK(plan_build(&ti, &op, "rs63", &pl, err, sizeof(err)) == 0,
          "plan: %s", err);
    if (failures) {
        rj_topo_free(t);
        return;
    }

    for (int z = 0; z < 2; z++) {
        rj_cpu_info cpu;
        memset(&cpu, 0, sizeof(cpu));
        cpu.rvv = 1;
        cpu.rvv_zicbop = z;
        cand_t cands[256];
        int nc = plan_candidates(&pl, &cpu, cands, 256, 1, 128);
        CHECK(nc > 0, "zicbop=%d: no candidates", z);
        int n_nt = 0, n_pf = 0;
        for (int i = 0; i < nc; i++) {
            if (cands[i].isa != ISA_RVV)
                continue;
            if (cands[i].nt)
                n_nt++;
            if (cands[i].prefetch_dist == 128)
                n_pf++;
        }
        /* NT twins need no gating (HINT NOP everywhere). */
        CHECK(n_nt > 0, "zicbop=%d: no RVV NT twins enumerated", z);
        if (z)
            CHECK(n_pf > 0,
                  "zicbop=1: no RVV prefetch twins enumerated");
        else
            CHECK(n_pf == 0,
                  "zicbop=0: RVV prefetch twins must not be enumerated "
                  "(backend would ignore them)");
    }

    plan_free(&pl);
    rj_op_free(&op);
    topo_int_free(&ti);
    rj_topo_free(t);
}

/* ---------------- B-E helpers ---------------- */

/* Build one RVV candidate directly, bypassing the planner's twin
 * policy (mirrors tests/test_riscv_remat.c). */
static int build_cand(const plan_t *pl, const rj_cpu_info *cpu, int U,
                      int nt, size_t pf_dist, version_t *v, char *err,
                      size_t errlen)
{
    rj_options o;
    rj_options_default(&o);
    o.benchmark = 0;
    cand_t cand;
    memset(&cand, 0, sizeof(cand));
    cand.isa = ISA_RVV;
    cand.mul = MUL_NIBBLE;
    cand.U = U;
    cand.nt = nt;
    cand.prefetch_dist = pf_dist;
    memset(v, 0, sizeof(*v));
    return jit_build_version(pl, &cand, cpu, &o, v, err, errlen);
}

/* Encode correctness of a built version against the scalar reference. */
static void check_correct(const char *name, const plan_t *pl,
                          const topo_int_t *ti, const version_t *v)
{
    int nd = ti->ndata, np = ti->npar;
    for (int li = 0; li < nlens && !failures; li++) {
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
        v->fn((const void **)in, (void **)out, len);
        ref_run(pl, (const uint8_t **)in, exp, len);
        for (int j = 0; j < np; j++)
            CHECK(memcmp(out[j], exp[j], len) == 0,
                  "%s [%s] len=%zu parity%d mismatch", name, v->desc,
                  len, j);
        for (int i = 0; i < nd; i++)
            free(in[i]);
        for (int j = 0; j < np; j++) {
            free(out[j]);
            free(exp[j]);
        }
    }
}

/* Scan the code: count NTL.ALL / vse8.v / prefetch.r words. */
static void scan_code(const version_t *v, int *n_ntl, int *n_vse,
                      int *n_pf)
{
    *n_ntl = *n_vse = *n_pf = 0;
    for (size_t i = 0; i + 4 <= v->code_size; i += 4) {
        uint32_t w = word_at(v->mem + i);
        if (w == W_NTL_ALL)
            (*n_ntl)++;
        if (is_vse8(w))
            (*n_vse)++;
        if (is_prefetch_r(w))
            (*n_pf)++;
    }
}

static void test_nt_twin(const plan_t *pl, const topo_int_t *ti,
                         const rj_cpu_info *cpu)
{
    char err[256];
    version_t v;
    CHECK(build_cand(pl, cpu, 4, 1, 0, &v, err, sizeof(err)) == 0,
          "nt build: %s", err);
    if (failures)
        return;
    CHECK(strstr(v.desc, "+nt") != NULL, "nt desc lacks +nt: %s", v.desc);
    int n_ntl, n_vse, n_pf;
    scan_code(&v, &n_ntl, &n_vse, &n_pf);
    CHECK(n_vse >= 1, "nt [%s]: no vse8.v found", v.desc);
    CHECK(n_ntl == n_vse,
          "nt [%s]: ntl.all=%d != vse8.v=%d", v.desc, n_ntl, n_vse);
    /* every vse8.v immediately preceded by NTL.ALL (the hint applies
     * to the immediately subsequent instruction) */
    int adj_ok = 1;
    for (size_t i = 4; i + 4 <= v.code_size; i += 4)
        if (is_vse8(word_at(v.mem + i)) &&
            word_at(v.mem + i - 4) != W_NTL_ALL)
            adj_ok = 0;
    CHECK(adj_ok, "nt [%s]: vse8.v without preceding ntl.all", v.desc);
    CHECK(n_pf == 0, "nt [%s]: unexpected prefetch.r", v.desc);
    check_correct("nt", pl, ti, &v);
    jit_free_version(&v);

    /* temporal twin: no hint words at all */
    version_t tv;
    CHECK(build_cand(pl, cpu, 4, 0, 0, &tv, err, sizeof(err)) == 0,
          "temporal build: %s", err);
    if (failures)
        return;
    CHECK(strstr(tv.desc, "+nt") == NULL, "temporal desc has +nt: %s",
          tv.desc);
    scan_code(&tv, &n_ntl, &n_vse, &n_pf);
    CHECK(n_ntl == 0, "temporal [%s]: %d ntl.all words", tv.desc, n_ntl);
    CHECK(n_vse >= 1, "temporal [%s]: no vse8.v", tv.desc);
    jit_free_version(&tv);
}

static void test_prefetch_twin(const plan_t *pl, const topo_int_t *ti,
                               const rj_cpu_info *cpu)
{
    char err[256];
    version_t v;
    CHECK(build_cand(pl, cpu, 4, 0, 128, &v, err, sizeof(err)) == 0,
          "prefetch build: %s", err);
    if (failures)
        return;
    CHECK(strstr(v.desc, "+pf") != NULL, "prefetch desc lacks +pf: %s",
          v.desc);
    int n_ntl, n_vse, n_pf;
    scan_code(&v, &n_ntl, &n_vse, &n_pf);
    CHECK(n_pf >= 1, "prefetch [%s]: no prefetch.r words", v.desc);
    int off_ok = 1;
    for (size_t i = 0; i + 4 <= v.code_size; i += 4) {
        uint32_t w = word_at(v.mem + i);
        if (is_prefetch_r(w) && prefetch_r_off(w) != 128)
            off_ok = 0;
    }
    CHECK(off_ok, "prefetch [%s]: prefetch.r with wrong offset",
          v.desc);
    CHECK(n_ntl == 0, "prefetch [%s]: unexpected ntl.all", v.desc);
    check_correct("prefetch", pl, ti, &v);
    jit_free_version(&v);
}

/* Exact gate: prefetch_dist without Zicbop degrades to temporal --
 * no "+pf", no prefetch words, still correct. */
static void test_prefetch_gate(const plan_t *pl, const topo_int_t *ti,
                               const rj_cpu_info *cpu_noz)
{
    char err[256];
    version_t v;
    CHECK(build_cand(pl, cpu_noz, 4, 0, 128, &v, err, sizeof(err)) == 0,
          "nozicbop build: %s", err);
    if (failures)
        return;
    CHECK(strstr(v.desc, "+pf") == NULL,
          "nozicbop desc must not have +pf: %s", v.desc);
    int n_ntl, n_vse, n_pf;
    scan_code(&v, &n_ntl, &n_vse, &n_pf);
    CHECK(n_pf == 0, "nozicbop [%s]: %d prefetch.r words", v.desc, n_pf);
    check_correct("nozicbop", pl, ti, &v);
    jit_free_version(&v);
}

/* Unencodable distances fail the build cleanly (never wrong code). */
static void test_prefetch_reject(const plan_t *pl, const rj_cpu_info *cpu)
{
    char err[256];
    version_t v;
    static const size_t bad[] = { 100, 4096, 8192, 16 };
    for (size_t k = 0; k < sizeof(bad) / sizeof(bad[0]); k++) {
        memset(&v, 0, sizeof(v));
        int rc = build_cand(pl, cpu, 4, 0, bad[k], &v, err, sizeof(err));
        CHECK(rc != 0, "prefetch_dist=%zu: build should fail", bad[k]);
        if (rc != 0)
            CHECK(strstr(err, "not encodable") != NULL,
                  "prefetch_dist=%zu: error should say not encodable: %s",
                  bad[k], err);
        else
            jit_free_version(&v);
    }
    /* boundary: 32 and 2016 build fine */
    static const size_t good[] = { 32, 2016 };
    for (size_t k = 0; k < sizeof(good) / sizeof(good[0]); k++) {
        memset(&v, 0, sizeof(v));
        int rc = build_cand(pl, cpu, 4, 0, good[k], &v, err, sizeof(err));
        CHECK(rc == 0, "prefetch_dist=%zu: build: %s", good[k], err);
        if (rc == 0)
            jit_free_version(&v);
    }
}

/* Noop plan (n_out == 0): no loop is emitted, so a prefetch twin
 * degrades to temporal -- no "+pf", no prefetch words. */
static void test_prefetch_noop(const rj_cpu_info *cpu)
{
    char err[256];
    rj_topology *t = rj_topo_raid0(4);
    CHECK(t != NULL, "rj_topo_raid0(4)");
    if (!t)
        return;
    topo_int_t ti;
    CHECK(topo_int_build(&ti, t, err, sizeof(err)) == 0, "topo: %s", err);
    rj_op op = rj_op_encode();
    plan_t pl;
    CHECK(plan_build(&ti, &op, "raid0", &pl, err, sizeof(err)) == 0,
          "plan: %s", err);
    if (!failures) {
        version_t v;
        CHECK(build_cand(&pl, cpu, 4, 0, 128, &v, err, sizeof(err)) == 0,
              "noop prefetch build: %s", err);
        if (!failures) {
            CHECK(strstr(v.desc, "+pf") == NULL,
                  "noop desc must not have +pf: %s", v.desc);
            int n_ntl, n_vse, n_pf;
            scan_code(&v, &n_ntl, &n_vse, &n_pf);
            CHECK(n_pf == 0, "noop [%s]: %d prefetch.r words", v.desc,
                  n_pf);
            jit_free_version(&v);
        }
        plan_free(&pl);
    }
    rj_op_free(&op);
    topo_int_free(&ti);
    rj_topo_free(t);
}

#endif /* __riscv */

int main(void)
{
#ifdef __riscv
    test_planner_twins();
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    if (!cpu.rvv) {
        printf("test_riscv_prefetch: SKIP (no RVV)\n");
        return failures != 0;
    }
    /* Synthetic Zicbop: QEMU user-mode does not report multi-letter
     * extensions (guest sees the host cpuinfo), but qemu's -cpu max
     * executes the prefetch hints. Real hardware is detected via
     * /proc/cpuinfo in rj_cpu_detect. */
    rj_cpu_info cpu_z = cpu;
    cpu_z.rvv_zicbop = 1;

    rj_topology *t = rj_topo_rs(6, 3);
    CHECK(t != NULL, "rj_topo_rs(6,3)");
    if (t) {
        char err[256];
        topo_int_t ti;
        if (topo_int_build(&ti, t, err, sizeof(err)) != 0) {
            CHECK(0, "topo: %s", err);
        } else {
            rj_op op = rj_op_encode();
            plan_t pl;
            if (plan_build(&ti, &op, "rs63", &pl, err, sizeof(err)) !=
                0) {
                CHECK(0, "plan: %s", err);
            } else {
                test_nt_twin(&pl, &ti, &cpu_z);
                test_prefetch_twin(&pl, &ti, &cpu_z);
                test_prefetch_gate(&pl, &ti, &cpu);
                test_prefetch_reject(&pl, &cpu_z);
                test_prefetch_noop(&cpu_z);
                plan_free(&pl);
            }
            rj_op_free(&op);
            topo_int_free(&ti);
        }
        rj_topo_free(t);
    }
#else
    printf("test_riscv_prefetch: SKIP (RVV-only test)\n");
#endif

    if (failures)
        printf("test_riscv_prefetch: %d FAILURES\n", failures);
    else
        printf("test_riscv_prefetch: all OK\n");
    return failures != 0;
}
