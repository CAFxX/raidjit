/* notail x xorsched twin combination (merge plan §5a).
 *
 * The length-dispatch (A2) notail twin and the XOR-schedule (A3) twin are
 * orthogonal candidate axes that must compose: the stage-2 shootout in
 * api.c builds the notail twin of the stage-1 winner, which may itself
 * be an xsched twin.
 *
 * 1. White-box: sharing-heavy 20-in/4-out P_XOR plan (the A3 duel shape).
 *    plan_candidates must enumerate an xorsched=1 twin (fail loudly if
 *    not: the test's premise is a config where the twin exists). Take
 *    each enumerated twin, set notail=1, jit_build_version must succeed;
 *    desc must contain "+xsched" before " notail" (C7 tag order, which
 *    A2's twin.desc == winner.desc + " notail" invariant relies on);
 *    model-exactness: v.uops_per_byte == xorsched_version_uops on the
 *    combo (A3-m1 rounding); differential vs the scalar reference at
 *    twin-admissible lengths (group, 2*group, one large multiple).
 * 2. E2E: rj_topo_custom(20,4,t1_mat) (same shape, public API) with
 *    benchmark=1. rj_get_fn bit-exact vs scalar ref at admitted and
 *    non-admitted lengths. If the stage-1 winner was an xsched twin
 *    ("+xsched" in the desc), api.c's stage-2 shootout necessarily ran
 *    (it is unconditional when benchmark=1 and the plan is not noop);
 *    the test then proves the twin was a real contender by rebuilding
 *    every enumerated xsched twin's notail twin white-box (a twin build
 *    failure degrades silently to general-only, so a build failure here
 *    would make the shootout vacuous). Reports whether the notail twin
 *    won stage 2 ("+notail" present) or the general version won.
 *
 * Buffers are 64-byte aligned: the AVX backends emit aligned vector
 * loads (vmovdqa), so plain malloc'd buffers fault. (A test-harness
 * bug of exactly this form was caught during development: the naive
 * U=7 twin segfaulted on malloc'd buffers, which misdirected the
 * investigation at the emitter before the alignment cause was found.)
 *
 * Mutation / red-green evidence (run manually, documented here):
 *  (a) jit.c desc: swap the tag order (" notail" before "+xsched") ->
 *      the C7 order check fails.
 *  (b) plan_candidates: force xorsched=0 on all twins -> the
 *      "no xsched twin enumerated" loud failure fires.
 *  (c) xorsched emitter: corrupt one fan-out XOR -> the differential
 *      vs scalar ref fails at group multiples.
 *  (d) 2026-10-03: the e2e C7 gate fired on a legitimate
 *      "custom(20,4) xor avx512 U=3 +notail +maskedtail" desc (plain
 *      winner, notail won the shootout, maskedtail won its tail duel;
 *      benchmark noise chose that path) -- track Bx86
 *      B2_make_test.log:574. The gate wrongly assumed notail-installed
 *      implies an xsched winner. Fixed the gate to require "+xsched"
 *      present (e2e_applies_xsched_notail_order); the gate + order
 *      check are table-tested in test_desc_gate_logic (the old gate
 *      fails case 0; && -> || in the new gate fails case 0; a
 *      synthetic wrong-order desc fails the order check).
 */
#include "../include/raidjit.h"
#include "../src/plan.h"
#include "../src/jit.h"
#include "../src/cpu.h"
#include "../src/ref.h"
#include "../src/xorsched.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

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

static uint64_t rng_state = 0x9e3779b97f4a7c15ULL;
static uint64_t xrng(void)
{
    uint64_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    rng_state = x;
    return x;
}

/* 64-byte aligned buffer: the AVX backends emit aligned vector loads. */
static void *abuf(size_t n)
{
    void *p = NULL;
    if (posix_memalign(&p, 64, n ? n : 64)) {
        printf("out of memory\n");
        exit(1);
    }
    return p;
}

/* The A3 duel shape: 20 inputs, 4 outputs, heavy cross-row sharing. */
static const uint8_t t1_mat[80] = {
    1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,
    1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,
    0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,
};

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

/* divisor for uops_per_byte -> uops (backend-specific; mirrors
 * test_xorsched.c) */
static long sve_true_vl_bytes(const rj_cpu_info *cpu)
{
#ifdef __aarch64__
    /* PR_SVE_GET_VL (which fills cpu->sve_vl) under-reports 8x under
     * qemu-user 8.2.2 (reports N/8 bytes for -cpu
     * max,sve-default-vector-length=N, true VL is N bytes). Probe the
     * true VL via cntb, the same ground truth src/aarch64/jit.c and
     * tests/test_boundaries.c use. */
    unsigned long probe = 0;
    __asm__ volatile(".arch_extension sve\n\tcntb %0" : "=r"(probe));
    if (probe >= 16)
        return (long)probe;
#endif
    return cpu->sve_vl > 0 ? (long)cpu->sve_vl : 16;
}

static long uop_divisor(isa_t isa, int U, int lmul, const rj_cpu_info *cpu)
{
    switch (isa) {
    case ISA_SSE: return (long)U * 16;
    case ISA_AVX2: return (long)U * 32;
    case ISA_AVX512: return (long)U * 64;
    case ISA_NEON: return (long)U * 16;
    case ISA_SVE: return (long)U * sve_true_vl_bytes(cpu);
    /* RVV: the backend normalizes uops_per_byte to U*NOMVL*lmul
     * nominal bytes (src/riscv/jit.c), so the divisor must scale by
     * the candidate's lmul (0 = 1); omitting it under-converts every
     * lmul>1 twin (X1: model 33 vs emitter 8 at U=1 lmul=4). */
    case ISA_RVV: return (long)U * 16 * (lmul ? lmul : 1); /* NOMVL */
    default: return (long)U * 16;
    }
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

/* Build a P_XOR plan from the shared matrix, with the schedule attached
 * (as plan_build does). */
static void make_plan(plan_t *pl, const char *name)
{
    memset(pl, 0, sizeof(*pl));
    pl->kind = P_XOR;
    pl->n_in = 20;
    pl->n_out = 4;
    pl->mat = malloc(sizeof(t1_mat));
    memcpy(pl->mat, t1_mat, sizeof(t1_mat));
    snprintf(pl->name, sizeof(pl->name), "%s", name);
    pl->xor_sched = xorsched_build(pl);
}

static void free_plan(plan_t *pl)
{
    xorsched_free(pl->xor_sched);
    free(pl->mat);
}

/* White-box: the notail twin of every enumerated xsched twin must build,
 * carry both desc tags in C7 order, be model-exact, and match the scalar
 * reference at twin-admissible lengths. Returns the number of combos
 * exercised. */
static int test_whitebox(const rj_cpu_info *cpu)
{
    plan_t pl;
    make_plan(&pl, "notail-xorsched");
    if (!pl.xor_sched) {
        printf("FAIL: no schedule for the sharing-heavy shape\n");
        failures++;
        free_plan(&pl);
        return 0;
    }

    cand_t out[256];
    int n = plan_candidates(&pl, cpu, out, 256, 0, 0);
    int ntwins = 0;
    for (int i = 0; i < n; i++)
        if (out[i].xorsched)
            ntwins++;
    /* Non-vacuity: this shape must enumerate xsched twins. */
    CHECK(ntwins > 0, "no xorsched twin enumerated for t1 shape (n=%d)", n);
    if (ntwins == 0) {
        free_plan(&pl);
        return 0;
    }

    rj_options opts;
    memset(&opts, 0, sizeof(opts));
    char err[256];
    int ncombo = 0;

    for (int i = 0; i < n; i++) {
        if (!out[i].xorsched)
            continue;
        cand_t twinc = out[i];
        twinc.notail = 1;
        version_t v;
        memset(&v, 0, sizeof(v));
        if (jit_build_version(&pl, &twinc, cpu, &opts, &v, err,
                              sizeof(err)) != 0) {
            CHECK(0, "%s U=%d: notail+xorsched twin build failed: %s",
                  isa_str(out[i].isa), out[i].U, err);
            continue;
        }
        /* C7 tag order: "+xsched" immediately after "+ternlog", and
         * " notail" last. */
        const char *px = strstr(v.desc, "+xsched");
        const char *pn = strstr(v.desc, " notail");
        CHECK(px != NULL && pn != NULL && px < pn,
              "%s U=%d: desc tag order wrong: %s", isa_str(out[i].isa),
              out[i].U, v.desc);
        CHECK(v.notail == 1, "%s U=%d: version not flagged notail",
              isa_str(out[i].isa), out[i].U);

        /* Model-exactness on the combo (A3-m1 rounding). The notail
         * twin's uops_per_byte must equal the scheduled model's
         * notail-aware prediction exactly: on x86/AArch64 the notail
         * twin omits only the uncounted scalar tail, while on RVV it
         * emits just the single nu=U body (not the double emission),
         * which the model's notail=1 accounts for. */
        long div = uop_divisor(out[i].isa, out[i].U, out[i].lmul, cpu);
        long ps = xorsched_version_uops(pl.xor_sched, &pl, out[i].isa,
                                        out[i].U, 1, 0, 1);
        long es = (long)(v.uops_per_byte * div + 0.5);
        CHECK(es == ps, "%s U=%d: combo model %ld != emitter %ld",
              isa_str(out[i].isa), out[i].U, ps, es);

        size_t G = v.group_bytes;
        CHECK(G > 0, "%s U=%d: group_bytes not set", isa_str(out[i].isa),
              out[i].U);
        if (G == 0) {
            jit_free_version(&v);
            continue;
        }
        size_t lens[3];
        lens[0] = G;
        lens[1] = 2 * G;
        lens[2] = (65536 / G) * G;

        size_t maxlen = lens[2] + 64;
        uint8_t *in[20], *got[4], *ref[4];
        const void *inv[20];
        void *gotv[4];
        for (int k = 0; k < 20; k++) {
            in[k] = abuf(maxlen);
            inv[k] = in[k];
        }
        for (int j = 0; j < 4; j++) {
            got[j] = abuf(maxlen);
            ref[j] = abuf(maxlen);
            gotv[j] = got[j];
        }
        for (int li = 0; li < 3; li++) {
            size_t len = lens[li];
            for (int k = 0; k < 20; k++)
                for (size_t b = 0; b < len; b++)
                    in[k][b] = (uint8_t)xrng();
            v.fn(inv, gotv, len);
            scalar_ref(&pl, in, ref, len);
            for (int j = 0; j < 4; j++)
                CHECK(memcmp(got[j], ref[j], len) == 0,
                      "%s U=%d: combo mismatch at len=%zu out=%d",
                      isa_str(out[i].isa), out[i].U, len, j);
        }
        for (int k = 0; k < 20; k++)
            free(in[k]);
        for (int j = 0; j < 4; j++) {
            free(got[j]);
            free(ref[j]);
        }
        ncombo++;
        jit_free_version(&v);
    }
    free_plan(&pl);
    printf("note: white-box notail+xorsched combos exercised: %d\n", ncombo);
    return ncombo;
}

/* Plausible groups so the length matrix hits admitted and non-admitted
 * lengths whatever the winner's group is. */
static const size_t egroups[] = { 16, 32, 48, 64, 80, 96, 112, 128,
                                  160, 192, 224, 256, 320, 384, 448, 512 };

/* E2E through the public API. */
static void test_e2e(void); /* (forward decl: helpers below precede it) */

/* The C7 tag-order check applies only when the installed notail twin
 * belongs to an xsched stage-1 winner: the "+xsched" tag comes from the
 * winner's own body, " notail" is appended by rj_get_info for the
 * installed stage-2 twin. A plain winner's notail twin carries no
 * +xsched tag -- e.g. the observed "custom(20,4) xor avx512 U=3
 * +notail +maskedtail" (plain winner, notail won the shootout,
 * maskedtail won its tail duel) -- and demanding one there is a false
 * positive. 2026-10-03: the old gate (notail present alone) fired
 * exactly that false positive (track Bx86 B2_make_test.log:574); the
 * gate below is the fix. */
static int e2e_applies_xsched_notail_order(const char *desc)
{
    return strstr(desc, "+xsched") != NULL && strstr(desc, "notail") != NULL;
}

/* C7 order on the public desc: "+xsched" (winner's body) before
 * "notail" (appended stage-2 twin tag). Returns 1 when both tags are
 * present in order; the gate above decides applicability. */
static int e2e_xsched_notail_order_ok(const char *desc)
{
    const char *px = strstr(desc, "+xsched");
    const char *pn = strstr(desc, "notail");
    return px != NULL && pn != NULL && px < pn;
}

/* Deterministic unit test for the C7 gate logic (the e2e path above
 * depends on benchmark noise for which winner/twin installs). */
static void test_desc_gate_logic(void)
{
    struct {
        const char *desc;
        int applies;  /* e2e_applies_xsched_notail_order */
        int order_ok; /* e2e_xsched_notail_order_ok, when applies */
    } cases[] = {
        /* the observed false positive: plain winner, notail + maskedtail
         * installed -- the old gate fired, the fixed gate must not */
        { "custom(20,4) xor avx512 U=3 +notail +maskedtail", 0, 0 },
        /* xsched winner, general won stage-2: xsched tag present but no
         * notail twin installed -- gate must not apply */
        { "custom(20,4) xor avx512 U=3+xsched +maskedtail", 0, 0 },
        /* xsched winner, notail twin installed: gate applies, order ok */
        { "custom(20,4) xor avx512 U=3+xsched +notail", 1, 1 },
        { "custom(20,4) xor avx512 U=3+xsched +notail +maskedtail", 1, 1 },
        /* xsched winner, general won stage 2: no notail tag */
        { "custom(20,4) xor avx512 U=3+xsched", 0, 0 },
        /* plain winner, general won stage 2 */
        { "custom(20,4) xor avx512 U=3", 0, 0 },
        /* synthetic wrong order: the order check must still fire */
        { "custom(20,4) xor avx512 U=3 +notail+xsched", 1, 0 },
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        int ap = e2e_applies_xsched_notail_order(cases[i].desc);
        CHECK(ap == cases[i].applies,
              "gate logic case %zu (%s): applies=%d, want %d", i,
              cases[i].desc, ap, cases[i].applies);
        if (cases[i].applies) {
            int ok = e2e_xsched_notail_order_ok(cases[i].desc);
            CHECK(ok == cases[i].order_ok,
                  "gate logic case %zu (%s): order_ok=%d, want %d", i,
                  cases[i].desc, ok, cases[i].order_ok);
        }
    }
    printf("note: desc gate-logic table: %zu cases checked\n",
           sizeof(cases) / sizeof(cases[0]));
}

static void test_e2e(void)
{
    char err[256];
    rj_topology *t = rj_topo_custom(20, 4, t1_mat);
    CHECK(t != NULL, "rj_topo_custom(20,4) failed");
    if (!t)
        return;
    rj_op op = rj_op_encode();

    rj_options o;
    rj_options_default(&o);
    o.benchmark = 1;
    o.bench_bytes = 1 << 16; /* keep the shootout fast */
    o.bench_iters = 3;

    rj_jit *j = rj_compile(t, &op, &o, err, sizeof(err));
    CHECK(j != NULL, "e2e compile: %s", err);
    if (!j) {
        rj_op_free(&op);
        rj_topo_free(t);
        return;
    }
    rj_fn fn = rj_get_fn(j);
    CHECK(fn != NULL, "e2e: fn non-null");
    rj_info info;
    rj_get_info(j, &info);
    int xsched_won_stage1 = strstr(info.desc, "+xsched") != NULL;
    int notail_won_stage2 = strstr(info.desc, "notail") != NULL;
    printf("note: e2e winner -> %s %s\n", info.desc,
           xsched_won_stage1
               ? (notail_won_stage2 ? "[xsched won stage 1, notail twin won stage 2]"
                                    : "[xsched won stage 1, general won stage 2]")
               : "[stage-1 winner was not an xsched twin]");
    if (e2e_applies_xsched_notail_order(info.desc)) {
        /* C7 order must hold on the public desc too. */
        CHECK(e2e_xsched_notail_order_ok(info.desc),
              "e2e desc tag order wrong: %s", info.desc);
    }

    /* Reference plan for the scalar oracle. */
    topo_int_t ti;
    plan_t pl;
    CHECK(topo_int_build(&ti, t, err, sizeof(err)) == 0, "e2e topo: %s",
          err);
    CHECK(plan_build(&ti, &op, "e2e", &pl, err, sizeof(err)) == 0,
          "e2e plan: %s", err);
    CHECK(pl.kind == P_XOR, "e2e: custom 0/1 matrix did not lower to P_XOR");

    size_t maxlen = 65537 + 64;
    uint8_t *ib[20], *ob[4], *rb[4];
    const void *in[20];
    void *out[4];
    for (int k = 0; k < 20; k++) {
        ib[k] = abuf(maxlen);
        in[k] = ib[k];
    }
    for (int j = 0; j < 4; j++) {
        ob[j] = abuf(maxlen);
        rb[j] = abuf(maxlen);
        out[j] = ob[j];
    }
    size_t lens[64];
    int nl = 0;
    lens[nl++] = 0;
    lens[nl++] = 1;
    for (size_t gi = 0; gi < sizeof(egroups) / sizeof(egroups[0]); gi++) {
        size_t g = egroups[gi];
        lens[nl++] = g - 1;
        lens[nl++] = g;
        lens[nl++] = g + 1;
    }
    lens[nl++] = 65537;
    for (int li = 0; li < nl; li++) {
        size_t len = lens[li];
        for (int k = 0; k < 20; k++)
            for (size_t b = 0; b < len; b++)
                ib[k][b] = (uint8_t)xrng();
        fn(in, out, len);
        ref_run(&pl, (const uint8_t **)ib, rb, len);
        for (int j = 0; j < 4; j++)
            CHECK(memcmp(ob[j], rb[j], len) == 0,
                  "e2e mismatch at len=%zu out=%d", len, j);
    }

    if (xsched_won_stage1) {
        /* The stage-2 shootout necessarily ran (unconditional in api.c
         * when benchmark=1 and the plan is not noop). Prove the notail
         * twin was a real contender, not a silent build failure: every
         * enumerated xsched twin's notail twin must build white-box. */
        rj_cpu_info cpu;
        rj_cpu_detect(&cpu);
        cand_t cands[256];
        int nc = plan_candidates(&pl, &cpu, cands, 256, 0, 0);
        int checked = 0;
        rj_options wo;
        memset(&wo, 0, sizeof(wo));
        for (int i = 0; i < nc; i++) {
            if (!cands[i].xorsched)
                continue;
            cand_t tc = cands[i];
            tc.notail = 1;
            version_t tv;
            memset(&tv, 0, sizeof(tv));
            char e2[256];
            int ok = jit_build_version(&pl, &tc, &cpu, &wo, &tv, e2,
                                       sizeof(e2)) == 0;
            CHECK(ok,
                  "e2e: notail twin of xsched winner candidate (%s U=%d) "
                  "failed to build: %s",
                  isa_str(cands[i].isa), cands[i].U, e2);
            if (ok)
                jit_free_version(&tv);
            checked++;
        }
        CHECK(checked > 0, "e2e: no xsched twin to re-check");
        printf("note: stage-2 shootout genuinely ran (%d xsched twin(s) "
               "re-built white-box)\n", checked);
    }

    for (int k = 0; k < 20; k++)
        free(ib[k]);
    for (int j = 0; j < 4; j++) {
        free(ob[j]);
        free(rb[j]);
    }
    plan_free(&pl);
    topo_int_free(&ti);
    rj_free(j);
    rj_op_free(&op);
    rj_topo_free(t);
}

int main(void)
{
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
#if defined(__x86_64__)
    if (!cpu.sse2) {
        printf("no vector ISA, skipping\n");
        return 0;
    }
#elif defined(__aarch64__)
    if (!cpu.neon && !(cpu.sve && cpu.sve_vl > 0)) {
        printf("no vector ISA, skipping\n");
        return 0;
    }
#elif defined(__riscv)
    if (!cpu.rvv) {
        printf("no vector ISA, skipping\n");
        return 0;
    }
#endif
    int ncombo = test_whitebox(&cpu);
    CHECK(ncombo > 0, "no notail+xorsched combo exercised white-box");
    test_desc_gate_logic();
    test_e2e();
    if (failures == 0)
        printf("all notail+xorsched combo tests passed\n");
    return failures ? 1 : 0;
}
