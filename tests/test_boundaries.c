/* Systematic main-loop/remainder boundary-length tests.
 *
 * The JIT emits a vectorized main loop over groups of U*VL bytes plus a
 * remainder path for the rest (scalar tail on x86-64/AArch64, strip-mined
 * remainder on RVV). Off-by-one errors at the main/remainder seam are the
 * classic JIT bug class: a byte skipped, doubled, or an overrun past len.
 *
 * For every (topology, op) case below and every enumerated candidate with
 * U in {1, 2, max-per-ISA}, this test runs the boundary lengths
 *     { 0, 1, VL-1, VL, VL+1, U-1, U, U+1,
 *       U*VL-1, U*VL, U*VL+1, 2*U*VL-1, 2*U*VL, 2*U*VL+1 }
 * (deduplicated) and checks three things per length:
 *   1. output bytes [0,len) match the scalar ref_run() oracle exactly;
 *   2. input bytes [0,len) are byte-identical afterwards (no clobber);
 *   3. no byte at or past len was touched (buffers are mmap'd with
 *      PROT_NONE guard pages on both sides and the data pages pre-poisoned
 *      with 0xA5; [len, page_end) must still be poison).  Item 3 makes
 *      length 0 prove "writes nothing": any store would fault or corrupt
 *      poison.
 *
 * Cases cover encode (P_XOR/P_HORNER/P_MULACC), recovery (single and double
 * data-disk failure), and parity-update (P_DELTA). Recovery inputs are
 * random survivor data; the
 * plan's solved matrix plus ref_run() is the oracle, so no re-encode is
 * needed. This is a boundary matrix, not a fuzz: representative
 * topologies only (raid5, raid6, one RS, one custom non-Vandermonde);
 * wider matrices are covered by test_pdelta_edge.c / test_large_mds.c.
 *
 * Portable: uses only the plan API + rj_fn calls, so it runs natively on
 * x86-64 and under QEMU for aarch64 (NEON, SVE 128/256) and riscv64
 * (VLEN 128/256/512). Whatever ISAs the planner offers for the detected
 * CPU are exercised; per-ISA coverage is asserted non-vacuous.
 */
#include "../include/raidjit.h"
#include "../src/plan.h"
#include "../src/jit.h"
#include "../src/cpu.h"
#include "../src/ref.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>
#include <unistd.h>

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

/* ---- xorshift RNG ---- */
static uint64_t rng_state = 0x123456789ABCDEFULL;
static uint64_t xrng(void)
{
    uint64_t x = rng_state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return rng_state = x;
}
static void rnd_bytes(uint8_t *p, size_t n)
{
    for (size_t i = 0; i < n; i++)
        p[i] = (uint8_t)(xrng() >> 33);
}

/* ---- guard-page buffers with poison ----
 * Layout: [PROT_NONE][data pages, pre-poisoned 0xA5][PROT_NONE].
 * Page-aligned, hence 64-byte aligned per the caller contract. */
typedef struct {
    uint8_t *p;
    size_t data_bytes;
} gbuf;

static size_t g_pagesz(void)
{
    static long ps = 0;
    if (!ps) {
        ps = sysconf(_SC_PAGESIZE);
        if (ps <= 0) {
            perror("sysconf(_SC_PAGESIZE)");
            exit(1);
        }
    }
    return (size_t)ps;
}

static gbuf g_alloc(size_t maxlen)
{
    size_t ps = g_pagesz();
    size_t npages = (maxlen + ps - 1) / ps;
    if (!npages)
        npages = 1;
    size_t total = (npages + 2) * ps;
    uint8_t *base = mmap(NULL, total, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED) {
        perror("mmap");
        exit(1);
    }
    if (mprotect(base, ps, PROT_NONE) != 0 ||
        mprotect(base + (npages + 1) * ps, ps, PROT_NONE) != 0) {
        perror("mprotect");
        exit(1);
    }
    gbuf g;
    g.p = base + ps;
    g.data_bytes = npages * ps;
    memset(g.p, 0xA5, g.data_bytes);
    return g;
}

static void g_free(gbuf *g)
{
    size_t ps = g_pagesz();
    if (munmap(g->p - ps, g->data_bytes + 2 * ps) != 0) {
        perror("munmap");
        exit(1);
    }
    g->p = NULL;
}

/* 1 iff [len, data_bytes) is still all poison */
static int g_poison_intact(const gbuf *g, size_t len)
{
    for (size_t i = len; i < g->data_bytes; i++)
        if (g->p[i] != 0xA5)
            return 0;
    return 1;
}

/* ---- vector length for a candidate ----
 * This must be the VL the JIT's main loop uses for its group (U*VL):
 * the boundary lengths are defined relative to the implemented seam. */
#if defined(__riscv)
/* RVV: the JIT sets vl = VLMAX once up front via vsetvli(t0, x0, e8, m1)
 * and the group is U*VLMAX. rj_cpu_info.rvv_vlen is a hardcoded 16-byte
 * fallback that does not track the actual VLEN (e.g. QEMU -cpu max,vlen=N),
 * so query the true runtime VLMAX directly. vsetvli touches no vector
 * registers; it only sets vl/vtype. Only called when the CPU has RVV
 * (plan_candidates only offers ISA_RVV then), so no SIGILL risk. */
static size_t rvv_vlmax_bytes(void)
{
    unsigned long vl = 0;
    __asm__ volatile(
        ".option push\n"
        ".option arch, +v\n"
        "vsetvli %0, x0, e8, m1, ta, mu\n"
        ".option pop"
        : "=r"(vl));
    return (size_t)vl;
}
#endif

#if defined(__aarch64__)
/* SVE: the JIT derives its group at runtime from cntb
 * (src/aarch64/jit.c: x13 = cntb * U), while cpu->sve_vl comes from
 * prctl(PR_SVE_GET_VL), which under-reports under qemu-user (e.g. 16
 * bytes reported while cntb says 128). Query the true VL directly.
 * Only called for SVE candidates, which require HWCAP_SVE, so cntb
 * cannot SIGILL. */
static size_t sve_vl_bytes(void)
{
    unsigned long vl = 0;
    __asm__ volatile(
        ".arch_extension sve\n"
        "cntb %0"
        : "=r"(vl));
    return (size_t)vl;
}
#endif

static size_t cand_vl(const cand_t *c, const rj_cpu_info *cpu)
{
    int v = isa_vl(c->isa);
    if (v > 0)
        return (size_t)v;
    if (c->isa == ISA_SVE) {
#if defined(__aarch64__)
        size_t probe = sve_vl_bytes();
        if (probe > 0)
            return probe;
#endif
        if (cpu->sve_vl > 0)
            return (size_t)cpu->sve_vl; /* bytes */
    }
    if (c->isa == ISA_RVV) {
#if defined(__riscv)
        size_t probe = rvv_vlmax_bytes();
        if (probe > 0)
            return probe;
#endif
        if (cpu->rvv_vlen > 0)
            return (size_t)cpu->rvv_vlen; /* VLEN in bytes */
    }
    return 0;
}

/* ---- one (plan, version) at one length ----
 * Buffers are pre-allocated for maxlen; this call poisons, fills [0,len),
 * runs the oracle + the JIT fn, and verifies outputs, input immutability,
 * and the poison region. */
static void run_one(const plan_t *pl, const version_t *v,
                    gbuf *in, gbuf *snap, gbuf *out, gbuf *exp,
                    const uint8_t **in_u8, const void **in_vp,
                    uint8_t **exp_u8, void **out_vp,
                    size_t len, const char *tag)
{
    int n_in = pl->n_in, n_out = pl->n_out;

    for (int i = 0; i < n_in; i++) {
        memset(in[i].p, 0xA5, in[i].data_bytes);
        rnd_bytes(in[i].p, len);
        memset(snap[i].p, 0xA5, snap[i].data_bytes);
        memcpy(snap[i].p, in[i].p, len);
        in_u8[i] = in[i].p;
        in_vp[i] = in[i].p;
    }
    for (int j = 0; j < n_out; j++) {
        memset(out[j].p, 0xA5, out[j].data_bytes);
        memset(exp[j].p, 0xA5, exp[j].data_bytes);
        exp_u8[j] = exp[j].p;
        out_vp[j] = out[j].p;
    }

    ref_run(pl, in_u8, exp_u8, len);
    v->fn(in_vp, out_vp, len);

    for (int j = 0; j < n_out; j++)
        CHECK(memcmp(out[j].p, exp[j].p, len) == 0,
              "%s [%s] len=%zu: out%d mismatch vs ref", tag, v->desc, len, j);
    for (int i = 0; i < n_in; i++) {
        CHECK(memcmp(in[i].p, snap[i].p, len) == 0,
              "%s [%s] len=%zu: input %d clobbered", tag, v->desc, len, i);
        CHECK(g_poison_intact(&in[i], len),
              "%s [%s] len=%zu: input %d overrun past len", tag, v->desc,
              len, i);
    }
    for (int j = 0; j < n_out; j++)
        CHECK(g_poison_intact(&out[j], len),
              "%s [%s] len=%zu: output %d overrun past len", tag, v->desc,
              len, j);
}

/* ---- all boundary lengths for one built candidate ----
 * Lengths: 0/1 (degenerate), VL-1/VL/VL+1 (sub-group), U-1/U/U+1 (RVV
 * strip-mining vs tail seam: the remainder path strips iff
 * remaining >= U), group-1/group/group+1 (main/remainder seam),
 * 2*group-1/2*group/2*group+1 (loop back-edge: the body runs twice). */
static void run_boundaries(const plan_t *pl, const version_t *v,
                           const cand_t *c, size_t vl, const char *tag)
{
    size_t U = (size_t)c->U;
    size_t group = U * vl;
    size_t lens[] = {
        0, 1,
        vl - 1, vl, vl + 1,
        U - 1, U, U + 1,
        group - 1, group, group + 1,
        2 * group - 1, 2 * group, 2 * group + 1,
    };
    size_t uniq[14];
    int nu = 0;
    for (size_t i = 0; i < sizeof(lens) / sizeof(lens[0]); i++) {
        int dup = 0;
        for (int k = 0; k < nu; k++)
            if (uniq[k] == lens[i])
                dup = 1;
        if (!dup)
            uniq[nu++] = lens[i];
    }

    int n_in = pl->n_in, n_out = pl->n_out;
    size_t maxlen = 2 * group + 1;
    gbuf *in = calloc((size_t)(n_in ? n_in : 1), sizeof(*in));
    gbuf *snap = calloc((size_t)(n_in ? n_in : 1), sizeof(*snap));
    gbuf *out = calloc((size_t)(n_out ? n_out : 1), sizeof(*out));
    gbuf *exp = calloc((size_t)(n_out ? n_out : 1), sizeof(*exp));
    const uint8_t **in_u8 =
        calloc((size_t)(n_in ? n_in : 1), sizeof(*in_u8));
    const void **in_vp = calloc((size_t)(n_in ? n_in : 1), sizeof(*in_vp));
    uint8_t **exp_u8 = calloc((size_t)(n_out ? n_out : 1), sizeof(*exp_u8));
    void **out_vp = calloc((size_t)(n_out ? n_out : 1), sizeof(*out_vp));
    if (!in || !snap || !out || !exp || !in_u8 || !in_vp || !exp_u8 || !out_vp) {
        printf("FAIL %s [%s]: oom\n", tag, v->desc);
        failures++;
        goto done;
    }
    for (int i = 0; i < n_in; i++) {
        in[i] = g_alloc(maxlen);
        snap[i] = g_alloc(maxlen);
    }
    for (int j = 0; j < n_out; j++) {
        out[j] = g_alloc(maxlen);
        exp[j] = g_alloc(maxlen);
    }

    for (int k = 0; k < nu && !failures; k++)
        run_one(pl, v, in, snap, out, exp, in_u8, in_vp, exp_u8, out_vp,
                uniq[k], tag);

    for (int i = 0; i < n_in; i++) {
        g_free(&in[i]);
        g_free(&snap[i]);
    }
    for (int j = 0; j < n_out; j++) {
        g_free(&out[j]);
        g_free(&exp[j]);
    }
done:
    free(in);
    free(snap);
    free(out);
    free(exp);
    free(in_u8);
    free(in_vp);
    free(exp_u8);
    free(out_vp);
}

/* ---- drive one (topology, op): enumerate candidates, filter U, run ----
 * U filter: {1, 2} plus each ISA's maximum offered unroll. The max unroll
 * is the widest group the planner can emit per ISA (8 on x86/NEON, 5 on
 * SVE/RVV); it must be boundary-tested, not just the small Us. Filtering
 * on the offered maximum (rather than a hardcoded set) keeps the check
 * valid if the planner's U tables change. */
static void drive(rj_topology *t, const rj_op *op, const char *tag,
                  const rj_cpu_info *cpu, int expect_kind /* -1: don't check */)
{
    char err[256];
    topo_int_t ti;
    CHECK(topo_int_build(&ti, t, err, sizeof(err)) == 0, "%s: topo: %s", tag,
          err);
    if (failures)
        return;
    plan_t pl;
    CHECK(plan_build(&ti, op, tag, &pl, err, sizeof(err)) == 0, "%s: plan: %s",
          tag, err);
    if (failures) {
        topo_int_free(&ti);
        return;
    }
    if (expect_kind >= 0)
        CHECK((int)pl.kind == expect_kind, "%s: expected plan kind %d, got %d",
              tag, expect_kind, (int)pl.kind);

    cand_t cands[256];
    int nc = plan_candidates(&pl, cpu, cands, 256, 0, 0);
    CHECK(nc > 0, "%s: no candidates", tag);

    /* Pass 1: per-ISA maximum offered U. */
    int isa_seen[7] = { 0, 0, 0, 0, 0, 0, 0 };
    int isa_maxu[7] = { 0, 0, 0, 0, 0, 0, 0 };
    for (int ci = 0; ci < nc; ci++) {
        cand_t *c = &cands[ci];
        if ((int)c->isa < 0 || (int)c->isa > 6)
            continue;
        isa_seen[c->isa] = 1;
        if (c->U > isa_maxu[c->isa])
            isa_maxu[c->isa] = c->U;
    }

    /* Pass 2: build & run U in {1, 2, max-per-ISA}. */
    int isa_maxu_hit[7] = { 0, 0, 0, 0, 0, 0, 0 };
    for (int ci = 0; ci < nc && !failures; ci++) {
        cand_t *c = &cands[ci];
        if ((int)c->isa < 0 || (int)c->isa > 6)
            continue;
        if (c->U != 1 && c->U != 2 && c->U != isa_maxu[c->isa])
            continue;
        size_t vl = cand_vl(c, cpu);
        if (!vl)
            continue;
        version_t v;
        rj_options o;
        rj_options_default(&o);
        o.benchmark = 0;
        memset(&v, 0, sizeof(v));
        if (jit_build_version(&pl, c, cpu, &o, &v, err, sizeof(err)) != 0) {
            CHECK(0, "%s: build isa=%d U=%d: %s", tag, c->isa, c->U, err);
            continue;
        }
        if (c->U == isa_maxu[c->isa])
            isa_maxu_hit[c->isa] = 1;
        run_boundaries(&pl, &v, c, vl, tag);
        jit_free_version(&v);
    }
    for (int i = 0; i < 7; i++)
        if (isa_seen[i] && !isa_maxu_hit[i])
            CHECK(0, "%s: isa %d offered max U=%d but it was never built",
                  tag, i, isa_maxu[i]);

    plan_free(&pl);
    topo_int_free(&ti);
    if (!failures)
        printf("ok %s\n", tag);
}

/* ---- cases ---- */
static void case_encode_raid5(const rj_cpu_info *cpu)
{
    rj_topology *t = rj_topo_raid5(4);
    CHECK(t != NULL, "rj_topo_raid5(4)");
    if (t) {
        rj_op op = rj_op_encode();
        drive(t, &op, "raid5(4) encode", cpu, P_XOR);
        rj_op_free(&op);
        rj_topo_free(t);
    }
}

static void case_encode_raid6(const rj_cpu_info *cpu)
{
    rj_topology *t = rj_topo_raid6(5);
    CHECK(t != NULL, "rj_topo_raid6(5)");
    if (t) {
        rj_op op = rj_op_encode();
        /* P_MULACC, not P_HORNER: rj_topo_raid6 emits the A2 2.0
         * searched matrix, which is non-Vandermonde, so
         * plan_select_strategy takes the multiply-accumulate branch
         * (deliberate 2.0 parity-format change). */
        drive(t, &op, "raid6(5) encode", cpu, P_MULACC);
        rj_op_free(&op);
        rj_topo_free(t);
    }
}

static void case_encode_rs63(const rj_cpu_info *cpu)
{
    rj_topology *t = rj_topo_rs(6, 3);
    CHECK(t != NULL, "rj_topo_rs(6,3)");
    if (t) {
        rj_op op = rj_op_encode();
        /* P_MULACC for the same reason as raid6(5) above: the searched
         * A2 2.0 matrix is non-Vandermonde. */
        drive(t, &op, "rs(6,3) encode", cpu, P_MULACC);
        rj_op_free(&op);
        rj_topo_free(t);
    }
}

static void case_encode_mulacc(const rj_cpu_info *cpu)
{
    /* Non-Vandermonde 4+2 matrix -> P_MULACC encode (the general
     * multiply-accumulate path, as opposed to the XOR/Horner fast paths).
     * The matrix need not be MDS: this test checks that the JIT implements
     * the plan's matrix product exactly at the loop/remainder seams, not
     * erasure-correctness. Row 1 is not a geometric progression, so
     * mat_is_horner_vandermonde() rejects it. */
    static const uint8_t mat[2 * 4] = {
        1, 1, 1, 1, /* P: xor */
        1, 2, 3, 5, /* Q: not geometric -> P_MULACC */
    };
    rj_topology *t = rj_topo_custom(4, 2, mat);
    CHECK(t != NULL, "rj_topo_custom(4,2)");
    if (t) {
        rj_op op = rj_op_encode();
        drive(t, &op, "custom(4,2) encode", cpu, P_MULACC);
        rj_op_free(&op);
        rj_topo_free(t);
    }
}

static void case_recover(const rj_cpu_info *cpu, rj_topology *t, const char *tname,
                         const int *failed, int nfailed)
{
    char tag[128];
    snprintf(tag, sizeof(tag), "%s recover{", tname);
    for (int f = 0; f < nfailed; f++)
        snprintf(tag + strlen(tag), sizeof(tag) - strlen(tag), "%s%d",
                 f ? "," : "", failed[f]);
    snprintf(tag + strlen(tag), sizeof(tag) - strlen(tag), "}");
    rj_op op = rj_op_recover(failed, nfailed);
    drive(t, &op, tag, cpu, -1);
    rj_op_free(&op);
}

static void case_pdelta(const rj_cpu_info *cpu, rj_topology *t, const char *tname,
                        const int *changed, int nchanged)
{
    char tag[128];
    snprintf(tag, sizeof(tag), "%s pdelta{", tname);
    for (int k = 0; k < nchanged; k++)
        snprintf(tag + strlen(tag), sizeof(tag) - strlen(tag), "%s%d",
                 k ? "," : "", changed[k]);
    snprintf(tag + strlen(tag), sizeof(tag) - strlen(tag), "}");
    rj_op op = rj_op_parity_update(changed, nchanged);
    drive(t, &op, tag, cpu, P_DELTA);
    rj_op_free(&op);
}

int main(void)
{
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);

    case_encode_raid5(&cpu);
    case_encode_raid6(&cpu);
    case_encode_rs63(&cpu);
    case_encode_mulacc(&cpu);

    {
        rj_topology *t = rj_topo_raid6(5);
        CHECK(t != NULL, "rj_topo_raid6(5)");
        if (t) {
            int f01[2] = { 0, 1 };
            int f0[1] = { 0 };
            case_recover(&cpu, t, "raid6(5)", f01, 2);
            case_recover(&cpu, t, "raid6(5)", f0, 1);
            rj_topo_free(t);
        }
    }
    {
        rj_topology *t = rj_topo_raid5(4);
        CHECK(t != NULL, "rj_topo_raid5(4)");
        if (t) {
            int f2[1] = { 2 };
            int c0[1] = { 0 };
            case_recover(&cpu, t, "raid5(4)", f2, 1);
            case_pdelta(&cpu, t, "raid5(4)", c0, 1);
            rj_topo_free(t);
        }
    }
    {
        rj_topology *t = rj_topo_raid6(5);
        CHECK(t != NULL, "rj_topo_raid6(5)");
        if (t) {
            int c13[2] = { 1, 3 };
            case_pdelta(&cpu, t, "raid6(5)", c13, 2);
            rj_topo_free(t);
        }
    }

    if (!failures)
        printf("test_boundaries: all OK\n");
    else
        printf("test_boundaries: %d FAILURES\n", failures);
    return failures != 0;
}
