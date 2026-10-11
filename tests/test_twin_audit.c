/* Twin x Twin audit (escape-protocol artifact).
 *
 * Origin (2026-10-07): I8's pipe twin won heuristic_select, and api.c
 * built the predtail tail-twin from the winner unconditionally. The
 * aarch64 backend had no pipe+predtail path: the predicated tail reuses
 * the non-pipelined gen_body whose register layout is dead in pipe mode,
 * so the build tripped a V() register assert -> SIGABRT. Reachable via
 * plain rj_compile, not just the doc tool. Fixed in 4f1e69c (backend
 * rejects cleanly; api.c/gen_kernels skip the twin for pipe winners).
 *
 * The class: any twin axis the planner can set on a winning candidate
 * (plan_candidates: nt, prefetch_dist, ternlog, remat, xorsched, xorbit,
 * mshift, lmul, pipe) crossed with any twin api.c builds from the
 * winner (notail, predtail, maskedtail, permi2b). The backend must build
 * or reject gracefully (rc != 0) -- never abort. The backends encode
 * "the planner never emits this" as assert(); every such assert is a
 * load-bearing claim about the planner, and this test checks the
 * reachable half of the claim empirically.
 *
 * What runs by default (fast, CI-safe): for each of the four plan
 * kinds, take plan_candidates' ACTUAL outputs on a synthetic
 * fully-featured CPU and, for each, build the base plus base x
 * {notail, predtail, maskedtail, permi2b} -- exactly api.c's stage-2
 * construction. An abort fails the test loudly; the last fflush'd
 * "building ..." line names the killer combo. Non-tautological:
 * reverting 4f1e69c's api.c skip makes the pipe x predtail case abort
 * the test process (verified by the original bug report).
 *
 * Full mode (TWIN_AUDIT_FULL=1): the complete twin cross product per
 * plan kind, fork-isolated per combination, printing a
 * built/rejected/aborted table. Combinations that abort are
 * assert-trips: each must be shown unreachable from the planner (the
 * documented twin-exclusion pairs) or fixed like 4f1e69c. Slow
 * (~10-20 min per backend); for audits, not CI.
 */
#include "../include/raidjit.h"
#include "../src/plan.h"
#include "../src/jit.h"
#include "../src/cpu.h"
#include "../src/gf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
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

enum { OC_BUILT = 0, OC_REJECTED = 1, OC_ABORTED = 2 };

/* Synthetic fully-featured CPU for this backend's arch (the
 * gen_kernels.c pattern): plan_candidates then emits every ISA the
 * backend implements, independent of host features. */
static void fill_cpu(rj_cpu_info *cpu)
{
    memset(cpu, 0, sizeof(*cpu));
#if defined(__x86_64__)
    cpu->sse2 = cpu->ssse3 = 1;
    cpu->avx = cpu->avx2 = 1;
    cpu->avx512f = cpu->avx512bw = cpu->avx512vl = 1;
    cpu->gfni = 1;
    cpu->vbmi = 1;
    snprintf(cpu->brand, sizeof(cpu->brand), "audit-avx512");
#elif defined(__aarch64__)
    cpu->neon = 1;
    cpu->sve = 1;
    cpu->sve2 = 1;
    cpu->sve_vl = 16;
    snprintf(cpu->brand, sizeof(cpu->brand), "audit-sve2");
#elif defined(__riscv)
    cpu->rvv = 1;
    cpu->rvv_vlen = 16;
    cpu->rvv_zvbc32e = 1;
    cpu->rvv_zicbop = 1;
    snprintf(cpu->brand, sizeof(cpu->brand), "audit-rvv");
#endif
}

/* Human-readable twin combo, e.g. "nt+pipe+notail lmul=2 pf=64". */
static void describe_cand(const cand_t *c, char *buf, size_t n)
{
    size_t w = 0;
#define FLAG(f, name)                                                      \
    do {                                                                   \
        if (c->f)                                                          \
            w += (size_t)snprintf(buf + w, n > w ? n - w : 0,               \
                                  "%s%s", w ? "+" : "", name);             \
    } while (0)
    FLAG(nt, "nt");
    FLAG(ternlog, "ternlog");
    FLAG(remat, "remat");
    FLAG(notail, "notail");
    FLAG(predtail, "predtail");
    FLAG(maskedtail, "maskedtail");
    FLAG(permi2b, "permi2b");
    FLAG(xorsched, "xorsched");
    FLAG(xorbit, "xorbit");
    FLAG(mshift, "mshift");
    FLAG(pipe, "pipe");
#undef FLAG
    if (c->prefetch_dist)
        w += (size_t)snprintf(buf + w, n > w ? n - w : 0, "%spf=%zu",
                              w ? "+" : "", c->prefetch_dist);
    if (c->lmul)
        w += (size_t)snprintf(buf + w, n > w ? n - w : 0, "%slmul=%d",
                              w ? "+" : "", c->lmul);
    if (w == 0)
        snprintf(buf, n, "(plain)");
}

/* Fork-isolated single build. Returns OC_BUILT / OC_REJECTED /
 * OC_ABORTED. The child never returns: it _exits. */
static int build_isolated(const plan_t *pl, const cand_t *cand,
                          const rj_cpu_info *cpu)
{
    fflush(stdout);
    fflush(stderr);
    pid_t pid = fork();
    if (pid < 0) {
        printf("FAIL: fork failed\n");
        return OC_ABORTED;
    }
    if (pid == 0) {
        rj_options o;
        rj_options_default(&o);
        o.benchmark = 0;
        version_t v;
        memset(&v, 0, sizeof(v));
        char err[256];
        int rc = jit_build_version(pl, cand, cpu, &o, &v, err, sizeof(err));
        if (rc == 0)
            jit_free_version(&v);
        _exit(rc == 0 ? OC_BUILT : OC_REJECTED);
    }
    int status = 0;
    while (waitpid(pid, &status, 0) < 0)
        ;
    if (WIFSIGNALED(status))
        return OC_ABORTED;
    if (WIFEXITED(status))
        return WEXITSTATUS(status);
    return OC_ABORTED;
}

/* Direct (non-fork) build for the reachable sweep: an abort here is
 * the test failing loudly, with the combo on the last log line. */
static int build_direct(const plan_t *pl, const cand_t *cand,
                        const rj_cpu_info *cpu, const char *desc)
{
    printf("building %s\n", desc);
    fflush(stdout);
    rj_options o;
    rj_options_default(&o);
    o.benchmark = 0;
    version_t v;
    memset(&v, 0, sizeof(v));
    char err[256];
    int rc = jit_build_version(pl, cand, cpu, &o, &v, err, sizeof(err));
    if (rc == 0)
        jit_free_version(&v);
    return rc;
}

static rj_topology *topo_vandermonde(int ndata, int npar)
{
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

struct plan_case {
    const char *name;
    plan_kind_t want_kind;
    rj_topology *topo; /* owned by caller */
    rj_op op;          /* owned by caller */
};

static int build_plan_cases(struct plan_case *cases, char *err,
                            size_t errlen)
{
    int n = 0;
    cases[n].topo = rj_topo_raid5(4);
    cases[n].op = rj_op_encode();
    cases[n].name = "P_XOR raid5(4)";
    cases[n].want_kind = P_XOR;
    n++;
    {
        uint8_t m[8] = {5, 7, 3, 9, 2, 4, 6, 8};
        cases[n].topo = rj_topo_custom(4, 2, m);
    }
    cases[n].op = rj_op_encode();
    cases[n].name = "P_MULACC custom(4,2)";
    cases[n].want_kind = P_MULACC;
    n++;
    cases[n].topo = topo_vandermonde(4, 1);
    cases[n].op = rj_op_encode();
    cases[n].name = "P_HORNER v(4,1)";
    cases[n].want_kind = P_HORNER;
    n++;
    {
        int ch[1] = {0};
        cases[n].topo = rj_topo_raid6(5);
        cases[n].op = rj_op_parity_update(ch, 1);
        cases[n].name = "P_DELTA raid6(5) ch={0}";
        cases[n].want_kind = P_DELTA;
        n++;
    }
    for (int i = 0; i < n; i++) {
        if (!cases[i].topo) {
            snprintf(err, errlen, "topo alloc failed for %s",
                     cases[i].name);
            return -1;
        }
    }
    return n;
}

static int build_one_plan(const struct plan_case *c, plan_t *pl,
                          topo_int_t *ti, char *err, size_t errlen)
{
    if (topo_int_build(ti, c->topo, err, errlen) != 0)
        return -1;
    if (plan_build(ti, &c->op, c->name, pl, err, errlen) != 0) {
        topo_int_free(ti);
        return -1;
    }
    return 0;
}

/* api.c stage-2 eligibility, mirrored here so the sweep constructs
 * exactly what rj_compile can construct. If api.c's gates change, this
 * must change with them (the full cross-product audit is the tool that
 * catches a missing gate; this sweep locks in the current contract). */
static int eligible_permi2b(const cand_t *base, const plan_t *pl,
                            const rj_cpu_info *cpu)
{
    return base->isa == ISA_AVX512 &&
           (pl->kind == P_MULACC || pl->kind == P_DELTA) &&
           base->mul == MUL_NIBBLE && !base->permi2b && cpu->vbmi &&
           plan_any_nontrivial(pl);
}

static int eligible_predtail(const cand_t *base)
{
    return base->isa == ISA_SVE && !base->pipe;
}

static int eligible_maskedtail(const cand_t *base, const plan_t *pl,
                               const rj_cpu_info *cpu)
{
    if (base->isa != ISA_AVX512 || base->nt)
        return 0;
    if (!cpu->avx512f || !cpu->avx512bw)
        return 0;
    if (pl->kind == P_XOR && !base->ternlog && !base->xorsched)
        return 1;
    if (base->mul == MUL_GFNI && cpu->gfni &&
        (pl->kind == P_MULACC || pl->kind == P_DELTA ||
         pl->kind == P_HORNER))
        return 1;
    return 0;
}

/* Reachable sweep: plan_candidates' real outputs x api.c's stage-2
 * twin constructions (with api.c's eligibility gates). Every
 * combination must build or reject gracefully -- never abort. */
static void sweep_reachable(void)
{
    char err[512];
    rj_cpu_info cpu;
    fill_cpu(&cpu);
    struct plan_case cases[4];
    int ncases = build_plan_cases(cases, err, sizeof(err));
    CHECK(ncases == 4, "plan cases: %s", err);
    if (ncases != 4)
        return;

#if defined(__x86_64__)
    size_t pf = 64;
#elif defined(__aarch64__)
    size_t pf = 64; /* multiple of 8 */
#elif defined(__riscv)
    size_t pf = 64; /* multiple of 32, in [32,2016] */
#else
    size_t pf = 0;
#endif

    long nbuilt = 0, nrejected = 0;
    for (int ci = 0; ci < ncases; ci++) {
        topo_int_t ti;
        plan_t pl;
        if (build_one_plan(&cases[ci], &pl, &ti, err, sizeof(err)) != 0) {
            CHECK(0, "plan %s: %s", cases[ci].name, err);
            continue;
        }
        if (pl.kind != cases[ci].want_kind)
            printf("note: %s built as kind %d (expected %d)\n",
                   cases[ci].name, (int)pl.kind,
                   (int)cases[ci].want_kind);
        cand_t cands[1024];
        int nc = plan_candidates(&pl, &cpu, cands, 1024, 1, pf);
        printf("plan %s: %d candidates\n", cases[ci].name, nc);
        for (int i = 0; i < nc; i++) {
            cand_t base = cands[i];
            char desc[256], full[320];
            /* The planner's own output must build (it was just
             * enumerated as viable). */
            describe_cand(&base, desc, sizeof(desc));
            snprintf(full, sizeof(full), "[%s isa=%d] %s", cases[ci].name,
                     (int)base.isa, desc);
            CHECK(build_direct(&pl, &base, &cpu, full) == 0,
                  "planner base failed to build: %s", full);
            nbuilt++;
            /* api.c stage-2 constructions from this winner, with
             * api.c's eligibility gates. Each must not abort;
             * graceful rejection is acceptable. */
            cand_t t;
            t = base;
            t.notail = 1;
            describe_cand(&t, desc, sizeof(desc));
            snprintf(full, sizeof(full), "[%s isa=%d] %s", cases[ci].name,
                     (int)base.isa, desc);
            if (build_direct(&pl, &t, &cpu, full) == 0)
                nbuilt++;
            else
                nrejected++;
            if (eligible_predtail(&base)) {
                t = base;
                t.predtail = 1;
                describe_cand(&t, desc, sizeof(desc));
                snprintf(full, sizeof(full), "[%s isa=%d] %s",
                         cases[ci].name, (int)base.isa, desc);
                if (build_direct(&pl, &t, &cpu, full) == 0)
                    nbuilt++;
                else
                    nrejected++;
            }
            if (eligible_maskedtail(&base, &pl, &cpu)) {
                t = base;
                t.maskedtail = 1;
                describe_cand(&t, desc, sizeof(desc));
                snprintf(full, sizeof(full), "[%s isa=%d] %s",
                         cases[ci].name, (int)base.isa, desc);
                if (build_direct(&pl, &t, &cpu, full) == 0)
                    nbuilt++;
                else
                    nrejected++;
            }
            if (eligible_permi2b(&base, &pl, &cpu)) {
                t = base;
                t.permi2b = 1;
                describe_cand(&t, desc, sizeof(desc));
                snprintf(full, sizeof(full), "[%s isa=%d] %s",
                         cases[ci].name, (int)base.isa, desc);
                if (build_direct(&pl, &t, &cpu, full) == 0)
                    nbuilt++;
                else
                    nrejected++;
            }
        }
        plan_free(&pl);
        topo_int_free(&ti);
    }
    for (int i = 0; i < ncases; i++) {
        rj_op_free(&cases[i].op);
        rj_topo_free(cases[i].topo);
    }
    printf("reachable sweep: %ld built, %ld gracefully rejected, "
           "0 aborted\n", nbuilt, nrejected);
}

/* Full cross-product audit (TWIN_AUDIT_FULL=1). Fork-isolated per
 * combination; prints the built/rejected/aborted table. */
static void sweep_full(void)
{
    char err[512];
    rj_cpu_info cpu;
    fill_cpu(&cpu);
    struct plan_case cases[4];
    int ncases = build_plan_cases(cases, err, sizeof(err));
    if (ncases != 4) {
        printf("FAIL: plan cases: %s\n", err);
        failures++;
        return;
    }
#if defined(__x86_64__)
    size_t pf = 64;
#elif defined(__aarch64__)
    size_t pf = 64;
#elif defined(__riscv)
    size_t pf = 64;
#else
    size_t pf = 0;
#endif
    /* Per-backend ISA under test. */
#if defined(__x86_64__)
    static const isa_t isas[] = {ISA_AVX512, ISA_AVX2};
#elif defined(__aarch64__)
    static const isa_t isas[] = {ISA_SVE, ISA_NEON};
#elif defined(__riscv)
    static const isa_t isas[] = {ISA_RVV};
#else
    static const isa_t isas[] = {(isa_t)-1};
#endif
    static const int lmuls[] = {0, 1, 2, 4, 8};

    long n_built = 0, n_rejected = 0, n_aborted = 0;
    for (int ci = 0; ci < ncases; ci++) {
        topo_int_t ti;
        plan_t pl;
        if (build_one_plan(&cases[ci], &pl, &ti, err, sizeof(err)) != 0) {
            printf("FAIL: plan %s: %s\n", cases[ci].name, err);
            failures++;
            continue;
        }
        for (size_t ii = 0; ii < sizeof(isas) / sizeof(isas[0]); ii++) {
            for (size_t li = 0;
                 li < sizeof(lmuls) / sizeof(lmuls[0]); li++) {
                /* 12 boolean twin axes. */
                for (int bits = 0; bits < 4096; bits++) {
                    cand_t c;
                    memset(&c, 0, sizeof(c));
                    c.isa = isas[ii];
                    c.mul = MUL_NIBBLE;
                    c.U = 2;
                    c.nt = (bits >> 0) & 1;
                    c.prefetch_dist = ((bits >> 1) & 1) ? pf : 0;
                    c.ternlog = (bits >> 2) & 1;
                    c.remat = (bits >> 3) & 1;
                    c.notail = (bits >> 4) & 1;
                    c.predtail = (bits >> 5) & 1;
                    c.maskedtail = (bits >> 6) & 1;
                    c.permi2b = (bits >> 7) & 1;
                    c.xorsched = (bits >> 8) & 1;
                    c.xorbit = (bits >> 9) & 1;
                    c.mshift = (bits >> 10) & 1;
                    c.pipe = (bits >> 11) & 1;
                    c.lmul = lmuls[li];
                    char desc[256];
                    describe_cand(&c, desc, sizeof(desc));
                    int oc = build_isolated(&pl, &c, &cpu);
                    if (oc == OC_BUILT)
                        n_built++;
                    else if (oc == OC_REJECTED)
                        n_rejected++;
                    else {
                        n_aborted++;
                        printf("ABORT [%s isa=%d] %s\n", cases[ci].name,
                               (int)c.isa, desc);
                    }
                }
            }
        }
        plan_free(&pl);
        topo_int_free(&ti);
    }
    for (int i = 0; i < ncases; i++) {
        rj_op_free(&cases[i].op);
        rj_topo_free(cases[i].topo);
    }
    printf("full sweep: %ld built, %ld rejected, %ld aborted\n", n_built,
           n_rejected, n_aborted);
    if (n_aborted)
        failures++;
}

/* Targeted audit: the "one extra twin" and exclusion-pair space. The
 * reachable sweep covers planner outputs x api.c twins; this covers
 * the defense-in-depth space: every body twin x every tail twin (the
 * pipe+predtail class), every documented body-twin exclusion pair, and
 * cross-ISA twins. Each must build or reject gracefully -- an abort
 * here means a backend assert on a combo that needs a reachability
 * verdict (see the audit notes). Fast enough for QEMU. */
static void sweep_targeted(void)
{
    char err[512];
    rj_cpu_info cpu;
    fill_cpu(&cpu);
    struct plan_case cases[4];
    int ncases = build_plan_cases(cases, err, sizeof(err));
    CHECK(ncases == 4, "plan cases: %s", err);
    if (ncases != 4)
        return;

    /* (plan_kind, flags) pairs to probe. Flags use the bit positions
     * from sweep_full. kind: 0=P_XOR, 1=P_MULACC, 2=P_HORNER,
     * 3=P_DELTA. */
    static const struct {
        int kind;
        isa_t isa;
        unsigned bits;
        const char *label;
    } probes[] = {
        /* pipe+predtail class (the original SIGABRT) */
        {0, ISA_SVE, (1u << 11) | (1u << 5), "pipe+predtail"},
        {0, ISA_AVX512, (1u << 11) | (1u << 6), "pipe+maskedtail"},
        {0, ISA_SVE, (1u << 11) | (1u << 4), "pipe+notail"},
        {2, ISA_SVE, (1u << 11) | (1u << 5), "pipe+predtail"},
        {2, ISA_AVX512, (1u << 11) | (1u << 6), "pipe+maskedtail"},
        /* body twin x tail twin */
        {0, ISA_AVX512, (1u << 9) | (1u << 5), "xorbit+predtail"},
        {0, ISA_AVX512, (1u << 9) | (1u << 6), "xorbit+maskedtail"},
        {0, ISA_AVX512, (1u << 10) | (1u << 5), "mshift+predtail"},
        {0, ISA_AVX512, (1u << 10) | (1u << 6), "mshift+maskedtail"},
        {1, ISA_AVX512, (1u << 7) | (1u << 5), "permi2b+predtail"},
        {1, ISA_AVX512, (1u << 7) | (1u << 6), "permi2b+maskedtail"},
        {0, ISA_AVX512, (1u << 2) | (1u << 6), "ternlog+maskedtail"},
        {0, ISA_SVE, (1u << 8) | (1u << 5), "xorsched+predtail"},
        {2, ISA_SVE, (1u << 3) | (1u << 4), "remat+notail"},
        {2, ISA_SVE, (1u << 3) | (1u << 5), "remat+predtail"},
        /* tail twin pairs (must stay disjoint) */
        {0, ISA_SVE, (1u << 4) | (1u << 5), "notail+predtail"},
        {0, ISA_AVX512, (1u << 4) | (1u << 6), "notail+maskedtail"},
        {0, ISA_SVE, (1u << 5) | (1u << 6), "predtail+maskedtail"},
        /* body-twin exclusion pairs */
        {0, ISA_AVX512, (1u << 9) | (1u << 2), "xorbit+ternlog"},
        {0, ISA_AVX512, (1u << 9) | (1u << 7), "xorbit+permi2b"},
        {0, ISA_AVX512, (1u << 9) | (1u << 8), "xorbit+xorsched"},
        {0, ISA_AVX512, (1u << 10) | (1u << 8), "mshift+xorsched"},
        {0, ISA_AVX512, (1u << 10) | (1u << 9), "mshift+xorbit"},
        {1, ISA_AVX512, (1u << 7) | (1u << 8), "permi2b+xorsched"},
        {2, ISA_SVE, (1u << 3) | (1u << 8), "remat+xorsched"},
        /* cross-ISA twins (backend must reject gracefully) */
        {0, ISA_AVX512, (1u << 5), "predtail-on-avx512"},
        {0, ISA_SVE, (1u << 6), "maskedtail-on-sve"},
        {1, ISA_SVE, (1u << 7), "permi2b-on-sve"},
        /* remat on non-HORNER / xorsched without schedule */
        {0, ISA_AVX512, (1u << 3), "remat-on-XOR"},
        {1, ISA_AVX512, (1u << 3), "remat-on-MULACC"},
        {0, ISA_AVX512, (1u << 8), "xorsched"},
        /* RVV: pipe x notail (riscv has no predtail/maskedtail) */
        {0, ISA_RVV, (1u << 11) | (1u << 4), "rvv-pipe+notail"},
        {2, ISA_RVV, (1u << 11) | (1u << 4), "rvv-pipe+notail"},
        {0, ISA_RVV, (1u << 8) | (1u << 4), "rvv-xorsched+notail"},
    };

    /* Map probe kind to plan_case index: cases[] are P_XOR, P_MULACC,
     * P_HORNER, P_DELTA in order (see build_plan_cases). */
    long nbuilt = 0, nrejected = 0;
    for (int ci = 0; ci < ncases; ci++) {
        topo_int_t ti;
        plan_t pl;
        if (build_one_plan(&cases[ci], &pl, &ti, err, sizeof(err)) != 0) {
            printf("FAIL: plan %s: %s\n", cases[ci].name, err);
            failures++;
            continue;
        }

        for (size_t qi = 0; qi < sizeof(probes) / sizeof(probes[0]);
             qi++) {
            if (probes[qi].kind != ci)
                continue;
            cand_t t;
            memset(&t, 0, sizeof(t));
            t.isa = probes[qi].isa;
            t.U = 4;
            t.mul = MUL_NIBBLE;
            unsigned b = probes[qi].bits;
            if (b & (1u << 0))
                t.nt = 1;
            if (b & (1u << 2))
                t.ternlog = 1;
            if (b & (1u << 3))
                t.remat = 1;
            if (b & (1u << 4))
                t.notail = 1;
            if (b & (1u << 5))
                t.predtail = 1;
            if (b & (1u << 6))
                t.maskedtail = 1;
            if (b & (1u << 7))
                t.permi2b = 1;
            if (b & (1u << 8))
                t.xorsched = 1;
            if (b & (1u << 9))
                t.xorbit = 1;
            if (b & (1u << 10))
                t.mshift = 1;
            if (b & (1u << 11))
                t.pipe = 1;
            printf("targeted [%s] %s\n", cases[ci].name,
                   probes[qi].label);
            int rc = build_isolated(&pl, &t, &cpu);
            if (rc == OC_BUILT)
                nbuilt++;
            else if (rc == OC_REJECTED)
                nrejected++;
            else {
                printf("FAIL: targeted [%s] %s aborted (rc=%d)\n",
                       cases[ci].name, probes[qi].label, rc);
                failures++;
            }
        }
        plan_free(&pl);
        topo_int_free(&ti);
    }
    for (int i = 0; i < ncases; i++) {
        rj_op_free(&cases[i].op);
        rj_topo_free(cases[i].topo);
    }
    printf("targeted sweep: %ld built, %ld gracefully rejected, "
           "0 aborted\n", nbuilt, nrejected);
    fflush(stdout);
}

int main(void)
{
    const char *full = getenv("TWIN_AUDIT_FULL");
    const char *targ = getenv("TWIN_AUDIT_TARGETED");
    if (full && full[0] == '1')
        sweep_full();
    else if (targ && targ[0] == '1')
        sweep_targeted();
    else
        sweep_reachable();
    if (!failures)
        printf("ok twin_audit\n");
    return failures != 0;
}
