/* Permanent regression test for the software-pipelined (double-buffered)
 * loop twin (track I8).
 *
 * Force-builds pipe candidates via jit_build_version() -- bypassing the
 * planner's twin policy -- and validates the jitted output against the
 * scalar reference (ref_run) and against the non-pipelined twin, on
 * canaried buffers. Lengths span 0 .. 2*group+7 plus a fixed ladder and
 * large odd sizes, covering: empty (prologue never runs), sub-group
 * (prologue never runs), exactly one group, group+1 (the boundary guard
 * blocks the next-iteration over-read), two groups (exactly one
 * boundary load), and many groups.
 *
 * Cases (per binary's ARCH; the ISA-specific ones skip without the ISA).
 * All force-built configs fit the register file
 * (plan_need_regs(...,pipe=1) <= isa_nregs, checked in-test):
 *  AArch64 (QEMU): custom Vandermonde(4,1) = P_HORNER, n_out=1, U=2,
 *    NEON (needs (1+2)*2+4+2 = 12 <= 16); custom Vandermonde(6,3),
 *    n_out=3, U=2, SVE (needs (3+2)*2+4+6 = 20 <= 24); custom
 *    P_MULACC with a dead column (exercises the column-skip in
 *    pipe_used_columns), n_out=2, U=1, NEON (needs (2+2)*1+7 = 11).
 *  RVV (QEMU): custom Vandermonde(6,3), U=2 (needs (3+2)*2+5+6 = 21
 *    <= 32); all-trivial Horner (c==1 fast path); custom P_MULACC
 *    dead-column, U=2 (needs (2+2)*2+7 = 15).
 *  x86-64 (native): custom Vandermonde(4,1), U=2, AVX512 (needs
 *    (1+2)*2+5+2 = 13 <= 32; force-built -- the planner gate is closed
 *    on x86-64 pending quiet-box benchmarks; AVX512 because AVX2
 *    HORNER has a pre-existing codegen issue unrelated to pipe);
 *    custom P_MULACC dead-column, U=1, AVX2 (needs (2+2)*1+6 = 10).
 * (rj_topo_rs now emits non-Vandermonde matrices via
 * matrix_search_default, i.e. P_MULACC, so the HORNER cases use the
 * explicit Vandermonde construction below.)
 *
 * Planner checks (portable, run on every ARCH):
 *  - the pipe twin is enumerated exactly where the static policy says:
 *    for each fitting base candidate (NEON/SVE/RVV, P_MULACC/P_HORNER,
 *    MUL_NIBBLE, no other twin flags) with
 *    plan_need_regs(...,pipe=1)*lmul <= isa_nregs(isa). Verified on a
 *    custom Vandermonde(6,3) (P_HORNER) and a custom P_MULACC plan.
 *  - NOT enumerated where it would spill: a NEON MULACC plan with
 *    n_out=2 at U=8 needs (2+2)*8+6 = 38 > 16 regs -> no pipe twin.
 *  - NOT enumerated on x86-64 (ISA gate closed): no pipe twin in
 *    plan_candidates for ISA_AVX512/AVX2/SSE.
 *  - over-pressure: force-building a pipe candidate that needs more
 *    regs than the file must fail cleanly (return -1), not abort.
 *
 * Over-read tripwire: one case mmaps the input buffers with a PROT_NONE
 * guard page immediately after len bytes and runs exactly one full group
 * (len == group). The boundary load of a miscompiled guard would fault
 * reading [len, len+group); a SIGSEGV handler converts the fault into a
 * FAIL instead of a crash. This is the sharp edge of the feature: the
 * boundary guard is the only thing standing between the last iteration
 * and an out-of-bounds read.
 *
 * Non-tautological: the differential pipe-vs-ref checks fail on any
 * codegen divergence; the planner enumeration checks fail if the gate
 * is widened (spill case) or narrowed (fit case); the guard-page case
 * fails (via SIGSEGV) if the boundary guard is removed. Verified by
 * mutation during development: with the boundary guard forced to
 * always-load, the guard-page case faults as designed.
 */
#include "../include/raidjit.h"
#include "../src/plan.h"
#include "../src/jit.h"
#include "../src/cpu.h"
#include "../src/gf.h"
#include "../src/ref.h"

#include <setjmp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

static const size_t fixed_lens[] = { 0, 1, 7, 15, 16, 17, 31, 32, 33,
                                     63, 64, 65, 100, 255, 256, 257,
                                     1000, 4096, 5000 };
static const int nfixed = sizeof(fixed_lens) / sizeof(fixed_lens[0]);

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

#define CANARY 64
static const uint8_t canary_byte = 0xA5;

/* Allocate len bytes with CANARY canary bytes on each side; returns the
 * interior pointer (free with canary_free). */
static uint8_t *canary_alloc(size_t len)
{
    uint8_t *p = abuf(len + 2 * CANARY);
    memset(p, canary_byte, CANARY);
    memset(p + CANARY + len, canary_byte, CANARY);
    return p + CANARY;
}

static void canary_free(uint8_t *interior)
{
    free(interior - CANARY);
}

static int canary_ok(uint8_t *interior, size_t len)
{
    for (int i = 0; i < CANARY; i++)
        if (interior[-CANARY + i] != canary_byte ||
            interior[len + i] != canary_byte)
            return 0;
    return 1;
}

/* Build one candidate (pipe flag per arg) and check it builds. */
static int build_one(const plan_t *pl, isa_t isa, mul_t mul, int U,
                     int pipe, const rj_cpu_info *cpu, const rj_options *o,
                     version_t *v, const char *name)
{
    char err[256];
    cand_t cand;
    memset(&cand, 0, sizeof cand);
    cand.isa = isa;
    cand.mul = mul;
    cand.U = U;
    cand.pipe = pipe;
    int rc = jit_build_version(pl, &cand, cpu, o, v, err, sizeof err);
    CHECK(rc == 0, "%s pipe=%d build: %s", name, pipe, err);
    return rc == 0;
}

/* Differential: pipe kernel vs scalar ref and vs the non-pipe twin, on
 * canaried buffers, over the fixed ladder plus group-relative lengths.
 * expect_kind pins the plan kind (P_HORNER for RS, P_MULACC for the
 * custom topology). */
static void test_pipe_case(const char *name, rj_topology *t, isa_t isa,
                           mul_t mul, int U, plan_kind_t expect_kind,
                           int main_loop_only)
{
    char err[256];
    topo_int_t ti;
    if (topo_int_build(&ti, t, err, sizeof err) != 0) {
        CHECK(0, "%s topo: %s", name, err);
        return;
    }
    rj_op op = rj_op_encode();
    plan_t pl;
    if (plan_build(&ti, &op, name, &pl, err, sizeof err) != 0) {
        CHECK(0, "%s plan: %s", name, err);
        rj_op_free(&op);
        topo_int_free(&ti);
        return;
    }
    CHECK(pl.kind == expect_kind, "%s: kind %d, want %d", name, pl.kind,
          expect_kind);

    int need = plan_need_regs(isa, mul, &pl, U, 0, 1, 0, 0);
    CHECK(need <= isa_nregs(isa), "%s: pipe needs %d regs > %d", name,
          need, isa_nregs(isa));
    if (failures) {
        plan_free(&pl);
        rj_op_free(&op);
        topo_int_free(&ti);
        return;
    }

    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    rj_options o;
    rj_options_default(&o);
    version_t vp, vn;
    int okp = build_one(&pl, isa, mul, U, 1, &cpu, &o, &vp, name);
    int okn = build_one(&pl, isa, mul, U, 0, &cpu, &o, &vn, name);
    if (!okp || !okn) {
        if (okp)
            jit_free_version(&vp);
        if (okn)
            jit_free_version(&vn);
        plan_free(&pl);
        rj_op_free(&op);
        topo_int_free(&ti);
        return;
    }
    size_t group = vp.group_bytes;
    CHECK(group > 0, "%s: group_bytes=%zu", name, group);

    /* Lengths: the fixed ladder plus group-relative sharp edges. */
    size_t lens[64];
    int nl = 0;
    if (main_loop_only) {
        /* Pre-existing x86 HORNER tail bug (non-pipe also crashes);
         * test the main loop (the pipe twin's domain) only. */
        lens[nl++] = 0;
        lens[nl++] = group;
        lens[nl++] = 2 * group;
        lens[nl++] = 3 * group + 5 - ((3 * group + 5) % group);
    } else {
        for (int i = 0; i < nfixed && nl < 64; i++)
            lens[nl++] = fixed_lens[i];
        size_t rel[] = { group > 1 ? group - 1 : 0, group, group + 1,
                         2 * group > 1 ? 2 * group - 1 : 0, 2 * group,
                         2 * group + 1, 2 * group + 7, 3 * group + 5 };
        for (size_t i = 0; i < sizeof(rel) / sizeof(rel[0]) && nl < 64;
             i++) {
            int dup = 0;
            for (int k = 0; k < nl; k++)
                if (lens[k] == rel[i])
                    dup = 1;
            if (!dup)
                lens[nl++] = rel[i];
        }
    }

    int nd = ti.ndata, np = ti.npar;
    for (int li = 0; li < nl; li++) {
        size_t len = lens[li];
        uint8_t **d = malloc(sizeof(*d) * (size_t)nd);
        uint8_t **pp = malloc(sizeof(*pp) * (size_t)np);
        uint8_t **pn = malloc(sizeof(*pn) * (size_t)np);
        uint8_t **pr = malloc(sizeof(*pr) * (size_t)np);
        for (int i = 0; i < nd; i++) {
            d[i] = canary_alloc(len);
            randfill(d[i], len);
        }
        for (int j = 0; j < np; j++) {
            pp[j] = canary_alloc(len);
            pn[j] = canary_alloc(len);
            pr[j] = abuf(len);
            randfill(pp[j], len);
            memcpy(pn[j], pp[j], len);
        }
        ref_run(&pl, (const uint8_t **)d, pr, len);
        const void **in = malloc(sizeof(*in) * (size_t)nd);
        void **outp = malloc(sizeof(*outp) * (size_t)np);
        void **outn = malloc(sizeof(*outn) * (size_t)np);
        for (int i = 0; i < nd; i++)
            in[i] = d[i];
        for (int j = 0; j < np; j++) {
            outp[j] = pp[j];
            outn[j] = pn[j];
        }
        vp.fn(in, outp, len);
        vn.fn(in, outn, len);
        for (int j = 0; j < np; j++) {
            CHECK(memcmp(pp[j], pr[j], len) == 0,
                  "%s len=%zu par%d: pipe output != ref", name, len, j);
            CHECK(memcmp(pn[j], pr[j], len) == 0,
                  "%s len=%zu par%d: non-pipe output != ref", name, len,
                  j);
            CHECK(memcmp(pp[j], pn[j], len) == 0,
                  "%s len=%zu par%d: pipe != non-pipe", name, len, j);
        }
        for (int i = 0; i < nd; i++) {
            CHECK(canary_ok(d[i], len), "%s len=%zu: input canary %d "
                  "clobbered", name, len, i);
            canary_free(d[i]);
        }
        for (int j = 0; j < np; j++) {
            CHECK(canary_ok(pp[j], len), "%s len=%zu: pipe output canary "
                  "%d clobbered", name, len, j);
            CHECK(canary_ok(pn[j], len), "%s len=%zu: non-pipe output "
                  "canary %d clobbered", name, len, j);
            canary_free(pp[j]);
            canary_free(pn[j]);
            free(pr[j]);
        }
        free(d);
        free(pp);
        free(pn);
        free(pr);
        free(in);
        free(outp);
        free(outn);
        if (failures)
            break;
    }
    jit_free_version(&vp);
    jit_free_version(&vn);
    plan_free(&pl);
    rj_op_free(&op);
    topo_int_free(&ti);
    if (!failures)
        printf("ok pipe %s (need=%d regs)\n", name, need);
}

/* Guard-page over-read tripwire: mmap input buffers with a PROT_NONE
 * page immediately after len bytes, run exactly one full group
 * (len == group). A miscompiled boundary guard (unconditional
 * next-iteration load) reads [len, len+group) and faults; the SIGSEGV
 * handler converts the fault into a FAIL. */
static sigjmp_buf segv_jb;
static volatile sig_atomic_t segv_armed = 0;

static void segv_handler(int sig)
{
    (void)sig;
    if (segv_armed)
        siglongjmp(segv_jb, 1);
}

/* Call v->fn under a SIGSEGV guard. Returns 1 if the call faulted, 0
 * otherwise. Kept as a separate function so no caller locals are live
 * across the sigsetjmp (-Wclobbered). */
static int call_guarded(version_t *v, const void **in, void **out,
                        size_t len)
{
    struct sigaction sa, old_sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = segv_handler;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, &old_sa);
    segv_armed = 1;
    int r = sigsetjmp(segv_jb, 1);
    if (r == 0)
        v->fn(in, out, len);
    segv_armed = 0;
    sigaction(SIGSEGV, &old_sa, NULL);
    return r != 0;
}

static void test_guard_page(const char *name, rj_topology *t, isa_t isa,
                            mul_t mul, int U, plan_kind_t expect_kind)
{
    char err[256];
    topo_int_t ti;
    if (topo_int_build(&ti, t, err, sizeof err) != 0) {
        CHECK(0, "%s topo: %s", name, err);
        return;
    }
    rj_op op = rj_op_encode();
    plan_t pl;
    if (plan_build(&ti, &op, name, &pl, err, sizeof err) != 0) {
        CHECK(0, "%s plan: %s", name, err);
        rj_op_free(&op);
        topo_int_free(&ti);
        return;
    }
    if (pl.kind != expect_kind) {
        CHECK(0, "%s: kind %d, want %d", name, pl.kind, expect_kind);
        plan_free(&pl);
        rj_op_free(&op);
        topo_int_free(&ti);
        return;
    }
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    rj_options o;
    rj_options_default(&o);
    version_t v;
    if (!build_one(&pl, isa, mul, U, 1, &cpu, &o, &v, name)) {
        plan_free(&pl);
        rj_op_free(&op);
        topo_int_free(&ti);
        return;
    }
    size_t group = v.group_bytes;
    size_t pagesz = (size_t)sysconf(_SC_PAGESIZE);

    int nd = ti.ndata, np = ti.npar;
    uint8_t **d = NULL;
    uint8_t **p = NULL;
    uint8_t **pr = NULL;
    const void **in = NULL;
    void **out = NULL;
    /* Keep the mmap bases for munmap. */
    uint8_t **dbase = NULL;
    /* Place the len==group buffer at the end of a page, with the next
     * page PROT_NONE. A boundary-guard over-read of [len, len+group)
     * then faults immediately. group < pagesz is required (true: group
     * is U*VL, at most a few hundred bytes). */
    CHECK(group < pagesz, "%s: group=%zu >= pagesz", name, group);
    if (group >= pagesz)
        goto out;

    d = malloc(sizeof(*d) * (size_t)nd);
    dbase = malloc(sizeof(*dbase) * (size_t)nd);
    p = malloc(sizeof(*p) * (size_t)np);
    pr = malloc(sizeof(*pr) * (size_t)np);
    for (int i = 0; i < nd; i++) {
        d[i] = NULL;
        dbase[i] = NULL;
    }
    for (int j = 0; j < np; j++) {
        p[j] = NULL;
        pr[j] = NULL;
    }
    for (int i = 0; i < nd; i++) {
        /* Two pages: first readable, second PROT_NONE. Data sits at
         * the end of the first page so [len, len+group) hits the
         * guard. */
        uint8_t *map = mmap(NULL, 2 * pagesz, PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (map == MAP_FAILED) {
            CHECK(0, "%s: mmap failed", name);
            goto out;
        }
        if (mprotect(map + pagesz, pagesz, PROT_NONE) != 0) {
            CHECK(0, "%s: mprotect failed", name);
            munmap(map, 2 * pagesz);
            goto out;
        }
        dbase[i] = map;
        d[i] = map + pagesz - group; /* len == group at page end */
        randfill(d[i], group);
    }
    for (int j = 0; j < np; j++) {
        p[j] = abuf(group);
        pr[j] = abuf(group);
        randfill(p[j], group);
    }
    ref_run(&pl, (const uint8_t **)d, pr, group);
    in = malloc(sizeof(*in) * (size_t)nd);
    out = malloc(sizeof(*out) * (size_t)np);
    for (int i = 0; i < nd; i++)
        in[i] = d[i];
    for (int j = 0; j < np; j++)
        out[j] = p[j];

    int faulted = call_guarded(&v, in, out, group);
    CHECK(!faulted, "%s: pipe kernel faulted on guard page (len=group=%"
          "zu): boundary guard over-reads", name, group);
    if (!faulted)
        for (int j = 0; j < np; j++)
            CHECK(memcmp(p[j], pr[j], group) == 0,
                  "%s: guard-page run par%d != ref", name, j);

out:
    for (int i = 0; i < nd; i++)
        if (dbase[i])
            munmap(dbase[i], 2 * pagesz);
    free(dbase);
    for (int j = 0; j < np; j++) {
        free(p[j]);
        free(pr[j]);
    }
    free(d);
    free(p);
    free(pr);
    free(in);
    free(out);
    jit_free_version(&v);
    plan_free(&pl);
    rj_op_free(&op);
    topo_int_free(&ti);
    if (!failures)
        printf("ok pipe guard-page %s (group=%zu)\n", name, group);
}

/* A custom ndata x npar Vandermonde topology in "Horner order" (the
 * same construction rj_topo_rs used historically): M[j][i] =
 * g^((j+1)*(ndata-1-i)), g=2, so horner_c[j] = 2^(j+1) and every row
 * is nontrivial. plan_build must see P_HORNER. (rj_topo_rs now emits
 * non-Vandermonde matrices via matrix_search_default, i.e. P_MULACC,
 * so the HORNER cases need this explicit construction.) */
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

/* A custom P_MULACC topology with a dead (all-zero) column: exercises
 * the column-skip in the pipe bodies. Matrix (n_in=4, n_out=2):
 *   row0: 5 0 7 0
 *   row1: 3 0 9 0
 * columns 1 and 3 are all-zero -> skipped by pipe_used_columns. */
static rj_topology *topo_mulacc_deadcol(void)
{
    uint8_t m[8] = { 5, 0, 7, 0, 3, 0, 9, 0 };
    return rj_topo_custom(4, 2, m);
}

#if defined(__x86_64__) || defined(__aarch64__)
/* A MULACC plan for spill/over-pressure tests: n_out=2. */
static rj_topology *topo_mulacc_wide(void)
{
    uint8_t m[16];
    for (int i = 0; i < 16; i++)
        m[i] = (uint8_t)(3 + (i % 5));
    return rj_topo_custom(8, 2, m);
}
#endif

static int has_pipe_twin(cand_t *cands, int n, isa_t isa, mul_t mul,
                         int U, int lmul)
{
    for (int i = 0; i < n; i++)
        if (cands[i].pipe && cands[i].isa == isa && cands[i].mul == mul &&
            cands[i].U == U &&
            (cands[i].lmul ? cands[i].lmul : 1) == lmul)
            return 1;
    return 0;
}

/* Planner gate checks (portable): the pipe twin is enumerated exactly
 * where the static policy says, and nowhere else. */
static void test_planner_pipe(void)
{
    char err[256];
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    cand_t cands[1024];

    /* Custom Vandermonde(6,3) is P_HORNER, n_out=3. For each ISA the
     * pipe twin must be present for every fitting base (MULACC/HORNER
     * x MUL_NIBBLE, no other twins) with
     * plan_need_regs(...,1)*lmul <= isa_nregs, and absent otherwise. */
    rj_topology *t63 = topo_vandermonde(6, 3);
    CHECK(t63 != NULL, "topo_vandermonde(6,3)");
    if (t63) {
        topo_int_t ti;
        CHECK(topo_int_build(&ti, t63, err, sizeof err) == 0, "topo: %s",
              err);
        rj_op op = rj_op_encode();
        plan_t pl;
        CHECK(plan_build(&ti, &op, "v(6,3)", &pl, err, sizeof err) == 0,
              "plan: %s", err);
        CHECK(pl.kind == P_HORNER, "expected P_HORNER, got %d", pl.kind);
        int n = plan_candidates(&pl, &cpu, cands, 1024, 0, 0);
        CHECK(n > 0, "no candidates");
        /* For every enumerated base candidate, the pipe twin must
         * exist iff the policy admits it. */
        int nbase = n;
        for (int i = 0; i < nbase; i++) {
            cand_t *b = &cands[i];
            if (b->pipe)
                continue;
            if (pl.kind != P_MULACC && pl.kind != P_HORNER)
                continue;
            if (b->mul != MUL_NIBBLE)
                continue;
            if (b->ternlog || b->permi2b || b->remat || b->xorsched ||
                b->xorbit || b->mshift)
                continue;
            int lmul = (b->isa == ISA_RVV && b->lmul) ? b->lmul : 1;
            int want;
            if (b->isa != ISA_NEON && b->isa != ISA_SVE &&
                b->isa != ISA_RVV)
                want = 0; /* x86-64 gate closed */
            else
                want = plan_need_regs(b->isa, b->mul, &pl, b->U, 0, 1, 0, 0) *
                           lmul <=
                       isa_nregs(b->isa);
            int found = has_pipe_twin(cands, n, b->isa, b->mul, b->U,
                                      lmul);
            /* nt/prefetch twins of the pipe twin share (isa,mul,U);
             * has_pipe_twin only needs one witness. */
            CHECK(found == want,
                  "rs(6,3) %s U=%d lmul=%d: pipe twin %s, policy says %s",
                  isa_name(b->isa), b->U, lmul,
                  found ? "enumerated" : "absent",
                  want ? "enumerate" : "skip");
        }
        /* x86-64: the ISA gate is closed -- no pipe twin at all. */
        for (int i = 0; i < n; i++)
            CHECK(!cands[i].pipe ||
                      (cands[i].isa != ISA_AVX512 &&
                       cands[i].isa != ISA_AVX2 && cands[i].isa != ISA_SSE),
                  "x86-64 pipe twin enumerated (gate closed)");
        plan_free(&pl);
        rj_op_free(&op);
        topo_int_free(&ti);
        rj_topo_free(t63);
    }

    /* NEON spill case: MULACC n_out=2. At U=3 the base needs
     * 2*3+8 = 14 (fits), but the pipe twin needs (2+2)*3+7 = 19 >
     * 16: the pipe twin must NOT be enumerated. (AArch64-only: the
     * planner doesn't enumerate NEON candidates on other ISAs.) */
#if defined(__aarch64__)
    rj_topology *tw = topo_mulacc_wide();
    CHECK(tw != NULL, "topo_mulacc_wide");
    if (tw) {
        topo_int_t ti;
        CHECK(topo_int_build(&ti, tw, err, sizeof err) == 0, "topo: %s",
              err);
        rj_op op = rj_op_encode();
        plan_t pl;
        CHECK(plan_build(&ti, &op, "wide", &pl, err, sizeof err) == 0,
              "plan: %s", err);
        CHECK(pl.kind == P_MULACC, "expected P_MULACC, got %d", pl.kind);
        int need_base = plan_need_regs(ISA_NEON, MUL_NIBBLE, &pl, 3, 0, 0, 0, 0);
        int need_pipe = plan_need_regs(ISA_NEON, MUL_NIBBLE, &pl, 3, 0, 1, 0, 0);
        CHECK(need_base == 14, "wide U=3 base need=%d, want 14", need_base);
        CHECK(need_pipe == 19, "wide U=3 pipe need=%d, want 19", need_pipe);
        int n = plan_candidates(&pl, &cpu, cands, 1024, 0, 0);
        /* The base must exist (it fits). */
        int base_found = 0;
        for (int i = 0; i < n; i++)
            if (!cands[i].pipe && cands[i].isa == ISA_NEON &&
                cands[i].U == 3)
                base_found = 1;
        CHECK(base_found, "NEON U=3 base missing (should fit)");
        /* The pipe twin must NOT be enumerated (would spill). */
        for (int i = 0; i < n; i++)
            CHECK(!(cands[i].pipe && cands[i].isa == ISA_NEON &&
                    cands[i].U == 3),
                  "NEON U=3 pipe twin enumerated despite 19 > 16 regs");
        plan_free(&pl);
        rj_op_free(&op);
        topo_int_free(&ti);
        rj_topo_free(tw);
    }
#endif

    /* Over-pressure: force-building a pipe candidate past the regfile
     * must fail cleanly (return -1), not abort. Uses the ISA valid for
     * the current architecture. */
#if defined(__x86_64__)
    /* AVX2 has 16 regs; MULACC n_out=2 U=4 pipe needs (2+2)*4+6 = 22. */
    rj_topology *tw2 = topo_mulacc_wide();
    if (tw2) {
        topo_int_t ti;
        if (topo_int_build(&ti, tw2, err, sizeof err) == 0) {
            rj_op op = rj_op_encode();
            plan_t pl;
            if (plan_build(&ti, &op, "wide", &pl, err, sizeof err) ==
                0) {
                CHECK(pl.kind == P_MULACC, "expected P_MULACC, got %d",
                      pl.kind);
                int need =
                    plan_need_regs(ISA_AVX2, MUL_NIBBLE, &pl, 4, 0, 1, 0, 0);
                CHECK(need == 22, "wide AVX2 U=4 pipe need=%d, want 22",
                      need);
                rj_options o;
                rj_options_default(&o);
                cand_t cand;
                memset(&cand, 0, sizeof cand);
                cand.isa = ISA_AVX2;
                cand.mul = MUL_NIBBLE;
                cand.U = 4;
                cand.pipe = 1;
                version_t v;
                int rc = jit_build_version(&pl, &cand, &cpu, &o, &v, err,
                                           sizeof err);
                CHECK(rc != 0, "over-pressure pipe build succeeded "
                      "(want clean fail)");
                if (rc == 0)
                    jit_free_version(&v);
                plan_free(&pl);
            }
            rj_op_free(&op);
            topo_int_free(&ti);
        }
        rj_topo_free(tw2);
    }
#elif defined(__aarch64__)
    /* NEON has 16 regs; MULACC n_out=2 U=3 pipe needs (2+2)*3+7 = 19. */
    rj_topology *tw2 = topo_mulacc_wide();
    if (tw2) {
        topo_int_t ti;
        if (topo_int_build(&ti, tw2, err, sizeof err) == 0) {
            rj_op op = rj_op_encode();
            plan_t pl;
            if (plan_build(&ti, &op, "wide", &pl, err, sizeof err) ==
                0) {
                int need =
                    plan_need_regs(ISA_NEON, MUL_NIBBLE, &pl, 3, 0, 1, 0, 0);
                CHECK(need == 19, "wide NEON U=3 pipe need=%d, want 19",
                      need);
                rj_options o;
                rj_options_default(&o);
                cand_t cand;
                memset(&cand, 0, sizeof cand);
                cand.isa = ISA_NEON;
                cand.mul = MUL_NIBBLE;
                cand.U = 3;
                cand.pipe = 1;
                version_t v;
                int rc = jit_build_version(&pl, &cand, &cpu, &o, &v, err,
                                           sizeof err);
                CHECK(rc != 0, "over-pressure pipe build succeeded "
                      "(want clean fail)");
                if (rc == 0)
                    jit_free_version(&v);
                plan_free(&pl);
            }
            rj_op_free(&op);
            topo_int_free(&ti);
        }
        rj_topo_free(tw2);
    }
#endif

    if (!failures)
        printf("ok pipe planner gate\n");
}

/* Regression test (2026-10-07): pipe+predtail must fail gracefully,
 * never crash. The predicated tail reuses the non-pipelined gen_body,
 * whose register layout (cg->t live) is dead in pipe mode (cg->t = -1);
 * before the fix, force-building pipe+predtail tripped the V()
 * register assert (SIGABRT). I8's pipe twin can win heuristic_select,
 * and api.c built the predtail twin from the winner unconditionally,
 * so this was reachable via rj_compile, not just the doc tool.
 * Non-tautological: without the backend's graceful rejection this test
 * aborts the process instead of failing a CHECK. */
static void test_pipe_predtail_reject(void)
{
#if defined(__aarch64__)
    char err[256];
    rj_topology *t = rj_topo_raid6(5);
    topo_int_t ti;
    if (topo_int_build(&ti, t, err, sizeof err) != 0) {
        CHECK(0, "pipe+predtail topo: %s", err);
        rj_topo_free(t);
        return;
    }
    rj_op op = rj_op_encode();
    plan_t pl;
    if (plan_build(&ti, &op, "pipe-predtail-reject", &pl, err,
                   sizeof err) != 0) {
        CHECK(0, "pipe+predtail plan: %s", err);
        rj_op_free(&op);
        topo_int_free(&ti);
        rj_topo_free(t);
        return;
    }
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    if (!cpu.sve) {
        printf("SKIP pipe+predtail: no SVE on this CPU\n");
        plan_free(&pl);
        rj_op_free(&op);
        topo_int_free(&ti);
        rj_topo_free(t);
        return;
    }
    rj_options o;
    rj_options_default(&o);
    cand_t cand;
    memset(&cand, 0, sizeof cand);
    cand.isa = ISA_SVE;
    cand.mul = MUL_NIBBLE;
    cand.U = 4;
    cand.pipe = 1;
    cand.predtail = 1;
    version_t v;
    memset(&v, 0, sizeof v);
    int rc = jit_build_version(&pl, &cand, &cpu, &o, &v, err, sizeof err);
    /* Must fail gracefully (rc != 0 with an explanatory err), never
     * abort. A successful build would also be acceptable if the
     * combination is ever implemented, but today it must reject. */
    CHECK(rc != 0, "pipe+predtail unexpectedly built (no rejection)");
    if (rc != 0)
        CHECK(strstr(err, "pipe") != NULL,
              "pipe+predtail err should name pipe, got: %s", err);
    else
        jit_free_version(&v);
    plan_free(&pl);
    rj_op_free(&op);
    topo_int_free(&ti);
    rj_topo_free(t);
#else
    printf("SKIP pipe+predtail: aarch64-only\n");
#endif
}

int main(void)
{
#if defined(__aarch64__)
    {
        rj_topology *t = topo_vandermonde(4, 1);
        test_pipe_case("a64-v41-horner-neon-u2", t, ISA_NEON, MUL_NIBBLE,
                       2, P_HORNER, 0);
        test_guard_page("a64-v41-horner-neon-u2", t, ISA_NEON, MUL_NIBBLE,
                        2, P_HORNER);
        rj_topo_free(t);
    }
    {
        rj_topology *t = topo_vandermonde(6, 3);
        test_pipe_case("a64-v63-horner-sve-u2", t, ISA_SVE, MUL_NIBBLE,
                       2, P_HORNER, 0);
        test_guard_page("a64-v63-horner-sve-u2", t, ISA_SVE, MUL_NIBBLE,
                        2, P_HORNER);
        rj_topo_free(t);
    }
    {
        rj_topology *t = topo_mulacc_deadcol();
        test_pipe_case("a64-mulacc-deadcol-neon-u1", t, ISA_NEON,
                       MUL_NIBBLE, 1, P_MULACC, 0);
        rj_topo_free(t);
    }
#elif defined(__riscv)
    {
        rj_topology *t = topo_vandermonde(6, 3);
        test_pipe_case("rvv-v63-horner-u2", t, ISA_RVV, MUL_NIBBLE, 2,
                       P_HORNER, 0);
        test_guard_page("rvv-v63-horner-u2", t, ISA_RVV, MUL_NIBBLE, 2,
                        P_HORNER);
        rj_topo_free(t);
    }
    {
        /* All-trivial Horner: force horner_c to 1 (mat all-ones) to
         * exercise the c==1 fast path in the pipe body. */
        rj_topology *t = topo_vandermonde(6, 3);
        topo_int_t ti;
        char err[256];
        if (topo_int_build(&ti, t, err, sizeof err) == 0) {
            rj_op op = rj_op_encode();
            plan_t pl;
            if (plan_build(&ti, &op, "trivial", &pl, err, sizeof err) ==
                0 && pl.kind == P_HORNER) {
                for (int j = 0; j < pl.n_out; j++)
                    pl.horner_c[j] = 1;
                memset(pl.mat, 1,
                       (size_t)pl.n_out * (size_t)pl.n_in);
                /* Build directly from the hand-modified plan. */
                rj_cpu_info cpu;
                rj_cpu_detect(&cpu);
                rj_options o;
                rj_options_default(&o);
                version_t vp, vn;
                int okp =
                    build_one(&pl, ISA_RVV, MUL_NIBBLE, 2, 1, &cpu, &o,
                              &vp, "rvv-trivial");
                int okn =
                    build_one(&pl, ISA_RVV, MUL_NIBBLE, 2, 0, &cpu, &o,
                              &vn, "rvv-trivial");
                if (okp && okn) {
                    size_t len = 1000;
                    uint8_t *d0 = abuf(len), *d1 = abuf(len);
                    uint8_t *d2 = abuf(len), *d3 = abuf(len);
                    uint8_t *d4 = abuf(len), *d5 = abuf(len);
                    uint8_t *pp[3] = { abuf(len), abuf(len), abuf(len) };
                    uint8_t *pr[3] = { abuf(len), abuf(len), abuf(len) };
                    const uint8_t *dd[6] = { d0, d1, d2, d3, d4, d5 };
                    for (int i = 0; i < 6; i++)
                        randfill((void *)dd[i], len);
                    ref_run(&pl, dd, pr, len);
                    const void *in[6] = { d0, d1, d2, d3, d4, d5 };
                    void *out[3] = { pp[0], pp[1], pp[2] };
                    vp.fn(in, out, len);
                    for (int j = 0; j < 3; j++)
                        CHECK(memcmp(pp[j], pr[j], len) == 0,
                              "rvv-trivial len=1000 par%d != ref", j);
                    for (int i = 0; i < 6; i++)
                        free((void *)dd[i]);
                    for (int j = 0; j < 3; j++) {
                        free(pp[j]);
                        free(pr[j]);
                    }
                    printf("ok pipe rvv-trivial (c==1 fast path)\n");
                    jit_free_version(&vp);
                    jit_free_version(&vn);
                }
                plan_free(&pl);
            }
            rj_op_free(&op);
            topo_int_free(&ti);
        }
        rj_topo_free(t);
    }
    {
        rj_topology *t = topo_mulacc_deadcol();
        test_pipe_case("rvv-mulacc-deadcol-u2", t, ISA_RVV, MUL_NIBBLE, 2,
                       P_MULACC, 0);
        rj_topo_free(t);
    }
#elif defined(__x86_64__)
    {
        /* x86-64: force-built (the planner gate is closed pending
         * quiet-box benchmarks); differential correctness only.
         * HORNER on AVX512 (AVX2 HORNER has a pre-existing codegen
         * issue, unrelated to the pipe twin; AVX512 is used here). */
        rj_topology *t = topo_vandermonde(4, 1);
        test_pipe_case("x86-v41-horner-avx512-u2", t, ISA_AVX512,
                       MUL_NIBBLE, 2, P_HORNER, 1);
        test_guard_page("x86-v41-horner-avx512-u2", t, ISA_AVX512,
                        MUL_NIBBLE, 2, P_HORNER);
        rj_topo_free(t);
    }
    {
        rj_topology *t = topo_mulacc_deadcol();
        test_pipe_case("x86-mulacc-deadcol-avx2-u1", t, ISA_AVX2,
                       MUL_NIBBLE, 1, P_MULACC, 0);
        rj_topo_free(t);
    }
#else
    printf("SKIP: no pipe case for this ARCH\n");
#endif
    test_planner_pipe();
    test_pipe_predtail_reject();

    if (failures == 0)
        printf("test_pipe: all OK\n");
    else
        printf("test_pipe: %d FAILURES\n", failures);
    return failures != 0;
}
