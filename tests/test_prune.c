/* test_prune: static-dominance pruning of the benchmark candidate list.
 *
 * Part 1 (unit, deterministic, no wall clock): synthetic version lists
 * with known (kind, isa, mul, U, nt, pf, tl, uops/B, mem/B, code)
 * exercise the dominance rule directly. The rule prunes ONLY the plain
 * member of a +ternlog twin pair (same plan kind, isa/mul/U/nt/pf/
 * remat, tl 1 vs 0) when the twin pareto-dominates it on the triple,
 * and only for plan kinds P_MULACC / P_DELTA (kind guard in
 * twin_dominates):
 *   - twin prunes plain (P_MULACC shape); order stable; multiple pairs
 *   - P_XOR twin (plain pareto-better on the triple) SURVIVES, and a
 *     P_XOR twin that DOES dominate on the triple also SURVIVES: the
 *     kind guard, not the triple values, enforces the direction
 *   - cross-U, cross-nt, cross-ISA, cross-mul, cross-remat, cross-kind
 *     pairs never prune
 *   - prefetch: only same-prefetch-dist pairs interact
 *   - twin that is not pareto-better does not prune (triple gate)
 *   - a plain whose twin failed to build (twin absent) survives
 *   - eps ties never prune; empty/singleton safe
 *   - RJ_NO_PRUNE / PRUNE_DISABLE disable pruning
 * Fail-capable: every assertion pins a specific dominance outcome.
 *
 * Part 4 (merge §5b, prune x xorsched non-interference): pins the C13
 * verification -- adversarial P_XOR triples (naive / xsched twin /
 * +ternlog twin with the +ternlog twin pareto-dominating the xsched
 * twin) prune nothing (kind-guard refusal); the mem-asymmetry pin (a
 * mem-worse twin never dominates, independent of the kind guard); and
 * the notail/xorsched field pinning in twin_dominates.
 *
 * Design history (2026-10-02): a BROADER rule (triple dominance within
 * the same (isa,mul), across U) was implemented first and REJECTED by
 * Parts 2+3 below -- it pruned real winners (P_XOR U=4/U=5, the P_XOR
 * twin at 1 MiB, P_DELTA AVX2). The failure mode: for P_XOR the model
 * reports U-invariant uops/B, so U=1 "dominated" U=8..U=2 on code size
 * alone. The rule was narrowed to twin-only prune-plain for
 * P_MULACC/P_DELTA; the rejection is documented in
 * docs/candidate_pruning.md. Hardening (2026-10-02, reviewer+attacker
 * findings): the kind restriction is now enforced by CODE (the kind
 * guard in twin_dominates), so a future model/codegen change that
 * makes a P_XOR twin triple-dominant cannot silently start pruning;
 * the Part-3 argmin tripwire below is the loud detector.
 *
 * Part 2 (end-to-end smoke, wall clock): rj_compile via the production
 * path (api.c calls prune_versions before bench_select) must succeed
 * with pruning enabled and with RJ_NO_PRUNE=1, over the topology x
 * length matrix, and each winner is checked against the scalar
 * reference. Winner equality across two independent compilations is
 * deliberately NOT asserted: benchmark noise flips winners between
 * runs even with pruning disabled, so that assertion would be flaky
 * by construction. Deterministic winner agreement is Part 3.
 *
 * Part 2b (hook wiring): with verbose=1 the api.c prune hook prints
 * "raidjit: prune: dropped ..." to stderr when it prunes; hook removal
 * makes the line disappear and fails the test (negative control:
 * RJ_NO_PRUNE=1 also removes the line). stderr is captured via a pipe.
 *
 * Part 3 (replay, deterministic given measured times): bench_select on
 * the full list gives real measured times (recovered from gbps); then
 * (a) select_by_time on the full vs pruned lists must agree (across
 * several noise windows), (b) the full run's winner must be in the
 * pruned set, and (c) TRIPWIRE: the UNPRUNED run's argmin-time
 * candidate must NOT be in the pruned set. (c) is the loud detector:
 * pruning safety reduces to "no pruned version is the argmin-time
 * candidate" (removing a pruned argmin re-centers select_by_time's
 * noise window and flips winners); any future model/codegen change
 * that breaks the precondition fails here instead of flipping winners
 * silently. Deterministic: both selections see identical measured
 * times, so noise cannot fake a pass or a failure.
 *
 * Mutation evidence (run 2026-10-02, all caught; logs committed under
 * tests/prune_mutations/):
 *  M1: twin_dominates() without the U equality (cross-U pruning) ->
 *      unit_cross_u_twin_no_prune FAILS: the U=1 twin pareto-dominates
 *      the U=8 plain on the U-invariant triple and prunes it.
 *      (Corrected 2026-10-02: the earlier claim that M1 makes
 *      unit_u_variants_survive fail was WRONG -- re-running Part 1
 *      with M1 applied passes there, because that fixture's code
 *      sizes grow with U so no cross-U dominance exists. The
 *      dedicated P_XOR-like U-invariant fixture is what catches M1.
 *      During design, the broader rule this mutation approximates
 *      pruned real P_XOR U=4/U=5 winners.)
 *  M2: twin_dominates() symmetric in tl (prune twin-in-favor-of-plain)
 *      -> unit_mulacc_plain_dominant_twin_survives FAILS: the twin is
 *      pruned in favor of the triple-dominant plain. (On the pre-kind-
 *      guard code this instead made unit_pxor_twin_survives fail and
 *      flipped xor2/1M in Part 3; the kind guard now contains that
 *      direction for P_XOR, so the M2 fixture uses P_MULACC.)
 *  M3: twin_dominates() without the triple gate (prune plain whenever
 *      a same-class twin exists) -> unit_twin_not_pareto_survives FAILS.
 *  M4: prune_versions() prunes the twin instead of the plain (inverted
 *      keep) -> 12 unit checks FAIL, starting with unit_twin_prunes_plain
 *      (every twin-pruning unit test breaks; Part 3 shows no flips on
 *      this corpus because the winners are GFNI, not nibble twins --
 *      the unit tests are the M4 detector).
 *  M5: api.c hook removed (no prune call) -> Part 2 still passes (safe
 *      by design: pruning only removes); hook_wiring (Part 2b) FAILS:
 *      the verbose prune line disappears. Not a correctness mutation,
 *      but the wiring is now pinned.
 */
#include "../include/raidjit.h"
#include "../src/plan.h"
#include "../src/jit.h"
#include "../src/bench.h"
#include "../src/prune.h"
#include "../src/cpu.h"
#include "../src/gf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int failures = 0;
static int nchecks = 0;
#define CHECK(cond, ...)                                                   \
    do {                                                                   \
        nchecks++;                                                         \
        if (!(cond)) {                                                     \
            printf("FAIL %s:%d: ", __func__, __LINE__);                    \
            printf(__VA_ARGS__);                                           \
            printf("\n");                                                  \
            failures++;                                                    \
        }                                                                  \
    } while (0)

/* ---------------- Part 1: unit tests on synthetic versions ---------------- */

/* Build a synthetic version_t. mem/code/uops are the static cost model;
 * the version is never executed (mem == NULL), so jit_free_version() on
 * a pruned synthetic is a safe no-op. kind is set explicitly: the kind
 * guard means pruning can only fire for P_MULACC/P_DELTA. */
static void mkv(version_t *v, plan_kind_t kind, isa_t isa, mul_t mul, int U,
                int nt, size_t pf, int tl, double uops, double mem,
                size_t code, const char *desc)
{
    memset(v, 0, sizeof(*v));
    v->kind = kind;
    v->cand.isa = isa;
    v->cand.mul = mul;
    v->cand.U = U;
    v->cand.nt = nt;
    v->cand.prefetch_dist = pf;
    v->cand.ternlog = tl;
    v->uops_per_byte = uops;
    v->mem_per_byte = mem;
    v->code_size = code;
    snprintf(v->desc, sizeof(v->desc), "%s", desc);
}

/* index of desc in vs[0..nv), or -1 */
static int find_desc(version_t *vs, int nv, const char *desc)
{
    for (int i = 0; i < nv; i++)
        if (strcmp(vs[i].desc, desc) == 0)
            return i;
    return -1;
}

/* P_MULACC shape: the +ternlog twin (fewer uops, same mem, smaller
 * code) dominates plain; plain is pruned, twin survives. Mirrors the
 * real rs(6,3) rec triples (1.346/32.016/5440 vs 1.502/32.016/5824).
 * This is the ONLY pruning the rule performs. */
static void unit_twin_prunes_plain(void)
{
    version_t vs[2];
    mkv(&vs[0], P_MULACC, ISA_AVX512, MUL_NIBBLE, 8, 0, 0, 0, 1.502,
        32.016, 5824, "plain");
    mkv(&vs[1], P_MULACC, ISA_AVX512, MUL_NIBBLE, 8, 0, 0, 1, 1.346,
        32.016, 5440, "twin");
    int n = prune_versions(vs, 2, 0);
    CHECK(n == 1, "want 1 survivor, got %d", n);
    CHECK(find_desc(vs, n, "twin") == 0, "twin must survive");
    CHECK(find_desc(vs, n, "plain") < 0, "plain must be pruned");
}

/* P_XOR shape: plain pareto-dominates the twin on the triple, but the
 * rule NEVER prunes a twin in favor of plain -- the P_XOR twin won the
 * 1 MiB benchmark (dense 4+2 recovery, 2026-10-02), so static dominance
 * in this direction is not trustworthy. Both survive. */
static void unit_pxor_twin_survives(void)
{
    version_t vs[2];
    mkv(&vs[0], P_XOR, ISA_AVX512, MUL_NIBBLE, 4, 0, 0, 0, 0.1250, 6.0,
        512, "plain");
    mkv(&vs[1], P_XOR, ISA_AVX512, MUL_NIBBLE, 4, 0, 0, 1, 0.1406, 7.0,
        512, "twin");
    int n = prune_versions(vs, 2, 0);
    CHECK(n == 2, "P_XOR twin must survive, got %d", n);
}

/* F1 pin: a P_XOR +ternlog twin that DOES pareto-dominate plain on the
 * triple must STILL survive. The twin direction is validated only for
 * P_MULACC/P_DELTA, and the kind guard (not the triple values)
 * enforces it: if the model ever claims a P_XOR twin dominates,
 * pruning is refused. Catches kind-guard removal. */
static int prune_capture_stderr(version_t *vs, int nv, unsigned flags,
                                char *buf, size_t buflen);
static void unit_pxor_twin_dominant_survives(void)
{
    version_t vs[2];
    mkv(&vs[0], P_XOR, ISA_AVX512, MUL_NIBBLE, 4, 0, 0, 0, 0.1500, 7.0,
        600, "plain");
    mkv(&vs[1], P_XOR, ISA_AVX512, MUL_NIBBLE, 4, 0, 0, 1, 0.1250, 6.0,
        512, "twin");
    int n = prune_versions(vs, 2, 0);
    CHECK(n == 2, "triple-dominant P_XOR twin must survive (kind guard),"
                  " got %d", n);
    CHECK(find_desc(vs, n, "twin") >= 0, "twin must be present");
    CHECK(find_desc(vs, n, "plain") >= 0, "plain must be present");

    /* F1: the refusal is LOUD under PRUNE_VERBOSE -- the exact case the
     * task names ("the model claims a P_XOR twin dominates") must emit
     * the verbose log. Fresh array: prune_versions() mutates input. */
    version_t vs3[2];
    mkv(&vs3[0], P_XOR, ISA_AVX512, MUL_NIBBLE, 4, 0, 0, 0, 0.1500, 7.0,
        600, "plain");
    mkv(&vs3[1], P_XOR, ISA_AVX512, MUL_NIBBLE, 4, 0, 0, 1, 0.1250, 6.0,
        512, "twin");
    char logb[1024];
    int ln = prune_capture_stderr(vs3, 2, PRUNE_VERBOSE, logb,
                                  sizeof(logb));
    CHECK(ln > 0 && strstr(logb, "NOT pruning") != NULL,
          "kind-guard refusal must be logged under PRUNE_VERBOSE");
    CHECK(strstr(logb, "P_XOR") != NULL,
          "kind-guard log must name the plan kind");
}

/* Capture stderr around a prune_versions() call into buf (NUL
 * terminated). Returns bytes captured, or -1 on capture failure. */
static int prune_capture_stderr(version_t *vs, int nv, unsigned flags,
                                char *buf, size_t buflen)
{
    int pipefd[2];
    if (pipe(pipefd) != 0)
        return -1;
    fflush(stderr);
    int saved = dup(STDERR_FILENO);
    if (saved < 0 || dup2(pipefd[1], STDERR_FILENO) < 0) {
        if (saved >= 0)
            close(saved);
        close(pipefd[0]);
        close(pipefd[1]);
        return -1;
    }
    close(pipefd[1]); /* stderr now holds the only write end */
    prune_versions(vs, nv, flags);
    fflush(stderr);
    dup2(saved, STDERR_FILENO); /* closes the pipe write end */
    close(saved);
    size_t got = 0;
    ssize_t nr;
    while (got + 1 < buflen &&
           (nr = read(pipefd[0], buf + got, buflen - 1 - got)) > 0)
        got += (size_t)nr;
    close(pipefd[0]);
    buf[got] = '\0';
    return (int)got;
}

/* Direction pin for validated kinds (M2 catcher): a P_MULACC pair
 * where PLAIN pareto-dominates the twin on the triple (hypothetical
 * codegen regression making +ternlog worse). Correct code prunes
 * nothing: the rule only ever prunes plain-in-favor-of-twin. M2
 * (symmetric tl) prunes the twin here -> FAILS. */
static void unit_mulacc_plain_dominant_twin_survives(void)
{
    version_t vs[2];
    mkv(&vs[0], P_MULACC, ISA_AVX512, MUL_NIBBLE, 8, 0, 0, 0, 1.346,
        32.016, 5440, "plain");
    mkv(&vs[1], P_MULACC, ISA_AVX512, MUL_NIBBLE, 8, 0, 0, 1, 1.502,
        32.016, 5824, "twin");
    int n = prune_versions(vs, 2, 0);
    CHECK(n == 2, "twin must not be pruned in favor of plain, got %d",
          n);
}

/* U variants never prune each other, even when a twin pair is present:
 * dominance requires identical U. (For P_XOR the model reports
 * U-invariant uops/B, so cross-U pruning picked U=1 over the real
 * U=4/U=5 winners -- the failure that narrowed this rule.) */
static void unit_u_variants_survive(void)
{
    version_t vs[4];
    mkv(&vs[0], P_MULACC, ISA_AVX512, MUL_NIBBLE, 8, 0, 0, 0, 1.502,
        32.016, 5824, "plain8");
    mkv(&vs[1], P_MULACC, ISA_AVX512, MUL_NIBBLE, 8, 0, 0, 1, 1.346,
        32.016, 5440, "twin8");
    mkv(&vs[2], P_MULACC, ISA_AVX512, MUL_NIBBLE, 1, 0, 0, 0, 1.510,
        32.100, 1152, "plain1");
    mkv(&vs[3], P_MULACC, ISA_AVX512, MUL_NIBBLE, 1, 0, 0, 1, 1.354,
        32.100, 1088, "twin1");
    int n = prune_versions(vs, 4, 0);
    CHECK(n == 2, "want 2 survivors (twins), got %d", n);
    CHECK(find_desc(vs, n, "twin8") >= 0, "twin8 must survive");
    CHECK(find_desc(vs, n, "twin1") >= 0, "twin1 must survive");
}

/* F9/M1 catcher: P_XOR-like U-invariant model -- uops/B and mem/B
 * identical across U, code smaller at U=1 (unrolling benefits invisible
 * to the model, the exact failure mode of the rejected broader rule).
 * With the U equality dropped (mutation M1) the U=1 twin
 * pareto-dominates the U=8 plain on the triple and prunes it -> FAILS.
 * With the rule intact, U differs -> both survive. */
static void unit_cross_u_twin_no_prune(void)
{
    version_t vs[2];
    mkv(&vs[0], P_MULACC, ISA_AVX512, MUL_NIBBLE, 8, 0, 0, 0, 0.1250,
        6.0, 2048, "plain8");
    mkv(&vs[1], P_MULACC, ISA_AVX512, MUL_NIBBLE, 1, 0, 0, 1, 0.1250,
        6.0, 1024, "twin1");
    int n = prune_versions(vs, 2, 0);
    CHECK(n == 2, "cross-U twin must not prune, got %d", n);
    CHECK(find_desc(vs, n, "plain8") >= 0, "plain8 must survive");
}

/* NT twins: dominance requires identical nt. A twin prunes only the
 * plain with the same nt. */
static void unit_nt_twins_survive(void)
{
    version_t vs[4];
    mkv(&vs[0], P_MULACC, ISA_AVX512, MUL_NIBBLE, 8, 0, 0, 0, 1.502,
        32.016, 5824, "plain");
    mkv(&vs[1], P_MULACC, ISA_AVX512, MUL_NIBBLE, 8, 0, 0, 1, 1.346,
        32.016, 5440, "twin");
    mkv(&vs[2], P_MULACC, ISA_AVX512, MUL_NIBBLE, 8, 1, 0, 0, 1.502,
        32.016, 5824, "plain+nt");
    mkv(&vs[3], P_MULACC, ISA_AVX512, MUL_NIBBLE, 8, 1, 0, 1, 1.346,
        32.016, 5440, "twin+nt");
    int n = prune_versions(vs, 4, 0);
    CHECK(n == 2, "want 2 survivors (twins), got %d", n);
    CHECK(find_desc(vs, n, "twin") >= 0, "twin must survive");
    CHECK(find_desc(vs, n, "twin+nt") >= 0, "twin+nt must survive");
    /* cross-nt must not prune: twin (nt=0) vs plain+nt (nt=1) */
    version_t vs2[2];
    mkv(&vs2[0], P_MULACC, ISA_AVX512, MUL_NIBBLE, 8, 1, 0, 0, 1.502,
        32.016, 5824, "plain+nt");
    mkv(&vs2[1], P_MULACC, ISA_AVX512, MUL_NIBBLE, 8, 0, 0, 1, 1.346,
        32.016, 5440, "twin");
    CHECK(prune_versions(vs2, 2, 0) == 2, "cross-nt must not prune");
}

/* Cross-ISA and cross-mul pairs never prune (different isa/mul). */
static void unit_cross_isa_mul_survive(void)
{
    version_t vs[4];
    mkv(&vs[0], P_MULACC, ISA_AVX512, MUL_NIBBLE, 8, 0, 0, 0, 1.502,
        32.016, 5824, "plain512");
    mkv(&vs[1], P_MULACC, ISA_AVX512, MUL_NIBBLE, 8, 0, 0, 1, 1.346,
        32.016, 5440, "twin512");
    mkv(&vs[2], P_MULACC, ISA_AVX2, MUL_NIBBLE, 8, 0, 0, 0, 3.004,
        32.062, 3200, "plainavx2");
    mkv(&vs[3], P_MULACC, ISA_AVX512, MUL_GFNI, 8, 0, 0, 0, 0.660,
        18.141, 3136, "gfni");
    int n = prune_versions(vs, 4, 0);
    CHECK(n == 3, "want 3 survivors, got %d", n);
    CHECK(find_desc(vs, n, "twin512") >= 0, "twin512 must survive");
    CHECK(find_desc(vs, n, "plainavx2") >= 0, "plainavx2 must survive");
    CHECK(find_desc(vs, n, "gfni") >= 0, "gfni must survive");
}

/* Prefetch: dominance requires identical prefetch_dist. A prefetch
 * twin prunes only a prefetch plain, never a non-prefetch one. */
static void unit_prefetch_same_dist(void)
{
    version_t vs[3];
    mkv(&vs[0], P_MULACC, ISA_AVX512, MUL_NIBBLE, 8, 0, 0, 0, 1.502,
        32.016, 5824, "plain");
    mkv(&vs[1], P_MULACC, ISA_AVX512, MUL_NIBBLE, 8, 0, 4096, 1, 1.346,
        32.016, 5440, "pf-twin");
    mkv(&vs[2], P_MULACC, ISA_AVX512, MUL_NIBBLE, 8, 0, 4096, 0, 1.502,
        32.016, 5824, "pf-plain");
    int n = prune_versions(vs, 3, 0);
    CHECK(n == 2, "want 2 survivors, got %d", n);
    CHECK(find_desc(vs, n, "plain") >= 0,
          "non-prefetch plain must survive pf-twin");
    CHECK(find_desc(vs, n, "pf-twin") >= 0, "pf-twin must survive");
    CHECK(find_desc(vs, n, "pf-plain") < 0, "pf-plain must be pruned");
}

/* remat (reviewer1 4): a twin with a different remat flag is a
 * different codegen shape, not a multiply twin -- never prunes. */
static void unit_remat_no_prune(void)
{
    version_t vs[2];
    mkv(&vs[0], P_HORNER, ISA_AVX512, MUL_NIBBLE, 8, 0, 0, 0, 1.502,
        32.016, 5824, "plain");
    mkv(&vs[1], P_HORNER, ISA_AVX512, MUL_NIBBLE, 8, 0, 0, 1, 1.346,
        32.016, 5440, "twin");
    vs[0].cand.remat = 0;
    vs[1].cand.remat = 1; /* remat differs: not a twin pair */
    int n = prune_versions(vs, 2, 0);
    CHECK(n == 2, "cross-remat must not prune, got %d", n);
    /* same remat, but P_HORNER has no validated twin direction: the
     * kind guard refuses even a triple-dominant pair */
    vs[1].cand.remat = 0;
    n = prune_versions(vs, 2, 0);
    CHECK(n == 2, "P_HORNER twin must not prune (kind guard), got %d",
          n);
}

/* Cross-kind pairs never prune (different plan kinds). */
static void unit_cross_kind_survive(void)
{
    version_t vs[2];
    mkv(&vs[0], P_DELTA, ISA_AVX512, MUL_NIBBLE, 8, 0, 0, 0, 1.502,
        32.016, 5824, "plain-delta");
    mkv(&vs[1], P_MULACC, ISA_AVX512, MUL_NIBBLE, 8, 0, 0, 1, 1.346,
        32.016, 5440, "twin-mulacc");
    int n = prune_versions(vs, 2, 0);
    CHECK(n == 2, "cross-kind must not prune, got %d", n);
}

/* F5: a plain whose +ternlog twin failed jit_build_version (so the
 * twin is absent from the version list) must survive: with no twin
 * present there is nothing that can dominate it. */
static void unit_plain_without_twin_survives(void)
{
    version_t vs[2];
    mkv(&vs[0], P_MULACC, ISA_AVX512, MUL_NIBBLE, 8, 0, 0, 0, 1.502,
        32.016, 5824, "plain");
    mkv(&vs[1], P_MULACC, ISA_AVX2, MUL_NIBBLE, 8, 0, 0, 0, 3.004,
        32.062, 3200, "other");
    int n = prune_versions(vs, 2, 0);
    CHECK(n == 2, "plain without twin must survive, got %d", n);
    CHECK(find_desc(vs, n, "plain") >= 0, "plain must be present");
}

/* Non-strict ties (within eps) never prune. Uses a tl=1/tl=0 pair. */
static void unit_eps_ties_survive(void)
{
    version_t vs[2];
    /* differences of 1e-12: below PRUNE_EPS -> tie -> both survive */
    mkv(&vs[0], P_MULACC, ISA_AVX512, MUL_NIBBLE, 8, 0, 0, 0, 1.502,
        32.016, 5824, "plain");
    mkv(&vs[1], P_MULACC, ISA_AVX512, MUL_NIBBLE, 8, 0, 0, 1,
        1.502 - 1e-12, 32.016, 5824, "twin");
    int n = prune_versions(vs, 2, 0);
    CHECK(n == 2, "eps-tie must survive, got %d", n);
    /* differences of 1e-6: above PRUNE_EPS -> strict -> plain pruned.
     * Fresh array: prune_versions() mutates its input, so the second
     * check rebuilds both entries (see AGENTS.md: fresh fixtures per
     * mutating call). */
    mkv(&vs[0], P_MULACC, ISA_AVX512, MUL_NIBBLE, 8, 0, 0, 0, 1.502,
        32.016, 5824, "plain");
    mkv(&vs[1], P_MULACC, ISA_AVX512, MUL_NIBBLE, 8, 0, 0, 1,
        1.502 - 1e-6, 32.016, 5824, "twin");
    n = prune_versions(vs, 2, 0);
    CHECK(n == 1 && find_desc(vs, n, "twin") == 0,
          "above-eps difference must prune, got %d", n);
}

/* A twin that is NOT pareto-better does not prune: the triple gate
 * is load-bearing, not the mere existence of a twin. Here the twin
 * has smaller uops but BIGGER code (hypothetical codegen regression);
 * both survive. */
static void unit_twin_not_pareto_survives(void)
{
    version_t vs[2];
    mkv(&vs[0], P_MULACC, ISA_AVX512, MUL_NIBBLE, 8, 0, 0, 0, 1.502,
        32.016, 5824, "plain");
    mkv(&vs[1], P_MULACC, ISA_AVX512, MUL_NIBBLE, 8, 0, 0, 1, 1.346,
        32.016, 6000, "twin");
    int n = prune_versions(vs, 2, 0);
    CHECK(n == 2, "non-pareto twin must not prune, got %d", n);
}

/* Empty and singleton lists. */
static void unit_edge_lists(void)
{
    version_t vs[1];
    CHECK(prune_versions(vs, 0, 0) == 0, "empty list -> 0");
    mkv(vs, P_MULACC, ISA_AVX512, MUL_GFNI, 8, 0, 0, 0, 0.660, 18.141,
        3136, "only");
    CHECK(prune_versions(vs, 1, 0) == 1, "singleton -> 1");
}

/* Disable switch: env var and flag. Each check uses a FRESH array:
 * prune_versions() mutates its input (pruned versions are finalized,
 * survivors compacted), so reusing the array across checks would test
 * garbage. */
static void mk_twin_pair(version_t *vs)
{
    mkv(&vs[0], P_MULACC, ISA_AVX512, MUL_NIBBLE, 8, 0, 0, 0, 1.502,
        32.016, 5824, "plain");
    mkv(&vs[1], P_MULACC, ISA_AVX512, MUL_NIBBLE, 8, 0, 0, 1, 1.346,
        32.016, 5440, "twin");
}
static void unit_disable(void)
{
    version_t vs[2];

    mk_twin_pair(vs);
    setenv("RJ_NO_PRUNE", "1", 1);
    CHECK(prune_versions(vs, 2, 0) == 2, "RJ_NO_PRUNE=1 must disable");
    unsetenv("RJ_NO_PRUNE");

    mk_twin_pair(vs);
    setenv("RJ_NO_PRUNE", "0", 1);
    CHECK(prune_versions(vs, 2, 0) == 1, "RJ_NO_PRUNE=0 must not disable");
    unsetenv("RJ_NO_PRUNE");

    mk_twin_pair(vs);
    setenv("RJ_NO_PRUNE", "", 1);
    CHECK(prune_versions(vs, 2, 0) == 1, "RJ_NO_PRUNE='' must not disable");
    unsetenv("RJ_NO_PRUNE");

    /* the contract is "non-empty and != "0"": these all disable */
    mk_twin_pair(vs);
    setenv("RJ_NO_PRUNE", "false", 1);
    CHECK(prune_versions(vs, 2, 0) == 2,
          "RJ_NO_PRUNE=false must disable");
    unsetenv("RJ_NO_PRUNE");

    mk_twin_pair(vs);
    setenv("RJ_NO_PRUNE", "00", 1);
    CHECK(prune_versions(vs, 2, 0) == 2, "RJ_NO_PRUNE=00 must disable");
    unsetenv("RJ_NO_PRUNE");

    mk_twin_pair(vs);
    setenv("RJ_NO_PRUNE", "0 ", 1);
    CHECK(prune_versions(vs, 2, 0) == 2,
          "RJ_NO_PRUNE='0 ' must disable");
    unsetenv("RJ_NO_PRUNE");

    mk_twin_pair(vs);
    CHECK(prune_versions(vs, 2, PRUNE_DISABLE) == 2,
          "PRUNE_DISABLE flag must disable");

    mk_twin_pair(vs);
    CHECK(prune_versions(vs, 2, 0) == 1, "no flag must prune");
}

/* Survivors keep their original relative order; multiple plains pruned
 * at once (one per twin pair). */
static void unit_order_and_many(void)
{
    version_t vs[5];
    mkv(&vs[0], P_MULACC, ISA_AVX512, MUL_NIBBLE, 8, 0, 0, 1, 1.346,
        32.016, 5440, "twin8");
    mkv(&vs[1], P_MULACC, ISA_AVX512, MUL_NIBBLE, 8, 0, 0, 0, 1.502,
        32.016, 5824, "plain8");
    mkv(&vs[2], P_MULACC, ISA_AVX2, MUL_NIBBLE, 4, 0, 0, 0, 3.008,
        32.062, 3200, "avx2");
    mkv(&vs[3], P_MULACC, ISA_AVX512, MUL_NIBBLE, 4, 0, 0, 0, 1.504,
        32.031, 3328, "plain4");
    mkv(&vs[4], P_MULACC, ISA_AVX512, MUL_NIBBLE, 4, 0, 0, 1, 1.348,
        32.031, 3136, "twin4");
    int n = prune_versions(vs, 5, 0);
    CHECK(n == 3, "want 3 survivors, got %d", n);
    CHECK(find_desc(vs, n, "twin8") == 0, "order: twin8 first");
    CHECK(find_desc(vs, n, "avx2") == 1, "order: avx2 second");
    CHECK(find_desc(vs, n, "twin4") == 2, "order: twin4 third");
}

/* ---------------- Part 2+3: matrix over real codegen ---------------- */

typedef struct {
    const char *name;
    rj_topology *(*topo)(void);
    rj_op (*op)(void);
    size_t bench_bytes;
    size_t prefetch; /* prefetch_distance; 0 = none */
} cfg_t;

static rj_topology *t_raid5_4(void) { return rj_topo_raid5(4); }
static rj_topology *t_rs63(void) { return rj_topo_rs(6, 3); }
static rj_op o_enc(void) { return rj_op_encode(); }
static int rec01_f[] = { 0, 1 };
static rj_op o_rec01(void) { return rj_op_recover(rec01_f, 2); }
/* The helpers below feed x86-64-only matrix configs (ternlog twins and
 * large recovery shapes exist only on x86-64); guard them so the
 * cross builds (aarch64/riscv64, -Werror=unused-function) stay clean.
 * (Latent pre-existing issue: the original file had the same unused
 * helpers on cross builds; exposed by running the QEMU targets for
 * real, 2026-10-02.) */
#if defined(__x86_64__)
static rj_topology *t_raid6_6(void) { return rj_topo_raid6(6); }
static rj_topology *t_rs103(void) { return rj_topo_rs(10, 3); }
static rj_topology *t_rs203(void) { return rj_topo_rs(20, 3); }
/* dense 4+2 custom matrix, recover data disk 0 + parity disk 0:
 * the recovery plan is P_XOR with n_out=2, so the AVX-512 VPTERNLOGD
 * twin is enumerated (the ship-criterion twin case). */
static uint8_t dense42_mat[8];
static rj_topology *t_dense42(void)
{
    memset(dense42_mat, 1, sizeof dense42_mat);
    return rj_topo_custom(4, 2, dense42_mat);
}
/* rs(10,4) over a Cauchy matrix (MDS by construction: x_i = i+1,
 * y_j = nd+1+j, M = inv(x^y)); same construction as
 * tools/perf_guard.c. A large-recovery config where pruning fires. */
static uint8_t cauchy104_mat[4 * 10];
static rj_topology *t_rs104c(void)
{
    for (int j = 0; j < 4; j++)
        for (int i = 0; i < 10; i++)
            cauchy104_mat[(size_t)j * 10 + (size_t)i] =
                gf_inv((uint8_t)((i + 1) ^ (10 + 1 + j)));
    return rj_topo_custom(10, 4, cauchy104_mat);
}
static int rec012_f[] = { 0, 1, 2 };
static rj_op o_rec012(void) { return rj_op_recover(rec012_f, 3); }
static int rec04_f[] = { 0, 4 };
static rj_op o_rec04(void) { return rj_op_recover(rec04_f, 2); }
static int pupd02_f[] = { 0, 2 };
static rj_op o_pupd02(void) { return rj_op_parity_update(pupd02_f, 2); }
#endif /* __x86_64__ */

/* Full matrix on x86-64 (where pruning fires); a 2-config smoke
 * elsewhere (pruning is a no-op there: no ternlog twins on
 * AArch64/RVV, but the plumbing must still agree). */
#if defined(__x86_64__)
static const cfg_t cfgs[] = {
    { "xor1/64K", t_raid5_4, o_enc, 1 << 16, 0 },     /* P_XOR n_out=1 */
    { "xor2/64K", t_dense42, o_rec04, 1 << 16, 0 },   /* P_XOR n_out=2+tl */
    { "horner/64K", t_raid6_6, o_enc, 1 << 16, 0 },   /* P_HORNER */
    { "mulacc/64K", t_rs63, o_rec01, 1 << 16, 0 },    /* P_MULACC */
    { "delta/64K", t_raid5_4, o_pupd02, 1 << 16, 0 }, /* P_DELTA */
    { "mulacc/1M", t_rs63, o_rec01, 1 << 20, 0 },
    { "xor2/1M", t_dense42, o_rec04, 1 << 20, 0 },
    /* F6: prefetch_dist != 0 -- the pf axis of the twin gate is
     * exercised on real codegen (prefetch twins prune prefetch
     * plains only). */
    { "mulacc-pf/64K", t_rs63, o_rec01, 1 << 16, 4096 },
    /* F7: large recovery configs -- the highest-leverage pruning
     * configs (many P_MULACC twin pairs), previously uncovered. */
    { "rs103rec/64K", t_rs103, o_rec01, 1 << 16, 0 },
    { "rs203rec/64K", t_rs203, o_rec012, 1 << 16, 0 },
    { "rs104crec/64K", t_rs104c, o_rec01, 1 << 16, 0 },
};
#else
static const cfg_t cfgs[] = {
    { "xor1/64K", t_raid5_4, o_enc, 1 << 16, 0 },
    { "mulacc/64K", t_rs63, o_rec01, 1 << 16, 0 },
};
#endif
#define NCFG (sizeof(cfgs) / sizeof(cfgs[0]))

static const cfg_t *find_cfg(const char *name)
{
    for (size_t i = 0; i < NCFG; i++)
        if (strcmp(cfgs[i].name, name) == 0)
            return &cfgs[i];
    return NULL;
}

/* ---------------- F10: scalar reference for the Part-2 winners ---------------- */

static uint64_t tprng_state = 0x243f6a8885a308d3ULL;
static uint64_t tprng(void)
{
    uint64_t x = tprng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return tprng_state = x;
}

static void *tabuf(size_t n)
{
    void *p = NULL;
    if (posix_memalign(&p, 64, n ? n : 64)) {
        fprintf(stderr, "test_prune: oom\n");
        exit(1);
    }
    return p;
}

static void ref_encode_mat(const uint8_t *mat, int nd, int np,
                           uint8_t **d, uint8_t **p, size_t len)
{
    for (int j = 0; j < np; j++)
        for (size_t k = 0; k < len; k++) {
            uint8_t a = 0;
            for (int i = 0; i < nd; i++) {
                uint8_t c = mat[(size_t)j * (size_t)nd + (size_t)i];
                if (c)
                    a ^= gf_mul(d[i][k], c);
            }
            p[j][k] = a;
        }
}

/* Verify the compiled winner against the scalar reference at a small
 * length. Returns 0 on match, 1 on mismatch. Generic over the op
 * (encode / recover / parity_update); mirrors tools/perf_guard.c's
 * verify(). This implements the reference comparison the old Part-2
 * header claimed but never performed. */
static int verify_winner(const rj_op *op, rj_fn fn, const topo_int_t *ti)
{
    int nd = ti->ndata, np = ti->npar;
    size_t len = 1000; /* exercises vector body + scalar tail */
    int nbuf = nd + np;
    uint8_t **b = malloc(sizeof(*b) * (size_t)(nbuf + np));
    uint8_t **want = malloc(sizeof(*want) * (size_t)np);
    for (int i = 0; i < nbuf + np; i++)
        b[i] = tabuf(len);
    for (int j = 0; j < np; j++)
        want[j] = tabuf(len);
    for (int i = 0; i < nd; i++)
        for (size_t k = 0; k < len; k++)
            b[i][k] = (uint8_t)tprng();

    const void **in;
    void **out;
    int ok = 1;

    if (!op->is_recover && !op->is_parity_update) { /* encode */
        ref_encode_mat(ti->mat, nd, np, b, want, len);
        in = malloc(sizeof(*in) * (size_t)nd);
        out = malloc(sizeof(*out) * (size_t)np);
        for (int i = 0; i < nd; i++)
            in[i] = b[i];
        for (int j = 0; j < np; j++) {
            memset(b[nd + j], 0xA5, len);
            out[j] = b[nd + j];
        }
        fn(in, out, len);
        for (int j = 0; j < np && ok; j++)
            if (memcmp(b[nd + j], want[j], len))
                ok = 0;
        free(in);
        free(out);
    } else if (op->is_recover) {
        ref_encode_mat(ti->mat, nd, np, b, want, len);
        for (int j = 0; j < np; j++)
            memcpy(b[nd + j], want[j], len);
        int ns = nd + np - op->nfailed;
        in = malloc(sizeof(*in) * (size_t)ns);
        out = malloc(sizeof(*out) * (size_t)op->nfailed);
        int si = 0, fi = 0;
        uint8_t **orig_fail =
            malloc(sizeof(*orig_fail) * (size_t)op->nfailed);
        for (int d = 0; d < nd + np; d++) {
            int isf = 0;
            for (int f = 0; f < op->nfailed; f++)
                if (op->failed[f] == d)
                    isf = 1;
            if (isf) {
                orig_fail[fi] = tabuf(len);
                memcpy(orig_fail[fi], b[d], len);
                memset(b[d], 0x5A, len); /* clobber "failed" disk */
                out[fi++] = b[d];
            } else {
                in[si++] = b[d];
            }
        }
        fn(in, out, len);
        for (int f = 0; f < op->nfailed && ok; f++)
            if (memcmp(out[f], orig_fail[f], len))
                ok = 0;
        for (int f = 0; f < op->nfailed; f++)
            free(orig_fail[f]);
        free(orig_fail);
        free(in);
        free(out);
    } else { /* parity update */
        int nc = op->nchanged;
        uint8_t **old = malloc(sizeof(*old) * (size_t)nc);
        uint8_t **nw = malloc(sizeof(*nw) * (size_t)nc);
        for (int i = 0; i < nc; i++) {
            old[i] = tabuf(len);
            nw[i] = tabuf(len);
            for (size_t k = 0; k < len; k++) {
                old[i][k] = (uint8_t)tprng();
                nw[i][k] = (uint8_t)tprng();
            }
            memcpy(b[op->changed[i]], old[i], len);
        }
        ref_encode_mat(ti->mat, nd, np, b, want, len); /* old parities */
        for (int i = 0; i < nc; i++)
            memcpy(b[op->changed[i]], nw[i], len);
        uint8_t **want2 = malloc(sizeof(*want2) * (size_t)np);
        for (int j = 0; j < np; j++)
            want2[j] = tabuf(len);
        ref_encode_mat(ti->mat, nd, np, b, want2, len); /* new parities */
        in = malloc(sizeof(*in) * (size_t)(2 * nc + np));
        out = malloc(sizeof(*out) * (size_t)np);
        for (int i = 0; i < nc; i++) {
            in[i] = old[i];
            in[nc + i] = nw[i];
        }
        for (int j = 0; j < np; j++) {
            memcpy(b[nd + j], want[j], len);
            in[2 * nc + j] = b[nd + j];
            memset(b[nd + np + j], 0, len);
            out[j] = b[nd + np + j];
        }
        fn(in, out, len);
        for (int j = 0; j < np && ok; j++)
            if (memcmp(out[j], want2[j], len))
                ok = 0;
        for (int i = 0; i < nc; i++) {
            free(old[i]);
            free(nw[i]);
        }
        for (int j = 0; j < np; j++)
            free(want2[j]);
        free(old);
        free(nw);
        free(want2);
        free(in);
        free(out);
    }

    for (int i = 0; i < nbuf + np; i++)
        free(b[i]);
    for (int j = 0; j < np; j++)
        free(want[j]);
    free(b);
    free(want);
    return !ok;
}

/* Compile one config; returns the winner desc in `out`, or
 * "COMPILE-FAIL:..." on compile or reference-verification failure.
 * prune_off=1 sets RJ_NO_PRUNE=1 for the unpruned side. */
static void compile_one(const cfg_t *c, int prune_off, char *out,
                        size_t outlen)
{
    char err[256];
    rj_topology *t = c->topo();
    rj_op op = c->op();
    rj_options o;
    rj_options_default(&o);
    o.bench_bytes = c->bench_bytes;
    o.bench_iters = 5;
    o.prefetch_distance = c->prefetch;
    if (prune_off)
        setenv("RJ_NO_PRUNE", "1", 1);
    else
        unsetenv("RJ_NO_PRUNE");
    rj_jit *j = rj_compile(t, &op, &o, err, sizeof(err));
    if (prune_off)
        unsetenv("RJ_NO_PRUNE");
    if (!j) {
        snprintf(out, outlen, "COMPILE-FAIL:%.100s", err);
        rj_op_free(&op);
        rj_topo_free(t);
        return;
    }
    /* the winner must be correct: run it against the scalar reference
     * at a boundary length (F10 -- this comparison was claimed but
     * missing; bench_select's own validation is the first pair of
     * eyes, this is the second on the shipped path) */
    {
        topo_int_t ti;
        char e3[256];
        if (topo_int_build(&ti, t, e3, sizeof(e3)) != 0) {
            snprintf(out, outlen, "COMPILE-FAIL:topo_int:%.80s", e3);
        } else {
            int bad = verify_winner(&op, rj_get_fn(j), &ti);
            topo_int_free(&ti);
            if (bad) {
                snprintf(out, outlen, "COMPILE-FAIL:ref-mismatch");
            } else {
                rj_info info;
                rj_get_info(j, &info);
                snprintf(out, outlen, "%.120s", info.desc);
            }
        }
    }
    rj_free(j);
    rj_op_free(&op);
    rj_topo_free(t);
}

/* Part 2: end-to-end smoke via the production rj_compile path (which
 * calls prune_versions before bench_select when benchmarking).
 * Asserts compilation succeeds with pruning enabled and disabled, and
 * that each winner matches the scalar reference. Winner EQUALITY
 * across two independent rj_compile runs is NOT asserted here:
 * benchmark noise alone flips winners between runs even with pruning
 * disabled (observed 2026-10-02), so that check would be flaky.
 * Deterministic pruned-vs-unpruned winner agreement on real measured
 * times is Part 3's job. */
static void e2e_smoke(void)
{
    for (size_t ci = 0; ci < NCFG; ci++) {
        char d_pruned[128], d_full[128];
        compile_one(&cfgs[ci], 0, d_pruned, sizeof(d_pruned));
        compile_one(&cfgs[ci], 1, d_full, sizeof(d_full));
        CHECK(strncmp(d_pruned, "COMPILE-FAIL", 12) != 0,
              "%s pruned compile/verify failed: %s", cfgs[ci].name,
              d_pruned);
        CHECK(strncmp(d_full, "COMPILE-FAIL", 12) != 0,
              "%s unpruned compile/verify failed: %s", cfgs[ci].name,
              d_full);
        if (failures == 0)
            printf("  smoke %s -> %s\n", cfgs[ci].name, d_pruned);
    }
}

/* Compile cfg with verbose=1, capturing stderr into buf (NUL-
 * terminated). prune_off=1 sets RJ_NO_PRUNE=1. Returns bytes captured,
 * or -1 on capture/compile failure (*ok=0 on compile failure).
 * x86-64 only: the sole user is hook_wiring's x86 branch. */
#if defined(__x86_64__)
static int compile_capture_stderr(const cfg_t *c, int prune_off, char *buf,
                                  size_t buflen, int *ok)
{
    *ok = 0;
    int pipefd[2];
    if (pipe(pipefd) != 0)
        return -1;
    fflush(stderr);
    int saved = dup(STDERR_FILENO);
    if (saved < 0) {
        close(pipefd[0]);
        close(pipefd[1]);
        return -1;
    }
    if (dup2(pipefd[1], STDERR_FILENO) < 0) {
        close(saved);
        close(pipefd[0]);
        close(pipefd[1]);
        return -1;
    }
    close(pipefd[1]); /* stderr now holds the only write end */

    char err[256];
    rj_topology *t = c->topo();
    rj_op op = c->op();
    rj_options o;
    rj_options_default(&o);
    o.bench_bytes = c->bench_bytes;
    o.bench_iters = 5;
    o.prefetch_distance = c->prefetch;
    o.verbose = 1;
    if (prune_off)
        setenv("RJ_NO_PRUNE", "1", 1);
    else
        unsetenv("RJ_NO_PRUNE");
    rj_jit *j = rj_compile(t, &op, &o, err, sizeof(err));
    if (prune_off)
        unsetenv("RJ_NO_PRUNE");
    if (j) {
        *ok = 1;
        rj_free(j);
    }
    rj_op_free(&op);
    rj_topo_free(t);

    fflush(stderr);
    dup2(saved, STDERR_FILENO); /* closes the pipe write end */
    close(saved);
    size_t got = 0;
    ssize_t nr;
    while (got + 1 < buflen &&
           (nr = read(pipefd[0], buf + got, buflen - 1 - got)) > 0)
        got += (size_t)nr;
    close(pipefd[0]);
    buf[got] = '\0';
    return (int)got;
}
#endif /* __x86_64__ */

/* Part 2b (F11): pin the production-path wiring. With verbose=1 the
 * api.c hook prints "raidjit: prune: dropped ..." to stderr exactly
 * when it prunes; if the hook were removed, the line disappears and
 * this test fails. Negative control: RJ_NO_PRUNE=1 also removes the
 * line. Only meaningful where pruning fires (x86-64 with AVX-512
 * ternlog twins); skipped elsewhere. */
static void hook_wiring(void)
{
#if defined(__x86_64__)
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    if (!cpu.avx512f) {
        printf("  hook wiring: skipped (no AVX-512 on this host)\n");
        return;
    }
    const cfg_t *c = find_cfg("mulacc/64K");
    CHECK(c != NULL, "hook wiring: mulacc/64K config missing");
    if (!c)
        return;
    char buf[16384];
    int ok = 0;
    int n = compile_capture_stderr(c, 0, buf, sizeof(buf), &ok);
    CHECK(ok, "hook wiring: pruned compile failed");
    CHECK(n >= 0, "hook wiring: stderr capture failed");
    const char *mark = strstr(buf, "raidjit: prune: dropped ");
    int dropped = -1, total = -1;
    if (mark)
        sscanf(mark, "raidjit: prune: dropped %d of %d", &dropped,
               &total);
    CHECK(mark != NULL && dropped > 0,
          "hook wiring: prune line missing/empty in verbose rj_compile "
          "(hook removed?)");
    if (mark != NULL && dropped > 0)
        printf("  hook wiring: prune line OK (dropped %d of %d)\n",
               dropped, total);
    /* negative control: pruning disabled -> no prune line */
    n = compile_capture_stderr(c, 1, buf, sizeof(buf), &ok);
    CHECK(ok, "hook wiring: unpruned compile failed");
    CHECK(n >= 0, "hook wiring: stderr capture failed (unpruned)");
    CHECK(strstr(buf, "raidjit: prune: dropped ") == NULL,
          "hook wiring: prune line present with RJ_NO_PRUNE=1");
#else
    printf("  hook wiring: skipped (non-x86_64)\n");
    (void)find_cfg;
#endif
}

/* Build all versions for a config (caller frees survivors via
 * jit_free_version; prune_versions finalizes the pruned ones). */
static int build_all(const cfg_t *c, version_t **out)
{
    char err[256];
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    rj_topology *t = c->topo();
    rj_op op = c->op();
    topo_int_t ti;
    plan_t pl;
    if (topo_int_build(&ti, t, err, sizeof(err)) ||
        plan_build(&ti, &op, "t", &pl, err, sizeof(err))) {
        printf("FAIL setup %s: %s\n", c->name, err);
        failures++;
        rj_op_free(&op);
        rj_topo_free(t);
        return -1;
    }
    cand_t cands[512];
    int nc = plan_candidates(&pl, &cpu, cands, 512, 1, c->prefetch);
    version_t *vs = calloc((size_t)nc, sizeof(*vs));
    rj_options o;
    rj_options_default(&o);
    o.bench_bytes = c->bench_bytes;
    o.bench_iters = 5;
    int nv = 0;
    for (int i = 0; i < nc; i++) {
        char e2[256];
        if (jit_build_version(&pl, &cands[i], &cpu, &o, &vs[nv], e2,
                              sizeof(e2)) == 0)
            nv++;
    }
    /* Clear the version_t.plan borrows: the plan is freed below, and
     * bench_select must use the shared plan passed by the caller, not
     * a dangling borrow. (Track A5 added the borrow; tests that free
     * the plan before bench_select must clear it.) */
    for (int i = 0; i < nv; i++)
        vs[i].plan = NULL;
    plan_free(&pl);
    topo_int_free(&ti);
    rj_op_free(&op);
    rj_topo_free(t);
    *out = vs;
    return nv;
}

/* Part 3: deterministic replay. bench_select on the full list gives
 * real measured times (recovered from gbps); then:
 *  (a) select_by_time on the full vs pruned lists must agree (across
 *      several noise windows),
 *  (b) the full run's winner must be in the pruned set, and
 *  (c) TRIPWIRE (attacker 16): the UNPRUNED run's argmin-time
 *      candidate must NOT be in the pruned set.
 * Deterministic: both selections see the same times, so measurement
 * noise cannot fake a pass or a failure. */
static void replay_agreement(void)
{
    static const double windows[] = { 0.015, 0.03, 0.06 };
    for (size_t ci = 0; ci < NCFG; ci++) {
        const cfg_t *c = &cfgs[ci];
        version_t *full = NULL;
        int nv = build_all(c, &full);
        if (nv <= 0)
            continue;
        char err[256];
        rj_cpu_info cpu;
        rj_cpu_detect(&cpu);
        topo_int_t ti;
        plan_t pl;
        rj_topology *t = c->topo();
        rj_op op = c->op();
        if (topo_int_build(&ti, t, err, sizeof(err)) ||
            plan_build(&ti, &op, "t", &pl, err, sizeof(err))) {
            printf("FAIL replay setup %s: %s\n", c->name, err);
            failures++;
            continue;
        }
        rj_options o;
        rj_options_default(&o);
        o.bench_bytes = c->bench_bytes;
        o.bench_iters = 5;
        int w_full = bench_select(full, nv, &pl, &o, 0, NULL, err, sizeof(err));
        CHECK(w_full >= 0, "%s: bench_select failed: %s", c->name, err);
        if (w_full < 0) {
            plan_free(&pl);
            topo_int_free(&ti);
            rj_op_free(&op);
            rj_topo_free(t);
            continue;
        }
        /* recover relative per-candidate times from measured gbps */
        double *secs = malloc((size_t)nv * sizeof(*secs));
        for (int i = 0; i < nv; i++)
            secs[i] = full[i].gbps > 0 ? 1.0 / full[i].gbps : 1e30;

        /* prune an independent second build (prune_versions finalizes
         * the pruned ones, so the two builds must not share structs) */
        version_t *pv = NULL;
        int nv2 = build_all(c, &pv);
        CHECK(nv2 == nv, "%s: rebuild gave %d vs %d versions", c->name,
              nv2, nv);
        int np = prune_versions(pv, nv2, 0);
        CHECK(np > 0 && np <= nv, "%s: pruned count %d of %d", c->name,
              np, nv);

        /* membership: the full run's winner must survive pruning */
        int wsurvives = find_desc(pv, np, full[w_full].desc);
        CHECK(wsurvives >= 0,
              "%s: full winner '%s' was pruned (real flip!)", c->name,
              full[w_full].desc);

        /* TRIPWIRE (F8): the unpruned run's argmin-time candidate must
         * NOT be in the pruned set. Pruning safety reduces to this
         * precondition: removing a pruned argmin re-centers
         * select_by_time's noise window and flips winners (attacker
         * result: 19,453 flips / 6M fuzz, 100% with pruned argmin).
         * Any future model/codegen change that makes a pruned version
         * the argmin fails LOUDLY here instead of flipping winners
         * silently. (The 1e30 guard skips the check when the argmin is
         * a failed-benchmark sentinel rather than a real
         * measurement.) */
        int argmin = 0;
        for (int i = 1; i < nv; i++)
            if (secs[i] < secs[argmin])
                argmin = i;
        if (secs[argmin] < 1e29) {
            CHECK(find_desc(pv, np, full[argmin].desc) >= 0,
                  "%s: TRIPWIRE: unpruned argmin-time '%s' was pruned",
                  c->name, full[argmin].desc);
        }

        /* map pruned survivors back to times by desc */
        double *psecs = malloc((size_t)np * sizeof(*psecs));
        for (int i = 0; i < np; i++) {
            int fi = find_desc(full, nv, pv[i].desc);
            psecs[i] = fi >= 0 ? secs[fi] : 1e30;
        }
        for (size_t wi = 0; wi < sizeof(windows) / sizeof(windows[0]);
             wi++) {
            int a = select_by_time(full, nv, secs, windows[wi]);
            int b = select_by_time(pv, np, psecs, windows[wi]);
            CHECK(strcmp(full[a].desc, pv[b].desc) == 0,
                  "%s win=%.2f%%: full->'%s' pruned->'%s'", c->name,
                  windows[wi] * 100.0, full[a].desc, pv[b].desc);
        }
        printf("  replay %s: full=%d pruned=%d winner='%s'\n", c->name,
               nv, np, full[w_full].desc);

        for (int i = 0; i < nv; i++)
            jit_free_version(&full[i]);
        for (int i = 0; i < np; i++)
            jit_free_version(&pv[i]);
        free(full);
        free(pv);
        free(secs);
        free(psecs);
        plan_free(&pl);
        topo_int_free(&ti);
        rj_op_free(&op);
        rj_topo_free(t);
    }
}

/* ---------------- Part 4: prune x xorsched non-interference (merge §5b) ---
 *
 * C13 verification pins: A1's rule can never drop an xsched twin or its
 * naive counterpart.
 *  1. Adversarial P_XOR triple: naive (xorsched=0,ternlog=0), xsched twin
 *     (xorsched=1,ternlog=0), +ternlog twin (xorsched=0,ternlog=1), with
 *     the +ternlog twin pareto-dominating the xsched twin on the triple.
 *     prune_versions must drop NOTHING (kind-guard refusal).
 *  2. Mem-asymmetry pin (independent of the kind guard): on P_MULACC, a
 *     +ternlog twin with strictly better uops but strictly worse mem
 *     (the dirty-sharing double-load case: mem_sched > mem_naive) must
 *     not prune the plain counterpart -- pareto requires mem not-worse
 *     (PRUNE_EPS), independent of the kind guard.
 *  3. Hardening: notail/xorsched mismatched pairs never satisfy the twin
 *     relation (pins the C13 field-comparison addition in twin_dominates).
 * Positive control (P_MULACC +ternlog pair still prunes) is covered by
 * Part 1's unit_twin_prunes_plain. */

/* mkv with the xorsched/notail axes (both 0 in Part 1's mkv). */
static void mkvx(version_t *v, plan_kind_t kind, isa_t isa, mul_t mul, int U,
                 int nt, size_t pf, int tl, int xs, int ntl, double uops,
                 double mem, size_t code, const char *desc)
{
    mkv(v, kind, isa, mul, U, nt, pf, tl, uops, mem, code, desc);
    v->cand.xorsched = xs;
    v->cand.notail = ntl;
}

/* Adversarial P_XOR triple: the +ternlog twin pareto-dominates the
 * xsched twin on the (uops, mem, code) triple, yet nothing may be
 * pruned -- the kind guard refuses P_XOR unconditionally, regardless of
 * the triple values. Catches kind-guard removal AND any future attempt
 * to let the triple overrule the guard. */
static void unit_pxor_xsched_triple_survives(void)
{
    version_t vs[3];
    /* naive P_XOR (xorsched=0, ternlog=0) */
    mkvx(&vs[0], P_XOR, ISA_AVX512, MUL_NIBBLE, 4, 0, 0, 0, 0, 0,
         0.1500, 7.0, 600, "naive");
    /* xsched twin (xorsched=1, ternlog=0): better uops via CSE */
    mkvx(&vs[1], P_XOR, ISA_AVX512, MUL_NIBBLE, 4, 0, 0, 0, 1, 0,
         0.1200, 7.0, 580, "xsched");
    /* +ternlog twin (xorsched=0, ternlog=1): adversarially set to
     * pareto-dominate the xsched twin on the triple */
    mkvx(&vs[2], P_XOR, ISA_AVX512, MUL_NIBBLE, 4, 0, 0, 1, 0, 0,
         0.1100, 6.5, 560, "ternlog");
    int n = prune_versions(vs, 3, 0);
    CHECK(n == 3, "P_XOR triple: nothing may be pruned, got %d", n);
    CHECK(find_desc(vs, n, "naive") >= 0, "naive must survive");
    CHECK(find_desc(vs, n, "xsched") >= 0, "xsched twin must survive");
    CHECK(find_desc(vs, n, "ternlog") >= 0, "ternlog twin must survive");
}

/* Mem-asymmetry pin, independent of the kind guard: on P_MULACC (a
 * validated kind), a +ternlog twin with strictly better uops but
 * strictly worse mem must NOT prune plain. pareto requires
 * a->mem_per_byte <= b->mem_per_byte + PRUNE_EPS; the dirty-sharing
 * double-load case (mem_sched > mem_naive, A3 R1-4) is a second,
 * independent barrier behind the kind guard. */
static void unit_mem_asymmetry_no_prune(void)
{
    version_t vs[2];
    mkvx(&vs[0], P_MULACC, ISA_AVX512, MUL_NIBBLE, 8, 0, 0, 0, 0, 0,
         1.502, 32.016, 5824, "plain");
    /* twin: strictly better uops, strictly worse mem */
    mkvx(&vs[1], P_MULACC, ISA_AVX512, MUL_NIBBLE, 8, 0, 0, 1, 0, 0,
         1.346, 33.500, 5440, "twin-memworse");
    int n = prune_versions(vs, 2, 0);
    CHECK(n == 2, "mem-worse twin must not prune, got %d", n);
    CHECK(find_desc(vs, n, "plain") >= 0, "plain must survive");
    CHECK(find_desc(vs, n, "twin-memworse") >= 0, "twin must survive");
}

/* Hardening pin: notail/xorsched mismatched pairs never satisfy the twin
 * relation, even when the triple would dominate. Each pair below is a
 * would-be twin except for one pinned field. */
static void unit_twin_axes_pinned(void)
{
    /* xorsched mismatch: a=(tl=1,xs=1) vs b=(tl=0,xs=0), triple-dominant */
    {
        version_t vs[2];
        mkvx(&vs[0], P_MULACC, ISA_AVX512, MUL_NIBBLE, 8, 0, 0, 0, 0, 0,
             1.502, 32.016, 5824, "plain");
        mkvx(&vs[1], P_MULACC, ISA_AVX512, MUL_NIBBLE, 8, 0, 0, 1, 1, 0,
             1.346, 32.016, 5440, "twin-xs");
        int n = prune_versions(vs, 2, 0);
        CHECK(n == 2, "xorsched-mismatched pair must not prune, got %d",
              n);
    }
    /* notail mismatch: a=(tl=1,notail=1) vs b=(tl=0,notail=0) */
    {
        version_t vs[2];
        mkvx(&vs[0], P_MULACC, ISA_AVX512, MUL_NIBBLE, 8, 0, 0, 0, 0, 0,
             1.502, 32.016, 5824, "plain");
        mkvx(&vs[1], P_MULACC, ISA_AVX512, MUL_NIBBLE, 8, 0, 0, 1, 0, 1,
             1.346, 32.016, 5440, "twin-notail");
        int n = prune_versions(vs, 2, 0);
        CHECK(n == 2, "notail-mismatched pair must not prune, got %d", n);
    }
    /* xorsched twin vs its naive counterpart on P_XOR: neither may
     * prune the other even with adversarial triples (both directions). */
    {
        version_t vs[2];
        mkvx(&vs[0], P_XOR, ISA_AVX512, MUL_NIBBLE, 4, 0, 0, 0, 0, 0,
             0.1500, 7.0, 600, "naive");
        mkvx(&vs[1], P_XOR, ISA_AVX512, MUL_NIBBLE, 4, 0, 0, 0, 1, 0,
             0.1100, 6.0, 560, "xsched-dom");
        int n = prune_versions(vs, 2, 0);
        CHECK(n == 2, "P_XOR naive/xsched pair must not prune, got %d",
              n);
    }
}

int main(void)
{
    printf("prune unit:\n");
    unit_twin_prunes_plain();
    unit_pxor_twin_survives();
    unit_pxor_twin_dominant_survives();
    unit_mulacc_plain_dominant_twin_survives();
    unit_u_variants_survive();
    unit_cross_u_twin_no_prune();
    unit_nt_twins_survive();
    unit_cross_isa_mul_survive();
    unit_prefetch_same_dist();
    unit_remat_no_prune();
    unit_cross_kind_survive();
    unit_plain_without_twin_survives();
    unit_eps_ties_survive();
    unit_twin_not_pareto_survives();
    unit_edge_lists();
    unit_disable();
    unit_order_and_many();
    printf("prune x xorsched non-interference (merge §5b):\n");
    unit_pxor_xsched_triple_survives();
    unit_mem_asymmetry_no_prune();
    unit_twin_axes_pinned();
    printf("prune e2e smoke (rj_compile with/without pruning):\n");
    e2e_smoke();
    printf("prune hook wiring (verbose prune line):\n");
    hook_wiring();
    printf("prune replay (deterministic):\n");
    replay_agreement();
    if (failures) {
        printf("PRUNE TESTS FAILED (%d/%d checks)\n", failures, nchecks);
        return 1;
    }
    printf("prune: all OK (%d checks)\n", nchecks);
    return 0;
}
