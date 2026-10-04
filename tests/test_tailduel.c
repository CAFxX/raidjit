/* test_tailduel: the tail-duel install gate (track B1x86, src/bench.c).
 *
 * 1. Unit (portable, deterministic): tail_duel_decide with synthetic
 *    medians -- a strict tail win outside the noise window installs; a
 *    tie, a within-window win, or a large-size regression drops.
 * 2. E2E (AVX-512F+BW only): bench_tail_duel on a real maskedtail twin vs
 *    the general version installs (the twin wins clearly at tail lengths
 *    on this host); dueling a version against itself does NOT install
 *    (tie -> drop). Plus an rj_compile end-to-end proving the dispatcher
 *    actually installs the masked twin ("+maskedtail" in info.desc).
 * 3. Wiring (portable, deterministic): api_tail_duel_gate (src/api.c)
 *    with mock duel runners -- a duel win installs the twin into
 *    twins[], a loss drops it (freed, not installed), an error is fatal.
 *    This pins the SVE predtail twin's install path (track B1sve):
 *    without the gate the twin was installed whenever the notail twin
 *    won the large-size shootout, with no tail-length measurement.
 *
 * Red-green: the decision unit tests fail if tail_duel_decide's
 * strictness is relaxed (see the header comment for the mutations);
 * the wiring tests fail if the gate ignores the duel result (e.g. an
 * unconditional install fails the loss case; a dropped twin that is
 * not freed fails under ASan).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "../src/bench.h"
#include "../src/api.h"
#include "../src/jit.h"
#include "../src/plan.h"
#include "../src/cpu.h"
#include "../include/raidjit.h"

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

/* ---------------- 1. decision unit tests ---------------- */

/* Red-green mutations this guards (all must flip at least one CHECK):
 * (a) `st < sg * (1.0 - window)` -> `st <= sg * (1.0 + window)` (tie
 *     installs): the tie/within-window cases fail.
 * (b) drop the large-size regression check: the regression case fails.
 * (c) `lt <= lg * (1.0 + tol)` -> unwindowed `<`: the
 *     within-tolerance-large case fails.
 * (d) rewrite the `!(...)` fail-closed comparisons as direct ones
 *     (e.g. `st >= sg * (1.0 - window)`): a NaN comparison is false,
 *     so the direct form skips the drop and can install -- the NaN
 *     cases below fail. */
static void test_decide(void)
{
    /* clear strict win on every decisive length, tie at large: install */
    {
        double tm[] = { 1.0, 1.0, 1.0 };
        double gm[] = { 2.0, 2.0, 2.0 };
        double tl[] = { 1.0, 1.0 };
        double gl[] = { 1.0, 1.0 };
        CHECK(tail_duel_decide(tm, gm, 3, tl, gl, 2, 0.015) == 1,
              "strict win installs");
    }
    /* exact tie on the decisive set: drop (no strict win) */
    {
        double tm[] = { 1.0, 1.0, 1.0 };
        double gm[] = { 1.0, 1.0, 1.0 };
        double tl[] = { 1.0 };
        double gl[] = { 1.0 };
        CHECK(tail_duel_decide(tm, gm, 3, tl, gl, 1, 0.015) == 0,
              "tie drops");
    }
    /* win inside the noise window: drop */
    {
        double tm[] = { 0.99, 0.99 };
        double gm[] = { 1.0, 1.0 };
        double tl[] = { 1.0 };
        double gl[] = { 1.0 };
        CHECK(tail_duel_decide(tm, gm, 2, tl, gl, 1, 0.015) == 0,
              "within-window win drops");
    }
    /* win just outside the window: install (boundary) */
    {
        double tm[] = { 0.98, 0.98 };
        double gm[] = { 1.0, 1.0 };
        double tl[] = { 1.0 };
        double gl[] = { 1.0 };
        /* 0.98 < 1.0 * (1 - 0.015) = 0.985 */
        CHECK(tail_duel_decide(tm, gm, 2, tl, gl, 1, 0.015) == 1,
              "just-outside-window win installs");
    }
    /* strict tail win but a large-size regression: drop. The gate is on
     * the SUM of the large lengths: a consistent 10% slowdown across
     * two lengths fails. */
    {
        double tm[] = { 1.0, 1.0, 1.0 };
        double gm[] = { 2.0, 2.0, 2.0 };
        double tl[] = { 1.10, 1.10 };
        double gl[] = { 1.0, 1.0 };
        CHECK(tail_duel_decide(tm, gm, 3, tl, gl, 2, 0.015) == 0,
              "large regression drops");
    }
    /* strict tail win, large slower-but-within-tolerance: install.
     * (A single noisy length is diluted by the sum; the 5% floor keeps
     * the sanity check from flaking on tiny measured windows.) */
    {
        double tm[] = { 1.0, 1.0 };
        double gm[] = { 2.0, 2.0 };
        double tl[] = { 1.04 };
        double gl[] = { 1.0 };
        CHECK(tail_duel_decide(tm, gm, 2, tl, gl, 1, 0.015) == 1,
              "within-tolerance large installs");
    }
    /* twin slower on tails: drop (sanity) */
    {
        double tm[] = { 3.0, 3.0 };
        double gm[] = { 2.0, 2.0 };
        double tl[] = { 1.0 };
        double gl[] = { 1.0 };
        CHECK(tail_duel_decide(tm, gm, 2, tl, gl, 1, 0.015) == 0,
              "slower twin drops");
    }
    /* NaN anywhere fails closed (drop): the !(...) comparisons are
     * NaN-safe by construction (a NaN comparison is false, and the
     * negation drops the twin). A NaN score must never install. */
    {
        double nan = NAN;
        double tm[] = { 1.0, nan, 1.0 };
        double gm[] = { 2.0, 2.0, 2.0 };
        double tl[] = { 1.0, 1.0 };
        double gl[] = { 1.0, 1.0 };
        CHECK(tail_duel_decide(tm, gm, 3, tl, gl, 2, 0.015) == 0,
              "NaN in twin medians drops");
    }
    {
        double nan = NAN;
        double tm[] = { 1.0, 1.0, 1.0 };
        double gm[] = { 2.0, nan, 2.0 };
        double tl[] = { 1.0, 1.0 };
        double gl[] = { 1.0, 1.0 };
        CHECK(tail_duel_decide(tm, gm, 3, tl, gl, 2, 0.015) == 0,
              "NaN in general medians drops");
    }
    {
        double nan = NAN;
        double tm[] = { 1.0, 1.0, 1.0 };
        double gm[] = { 2.0, 2.0, 2.0 };
        double tl[] = { 1.0, nan };
        double gl[] = { 1.0, 1.0 };
        CHECK(tail_duel_decide(tm, gm, 3, tl, gl, 2, 0.015) == 0,
              "NaN in twin large drops");
    }
    {
        double nan = NAN;
        double tm[] = { 1.0, 1.0, 1.0 };
        double gm[] = { 2.0, 2.0, 2.0 };
        double tl[] = { 1.0, 1.0 };
        double gl[] = { 1.0, nan };
        CHECK(tail_duel_decide(tm, gm, 3, tl, gl, 2, 0.015) == 0,
              "NaN in general large drops");
    }
}

/* ---------------- 2. e2e (AVX-512F+BW only) ---------------- */

static int have_avx512(void)
{
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    return cpu.avx512f && cpu.avx512bw;
}

/* Build a P_XOR AVX-512 plan and its general + maskedtail versions. */
static int build_pair(plan_t *pl, version_t *gen, version_t *twin,
                      char *err, size_t errlen)
{
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    rj_topology *t = rj_topo_raid5(4);
    rj_op op = rj_op_encode();
    topo_int_t ti;
    if (topo_int_build(&ti, t, err, errlen)) {
        rj_topo_free(t);
        return -1;
    }
    if (plan_build(&ti, &op, "duel", pl, err, errlen)) {
        topo_int_free(&ti);
        rj_topo_free(t);
        return -1;
    }
    rj_op_free(&op);
    topo_int_free(&ti);
    rj_topo_free(t);
    if (pl->kind != P_XOR) {
        snprintf(err, errlen, "plan kind=%d, want P_XOR", pl->kind);
        plan_free(pl);
        return -1;
    }
    cand_t cands[64];
    int nc = plan_candidates(pl, &cpu, cands, 64, 0, 0);
    rj_options o;
    rj_options_default(&o);
    int got = 0;
    for (int i = 0; i < nc && !got; i++) {
        if (cands[i].isa != ISA_AVX512 || cands[i].U != 8)
            continue;
        char e2[256];
        if (jit_build_version(pl, &cands[i], &cpu, &o, gen, e2,
                              sizeof(e2)))
            continue;
        cand_t mc = cands[i];
        mc.maskedtail = 1;
        if (jit_build_version(pl, &mc, &cpu, &o, twin, e2, sizeof(e2))) {
            jit_free_version(gen);
            continue;
        }
        got = 1;
    }
    if (!got) {
        snprintf(err, errlen, "no buildable AVX-512 U=8 candidate");
        plan_free(pl);
        return -1;
    }
    return 0;
}

static void test_duel_e2e(void)
{
    char err[256];
    plan_t pl;
    version_t gen, twin;
    memset(&gen, 0, sizeof(gen));
    memset(&twin, 0, sizeof(twin));
    CHECK(build_pair(&pl, &gen, &twin, err, sizeof(err)) == 0,
          "build pair: %s", err);
    if (failures)
        return;

    rj_options o;
    rj_options_default(&o);
    /* Use the default bench_iters (7): the duel's noise window needs
     * enough samples to be robust; iters=5 flakes on VM noise (the
     * large-size no-regression gate trips on outliers). */

    /* must-install: the real masked twin wins clearly at tail lengths */
    int r = bench_tail_duel(&twin, &gen, &pl, &o, 0, err, sizeof(err));
    if (r == 0) {
        /* Retry once: the gate is DESIGNED to drop on noise (the safe
         * direction), so a spurious drop on a loaded VM is not a code
         * bug. A second loss is a real failure. Sensitivity cost of the
         * retry (accepted): a genuine twin regression that only just
         * loses could pass on a lucky second run -- but the twin wins
         * by ~2-30x here, far from the boundary, so the retry only
         * filters noise, not real regressions. If a future twin's
         * margin ever shrinks toward the noise window, remove the
         * retry rather than let it mask a real loss. */
        r = bench_tail_duel(&twin, &gen, &pl, &o, 0, err, sizeof(err));
    }
    CHECK(r == 1, "real duel installs (r=%d err='%s')", r, err);
    CHECK(twin.gbps_measured == 1 && twin.gbps > 0,
          "duel sets twin gbps (%f)", twin.gbps);

    /* must-not-install: a version dueled against itself ties -> drop */
    r = bench_tail_duel(&gen, &gen, &pl, &o, 0, err, sizeof(err));
    CHECK(r == 0, "self-duel drops (r=%d err='%s')", r, err);

    jit_free_version(&gen);
    jit_free_version(&twin);
    plan_free(&pl);
}

/* rj_compile end-to-end: the dispatcher installs the masked twin.
 * Returns 1 if installed, 0 if dropped, -1 if rj_compile failed. */
static int compile_installs_maskedtail(char *err, size_t errlen)
{
    rj_topology *t = rj_topo_raid5(4);
    rj_op op = rj_op_encode();
    rj_options o;
    rj_options_default(&o);
    o.benchmark = 1;
    o.bench_bytes = 1 << 16;
    /* default bench_iters: the tail duel needs the full sample count
     * for a robust noise window (see test_duel_e2e). */
    rj_jit *j = rj_compile(t, &op, &o, err, errlen);
    int installed = -1;
    if (j) {
        rj_info info;
        rj_get_info(j, &info);
        printf("note: compile e2e desc: %s\n", info.desc);
        installed = strstr(info.desc, "maskedtail") != NULL ? 1 : 0;
        rj_free(j);
    }
    rj_op_free(&op);
    rj_topo_free(t);
    return installed;
}

static void test_compile_e2e(void)
{
    char err[256];
    int installed = compile_installs_maskedtail(err, sizeof(err));
    if (installed == 0) {
        /* Retry once: the tail duel is DESIGNED to drop on noise (the
         * safe direction), so a spurious drop on a loaded VM is not a
         * code bug. A second drop is a real failure. */
        installed = compile_installs_maskedtail(err, sizeof(err));
    }
    CHECK(installed == 1, "dispatcher installed the masked twin (%s)",
          installed == -1 ? err : "twin dropped");
}

/* ---------------- 3. install-gate wiring (portable, mock duel) ---------------- */

/* Mock duel runners: they record the twin/general pointers they were
 * called with (proving the gate invokes the duel on the twin path --
 * the production runner is bench_tail_duel) and return a canned
 * verdict. The twin's fn is never called, so no ISA codegen runs. */
static const version_t *mock_tw_seen, *mock_gen_seen;
static int mock_calls;

static int mock_duel_win(version_t *twin, version_t *general,
                         const plan_t *pl, const rj_options *opts,
                         int verbose, char *err, size_t errlen)
{
    (void)pl;
    (void)opts;
    (void)verbose;
    (void)err;
    (void)errlen;
    mock_tw_seen = twin;
    mock_gen_seen = general;
    mock_calls++;
    twin->gbps = 42.0; /* the duel sets the twin's tail-length GB/s */
    return 1;
}

static int mock_duel_lose(version_t *twin, version_t *general,
                          const plan_t *pl, const rj_options *opts,
                          int verbose, char *err, size_t errlen)
{
    (void)pl;
    (void)opts;
    (void)verbose;
    (void)err;
    (void)errlen;
    mock_tw_seen = twin;
    mock_gen_seen = general;
    mock_calls++;
    return 0;
}

static int mock_duel_error(version_t *twin, version_t *general,
                           const plan_t *pl, const rj_options *opts,
                           int verbose, char *err, size_t errlen)
{
    (void)twin;
    (void)general;
    (void)pl;
    (void)opts;
    (void)verbose;
    mock_calls++;
    snprintf(err, errlen, "mock duel failure");
    return -1;
}

/* Red-green: an unconditional install (ignoring the duel result) fails
 * the loss case; a gate that never invokes the duel fails the
 * mock_calls checks; a gate that leaks the dropped twin fails under
 * ASan (the twin is freed exactly once by the gate). */
static void test_gate_wiring(void)
{
    char err[256];
    rj_options o;
    rj_options_default(&o);

    /* duel win -> twin installed at twins[0], duel invoked on (twin,
     * general), duel-set gbps carried into the installed twin. */
    {
        version_t twins[2];
        memset(twins, 0, sizeof(twins));
        int ntwins = 0;
        version_t tw, gen;
        memset(&tw, 0, sizeof(tw));
        memset(&gen, 0, sizeof(gen));
        snprintf(tw.desc, sizeof(tw.desc), "winner predtail");
        tw.group_bytes = 512;
        gen.group_bytes = 512;
        mock_calls = 0;
        mock_tw_seen = mock_gen_seen = NULL;
        int r = api_tail_duel_gate(&tw, &gen, NULL, &o, 0, twins,
                                   &ntwins, mock_duel_win, "predtail",
                                   err, sizeof(err));
        CHECK(r == 0, "duel win installs at index 0 (got %d)", r);
        CHECK(ntwins == 1, "win: ntwins 0->1 (got %d)", ntwins);
        CHECK(mock_calls == 1, "win: duel invoked exactly once");
        CHECK(mock_tw_seen == &tw && mock_gen_seen == &gen,
              "win: duel invoked with (twin, general)");
        CHECK(strcmp(twins[0].desc, "winner predtail") == 0,
              "win: twin adopted into twins[] ('%s')", twins[0].desc);
        CHECK(twins[0].gbps == 42.0,
              "win: duel-set gbps carried over (%f)", twins[0].gbps);
        jit_free_version(&twins[0]);
    }
    /* duel loss -> twin dropped (freed by the gate), not installed. */
    {
        version_t twins[2];
        memset(twins, 0, sizeof(twins));
        int ntwins = 0;
        version_t tw, gen;
        memset(&tw, 0, sizeof(tw));
        memset(&gen, 0, sizeof(gen));
        mock_calls = 0;
        int r = api_tail_duel_gate(&tw, &gen, NULL, &o, 0, twins,
                                   &ntwins, mock_duel_lose, "predtail",
                                   err, sizeof(err));
        CHECK(r == -1, "duel loss drops the twin (got %d)", r);
        CHECK(ntwins == 0, "loss: ntwins unchanged (got %d)", ntwins);
        CHECK(mock_calls == 1, "loss: duel still invoked (gate ran)");
    }
    /* duel error -> fatal: nothing installed, twin consumed by the
     * gate (caller must fail compilation). */
    {
        version_t twins[2];
        memset(twins, 0, sizeof(twins));
        int ntwins = 0;
        version_t tw, gen;
        memset(&tw, 0, sizeof(tw));
        memset(&gen, 0, sizeof(gen));
        mock_calls = 0;
        int r = api_tail_duel_gate(&tw, &gen, NULL, &o, 0, twins,
                                   &ntwins, mock_duel_error, "predtail",
                                   err, sizeof(err));
        CHECK(r == -2, "duel error is fatal (got %d)", r);
        CHECK(ntwins == 0, "error: nothing installed (got %d)", ntwins);
        CHECK(mock_calls == 1, "error: duel invoked (gate ran)");
    }
    /* a pre-existing twin (e.g. the notail twin in slot 0) keeps its
     * slot; the duel-winning twin appends after it. */
    {
        version_t twins[2];
        memset(twins, 0, sizeof(twins));
        int ntwins = 1;
        snprintf(twins[0].desc, sizeof(twins[0].desc), "winner notail");
        version_t tw, gen;
        memset(&tw, 0, sizeof(tw));
        memset(&gen, 0, sizeof(gen));
        snprintf(tw.desc, sizeof(tw.desc), "winner predtail");
        mock_calls = 0;
        int r = api_tail_duel_gate(&tw, &gen, NULL, &o, 0, twins,
                                   &ntwins, mock_duel_win, "predtail",
                                   err, sizeof(err));
        CHECK(r == 1, "win appends at index 1 (got %d)", r);
        CHECK(ntwins == 2, "win: ntwins 1->2 (got %d)", ntwins);
        CHECK(strcmp(twins[0].desc, "winner notail") == 0,
              "existing twin keeps slot 0 ('%s')", twins[0].desc);
        CHECK(strcmp(twins[1].desc, "winner predtail") == 0,
              "gated twin takes slot 1 ('%s')", twins[1].desc);
        jit_free_version(&twins[0]);
        jit_free_version(&twins[1]);
    }
}

int main(void)
{
    test_decide();
    test_gate_wiring();
    if (!have_avx512()) {
        printf("SKIP e2e: need AVX-512F+BW\n");
    } else {
        test_duel_e2e();
        test_compile_e2e();
    }
    printf(failures ? "FAIL test_tailduel\n" : "PASS test_tailduel\n");
    return failures != 0;
}
