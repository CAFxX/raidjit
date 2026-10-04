/* test_permi2b: correctness + planner gating of the VPERMI2B nibble twin
 * (track B2, SOTA item 15).
 *
 * The twin replaces the two-load vpshufb nibble ping-pong with a 2-table
 * vpermi2b lookup (CK_NIBBOTH interleaved tables, +16 folded via a
 * memory-source VPORB) for AVX-512 MUL_NIBBLE P_MULACC/P_DELTA. Tests:
 *
 *  1. Correctness vs ref_run for every AVX-512 nibble candidate's
 *     permi2b=0 and permi2b=1 versions (incl. ternlog x permi2b where
 *     the planner enumerates the combination), across:
 *       topologies: custom(4,2) and custom(5,3) encodes (P_MULACC,
 *         several distinct nontrivial coefficients -> pool-dedup paths),
 *         rs(6,3) recover (P_MULACC recovery matrix), raid6_rs(6)
 *         parity update (P_DELTA, mixed XOR row + nibble row: the
 *         permi2b branch must fire only for the nontrivial row).
 *       lengths: 0..2*group+7 (group boundaries +/-1, odd lengths,
 *         tails) plus one large odd streaming length.
 *     Buffers are canaried: no overrun, outputs bit-exact.
 *  2. Planner gating: stage-1 (plan_candidates) never enumerates a
 *     permi2b candidate -- the twin is built from the stage-1 winner in
 *     api.c and duelled there (2026-10-03 follow-up; before that it was
 *     enumerated when (AVX-512, MUL_NIBBLE, MULACC/DELTA, VBMI)). What
 *     is pinned: zero permi2b flags out of plan_candidates on a VBMI
 *     host and with vbmi simulated off alike; MULACC/DELTA plans still
 *     enumerate AVX-512 MUL_NIBBLE candidates so the api.c duel is
 *     non-vacuous; RAID-5 (P_XOR) and rs encode (P_HORNER) enumerate
 *     none.
 *  3. desc tags: twin carries "+permi2b", plain does not.
 *  4. Red-green: the comparison harness is shown to REPORT a mismatch
 *     when run against a deliberately wrong reference (wrong-coefficient
 *     topology), proving the check is not tautological.
 *  5. notail x permi2b combo (white-box): api.c builds the notail twin
 *     from the stage-2 winner, which may be the permi2b twin on a host
 *     where it wins its duel. The combo must build and be bit-exact at
 *     group multiples; red-green leg included.
 *  6. E2E: rj_compile with benchmarking on proves the api.c stage-2
 *     duel runs exactly when the stage-1 winner is duel-eligible
 *     (observed via the verbose log), and the installed function is
 *     bit-exact at admitted and tail lengths whichever side won. The
 *     duel's outcome is empirical and is not asserted.
 *  7. Duel mechanics (white-box): replicate the api.c stage-2 block
 *     step by step on a forced AVX-512 nibble winner -- twin build
 *     from the winner's candidate, real bench_select duel, winner
 *     swap on a strict win, notail twin built from the final winner
 *     -- all bit-exact. Deterministic; covers the eligible branch the
 *     public API never takes on this host (GFNI wins stage-1). The
 *     forced-swap leg mirrors the api.c swap verbatim, including the
 *     measured-gbps copy-back (P1-1): it asserts the installed twin
 *     reports the measured gbps (gbps >= 0, gbps_measured == 1, equal
 *     to the duel measurement), not jit_build_version's -1 sentinel.
 *  8. Duel eligibility (white-box, P1-2): api_permi2b_duel_eligible --
 *     the real api.c predicate -- table-tested: an all-trivial P_DELTA
 *     plan (all-{0,1} delta matrix, where the twin is byte-identical
 *     to plain and the duel would be vacuous) must read ineligible
 *     while nontrivial MULACC/DELTA plans stay eligible; every
 *     non-shape axis still gates. Plus an end-to-end compile of the
 *     all-trivial plan asserting the duel never runs there.
 *
 * Skipped loudly (not silently) when the host lacks AVX-512 or VBMI: the
 * twin is never enumerated there by construction.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h> /* pipe/dup2/read for the e2e stderr capture */

#include "../include/raidjit.h"
#include "../src/jit.h"
#include "../src/plan.h"
#include "../src/bench.h"
#include "../src/api.h"
#include "../src/ref.h"

static int failures = 0;
#define CHECK(c, ...) do {                                             \
    if (!(c)) {                                                        \
        printf("FAIL "); printf(__VA_ARGS__); printf("\n"); failures++; \
    }                                                                  \
} while (0)

static void *abuf(size_t n)
{
    void *p = NULL;
    if (posix_memalign(&p, 64, n ? n : 64)) {
        printf("oom\n");
        exit(1);
    }
    return p;
}

/* Deterministic input pattern: covers all nibble values in both halves. */
static void fill_in(uint8_t **in, int n_in, size_t maxlen)
{
    for (int i = 0; i < n_in; i++)
        for (size_t k = 0; k < maxlen; k++)
            in[i][k] = (uint8_t)((i * 131 + k * 17 + (k >> 4) * 5) & 0xff);
}

/* Compare one built version against ref_run at lengths 0..2*group+7 plus
 * one large odd streaming length. Returns the number of
 * mismatches/overwrites found. */
static int check_version(const plan_t *pl, const version_t *v,
                         const char *name, int U, int permi2b)
{
    size_t group = v->group_bytes;
    int n_in = pl->n_in, n_out = pl->n_out;
    size_t maxlen = 2 * group + 7;
    size_t biglen = (1 << 20) + 13; /* large odd streaming length */
    size_t bufsz = biglen + 128;    /* canary zone past the big length */

    uint8_t **in = malloc(sizeof(*in) * (size_t)n_in);
    uint8_t **out = malloc(sizeof(*out) * (size_t)n_out);
    uint8_t **ref = malloc(sizeof(*ref) * (size_t)n_out);
    for (int i = 0; i < n_in; i++) {
        in[i] = abuf(bufsz);
        memset(in[i], 0, bufsz);
    }
    fill_in(in, n_in, biglen);
    for (int j = 0; j < n_out; j++) {
        out[j] = abuf(bufsz);
        ref[j] = abuf(bufsz);
    }

    int bad = 0;
    for (size_t li = 0; li <= 2 * group + 7; li++) {
        size_t len = (li == 2 * group + 7) ? biglen : li;
        /* canary the whole buffer; only [0,len) may change */
        for (int j = 0; j < n_out; j++) {
            memset(out[j], 0xA5, bufsz);
            memset(ref[j], 0xA5, bufsz);
        }
        ref_run(pl, (const uint8_t **)in, ref, len);
        v->fn((const void **)in, (void **)out, len);
        for (int j = 0; j < n_out; j++) {
            if (memcmp(out[j], ref[j], len) != 0) {
                if (bad < 3)
                    printf("FAIL %s U=%d p2b=%d len=%zu out=%d "
                           "mismatch\n", name, U, permi2b, len, j);
                bad++;
            }
            for (size_t k = len; k < len + 64 && k < bufsz; k++) {
                if (out[j][k] != 0xA5) {
                    if (bad < 3)
                        printf("FAIL %s U=%d p2b=%d len=%zu out=%d "
                               "overwrite at k=%zu\n", name, U, permi2b,
                               len, j, k);
                    bad++;
                    break;
                }
            }
        }
        if (bad >= 10)
            break;
    }

    for (int i = 0; i < n_in; i++)
        free(in[i]);
    for (int j = 0; j < n_out; j++) {
        free(out[j]);
        free(ref[j]);
    }
    free(in);
    free(out);
    free(ref);
    (void)maxlen;
    return bad;
}

/* Build and check the permi2b=0/1 versions of every AVX-512 nibble
 * candidate (incl. ternlog combos). want_kind: expected plan kind, or -1
 * to skip the check. */
static void test_one_topo(rj_topology *t, rj_op op, const char *name,
                          int want_kind)
{
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    topo_int_t ti;
    char err[256];
    if (topo_int_build(&ti, t, err, sizeof(err))) {
        printf("FAIL %s topo: %s\n", name, err);
        failures++;
        return;
    }
    plan_t pl;
    if (plan_build(&ti, &op, name, &pl, err, sizeof(err))) {
        printf("FAIL %s plan: %s\n", name, err);
        failures++;
        topo_int_free(&ti);
        return;
    }
    if (want_kind >= 0 && (int)pl.kind != want_kind) {
        printf("FAIL %s plan kind=%d, want %d\n", name, pl.kind,
               want_kind);
        failures++;
        plan_free(&pl);
        topo_int_free(&ti);
        return;
    }

    cand_t cands[160];
    int nc = plan_candidates(&pl, &cpu, cands, 160, 0, 0);
    int ntested = 0, ntwins = 0;
    for (int ci = 0; ci < nc; ci++) {
        if (cands[ci].isa != ISA_AVX512 || cands[ci].mul != MUL_NIBBLE)
            continue;
        for (int p2b = 0; p2b <= 1; p2b++) {
            cand_t c = cands[ci];
            c.permi2b = p2b;
            rj_options o;
            rj_options_default(&o);
            version_t v;
            if (jit_build_version(&pl, &c, &cpu, &o, &v, err,
                                  sizeof(err))) {
                printf("FAIL %s U=%d tl=%d p2b=%d build: %s\n", name, c.U,
                       c.ternlog, p2b, err);
                failures++;
                continue;
            }
            int hastag = strstr(v.desc, "+permi2b") != NULL;
            if (!!hastag != !!p2b) {
                printf("FAIL %s U=%d tl=%d p2b=%d desc tag wrong: %s\n",
                       name, c.U, c.ternlog, p2b, v.desc);
                failures++;
            }
            int bad = check_version(&pl, &v, name, c.U, p2b);
            if (bad) {
                printf("FAIL %s U=%d tl=%d p2b=%d: %d mismatches\n", name,
                       c.U, c.ternlog, p2b, bad);
                failures++;
            } else {
                printf("ok %s U=%d tl=%d p2b=%d [%s]\n", name, c.U,
                       c.ternlog, p2b, v.desc);
            }
            jit_free_version(&v);
            ntested++;
            ntwins += p2b;
        }
    }
    if (!ntested) {
        printf("FAIL %s: no AVX-512 nibble candidates tested\n", name);
        failures++;
    } else if (!ntwins) {
        printf("FAIL %s: no permi2b twins tested\n", name);
        failures++;
    }
    plan_free(&pl);
    topo_int_free(&ti);
}

/* Planner gating checks. Since the 2026-10-03 B2 follow-up, stage-1
 * (plan_candidates) never enumerates permi2b twins: api.c builds the
 * twin from the stage-1 winner and duels it with the standard
 * bench_select protocol, cutting the compile cost from +50% stage-1
 * builds to +1 build + 1 duel (only when the winner is a nibble
 * candidate). Pinned here:
 *  - no candidate ever carries permi2b=1 out of plan_candidates, on a
 *    VBMI host and with vbmi simulated off alike;
 *  - MULACC/DELTA nibble plans still enumerate AVX-512 MUL_NIBBLE
 *    candidates, so the api.c duel's eligibility is non-vacuous;
 *  - P_XOR / P_HORNER plans enumerate no permi2b candidates (as
 *    before). */
static void test_gating(rj_topology *t, rj_op op, const char *name,
                        int want_nibble)
{
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    topo_int_t ti;
    char err[256];
    if (topo_int_build(&ti, t, err, sizeof(err))) {
        printf("FAIL gate %s topo: %s\n", name, err);
        failures++;
        return;
    }
    plan_t pl;
    if (plan_build(&ti, &op, name, &pl, err, sizeof(err))) {
        printf("FAIL gate %s plan: %s\n", name, err);
        failures++;
        topo_int_free(&ti);
        return;
    }
    cand_t cands[160];
    int nc = plan_candidates(&pl, &cpu, cands, 160, 0, 0);
    int ntw = 0, nnibble = 0;
    for (int i = 0; i < nc; i++) {
        if (cands[i].permi2b) {
            ntw++;
            /* If the flag ever reappears in stage-1, it must still be
             * gated to (AVX-512, MUL_NIBBLE, MULACC/DELTA). */
            CHECK(cands[i].isa == ISA_AVX512,
                  "gate %s: permi2b on non-AVX512 isa %d", name,
                  cands[i].isa);
            CHECK(cands[i].mul == MUL_NIBBLE,
                  "gate %s: permi2b on non-nibble mul %d", name,
                  cands[i].mul);
            CHECK(pl.kind == P_MULACC || pl.kind == P_DELTA,
                  "gate %s: permi2b on plan kind %d", name, pl.kind);
        }
        if (cands[i].isa == ISA_AVX512 && cands[i].mul == MUL_NIBBLE)
            nnibble++;
    }
    CHECK(ntw == 0,
          "gate %s: stage-1 enumerated %d permi2b twins (want 0)", name,
          ntw);
    if (want_nibble)
        CHECK(nnibble > 0,
              "gate %s: no AVX-512 nibble candidate (api.c duel vacuous)",
              name);
    printf("gate %s: %d permi2b twins in stage-1 (want 0), %d AVX-512 "
           "nibble candidates\n", name, ntw, nnibble);

    /* vbmi=0 simulation: still no twins (the flag is gone from stage-1
     * on every host now, not just non-VBMI ones). */
    {
        rj_cpu_info novbmi = cpu;
        novbmi.vbmi = 0;
        int nc2 = plan_candidates(&pl, &novbmi, cands, 160, 0, 0);
        int ntw2 = 0;
        for (int i = 0; i < nc2; i++)
            ntw2 += !!cands[i].permi2b;
        CHECK(ntw2 == 0,
              "gate %s: vbmi=0 enumerates %d permi2b twins", name,
              ntw2);
        printf("gate %s: vbmi=0 -> %d twins OK\n", name, ntw2);
    }
    plan_free(&pl);
    topo_int_free(&ti);
}

/* notail x permi2b combo (white-box): api.c builds the notail twin
 * from the stage-2 winner, which may itself be the permi2b twin on a
 * host where it wins its duel. The combo must build and be bit-exact
 * vs the scalar reference at twin-admissible (group-multiple) lengths.
 * Red-green: the same combo compared against a deliberately wrong
 * reference must report mismatches (proves the check discriminates). */
static void test_notail_combo(void)
{
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    char err[256];
    /* true topology: custom(4,2), Q row [1,2,4,8] */
    uint8_t cmat[8] = { 1, 1, 1, 1, 1, 2, 4, 8 };
    /* wrong topology: Q row [1,2,4,9] -- red-green leg */
    uint8_t wmat[8] = { 1, 1, 1, 1, 1, 2, 4, 9 };
    rj_topology *t = rj_topo_custom(4, 2, cmat);
    rj_topology *wt = rj_topo_custom(4, 2, wmat);
    rj_op op = rj_op_encode();
    topo_int_t ti, wti;
    plan_t pl, wpl;
    if (topo_int_build(&ti, t, err, sizeof(err)) ||
        topo_int_build(&wti, wt, err, sizeof(err)) ||
        plan_build(&ti, &op, "ntc", &pl, err, sizeof(err)) ||
        plan_build(&wti, &op, "ntcw", &wpl, err, sizeof(err))) {
        printf("FAIL notail-combo setup: %s\n", err);
        failures++;
        goto out;
    }
    cand_t cands[160];
    int nc = plan_candidates(&pl, &cpu, cands, 160, 0, 0);
    int ci = -1;
    for (int i = 0; i < nc; i++)
        if (cands[i].isa == ISA_AVX512 && cands[i].mul == MUL_NIBBLE &&
            !cands[i].ternlog) {
            ci = i;
            break;
        }
    CHECK(ci >= 0, "notail-combo: no AVX-512 nibble candidate");
    if (ci < 0)
        goto out2;
    {
        cand_t c = cands[ci];
        c.permi2b = 1;
        c.notail = 1;
        rj_options o;
        rj_options_default(&o);
        version_t v;
        if (jit_build_version(&pl, &c, &cpu, &o, &v, err, sizeof(err))) {
            printf("FAIL notail-combo build: %s\n", err);
            failures++;
            goto out2;
        }
        CHECK(strstr(v.desc, "+permi2b") != NULL &&
                  strstr(v.desc, "notail") != NULL,
              "notail-combo desc tags wrong: %s", v.desc);
        size_t group = v.group_bytes;
        int n_in = pl.n_in, n_out = pl.n_out;
        size_t lens[3] = { group, 2 * group, 3 * group };
        size_t bufsz = 3 * group + 64;
        uint8_t **in = malloc(sizeof(*in) * (size_t)n_in);
        uint8_t **out = malloc(sizeof(*out) * (size_t)n_out);
        uint8_t **ref = malloc(sizeof(*ref) * (size_t)n_out);
        for (int i = 0; i < n_in; i++) {
            in[i] = abuf(bufsz);
            memset(in[i], 0, bufsz);
        }
        fill_in(in, n_in, bufsz);
        for (int j = 0; j < n_out; j++) {
            out[j] = abuf(bufsz);
            ref[j] = abuf(bufsz);
        }
        int bad = 0;
        for (int li = 0; li < 3; li++) {
            size_t len = lens[li];
            for (int j = 0; j < n_out; j++) {
                memset(out[j], 0xA5, bufsz);
                memset(ref[j], 0xA5, bufsz);
            }
            ref_run(&pl, (const uint8_t **)in, ref, len);
            v.fn((const void **)in, (void **)out, len);
            for (int j = 0; j < n_out; j++)
                if (memcmp(out[j], ref[j], len) != 0) {
                    printf("FAIL notail-combo len=%zu out=%d mismatch\n",
                           len, j);
                    bad++;
                }
        }
        CHECK(bad == 0, "notail-combo: %d mismatches", bad);
        if (!bad)
            printf("notail-combo: permi2b+notail bit-exact at multiples "
                   "OK [%s]\n", v.desc);
        /* Red-green: the same outputs against the WRONG reference must
         * mismatch, proving the comparison discriminates. */
        for (int j = 0; j < n_out; j++) {
            memset(out[j], 0xA5, bufsz);
            memset(ref[j], 0xA5, bufsz);
        }
        ref_run(&wpl, (const uint8_t **)in, ref, 2 * group);
        v.fn((const void **)in, (void **)out, 2 * group);
        int wbad = 0;
        for (int j = 0; j < n_out; j++)
            wbad += memcmp(out[j], ref[j], 2 * group) != 0;
        CHECK(wbad > 0, "notail-combo red-green: wrong reference produced "
                         "no mismatch (check is tautological)");
        if (wbad > 0)
            printf("notail-combo red-green: wrong reference detected OK\n");
        for (int i = 0; i < n_in; i++)
            free(in[i]);
        for (int j = 0; j < n_out; j++) {
            free(out[j]);
            free(ref[j]);
        }
        free(in);
        free(out);
        free(ref);
        jit_free_version(&v);
    }
out2:
    plan_free(&pl);
    plan_free(&wpl);
    topo_int_free(&ti);
    topo_int_free(&wti);
out:
    rj_topo_free(t);
    rj_topo_free(wt);
    rj_op_free(&op);
}

/* Run rj_compile with stderr captured (api.c logs the permi2b duel's
 * progress there when verbose=1). Returns bytes captured, or -1 on
 * setup failure; *jout is the compile result (NULL on failure). */
static int compile_capture_stderr(rj_topology *t, const rj_op *op,
                                  rj_options *o, rj_jit **jout, char *buf,
                                  size_t buflen, char *err, size_t errlen)
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
    rj_jit *j = rj_compile(t, op, o, err, errlen);
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
    *jout = j;
    return (int)got;
}

/* E2E: the api.c stage-2 permi2b duel. Compile a MULACC nibble plan with
 * benchmarking on and verbose=1. The duel's progress messages
 * ("permi2b twin won its duel" / "permi2b twin lost its duel, dropped")
 * prove the duel ran exactly when the stage-1 winner is duel-eligible
 * (AVX-512, +nibble, MULACC/DELTA -- read off the installed winner's
 * desc; VBMI holds since main() skips without it), and prove it was
 * skipped otherwise. The duel's outcome is empirical (host-dependent)
 * and is NOT asserted -- only that the mechanism fires correctly and
 * the installed function is bit-exact vs the scalar reference at
 * admitted and tail lengths, whichever side won. */
static void test_e2e_duel(void)
{
    char err[256];
    uint8_t cmat[8] = { 1, 1, 1, 1, 1, 2, 4, 8 };
    rj_topology *t = rj_topo_custom(4, 2, cmat);
    CHECK(t != NULL, "e2e topo");
    if (!t)
        return;
    rj_op op = rj_op_encode();
    rj_options o;
    rj_options_default(&o);
    o.benchmark = 1;
    o.bench_bytes = 1 << 16; /* keep the duel fast */
    o.bench_iters = 3;
    o.verbose = 1;

    char log[65536];
    rj_jit *j = NULL;
    int got = compile_capture_stderr(t, &op, &o, &j, log, sizeof(log),
                                     err, sizeof(err));
    CHECK(got >= 0, "e2e: stderr capture failed");
    CHECK(j != NULL, "e2e compile: %s", err);
    if (!j) {
        rj_op_free(&op);
        rj_topo_free(t);
        return;
    }
    rj_info info;
    rj_get_info(j, &info);
    /* Duel eligibility, from the installed winner's desc: api.c's
     * predicate is (AVX-512, MULACC|DELTA, MUL_NIBBLE, vbmi). If the
     * twin won, j->v IS the twin and the desc still shows avx512 /
     * +nibble / mulacc, so the read-off is the same. */
    int eligible = strstr(info.desc, "avx512") != NULL &&
                   strstr(info.desc, "+nibble") != NULL &&
                   (strcmp(info.strategy, "mulacc") == 0 ||
                    strcmp(info.strategy, "delta") == 0);
    int saw_duel = strstr(log, "permi2b twin") != NULL;
    printf("note: e2e winner -> %s [%s, duel %s]\n", info.desc,
           eligible ? "duel-eligible" : "not duel-eligible",
           saw_duel ? "ran" : "did not run");
    if (eligible)
        CHECK(saw_duel, "e2e: stage-1 winner is duel-eligible but the "
                        "permi2b duel did not run: %s", info.desc);
    else
        CHECK(!saw_duel, "e2e: permi2b duel ran for a non-eligible "
                         "winner: %s", info.desc);

    /* Bit-exactness of the installed function (whichever side won),
     * at admitted and tail lengths. */
    rj_fn fn = rj_get_fn(j);
    CHECK(fn != NULL, "e2e: fn non-null");
    if (fn) {
        topo_int_t ti;
        plan_t pl;
        if (topo_int_build(&ti, t, err, sizeof(err)) == 0 &&
            plan_build(&ti, &op, "e2e", &pl, err, sizeof(err)) == 0) {
            int n_in = pl.n_in, n_out = pl.n_out;
            size_t maxlen = 65537 + 64;
            uint8_t **ib = malloc(sizeof(*ib) * (size_t)n_in);
            uint8_t **ob = malloc(sizeof(*ob) * (size_t)n_out);
            uint8_t **rb = malloc(sizeof(*rb) * (size_t)n_out);
            for (int k = 0; k < n_in; k++)
                ib[k] = abuf(maxlen);
            for (int j = 0; j < n_out; j++) {
                ob[j] = abuf(maxlen);
                rb[j] = abuf(maxlen);
            }
            /* Plausible groups so the length matrix hits admitted and
             * tail lengths whatever the winner's group is. */
            size_t lens[] = { 1, 63, 64, 65, 127, 128, 129, 255, 256,
                              257, 511, 512, 513, 65537 };
            int bad = 0;
            for (size_t li = 0; li < sizeof(lens) / sizeof(lens[0]); li++) {
                size_t len = lens[li];
                for (int k = 0; k < n_in; k++)
                    for (size_t b = 0; b < len; b++)
                        ib[k][b] = (uint8_t)((k * 131 + b * 17) & 0xff);
                fn((const void **)ib, (void **)ob, len);
                ref_run(&pl, (const uint8_t **)ib, rb, len);
                for (int j = 0; j < n_out; j++)
                    if (memcmp(ob[j], rb[j], len) != 0) {
                        printf("FAIL e2e len=%zu out=%d mismatch\n", len,
                               j);
                        bad++;
                    }
            }
            CHECK(bad == 0, "e2e: %d mismatches", bad);
            if (!bad)
                printf("e2e: installed function bit-exact OK [%s]\n",
                       info.desc);
            for (int k = 0; k < n_in; k++)
                free(ib[k]);
            for (int j = 0; j < n_out; j++) {
                free(ob[j]);
                free(rb[j]);
            }
            free(ib);
            free(ob);
            free(rb);
            plan_free(&pl);
            topo_int_free(&ti);
        } else {
            printf("FAIL e2e ref plan: %s\n", err);
            failures++;
        }
    }
    rj_free(j);
    rj_op_free(&op);
    rj_topo_free(t);
}

/* White-box replication of the api.c stage-2 permi2b duel mechanics.
 * The e2e test above pins the api.c eligibility predicate (duel runs
 * iff the stage-1 winner is duel-eligible); this pins the duel + swap
 * + tail-from-winner mechanics deterministically, with a forced
 * AVX-512 nibble "winner" (through the public API the GFNI candidate
 * wins stage-1 on this host, so the eligible branch would never be
 * exercised there). Steps mirror src/api.c's stage-2 permi2b block:
 * build the twin from the winner's candidate, duel with the real
 * bench_select protocol, swap on a strict win, then build the notail
 * twin from the final winner exactly like api.c's notail block. */
static void test_duel_mechanics(void)
{
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    char err[256];
    uint8_t cmat[8] = { 1, 1, 1, 1, 1, 2, 4, 8 };
    rj_topology *t = rj_topo_custom(4, 2, cmat);
    rj_op op = rj_op_encode();
    topo_int_t ti;
    plan_t pl;
    if (topo_int_build(&ti, t, err, sizeof(err)) ||
        plan_build(&ti, &op, "duel", &pl, err, sizeof(err))) {
        printf("FAIL duel-mechanics setup: %s\n", err);
        failures++;
        goto out;
    }
    cand_t cands[160];
    int nc = plan_candidates(&pl, &cpu, cands, 160, 0, 0);
    int ci = -1;
    for (int i = 0; i < nc; i++)
        if (cands[i].isa == ISA_AVX512 && cands[i].mul == MUL_NIBBLE &&
            !cands[i].ternlog && !cands[i].permi2b) {
            ci = i;
            break;
        }
    CHECK(ci >= 0, "duel-mechanics: no AVX-512 nibble winner candidate");
    if (ci < 0)
        goto out2;
    {
        rj_options bo;
        rj_options_default(&bo);
        bo.benchmark = 1;
        bo.bench_bytes = 1 << 16;
        bo.bench_iters = 3;
        version_t wver;
        memset(&wver, 0, sizeof(wver));
        if (jit_build_version(&pl, &cands[ci], &cpu, &bo, &wver, err,
                              sizeof(err))) {
            printf("FAIL duel-mechanics winner build: %s\n", err);
            failures++;
            goto out2;
        }
        /* --- mirrors src/api.c: p2c = winner.cand; p2c.permi2b = 1 --- */
        cand_t p2c = wver.cand;
        p2c.permi2b = 1;
        version_t ptwin;
        memset(&ptwin, 0, sizeof(ptwin));
        if (jit_build_version(&pl, &p2c, &cpu, &bo, &ptwin, err,
                              sizeof(err))) {
            printf("FAIL duel-mechanics twin build: %s\n", err);
            failures++;
            jit_free_version(&wver);
            goto out2;
        }
        version_t pair2[2];
        pair2[0] = wver;  /* shallow; still owned by wver */
        pair2[1] = ptwin; /* shallow; owned by ptwin */
        int w3 = bench_select(pair2, 2, &pl, &bo, 0, NULL, err,
                              sizeof(err));
        CHECK(w3 >= 0, "duel-mechanics bench_select failed: %s", err);
        if (w3 < 0) {
            jit_free_version(&wver);
            jit_free_version(&ptwin);
            goto out2;
        }
        printf("note: duel-mechanics: %s won the duel\n",
               w3 == 1 ? "permi2b twin" : "plain winner");
        /* --- mirrors src/api.c: swap on strict win --- */
        version_t final = wver;
        if (w3 == 1) {
            jit_free_version(&wver);
            final = ptwin;
            CHECK(strstr(final.desc, "+permi2b") != NULL,
                  "duel-mechanics: swapped winner lacks +permi2b: %s",
                  final.desc);
        } else {
            jit_free_version(&ptwin);
            CHECK(strstr(final.desc, "+permi2b") == NULL,
                  "duel-mechanics: unswapped winner has +permi2b: %s",
                  final.desc);
        }
        /* The final winner is bit-exact (bench_select validated both). */
        {
            int n_in = pl.n_in, n_out = pl.n_out;
            size_t maxlen = 8192 + 64;
            uint8_t **in = malloc(sizeof(*in) * (size_t)n_in);
            uint8_t **out = malloc(sizeof(*out) * (size_t)n_out);
            uint8_t **ref = malloc(sizeof(*ref) * (size_t)n_out);
            for (int i = 0; i < n_in; i++) {
                in[i] = abuf(maxlen);
                memset(in[i], 0, maxlen);
            }
            fill_in(in, n_in, maxlen);
            for (int j = 0; j < n_out; j++) {
                out[j] = abuf(maxlen);
                ref[j] = abuf(maxlen);
            }
            size_t lens[4] = { 1, 511, 512, 513 };
            int bad = 0;
            for (int li = 0; li < 4; li++) {
                ref_run(&pl, (const uint8_t **)in, ref, lens[li]);
                final.fn((const void **)in, (void **)out, lens[li]);
                for (int j = 0; j < n_out; j++)
                    if (memcmp(out[j], ref[j], lens[li]) != 0)
                        bad++;
            }
            CHECK(bad == 0, "duel-mechanics: final winner %d mismatches",
                  bad);
            /* --- mirrors src/api.c: notail twin from the final winner --- */
            cand_t tc = final.cand;
            tc.notail = 1;
            version_t tver;
            memset(&tver, 0, sizeof(tver));
            rj_options wo;
            rj_options_default(&wo);
            if (jit_build_version(&pl, &tc, &cpu, &wo, &tver, err,
                                  sizeof(err))) {
                printf("FAIL duel-mechanics notail-from-winner build: "
                       "%s\n", err);
                failures++;
            } else {
                size_t g = tver.group_bytes;
                int tbad = 0;
                for (int m = 1; m <= 3; m++) {
                    ref_run(&pl, (const uint8_t **)in, ref, g * (size_t)m);
                    tver.fn((const void **)in, (void **)out,
                            g * (size_t)m);
                    for (int j = 0; j < n_out; j++)
                        if (memcmp(out[j], ref[j], g * (size_t)m) != 0)
                            tbad++;
                }
                CHECK(tbad == 0, "duel-mechanics: notail-from-winner %d "
                                 "mismatches", tbad);
                if (!tbad)
                    printf("duel-mechanics: notail twin from final winner "
                           "bit-exact OK [%s]\n", tver.desc);
                jit_free_version(&tver);
            }
            for (int i = 0; i < n_in; i++)
                free(in[i]);
            for (int j = 0; j < n_out; j++) {
                free(out[j]);
                free(ref[j]);
            }
            free(in);
            free(out);
            free(ref);
        }
        jit_free_version(&final);
        /* Swap-path simulation: force the twin-win branch to pin the
         * swap mechanics (the real duel above never takes it on this
         * host -- the twin loses). Uses fresh builds and a fresh
         * bench_select (which measures into the local pair copies, NOT
         * into w2/t2 -- exactly like api.c); then mirrors the api.c
         * w3 == 1 sequence verbatim, INCLUDING the P1-1 fix: the
         * measured gbps/gbps_measured are copied back into the twin
         * before the swap, since bench_select's measurements land in
         * the pair copies while the twin struct still carries
         * jit_build_version's gbps=-1 / gbps_measured=0 sentinel.
         * If this mirror's copy-back lines are removed (simulating a
         * revert of the api.c fix), the gbps CHECKs below fail.
         * The swapped winner must be fully functional (bit-exact,
         * +permi2b tag, measured gbps) and freeable exactly once. */
        {
            version_t w2, t2;
            memset(&w2, 0, sizeof(w2));
            memset(&t2, 0, sizeof(t2));
            if (jit_build_version(&pl, &cands[ci], &cpu, &bo, &w2, err,
                                  sizeof(err)) ||
                jit_build_version(&pl, &p2c, &cpu, &bo, &t2, err,
                                  sizeof(err))) {
                printf("FAIL duel-mechanics swap-leg build: %s\n", err);
                failures++;
            } else {
                version_t pair3[2];
                pair3[0] = w2;  /* shallow; still owned by w2 */
                pair3[1] = t2; /* shallow; owned by t2 */
                int w3s = bench_select(pair3, 2, &pl, &bo, 0, NULL, err,
                                       sizeof(err));
                CHECK(w3s >= 0,
                      "duel-mechanics swap-leg bench_select failed: %s",
                      err);
                if (w3s < 0) {
                    jit_free_version(&w2);
                    jit_free_version(&t2);
                } else {
                    /* --- mirrors src/api.c's w3 == 1 branch --- */
                    t2.gbps = pair3[1].gbps;
                    t2.gbps_measured = pair3[1].gbps_measured;
                    jit_free_version(&w2);
                    w2 = t2; /* the api.c swap */
                    CHECK(strstr(w2.desc, "+permi2b") != NULL,
                          "duel-mechanics swap: winner lacks +permi2b: %s",
                          w2.desc);
                    /* P1-1: the installed twin must report the measured
                     * throughput, not jit_build_version's sentinel. */
                    CHECK(w2.gbps_measured == 1,
                          "duel-mechanics swap: gbps_measured=%d (want 1): "
                          "%s",
                          w2.gbps_measured, w2.desc);
                    CHECK(w2.gbps >= 0.0,
                          "duel-mechanics swap: gbps=%f still the -1 "
                          "sentinel after swap: %s",
                          w2.gbps, w2.desc);
                    CHECK(w2.gbps == pair3[1].gbps,
                          "duel-mechanics swap: gbps=%f != measured %f "
                          "after swap",
                          w2.gbps, pair3[1].gbps);
                    if (w2.gbps_measured == 1 && w2.gbps >= 0.0)
                        printf("duel-mechanics: forced-swap winner "
                               "reports measured gbps=%.2f OK [%s]\n",
                               w2.gbps, w2.desc);
                int sbad = 0;
                for (int li = 0; li < 4; li++) {
                    size_t L[4] = { 1, 511, 512, 513 };
                    uint8_t **in = malloc(sizeof(*in) * (size_t)pl.n_in);
                    uint8_t **out = malloc(sizeof(*out) * (size_t)pl.n_out);
                    uint8_t **ref = malloc(sizeof(*ref) * (size_t)pl.n_out);
                    for (int i = 0; i < pl.n_in; i++) {
                        in[i] = abuf(8192 + 64);
                        memset(in[i], 0, 8192 + 64);
                    }
                    fill_in(in, pl.n_in, 8192 + 64);
                    for (int j = 0; j < pl.n_out; j++) {
                        out[j] = abuf(8192 + 64);
                        ref[j] = abuf(8192 + 64);
                    }
                    ref_run(&pl, (const uint8_t **)in, ref, L[li]);
                    w2.fn((const void **)in, (void **)out, L[li]);
                    for (int j = 0; j < pl.n_out; j++)
                        if (memcmp(out[j], ref[j], L[li]) != 0)
                            sbad++;
                    for (int i = 0; i < pl.n_in; i++)
                        free(in[i]);
                    for (int j = 0; j < pl.n_out; j++) {
                        free(out[j]);
                        free(ref[j]);
                    }
                    free(in);
                    free(out);
                    free(ref);
                }
                CHECK(sbad == 0,
                      "duel-mechanics swap: swapped winner %d mismatches",
                      sbad);
                if (!sbad)
                    printf("duel-mechanics: forced-swap winner bit-exact "
                           "OK [%s]\n", w2.desc);
                jit_free_version(&w2); /* exactly once: no double-free */
                }
            }
        }
    }
out2:
    plan_free(&pl);
    topo_int_free(&ti);
out:
    rj_topo_free(t);
    rj_op_free(&op);
}

/* Red-green: the check_version harness must REPORT a mismatch when the
 * reference is deliberately wrong (wrong-coefficient topology). A
 * harness that cannot fail is tautological. */
static void test_redgreen(void)
{
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    /* true topology: custom(4,2), Q row [1,2,4,8] */
    uint8_t cmat[8] = { 1, 1, 1, 1, 1, 2, 4, 8 };
    /* wrong topology: Q row [1,2,4,9] -- coefficient flip the harness
     * must catch */
    uint8_t wmat[8] = { 1, 1, 1, 1, 1, 2, 4, 9 };
    rj_topology *t = rj_topo_custom(4, 2, cmat);
    rj_topology *wt = rj_topo_custom(4, 2, wmat);
    rj_op op = rj_op_encode();
    topo_int_t ti, wti;
    char err[256];
    plan_t pl, wpl;
    if (topo_int_build(&ti, t, err, sizeof(err)) ||
        topo_int_build(&wti, wt, err, sizeof(err)) ||
        plan_build(&ti, &op, "rg", &pl, err, sizeof(err)) ||
        plan_build(&wti, &op, "rgw", &wpl, err, sizeof(err))) {
        printf("FAIL redgreen setup: %s\n", err);
        failures++;
        goto out;
    }
    cand_t cands[160];
    int nc = plan_candidates(&pl, &cpu, cands, 160, 0, 0);
    int ci = -1;
    for (int i = 0; i < nc; i++)
        if (cands[i].isa == ISA_AVX512 && cands[i].mul == MUL_NIBBLE &&
            !cands[i].ternlog) {
            ci = i;
            break;
        }
    CHECK(ci >= 0, "redgreen: no AVX-512 nibble candidate");
    if (ci < 0)
        goto out2;
    cand_t c = cands[ci];
    c.permi2b = 1;
    rj_options o;
    rj_options_default(&o);
    version_t v;
    if (jit_build_version(&pl, &c, &cpu, &o, &v, err, sizeof(err))) {
        printf("FAIL redgreen build: %s\n", err);
        failures++;
        goto out2;
    }
    /* Run the twin's output against the WRONG reference: the detector
     * must fire (bad > 0), proving the comparison discriminates. */
    size_t group = v.group_bytes;
    int n_in = pl.n_in, n_out = pl.n_out;
    size_t len = group + 3;
    uint8_t **in = malloc(sizeof(*in) * (size_t)n_in);
    uint8_t **out = malloc(sizeof(*out) * (size_t)n_out);
    uint8_t **ref = malloc(sizeof(*ref) * (size_t)n_out);
    for (int i = 0; i < n_in; i++)
        in[i] = abuf(len);
    fill_in(in, n_in, len);
    for (int j = 0; j < n_out; j++) {
        out[j] = abuf(len);
        ref[j] = abuf(len);
    }
    ref_run(&wpl, (const uint8_t **)in, ref, len); /* wrong reference */
    v.fn((const void **)in, (void **)out, len);
    int bad = 0;
    for (int j = 0; j < n_out; j++)
        if (memcmp(out[j], ref[j], len) != 0)
            bad++;
    CHECK(bad > 0,
          "redgreen: wrong-coefficient reference produced NO mismatch "
          "(harness is tautological)");
    if (bad > 0)
        printf("redgreen: wrong reference detected (%d rows differ) OK\n",
               bad);
    for (int i = 0; i < n_in; i++)
        free(in[i]);
    for (int j = 0; j < n_out; j++) {
        free(out[j]);
        free(ref[j]);
    }
    free(in);
    free(out);
    free(ref);
    jit_free_version(&v);
out2:
    plan_free(&pl);
    plan_free(&wpl);
    topo_int_free(&ti);
    topo_int_free(&wti);
out:
    rj_topo_free(t);
    rj_topo_free(wt);
    rj_op_free(&op);
}

/* B2 P1-2: the stage-2 permi2b duel's nontrivial-coefficient gate.
 * The vpermi2b branch fires only for coefficients outside {0,1}, so on
 * an all-trivial plan (reachable: P_DELTA with an all-{0,1} delta
 * matrix -- plan_build forces P_DELTA for parity_update even then) the
 * twin is byte-identical to plain and the duel is vacuous: the twin
 * could "win" on noise and install under a misleading +permi2b desc.
 * This pins api_permi2b_duel_eligible (the real api.c predicate)
 * table-driven: the all-trivial P_DELTA plan must read ineligible
 * while nontrivial MULACC/DELTA plans with the same winner shape stay
 * eligible, and every non-shape axis still gates. Red-green: removing
 * the plan_any_nontrivial term from the predicate flips the
 * all-trivial case to eligible (shown by mutation run).
 * The e2e leg compiles the all-trivial plan end-to-end and asserts the
 * duel never runs there (no "permi2b twin" progress line) -- the
 * white-box table is the mutation-capable pin, the e2e the public-API
 * one. */
static void test_duel_eligibility(void)
{
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    char err[256];
    uint8_t cmat[8] = { 1, 1, 1, 1, 1, 2, 4, 8 }; /* P=[1]*4, Q=[1,2,4,8] */
    rj_topology *t = rj_topo_custom(4, 2, cmat);
    CHECK(t != NULL, "duel-eligibility topo");
    if (!t)
        return;
    /* The winner shape the api.c block duels: AVX-512 MUL_NIBBLE.
     * api_permi2b_duel_eligible only reads winner->cand. */
    version_t nw;
    memset(&nw, 0, sizeof(nw));
    nw.cand.isa = ISA_AVX512;
    nw.cand.mul = MUL_NIBBLE;

    /* --- all-trivial P_DELTA: changed={0} selects column 0 = [1,1] --- */
    {
        int changed[] = { 0 };
        rj_op op = rj_op_parity_update(changed, 1);
        topo_int_t ti;
        plan_t pl;
        if (topo_int_build(&ti, t, err, sizeof(err)) ||
            plan_build(&ti, &op, "elig", &pl, err, sizeof(err))) {
            printf("FAIL duel-eligibility trivial plan: %s\n", err);
            failures++;
        } else {
            CHECK(pl.kind == P_DELTA,
                  "duel-eligibility: trivial plan kind %d, want P_DELTA",
                  pl.kind);
            CHECK(plan_any_nontrivial(&pl) == 0,
                  "duel-eligibility: trivial plan reports nontrivial "
                  "coefficients");
            CHECK(api_permi2b_duel_eligible(&nw, &pl, &cpu) == 0,
                  "duel-eligibility: duel NOT skipped for all-trivial "
                  "P_DELTA plan (vacuous duel can install a "
                  "byte-identical twin under +permi2b on noise)");
            /* Mechanism premise: on this plan the twin really is
             * byte-identical to plain (the permi2b branch never fires),
             * so a duel win here would install noise under a
             * misleading desc. */
            {
                rj_options bo;
                rj_options_default(&bo);
                bo.benchmark = 0;
                cand_t cands[160];
                int nc = plan_candidates(&pl, &cpu, cands, 160, 0, 0);
                int ci = -1;
                for (int i = 0; i < nc; i++)
                    if (cands[i].isa == ISA_AVX512 &&
                        cands[i].mul == MUL_NIBBLE && !cands[i].ternlog) {
                        ci = i;
                        break;
                    }
                CHECK(ci >= 0,
                      "duel-eligibility: no AVX-512 nibble candidate for "
                      "the all-trivial plan");
                if (ci >= 0) {
                    cand_t tc = cands[ci];
                    tc.permi2b = 1;
                    version_t v0, v1;
                    memset(&v0, 0, sizeof(v0));
                    memset(&v1, 0, sizeof(v1));
                    if (jit_build_version(&pl, &cands[ci], &cpu, &bo,
                                          &v0, err, sizeof(err)) ||
                        jit_build_version(&pl, &tc, &cpu, &bo, &v1, err,
                                          sizeof(err))) {
                        printf("FAIL duel-eligibility twin build: %s\n",
                               err);
                        failures++;
                    } else {
                        CHECK(v0.code_size == v1.code_size &&
                                  memcmp(v0.mem, v1.mem, v0.code_size) ==
                                      0,
                              "duel-eligibility: twin differs from plain "
                              "on the all-trivial plan (the vacuous-duel "
                              "premise is wrong)");
                        CHECK(strstr(v1.desc, "+permi2b") != NULL,
                              "duel-eligibility: twin lacks the +permi2b "
                              "tag whose misleading install this gates");
                        if (v0.code_size == v1.code_size &&
                            memcmp(v0.mem, v1.mem, v0.code_size) == 0)
                            printf("duel-eligibility: all-trivial twin "
                                   "byte-identical to plain (%zu B) "
                                   "despite %s\n",
                                   v0.code_size, v1.desc);
                        jit_free_version(&v0);
                        jit_free_version(&v1);
                    }
                }
            }
            /* End-to-end through the public API: the duel must never
             * run on the all-trivial plan (no "permi2b twin" progress
             * line), whichever side wins stage-1. */
            {
                rj_options o;
                rj_options_default(&o);
                o.benchmark = 1;
                o.bench_bytes = 1 << 16;
                o.bench_iters = 3;
                o.verbose = 1;
                char log[65536];
                rj_jit *j = NULL;
                int got = compile_capture_stderr(t, &op, &o, &j, log,
                                                 sizeof(log), err,
                                                 sizeof(err));
                CHECK(got >= 0, "duel-eligibility e2e: capture failed");
                CHECK(j != NULL, "duel-eligibility e2e compile: %s",
                      err);
                if (j) {
                    int saw_duel = strstr(log, "permi2b twin") != NULL;
                    CHECK(!saw_duel,
                          "duel-eligibility e2e: permi2b duel ran on an "
                          "all-trivial plan");
                    if (!saw_duel)
                        printf("duel-eligibility: e2e duel skipped on "
                               "all-trivial plan OK\n");
                    rj_free(j);
                }
            }
            plan_free(&pl);
            topo_int_free(&ti);
        }
        rj_op_free(&op);
    }

    /* --- nontrivial P_DELTA: changed={1} selects column 1 = [1,2] --- */
    {
        int changed[] = { 1 };
        rj_op op = rj_op_parity_update(changed, 1);
        topo_int_t ti;
        plan_t pl;
        if (topo_int_build(&ti, t, err, sizeof(err)) ||
            plan_build(&ti, &op, "elig", &pl, err, sizeof(err))) {
            printf("FAIL duel-eligibility nontrivial-delta plan: %s\n",
                   err);
            failures++;
        } else {
            CHECK(pl.kind == P_DELTA, "duel-eligibility: delta plan kind "
                                      "%d, want P_DELTA", pl.kind);
            CHECK(plan_any_nontrivial(&pl) != 0,
                  "duel-eligibility: nontrivial plan reports all-trivial");
            CHECK(api_permi2b_duel_eligible(&nw, &pl, &cpu) == 1,
                  "duel-eligibility: duel skipped for a nontrivial "
                  "P_DELTA plan (behavior change)");
            plan_free(&pl);
            topo_int_free(&ti);
        }
        rj_op_free(&op);
    }

    /* --- nontrivial P_MULACC: the duel still runs as before --- */
    {
        rj_op op = rj_op_encode();
        topo_int_t ti;
        plan_t pl;
        if (topo_int_build(&ti, t, err, sizeof(err)) ||
            plan_build(&ti, &op, "elig", &pl, err, sizeof(err))) {
            printf("FAIL duel-eligibility mulacc plan: %s\n", err);
            failures++;
        } else {
            CHECK(pl.kind == P_MULACC, "duel-eligibility: plan kind %d, "
                                       "want P_MULACC", pl.kind);
            CHECK(api_permi2b_duel_eligible(&nw, &pl, &cpu) == 1,
                  "duel-eligibility: duel skipped for a nontrivial "
                  "P_MULACC plan (behavior change)");
            /* every non-shape axis still gates */
            version_t gw = nw;
            gw.cand.mul = MUL_GFNI;
            CHECK(api_permi2b_duel_eligible(&gw, &pl, &cpu) == 0,
                  "duel-eligibility: GFNI winner eligible");
            version_t xw = nw;
            xw.cand.isa = ISA_AVX2;
            CHECK(api_permi2b_duel_eligible(&xw, &pl, &cpu) == 0,
                  "duel-eligibility: AVX2 winner eligible");
            version_t pw = nw;
            pw.cand.permi2b = 1;
            CHECK(api_permi2b_duel_eligible(&pw, &pl, &cpu) == 0,
                  "duel-eligibility: permi2b winner re-duelled");
            rj_cpu_info novbmi = cpu;
            novbmi.vbmi = 0;
            CHECK(api_permi2b_duel_eligible(&nw, &pl, &novbmi) == 0,
                  "duel-eligibility: non-VBMI host eligible");
            plan_free(&pl);
            topo_int_free(&ti);
        }
        rj_op_free(&op);
    }

    /* --- P_XOR: no nibble path, never eligible --- */
    {
        rj_topology *r5 = rj_topo_raid5(4);
        rj_op op = rj_op_encode();
        topo_int_t ti;
        plan_t pl;
        if (topo_int_build(&ti, r5, err, sizeof(err)) ||
            plan_build(&ti, &op, "elig", &pl, err, sizeof(err))) {
            printf("FAIL duel-eligibility xor plan: %s\n", err);
            failures++;
        } else {
            CHECK(pl.kind == P_XOR, "duel-eligibility: plan kind %d, "
                                    "want P_XOR", pl.kind);
            CHECK(api_permi2b_duel_eligible(&nw, &pl, &cpu) == 0,
                  "duel-eligibility: P_XOR winner eligible");
            plan_free(&pl);
            topo_int_free(&ti);
        }
        rj_op_free(&op);
        rj_topo_free(r5);
    }
    rj_topo_free(t);
}

int main(void)
{
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    if (!cpu.avx512f || !cpu.avx512bw) {
        printf("SKIP: host lacks AVX-512F/BW\n");
        return 0;
    }
    if (!cpu.vbmi) {
        printf("SKIP: host lacks VBMI (the twin's instructions need it)\n");
        return 0;
    }

    /* 1. correctness: topologies x lengths x U x {plain,permi2b} x
     * {noternlog,ternlog} */
    {
        uint8_t cmat[8] = { 1, 1, 1, 1, 1, 2, 4, 8 };
        rj_topology *rc = rj_topo_custom(4, 2, cmat);
        rj_op op = rj_op_encode();
        test_one_topo(rc, op, "custom(4,2) encode", P_MULACC);
        rj_op_free(&op);
        rj_topo_free(rc);
    }
    {
        uint8_t cmat[15] = { 1, 1, 1, 1, 1,
                             1, 2, 3, 5, 7,
                             8, 4, 2, 1, 9 };
        rj_topology *rc = rj_topo_custom(5, 3, cmat);
        rj_op op = rj_op_encode();
        test_one_topo(rc, op, "custom(5,3) encode", P_MULACC);
        rj_op_free(&op);
        rj_topo_free(rc);
    }
    {
        /* P_MULACC recovery matrix (solved, non-Vandermonde) */
        rj_topology *rs = rj_topo_rs(6, 3);
        int failed[] = { 0, 1, 2 };
        rj_op op = rj_op_recover(failed, 3);
        test_one_topo(rs, op, "rs(6,3) recover{0,1,2}", P_MULACC);
        rj_op_free(&op);
        rj_topo_free(rs);
    }
    {
        /* P_DELTA: mixed XOR row (P) + nibble row (Q); the permi2b
         * branch must fire only for the nontrivial row. */
        rj_topology *r6 = rj_topo_raid6_rs(6);
        int changed[] = { 0, 2 };
        rj_op op = rj_op_parity_update(changed, 2);
        test_one_topo(r6, op, "raid6_rs(6) delta", P_DELTA);
        rj_op_free(&op);
        rj_topo_free(r6);
    }

    /* 2. planner gating */
    {
        uint8_t cmat[8] = { 1, 1, 1, 1, 1, 2, 4, 8 };
        rj_topology *rc = rj_topo_custom(4, 2, cmat);
        rj_op op = rj_op_encode();
        test_gating(rc, op, "custom(4,2) encode", 1);
        rj_op_free(&op);
        rj_topo_free(rc);
    }
    {
        rj_topology *r6 = rj_topo_raid6_rs(6);
        int changed[] = { 1 };
        rj_op op = rj_op_parity_update(changed, 1);
        test_gating(r6, op, "raid6_rs(6) delta", 1);
        rj_op_free(&op);
        rj_topo_free(r6);
    }
    {
        /* RAID-5 is P_XOR: no nibble path, no twins */
        rj_topology *r5 = rj_topo_raid5(4);
        rj_op op = rj_op_encode();
        test_gating(r5, op, "raid5 encode", 0);
        rj_op_free(&op);
        rj_topo_free(r5);
    }
    {
        /* rs encode is P_HORNER: permi2b does not apply */
        rj_topology *rs = rj_topo_rs(6, 3);
        rj_op op = rj_op_encode();
        test_gating(rs, op, "rs(6,3) encode", 0);
        rj_op_free(&op);
        rj_topo_free(rs);
    }

    /* 3. red-green */
    test_redgreen();

    /* 4. notail x permi2b combo */
    test_notail_combo();

    /* 5. e2e: the api.c stage-2 duel */
    test_e2e_duel();

    /* 6. duel mechanics (white-box) */
    test_duel_mechanics();

    /* 7. duel eligibility (white-box + e2e): the nontrivial-coefficient
     * gate */
    test_duel_eligibility();

    if (failures)
        printf("test_permi2b: %d FAILURES\n", failures);
    else
        printf("test_permi2b: all ok\n");
    return failures != 0;
}
