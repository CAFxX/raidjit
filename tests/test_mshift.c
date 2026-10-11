/* test_mshift: correctness + planner gating of the VPMULTISHIFTQB
 * high-nibble twin (track I2, SOTA item 2), plus AVX512_BITAlg CPUID
 * detection (track I1's corrected gate).
 *
 * The twin replaces the nibble split's VPSRLW with a VPMULTISHIFTQB
 * (AVX512_VBMI): thi[k] = (t's lane >> (8j+4)) & 0xFF per 64-bit lane,
 * i.e. {(t[k+1] & 0x0F), (t[k] >> 4)}; the trailing 0x0F AND stays.
 * Same 3-uop split either way -- the win, if any, is port placement
 * (shuffle port vs shift port), so the plan_candidates gate is closed
 * (strict uop win never fires) and the twin is built directly here.
 * Tests:
 *
 *  1. Correctness vs ref_run for every AVX-512 nibble candidate's
 *     mshift=0 and mshift=1 versions (incl. ternlog x mshift, which the
 *     planner would twin if the gate opened), across:
 *       topologies: custom(4,2) and custom(5,3) encodes (P_MULACC),
 *         rs(6,3) recover (P_MULACC recovery matrix), raid6_rs(6)
 *         parity update (P_DELTA, mixed XOR row + nibble row: the
 *         multishift branch must fire only for the nontrivial row).
 *       lengths: 0..2*group+7 (group boundaries +/-1, odd lengths,
 *         tails) plus one large odd streaming length.
 *     Buffers are canaried: no overrun, outputs bit-exact.
 *  2. Planner gating: plan_candidates never enumerates an mshift
 *     candidate (the strict-uop-win gate is closed: both splits are 3
 *     uops), on a VBMI host and with vbmi simulated off alike; the
 *     +mshift desc tag appears exactly on mshift=1 builds.
 *  3. Red-green: the comparison harness REPORTS a mismatch when run
 *     against a deliberately wrong reference (wrong-coefficient
 *     topology), proving the check is not tautological.
 *  4. BITAlg detection: rj_cpu_detect's bitalg bit matches an
 *     independent CPUID leaf-7 ECX bit-12 read with the same AVX512F +
 *     OS-XCR0 gating (track I1: VPSHUFBITQMB is AVX512_BITAlg, not
 *     VBMI -- gating it on vbmi alone would #UD on e.g. Cannon Lake).
 *
 * Skipped loudly (not silently) when the host lacks AVX-512 or VBMI:
 * the twin's instruction needs VBMI to execute.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "../include/raidjit.h"
#include "../src/jit.h"
#include "../src/plan.h"
#include "../src/ref.h"
#include "../src/gf.h" /* rs_default_matrix (Vandermonde P_HORNER topo) */

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
                         const char *name, int U, int mshift)
{
    size_t group = v->group_bytes;
    int n_in = pl->n_in, n_out = pl->n_out;
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
                    printf("FAIL %s U=%d mshift=%d len=%zu out=%d "
                           "mismatch\n", name, U, mshift, len, j);
                bad++;
            }
            for (size_t k = len; k < len + 64 && k < bufsz; k++) {
                if (out[j][k] != 0xA5) {
                    if (bad < 3)
                        printf("FAIL %s U=%d mshift=%d len=%zu out=%d "
                               "overwrite at k=%zu\n", name, U, mshift,
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
    return bad;
}

/* Build and check the mshift=0/1 versions of every AVX-512 nibble
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
        for (int ms = 0; ms <= 1; ms++) {
            cand_t c = cands[ci];
            c.mshift = ms;
            rj_options o;
            rj_options_default(&o);
            version_t v;
            if (jit_build_version(&pl, &c, &cpu, &o, &v, err,
                                  sizeof(err))) {
                printf("FAIL %s U=%d tl=%d mshift=%d build: %s\n", name,
                       c.U, c.ternlog, ms, err);
                failures++;
                continue;
            }
            int hastag = strstr(v.desc, "+mshift") != NULL;
            if (!!hastag != !!ms) {
                printf("FAIL %s U=%d tl=%d mshift=%d desc tag wrong: %s\n",
                       name, c.U, c.ternlog, ms, v.desc);
                failures++;
            }
            int bad = check_version(&pl, &v, name, c.U, ms);
            if (bad) {
                printf("FAIL %s U=%d tl=%d mshift=%d: %d mismatches\n",
                       name, c.U, c.ternlog, ms, bad);
                failures++;
            } else {
                printf("ok %s U=%d tl=%d mshift=%d [%s]\n", name, c.U,
                       c.ternlog, ms, v.desc);
            }
            jit_free_version(&v);
            ntested++;
            ntwins += ms;
        }
    }
    if (!ntested) {
        printf("FAIL %s: no AVX-512 nibble candidates tested\n", name);
        failures++;
    } else if (!ntwins) {
        printf("FAIL %s: no mshift twins tested\n", name);
        failures++;
    }
    plan_free(&pl);
    topo_int_free(&ti);
}

/* Planner gating: the I2 strict-uop-win gate is closed (both splits are
 * 3 uops), so plan_candidates must never enumerate mshift=1 -- on a
 * VBMI host and with vbmi simulated off alike. */
static void test_gating(rj_topology *t, rj_op op, const char *name)
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
    int ntw = 0;
    for (int i = 0; i < nc; i++)
        ntw += !!cands[i].mshift;
    CHECK(ntw == 0,
          "gate %s: plan_candidates enumerated %d mshift twins (want 0; "
          "the I2 strict-uop-win gate is closed)", name, ntw);
    printf("gate %s: %d mshift twins enumerated (want 0)\n", name, ntw);

    /* vbmi=0 simulation: still none. */
    {
        rj_cpu_info novbmi = cpu;
        novbmi.vbmi = 0;
        int nc2 = plan_candidates(&pl, &novbmi, cands, 160, 0, 0);
        int ntw2 = 0;
        for (int i = 0; i < nc2; i++)
            ntw2 += !!cands[i].mshift;
        CHECK(ntw2 == 0,
              "gate %s: vbmi=0 enumerates %d mshift twins", name, ntw2);
        printf("gate %s: vbmi=0 -> %d twins OK\n", name, ntw2);
    }
    plan_free(&pl);
    topo_int_free(&ti);
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
    {
        cand_t c = cands[ci];
        c.mshift = 1;
        rj_options o;
        rj_options_default(&o);
        version_t v;
        if (jit_build_version(&pl, &c, &cpu, &o, &v, err, sizeof(err))) {
            printf("FAIL redgreen build: %s\n", err);
            failures++;
            goto out2;
        }
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
        ref_run(&wpl, (const uint8_t **)in, ref, len); /* wrong ref */
        v.fn((const void **)in, (void **)out, len);
        int bad = 0;
        for (int j = 0; j < n_out; j++)
            if (memcmp(out[j], ref[j], len) != 0)
                bad++;
        CHECK(bad > 0,
              "redgreen: wrong-coefficient reference produced NO mismatch "
              "(harness is tautological)");
        if (bad > 0)
            printf("redgreen: wrong reference detected (%d rows differ) "
                   "OK\n", bad);
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

#if defined(__x86_64__)
/* Independent CPUID read for the bitalg cross-check (does not share code
 * with src/x86_64/cpu.c). */
static void test_cpuid(unsigned leaf, unsigned sub,
                       unsigned *a, unsigned *b, unsigned *c, unsigned *d)
{
    __asm__ volatile("cpuid"
                     : "=a"(*a), "=b"(*b), "=c"(*c), "=d"(*d)
                     : "a"(leaf), "c"(sub));
}

static unsigned long long test_xgetbv(unsigned c)
{
    unsigned lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(c));
    return ((unsigned long long)hi << 32) | lo;
}
#endif

/* BITAlg detection (track I1's corrected gate): rj_cpu_detect's bitalg
 * bit must match an independent CPUID leaf-7 ECX bit-12 read with the
 * same AVX512F-HW + OS-XCR0 gating the detector applies. VPSHUFBITQMB is
 * AVX512_BITAlg, not VBMI: gating it on vbmi alone would #UD on CPUs
 * with VBMI but no BITAlg (e.g. Cannon Lake). */
static void test_bitalg_detect(void)
{
#if !defined(__x86_64__)
    printf("SKIP bitalg-detect: not x86-64\n");
    return;
#else
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    unsigned a, b, c, d;
    test_cpuid(0, 0, &a, &b, &c, &d);
    int expect = 0;
    if (a >= 7) {
        test_cpuid(7, 0, &a, &b, &c, &d);
        int bit12 = (c >> 12) & 1;
        int f_hw = (b >> 16) & 1;
        test_cpuid(1, 0, &a, &b, &c, &d);
        int osxsave = (c >> 27) & 1;
        unsigned long long xcr0 = osxsave ? test_xgetbv(0) : 0;
        int os_ok = ((xcr0 & 0xE6) == 0xE6);
        expect = bit12 && f_hw && os_ok;
        printf("note: CPUID leaf7 ECX bit12=%d avx512f_hw=%d xcr0_ok=%d\n",
               bit12, f_hw, os_ok);
    }
    CHECK(cpu.bitalg == expect,
          "bitalg-detect: rj_cpu_detect reports bitalg=%d, independent "
          "CPUID read says %d", cpu.bitalg, expect);
    if (cpu.bitalg == expect)
        printf("bitalg-detect: rj_cpu_detect bitalg=%d matches CPUID OK\n",
               cpu.bitalg);
#endif
}

int main(void)
{
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);

    /* BITAlg detection has no VBMI requirement; run it first so a
     * non-VBMI host still pins the I1 gate. */
    test_bitalg_detect();

    if (!cpu.avx512f || !cpu.avx512bw) {
        printf("SKIP: host lacks AVX-512F/BW\n");
        return failures != 0;
    }
    if (!cpu.vbmi) {
        printf("SKIP: host lacks VBMI (the twin's instructions need it)\n");
        return failures != 0;
    }

    /* 1. correctness: topologies x lengths x U x {plain,mshift} x
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
        /* P_DELTA: mixed XOR row (P) + nibble row (Q); the multishift
         * branch must fire only for the nontrivial row. */
        rj_topology *r6 = rj_topo_raid6_rs(6);
        int changed[] = { 0, 2 };
        rj_op op = rj_op_parity_update(changed, 2);
        test_one_topo(r6, op, "raid6_rs(6) delta", P_DELTA);
        rj_op_free(&op);
        rj_topo_free(r6);
    }
    {
        /* P_HORNER: a custom Vandermonde topology (rs_default_matrix)
         * exercises the Horner nibble-split site (split of acc, not t).
         * (rj_topo_rs now emits A2 2.0 searched non-Vandermonde matrices
         * -> P_MULACC; the historical default matrix keeps P_HORNER.) */
        uint8_t mat[12];
        rs_default_matrix(6, 2, mat);
        rj_topology *rs = rj_topo_custom(6, 2, mat);
        rj_op op = rj_op_encode();
        test_one_topo(rs, op, "vandermonde(6,2) encode", P_HORNER);
        rj_op_free(&op);
        rj_topo_free(rs);
    }

    /* 2. planner gating (closed gate) */
    {
        uint8_t cmat[8] = { 1, 1, 1, 1, 1, 2, 4, 8 };
        rj_topology *rc = rj_topo_custom(4, 2, cmat);
        rj_op op = rj_op_encode();
        test_gating(rc, op, "custom(4,2) encode");
        rj_op_free(&op);
        rj_topo_free(rc);
    }

    /* 3. red-green */
    test_redgreen();

    if (failures)
        printf("FAILURES: %d\n", failures);
    else
        printf("ALL OK\n");
    return failures != 0;
}
