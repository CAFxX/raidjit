/* B4 (SOTA item 18): prefetch-dimension tests.
 *
 * 1. Self-attack: prefetch past the buffer end. A prefetch twin with a
 *    huge distance (PF_DIST: 1 MB on x86, 32760 on AArch64, 2016 on
 *    RISC-V -- each ISA's encodable maximum, so the emitted prefetch
 *    genuinely reaches far past the tiny test buffers) run at tiny
 *    lengths must not fault (prefetch hints are non-faulting on
 *    x86/AArch64/RISC-V) and must produce output identical to the
 *    temporal twin. Covers tail-only lengths, exact-group lengths, and
 *    group+1 (last-iteration prefetch reaches far past the end).
 *    On AArch64 the test also scans the emitted code to prove every
 *    PRFM word carries the full PF_DIST offset -- without that check a
 *    silently-wrapped offset (e.g. 1 MB -> 0) would pass vacuously.
 * 2. Planner: prefetch twins are enumerated for every base candidate
 *    (and only when asked); the NT+prefetch cross product is covered in
 *    tests/test_nt.c (test_nt_prefetch_combo).
 * 3. Backend gate (AArch64): unencodable prefetch distances fail the
 *    twin build cleanly ("not encodable"), never silently truncated --
 *    the same rule the RISC-V Zicbop backend enforces
 *    (tests/test_riscv_prefetch.c:test_prefetch_reject).
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

static int fails;
#define CHECK(c, ...) do {                                             \
    if (!(c)) {                                                        \
        printf("FAIL "); printf(__VA_ARGS__); printf("\n"); fails++;   \
    }                                                                  \
} while (0)

/* Self-attack distance: far past the end of the tiny test buffers
 * (max test length 4096). Each ISA uses its encodable maximum so the
 * emitted prefetch genuinely reaches past the end:
 * - RISC-V Zicbop prefetch.r encodes only 32..2016 (multiple of 32);
 * - AArch64 PRFM encodes imm12 x 8 = 0..32760 (multiple of 8); larger
 *   distances are rejected by the backend (test_unencodable_reject),
 *   so a 1 MB distance there would fail the build, not test anything. */
#if defined(__riscv)
#define PF_DIST ((size_t)2016)
#elif defined(__aarch64__)
#define PF_DIST ((size_t)32760)
#else
#define PF_DIST ((size_t)1 << 20)
#endif

#if defined(__aarch64__)
/* PRFM <prfop>, [xRn, #imm12*8]: fixed bits are [31:22] = 0xF98. */
#define PRFM_MASK 0xFFC00000u
#define PRFM_WORD 0xF9800000u
static unsigned prfm_off(uint32_t w) { return ((w >> 10) & 0xfffu) << 3; }

/* The self-attack is only real if the emitted PRFM words actually carry
 * the huge PF_DIST offset: a silently-wrapped offset (e.g. 1 MB -> 0)
 * would make test_past_end pass vacuously. Scan the twin's code and
 * require every PRFM word to encode exactly PF_DIST. */
static void check_prfm_shape(const version_t *v, const char *tname)
{
    int n_prfm = 0;
    int off_ok = 1;
    const uint8_t *code = (const uint8_t *)v->mem;
    for (size_t i = 0; i + 4 <= v->code_size; i += 4) {
        uint32_t w;
        memcpy(&w, code + i, 4);
        if ((w & PRFM_MASK) == PRFM_WORD) {
            n_prfm++;
            if (prfm_off(w) != PF_DIST)
                off_ok = 0;
        }
    }
    CHECK(n_prfm > 0, "%s [%s]: no PRFM words emitted", tname, v->desc);
    CHECK(off_ok, "%s [%s]: PRFM word with wrapped offset (want %zu)",
          tname, v->desc, PF_DIST);
}
#endif

static uint64_t xrng_state = 0x9E3779B97F4A7C15ULL;
static uint64_t xrng(void)
{
    uint64_t x = xrng_state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return xrng_state = x;
}

/* Run fn vs the scalar reference at len; 0 on match. */
static int check_vs_ref(const plan_t *pl, rj_fn fn, int n_in, int n_out,
                        size_t len)
{
    uint8_t *din[64], *dout[64], *ref[64];
    const void *in[64];
    void *out[64];
    for (int i = 0; i < n_in; i++) {
        if (posix_memalign((void **)&din[i], 64, len ? len : 64))
            abort();
        for (size_t k = 0; k < len; k++)
            din[i][k] = (uint8_t)xrng();
        in[i] = din[i];
    }
    for (int j = 0; j < n_out; j++) {
        if (posix_memalign((void **)&dout[j], 64, len ? len : 64))
            abort();
        if (posix_memalign((void **)&ref[j], 64, len ? len : 64))
            abort();
        out[j] = dout[j];
    }
    fn(in, out, len);
#if defined(__x86_64__)
    __asm__ volatile("sfence" ::: "memory");
#elif defined(__aarch64__)
    __asm__ volatile("dmb ishst" ::: "memory");
#endif
    ref_run(pl, (const uint8_t **)in, ref, len);
    int bad = 0;
    for (int j = 0; j < n_out; j++)
        if (memcmp(dout[j], ref[j], len) != 0)
            bad = 1;
    for (int i = 0; i < n_in; i++)
        free(din[i]);
    for (int j = 0; j < n_out; j++) {
        free(dout[j]);
        free(ref[j]);
    }
    return bad;
}

static void test_past_end(const rj_cpu_info *cpu, rj_topology *t, rj_op *op,
                          const char *tname)
{
    char err[256];
    topo_int_t ti;
    if (topo_int_build(&ti, t, err, sizeof err) != 0) {
        CHECK(0, "%s topo: %s", tname, err);
        return;
    }
    plan_t pl;
    if (plan_build(&ti, op, tname, &pl, err, sizeof err) != 0) {
        CHECK(0, "%s plan: %s", tname, err);
        topo_int_free(&ti);
        return;
    }
    /* one prefetch twin per ISA (first candidate of each ISA), huge
     * distance: the attack is "prefetch far past the buffer end". */
    cand_t cands[512];
    int nc = plan_candidates(&pl, cpu, cands, 512, 0, PF_DIST);
    CHECK(nc > 0, "%s: no candidates", tname);
    rj_options opts;
    rj_options_default(&opts);
    opts.benchmark = 0;

    int seen_isa[8] = {0};
    size_t lens[] = {1, 63, 64, 512, 513, 4096};
    for (int ci = 0; ci < nc && !fails; ci++) {
        if (!cands[ci].prefetch_dist || seen_isa[cands[ci].isa])
            continue;
        seen_isa[cands[ci].isa] = 1;
        version_t v;
        memset(&v, 0, sizeof v);
        if (jit_build_version(&pl, &cands[ci], cpu, &opts, &v,
                              err, sizeof err) != 0) {
            CHECK(0, "%s isa=%d: build: %s", tname, cands[ci].isa, err);
            continue;
        }
#if defined(__aarch64__)
        check_prfm_shape(&v, tname);
#endif
        for (size_t li = 0; li < sizeof(lens) / sizeof(lens[0]); li++) {
            /* Must not fault; output must match the scalar reference. */
            CHECK(!check_vs_ref(&pl, (rj_fn)v.fn, pl.n_in, pl.n_out,
                                lens[li]),
                  "%s [%s] len=%zu: prefetch-past-end mismatch (or fault)",
                  tname, v.desc, lens[li]);
        }
        jit_free_version(&v);
    }
    plan_free(&pl);
    topo_int_free(&ti);
}

static void test_enumeration(const rj_cpu_info *cpu)
{
    /* Planner level: prefetch twins exist iff a distance is given, one
     * per base candidate (RVV: only with Zicbop). */
    char err[256];
    rj_topology *t = rj_topo_rs(6, 3);
    topo_int_t ti;
    CHECK(topo_int_build(&ti, t, err, sizeof err) == 0, "topo: %s", err);
    rj_op op = rj_op_encode();
    plan_t pl;
    CHECK(plan_build(&ti, &op, "rs(6,3)", &pl, err, sizeof err) == 0,
          "plan: %s", err);

    cand_t c0[512], c1[512];
    int n0 = plan_candidates(&pl, cpu, c0, 512, 0, 0);
    int n1 = plan_candidates(&pl, cpu, c1, 512, 0, 4096);
    CHECK(n0 > 0 && n1 > 0, "candidates %d/%d", n0, n1);
    int pf0 = 0, pf1 = 0;
    for (int i = 0; i < n0; i++)
        if (c0[i].prefetch_dist)
            pf0++;
    for (int i = 0; i < n1; i++)
        if (c1[i].prefetch_dist)
            pf1++;
    CHECK(pf0 == 0, "distance 0 must enumerate no prefetch twins (%d)", pf0);
#if defined(__riscv)
    if (cpu->rvv_zicbop)
        CHECK(pf1 == n0, "want one prefetch twin per base (%d vs %d)", pf1, n0);
    else
        CHECK(pf1 == 0, "no Zicbop: want no prefetch twins (%d)", pf1);
#else
    CHECK(pf1 == n0, "want one prefetch twin per base (%d vs %d)", pf1, n0);
#endif
    /* every prefetch twin matches a base twin on all other axes */
    for (int i = 0; i < n1 && !fails; i++) {
        if (!c1[i].prefetch_dist)
            continue;
        int found = 0;
        for (int k = 0; k < n0; k++) {
            if (c0[k].isa == c1[i].isa && c0[k].mul == c1[i].mul &&
                c0[k].U == c1[i].U && c0[k].nt == c1[i].nt &&
                c0[k].ternlog == c1[i].ternlog &&
                c0[k].remat == c1[i].remat) {
                found = 1;
                break;
            }
        }
        CHECK(found, "prefetch twin %d has no base twin", i);
    }
    CHECK(c1[0].prefetch_dist == 0 || n1 > n0,
          "prefetch twins must extend the candidate list");
    plan_free(&pl);
    rj_op_free(&op);
    topo_int_free(&ti);
    rj_topo_free(t);
}

#if defined(__aarch64__)
/* Unencodable prefetch distances fail the twin build cleanly (never
 * silently truncated): PRFM imm12 x 8 -> 0..32760, multiple of 8.
 * This is the regression test for the backend gate -- with the gate
 * reverted these builds succeed (offset wrapped) and the CHECKs fire. */
static void test_unencodable_reject(const rj_cpu_info *cpu)
{
    char err[256];
    rj_topology *t = rj_topo_rs(6, 3);
    topo_int_t ti;
    if (topo_int_build(&ti, t, err, sizeof err) != 0) {
        CHECK(0, "reject topo: %s", err);
        rj_topo_free(t);
        return;
    }
    rj_op op = rj_op_encode();
    plan_t pl;
    if (plan_build(&ti, &op, "rs(6,3)", &pl, err, sizeof err) != 0) {
        CHECK(0, "reject plan: %s", err);
        topo_int_free(&ti);
        rj_topo_free(t);
        return;
    }
    /* first prefetch twin of each ISA, to exercise every backend */
    cand_t cands[512];
    int nc = plan_candidates(&pl, cpu, cands, 512, 0, 32760);
    CHECK(nc > 0, "reject: no candidates");
    rj_options opts;
    rj_options_default(&opts);
    opts.benchmark = 0;

    static const size_t bad[] = { 100, 32761, 32768,
                                  (size_t)1 << 20, (size_t)1 << 33 };
    static const size_t good[] = { 8, 32760 };
    int seen_isa[8] = {0};
    for (int ci = 0; ci < nc && !fails; ci++) {
        if (!cands[ci].prefetch_dist || seen_isa[cands[ci].isa])
            continue;
        seen_isa[cands[ci].isa] = 1;
        for (size_t k = 0; k < sizeof(bad) / sizeof(bad[0]); k++) {
            version_t v;
            memset(&v, 0, sizeof v);
            cands[ci].prefetch_dist = bad[k];
            int rc = jit_build_version(&pl, &cands[ci], cpu, &opts, &v,
                                       err, sizeof err);
            CHECK(rc != 0, "isa=%d prefetch_dist=%zu: build should fail",
                  cands[ci].isa, bad[k]);
            if (rc != 0)
                CHECK(strstr(err, "not encodable") != NULL,
                      "isa=%d prefetch_dist=%zu: error should say "
                      "not encodable: %s", cands[ci].isa, bad[k], err);
            else
                jit_free_version(&v);
        }
        for (size_t k = 0; k < sizeof(good) / sizeof(good[0]); k++) {
            version_t v;
            memset(&v, 0, sizeof v);
            cands[ci].prefetch_dist = good[k];
            int rc = jit_build_version(&pl, &cands[ci], cpu, &opts, &v,
                                       err, sizeof err);
            CHECK(rc == 0, "isa=%d prefetch_dist=%zu: build: %s",
                  cands[ci].isa, good[k], err);
            if (rc == 0)
                jit_free_version(&v);
        }
    }
    plan_free(&pl);
    rj_op_free(&op);
    topo_int_free(&ti);
    rj_topo_free(t);
}
#endif

int main(void)
{
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);

    test_enumeration(&cpu);
#if defined(__aarch64__)
    test_unencodable_reject(&cpu);
#endif

    rj_topology *t;
    rj_op op;
    t = rj_topo_raid6(8);
    op = rj_op_encode();
    test_past_end(&cpu, t, &op, "raid6(8) enc");
    rj_op_free(&op);
    rj_topo_free(t);
    t = rj_topo_rs(6, 3);
    op = rj_op_encode();
    test_past_end(&cpu, t, &op, "rs(6,3) enc");
    rj_op_free(&op);
    rj_topo_free(t);
    {
        int f[] = {1, 7};
        t = rj_topo_rs(6, 3);
        op = rj_op_recover(f, 2);
        test_past_end(&cpu, t, &op, "rs(6,3) rec2");
        rj_op_free(&op);
        rj_topo_free(t);
    }

    if (!fails)
        printf("b4_prefetch: all OK\n");
    return fails != 0;
}
