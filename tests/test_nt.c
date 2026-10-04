/* Non-temporal store variants: correctness + codegen-shape tests.
 *
 * For every candidate's NT twin (plan_candidates with allow_nt=1):
 *   1. output matches the scalar GF(2^8) reference at scalar, boundary,
 *      multi-group and large lengths (buffers 64-byte aligned, canaried),
 *      with the test performing the caller-side fence (SFENCE / DMB ISHST)
 *      after each NT call, modeling the documented caller contract;
 *   2. the generated code contains NO fence instruction. x86 NT stores
 *      (MOVNTDQ/VMOVNTDQ) and AArch64 NT stores (STNP/STNT1B) are weakly
 *      ordered and the jitted code deliberately emits no fence: callers
 *      that read the NT-written data must fence themselves (SFENCE /
 *      DMB ISHST). RISC-V NT is the Zihintntl NTL.ALL hint, which does
 *      not change memory ordering, so no caller fence is needed there.
 *      The NT twin must still contain at least one NT store marker
 *      (x86: opcode E7 with 66 prefix; AArch64: STNP/STNT1B words;
 *      RISC-V: NTL.ALL words, one per vector store);
 *   3. the temporal twin of the same candidate contains no NT marker.
 * Covers encode and recover ops, all ISA levels the synthetic CPU offers.
 *
 * B4 extensions (SOTA item 18):
 *   4. large-length NT recovery (2 MiB, rs(6,3) double failure): the NT
 *      regime is multi-MiB cold streaming; the per-length sweep above
 *      tops out at 65537 B. First NT twin per ISA only.
 *   5. wider recovery failure patterns: all-parity, data+parity,
 *      rs(12,3) double failure, raid6(8) double failure.
 *   6. prefetch x NT combination: the planner must enumerate all four
 *      (nt, prefetch) combos, and every NT+prefetch twin must produce
 *      correct output while honoring the fence discipline (plus >= 1
 *      prefetch marker in the emitted code).
 */
#include "../include/raidjit.h"
#include "../src/plan.h"
#include "../src/jit.h"
#include "../src/gf.h"
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

/* Caller-side fence for the NT contract (docs/limitations.md §9): the
 * jitted NT code deliberately emits no fence, so the caller must fence
 * after the call before touching NT-written outputs. The test models a
 * correct caller: fence after every NT invocation. */
static inline void caller_fence(void)
{
#if defined(__x86_64__)
    __asm__ volatile("sfence" ::: "memory");
#elif defined(__aarch64__)
    __asm__ volatile("dmb ishst" ::: "memory");
#else
    /* RISC-V NTL.ALL is a hint: no ordering change, but keep a compiler
     * barrier so the test's loads cannot migrate above the call. */
    __asm__ volatile("" ::: "memory");
#endif
}

static uint64_t xrng_state = 0x243F6A8885A308D3ULL;
static uint64_t xrng(void)
{
    uint64_t x = xrng_state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return xrng_state = x;
}

/* NT store = opcode E7 with a 66 prefix in one of our three forms:
 *   66 [REX] 0F E7 (SSE) | VEX.66.0F E7 (C4/C5 .. E7) |
 *   EVEX.66.0F E7 (62 .. E7).
 * Count E7 bytes plausibly positioned as the opcode. This is a shape
 * check only; exact decode is covered by test_scan_align. */
static int count_nt_stores(const uint8_t *code, size_t n)
{
    int c = 0;
#if defined(__aarch64__)
    /* AArch64 NT stores: STNP (Qt) and STNT1B. Word-aligned scan. */
    for (size_t i = 0; i + 4 <= n; i += 4) {
        uint32_t w = (uint32_t)code[i] | ((uint32_t)code[i + 1] << 8) |
                     ((uint32_t)code[i + 2] << 16) |
                     ((uint32_t)code[i + 3] << 24);
        if ((w & 0xFFC00000u) == 0xAC000000u)
            c++; /* STNP */
        else if ((w & 0xFFFFE000u) == 0xE410E000u)
            c++; /* STNT1B */
    }
#elif defined(__riscv)
    /* RVV NT marker: Zihintntl NTL.ALL = 0x00500033, one per vector
     * store. Word-aligned scan. */
    for (size_t i = 0; i + 4 <= n; i += 4) {
        uint32_t w = (uint32_t)code[i] | ((uint32_t)code[i + 1] << 8) |
                     ((uint32_t)code[i + 2] << 16) |
                     ((uint32_t)code[i + 3] << 24);
        if (w == 0x00500033u)
            c++;
    }
#else
    for (size_t i = 0; i < n; i++) {
        if (code[i] != 0xE7)
            continue;
        if (i >= 2 && code[i - 1] == 0x0F) {
            if (code[i - 2] == 0x66)
                c++; /* SSE: 66 0F E7 */
            else if (i >= 3 && code[i - 2] >= 0x40 &&
                     code[i - 2] <= 0x4F && code[i - 3] == 0x66)
                c++; /* SSE: 66 REX 0F E7 */
        } else if (i >= 3 && code[i - 3] == 0xC4 &&
                   (code[i - 1] & 0x03) == 0x01)
            c++; /* VEX 3-byte: C4 .. E7, pp=01 */
        else if (i >= 2 && code[i - 2] == 0xC5 &&
                 (code[i - 1] & 0x03) == 0x01)
            c++; /* VEX 2-byte: C5 .. E7, pp=01 */
        else if (i >= 4 && code[i - 4] == 0x62 &&
                 (code[i - 3] & 0x03) == 0x01)
            c++; /* EVEX: 62 P0 .. E7, P0.pp=01 */
    }
#endif
    return c;
}

/* count fence instructions that the NT contract forbids in jitted code */
static int count_fences(const uint8_t *code, size_t n)
{
    int c = 0;
#if defined(__aarch64__)
    /* any DMB: the jitted code must not fence; the caller does */
    for (size_t i = 0; i + 4 <= n; i += 4) {
        uint32_t w = (uint32_t)code[i] | ((uint32_t)code[i + 1] << 8) |
                     ((uint32_t)code[i + 2] << 16) |
                     ((uint32_t)code[i + 3] << 24);
        if ((w & 0xFFFFF0FFu) == 0xD50330BFu)
            c++;
    }
#elif defined(__riscv)
    /* any FENCE (0x0FF0000F): the jitted code must not fence; on RVV
     * the NTL.ALL hint needs no caller fence either */
    for (size_t i = 0; i + 4 <= n; i += 4) {
        uint32_t w = (uint32_t)code[i] | ((uint32_t)code[i + 1] << 8) |
                     ((uint32_t)code[i + 2] << 16) |
                     ((uint32_t)code[i + 3] << 24);
        if (w == 0x0FF0000Fu)
            c++;
    }
#else
    static const uint8_t sfence[] = {0x0F, 0xAE, 0xF8};
    for (size_t i = 0; i + 3 <= n; i++)
        if (code[i] == sfence[0] && code[i + 1] == sfence[1] &&
            code[i + 2] == sfence[2])
            c++;
#endif
    return c;
}

static void test_one(const rj_cpu_info *cpu, rj_topology *t,
                     const char *tname, int nfailed, const int *failed)
{
    char err[256];
    topo_int_t ti;
    if (topo_int_build(&ti, t, err, sizeof err) != 0) {
        CHECK(0, "%s topo: %s", tname, err);
        return;
    }
    rj_op op = nfailed ? rj_op_recover(failed, nfailed) : rj_op_encode();
    plan_t pl;
    if (plan_build(&ti, &op, tname, &pl, err, sizeof err) != 0) {
        CHECK(0, "%s plan: %s", tname, err);
        rj_op_free(&op); topo_int_free(&ti);
        return;
    }
    /* encode plan for the recover reference stripe */
    plan_t ple;
    rj_op ope = rj_op_encode();
    if (nfailed && plan_build(&ti, &ope, tname, &ple, err, sizeof err) != 0) {
        CHECK(0, "%s enc plan: %s", tname, err);
        rj_op_free(&ope); rj_op_free(&op);
        plan_free(&pl); topo_int_free(&ti);
        return;
    }
    cand_t cands[128];
    int nc = plan_candidates(&pl, cpu, cands, 128, 1, 0);
    CHECK(nc > 0, "%s: no candidates", tname);

    rj_options opts;
    rj_options_default(&opts);
    opts.benchmark = 0;

    for (int ci = 0; ci < nc; ci++) {
        if (!cands[ci].nt)
            continue; /* NT twins only here */
        version_t v;
        memset(&v, 0, sizeof v);
        if (jit_build_version(&pl, &cands[ci], cpu, &opts, &v,
                              err, sizeof err) != 0) {
            CHECK(0, "%s NT cand isa=%d U=%d: build: %s", tname,
                  cands[ci].isa, cands[ci].U, err);
            continue;
        }
        /* shape: NO fence (caller-fence contract), >=1 NT store
         * (non-noop plans) */
        if (!pl.noop) {
            int nf = count_fences(v.mem, v.code_size);
            CHECK(nf == 0, "%s [%s]: NT code must not contain a fence (caller fences), found %d",
                  tname, v.desc, nf);
            int nnt = count_nt_stores(v.mem, v.code_size);
            CHECK(nnt >= 1, "%s [%s]: want >=1 NT store, found %d",
                  tname, v.desc, nnt);
        }
        /* temporal twin must have no NT store */
        {
            cand_t tc = cands[ci];
            tc.nt = 0;
            version_t tv;
            memset(&tv, 0, sizeof tv);
            if (jit_build_version(&pl, &tc, cpu, &opts, &tv,
                                  err, sizeof err) == 0) {
                int nnt = count_nt_stores(tv.mem, tv.code_size);
                CHECK(nnt == 0, "%s [%s]: temporal twin has NT store",
                      tname, tv.desc);
                jit_free_version(&tv);
            }
        }

        /* correctness vs scalar reference.
         * Encode: compare vs ref_run on the same plan.
         * Recover: encode a full stripe with the encode plan, wipe the
         * failed disks, recover into them, compare with the pristine
         * copies; survivors must be untouched. */
        int nd = ti.ndata, np = ti.npar, N = nd + np;
        int vl = cands[ci].isa == ISA_SVE ? cpu->sve_vl : isa_vl(cands[ci].isa);
        int group = cands[ci].U * vl;
        size_t lens[6] = {1, (size_t)(group > 1 ? group - 1 : 1),
                          (size_t)group, (size_t)group + 1, 4096, 65537};
        for (int li = 0; li < 6 && !fails; li++) {
            size_t len = lens[li];
            size_t alen = len ? len : 64;
            uint8_t *all[64], *keep[64];
            for (int i = 0; i < N; i++) {
                if (posix_memalign((void **)&all[i], 64, alen))
                    abort();
                if (posix_memalign((void **)&keep[i], 64, alen))
                    abort();
                if (i < nd)
                    for (size_t k = 0; k < len; k++)
                        all[i][k] = (uint8_t)xrng();
            }
            if (!nfailed) {
                const void *in[64];
                void *out[64];
                uint8_t *exp[64];
                for (int i = 0; i < nd; i++)
                    in[i] = all[i];
                for (int j = 0; j < np; j++) {
                    memset(all[nd + j], 0x5A, alen);
                    out[j] = all[nd + j];
                    if (posix_memalign((void **)&exp[j], 64, alen))
                        abort();
                }
                v.fn(in, out, len);
                caller_fence(); /* NT contract: caller fences before reading */
                ref_run(&pl, (const uint8_t **)in, exp, len);
                for (int j = 0; j < np; j++)
                    CHECK(memcmp(all[nd + j], exp[j], len) == 0,
                          "%s [%s] len=%zu parity%d mismatch", tname,
                          v.desc, len, j);
                for (int j = 0; j < np; j++)
                    free(exp[j]);
            } else {
                /* full stripe, then wipe failed disks */
                const void *ein[64];
                void *eout[64];
                for (int i = 0; i < nd; i++)
                    ein[i] = all[i];
                for (int j = 0; j < np; j++)
                    eout[j] = all[nd + j];
                ref_run(&ple, (const uint8_t **)ein, (uint8_t **)eout,
                        len);
                for (int i = 0; i < N; i++)
                    memcpy(keep[i], all[i], len);
                for (int f = 0; f < nfailed; f++)
                    memset(all[failed[f]], 0xA5, alen);
                const void *in[64];
                void *out[64];
                int ni = 0, fi = 0, nf = 0;
                for (int dsk = 0; dsk < N; dsk++) {
                    if (fi < nfailed && failed[fi] == dsk) {
                        out[nf++] = all[dsk];
                        fi++;
                    } else {
                        in[ni++] = all[dsk];
                    }
                }
                v.fn(in, out, len);
                caller_fence(); /* NT contract: caller fences before reading */
                for (int f = 0; f < nfailed; f++)
                    CHECK(memcmp(all[failed[f]], keep[failed[f]], len) == 0,
                          "%s [%s] recover disk%d len=%zu mismatch", tname,
                          v.desc, failed[f], len);
                for (int dsk = 0; dsk < N; dsk++) {
                    int isf = 0;
                    for (int f = 0; f < nfailed; f++)
                        if (failed[f] == dsk)
                            isf = 1;
                    if (!isf)
                        CHECK(memcmp(all[dsk], keep[dsk], len) == 0,
                              "%s [%s] survivor disk%d modified", tname,
                              v.desc, dsk);
                }
            }
            for (int i = 0; i < N; i++) {
                free(all[i]);
                free(keep[i]);
            }
        }
        jit_free_version(&v);
        if (fails)
            break;
    }
    if (nfailed)
        plan_free(&ple);
    rj_op_free(&ope);
    plan_free(&pl);
    rj_op_free(&op);
    topo_int_free(&ti);
}

/* Recover-leg correctness core shared by the B4 extensions: build a full
 * stripe with the encode plan, wipe the failed disks, run the version,
 * take the caller-side fence (the NT contract), then compare the failed
 * disks against the pristine copies and assert survivors are untouched.
 * `tag` names the calling test for failure messages. */
static void check_recover(const version_t *v, const plan_t *ple,
                          const topo_int_t *ti, const int *failed, int nfailed,
                          size_t len, const char *tag)
{
    int nd = ti->ndata, np = ti->npar, N = nd + np;
    uint8_t *all[64], *keep[64];
    for (int i = 0; i < N; i++) {
        if (posix_memalign((void **)&all[i], 64, len))
            abort();
        if (posix_memalign((void **)&keep[i], 64, len))
            abort();
        if (i < nd)
            for (size_t k = 0; k < len; k++)
                all[i][k] = (uint8_t)xrng();
    }
    const void *ein[64];
    void *eout[64];
    for (int i = 0; i < nd; i++)
        ein[i] = all[i];
    for (int j = 0; j < np; j++)
        eout[j] = all[nd + j];
    ref_run(ple, (const uint8_t **)ein, (uint8_t **)eout, len);
    for (int i = 0; i < N; i++)
        memcpy(keep[i], all[i], len);
    for (int f = 0; f < nfailed; f++)
        memset(all[failed[f]], 0xA5, len);
    const void *in[64];
    void *out[64];
    int ni = 0, fi = 0, nf = 0;
    for (int dsk = 0; dsk < N; dsk++) {
        if (fi < nfailed && failed[fi] == dsk) {
            out[nf++] = all[dsk];
            fi++;
        } else {
            in[ni++] = all[dsk];
        }
    }
    v->fn(in, out, len);
    caller_fence(); /* NT contract: caller fences before reading */
    for (int f = 0; f < nfailed; f++)
        CHECK(memcmp(all[failed[f]], keep[failed[f]], len) == 0,
              "%s [%s] disk%d len=%zu mismatch", tag, v->desc, failed[f],
              len);
    for (int dsk = 0; dsk < N; dsk++) {
        int isf = 0;
        for (int f = 0; f < nfailed; f++)
            if (failed[f] == dsk)
                isf = 1;
        if (!isf)
            CHECK(memcmp(all[dsk], keep[dsk], len) == 0,
                  "%s [%s] survivor disk%d modified", tag, v->desc, dsk);
    }
    for (int i = 0; i < N; i++) {
        free(all[i]);
        free(keep[i]);
    }
}

/* Large-length NT recovery (track B4, SOTA item 18): the NT regime is
 * multi-MiB cold streaming; test_one's sweep tops out at 65537 B.
 * First NT twin per ISA only (one build per ISA keeps this fast).
 * Data integrity, survivors untouched, fence discipline. */
static void test_nt_large_recover(const rj_cpu_info *cpu)
{
    char err[256];
    rj_topology *t = rj_topo_rs(6, 3);
    if (!t) {
        CHECK(0, "large: topo");
        return;
    }
    topo_int_t ti;
    if (topo_int_build(&ti, t, err, sizeof err) != 0) {
        CHECK(0, "large topo: %s", err);
        rj_topo_free(t);
        return;
    }
    int failed[] = {1, 7};
    rj_op op = rj_op_recover(failed, 2);
    plan_t pl;
    if (plan_build(&ti, &op, "rs(6,3)", &pl, err, sizeof err) != 0) {
        CHECK(0, "large plan: %s", err);
        rj_op_free(&op);
        topo_int_free(&ti);
        rj_topo_free(t);
        return;
    }
    rj_op ope = rj_op_encode();
    plan_t ple;
    if (plan_build(&ti, &ope, "rs(6,3)", &ple, err, sizeof err) != 0) {
        CHECK(0, "large enc plan: %s", err);
        rj_op_free(&ope);
        rj_op_free(&op);
        plan_free(&pl);
        topo_int_free(&ti);
        rj_topo_free(t);
        return;
    }
    cand_t cands[512];
    int nc = plan_candidates(&pl, cpu, cands, 512, 1, 0);
    CHECK(nc > 0, "large: no candidates");
    rj_options opts;
    rj_options_default(&opts);
    opts.benchmark = 0;

    int seen_isa[8] = {0};
    size_t len = (size_t)2 << 20;
    int ntested = 0;
    for (int ci = 0; ci < nc && !fails; ci++) {
        if (!cands[ci].nt || seen_isa[cands[ci].isa])
            continue;
        seen_isa[cands[ci].isa] = 1;
        ntested++;
        version_t v;
        memset(&v, 0, sizeof v);
        if (jit_build_version(&pl, &cands[ci], cpu, &opts, &v,
                              err, sizeof err) != 0) {
            CHECK(0, "large isa=%d: build: %s", cands[ci].isa, err);
            continue;
        }
        if (!pl.noop) {
            CHECK(count_fences(v.mem, v.code_size) == 0,
                  "large [%s]: NT code must not contain a fence", v.desc);
            CHECK(count_nt_stores(v.mem, v.code_size) >= 1,
                  "large [%s]: want >=1 NT store", v.desc);
        }
        check_recover(&v, &ple, &ti, failed, 2, len, "large");
        jit_free_version(&v);
    }
    CHECK(ntested > 0, "large: no NT twin tested");
    plan_free(&ple);
    rj_op_free(&ope);
    plan_free(&pl);
    rj_op_free(&op);
    topo_int_free(&ti);
    rj_topo_free(t);
}

/* Prefetch marker scan (shape check only, like count_nt_stores):
 * x86 PREFETCHT0 = 0F 18 /1; AArch64 PRFM PLDL1STRM word;
 * RISC-V Zicbop prefetch.r word. */
static int count_prefetches(const uint8_t *code, size_t n)
{
    int c = 0;
#if defined(__aarch64__)
    for (size_t i = 0; i + 4 <= n; i += 4) {
        uint32_t w = (uint32_t)code[i] | ((uint32_t)code[i + 1] << 8) |
                     ((uint32_t)code[i + 2] << 16) |
                     ((uint32_t)code[i + 3] << 24);
        /* PRFM (immediate): 0xF9800000 | (off/8 << 10) | (rn << 5) |
         * prfop; PLDL1STRM prfop = 1. */
        if ((w & 0xFFC0001Fu) == (0xF9800000u | 0x01u))
            c++;
    }
#elif defined(__riscv)
    for (size_t i = 0; i + 4 <= n; i += 4) {
        uint32_t w = (uint32_t)code[i] | ((uint32_t)code[i + 1] << 8) |
                     ((uint32_t)code[i + 2] << 16) |
                     ((uint32_t)code[i + 3] << 24);
        /* prefetch.r: ORI x0,rs1,imm HINT, imm[4:0] = 0b00001 */
        if ((w & 0x00007f7fu) == 0x00006013u && ((w >> 20) & 0x1fu) == 1)
            c++;
    }
#else
    for (size_t i = 0; i + 3 <= n; i++) {
        if (code[i] == 0x0F && code[i + 1] == 0x18 &&
            ((code[i + 2] >> 3) & 7) == 1)
            c++; /* PREFETCHT0: 0F 18 /1 */
    }
#endif
    return c;
}

/* prefetch x NT combination (track B4): the planner must enumerate all
 * four (nt, prefetch) combos, and every NT+prefetch twin must produce
 * correct output while honoring the fence discipline. */
static void test_nt_prefetch_combo(const rj_cpu_info *cpu)
{
    char err[256];
    /* Local CPU copy: force Zicbop on RISC-V so the prefetch leg of the
     * cross product exists (the file's CPU is synthetic throughout;
     * prefetch.r is a HINT, safe to emit anywhere). Without Zicbop the
     * planner correctly enumerates no prefetch twins and there would be
     * no (nt, prefetch) combo to test. */
    rj_cpu_info lc = *cpu;
#if defined(__riscv)
    lc.rvv_zicbop = 1;
    /* Zicbop prefetch.r encodes 32..2016 in multiples of 32: the combo
     * distance must be encodable or the backend (correctly) fails the
     * twin build. */
    size_t pf_dist = 1024;
#else
    size_t pf_dist = 4096;
#endif
    rj_topology *t = rj_topo_rs(6, 3);
    if (!t) {
        CHECK(0, "combo: topo");
        return;
    }
    topo_int_t ti;
    if (topo_int_build(&ti, t, err, sizeof err) != 0) {
        CHECK(0, "combo topo: %s", err);
        rj_topo_free(t);
        return;
    }
    int failed[] = {1, 7};
    rj_op op = rj_op_recover(failed, 2);
    plan_t pl;
    if (plan_build(&ti, &op, "rs(6,3)", &pl, err, sizeof err) != 0) {
        CHECK(0, "combo plan: %s", err);
        rj_op_free(&op);
        topo_int_free(&ti);
        rj_topo_free(t);
        return;
    }
    rj_op ope = rj_op_encode();
    plan_t ple;
    if (plan_build(&ti, &ope, "rs(6,3)", &ple, err, sizeof err) != 0) {
        CHECK(0, "combo enc plan: %s", err);
        rj_op_free(&ope);
        rj_op_free(&op);
        plan_free(&pl);
        topo_int_free(&ti);
        rj_topo_free(t);
        return;
    }
    cand_t cands[512];
    int nc = plan_candidates(&pl, &lc, cands, 512, 1, pf_dist);
    CHECK(nc > 0, "combo: no candidates");
    int have[2][2] = {{0, 0}, {0, 0}};
    for (int ci = 0; ci < nc; ci++)
        have[cands[ci].nt != 0][cands[ci].prefetch_dist != 0]++;
    CHECK(have[0][0] > 0, "combo: missing temporal base (have %d/%d/%d/%d)",
          have[0][0], have[1][0], have[0][1], have[1][1]);
    CHECK(have[1][0] > 0, "combo: missing NT twin (have %d/%d/%d/%d)",
          have[0][0], have[1][0], have[0][1], have[1][1]);
    CHECK(have[0][1] > 0, "combo: missing prefetch twin (have %d/%d/%d/%d)",
          have[0][0], have[1][0], have[0][1], have[1][1]);
    CHECK(have[1][1] > 0, "combo: missing NT+prefetch twin (have %d/%d/%d/%d)",
          have[0][0], have[1][0], have[0][1], have[1][1]);

    rj_options opts;
    rj_options_default(&opts);
    opts.benchmark = 0;
    size_t lens[3] = {4096, 65536, 262144};
    int ntested = 0;
    for (int ci = 0; ci < nc && !fails; ci++) {
        if (!cands[ci].nt || !cands[ci].prefetch_dist)
            continue;
        ntested++;
        version_t v;
        memset(&v, 0, sizeof v);
        if (jit_build_version(&pl, &cands[ci], &lc, &opts, &v,
                              err, sizeof err) != 0) {
            CHECK(0, "combo isa=%d U=%d: build: %s", cands[ci].isa,
                  cands[ci].U, err);
            continue;
        }
        if (!pl.noop) {
            CHECK(count_fences(v.mem, v.code_size) == 0,
                  "combo [%s]: NT code must not contain a fence", v.desc);
            CHECK(count_nt_stores(v.mem, v.code_size) >= 1,
                  "combo [%s]: want >=1 NT store", v.desc);
            CHECK(count_prefetches(v.mem, v.code_size) >= 1,
                  "combo [%s]: want >=1 prefetch", v.desc);
        }
        for (int li = 0; li < 3 && !fails; li++)
            check_recover(&v, &ple, &ti, failed, 2, lens[li], "combo");
        jit_free_version(&v);
    }
    CHECK(ntested > 0, "combo: no NT+prefetch twin tested");
    plan_free(&ple);
    rj_op_free(&ope);
    plan_free(&pl);
    rj_op_free(&op);
    topo_int_free(&ti);
    rj_topo_free(t);
}

int main(void)
{
    rj_cpu_info cpu;
    memset(&cpu, 0, sizeof cpu);
#if defined(__aarch64__)
    /* synthetic max CPU: NEON + SVE2 */
    cpu.neon = cpu.sve = cpu.sve2 = 1;
    cpu.sve_vl = 16; /* bytes */
#elif defined(__riscv)
    /* synthetic max CPU: RVV */
    cpu.rvv = 1;
#else
    /* synthetic max CPU: every ISA level incl. GFNI */
    cpu.sse2 = cpu.ssse3 = cpu.avx = cpu.avx2 = 1;
    cpu.avx512f = cpu.avx512bw = cpu.avx512vl = cpu.gfni = 1;
#endif

    rj_topology *t;
    t = rj_topo_raid1();
    test_one(&cpu, t, "raid1", 0, NULL);
    rj_topo_free(t);
    t = rj_topo_raid5(8);
    test_one(&cpu, t, "raid5(8)", 0, NULL);
    rj_topo_free(t);
    t = rj_topo_raid6(8);
    test_one(&cpu, t, "raid6(8)", 0, NULL);
    rj_topo_free(t);
    t = rj_topo_rs(6, 3);
    test_one(&cpu, t, "rs(6,3)", 0, NULL);
    rj_topo_free(t);
    {
        int f1[] = {2};
        t = rj_topo_rs(6, 3);
        test_one(&cpu, t, "rs(6,3) rec", 1, f1);
        rj_topo_free(t);
    }
    {
        int f2[] = {1, 7};
        t = rj_topo_rs(6, 3);
        test_one(&cpu, t, "rs(6,3) rec2", 2, f2);
        rj_topo_free(t);
    }
    /* B4 (SOTA item 18): wider NT recovery failure patterns. */
    {
        int fap[] = {6, 7, 8};
        t = rj_topo_rs(6, 3);
        test_one(&cpu, t, "rs(6,3) rec-allpar", 3, fap);
        rj_topo_free(t);
    }
    {
        int fdp[] = {0, 8};
        t = rj_topo_rs(6, 3);
        test_one(&cpu, t, "rs(6,3) rec-datapar", 2, fdp);
        rj_topo_free(t);
    }
    {
        int f2[] = {3, 9};
        t = rj_topo_rs(12, 3);
        if (t) {
            test_one(&cpu, t, "rs(12,3) rec2", 2, f2);
            rj_topo_free(t);
        }
    }
    {
        int f2[] = {0, 8};
        t = rj_topo_raid6(8);
        test_one(&cpu, t, "raid6(8) rec2", 2, f2);
        rj_topo_free(t);
    }
    /* B4: large-length NT recovery + prefetch x NT combination. */
    test_nt_large_recover(&cpu);
    test_nt_prefetch_combo(&cpu);

    if (!fails)
        printf("nt: all OK\n");
    return fails != 0;
}
