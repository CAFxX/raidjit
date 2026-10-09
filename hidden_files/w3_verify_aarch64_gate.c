/* W3 worker verification (AArch64): CRC Phase 2 gate CLOSED, fallback correct.
 *
 * Per crc_fold_phase2_SPEC.md §2, on AArch64 the interleave gate is CLOSED
 * for Phase 2 (no Apple-silicon µarch detection exists, so a conservative
 * allowlist is unimplementable). This verifies:
 *   1. fused_crc_interleave_ok() returns 0 on AArch64 (compile-time
 *      #elif defined(__aarch64__) branch -- holds under ANY cpu, including
 *      QEMU -cpu max).
 *   2. plan_candidates returns 0 candidates when pl.want_crc=1 (the gate
 *      check fires; no crc_fold=1 candidate is ever enumerated).
 *   3. The scalar fallback path -- Phase-1 rj_crc32_fast over the tile
 *      buffers, exercised end-to-end via rj_fuse_compile([ENCODE, CRC]) --
 *      produces CRCs bit-exact vs rj_crc32_update and an independent
 *      bitwise oracle.
 *
 * Kept in hidden_files as worker verification evidence, not part of the
 * shipped test suite.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "raidjit.h"
#include "raidjit_fused.h"
#include "plan.h"
#include "fused.h"
#include "cpu.h"

static int failures = 0;
#define CHECK(cond, ...) do {                                          \
    if (!(cond)) { printf("FAIL %s:%d: ", __func__, __LINE__);        \
        printf(__VA_ARGS__); printf("\n"); failures++; }              \
} while (0)

static uint64_t rng_state = 0xC0FFEE123456789ULL;
static uint64_t xrng(void)
{
    uint64_t x = rng_state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return rng_state = x;
}
static void *abuf(size_t n)
{
    void *p = NULL;
    if (posix_memalign(&p, 64, n ? n : 64)) { fprintf(stderr, "oom\n"); exit(1); }
    return p;
}
static void randfill(void *p, size_t n)
{
    uint8_t *b = p;
    for (size_t i = 0; i < n; i++)
        b[i] = (uint8_t)xrng();
}

/* Independent bitwise CRC-32/ISO-HDLC (shares no code with src/crc*). */
static uint32_t crc32_bitwise(const uint8_t *p, size_t n)
{
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; i++) {
        crc ^= p[i];
        for (int k = 0; k < 8; k++)
            crc = (crc & 1) ? (crc >> 1) ^ 0xEDB88320u : crc >> 1;
    }
    return crc ^ 0xFFFFFFFFu;
}

#if defined(__aarch64__)

static void test_gate_closed(void)
{
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    int ok = fused_crc_interleave_ok(&cpu);
    printf("gate: fused_crc_interleave_ok() = %d\n", ok);
    CHECK(ok == 0, "gate must be 0 on AArch64, got %d", ok);
    CHECK(fused_crc_interleave_ok(NULL) == 0, "gate(NULL) must be 0");
}

static void test_planner_no_fold_candidates(void)
{
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    char err[256];
    rj_topology *topo = rj_topo_rs(6, 3);
    CHECK(topo != NULL, "topo_rs failed");
    topo_int_t ti;
    if (topo_int_build(&ti, topo, err, sizeof err)) {
        printf("FAIL: topo_int_build: %s\n", err); failures++;
        rj_topo_free(topo); return;
    }
    rj_op op = rj_op_encode();
    plan_t pl;
    if (plan_build(&ti, &op, "enc", &pl, err, sizeof err)) {
        printf("FAIL: plan_build: %s\n", err); failures++;
        goto out;
    }

    cand_t cands[512];
    pl.want_crc = 1;
    int nc = plan_candidates(&pl, &cpu, cands, 512, 0, 0);
    printf("planner: want_crc=1 -> %d candidates\n", nc);
    CHECK(nc == 0, "want_crc=1 with closed gate must yield 0 candidates, got %d", nc);

    /* want_crc=0: normal enumeration, and no crc_fold anywhere. */
    pl.want_crc = 0;
    nc = plan_candidates(&pl, &cpu, cands, 512, 0, 0);
    printf("planner: want_crc=0 -> %d candidates\n", nc);
    CHECK(nc > 0, "want_crc=0 must enumerate candidates");
    for (int i = 0; i < nc; i++)
        CHECK(cands[i].crc_fold == 0,
              "candidate %d has crc_fold=1 with closed gate", i);

    plan_free(&pl);
out:
    topo_int_free(&ti);
    rj_op_free(&op);
    rj_topo_free(topo);
}

/* End-to-end: fused [ENCODE, CRC->0] (the gate-closed fallback driver
 * path: Phase-1 rj_crc32_fast over tile buffers) vs bitwise oracle and
 * rj_crc32_update. */
static void test_fused_scalar_fallback(void)
{
    static const size_t lens[] = { 1, 7, 63, 64, 65, 1000, 4096, 65537 };
    struct { rj_topology *t; const char *name; int nd, np; } ts[] = {
        { rj_topo_raid5(4), "raid5(4)", 4, 1 },
        { rj_topo_raid6_rs(6), "raid6_rs(6)", 6, 2 },
        { rj_topo_rs(6, 3), "rs(6,3)", 6, 3 },
    };
    rj_fuse_options fopts;
    rj_fuse_options_default(&fopts);

    for (size_t ti = 0; ti < 3; ti++) {
        rj_topology *t = ts[ti].t;
        int nd = ts[ti].nd, np = ts[ti].np;
        char err[256];
        rj_fop_t fops[2];
        memset(fops, 0, sizeof fops);
        fops[0].kind = RJ_FOP_ENCODE;
        fops[1].kind = RJ_FOP_CRC;
        fops[1].target = 0;
        rj_fused *f = rj_fuse_compile(t, fops, 2, &fopts, err, sizeof err);
        CHECK(f != NULL, "%s: fuse failed: %s", ts[ti].name, err);
        if (!f) { rj_topo_free(t); continue; }
        rj_fn_fused ffn = rj_get_fn_fused(f);
        CHECK(ffn != NULL, "%s: fused fn NULL", ts[ti].name);

        for (size_t li = 0; li < sizeof lens / sizeof lens[0]; li++) {
            size_t len = lens[li];
            const void **in = malloc(sizeof *in * (size_t)nd);
            void **out = malloc(sizeof *out * (size_t)np);
            for (int i = 0; i < nd; i++) {
                uint8_t *b = abuf(len); randfill(b, len); in[i] = b;
            }
            for (int p = 0; p < np; p++)
                out[p] = abuf(len);
            uint32_t crc[16];
            if (ffn) {
                ffn(in, out, len, crc);
                for (int i = 0; i < nd; i++) {
                    uint32_t o = crc32_bitwise(in[i], len);
                    uint32_t fu = 0xFFFFFFFFu;
                    fu = rj_crc32_fast(fu, in[i], len);
                    fu ^= 0xFFFFFFFFu;
                    uint32_t uu = rj_crc32_update(0xFFFFFFFFu, in[i], len)
                                  ^ 0xFFFFFFFFu;
                    CHECK(crc[i] == o,
                          "%s len=%zu: fused crc_in[%d]=%08x != oracle %08x",
                          ts[ti].name, len, i, crc[i], o);
                    CHECK(crc[i] == fu,
                          "%s len=%zu: fused crc_in[%d] != rj_crc32_fast",
                          ts[ti].name, len, i);
                    CHECK(crc[i] == uu,
                          "%s len=%zu: fused crc_in[%d] != rj_crc32_update",
                          ts[ti].name, len, i);
                }
                for (int p = 0; p < np; p++) {
                    uint32_t o = crc32_bitwise(out[p], len);
                    CHECK(crc[nd + p] == o,
                          "%s len=%zu: fused crc_out[%d]=%08x != oracle %08x",
                          ts[ti].name, len, p, crc[nd + p], o);
                }
            }
            for (int i = 0; i < nd; i++) free((void *)in[i]);
            for (int p = 0; p < np; p++) free(out[p]);
            free(in); free(out);
        }
        rj_free_fused(f);
        rj_topo_free(t);
    }
}

#endif /* __aarch64__ */

int main(void)
{
#if !defined(__aarch64__)
    printf("SKIP: AArch64-gate verification is aarch64-only\n");
    return 0;
#else
    test_gate_closed();
    test_planner_no_fold_candidates();
    test_fused_scalar_fallback();
    if (failures)
        printf("RESULT: FAIL (%d failures)\n", failures);
    else
        printf("RESULT: PASS\n");
    return failures != 0;
#endif
}
