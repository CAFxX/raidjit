/* Partial-stripe parity update (P_DELTA): correctness + API tests.
 *
 * For a topology with ndata data disks and npar parity disks, and a set of
 * changed data disks C = {c0..c_{nc-1}}:
 *   new_p[j] = old_p[j] XOR Σ_i ((old[i] XOR new[i]) * M[j][c_i])
 * where M is the topology's encode matrix.
 *
 * Input layout (non-interleaved, documented in raidjit.h):
 *   [old_0..old_{nc-1}, new_0..new_{nc-1}, oldpar_0..oldpar_{np-1}]
 * out[j] may alias oldpar_j (in-place update).
 *
 * Tests:
 *  1. API validation: empty/NULL/out-of-range/duplicate changed lists;
 *     unsorted input is sorted; constructor owns a private copy.
 *  2. Correctness vs full re-encode after applying changes, for RAID-1/5/6
 *     and a custom RS topology; 1, several, and all data disks changed.
 *  3. Lengths: 0, 1, vector-boundary ±1, multi-group, large odd.
 *  4. Out-of-place and in-place (out aliases oldpar) operation.
 *  5. Canaried buffers: no overrun, inputs unchanged, outputs exact.
 *  6. Benchmark mode (validates every candidate: SSE/AVX2/AVX-512
 *     nibble/GFNI, temporal + NT) at representative lengths.
 */
#include "../include/raidjit.h"

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

static uint64_t xrng_state = 0x9E3779B97F4A7C15ULL;
static uint64_t xrng(void)
{
    uint64_t x = xrng_state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return xrng_state = x;
}
static void rnd_bytes(uint8_t *p, size_t n)
{
    for (size_t i = 0; i < n; i++)
        p[i] = (uint8_t)(xrng() >> 33);
}
static uint8_t *alloc64(size_t n)
{
    /* 64-byte aligned, with 64-byte canary zones on both sides */
    uint8_t *raw = NULL;
    if (posix_memalign((void **)&raw, 64, n + 128))
        abort();
    memset(raw, 0xA5, 64);
    memset(raw + 64 + n, 0x5A, 64);
    return raw + 64;
}
static void free64(uint8_t *p, size_t n)
{
    uint8_t *raw = p - 64;
    int ok = 1;
    for (int i = 0; i < 64; i++)
        if (raw[i] != 0xA5 || raw[64 + n + i] != 0x5A)
            ok = 0;
    CHECK(ok, "canary corruption (buffer overrun/underrun)");
    free(raw);
}
static int check_canary(const uint8_t *p, size_t n)
{
    const uint8_t *raw = p - 64;
    for (int i = 0; i < 64; i++)
        if (raw[i] != 0xA5 || raw[64 + n + i] != 0x5A)
            return 0;
    return 1;
}

/* ------------------------------------------------------------------ */
/* 1. API validation                                                  */
/* ------------------------------------------------------------------ */

static void test_api_validation(void)
{
    char err[256];

    /* NULL changed list */
    {
        rj_op op = rj_op_parity_update(NULL, 3);
        rj_topology *t = rj_topo_raid6(4);
        rj_options o; rj_options_default(&o);
        rj_jit *j = rj_compile(t, &op, &o, err, sizeof(err));
        CHECK(!j, "NULL changed list should fail compile");
        rj_op_free(&op); rj_topo_free(t);
    }
    /* empty changed list */
    {
        int ch[1] = { 0 };
        rj_op op = rj_op_parity_update(ch, 0);
        rj_topology *t = rj_topo_raid6(4);
        rj_options o; rj_options_default(&o);
        rj_jit *j = rj_compile(t, &op, &o, err, sizeof(err));
        CHECK(!j, "empty changed list should fail compile");
        rj_op_free(&op); rj_topo_free(t);
    }
    /* negative nchanged */
    {
        int ch[1] = { 0 };
        rj_op op = rj_op_parity_update(ch, -1);
        rj_topology *t = rj_topo_raid6(4);
        rj_options o; rj_options_default(&o);
        rj_jit *j = rj_compile(t, &op, &o, err, sizeof(err));
        CHECK(!j, "negative nchanged should fail compile");
        rj_op_free(&op); rj_topo_free(t);
    }
    /* out-of-range index */
    {
        int ch[2] = { 0, 4 }; /* raid6(4) has data disks 0..3 */
        rj_op op = rj_op_parity_update(ch, 2);
        rj_topology *t = rj_topo_raid6(4);
        rj_options o; rj_options_default(&o);
        rj_jit *j = rj_compile(t, &op, &o, err, sizeof(err));
        CHECK(!j, "out-of-range changed index should fail compile");
        rj_op_free(&op); rj_topo_free(t);
    }
    /* negative index */
    {
        int ch[1] = { -1 };
        rj_op op = rj_op_parity_update(ch, 1);
        rj_topology *t = rj_topo_raid6(4);
        rj_options o; rj_options_default(&o);
        rj_jit *j = rj_compile(t, &op, &o, err, sizeof(err));
        CHECK(!j, "negative changed index should fail compile");
        rj_op_free(&op); rj_topo_free(t);
    }
    /* nchanged > RJ_MAX_DISKS (256) is clamped, not a hard error;
     * the clamped list still validates (indices 0..255 valid for a
     * wide topology, but raid6(4) rejects out-of-range) */
    {
        int ch[300];
        for (int i = 0; i < 300; i++)
            ch[i] = i; /* 0..299; raid6(4) only has 0..3 */
        rj_op op = rj_op_parity_update(ch, 300);
        CHECK(op.nchanged == 256, "nchanged>256 should clamp to 256, got %d",
              op.nchanged);
        CHECK(!op.oom, "clamped allocation should not set oom");
        rj_topology *t = rj_topo_raid6(4);
        rj_options o; rj_options_default(&o);
        rj_jit *j = rj_compile(t, &op, &o, err, sizeof(err));
        CHECK(!j, "clamped list with out-of-range indices should fail");
        rj_op_free(&op); rj_topo_free(t);
    }
    /* malformed op: both recover and parity_update flags */
    {
        int ch[1] = { 0 };
        int fl[1] = { 4 };
        rj_op op = rj_op_parity_update(ch, 1);
        /* hand-set the recover flag to simulate a malformed op */
        op.is_recover = 1;
        op.failed = malloc(sizeof(int));
        op.failed[0] = fl[0];
        op.nfailed = 1;
        rj_topology *t = rj_topo_raid6(4);
        rj_options o; rj_options_default(&o);
        rj_jit *j = rj_compile(t, &op, &o, err, sizeof(err));
        CHECK(!j, "recover+parity_update should fail compile");
        CHECK(strstr(err, "malformed") != NULL,
              "malformed op error should mention 'malformed', got: %s", err);
        rj_op_free(&op); rj_topo_free(t);
    }
    /* duplicates are deduplicated: {2,2,1} -> {1,2}, compiles fine */
    {
        int ch[3] = { 2, 2, 1 };
        rj_op op = rj_op_parity_update(ch, 3);
        rj_topology *t = rj_topo_raid6(4);
        rj_options o; rj_options_default(&o); o.benchmark = 0;
        rj_jit *j = rj_compile(t, &op, &o, err, sizeof(err));
        CHECK(j, "duplicate changed list should compile (dedup): %s", err);
        if (j) {
            /* Correctness (not the name) verifies dedup: the JIT is run
             * against the reference below. */
            rj_free(j);
        }
        rj_op_free(&op); rj_topo_free(t);
    }
    /* unsorted input is sorted in the plan name */
    {
        int ch[3] = { 3, 0, 2 };
        rj_op op = rj_op_parity_update(ch, 3);
        rj_topology *t = rj_topo_raid5(5);
        rj_options o; rj_options_default(&o); o.benchmark = 0;
        rj_jit *j = rj_compile(t, &op, &o, err, sizeof(err));
        CHECK(j, "unsorted changed list should compile: %s", err);
        if (j) {
            /* Correctness verifies the sort; the plan name is internal. */
            rj_free(j);
        }
        rj_op_free(&op); rj_topo_free(t);
    }
    /* constructor owns a private copy: mutating the caller's array
     * after construction must not affect compilation */
    {
        int ch[2] = { 1, 2 };
        rj_op op = rj_op_parity_update(ch, 2);
        ch[0] = 99; ch[1] = 99; /* corrupt caller's array */
        rj_topology *t = rj_topo_raid6(4);
        rj_options o; rj_options_default(&o); o.benchmark = 0;
        rj_jit *j = rj_compile(t, &op, &o, err, sizeof(err));
        CHECK(j, "constructor must own a private copy: %s", err);
        if (j)
            rj_free(j);
        rj_op_free(&op); rj_topo_free(t);
    }
    /* NULL topology / NULL op */
    {
        rj_options o; rj_options_default(&o);
        int ch[1] = { 0 };
        rj_op op = rj_op_parity_update(ch, 1);
        rj_jit *j = rj_compile(NULL, &op, &o, err, sizeof(err));
        CHECK(!j, "NULL topology should fail");
        rj_op_free(&op);
    }
}

/* ------------------------------------------------------------------ */
/* 2-5. Correctness vs full re-encode                                 */
/* ------------------------------------------------------------------ */

struct topo_case {
    const char *name;
    rj_topology *(*make)(void);
    int ndata, npar;
};

static rj_topology *mk_r1(void) { return rj_topo_raid1(); }
static rj_topology *mk_r5(void) { return rj_topo_raid5(6); }
static rj_topology *mk_r6(void) { return rj_topo_raid6(8); }
static rj_topology *mk_rs43(void) { return rj_topo_rs(4, 3); }

/* run one delta case; inplace!=0 aliases out[j] with oldpar[j] */
static void run_delta_case(rj_topology *t, int ndata, int npar,
                           const int *changed, int nchanged,
                           size_t len, int inplace, int benchmark,
                           const char *tag)
{
    char err[512];
    rj_op op = rj_op_parity_update(changed, nchanged);
    rj_options o; rj_options_default(&o);
    o.benchmark = benchmark;
    o.nt_threshold = benchmark ? 1 : 0; /* also cover NT twins when bench */
    if (benchmark) {
        /* Short bench for QEMU: full 1MiB×11iter benchmarks are ~25s per
         * compile under RVV emulation (vs seconds natively). Reduce to
         * 4KiB×3iter; correctness validation is unaffected (bench.c
         * checksums every candidate regardless of working-set size). */
        o.bench_bytes = 4096;
        o.bench_iters = 3;
    }
    rj_jit *j = rj_compile(t, &op, &o, err, sizeof(err));
    CHECK(j, "%s: compile failed: %s", tag, err);
    if (!j) { rj_op_free(&op); return; }

    /* original data */
    uint8_t **data = malloc(sizeof(*data) * (size_t)ndata);
    uint8_t **orig = malloc(sizeof(*orig) * (size_t)ndata);
    for (int i = 0; i < ndata; i++) {
        data[i] = alloc64(len); orig[i] = alloc64(len);
        rnd_bytes(data[i], len);
        memcpy(orig[i], data[i], len);
    }
    /* full encode of original -> old parities */
    rj_op eop = rj_op_encode();
    rj_options eo; rj_options_default(&eo); eo.benchmark = 0;
    rj_jit *je = rj_compile(t, &eop, &eo, err, sizeof(err));
    CHECK(je, "%s: encode compile failed: %s", tag, err);
    if (!je) { rj_free(j); rj_op_free(&op); return; }
    uint8_t **oldp = malloc(sizeof(*oldp) * (size_t)npar);
    uint8_t **newp = malloc(sizeof(*newp) * (size_t)npar);
    uint8_t **wantp = malloc(sizeof(*wantp) * (size_t)npar);
    for (int k = 0; k < npar; k++) {
        oldp[k] = alloc64(len); newp[k] = alloc64(len); wantp[k] = alloc64(len);
    }
    const void **di = malloc(sizeof(*di) * (size_t)ndata);
    for (int i = 0; i < ndata; i++) di[i] = data[i];
    void **oo = malloc(sizeof(*oo) * (size_t)npar);
    for (int k = 0; k < npar; k++) oo[k] = oldp[k];
    rj_get_fn(je)(di, oo, len);

    /* build old/new changed buffers; apply new data to data[] */
    /* need sorted unique changed list as the plan sees it */
    int *cs = malloc(sizeof(*cs) * (size_t)nchanged);
    memcpy(cs, changed, sizeof(*cs) * (size_t)nchanged);
    /* simple sort + dedup */
    for (int a = 0; a < nchanged; a++)
        for (int b = a + 1; b < nchanged; b++)
            if (cs[a] > cs[b]) { int tmp = cs[a]; cs[a] = cs[b]; cs[b] = tmp; }
    int nc = 0;
    for (int a = 0; a < nchanged; a++)
        if (a == 0 || cs[a] != cs[a - 1])
            cs[nc++] = cs[a];

    uint8_t **oldc = malloc(sizeof(*oldc) * (size_t)nc);
    uint8_t **newc = malloc(sizeof(*newc) * (size_t)nc);
    for (int k = 0; k < nc; k++) {
        oldc[k] = alloc64(len); newc[k] = alloc64(len);
        memcpy(oldc[k], data[cs[k]], len);
        rnd_bytes(newc[k], len);
        memcpy(data[cs[k]], newc[k], len);
    }

    /* delta inputs: [old_0..old_{nc-1}, new_0..new_{nc-1}, oldpar_0..] */
    const void **din = malloc(sizeof(*din) * (size_t)(2 * nc + npar));
    void **dout = malloc(sizeof(*dout) * (size_t)npar);
    for (int k = 0; k < nc; k++) din[k] = oldc[k];
    for (int k = 0; k < nc; k++) din[nc + k] = newc[k];
    for (int k = 0; k < npar; k++) {
        if (inplace) {
            memcpy(newp[k], oldp[k], len);
            din[2 * nc + k] = newp[k];
            dout[k] = newp[k];
        } else {
            din[2 * nc + k] = oldp[k];
            dout[k] = newp[k];
        }
    }
    /* snapshot inputs to verify they are not modified */
    uint8_t **oldc_snap = malloc(sizeof(*oldc_snap) * (size_t)nc);
    uint8_t **newc_snap = malloc(sizeof(*newc_snap) * (size_t)nc);
    uint8_t **oldp_snap = malloc(sizeof(*oldp_snap) * (size_t)npar);
    for (int k = 0; k < nc; k++) {
        oldc_snap[k] = alloc64(len); newc_snap[k] = alloc64(len);
        memcpy(oldc_snap[k], oldc[k], len);
        memcpy(newc_snap[k], newc[k], len);
    }
    for (int k = 0; k < npar; k++) {
        oldp_snap[k] = alloc64(len);
        memcpy(oldp_snap[k], oldp[k], len);
    }

    rj_get_fn(j)(din, dout, len);

    /* reference: full encode of modified data */
    for (int i = 0; i < ndata; i++) di[i] = data[i];
    for (int k = 0; k < npar; k++) oo[k] = wantp[k];
    rj_get_fn(je)(di, oo, len);

    for (int k = 0; k < npar; k++) {
        uint8_t *got = (uint8_t *)dout[k];
        if (memcmp(got, wantp[k], len) != 0) {
            size_t bad = 0;
            for (size_t b = 0; b < len; b++)
                if (got[b] != wantp[k][b]) bad++;
            CHECK(0, "%s: par%d mismatch (%zu/%zu bytes)", tag, k, bad, len);
        }
        CHECK(check_canary(got, len), "%s: par%d canary damaged", tag, k);
    }
    /* inputs must be unchanged (except in-place outputs) */
    for (int k = 0; k < nc; k++) {
        CHECK(memcmp(oldc[k], oldc_snap[k], len) == 0,
              "%s: oldc[%d] modified", tag, k);
        CHECK(memcmp(newc[k], newc_snap[k], len) == 0,
              "%s: newc[%d] modified", tag, k);
        CHECK(check_canary(oldc[k], len), "%s: oldc[%d] canary", tag, k);
        CHECK(check_canary(newc[k], len), "%s: newc[%d] canary", tag, k);
    }
    if (!inplace) {
        for (int k = 0; k < npar; k++) {
            CHECK(memcmp(oldp[k], oldp_snap[k], len) == 0,
                  "%s: oldpar[%d] modified (out-of-place)", tag, k);
            CHECK(check_canary(oldp[k], len), "%s: oldp[%d] canary", tag, k);
        }
    }
    /* unchanged data disks must be untouched */
    for (int i = 0; i < ndata; i++) {
        int is_changed = 0;
        for (int k = 0; k < nc; k++)
            if (cs[k] == i) is_changed = 1;
        if (!is_changed)
            CHECK(memcmp(data[i], orig[i], len) == 0,
                  "%s: unchanged disk %d modified", tag, i);
    }

    for (int i = 0; i < ndata; i++) { free64(data[i], len); free64(orig[i], len); }
    for (int k = 0; k < npar; k++) {
        free64(oldp[k], len); free64(newp[k], len); free64(wantp[k], len);
        free64(oldp_snap[k], len);
    }
    for (int k = 0; k < nc; k++) {
        free64(oldc[k], len); free64(newc[k], len);
        free64(oldc_snap[k], len); free64(newc_snap[k], len);
    }
    free(data); free(orig); free(oldp); free(newp); free(wantp);
    free(di); free(oo); free(oldc); free(newc); free(din); free(dout);
    free(oldc_snap); free(newc_snap); free(oldp_snap); free(cs);
    rj_free(j); rj_free(je);
    rj_op_free(&op); rj_op_free(&eop);
}

static void test_correctness(void)
{
    struct topo_case cases[] = {
        { "raid1", mk_r1, 1, 1 },
        { "raid5(6)", mk_r5, 6, 1 },
        { "raid6(8)", mk_r6, 8, 2 },
        { "rs(4,3)", mk_rs43, 4, 3 },
    };
    /* lengths: 0, 1, boundaries around 64/512/4096, multi-group, large odd */
    size_t lens[] = { 0, 1, 2, 15, 16, 17, 31, 32, 33, 63, 64, 65,
                      127, 128, 129, 255, 256, 257, 511, 512, 513,
                      1000, 1023, 1024, 1025, 4095, 4096, 4097, 9999 };
    int nlens = (int)(sizeof(lens) / sizeof(lens[0]));

    for (unsigned ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ci++) {
        for (int li = 0; li < nlens; li++) {
            size_t len = lens[li];
            int nd = cases[ci].ndata;
            /* change patterns: first disk, last disk, two disks,
             * all disks, unsorted+duplicate */
            int p1[1] = { 0 };
            int p2[1] = { nd - 1 };
            int p3[2] = { 1, nd - 2 };
            int *p4 = malloc(sizeof(int) * (size_t)nd);
            for (int i = 0; i < nd; i++) p4[i] = i;
            int p5[4] = { nd - 1, 0, nd - 1, 1 }; /* unsorted + dup */
            const int *patterns[5] = { p1, p2, p3, p4, p5 };
            int pn[5] = { 1, 1, 2, nd, 4 };
            for (int pi = 0; pi < 5; pi++) {
                /* skip patterns that need more data disks than available */
                if (nd < 2 && pi >= 2)
                    continue;
                if (nd < 3 && pi == 2)
                    continue;
                char tag[128];
                snprintf(tag, sizeof(tag), "%s len=%zu pat=%d %s",
                         cases[ci].name, len, pi, "oop");
                rj_topology *t = cases[ci].make();
                run_delta_case(t, nd, cases[ci].npar, patterns[pi], pn[pi],
                               len, 0, 0, tag);
                snprintf(tag, sizeof(tag), "%s len=%zu pat=%d %s",
                         cases[ci].name, len, pi, "inplace");
                /* in-place only when len>0 (len=0 is a no-op either way) */
                run_delta_case(t, nd, cases[ci].npar, patterns[pi], pn[pi],
                               len, 1, 0, tag);
                rj_topo_free(t);
            }
            free(p4);
        }
    }
}

/* benchmark mode validates every candidate (all ISAs, temporal+NT) */
static void test_benchmark_validation(void)
{
    struct topo_case cases[] = {
        { "raid5(6)", mk_r5, 6, 1 },
        { "raid6(8)", mk_r6, 8, 2 },
        { "rs(4,3)", mk_rs43, 4, 3 },
    };
    size_t lens[] = { 1, 63, 64, 65, 511, 512, 513, 4096 };
    for (unsigned ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ci++) {
        for (unsigned li = 0; li < sizeof(lens) / sizeof(lens[0]); li++) {
            int ch[2] = { 0, cases[ci].ndata - 1 };
            char tag[128];
            snprintf(tag, sizeof(tag), "%s bench len=%zu %s",
                     cases[ci].name, lens[li], "oop");
            rj_topology *t = cases[ci].make();
            run_delta_case(t, cases[ci].ndata, cases[ci].npar, ch, 2,
                           lens[li], 0, 1, tag);
            snprintf(tag, sizeof(tag), "%s bench len=%zu %s",
                     cases[ci].name, lens[li], "inplace");
            run_delta_case(t, cases[ci].ndata, cases[ci].npar, ch, 2,
                           lens[li], 1, 1, tag);
            rj_topo_free(t);
        }
    }
}

/* zero-length parity update must not touch anything */
static void test_len_zero(void)
{
    rj_topology *t = rj_topo_raid6(8);
    int ch[2] = { 2, 5 };
    char err[512];
    rj_op op = rj_op_parity_update(ch, 2);
    rj_options o; rj_options_default(&o); o.benchmark = 0;
    rj_jit *j = rj_compile(t, &op, &o, err, sizeof(err));
    CHECK(j, "len-zero compile: %s", err);
    if (j) {
        uint8_t *in[6], *out[2];
        for (int k = 0; k < 6; k++) in[k] = alloc64(0);
        for (int k = 0; k < 2; k++) out[k] = alloc64(0);
        const void *din[6];
        for (int k = 0; k < 6; k++) din[k] = in[k];
        void *dout[2] = { out[0], out[1] };
        rj_get_fn(j)(din, dout, 0);
        /* nothing should have been written; canaries intact */
        for (int k = 0; k < 2; k++)
            CHECK(check_canary(out[k], 0), "len-zero canary");
        for (int k = 0; k < 6; k++) free64(in[k], 0);
        for (int k = 0; k < 2; k++) free64(out[k], 0);
        rj_free(j);
    }
    rj_op_free(&op); rj_topo_free(t);
}

int main(void)
{
    test_api_validation();
    test_len_zero();
    test_correctness();
    test_benchmark_validation();
    if (fails == 0)
        printf("test_pdelta: all OK\n");
    else
        printf("test_pdelta: %d FAILURES\n", fails);
    return fails != 0;
}
