/* Fused-kernel tests (queue item #8).
 *
 * The fusion planner must reject incompatible sequences loudly at
 * compile time, and fuse compatible ones bit-exactly:
 *
 *   - rejection matrix: null/empty/oversize/unknown-kind/duplicate-CRC/
 *     CRC-alone/CRC-not-earlier/bad-changed-list/loop_align garbage.
 *   - [ENCODE, CRC->0] bit-exact vs the #38 rj_compile_crc path
 *     (same topology/options/lengths; CRCs vs the independent bitwise
 *     oracle in test_crc.c's style, re-implemented here).
 *   - [PARITY_UPDATE, CRC->0]: update math matches a full re-encode of
 *     the new data (test_pdelta's oracle), and CRCs match the bitwise
 *     oracle over the update's in/out buffers.
 *   - [ENCODE, PARITY_UPDATE]: the verify pair -- the update applied
 *     to the encode's buffers equals a fresh full encode; read-only
 *     buffers are shared across the two sub-ops.
 *   - [ENCODE, ENCODE] and single-op sequences work; layout accessors
 *     report the rj_fn contract counts.
 *   - benchmark path: dispatcher composition (tail twins) bit-exact.
 *
 * Non-tautological: every positive case compares two independently
 * compiled pipelines (fused vs unfused); the rejection matrix asserts
 * specific failures that a permissive planner would not produce.
 */
#include "../include/raidjit_fused.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static rj_fuse_options fopts; /* benchmark off: correctness only */

static uint64_t rng_state = 0x123456789abcdefULL;
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

/* Independent bitwise CRC-32/ISO-HDLC (no shared code with src/crc.c). */
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

/* ---------------- rejection matrix ---------------- */

static rj_topology *mk_topo(void)
{
    return rj_topo_raid5(4);
}

static void expect_fail(const char *tag, const rj_topology *t,
                        const rj_fop_t *ops, int nops)
{
    char err[256];
    rj_fused *f = rj_fuse_compile(t, ops, nops, &fopts, err, sizeof(err));
    CHECK(f == NULL, "%s: must fail, got success", tag);
    if (f)
        rj_free_fused(f);
}

static void test_rejections(void)
{
    rj_topology *t = mk_topo();
    CHECK(t != NULL, "topo");
    rj_fop_t ops[5];
    memset(ops, 0, sizeof(ops));
    int ch[2] = { 0, 1 };

    expect_fail("null-topo", NULL, ops, 1);

    ops[0].kind = RJ_FOP_ENCODE;
    expect_fail("null-ops", t, NULL, 1);
    expect_fail("empty", t, ops, 0);
    expect_fail("oversize", t, ops, 5);

    memset(ops, 0, sizeof(ops));
    ops[0].kind = (rj_fop_kind_t)99;
    expect_fail("unknown-kind", t, ops, 1);

    /* CRC alone: no earlier data op. */
    memset(ops, 0, sizeof(ops));
    ops[0].kind = RJ_FOP_CRC;
    ops[0].target = 0;
    expect_fail("crc-alone", t, ops, 1);

    /* CRC targeting itself / a later op. */
    memset(ops, 0, sizeof(ops));
    ops[0].kind = RJ_FOP_ENCODE;
    ops[1].kind = RJ_FOP_CRC;
    ops[1].target = 1;
    expect_fail("crc-target-self", t, ops, 2);
    ops[1].target = 5;
    expect_fail("crc-target-oob", t, ops, 2);

    /* Duplicate CRC ops. */
    memset(ops, 0, sizeof(ops));
    ops[0].kind = RJ_FOP_ENCODE;
    ops[1].kind = RJ_FOP_CRC;
    ops[1].target = 0;
    ops[2].kind = RJ_FOP_CRC;
    ops[2].target = 0;
    expect_fail("dup-crc", t, ops, 3);

    /* Bad changed lists. */
    memset(ops, 0, sizeof(ops));
    ops[0].kind = RJ_FOP_PARITY_UPDATE;
    ops[0].changed = NULL;
    ops[0].nchanged = 1;
    expect_fail("pupd-null-changed", t, ops, 1);
    ops[0].changed = ch;
    ops[0].nchanged = 0;
    expect_fail("pupd-empty-changed", t, ops, 1);
    {
        int bad[1] = { 7 };
        ops[0].changed = bad;
        ops[0].nchanged = 1;
        expect_fail("pupd-changed-oob", t, ops, 1);
    }

    /* Bad loop_align. */
    {
        rj_fuse_options bad = fopts;
        bad.loop_align = 3;
        char err[256];
        memset(ops, 0, sizeof(ops));
        ops[0].kind = RJ_FOP_ENCODE;
        rj_fused *f = rj_fuse_compile(t, ops, 1, &bad, err, sizeof(err));
        CHECK(f == NULL, "loop_align=3: must fail");
        if (f)
            rj_free_fused(f);
    }

    /* NULL-safe accessors. */
    CHECK(rj_get_fn_fused(NULL) == NULL, "get_fn(NULL)");
    CHECK(rj_fused_nops(NULL) == 0, "nops(NULL)");
    CHECK(rj_fused_op_nin(NULL, 0) == -1, "nin(NULL)");
    CHECK(rj_fused_tile_bytes(NULL) == 0, "tile(NULL)");
    rj_free_fused(NULL); /* must not crash */

    rj_topo_free(t);
    printf("ok rejections\n");
}

/* ---------------- [ENCODE, CRC] == #38 ---------------- */

static void test_encode_crc_equals_38(void)
{
    /* Same topology, same options (benchmark=0), same lengths: the
     * fused [ENCODE,CRC->0] must be bit-exact vs rj_compile_crc, which
     * is now the same planner under the hood -- this pins the
     * refactor to the pre-#8 validated behavior. */
    static const size_t lens[] = { 1, 63, 64, 65, 1000, 4096,
                                   524287, 524288, 524289, 1048699 };
    struct {
        rj_topology *t;
        const char *name;
        int nd, np;
    } ts[] = {
        { rj_topo_raid5(4), "raid5(4)", 4, 1 },
        { rj_topo_raid6_rs(6), "raid6_rs(6)", 6, 2 },
        { rj_topo_rs(6, 3), "rs(6,3)", 6, 3 },
    };
    for (size_t ti = 0; ti < 3; ti++) {
        rj_topology *t = ts[ti].t;
        int nd = ts[ti].nd, np = ts[ti].np;
        char err[256];

        rj_fop_t fops[2];
        memset(fops, 0, sizeof(fops));
        fops[0].kind = RJ_FOP_ENCODE;
        fops[1].kind = RJ_FOP_CRC;
        fops[1].target = 0;
        rj_fused *f = rj_fuse_compile(t, fops, 2, &fopts, err,
                                      sizeof(err));
        CHECK(f != NULL, "%s: fuse failed: %s", ts[ti].name, err);
        if (!f) {
            rj_topo_free(t);
            continue;
        }
        rj_fn_fused ffn = rj_get_fn_fused(f);
        CHECK(ffn != NULL, "%s: fused fn NULL", ts[ti].name);

        rj_crc_options co;
        rj_crc_options_default(&co);
        co.benchmark = 0;
        rj_jit_crc *jc = rj_compile_crc(t, &co, err, sizeof(err));
        CHECK(jc != NULL, "%s: crc compile failed: %s", ts[ti].name,
              err);
        rj_fn_crc cfn = jc ? rj_get_fn_crc(jc) : NULL;

        /* Layout accessors pin the rj_fn contract counts. */
        CHECK(rj_fused_nops(f) == 2, "%s: nops", ts[ti].name);
        CHECK(rj_fused_op_nin(f, 0) == nd, "%s: nin", ts[ti].name);
        CHECK(rj_fused_op_nout(f, 0) == np, "%s: nout", ts[ti].name);
        CHECK(rj_fused_op_nin(f, 1) == 0, "%s: crc nin", ts[ti].name);
        CHECK(rj_fused_op_nout(f, 2) == -1, "%s: oob nout", ts[ti].name);
        CHECK(rj_fused_tile_bytes(f) == rj_crc_tile_bytes(jc),
              "%s: tile mismatch", ts[ti].name);

        for (size_t li = 0;
             li < sizeof(lens) / sizeof(lens[0]); li++) {
            size_t len = lens[li];
            const void **in = malloc(sizeof(*in) * (size_t)nd);
            void **oF = malloc(sizeof(*oF) * (size_t)np);
            void **oC = malloc(sizeof(*oC) * (size_t)np);
            for (int i = 0; i < nd; i++) {
                uint8_t *b = abuf(len);
                randfill(b, len);
                in[i] = b;
            }
            for (int p = 0; p < np; p++) {
                oF[p] = abuf(len);
                oC[p] = abuf(len);
            }
            uint32_t crcF[256], crcC[256], crcO[256];
            if (ffn && cfn) {
                ffn(in, oF, len, crcF);
                cfn(in, oC, len, crcC, crcO);
                for (int p = 0; p < np; p++)
                    CHECK(memcmp(oF[p], oC[p], len) == 0,
                          "%s len=%zu: fused vs #38 parity %d",
                          ts[ti].name, len, p);
                for (int i = 0; i < nd; i++)
                    CHECK(crcF[i] == crcC[i],
                          "%s len=%zu: fused vs #38 crc_in[%d]",
                          ts[ti].name, len, i);
                for (int p = 0; p < np; p++)
                    CHECK(crcF[nd + p] == crcO[p],
                          "%s len=%zu: fused vs #38 crc_out[%d]",
                          ts[ti].name, len, p);
                /* Spot-check against the bitwise oracle. */
                if (len <= 4096) {
                    for (int i = 0; i < nd; i++)
                        CHECK(crcF[i] == crc32_bitwise(in[i], len),
                              "%s len=%zu: oracle crc_in[%d]",
                              ts[ti].name, len, i);
                }
            }
            for (int i = 0; i < nd; i++)
                free((void *)in[i]);
            for (int p = 0; p < np; p++) {
                free(oF[p]);
                free(oC[p]);
            }
            free(in);
            free(oF);
            free(oC);
        }
        rj_free_fused(f);
        rj_free_crc(jc);
        rj_topo_free(t);
        printf("ok encode_crc_equals_38 %s\n", ts[ti].name);
    }
}

/* ---------------- [PARITY_UPDATE, CRC] ---------------- */

static rj_jit *plain_encode(const rj_topology *t)
{
    char err[256];
    rj_op op = rj_op_encode();
    rj_options o;
    rj_options_default(&o);
    o.benchmark = 0;
    rj_jit *j = rj_compile(t, &op, &o, err, sizeof(err));
    rj_op_free(&op);
    if (!j)
        fprintf(stderr, "plain encode compile: %s\n", err);
    return j;
}

static void test_pupdate_crc(void)
{
    /* [PARITY_UPDATE{0,2}, CRC->0] on raid6_rs(6): the update math
     * must match a full re-encode of the new data (the test_pdelta
     * oracle), and the CRCs must match the bitwise oracle over the
     * update's own in/out buffers. */
    rj_topology *t = rj_topo_raid6_rs(6);
    CHECK(t != NULL, "topo");
    if (!t)
        return;
    int np = 2;
    int ch[2] = { 0, 2 };
    int nc = 2;
    char err[256];

    rj_fop_t fops[2];
    memset(fops, 0, sizeof(fops));
    fops[0].kind = RJ_FOP_PARITY_UPDATE;
    fops[0].changed = ch;
    fops[0].nchanged = nc;
    fops[1].kind = RJ_FOP_CRC;
    fops[1].target = 0;
    rj_fused *f = rj_fuse_compile(t, fops, 2, &fopts, err, sizeof(err));
    CHECK(f != NULL, "pupd fuse failed: %s", err);
    if (!f) {
        rj_topo_free(t);
        return;
    }
    rj_fn_fused ffn = rj_get_fn_fused(f);
    int ni = 2 * nc + np; /* rj_fn parity-update contract */
    CHECK(rj_fused_op_nin(f, 0) == ni, "pupd nin=%d want %d",
          rj_fused_op_nin(f, 0), ni);
    CHECK(rj_fused_op_nout(f, 0) == np, "pupd nout");

    rj_jit *e = plain_encode(t);
    rj_fn efn = e ? rj_get_fn(e) : NULL;

    static const size_t lens[] = { 64, 1000, 70000 };
    for (size_t li = 0; li < 3; li++) {
        size_t len = lens[li];
        /* Update buffers: old[2], new[2], oldpar[2]; parities updated
         * in place (out aliases oldpar). */
        uint8_t *old[2], *newd[2], *par[2];
        const void *in[6];
        void *out[2];
        for (int k = 0; k < 2; k++) {
            old[k] = abuf(len);
            newd[k] = abuf(len);
            par[k] = abuf(len);
            randfill(old[k], len);
            randfill(newd[k], len);
            /* rj_fn contract: [old_0..old_{nc-1}, new_0..new_{nc-1},
             * oldpar_0..] -- all olds first, then all news. */
            in[k] = old[k];
            in[2 + k] = newd[k];
        }
        /* oldpar = encode(all-old); want = encode(with new applied). */
        uint8_t *data_old[6], *data_new[6];
        const void *din_old[6], *din_new[6];
        for (int i = 0; i < 6; i++) {
            data_old[i] = abuf(len);
            data_new[i] = abuf(len);
            din_old[i] = data_old[i];
            din_new[i] = data_new[i];
        }
        for (int k = 0; k < 2; k++) {
            int d = ch[k];
            memcpy(data_old[d], old[k], len);
            memcpy(data_new[d], newd[k], len);
        }
        for (int i = 0; i < 6; i++) {
            int ischg = (i == ch[0] || i == ch[1]);
            if (!ischg) {
                randfill(data_old[i], len);
                memcpy(data_new[i], data_old[i], len);
            }
        }
        void *oldp[2], *wantp[2];
        for (int p = 0; p < 2; p++) {
            oldp[p] = abuf(len);
            wantp[p] = abuf(len);
        }
        if (efn) {
            efn(din_old, oldp, len);
            efn(din_new, wantp, len);
        }
        for (int p = 0; p < 2; p++) {
            memcpy(par[p], oldp[p], len);
            in[4 + p] = par[p];
            out[p] = par[p]; /* in-place update */
        }
        uint32_t crc[8]; /* 6 in + 2 out */
        if (ffn && efn) {
            ffn(in, out, len, crc);
            for (int p = 0; p < 2; p++)
                CHECK(memcmp(par[p], wantp[p], len) == 0,
                      "len=%zu: fused update != re-encode (p=%d)", len,
                      p);
            for (int k = 0; k < 6; k++)
                CHECK(crc[k] == crc32_bitwise(in[k], len),
                      "len=%zu: oracle crc in[%d]", len, k);
            for (int p = 0; p < 2; p++)
                CHECK(crc[6 + p] == crc32_bitwise(par[p], len),
                      "len=%zu: oracle crc out[%d]", len, p);
        }
        for (int k = 0; k < 2; k++) {
            free(old[k]);
            free(newd[k]);
            free(par[k]);
            free(oldp[k]);
            free(wantp[k]);
        }
        for (int i = 0; i < 6; i++) {
            free(data_old[i]);
            free(data_new[i]);
        }
    }
    rj_free(e);
    rj_free_fused(f);
    rj_topo_free(t);
    printf("ok pupdate_crc\n");
}

/* ---------------- [ENCODE, PARITY_UPDATE] verify pair ---------------- */

static void test_encode_pupdate_pair(void)
{
    /* [ENCODE, PARITY_UPDATE{1}] on raid5(4): the update re-applies
     * disk 1's delta onto the encode's outputs. The two sub-ops share
     * the read-only new[] buffer with the encode's in[1]. */
    rj_topology *t = rj_topo_raid5(4);
    CHECK(t != NULL, "topo");
    if (!t)
        return;
    int ch[1] = { 1 };
    char err[256];

    rj_fop_t fops[2];
    memset(fops, 0, sizeof(fops));
    fops[0].kind = RJ_FOP_ENCODE;
    fops[1].kind = RJ_FOP_PARITY_UPDATE;
    fops[1].changed = ch;
    fops[1].nchanged = 1;
    rj_fused *f = rj_fuse_compile(t, fops, 2, &fopts, err, sizeof(err));
    CHECK(f != NULL, "pair fuse failed: %s", err);
    if (!f) {
        rj_topo_free(t);
        return;
    }
    rj_fn_fused ffn = rj_get_fn_fused(f);
    /* Flat layout: op0 in[0..4) out[0..1); op1 in[4..7) out[1..2). */
    CHECK(rj_fused_op_nin(f, 1) == 2 * 1 + 1, "pair pupd nin");
    CHECK(rj_fused_op_nout(f, 1) == 1, "pair pupd nout");

    rj_jit *e = plain_encode(t);
    rj_fn efn = e ? rj_get_fn(e) : NULL;

    static const size_t lens[] = { 128, 5000 };
    for (size_t li = 0; li < 2; li++) {
        size_t len = lens[li];
        /* Flat buffers: 4 + 3 inputs, 1 + 1 outputs. */
        uint8_t *din[4];
        const void *in[7];
        uint8_t *old1 = abuf(len);
        for (int i = 0; i < 4; i++) {
            din[i] = abuf(len);
            randfill(din[i], len);
            in[i] = din[i];
        }
        memcpy(old1, din[1], len);
        /* op1's inputs: old[1], new[1] (= din[1], shared read-only),
         * oldpar (encode's out[0], updated in place). */
        uint8_t *p0 = abuf(len), *p1 = abuf(len);
        void *out[2];
        out[0] = p0;
        out[1] = p1;
        in[4] = old1;
        in[5] = din[1]; /* shared with op0's in[1] */
        in[6] = p0;     /* oldpar: encode's out[0] */
        /* out[1] = p1: out-of-place update */

        /* Oracle: full encode with new data, then with old data for
         * the pre-update parity. */
        uint8_t *want_new = abuf(len);
        const void *en_in[4];
        void *en_out[1];
        for (int i = 0; i < 4; i++)
            en_in[i] = din[i];
        en_out[0] = want_new;
        if (efn)
            efn(en_in, en_out, len);
        if (ffn) {
            ffn(in, out, len, NULL);
            CHECK(memcmp(p0, want_new, len) == 0,
                  "len=%zu: encode parity mismatch", len);
            /* p1 must equal a fresh encode of the new data too: the
             * update applied disk 1's (old->new) delta, and old==new
             * here, so p1 == p0. */
            CHECK(memcmp(p1, want_new, len) == 0,
                  "len=%zu: update parity mismatch", len);
        }
        for (int i = 0; i < 4; i++)
            free(din[i]);
        free(old1);
        free(p0);
        free(p1);
        free(want_new);
    }
    rj_free(e);
    rj_free_fused(f);
    rj_topo_free(t);
    printf("ok encode_pupdate_pair\n");
}

/* ---------------- [ENCODE, ENCODE] and single-op ---------------- */

static void test_two_encodes_and_single(void)
{
    rj_topology *t = rj_topo_raid5(4);
    CHECK(t != NULL, "topo");
    if (!t)
        return;
    char err[256];

    /* Two encodes of different stripes in one call. */
    rj_fop_t fops[2];
    memset(fops, 0, sizeof(fops));
    fops[0].kind = RJ_FOP_ENCODE;
    fops[1].kind = RJ_FOP_ENCODE;
    rj_fused *f = rj_fuse_compile(t, fops, 2, &fopts, err, sizeof(err));
    CHECK(f != NULL, "2xencode fuse: %s", err);
    rj_jit *e = plain_encode(t);
    rj_fn efn = e ? rj_get_fn(e) : NULL;
    rj_fn_fused ffn = f ? rj_get_fn_fused(f) : NULL;
    size_t len = 3000;
    const void *in[8];
    void *outF[2], *outP[2];
    for (int i = 0; i < 8; i++) {
        uint8_t *b = abuf(len);
        randfill(b, len);
        in[i] = b;
    }
    for (int p = 0; p < 2; p++) {
        outF[p] = abuf(len);
        outP[p] = abuf(len);
    }
    if (ffn && efn) {
        ffn(in, outF, len, NULL);
        efn(&in[0], &outP[0], len);
        efn(&in[4], &outP[1], len);
        CHECK(memcmp(outF[0], outP[0], len) == 0, "2xencode p0");
        CHECK(memcmp(outF[1], outP[1], len) == 0, "2xencode p1");
    }
    for (int i = 0; i < 8; i++)
        free((void *)in[i]);
    for (int p = 0; p < 2; p++) {
        free(outF[p]);
        free(outP[p]);
    }
    rj_free_fused(f);

    /* Single-op fusion works and matches the plain path. */
    memset(fops, 0, sizeof(fops));
    fops[0].kind = RJ_FOP_ENCODE;
    f = rj_fuse_compile(t, fops, 1, &fopts, err, sizeof(err));
    CHECK(f != NULL, "single fuse: %s", err);
    ffn = f ? rj_get_fn_fused(f) : NULL;
    CHECK(rj_fused_nops(f) == 1, "single nops");
    const void *in1[4];
    void *oF[1], *oP[1];
    for (int i = 0; i < 4; i++) {
        uint8_t *b = abuf(len);
        randfill(b, len);
        in1[i] = b;
    }
    oF[0] = abuf(len);
    oP[0] = abuf(len);
    if (ffn && efn) {
        ffn(in1, oF, len, NULL);
        efn(in1, oP, len);
        CHECK(memcmp(oF[0], oP[0], len) == 0, "single parity");
    }
    for (int i = 0; i < 4; i++)
        free((void *)in1[i]);
    free(oF[0]);
    free(oP[0]);
    rj_free_fused(f);
    rj_free(e);
    rj_topo_free(t);
    printf("ok two_encodes_and_single\n");
}

/* ---------------- benchmark path (dispatcher composition) ---------------- */

static void test_benchmark_path(void)
{
    /* One benchmark-path compile: tail twins may install and the
     * per-tile calls route through the length dispatcher. */
    rj_topology *t = rj_topo_raid6_rs(6);
    CHECK(t != NULL, "topo");
    if (!t)
        return;
    char err[256];
    rj_fuse_options bfo;
    rj_fuse_options_default(&bfo); /* benchmark on */
    rj_fop_t fops[2];
    memset(fops, 0, sizeof(fops));
    fops[0].kind = RJ_FOP_ENCODE;
    fops[1].kind = RJ_FOP_CRC;
    fops[1].target = 0;
    rj_fused *f = rj_fuse_compile(t, fops, 2, &bfo, err, sizeof(err));
    CHECK(f != NULL, "benchmark-path fuse: %s", err);
    if (!f) {
        rj_topo_free(t);
        return;
    }
    rj_fn_fused ffn = rj_get_fn_fused(f);
    rj_op op = rj_op_encode();
    rj_options o;
    rj_options_default(&o); /* benchmark on */
    rj_jit *e = rj_compile(t, &op, &o, err, sizeof(err));
    rj_op_free(&op);
    CHECK(e != NULL, "benchmark-path plain: %s", err);
    rj_fn fn = e ? rj_get_fn(e) : NULL;

    static const size_t lens[] = { 1000, 100000 };
    for (size_t li = 0; li < 2; li++) {
        size_t len = lens[li];
        const void *in[6];
        void *outF[2], *outP[2];
        for (int i = 0; i < 6; i++) {
            uint8_t *b = abuf(len);
            randfill(b, len);
            in[i] = b;
        }
        for (int p = 0; p < 2; p++) {
            outF[p] = abuf(len);
            outP[p] = abuf(len);
        }
        uint32_t crc[8];
        if (ffn && fn) {
            ffn(in, outF, len, crc);
            fn(in, outP, len);
            for (int p = 0; p < 2; p++)
                CHECK(memcmp(outF[p], outP[p], len) == 0,
                      "bench len=%zu parity %d", len, p);
            for (int i = 0; i < 6; i++)
                CHECK(crc[i] == crc32_bitwise(in[i], len),
                      "bench len=%zu crc_in[%d]", len, i);
        }
        for (int i = 0; i < 6; i++)
            free((void *)in[i]);
        for (int p = 0; p < 2; p++) {
            free(outF[p]);
            free(outP[p]);
        }
    }
    rj_free(e);
    rj_free_fused(f);
    rj_topo_free(t);
    printf("ok benchmark_path\n");
}

int main(void)
{
    rj_fuse_options_default(&fopts);
    fopts.benchmark = 0;
    test_rejections();
    test_encode_crc_equals_38();
    test_pupdate_crc();
    test_encode_pupdate_pair();
    test_two_encodes_and_single();
    test_benchmark_path();
    if (failures) {
        printf("test_fused: %d FAILURES\n", failures);
        return 1;
    }
    printf("test_fused: all OK\n");
    return 0;
}
