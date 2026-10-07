/* Integrated-CRC encode pipeline (queue item #38).
 *
 * rj_compile_crc compiles the encode op through the public rj_compile
 * planner (full candidate enumeration / pruning / benchmarking -- the
 * CRC variant is planner-selectable in exactly the same sense as the
 * plain encode), then wraps the winning rj_fn in a cache-blocked fused
 * driver: per tile it runs the encode kernel once and accumulates
 * CRC-32 over the tile's input and parity buffers while they are hot
 * in cache. Every input byte is read from DRAM once.
 *
 * The entry point is one of a static trampoline pool bound to the
 * handle (the api_batched.c pattern): rj_fn_crc carries no context
 * pointer. Lifetime, thread-safety, and slot-reuse rules match
 * api_batched.h.
 */
#include "../include/raidjit_fused.h"
#include "crc.h"
#include "striping.h"

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ndata+npar <= RJ_MAX_DISKS (256) for every topology (src/topo.c), so
 * fixed-size per-call staging is bounded and overflow-free. */
#define CRC_MAX_BUFS 256

struct rj_jit_crc {
    rj_jit *enc;     /* underlying encode unit (owns the kernel) */
    rj_fn enc_fn;    /* dispatched entry point */
    int ndata, npar;
    size_t tile;     /* plan-time tile size (rj_tile_select) */
    char desc[192];  /* underlying winner's desc, for info */
    int slot;        /* trampoline slot, -1 when not installed */
};

void rj_crc_options_default(rj_crc_options *o)
{
    memset(o, 0, sizeof(*o));
    o->benchmark = 1;
    o->loop_align = 32;
}

static void crc_options_to_rj(const rj_crc_options *co, rj_options *o)
{
    rj_options_default(o);
    if (!co)
        return;
    o->benchmark = co->benchmark;
    o->bench_bytes = co->bench_bytes;
    o->bench_iters = co->bench_iters;
    o->verbose = co->verbose;
    o->loop_align = co->loop_align;
    o->emit_endbr = co->emit_endbr;
    o->emit_shadow_stack = co->emit_shadow_stack;
    o->nt_threshold = co->nt_threshold;
    o->prefetch_distance = co->prefetch_distance;
}

/* One fused call: tile loop over (encode kernel + CRC accumulation).
 * crc_in/crc_out may be NULL (that side is skipped). */
static void crc_driver_call(const struct rj_jit_crc *c, const void *in[],
                            void *out[], size_t len, uint32_t *crc_in,
                            uint32_t *crc_out)
{
    assert(in != NULL);
    assert(out != NULL || c->npar == 0);
    assert(crc_out == NULL || c->npar > 0);
    assert(len >= 1);
    int nd = c->ndata, np = c->npar;
    rj_fn fn = c->enc_fn;
    size_t tile = c->tile;

    uint32_t acc_in[CRC_MAX_BUFS], acc_out[CRC_MAX_BUFS];
    const void *tin[CRC_MAX_BUFS];
    void *tout[CRC_MAX_BUFS];
    for (int i = 0; i < nd; i++)
        rj_crc32_init(&acc_in[i]);
    for (int j = 0; j < np; j++)
        rj_crc32_init(&acc_out[j]);

    for (size_t off = 0; off < len; off += tile) {
        size_t tl = len - off < tile ? len - off : tile;
        for (int i = 0; i < nd; i++) {
            tin[i] = (const uint8_t *)in[i] + off;
            assert(((uintptr_t)tin[i] & 63) == 0); /* tile is a power of
                                                    * two >= 4096: tile
                                                    * offsets preserve
                                                    * the caller's
                                                    * 64-byte alignment */
        }
        for (int j = 0; j < np; j++)
            tout[j] = (uint8_t *)out[j] + off;
        fn(tin, np ? tout : NULL, tl);
        /* The tile's inputs and freshly written parities are hot in
         * cache here: this pass costs no extra DRAM reads. */
        if (crc_in)
            for (int i = 0; i < nd; i++)
                acc_in[i] = rj_crc32_update(acc_in[i], tin[i], tl);
        if (crc_out)
            for (int j = 0; j < np; j++)
                acc_out[j] = rj_crc32_update(acc_out[j], tout[j], tl);
    }

    if (crc_in)
        for (int i = 0; i < nd; i++)
            crc_in[i] = rj_crc32_final(acc_in[i]);
    if (crc_out)
        for (int j = 0; j < np; j++)
            crc_out[j] = rj_crc32_final(acc_out[j]);
}

/* ---------------- trampoline pool (api_batched.c pattern) ---------------- */

#define RJ_CRC_SLOTS 32

static struct rj_jit_crc *g_cslots[RJ_CRC_SLOTS];
static pthread_mutex_t g_cmu = PTHREAD_MUTEX_INITIALIZER;

#define DEF_CTRAMP(i)                                                      \
    static void rj_ctramp_##i(const void *in[], void *out[], size_t len,  \
                              uint32_t *crc_in, uint32_t *crc_out)        \
    {                                                                    \
        struct rj_jit_crc *c = g_cslots[i];                              \
        assert(c != NULL && "rj_fn_crc used after rj_free_crc");         \
        crc_driver_call(c, in, out, len, crc_in, crc_out);                \
    }

DEF_CTRAMP(0) DEF_CTRAMP(1) DEF_CTRAMP(2) DEF_CTRAMP(3)
DEF_CTRAMP(4) DEF_CTRAMP(5) DEF_CTRAMP(6) DEF_CTRAMP(7)
DEF_CTRAMP(8) DEF_CTRAMP(9) DEF_CTRAMP(10) DEF_CTRAMP(11)
DEF_CTRAMP(12) DEF_CTRAMP(13) DEF_CTRAMP(14) DEF_CTRAMP(15)
DEF_CTRAMP(16) DEF_CTRAMP(17) DEF_CTRAMP(18) DEF_CTRAMP(19)
DEF_CTRAMP(20) DEF_CTRAMP(21) DEF_CTRAMP(22) DEF_CTRAMP(23)
DEF_CTRAMP(24) DEF_CTRAMP(25) DEF_CTRAMP(26) DEF_CTRAMP(27)
DEF_CTRAMP(28) DEF_CTRAMP(29) DEF_CTRAMP(30) DEF_CTRAMP(31)

#if RJ_CRC_SLOTS != 32
#error "trampoline count must match RJ_CRC_SLOTS"
#endif

static rj_fn_crc ctramp_for(int slot)
{
    static rj_fn_crc tramps[RJ_CRC_SLOTS] = {
        rj_ctramp_0, rj_ctramp_1, rj_ctramp_2, rj_ctramp_3,
        rj_ctramp_4, rj_ctramp_5, rj_ctramp_6, rj_ctramp_7,
        rj_ctramp_8, rj_ctramp_9, rj_ctramp_10, rj_ctramp_11,
        rj_ctramp_12, rj_ctramp_13, rj_ctramp_14, rj_ctramp_15,
        rj_ctramp_16, rj_ctramp_17, rj_ctramp_18, rj_ctramp_19,
        rj_ctramp_20, rj_ctramp_21, rj_ctramp_22, rj_ctramp_23,
        rj_ctramp_24, rj_ctramp_25, rj_ctramp_26, rj_ctramp_27,
        rj_ctramp_28, rj_ctramp_29, rj_ctramp_30, rj_ctramp_31,
    };
    return tramps[slot];
}

/* ---------------- public API ---------------- */

rj_jit_crc *rj_compile_crc(const rj_topology *topo,
                           const rj_crc_options *opts,
                           char *err, size_t errlen)
{
    if (!topo) {
        if (err && errlen)
            snprintf(err, errlen, "null topology");
        return NULL;
    }
    rj_options o;
    crc_options_to_rj(opts, &o);
    if (o.loop_align < 0 ||
        (o.loop_align != 0 && (o.loop_align & (o.loop_align - 1)) != 0)) {
        if (err && errlen)
            snprintf(err, errlen,
                     "loop_align must be 0 or a power of two, got %d",
                     o.loop_align);
        return NULL;
    }

    rj_op op = rj_op_encode();
    if (op.oom) {
        if (err && errlen)
            snprintf(err, errlen, "out of memory");
        return NULL;
    }
    /* The CRC variant rides the full planner: candidate enumeration,
     * static pruning, and benchmarking select the encode kernel. */
    rj_jit *j = rj_compile(topo, &op, &o, err, errlen);
    rj_op_free(&op);
    if (!j)
        return NULL;

    struct rj_jit_crc *c = calloc(1, sizeof(*c));
    if (!c) {
        if (err && errlen)
            snprintf(err, errlen, "out of memory");
        rj_free(j);
        return NULL;
    }
    c->enc = j;
    c->enc_fn = rj_get_fn(j);
    c->ndata = rj_topo_ndata(topo);
    c->npar = rj_topo_npar(topo);
    if (c->ndata < 1 || c->ndata + c->npar > CRC_MAX_BUFS) {
        if (err && errlen)
            snprintf(err, errlen, "topology too wide for the CRC driver");
        rj_free(j);
        free(c);
        return NULL;
    }
    /* Plan-time tiling decision: nstreams = inputs + outputs touched
     * per tile. Stored once; the driver only reads it (zero runtime
     * cost for the decision). */
    c->tile = rj_tile_select((size_t)c->ndata + (size_t)c->npar);
    c->slot = -1;
    rj_info info;
    rj_get_info(j, &info);
    snprintf(c->desc, sizeof(c->desc), "%s", info.desc);
    return c;
}

rj_fn_crc rj_get_fn_crc(const rj_jit_crc *j)
{
    if (!j)
        return NULL;
    struct rj_jit_crc *c = (struct rj_jit_crc *)j;
    pthread_mutex_lock(&g_cmu);
    if (c->slot < 0) {
        for (int i = 0; i < RJ_CRC_SLOTS; i++) {
            if (!g_cslots[i]) {
                g_cslots[i] = c;
                c->slot = i;
                break;
            }
        }
    }
    int slot = c->slot;
    pthread_mutex_unlock(&g_cmu);
    return slot < 0 ? NULL : ctramp_for(slot);
}

void rj_free_crc(rj_jit_crc *j)
{
    if (!j)
        return;
    if (j->slot >= 0) {
        pthread_mutex_lock(&g_cmu);
        g_cslots[j->slot] = NULL;
        pthread_mutex_unlock(&g_cmu);
    }
    rj_free(j->enc);
    free(j);
}

void rj_get_info_crc(const rj_jit_crc *j, rj_info *info)
{
    memset(info, 0, sizeof(*info));
    if (!j)
        return;
    rj_get_info(j->enc, info);
    size_t n = strlen(info->desc);
    snprintf(info->desc + n, sizeof(info->desc) - n, " +crc");
}

int rj_crc_ndata(const rj_jit_crc *j)
{
    return j ? j->ndata : -1;
}

int rj_crc_npar(const rj_jit_crc *j)
{
    return j ? j->npar : -1;
}

size_t rj_crc_tile_bytes(const rj_jit_crc *j)
{
    return j ? j->tile : 0;
}
