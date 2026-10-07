/* Cache-blocked striping: plan-time tile-size selection.
 *
 * rj_tile_select is a pure function of nstreams: the largest power of
 * two with nstreams*tile <= LLC/2, clamped to [TILE_MIN, TILE_MAX].
 * Computed once per compile; the drivers below (#38, #8, #10) only
 * read the stored value, so the tiling decision costs nothing at
 * runtime.
 */
#include "striping.h"
#include "../include/raidjit_fused.h"

size_t rj_tile_select(size_t nstreams)
{
    if (nstreams == 0)
        nstreams = 1;
    size_t budget = RJ_STRIPING_LLC_BYTES / 2;
    size_t per_stream = budget / nstreams; /* nstreams >= 1: no div-by-0 */

    /* Largest power of two <= per_stream. per_stream >= 1 always
     * (budget >> nstreams for any sane nstreams), so the loop
     * terminates with tile >= 1. */
    size_t tile = (size_t)1 << 62;
    while (tile > per_stream)
        tile >>= 1;

    if (tile < RJ_STRIPING_TILE_MIN)
        tile = RJ_STRIPING_TILE_MIN;
    if (tile > RJ_STRIPING_TILE_MAX)
        tile = RJ_STRIPING_TILE_MAX;
    return tile;
}

/* Cache-blocked striping (queue item #10).
 *
 * rj_compile_tiled compiles one op through the public rj_compile
 * planner, then wraps the winning rj_fn in a tile loop sized by
 * rj_tile_select (plan time; zero runtime cost). The kernel sees only
 * disjoint tiles, so the result is bit-exact vs the untiled call.
 *
 * Entry point: a static trampoline pool bound to the handle (the
 * api_batched.c pattern); rj_fn carries no context pointer. Lifetime,
 * thread-safety, and slot-reuse rules match api_batched.h.
 */

#include "../include/raidjit_fused.h"

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct rj_tiled {
    rj_jit *inner;
    rj_fn fn;
    int nin, nout;
    size_t tile;
    int slot;
    rj_info cached;
    int cached_ok;
};

/* Per-op buffer counts from the rj_fn contract. Only the ops the
 * tiled driver supports are listed; anything else fails at compile
 * time (fail closed). */
static int tiled_counts(const rj_op *op, const rj_topology *topo,
                        int *nin, int *nout)
{
    int ndata = rj_topo_ndata(topo);
    int npar = rj_topo_npar(topo);
    if (ndata < 1 || ndata + npar > 256)
        return -1;
    if (op->is_parity_update) {
        int nc = op->nchanged;
        if (nc < 1 || nc > ndata)
            return -1;
        *nin = 2 * nc + npar;
        *nout = npar;
        return 0;
    }
    if (op->is_recover) {
        int nf = op->nfailed;
        if (nf < 1 || nf > npar)
            return -1;
        *nin = ndata + npar - nf;
        *nout = nf;
        return 0;
    }
    /* encode */
    *nin = ndata;
    *nout = npar;
    return 0;
}

rj_tiled *rj_compile_tiled(const rj_topology *topo, const rj_op *op,
                           const rj_options *opts, char *err,
                           size_t errlen)
{
    if (!topo || !op) {
        if (err && errlen)
            snprintf(err, errlen, "null topology or op");
        return NULL;
    }
    int nin = 0, nout = 0;
    if (tiled_counts(op, topo, &nin, &nout) != 0) {
        if (err && errlen)
            snprintf(err, errlen,
                     "tiled driver does not support this op");
        return NULL;
    }
    rj_jit *inner = rj_compile(topo, op, opts, err, errlen);
    if (!inner)
        return NULL;
    rj_tiled *t = calloc(1, sizeof(*t));
    if (!t) {
        rj_free(inner);
        if (err && errlen)
            snprintf(err, errlen, "out of memory");
        return NULL;
    }
    t->inner = inner;
    t->fn = rj_get_fn(inner);
    t->nin = nin;
    t->nout = nout;
    t->tile = rj_tile_select((size_t)nin + (size_t)nout);
    t->slot = -1;
    return t;
}

static void tiled_driver_call(const struct rj_tiled *t,
                              const void *const *in, void *const *out,
                              size_t len)
{
    assert(in != NULL);
    assert(out != NULL || t->nout == 0);
    assert(len >= 1);
    int nin = t->nin, nout = t->nout;
    size_t tile = t->tile;
    const void *tin[256];
    void *tout[256];
    for (size_t off = 0; off < len; off += tile) {
        size_t tl = len - off < tile ? len - off : tile;
        for (int i = 0; i < nin; i++) {
            tin[i] = (const uint8_t *)in[i] + off;
            assert(((uintptr_t)tin[i] & 63) == 0);
        }
        for (int j = 0; j < nout; j++)
            tout[j] = (uint8_t *)out[j] + off;
        t->fn(tin, nout ? tout : NULL, tl);
    }
}

#define RJ_TILED_SLOTS 32

static struct rj_tiled *g_tslots[RJ_TILED_SLOTS];
static pthread_mutex_t g_tmu = PTHREAD_MUTEX_INITIALIZER;

#define DEF_TTRAMP(i)                                                  \
    static void rj_ttramp_##i(const void *in[], void *out[],          \
                              size_t len)                           \
    {                                                                  \
        struct rj_tiled *t = g_tslots[i];                             \
        assert(t != NULL && "rj_fn used after rj_free_tiled");         \
        tiled_driver_call(t, in, out, len);                             \
    }

DEF_TTRAMP(0) DEF_TTRAMP(1) DEF_TTRAMP(2) DEF_TTRAMP(3)
DEF_TTRAMP(4) DEF_TTRAMP(5) DEF_TTRAMP(6) DEF_TTRAMP(7)
DEF_TTRAMP(8) DEF_TTRAMP(9) DEF_TTRAMP(10) DEF_TTRAMP(11)
DEF_TTRAMP(12) DEF_TTRAMP(13) DEF_TTRAMP(14) DEF_TTRAMP(15)
DEF_TTRAMP(16) DEF_TTRAMP(17) DEF_TTRAMP(18) DEF_TTRAMP(19)
DEF_TTRAMP(20) DEF_TTRAMP(21) DEF_TTRAMP(22) DEF_TTRAMP(23)
DEF_TTRAMP(24) DEF_TTRAMP(25) DEF_TTRAMP(26) DEF_TTRAMP(27)
DEF_TTRAMP(28) DEF_TTRAMP(29) DEF_TTRAMP(30) DEF_TTRAMP(31)

#if RJ_TILED_SLOTS != 32
#error "trampoline count must match RJ_TILED_SLOTS"
#endif

static rj_fn ttramp_for(int slot)
{
    static rj_fn tramps[RJ_TILED_SLOTS] = {
        rj_ttramp_0, rj_ttramp_1, rj_ttramp_2, rj_ttramp_3,
        rj_ttramp_4, rj_ttramp_5, rj_ttramp_6, rj_ttramp_7,
        rj_ttramp_8, rj_ttramp_9, rj_ttramp_10, rj_ttramp_11,
        rj_ttramp_12, rj_ttramp_13, rj_ttramp_14, rj_ttramp_15,
        rj_ttramp_16, rj_ttramp_17, rj_ttramp_18, rj_ttramp_19,
        rj_ttramp_20, rj_ttramp_21, rj_ttramp_22, rj_ttramp_23,
        rj_ttramp_24, rj_ttramp_25, rj_ttramp_26, rj_ttramp_27,
        rj_ttramp_28, rj_ttramp_29, rj_ttramp_30, rj_ttramp_31,
    };
    return tramps[slot];
}

rj_fn rj_get_fn_tiled(const rj_tiled *t)
{
    if (!t)
        return NULL;
    struct rj_tiled *tt = (struct rj_tiled *)t;
    pthread_mutex_lock(&g_tmu);
    if (tt->slot < 0) {
        for (int i = 0; i < RJ_TILED_SLOTS; i++) {
            if (!g_tslots[i]) {
                g_tslots[i] = tt;
                tt->slot = i;
                break;
            }
        }
    }
    int slot = tt->slot;
    pthread_mutex_unlock(&g_tmu);
    return slot < 0 ? NULL : ttramp_for(slot);
}

void rj_free_tiled(rj_tiled *t)
{
    if (!t)
        return;
    if (t->slot >= 0) {
        pthread_mutex_lock(&g_tmu);
        g_tslots[t->slot] = NULL;
        pthread_mutex_unlock(&g_tmu);
    }
    rj_free(t->inner);
    free(t);
}

void rj_get_info_tiled(const rj_tiled *t, rj_info *info)
{
    memset(info, 0, sizeof(*info));
    if (!t)
        return;
    struct rj_tiled *tt = (struct rj_tiled *)t;
    if (!tt->cached_ok) {
        rj_get_info(tt->inner, &tt->cached);
        size_t n = strlen(tt->cached.desc);
        if (n + 8 < sizeof(tt->cached.desc))
            memcpy(tt->cached.desc + n, " +tiled", 8);
        tt->cached_ok = 1;
    }
    *info = tt->cached;
}

size_t rj_tiled_tile_bytes(const rj_tiled *t)
{
    return t ? t->tile : 0;
}
