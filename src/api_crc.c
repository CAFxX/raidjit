/* Integrated-CRC encode (queue item #38), built on the fusion
 * planner (queue item #8).
 *
 * rj_compile_crc is rj_fuse_compile with the sequence
 * [ ENCODE, CRC targeting op 0 ]. The fused tile loop reads every
 * stripe byte once: per tile it runs the encode kernel, then
 * accumulates CRC-32 over the tile's input (ndata) and freshly
 * computed parity (npar) buffers while they are hot in cache.
 *
 * The public rj_fn_crc contract takes separate crc_in/crc_out arrays
 * and allows NULL for either side; the fused driver takes one flat
 * crc array (inputs first). The rj_fn_crc entry point is an adapter
 * trampoline that merges the two call-site arrays, delegates via
 * rj_fused_call (no extra trampoline slot), and splits the results
 * back. A NULL side skips the copy and the accumulation (kernel bytes
 * are unaffected).
 */
#include "fused.h"
#include "crc.h"

#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct rj_jit_crc {
    rj_fused *f;
    int ndata, npar;
    int slot; /* adapter trampoline slot, -1 when not installed */
    rj_info cached; /* sub-info for rj_get_info_crc; " crc" appended */
    int cached_ok;
};

void rj_crc_options_default(rj_crc_options *o)
{
    memset(o, 0, sizeof(*o));
    o->benchmark = 1;
    o->loop_align = 32;
}

rj_jit_crc *rj_compile_crc(const rj_topology *topo,
                           const rj_crc_options *opts,
                           char *err, size_t errlen)
{
    if (!topo) {
        if (err && errlen)
            snprintf(err, errlen, "null topology");
        return NULL;
    }
    rj_fuse_options fo;
    rj_fuse_options_default(&fo);
    if (opts) {
        fo.benchmark = opts->benchmark;
        fo.bench_bytes = opts->bench_bytes;
        fo.bench_iters = opts->bench_iters;
        fo.verbose = opts->verbose;
        fo.loop_align = opts->loop_align;
        fo.emit_endbr = opts->emit_endbr;
        fo.emit_shadow_stack = opts->emit_shadow_stack;
        fo.nt_threshold = opts->nt_threshold;
        fo.prefetch_distance = opts->prefetch_distance;
    }
    if (fo.loop_align < 0 ||
        (fo.loop_align != 0 && (fo.loop_align & (fo.loop_align - 1)) != 0)) {
        if (err && errlen)
            snprintf(err, errlen,
                     "loop_align must be 0 or a power of two, got %d",
                     fo.loop_align);
        return NULL;
    }

    rj_fop_t fops[2];
    memset(fops, 0, sizeof(fops));
    fops[0].kind = RJ_FOP_ENCODE;
    fops[1].kind = RJ_FOP_CRC;
    fops[1].target = 0;
    rj_fused *f = rj_fuse_compile(topo, fops, 2, &fo, err, errlen);
    if (!f)
        return NULL;

    rj_jit_crc *jc = calloc(1, sizeof(*jc));
    if (!jc) {
        rj_free_fused(f);
        if (err && errlen)
            snprintf(err, errlen, "out of memory");
        return NULL;
    }
    jc->f = f;
    jc->ndata = rj_fused_sub_ndata(f, 0);
    jc->npar = rj_fused_sub_npar(f, 0);
    jc->slot = -1;
    return jc;
}

/* ---------------- rj_fn_crc adapter trampolines ----------------
 * The adapter merges crc_in/crc_out into the flat crc[] the fused
 * driver wants, calls rj_fused_call directly (no extra trampoline
 * slot consumed), then splits results back. 32 static slots, the
 * api_batched.c pattern.
 */
#define RJ_CRC_SLOTS 32

static struct rj_jit_crc *g_slots[RJ_CRC_SLOTS];
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;

/* Merge: fused layout wants inputs first, then outputs. A NULL side is
 * skipped (bytes still processed; no CRC accumulated for that side). */
static void crc_call(struct rj_jit_crc *jc, const void *in[],
                     void *out[], size_t len, uint32_t *crc_in,
                     uint32_t *crc_out)
{
    int nd = jc->ndata, np = jc->npar;
    uint32_t flat[256]; /* ndata+npar <= RJ_MAX_DISKS (256) */
    int have = crc_in != NULL || crc_out != NULL;
    uint32_t *flatp = have ? flat : NULL;
    /* When only one side is present the driver accumulates into the
     * flat prefix (inputs-first order); results are split below. */
    rj_fused_call(jc->f, in, out, len, flatp);
    if (flatp) {
        int o = 0;
        if (crc_in) {
            for (int i = 0; i < nd; i++)
                crc_in[i] = flat[o++];
        } else {
            o += nd;
        }
        if (crc_out) {
            for (int j = 0; j < np; j++)
                crc_out[j] = flat[o++];
        }
    }
}

#define DEF_TRAMP(i)                                                       \
    static void rj_crc_tramp_##i(const void *in[], void *out[],         \
                                 size_t len, uint32_t *crc_in,          \
                                 uint32_t *crc_out)                    \
    {                                                                    \
        struct rj_jit_crc *jc = g_slots[i];                              \
        assert(jc != NULL && "rj_fn_crc used after rj_free_crc");        \
        crc_call(jc, in, out, len, crc_in, crc_out);                      \
    }

DEF_TRAMP(0) DEF_TRAMP(1) DEF_TRAMP(2) DEF_TRAMP(3)
DEF_TRAMP(4) DEF_TRAMP(5) DEF_TRAMP(6) DEF_TRAMP(7)
DEF_TRAMP(8) DEF_TRAMP(9) DEF_TRAMP(10) DEF_TRAMP(11)
DEF_TRAMP(12) DEF_TRAMP(13) DEF_TRAMP(14) DEF_TRAMP(15)
DEF_TRAMP(16) DEF_TRAMP(17) DEF_TRAMP(18) DEF_TRAMP(19)
DEF_TRAMP(20) DEF_TRAMP(21) DEF_TRAMP(22) DEF_TRAMP(23)
DEF_TRAMP(24) DEF_TRAMP(25) DEF_TRAMP(26) DEF_TRAMP(27)
DEF_TRAMP(28) DEF_TRAMP(29) DEF_TRAMP(30) DEF_TRAMP(31)

#if RJ_CRC_SLOTS != 32
#error "trampoline count must match RJ_CRC_SLOTS"
#endif

static rj_fn_crc tramp_for(int slot)
{
    static rj_fn_crc tramps[RJ_CRC_SLOTS] = {
        rj_crc_tramp_0, rj_crc_tramp_1, rj_crc_tramp_2, rj_crc_tramp_3,
        rj_crc_tramp_4, rj_crc_tramp_5, rj_crc_tramp_6, rj_crc_tramp_7,
        rj_crc_tramp_8, rj_crc_tramp_9, rj_crc_tramp_10, rj_crc_tramp_11,
        rj_crc_tramp_12, rj_crc_tramp_13, rj_crc_tramp_14, rj_crc_tramp_15,
        rj_crc_tramp_16, rj_crc_tramp_17, rj_crc_tramp_18, rj_crc_tramp_19,
        rj_crc_tramp_20, rj_crc_tramp_21, rj_crc_tramp_22, rj_crc_tramp_23,
        rj_crc_tramp_24, rj_crc_tramp_25, rj_crc_tramp_26, rj_crc_tramp_27,
        rj_crc_tramp_28, rj_crc_tramp_29, rj_crc_tramp_30, rj_crc_tramp_31,
    };
    return tramps[slot];
}

rj_fn_crc rj_get_fn_crc(const rj_jit_crc *jc)
{
    if (!jc)
        return NULL;
    struct rj_jit_crc *j = (struct rj_jit_crc *)jc;
    pthread_mutex_lock(&g_mu);
    if (j->slot < 0) {
        for (int i = 0; i < RJ_CRC_SLOTS; i++) {
            if (!g_slots[i]) {
                g_slots[i] = j;
                j->slot = i;
                break;
            }
        }
    }
    int slot = j->slot;
    pthread_mutex_unlock(&g_mu);
    return slot < 0 ? NULL : tramp_for(slot);
}

void rj_free_crc(rj_jit_crc *jc)
{
    if (!jc)
        return;
    if (jc->slot >= 0) {
        pthread_mutex_lock(&g_mu);
        g_slots[jc->slot] = NULL;
        pthread_mutex_unlock(&g_mu);
    }
    rj_free_fused(jc->f);
    free(jc);
}

void rj_get_info_crc(const rj_jit_crc *jc, rj_info *info)
{
    memset(info, 0, sizeof(*info));
    if (!jc)
        return;
    struct rj_jit_crc *j = (struct rj_jit_crc *)jc;
    if (!j->cached_ok) {
        rj_fused_sub_info(j->f, 0, &j->cached);
        size_t n = strlen(j->cached.desc);
        if (n + 5 < sizeof(j->cached.desc))
            memcpy(j->cached.desc + n, " +crc", 5);
        j->cached_ok = 1;
    }
    *info = j->cached;
}

int rj_crc_ndata(const rj_jit_crc *jc)
{
    return jc ? jc->ndata : -1;
}

int rj_crc_npar(const rj_jit_crc *jc)
{
    return jc ? jc->npar : -1;
}

size_t rj_crc_tile_bytes(const rj_jit_crc *jc)
{
    return jc ? rj_fused_tile_bytes(jc->f) : 0;
}
