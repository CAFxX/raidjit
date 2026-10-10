/* Generalized fusion planner (queue item #8).
 *
 * rj_fuse_compile takes a short sequence of ops over one topology and
 * fuses the compatible ones into a single kernel invocation: one entry
 * point, one cache-blocked tile loop. Each data op (ENCODE /
 * PARITY_UPDATE) is compiled through the public rj_compile planner
 * (full candidate enumeration / pruning / benchmarking), so every
 * sub-kernel is planner-selected exactly as if compiled alone. The CRC
 * op accumulates CRC-32 over its target's tile buffers while they are
 * hot in cache.
 *
 * Compatibility is checked up front; an incompatible sequence fails
 * loudly at compile time, never silently.
 *
 * The entry point is one of a static trampoline pool bound to the
 * handle (the api_batched.c pattern): rj_fn_fused carries no context
 * pointer. Lifetime, thread-safety, and slot-reuse rules match
 * api_batched.h.
 */
#include "fused.h"
#include "api.h"
#include "crc.h"
#include "striping.h"
#ifdef __x86_64__
#include "x86_64/crc_fold.h"
#endif

#include <assert.h>
#include <pthread.h>
#include <stdalign.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct rj_fused {
    int nops;
    rj_fop_kind_t kinds[RJ_FUSED_MAX_OPS];
    rj_jit *units[RJ_FUSED_MAX_OPS]; /* data ops; NULL for the CRC op */
    rj_fn fns[RJ_FUSED_MAX_OPS];
    int nin[RJ_FUSED_MAX_OPS], nout[RJ_FUSED_MAX_OPS];
    size_t base_in[RJ_FUSED_MAX_OPS], base_out[RJ_FUSED_MAX_OPS];
    size_t n_flat_in, n_flat_out;
    int crc_op;    /* sequence index of the CRC op, -1 when absent */
    int crc_tgt;   /* its target data-op index */
    int crc_fold;  /* Phase 2: 1 = the target unit has in-kernel CRC folding
                    * (4-arg kernel); 0 = scalar fallback (3-arg kernel) */
    int ndata, npar; /* topology dims (for the CRC target layout) */
    size_t tile;   /* plan-time tile size (rj_tile_select) */
    int slot;      /* trampoline slot, -1 when not installed */
};

void rj_fuse_options_default(rj_fuse_options *o)
{
    memset(o, 0, sizeof(*o));
    o->benchmark = 1;
    o->loop_align = 32;
}

static void fuse_options_to_rj(const rj_fuse_options *fo, rj_options *o)
{
    rj_options_default(o);
    if (!fo)
        return;
    o->benchmark = fo->benchmark;
    o->bench_bytes = fo->bench_bytes;
    o->bench_iters = fo->bench_iters;
    o->verbose = fo->verbose;
    o->loop_align = fo->loop_align;
    o->emit_endbr = fo->emit_endbr;
    o->emit_shadow_stack = fo->emit_shadow_stack;
    o->nt_threshold = fo->nt_threshold;
    o->prefetch_distance = fo->prefetch_distance;
}

/* Sort + dedupe a changed[] list. Returns the cleaned count (0 on
 * invalid input). n <= ndata <= 256, so the stack buffer is bounded. */
static int clean_changed(const int *changed, int nchanged, int ndata,
                         int *out)
{
    if (!changed || nchanged < 1 || nchanged > ndata)
        return 0;
    int n = 0;
    for (int i = 0; i < nchanged; i++) {
        if (changed[i] < 0 || changed[i] >= ndata)
            return 0;
        out[n++] = changed[i];
    }
    /* Insertion sort (n <= 256): no qsort comparator indirection. */
    for (int i = 1; i < n; i++) {
        int v = out[i], j = i - 1;
        while (j >= 0 && out[j] > v) {
            out[j + 1] = out[j];
            j--;
        }
        out[j + 1] = v;
    }
    int w = 0;
    for (int i = 0; i < n; i++)
        if (i == 0 || out[i] != out[i - 1])
            out[w++] = out[i];
    return w;
}

rj_fused *rj_fuse_compile(const rj_topology *topo, const rj_fop_t *ops,
                          int nops, const rj_fuse_options *opts,
                          char *err, size_t errlen)
{
    if (!topo || !ops) {
        if (err && errlen)
            snprintf(err, errlen, "null topology or op list");
        return NULL;
    }
    if (nops < 1 || nops > RJ_FUSED_MAX_OPS) {
        if (err && errlen)
            snprintf(err, errlen,
                     "fused sequence needs 1..%d ops, got %d",
                     RJ_FUSED_MAX_OPS, nops);
        return NULL;
    }
    rj_options o;
    fuse_options_to_rj(opts, &o);
    if (o.loop_align < 0 ||
        (o.loop_align != 0 && (o.loop_align & (o.loop_align - 1)) != 0)) {
        if (err && errlen)
            snprintf(err, errlen,
                     "loop_align must be 0 or a power of two, got %d",
                     o.loop_align);
        return NULL;
    }
    int ndata = rj_topo_ndata(topo);
    int npar = rj_topo_npar(topo);
    if (ndata < 1 || ndata + npar > 256) {
        if (err && errlen)
            snprintf(err, errlen, "topology too wide for the fused driver");
        return NULL;
    }

    /* ---- compatibility check (up front, before any compile) ---- */
    int crc_op = -1, crc_tgt = -1;
    for (int k = 0; k < nops; k++) {
        rj_fop_kind_t kd = ops[k].kind;
        if (kd != RJ_FOP_ENCODE && kd != RJ_FOP_PARITY_UPDATE &&
            kd != RJ_FOP_CRC) {
            if (err && errlen)
                snprintf(err, errlen, "fused op %d: unknown kind %d", k,
                         (int)kd);
            return NULL;
        }
        if (kd == RJ_FOP_PARITY_UPDATE) {
            int tmp[256];
            if (clean_changed(ops[k].changed, ops[k].nchanged, ndata,
                              tmp) < 1) {
                if (err && errlen)
                    snprintf(err, errlen,
                             "fused op %d: parity update needs 1..ndata "
                             "valid changed disks", k);
                return NULL;
            }
        }
        if (kd == RJ_FOP_CRC) {
            if (crc_op >= 0) {
                if (err && errlen)
                    snprintf(err, errlen,
                             "fused op %d: at most one CRC op per sequence",
                             k);
                return NULL;
            }
            int t = ops[k].target;
            if (t < 0 || t >= k || (ops[t].kind != RJ_FOP_ENCODE &&
                                   ops[t].kind != RJ_FOP_PARITY_UPDATE)) {
                if (err && errlen)
                    snprintf(err, errlen,
                             "fused op %d: CRC target must be an earlier "
                             "ENCODE/PARITY_UPDATE op", k);
                return NULL;
            }
            crc_op = k;
            crc_tgt = t;
        }
    }

    /* ---- compile each data op through the public planner ---- */
    struct rj_fused *f = calloc(1, sizeof(*f));
    if (!f) {
        if (err && errlen)
            snprintf(err, errlen, "out of memory");
        return NULL;
    }
    f->nops = nops;
    f->crc_op = crc_op;
    f->crc_tgt = crc_tgt;
    f->ndata = ndata;
    f->npar = npar;
    f->slot = -1;

    for (int k = 0; k < nops; k++) {
        rj_fop_kind_t kd = ops[k].kind;
        f->kinds[k] = kd;
        if (kd == RJ_FOP_CRC)
            continue;
        rj_op op;
        memset(&op, 0, sizeof(op));
        int ni = 0, no = 0;
        if (kd == RJ_FOP_ENCODE) {
            op = rj_op_encode();
            ni = ndata;
            no = npar;
        } else { /* PARITY_UPDATE */
            int ch[256];
            int nc = clean_changed(ops[k].changed, ops[k].nchanged,
                                   ndata, ch);
            assert(nc >= 1); /* re-validated above */
            op = rj_op_parity_update(ch, nc);
            ni = 2 * nc + npar;
            no = npar;
        }
        if (op.oom) {
            if (err && errlen)
                snprintf(err, errlen, "out of memory");
            goto fail;
        }
        rj_jit *u = NULL;
        /* Phase 2: for the CRC target, try in-kernel folding first (when
         * the uarch gate is open). If want_crc compile fails (no buildable
         * kernel), fall back to the plain compile + scalar CRC path. */
        if (k == crc_tgt) {
            rj_cpu_info cpu;
            rj_cpu_detect(&cpu);
            if (fused_crc_interleave_ok(&cpu)) {
                u = rj_compile_want_crc(topo, &op, &o, err, errlen);
                /* If want_crc fails, u==NULL and we fall through to the
                 * plain compile below. */
            }
        }
        if (!u)
            u = rj_compile(topo, &op, &o, err, errlen);
        rj_op_free(&op);
        if (!u)
            goto fail;
        f->units[k] = u;
        f->fns[k] = rj_get_fn(u);
        f->nin[k] = ni;
        f->nout[k] = no;
        /* Phase 2: record whether this unit folds (for the driver). */
        if (k == crc_tgt)
            f->crc_fold = rj_jit_handle_crc_fold(u);
    }

    /* ---- flat buffer layout (sequence order) ---- */
    size_t bi = 0, bo = 0;
    for (int k = 0; k < nops; k++) {
        f->base_in[k] = bi;
        f->base_out[k] = bo;
        bi += (size_t)f->nin[k];
        bo += (size_t)f->nout[k];
    }
    f->n_flat_in = bi;
    f->n_flat_out = bo;
    if (bi + bo > RJ_FUSED_MAX_FLAT) {
        if (err && errlen)
            snprintf(err, errlen,
                     "fused buffer layout too large (%zu buffers)",
                     bi + bo);
        goto fail;
    }

    /* ---- plan-time tiling decision (zero runtime cost) ---- */
    size_t nstreams = 0;
    for (int k = 0; k < nops; k++)
        nstreams += (size_t)f->nin[k] + (size_t)f->nout[k];
    f->tile = rj_tile_select(nstreams);
    return f;

fail:
    for (int k = 0; k < nops; k++)
        if (f->units[k])
            rj_free(f->units[k]);
    free(f);
    return NULL;
}

/* One fused call: per tile, run each data op's kernel, then (when a
 * CRC op is present and crc != NULL) accumulate CRC-32 over the
 * target's tile buffers while they are hot in cache. crc == NULL with
 * a CRC op present skips the accumulation (the kernels still run). */
static void fused_driver_call(const struct rj_fused *f,
                              const void *const *in, void *const *out,
                              size_t len, uint32_t *crc)
{
    assert(in != NULL);
    assert(out != NULL || f->n_flat_out == 0);
    assert(len >= 1);
    int nops = f->nops;
    size_t tile = f->tile;

    const void *ftin[RJ_FUSED_MAX_FLAT];
    void *ftout[RJ_FUSED_MAX_FLAT];
    /* The CRC target is one data op: its ni+no <= 1024 by the layout
     * cap (worst case PARITY_UPDATE: 768+256). */
    uint32_t acc[RJ_FUSED_MAX_FLAT / 2];
    int t = f->crc_tgt;
    int tn = t >= 0 ? f->nin[t] : 0;
    int tno = t >= 0 ? f->nout[t] : 0;
    assert(t < 0 || tn + tno <= RJ_FUSED_MAX_FLAT / 2);
    /* REDUX: fold state is 16 bytes per stream (one 128-bit state;
     * spec §3.1-§3.2). Allocated on the stack, 64-byte aligned (the
     * contract requires 16). */
    alignas(64) uint8_t fold_state[RJ_FUSED_MAX_FLAT / 2 * 16];
    int nstreams = tn + tno;
    if (f->crc_fold) {
        assert(t >= 0);
        /* Initialize fold state: the backend defines the init convention.
         * For x86-64: 0xFFFFFFFF in low 32 bits (reflected init), rest
         * zero. The backend's kernel entry loads this. */
        memset(fold_state, 0, (size_t)nstreams * 16);
        for (int i = 0; i < nstreams; i++) {
            uint32_t *p = (uint32_t *)(fold_state + i * 16);
            p[0] = 0xFFFFFFFFu;
        }
    } else {
        if (crc && t >= 0)
            for (int i = 0; i < tn + tno; i++)
                rj_crc32_init(&acc[i]);
    }

    for (size_t off = 0; off < len; off += tile) {
        size_t tl = len - off < tile ? len - off : tile;
        for (int k = 0; k < nops; k++) {
            if (f->kinds[k] == RJ_FOP_CRC)
                continue;
            size_t bii = f->base_in[k], boo = f->base_out[k];
            for (int i = 0; i < f->nin[k]; i++) {
                ftin[bii + (size_t)i] =
                    (const uint8_t *)in[bii + (size_t)i] + off;
                assert(((uintptr_t)ftin[bii + (size_t)i] & 63) == 0);
            }
            for (int j = 0; j < f->nout[k]; j++)
                ftout[boo + (size_t)j] =
                    (uint8_t *)out[boo + (size_t)j] + off;
            /* Phase 2: 4-arg call when folding, 3-arg otherwise. */
            if (f->crc_fold && k == t) {
                rj_fn_crc_fold fn4 = (rj_fn_crc_fold)(void *)f->fns[k];
                fn4(&ftin[bii], f->nout[k] ? &ftout[boo] : NULL, tl,
                    fold_state);
            } else {
                f->fns[k](&ftin[bii], f->nout[k] ? &ftout[boo] : NULL, tl);
            }
        }
        if (crc && t >= 0 && !f->crc_fold) {
            /* Scalar fallback: the target's tile buffers are hot in cache
             * here. (When folding, the kernel already updated fold_state;
             * no scalar pass needed.) */
            size_t bii = f->base_in[t], boo = f->base_out[t];
            for (int i = 0; i < tn; i++)
                acc[i] = rj_crc32_fast(acc[i], ftin[bii + (size_t)i],
                                       tl);
            for (int j = 0; j < tno; j++)
                acc[tn + j] = rj_crc32_fast(acc[tn + j],
                                           ftout[boo + (size_t)j], tl);
        }
    }

    if (crc && t >= 0) {
        if (f->crc_fold) {
            /* Phase 2 v2: handle tail bytes (len % 16) driver-side, then
             * finalize each stream's 128-bit state to CRC-32 via per-stream
             * Barrett reduction. No cross-lane combination (lanes are
             * independent streams). */
#ifdef __x86_64__
            size_t tail_len = len % 16;
            if (tail_len > 0) {
                size_t bii = f->base_in[t], boo = f->base_out[t];
                /* Tail bytes are at the end of each stream. */
                for (int i = 0; i < tn; i++) {
                    const uint8_t *p =
                        (const uint8_t *)in[bii + (size_t)i] + len - tail_len;
                    rj_crc_fold_tail(&fold_state[i * 16], p, tail_len);
                }
                for (int j = 0; j < tno; j++) {
                    const uint8_t *p =
                        (const uint8_t *)out[boo + (size_t)j] + len - tail_len;
                    rj_crc_fold_tail(&fold_state[(tn + j) * 16], p, tail_len);
                }
            }
            rj_crc_fold_finalize_x86(fold_state, nstreams, len, crc);
#else
            /* AArch64/RISC-V gates are closed; this is unreachable. */
            for (int i = 0; i < nstreams; i++)
                crc[i] = 0;
#endif
        } else {
            for (int i = 0; i < tn + tno; i++)
                crc[i] = rj_crc32_final(acc[i]);
        }
    }
}

/* Direct driver call for internal composition (no trampoline slot
 * consumed): same contract as rj_fn_fused. */
void rj_fused_call(const rj_fused *f, const void *const *in, void *const *out,
                   size_t len, uint32_t *crc)
{
    assert(f != NULL);
    fused_driver_call(f, in, out, len, crc);
}

/* ---------------- trampoline pool (api_batched.c pattern) ---------------- */

#define RJ_FUSED_SLOTS 32

static struct rj_fused *g_fslots[RJ_FUSED_SLOTS];
static pthread_mutex_t g_fmu = PTHREAD_MUTEX_INITIALIZER;

#define DEF_FTRAMP(i)                                                      \
    static void rj_ftramp_##i(const void *const *in, void *const *out,    \
                              size_t len, uint32_t *crc)                 \
    {                                                                    \
        struct rj_fused *f = g_fslots[i];                                \
        assert(f != NULL && "rj_fn_fused used after rj_free_fused");     \
        fused_driver_call(f, in, out, len, crc);                          \
    }

DEF_FTRAMP(0) DEF_FTRAMP(1) DEF_FTRAMP(2) DEF_FTRAMP(3)
DEF_FTRAMP(4) DEF_FTRAMP(5) DEF_FTRAMP(6) DEF_FTRAMP(7)
DEF_FTRAMP(8) DEF_FTRAMP(9) DEF_FTRAMP(10) DEF_FTRAMP(11)
DEF_FTRAMP(12) DEF_FTRAMP(13) DEF_FTRAMP(14) DEF_FTRAMP(15)
DEF_FTRAMP(16) DEF_FTRAMP(17) DEF_FTRAMP(18) DEF_FTRAMP(19)
DEF_FTRAMP(20) DEF_FTRAMP(21) DEF_FTRAMP(22) DEF_FTRAMP(23)
DEF_FTRAMP(24) DEF_FTRAMP(25) DEF_FTRAMP(26) DEF_FTRAMP(27)
DEF_FTRAMP(28) DEF_FTRAMP(29) DEF_FTRAMP(30) DEF_FTRAMP(31)

#if RJ_FUSED_SLOTS != 32
#error "trampoline count must match RJ_FUSED_SLOTS"
#endif

static rj_fn_fused ftramp_for(int slot)
{
    static rj_fn_fused tramps[RJ_FUSED_SLOTS] = {
        rj_ftramp_0, rj_ftramp_1, rj_ftramp_2, rj_ftramp_3,
        rj_ftramp_4, rj_ftramp_5, rj_ftramp_6, rj_ftramp_7,
        rj_ftramp_8, rj_ftramp_9, rj_ftramp_10, rj_ftramp_11,
        rj_ftramp_12, rj_ftramp_13, rj_ftramp_14, rj_ftramp_15,
        rj_ftramp_16, rj_ftramp_17, rj_ftramp_18, rj_ftramp_19,
        rj_ftramp_20, rj_ftramp_21, rj_ftramp_22, rj_ftramp_23,
        rj_ftramp_24, rj_ftramp_25, rj_ftramp_26, rj_ftramp_27,
        rj_ftramp_28, rj_ftramp_29, rj_ftramp_30, rj_ftramp_31,
    };
    return tramps[slot];
}

rj_fn_fused rj_get_fn_fused(const rj_fused *f)
{
    if (!f)
        return NULL;
    struct rj_fused *ff = (struct rj_fused *)f;
    pthread_mutex_lock(&g_fmu);
    if (ff->slot < 0) {
        for (int i = 0; i < RJ_FUSED_SLOTS; i++) {
            if (!g_fslots[i]) {
                g_fslots[i] = ff;
                ff->slot = i;
                break;
            }
        }
    }
    int slot = ff->slot;
    pthread_mutex_unlock(&g_fmu);
    return slot < 0 ? NULL : ftramp_for(slot);
}

void rj_free_fused(rj_fused *f)
{
    if (!f)
        return;
    if (f->slot >= 0) {
        pthread_mutex_lock(&g_fmu);
        g_fslots[f->slot] = NULL;
        pthread_mutex_unlock(&g_fmu);
    }
    for (int k = 0; k < f->nops; k++)
        if (f->units[k])
            rj_free(f->units[k]);
    free(f);
}

int rj_fused_op_nin(const rj_fused *f, int idx)
{
    if (!f || idx < 0 || idx >= f->nops)
        return -1;
    return f->nin[idx];
}

int rj_fused_op_nout(const rj_fused *f, int idx)
{
    if (!f || idx < 0 || idx >= f->nops)
        return -1;
    return f->nout[idx];
}

int rj_fused_nops(const rj_fused *f)
{
    return f ? f->nops : 0;
}

size_t rj_fused_tile_bytes(const rj_fused *f)
{
    return f ? f->tile : 0;
}

void rj_fused_sub_info(const rj_fused *f, int idx, rj_info *info)
{
    memset(info, 0, sizeof(*info));
    if (!f || idx < 0 || idx >= f->nops || !f->units[idx])
        return;
    rj_get_info(f->units[idx], info);
}

int rj_fused_sub_ndata(const rj_fused *f, int idx)
{
    if (!f || idx < 0 || idx >= f->nops)
        return -1;
    return f->kinds[idx] == RJ_FOP_ENCODE ? f->ndata : -1;
}

int rj_fused_sub_npar(const rj_fused *f, int idx)
{
    if (!f || idx < 0 || idx >= f->nops)
        return -1;
    return f->kinds[idx] == RJ_FOP_ENCODE ? f->npar : -1;
}

/* Phase 2 microarchitecture gate for interleaved CRC folding.
 * Conservative: returns 1 ONLY for the allowlisted (µarch, feature)
 * combinations where FINAL_SYNTHESIS.md shows the fold hides under DRAM
 * bandwidth. Everything else (including unknown/new µarchs) returns 0
 * for the scalar fallback path. */
#include "cpu.h"
int fused_crc_interleave_ok(const rj_cpu_info *cpu)
{
    if (!cpu)
        return 0;
#if defined(__x86_64__)
    /* REDUX: PCLMULQDQ (128-bit fold) + AVX512F required; SPR-only
     * (spec §3.6 -- ICL-SP/Zen4 are NOT carried over; re-opening them
     * needs per-µarch throughput data and a spec amendment). Unknown or
     * new µarchs fail closed. cpu->vpclmulqdq is deliberately NOT
     * required: the fold uses 128-bit PCLMULQDQ, not VPCLMULQDQ. */
    if (!cpu->pclmulqdq || !cpu->avx512f)
        return 0;
    if (rj_uarch_current() == RJ_UARCH_INTEL_SAPPHIRERAPIDS)
        return 1;
    return 0;
#elif defined(__aarch64__)
    /* AArch64: gate CLOSED for Phase 2 (see spec §2). The codebase has no
     * Apple-silicon µarch detection, making a conservative allowlist
     * unimplementable. Always scalar fallback. */
    (void)cpu;
    return 0;
#else
    /* RISC-V and others: never interleave in Phase 2 (conservative).
     * RISC-V throughput data does not exist; unknown ISAs default
     * to scalar. */
    (void)cpu;
    return 0;
#endif
}
