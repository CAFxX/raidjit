/* Length-specialized runtime dispatch. See dispatch.h and
 * docs/length_dispatch.md. */
#include "dispatch.h"

#include <assert.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

struct rj_dispatcher {
    rj_variant vars[RJ_DISPATCH_MAXV];
    int nvar;
    int slot; /* trampoline index, -1 when not installed */
};

void rj_variant_derive(rj_variant *v)
{
    size_t g = v->group;
    v->group_pow2 = g != 0 && (g & (g - 1)) == 0;
    v->group_mask = v->group_pow2 ? g - 1 : 0;
}

void rj_variant_sort(rj_variant *vars, int nvar)
{
    /* Insertion sort, fastest-first by gbps. nvar <= RJ_DISPATCH_MAXV. */
    for (int i = 1; i < nvar; i++) {
        rj_variant t = vars[i];
        int j = i;
        while (j > 0 && vars[j - 1].gbps < t.gbps) {
            vars[j] = vars[j - 1];
            j--;
        }
        vars[j] = t;
    }
}

/* ---------------- trampoline pool ---------------- */

static rj_dispatcher *g_slots[RJ_DISPATCH_SLOTS];
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;

/* The hot path: first admitting variant (fastest-first) wins. variants[0]
 * is the fastest measured; the variant set covers every len by the
 * create-time coverage check below, so the loop always terminates with
 * a call. */
static void dispatch_call(rj_dispatcher *d, const void *in[], void *out[],
                          size_t len)
{
    for (int i = 0; i < d->nvar; i++) {
        if (rj_variant_admits(&d->vars[i], len)) {
            d->vars[i].fn(in, out, len);
            return;
        }
    }
    /* Unreachable by construction: rj_dispatcher_create requires the
     * variant set to cover every len (a lone TAIL_SCALAR, or a
     * TAIL_MASKED tail-server plus a TAIL_NONE multiple-server), so the
     * loop above always returns. There is deliberately no
     * serve-the-last-variant fallback: a silent wrong-code fallback is
     * worse than the invariant violation being loud at the create
     * site. */
}

#define DEF_TRAMP(i)                                                     \
    static void rj_tramp_##i(const void *in[], void *out[], size_t len) \
    {                                                                  \
        /* No lock on the hot path: the slot is installed before the   \
         * entry point is published (rj_dispatcher_create returns) and \
         * released only by rj_dispatcher_destroy, which the owner    \
         * calls after its last dispatched call has returned -- the   \
         * same lifetime rule as the rj_jit itself. The mutex below   \
         * guards create/destroy only. */                              \
        rj_dispatcher *d = g_slots[i];                                 \
        /* Debug-only liveness tripwire (zero cost under NDEBUG): a    \
         * call through a dispatcher entry AFTER rj_free cleared the   \
         * slot but BEFORE it was reused aborts loudly instead of     \
         * faulting on a NULL deref or, worse, running. A slot REUSED \
         * by a newer dispatcher is NOT caught -- see the lifetime    \
         * contract in dispatch.h: never cache an rj_fn across        \
         * rj_free. */                                                \
        assert(d != NULL && "rj_fn used after rj_free");               \
        dispatch_call(d, in, out, len);                                \
    }

DEF_TRAMP(0) DEF_TRAMP(1) DEF_TRAMP(2) DEF_TRAMP(3)
DEF_TRAMP(4) DEF_TRAMP(5) DEF_TRAMP(6) DEF_TRAMP(7)
DEF_TRAMP(8) DEF_TRAMP(9) DEF_TRAMP(10) DEF_TRAMP(11)
DEF_TRAMP(12) DEF_TRAMP(13) DEF_TRAMP(14) DEF_TRAMP(15)
DEF_TRAMP(16) DEF_TRAMP(17) DEF_TRAMP(18) DEF_TRAMP(19)
DEF_TRAMP(20) DEF_TRAMP(21) DEF_TRAMP(22) DEF_TRAMP(23)
DEF_TRAMP(24) DEF_TRAMP(25) DEF_TRAMP(26) DEF_TRAMP(27)
DEF_TRAMP(28) DEF_TRAMP(29) DEF_TRAMP(30) DEF_TRAMP(31)

#if RJ_DISPATCH_SLOTS != 32
#error "trampoline count must match RJ_DISPATCH_SLOTS"
#endif

static rj_fn tramp_for(int slot)
{
    static rj_fn tramps[RJ_DISPATCH_SLOTS] = {
        rj_tramp_0, rj_tramp_1, rj_tramp_2, rj_tramp_3,
        rj_tramp_4, rj_tramp_5, rj_tramp_6, rj_tramp_7,
        rj_tramp_8, rj_tramp_9, rj_tramp_10, rj_tramp_11,
        rj_tramp_12, rj_tramp_13, rj_tramp_14, rj_tramp_15,
        rj_tramp_16, rj_tramp_17, rj_tramp_18, rj_tramp_19,
        rj_tramp_20, rj_tramp_21, rj_tramp_22, rj_tramp_23,
        rj_tramp_24, rj_tramp_25, rj_tramp_26, rj_tramp_27,
        rj_tramp_28, rj_tramp_29, rj_tramp_30, rj_tramp_31,
    };
    return tramps[slot];
}

rj_dispatcher *rj_dispatcher_create(const rj_variant *vars, int nvar)
{
    if (!vars || nvar < 1 || nvar > RJ_DISPATCH_MAXV)
        return NULL;
    /* Coverage: the variant set must admit every len, otherwise
     * dispatch_call's loop could fall through. A lone TAIL_SCALAR covers
     * everything (legacy); otherwise the set needs a tail-server
     * (TAIL_MASKED, tails only) plus a multiple-server (TAIL_NONE).
     * api.c always installs the general variant, so the first case is
     * the norm; the second keeps a {masked, notail} set valid. Callers
     * must not pass TAIL_SCALAR_MULT: only create() derives it. (The
     * check is exact when all twins share one group, as api.c builds
     * them; mixed groups are conservative -- a set that passes may still
     * miss a length if groups differ, so keep twins same-group.) */
    int has_scalar = 0, has_masked = 0, has_notail = 0;
    for (int i = 0; i < nvar; i++) {
        if (!vars[i].fn)
            return NULL;
        switch (vars[i].tail) {
        case TAIL_SCALAR:
            has_scalar = 1;
            break;
        case TAIL_MASKED:
            has_masked = 1;
            break;
        case TAIL_NONE:
            has_notail = 1;
            break;
        default:
            return NULL;
        }
    }
    if (!has_scalar && !(has_masked && has_notail))
        return NULL;

    rj_dispatcher *d = calloc(1, sizeof(*d));
    if (!d)
        return NULL;
    memcpy(d->vars, vars, sizeof(rj_variant) * (size_t)nvar);
    d->nvar = nvar;
    /* Length-aware tail-twin routing (docs/length_dispatch.md §12): when
     * a masked/predicated tail twin is installed it owns tail lengths,
     * so the scalar general variant defers to it (admits multiples
     * only). Without a masked twin the scalar keeps admit-everything.
     * The mechanism is ISA-agnostic: any TAIL_MASKED twin (x86-64 k1,
     * SVE whilelt) triggers it. */
    if (has_masked)
        for (int i = 0; i < nvar; i++)
            if (d->vars[i].tail == TAIL_SCALAR)
                d->vars[i].tail = TAIL_SCALAR_MULT;
    /* Fastest-first, always: the hot path relies on it, so enforce it
     * here rather than trusting every caller to sort. Idempotent. */
    rj_variant_sort(d->vars, d->nvar);
    d->slot = -1;

    pthread_mutex_lock(&g_mu);
    for (int i = 0; i < RJ_DISPATCH_SLOTS; i++) {
        if (!g_slots[i]) {
            g_slots[i] = d;
            d->slot = i;
            break;
        }
    }
    pthread_mutex_unlock(&g_mu);
    if (d->slot < 0) {
        free(d);
        return NULL; /* pool exhausted: caller falls back to general-only */
    }
    return d;
}

rj_fn rj_dispatcher_fn(const rj_dispatcher *d)
{
    if (!d || d->slot < 0)
        return NULL;
    return tramp_for(d->slot);
}

int rj_dispatcher_nvar(const rj_dispatcher *d)
{
    return d ? d->nvar : 0;
}

int rj_dispatcher_pick(const rj_dispatcher *d, size_t len)
{
    if (!d)
        return -1;
    for (int i = 0; i < d->nvar; i++)
        if (rj_variant_admits(&d->vars[i], len))
            return i;
    return d->nvar - 1;
}

void rj_dispatcher_destroy(rj_dispatcher *d)
{
    if (!d)
        return;
    if (d->slot >= 0) {
        pthread_mutex_lock(&g_mu);
        g_slots[d->slot] = NULL;
        pthread_mutex_unlock(&g_mu);
    }
    free(d);
}
