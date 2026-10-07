/* Batched multi-stripe Tier-1 driver (rj_fn_batched / rj_get_fn_batched).
 *
 * Design: hidden_files/design_batched_api.md (Carlo's §13 decisions are
 * the spec). v1 is Tier 1 only: a C loop calling the existing dispatched
 * rj_fn once per stripe. Zero new codegen. S is per call, the pointer
 * layout is flat row-major, the fast path is void/unchecked, there is
 * no S enforcement in code (guidance lives in the header + docs), and
 * batch-aware benchmarking is deferred (the driver inherits the
 * single-stripe winner via rj_get_fn).
 *
 * Internal chunking (design §6a): the driver processes S in chunks of
 * at most Sc_stage stripes,
 *     Sc_stage = max(1, 65536 / ((nbufs_in+nbufs_out) * sizeof(void *))),
 * so auxiliary memory stays bounded by a constant for any S. In Tier 1
 * the driver stages no pointers at all -- each stripe's slice is taken
 * directly from the caller's flat arrays -- so the chunk loop is
 * degenerate today; it exists as the seam a future Tier-2 deep kernel
 * would plug its LLC-sized data chunking into, without an API change.
 */
#include "api_batched.h"

#include <assert.h>
#include <pthread.h>
#include <stdlib.h>

struct rj_batch {
    rj_fn fn;          /* dispatched per-stripe entry point */
    size_t nbufs_in;   /* per-stripe in[] count (rj_fn contract) */
    size_t nbufs_out;  /* per-stripe out[] count (rj_fn contract) */
    int slot;          /* trampoline slot index, -1 when not installed */
};

/* Bound on one chunk's pointer staging (design §6a): 64 KiB. */
#define BATCH_STAGE_BYTES ((size_t)65536)

static size_t batch_chunk(const struct rj_batch *b)
{
    size_t per = (b->nbufs_in + b->nbufs_out) * sizeof(void *);
    size_t sc = per ? BATCH_STAGE_BYTES / per : 1;
    return sc ? sc : 1;
}

/* The driver. Stripes complete in index order 0..S-1; when the call
 * returns, all S stripes' outputs are written (modulo the NT fence
 * contract, which is the caller's -- once per call, not per stripe).
 * No allocation, no syscalls, no fallible operations inside: the loop
 * cannot fail partway. Debug builds assert the cheap contract checks
 * (design §7.5); release builds treat violations as UB, like rj_fn. */
static void batch_call(const struct rj_batch *b, const void *const *in,
                       void *const *out, size_t S, size_t len)
{
    assert(in != NULL);
    assert(out != NULL || b->nbufs_out == 0); /* npar==0 encode: out may
                                              * be NULL, fn is a no-op */
    assert(S >= 1);
    rj_fn fn = b->fn;
    size_t ni = b->nbufs_in, no = b->nbufs_out;
    size_t sc = batch_chunk(b);
    for (size_t c = 0; c < S; c += sc) {
        size_t n = S - c < sc ? S - c : sc;
        for (size_t k = 0; k < n; k++) {
            size_t s = c + k;
            /* The rj_fn in[]/out[] parameters are non-const pointer
             * arrays; the callee never writes through them, so the
             * const-dropping casts are safe (the caller retains the
             * const-qualified originals). nbufs_out == 0 (npar == 0
             * encode) passes NULL straight through -- the fn is a
             * no-op -- rather than forming &out[0] of a NULL base. */
            fn((const void **)&in[s * ni],
               no ? (void **)&out[s * no] : NULL, len);
        }
    }
}

/* ---------------- trampoline pool ---------------- */

#define RJ_BATCH_SLOTS 32

static struct rj_batch *g_bslots[RJ_BATCH_SLOTS];
static pthread_mutex_t g_bmu = PTHREAD_MUTEX_INITIALIZER;

#define DEF_BTRAMP(i)                                                      \
    static void rj_btramp_##i(const void *const *in, void *const *out,    \
                              size_t S, size_t len)                      \
    {                                                                    \
        /* No lock on the hot path: the slot is installed before the    \
         * entry point is published (rj_batch_fn returns) and released  \
         * only by rj_batch_destroy, which the rj_jit owner calls after \
         * its last batched call has returned -- the same lifetime rule \
         * as the rj_jit itself. The mutex below guards install/destroy \
         * only. */                                                      \
        struct rj_batch *b = g_bslots[i];                                \
        /* Debug-only liveness tripwire (zero cost under NDEBUG). A     \
         * slot REUSED by a newer rj_jit is NOT caught -- see the       \
         * lifetime contract in api_batched.h. */                        \
        assert(b != NULL && "rj_fn_batched used after rj_free");         \
        batch_call(b, in, out, S, len);                                  \
    }

DEF_BTRAMP(0) DEF_BTRAMP(1) DEF_BTRAMP(2) DEF_BTRAMP(3)
DEF_BTRAMP(4) DEF_BTRAMP(5) DEF_BTRAMP(6) DEF_BTRAMP(7)
DEF_BTRAMP(8) DEF_BTRAMP(9) DEF_BTRAMP(10) DEF_BTRAMP(11)
DEF_BTRAMP(12) DEF_BTRAMP(13) DEF_BTRAMP(14) DEF_BTRAMP(15)
DEF_BTRAMP(16) DEF_BTRAMP(17) DEF_BTRAMP(18) DEF_BTRAMP(19)
DEF_BTRAMP(20) DEF_BTRAMP(21) DEF_BTRAMP(22) DEF_BTRAMP(23)
DEF_BTRAMP(24) DEF_BTRAMP(25) DEF_BTRAMP(26) DEF_BTRAMP(27)
DEF_BTRAMP(28) DEF_BTRAMP(29) DEF_BTRAMP(30) DEF_BTRAMP(31)

#if RJ_BATCH_SLOTS != 32
#error "trampoline count must match RJ_BATCH_SLOTS"
#endif

static rj_fn_batched btramp_for(int slot)
{
    static rj_fn_batched tramps[RJ_BATCH_SLOTS] = {
        rj_btramp_0, rj_btramp_1, rj_btramp_2, rj_btramp_3,
        rj_btramp_4, rj_btramp_5, rj_btramp_6, rj_btramp_7,
        rj_btramp_8, rj_btramp_9, rj_btramp_10, rj_btramp_11,
        rj_btramp_12, rj_btramp_13, rj_btramp_14, rj_btramp_15,
        rj_btramp_16, rj_btramp_17, rj_btramp_18, rj_btramp_19,
        rj_btramp_20, rj_btramp_21, rj_btramp_22, rj_btramp_23,
        rj_btramp_24, rj_btramp_25, rj_btramp_26, rj_btramp_27,
        rj_btramp_28, rj_btramp_29, rj_btramp_30, rj_btramp_31,
    };
    return tramps[slot];
}

struct rj_batch *rj_batch_create(rj_fn fn, size_t nbufs_in, size_t nbufs_out)
{
    if (!fn)
        return NULL;
    struct rj_batch *b = calloc(1, sizeof(*b));
    if (!b)
        return NULL;
    b->fn = fn;
    b->nbufs_in = nbufs_in;
    b->nbufs_out = nbufs_out;
    b->slot = -1;
    return b;
}

rj_fn_batched rj_batch_fn(struct rj_batch *b)
{
    if (!b)
        return NULL;
    pthread_mutex_lock(&g_bmu);
    if (b->slot < 0) {
        for (int i = 0; i < RJ_BATCH_SLOTS; i++) {
            if (!g_bslots[i]) {
                g_bslots[i] = b;
                b->slot = i;
                break;
            }
        }
    }
    int slot = b->slot;
    pthread_mutex_unlock(&g_bmu);
    /* Slot exhaustion: NULL, like rj_dispatcher_create's clean failure.
     * The caller (rj_get_fn_batched) propagates it; the single-stripe
     * rj_fn stays usable. */
    return slot < 0 ? NULL : btramp_for(slot);
}

void rj_batch_destroy(struct rj_batch *b)
{
    if (!b)
        return;
    if (b->slot >= 0) {
        pthread_mutex_lock(&g_bmu);
        g_bslots[b->slot] = NULL;
        pthread_mutex_unlock(&g_bmu);
    }
    free(b);
}
