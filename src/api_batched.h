#pragma once
#include "../include/raidjit.h"

#include <stddef.h>

/* Batched Tier-1 driver state (src/api_batched.c). Internal; not part of
 * the public API.
 *
 * The driver is a plain C loop over the already-compiled, dispatched
 * rj_fn: zero new codegen (design hidden_files/design_batched_api.md
 * §5). Because rj_fn_batched carries no context pointer, the entry point
 * handed out is one of a pool of static trampolines bound to slots --
 * the same portable pattern src/dispatch.c uses for the length
 * dispatcher (see dispatch.h "trampoline pool").
 *
 * LIFETIME CONTRACT: the rj_fn_batched returned by rj_get_fn_batched
 * shares its rj_jit's lifetime. rj_free releases the slot; a call
 * through the entry point after rj_free is dangling (the slot may be
 * reused by a newer rj_jit, turning a loud fault into silent wrong
 * output). Always re-fetch via rj_get_fn_batched after re-compiling.
 * Debug builds trip an assert on use-after-free-before-reuse; slot
 * reuse itself is NOT detectable -- the contract is the only
 * protection. */
struct rj_batch;

/* Create the per-rj_jit batch driver state. fn is the dispatched rj_fn
 * (what rj_get_fn(j) returns); ni/no are the per-stripe in[]/out[]
 * counts from the rj_fn contract for j's op (encode: ndata/npar;
 * recover: nsurv/nfailed; parity update: 2*nchanged+npar/npar). The
 * state is installed in a trampoline slot lazily by rj_batch_fn.
 * Returns NULL on OOM. */
struct rj_batch *rj_batch_create(rj_fn fn, size_t nbufs_in, size_t nbufs_out);

/* The rj_fn_batched entry point for this state: installs it in a static
 * trampoline slot on first call (idempotent afterwards) and returns the
 * trampoline. NULL on slot exhaustion (the caller sees NULL from
 * rj_get_fn_batched). Thread-safe: concurrent first calls serialize on
 * the pool mutex. */
rj_fn_batched rj_batch_fn(struct rj_batch *b);

/* Release the trampoline slot (if installed) and free the state. NULL
 * is a no-op. */
void rj_batch_destroy(struct rj_batch *b);
