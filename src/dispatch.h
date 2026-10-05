/* Length-specialized runtime dispatch (see docs/length_dispatch.md).
 *
 * A dispatcher is a portable C function with the rj_fn signature that
 * routes each call to the fastest compiled variant admitting that call's
 * length. Variants are sorted fastest-first (by measured GB/s from the
 * compile-time shootout); the first variant whose tail policy admits len
 * wins.
 *
 * Length-aware tail-twin routing (track B1x86, generalized for B1sve;
 * docs/length_dispatch.md §12): when a masked/predicated tail twin
 * (TAIL_MASKED) is installed, it OWNS tail lengths (len % group != 0)
 * and the scalar general variant defers to it (its policy is derived to
 * TAIL_SCALAR_MULT at create time, admitting multiples only). Tail-heavy
 * traffic then routes deterministically to the twin that won its tail
 * gate, instead of coin-flipping on GB/s noise. Without an installed
 * masked twin, TAIL_SCALAR admits every length (legacy behavior).
 *
 * rj_fn carries no context pointer, so one static function cannot close
 * over a per-rj_jit variant table. The portable answer is a small pool of
 * static trampolines (macro-generated below), each bound to a slot holding
 * one dispatcher's table. Slot exhaustion (more than RJ_DISPATCH_SLOTS
 * live dispatching rj_jit objects) fails rj_dispatcher_create cleanly and
 * the caller falls back to the general-only version -- never wrong code.
 *
 * Tail policies beyond TAIL_SCALAR/TAIL_NONE plug into rj_tail_policy +
 * rj_variant_admits (TAIL_MASKED arrived with track B1sve); the dispatcher
 * orders them purely by measured speed, no policy-name special cases.
 */
#pragma once
#include "../include/raidjit.h"

#include <stddef.h>

/* Tail policy of one compiled variant: what lengths its code can serve.
 * A variant's policy is a property of its codegen (a TAIL_NONE version
 * physically has no tail code), but whether the dispatcher *routes* to it
 * is decided by measurement (docs/length_dispatch.md §5) and, for tail
 * twins, by the installed variant set (§12: length-aware tail-twin
 * routing). */
typedef enum {
    TAIL_SCALAR = 0, /* scalar byte tail (x86-64/AArch64) or predicated
                      * strip-mining tail (RISC-V): admits every len --
                      * UNLESS a TAIL_MASKED twin shares the dispatcher,
                      * in which case rj_dispatcher_create derives this to
                      * TAIL_SCALAR_MULT (below). */
    TAIL_NONE = 1,   /* no tail code at all: admits only len % group == 0 */
    TAIL_MASKED = 2, /* fully predicated/masked vector tail (SVE whilelt
                      * twin, track B1sve; AVX-512 k1 twin, track B1x86):
                      * admits ONLY tail lengths (len % group != 0). A
                      * masked twin is installed only after winning its
                      * tail gate, so it deterministically serves the
                      * lengths it was measured on. A distinct policy (not
                      * a reuse of TAIL_SCALAR) so routing/diagnostics can
                      * tell the codegen apart. */
    TAIL_SCALAR_MULT = 3, /* DERIVED -- never set by callers. A TAIL_SCALAR
                           * variant in a dispatcher that also installs a
                           * TAIL_MASKED twin: the scalar tail still exists
                           * in the code, but tail lengths defer to the
                           * masked twin, so this admits only
                           * len % group == 0. rj_dispatcher_create
                           * performs the rewrite; without an installed
                           * masked twin TAIL_SCALAR keeps its
                           * admit-everything (legacy) behavior. */
} rj_tail_policy;

/* One length-specialized variant of a compiled plan. */
typedef struct {
    rj_fn fn;          /* variant entry point, rj_fn ABI */
    rj_tail_policy tail;
    size_t group;      /* full-iteration bytes; TAIL_NONE admission key */
    int group_pow2;    /* precomputed at build time: is group a power of 2 */
    size_t group_mask; /* group - 1, valid when group_pow2 */
    double gbps;       /* measured under the compile-time protocol */
    char desc[192];    /* diagnostics */
} rj_variant;

/* Length-admission predicate: the per-variant tail-policy hook. 1 if this
 * variant may be called with len, 0 otherwise.
 *
 * Length-aware tail-twin routing (docs/length_dispatch.md §12): a
 * TAIL_MASKED twin admits only tail lengths (len % group != 0); a
 * TAIL_SCALAR_MULT variant (derived at create time when a masked twin is
 * installed) admits only multiples. Without an installed masked twin,
 * TAIL_SCALAR admits everything (legacy). */
static inline int rj_variant_admits(const rj_variant *v, size_t len)
{
    if (v->group == 0) {
        /* Degenerate group: the multiple/tail tests are meaningless.
         * TAIL_SCALAR still admits everything (its scalar tail needs no
         * group); every other policy admits nothing. */
        return v->tail == TAIL_SCALAR;
    }
    switch (v->tail) {
    case TAIL_SCALAR:
        return 1;
    case TAIL_SCALAR_MULT:
    case TAIL_NONE:
    case TAIL_MASKED: {
        /* The multiple test needs a hardware DIV for non-power-of-2
         * groups (25-90 cycles). Compute it lazily: TAIL_SCALAR (the
         * common general variant) returns above without paying for a
         * result it discards. */
        int mult = v->group_pow2 ? (len & v->group_mask) == 0
                                 : len % v->group == 0;
        return v->tail == TAIL_MASKED ? !mult : mult;
    }
    }
    return 0; /* unknown future policy: conservative */
}

typedef struct rj_dispatcher rj_dispatcher;

/* Fill in the derived admission fields of a variant (group_pow2 /
 * group_mask) from group. Called by the api.c wiring at build time. */
void rj_variant_derive(rj_variant *v);

/* Sort variants fastest-first by gbps (in place, nvar >= 1). The general
 * variant must be present (it is the fallback); anything else is ordered
 * purely by measured speed. */
void rj_variant_sort(rj_variant *vars, int nvar);

/* Maximum live dispatchers. 32 static trampolines; exhaustion is a clean
 * NULL (caller falls back to general-only). */
#define RJ_DISPATCH_SLOTS 32

/* Create a dispatcher over nvar variants (1 <= nvar <= RJ_DISPATCH_MAXV).
 * Copies the variant array. Returns NULL on slot exhaustion or bad args.
 *
 * LIFETIME CONTRACT (footgun warning): the rj_fn returned by
 * rj_dispatcher_fn shares the rj_jit's lifetime -- rj_free destroys the
 * dispatcher and releases its slot, after which the entry point is DANGLING.
 * Never cache a dispatched rj_fn across rj_free: the slot may be reused by
 * a newer dispatcher, turning a loud segfault into SILENT WRONG OUTPUT
 * (the stale entry would route into another plan's variants). Always
 * re-fetch via rj_get_fn after re-compiling. Debug builds trip an assert
 * on use-after-free-before-reuse; slot reuse itself is NOT detectable --
 * the contract is the only protection. */
#define RJ_DISPATCH_MAXV 4
rj_dispatcher *rj_dispatcher_create(const rj_variant *vars, int nvar);

/* The rj_fn entry point to hand out (== one of the static trampolines). */
rj_fn rj_dispatcher_fn(const rj_dispatcher *d);

/* Number of installed variants (for tests/diagnostics). */
int rj_dispatcher_nvar(const rj_dispatcher *d);

/* White-box test hook: index (fastest-first order) of the variant that
 * would serve len. */
int rj_dispatcher_pick(const rj_dispatcher *d, size_t len);

void rj_dispatcher_destroy(rj_dispatcher *d);
