# Batched multi-stripe API (queue item #9, Tier 1)

One `rj_fn_batched` call processes S stripes that share a topology and an
op. v1 is a Tier-1 driver: a C loop over the already-compiled, dispatched
`rj_fn`, one call per stripe. Zero new codegen.

Design: [`hidden_files/design_batched_api.md`](../hidden_files/design_batched_api.md) (Carlo's §13 decisions are
the spec; this doc is the user-facing summary).

## 1. API

```c
typedef void (*rj_fn_batched)(const void *const *in, void *const *out,
                              size_t S, size_t len);
rj_fn_batched rj_get_fn_batched(const rj_jit *j);
```

`j` comes from `rj_compile(topo, op, opts, ...)` as usual. The op is fixed
at compile time, so every stripe in a batch shares it: one encode, one
recover with one fixed failed set, or one parity update with one fixed
changed set. (A dead disk is dead for all stripes, so one failed set per
recover batch matches the physical reality.)

## 2. Pointer layout

Flat, row-major over stripes. Stripe `s` owns

- `in[s*nbufs_in .. (s+1)*nbufs_in)`
- `out[s*nbufs_out .. (s+1)*nbufs_out)`

where `nbufs_in`/`nbufs_out` are exactly the per-stripe counts from the
`rj_fn` contract for `j`'s op:

| op | nbufs_in | nbufs_out |
|---|---|---|
| encode | ndata | npar |
| recover | nsurv | nfailed |
| parity update | 2*nchanged+npar | npar |

Only the *pointer arrays* are flat. The *data buffers* keep whatever
layout the caller already has -- each of the `S*(nbufs_in+nbufs_out)`
buffers is an independent buffer as far as the API is concerned. The
driver passes `&in[s*nbufs_in]` / `&out[s*nbufs_out]` straight to `rj_fn`,
so per-stripe semantics -- including the in-place parity-update aliasing
exception, which applies *within* a stripe -- are inherited verbatim.

## 3. Contract (unchecked, like `rj_fn`)

- `S >= 1`; `in`/`out` non-NULL (`out` may be NULL only when `npar == 0`,
  where the fn is a no-op).
- Every buffer 64-byte aligned.
- No two buffers overlap or alias, within or *across* stripes, except the
  per-stripe in-place parity-update exception.
- `len >= 1`, one stripe length in bytes for the whole batch (arbitrary,
  need not be a vector multiple). Stripes of different lengths go in
  separate calls.

Violations are undefined behavior. Debug builds (`NDEBUG` unset) assert
the cheap checks (non-NULL, `S >= 1`); release builds do not check.

## 4. Sizing guidance (documented, not enforced)

S is per call, not per compile: `rj_jit` stays specialized on
(topo, op, CPU) only, and one compiled unit serves any batch size. The
driver chunks S internally (at most 64 KiB of pointer staging per chunk),
so no S can blow auxiliary memory or the stack.

Guidance, not enforcement:

- Per-call overhead amortization saturates around ~10^4 stripes; larger
  batches buy little beyond that.
- `S > 2^20` is almost certainly a caller bug (e.g. bytes-vs-stripes
  confusion).

## 5. Execution and threading

Stripes complete in index order 0..S-1 on the calling thread; the driver
spawns no threads. When the call returns, all S outputs are written
(modulo the NT fence contract below).

Threading guarantees are exactly `rj_fn`'s ([`docs/limitations.md`](limitations.md)):
generated code keeps no mutable shared state and the driver adds none
(loop-local indices only), so concurrent batched calls on one `rj_jit*`
from any threads are safe. Not async-signal-safe, same as `rj_fn`.

## 6. NT-store fence contract

If the benchmark-selected winner is an NT twin, the caller must fence
once per *batched call* (not per stripe) before any other agent reads
the outputs: `SFENCE` on x86, `DMB ISHST` on AArch64 (RISC-V's
Zihintntl hint needs no fence). Fencing is entirely the caller's
responsibility; there is no machine-readable "NT won" flag (a deliberate
v1 decision).

## 7. Entry-point lifetime

The returned `rj_fn_batched` shares the `rj_jit`'s lifetime: never call
it after `rj_free` (the internal slot may be reused by a newer compile,
turning a loud fault into silent wrong output). Re-fetch via
`rj_get_fn_batched` after re-compiling. `rj_get_fn_batched` returns NULL
for a NULL `j`, or on (rare) internal trampoline-slot exhaustion -- the
single-stripe `rj_fn` from `rj_get_fn` stays usable.

## 8. What v1 does NOT do

- No new codegen (Tier 2 deep kernel is gated on Tier-1 measurements
  showing the driver loop is the bottleneck).
- No checked wrapper (`rj_batch_call`): deferred, not v1 scope.
- No batch-aware benchmarking: the driver inherits the single-stripe
  benchmark winner, including its length dispatch and tail/NT twins.
- No S enforcement in code; no per-stripe lengths; no per-stripe ops.
- No `nt_used` flag in `rj_info`.
