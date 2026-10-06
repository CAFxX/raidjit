# Batched multi-stripe API — design

**Item:** LATER #9, `FEATURES_CONSOLIDATED.md` §2 (duplicates brainstorm A7 —
single entry). Lane 2, queue item 7 in `EXECUTION_QUEUE.md` ("design only").

**Status:** DESIGN ONLY. No implementation. Implementation is Lane 5, queued
after this design is reviewed (see `EXECUTION_QUEUE.md` Lane 5).

**Scope of this document:** the API shape, the S-axis semantics, the
cache-footprint guard, error handling, threading, and interaction with the
existing API. All performance numbers below are **static estimates, not
measurements**; measuring is the implementation lane's job (with the
per-change perf guard).

---

## 1. Terminology

1. **Stripe** — one `(in[], out[], len)` unit as defined by the `rj_fn`
   contract in `include/raidjit.h`: `ndata` input buffers, `npar` output
   buffers (encode), or the recover / parity-update pointer lists.
2. **S (batch axis)** — the number of stripes processed by one batched call.
   S is a pure call-time multiplicity: every stripe in the batch shares one
   topology and one op (inherent — see §9).
3. **Chunk (Sc)** — a driver-internal subdivision of S. The caller never
   sees chunks; they exist only to bound staging memory and (for a future
   deep kernel) the resident working set.
4. **Tier 1 (driver)** — a C loop that invokes the already-compiled,
   length-dispatched `rj_fn` once per stripe. No new codegen.
5. **Tier 2 (deep kernel)** — a hypothetical future jitted kernel with the
   S-loop *inside* the generated code. Explicitly out of scope for the v1
   design (§10, Q4).

---

## 2. Motivation and non-goals

### Motivation

1. **Amortize per-call overhead.** Each `rj_fn` call pays prologue/epilogue,
   register saves, and length dispatch (notail / predtail / maskedtail twin
   selection). For small stripes this fixed cost is a real fraction of the
   call; one call for S stripes pays it once. (Static estimate, unmeasured.)
2. **Feed wide machines on small stripes.** A single short stripe is a tiny
   loop trip count with few concurrent streams; the core retires, the
   memory-level parallelism stays idle, and the I-cache is re-touched per
   call. Processing S stripes back-to-back in one call keeps the hot code
   resident and raises the effective stream count. This is the catalogue's
   "biggest system-level lever" claim — it needs measurement in Lane 5, but
   the mechanism is sound.
3. **Ergonomics.** Storage engines (the Ceph/DAOS-adjacent consumers in
   #44) naturally produce stripes in batches; forcing them through a
   one-stripe-at-a-time API pushes batching complexity into every caller.

### Non-goals

1. **No new codegen in v1.** Tier 1 reuses the compiled `rj_fn` unchanged;
   no emitter, planner, or twin changes.
2. **No threading inside the call.** One batched call runs on the calling
   thread (§8). Thread-parallel plans are PARKED item #11, a separate axis.
3. **No data reorganization.** The API accepts the caller's existing buffer
   layouts; it does not require interleaved or batch-major data (§10, C).
4. **No change to benchmark/selection.** The winning version is still chosen
   by the single-stripe benchmark protocol; batch-aware selection is
   deferred (§9, Q6).
5. **No plan cache.** There is none today (`docs/limitations.md`: "No code
   caching"); this design does not add one (§9).

---

## 3. API shape

Proposed additions to `include/raidjit.h` (names provisional — Q8):

```c
/* Batched fast path. Unchecked contract, exactly like rj_fn:
 * violations (S == 0, NULL pointers, misaligned buffers, aliasing across
 * stripes, len == 0) are undefined behavior. Processes stripes
 * 0 .. S-1 in order. void return: there is no mid-call failure mode
 * (§7). Inherits the NT-store fence contract of rj_fn (§9). */
typedef void (*rj_fn_batched)(const void *const *in, void *const *out,
                              size_t S, size_t len);

/* Retrieve the batched entry point for a compiled unit. The returned
 * function processes S stripes per call using the same jitted code
 * (and the same length dispatch) as rj_get_fn(j). */
rj_fn_batched rj_get_fn_batched(const rj_jit *j);

/* Checked wrapper: validates every parameter up front (see §7), then
 * delegates to rj_get_fn_batched(j). Returns 0 on success, -1 on
 * validation failure with a message in err (err may be NULL). Either all
 * S stripes are processed or none are. */
int rj_batch_call(const rj_jit *j,
                  const void *const *in, void *const *out,
                  size_t S, size_t len,
                  char *err, size_t errlen);
```

Parameter semantics:

1. `j` — compiled unit from `rj_compile(topo, op, opts, ...)`. The op
   (encode / recover with a fixed failed set / parity update with a fixed
   changed set) is therefore identical for every stripe in the batch.
2. `in` / `out` — flat pointer arrays, row-major over stripes: stripe `s`
   owns `in[s*nbufs_in .. (s+1)*nbufs_in)` and
   `out[s*nbufs_out .. (s+1)*nbufs_out)`, where `nbufs_in` / `nbufs_out`
   are exactly the per-stripe counts from the `rj_fn` contract for `j`'s op
   (encode: ndata/npar; recover: nsurv/nfailed; parity update:
   2*nchanged+npar / npar).
3. `S` — stripe count, `S >= 1` (checked wrapper enforces; fast path: UB).
4. `len` — **one** stripe length in bytes for the whole batch, same meaning
   as `rj_fn`'s `len` (arbitrary, need not be a vector multiple). Mixed
   lengths are handled by grouping equal-`len` stripes into separate calls
   (§10, D).

Why flat arrays rather than an array of per-stripe pointer arrays: one
contiguous block, no double indirection in the driver, trivial to build
from per-stripe arrays (`memcpy` of S pointer blocks). The driver
internally passes `&in[s*nbufs_in]` / `&out[s*nbufs_out]` straight to
`rj_fn`, so per-stripe semantics — including the in-place parity-update
aliasing exception, which applies *within* a stripe — are inherited
verbatim.

---

## 4. S-axis semantics

1. **Meaning of S.** S stripes, processed in stripe-index order 0..S-1.
   Stripes are logically independent: stripe `s` reads only its own
   `nbufs_in` inputs and writes only its own `nbufs_out` outputs.
2. **Layout in memory.** The *pointer arrays* are flat and contiguous
   (§3). The *data buffers* keep whatever layout the caller already uses —
   each of the `S*(nbufs_in+nbufs_out)` buffers is an independent
   allocation as far as the API is concerned. A caller that stores disk
   `d`'s S stripes contiguously (batch-major per disk) gets sequential
   access within a chunk for free, but the API does not require it.
3. **Alignment.** The 64-byte per-buffer contract from `rj_fn` applies to
   every buffer of every stripe, unchanged. In the checked wrapper,
   misalignment of any buffer is a validation failure; in the fast path it
   is UB (normally #GP on an aligned vector load/store, per the existing
   contract).
4. **Aliasing.** No two buffers may overlap or alias — within a stripe, or
   *across* stripes — except the documented in-place parity-update
   exception, which is per stripe (`out[s*np+j]` may alias
   `in[s*(2*nc+np)+2*nc+j]`). Cross-stripe aliasing in the fast path
   silently computes wrong results, same as today.
5. **Recover across a batch.** Because `j` is compiled for one fixed failed
   set, every stripe in a recover batch uses the same failed-disk list.
   This matches the physical reality (a dead disk is dead for all stripes)
   — no per-stripe op variation is expressible, by construction.
6. **S is per call, not per compile** (recommendation; Q1 asks Carlo to
   confirm). Rationale: `rj_jit` is already specialized on (topology, op,
   CPU); specializing it additionally on S would multiply compile cost and
   destroy `rj_jit*` reuse across batch sizes. The driver loop cost is
   O(S) pointer arithmetic — there is nothing for the JIT to specialize.

---

## 5. Execution model (Tier 1 driver)

```
rj_batch_call / rj_fn_batched(j, in, out, S, len):
    fn = <the same dispatched rj_fn that rj_get_fn(j) returns>
    for each chunk c of Sc stripes:          # §6
        for s in chunk:
            fn(&in[s*nbufs_in], &out[s*nbufs_out], len)
```

1. **Zero new codegen.** The driver calls the existing `rj_fn`, so length
   dispatch (scalar / notail / predtail / maskedtail twins), tail handling,
   NT-store twins, and prefetch twins all behave exactly as in the
   single-stripe path. No twin needs an S-aware variant.
2. **Ordering guarantee.** Stripes complete in index order; when the call
   returns, all S stripes' outputs are written (modulo the NT fence
   contract, §9).
3. **No allocation in the loop.** Chunk pointer staging is bounded (§6) and
   lives on the caller's stack frame or in one up-front allocation —
   never per stripe, never per chunk at unbounded size.
4. **What batching does NOT change:** total bytes moved are identical to S
   single calls (`S * (nbufs_in+nbufs_out) * len`). The win is strictly in
   per-call overhead amortization, I-cache residency of the hot loop, and
   higher concurrent-stream count for small stripes — all static
   hypotheses until Lane 5 measures them.

---

## 6. Cache-footprint guard

The guard has two layers. Both live inside the driver; the caller does no
chunking.

### 6a. Staging bound (always on)

The driver stages at most one chunk's worth of derived pointers. Bound the
staging to 64 KiB:

```
Sc_stage = max(1, 65536 / ((nbufs_in + nbufs_out) * sizeof(void *)))
```

E.g. a (10+4) encode: `Sc_stage = 65536 / (14*8) = 585` stripes per chunk.
This bounds auxiliary memory to a constant regardless of S; S itself is
unbounded by staging.

### 6b. Working-set chunking (streaming regime)

For large `len`, a batch's total traffic `S * (nbufs_in+nbufs_out) * len`
can exceed the LLC by orders of magnitude. In Tier 1 there is no
cross-stripe reuse to protect (each stripe streams once), so no LLC-sized
chunking is *required* for correctness or for the Tier-1 win — streaming
is streaming. The guard's real job here is defensive:

1. **Sanity bound on S.** The checked wrapper rejects `S > S_MAX`
   (proposed: `S_MAX = 1<<20`; Q5) with `ERANGE`-style error. Beyond ~10^4
   stripes there is no further amortization to gain (static estimate); a
   larger S is almost certainly a caller bug (e.g. bytes-vs-stripes
   confusion). The fast path documents S as unbounded-but-sane; the
   driver chunks internally so even absurd S cannot blow the stack.
2. **Forward compatibility with Tier 2.** If a deep kernel is ever built,
   its chunker would use `Sc_llc` such that
   `Sc_llc * (nbufs_in+nbufs_out) * len <= LLC/2`, keeping one chunk's
   per-disk data resident across the inner S-loop — that is where chunking
   buys actual cache reuse. Tier 1 reserves the chunking seam so Tier 2
   can plug in without API change.

### 6c. Workspace: none

Neither tier needs caller-provided workspace. Tier 1 needs only the
bounded pointer staging (§6a). Tier 2 (if built) keeps accumulators in
vector registers exactly like today's kernels. **The API therefore takes
no workspace parameter** — one less thing for callers to size wrong.

---

## 7. Error handling and partial failure

1. **Fast path (`rj_fn_batched`): void, unchecked.** Same philosophy as
   `rj_fn`: the contract is documented, violations are UB. There is no
   return channel by design — a `void` hot path stays a `void` hot path.
2. **Checked wrapper (`rj_batch_call`): all-or-nothing.** It validates, in
   order: `j != NULL`; `in != NULL && out != NULL`; `1 <= S <= S_MAX`;
   `len >= 1`; every one of the `S*(nbufs_in+nbufs_out)` pointers non-NULL
   and 64-byte aligned; no aliasing across the whole batch (pairwise range
   check, O((S*nbufs)^2) worst case — acceptable for a *checked* path; the
   fast path exists for callers who can't afford it) except the per-stripe
   in-place parity-update exception. Only after all checks pass does any
   stripe execute.
3. **Partial failure semantics: none.** Because validation is complete
   before the first byte is processed and the compute loop cannot fail
   (no allocation, no syscalls, no fallible operations inside), a call
   either processes all S stripes or zero. There is no "stripe 3 of 8
   failed" state to report — a deliberate simplification over
   per-stripe error codes.
4. **OOM:** the only allocation is the optional one-time staging block;
   if it fails, the wrapper returns -1/ENOMEM before processing anything.
5. **Misuse detection aid:** in debug builds (`NDEBUG` unset), the fast
   path may `assert()` the cheap checks (non-NULL, S>=1) — zero cost in
   release, matching the existing unchecked-for-speed stance.

---

## 8. Threading and reentrancy

1. **Same guarantees as `rj_fn`** (`docs/limitations.md`): generated code
   makes no calls and keeps no mutable shared state; the Tier-1 driver
   adds no shared state either (loop-local indices only). Concurrent
   batched calls on the same `rj_jit*` — from any threads — are safe.
2. **Not async-signal-safe**, same as `rj_fn` (no change).
3. **One call, one thread.** The driver does not spawn threads and does
   not shard S internally. A caller may shard a batch across threads
   itself (disjoint stripe subsets of one `rj_jit*`); safety follows from
   the no-cross-stripe-aliasing rule. This is explicitly *not* item #11
   (thread-parallel plans, PARKED): batching is about call amortization,
   not core scaling.
4. **Reentrancy:** the driver is reentrant (no statics); a signal handler
   must still not invoke it (see 2).

---

## 9. Interaction with existing API and the plan cache

1. **`rj_compile` / `rj_get_fn`: unchanged.** The batched entry point is a
   new accessor on the existing `rj_jit*`, exactly parallel to
   `rj_get_fn`. No new compile modes, no new `rj_options` fields in v1.
2. **There is no plan cache.** `docs/limitations.md` is explicit: each
   `rj_compile` re-emits and re-benchmarks from scratch; callers cache the
   `rj_jit*` themselves. The batch API follows the same rule: compile once
   for the (topo, op), reuse the `rj_jit*` across any number of batched
   calls with any S. This design adds no cache and needs none.
3. **NT-store fence contract carries over.** If the benchmark-selected
   winner is an NT twin, the caller must `SFENCE` (x86) / `DMB ISHST`
   (AArch64) after the batched call before any other agent reads the
   outputs — once per *call*, not per stripe. Note: `rj_info` currently
   exposes no machine-readable "NT won" flag (only the `desc` string); see
   Q7.
4. **Benchmark/selection interaction.** Winner selection uses the existing
   single-stripe protocol (`bench_bytes` per buffer). A batch-shaped
   benchmark (S stripes in flight during timing) might select differently
   — e.g. an NT or prefetch twin could win at batch scale while losing
   single-stripe. Deferred to measurement (Q6); v1 inherits the
   single-stripe winner.
5. **XOR-code topologies.** The expanded row model (`rj_topo_rdp` etc.)
   changes only what the per-stripe pointer lists *mean*, not their shape;
   batching composes with it unchanged.
6. **Compile cost.** Unchanged: one `rj_compile` per (topo, op) regardless
   of how many batch sizes the caller uses — a direct consequence of S
   being per call (§4.6).

---

## 10. Alternatives considered

- **A. Caller loops over `rj_fn` (status quo).** Zero API surface. Rejected
  as the *deliverable* — it leaves the amortization and ergonomics wins on
  the floor — but it remains available and is the behavior Tier 1
  internalizes.
- **B. Deep kernel with inner S-loop (Tier 2).** Highest theoretical
  ceiling: cross-stripe memory-level parallelism inside one jitted loop,
  one length-dispatch per call. Rejected for v1: needs new codegen in all
  three backends × every twin flavor (notail/predtail/maskedtail/NT/
  prefetch), plus either `npar*S` accumulator registers (register
  pressure) or re-streaming inputs per stripe (no traffic win). Revisit
  only if Tier-1 measurement shows the driver loop itself is the
  bottleneck (Q4).
- **C. Interleaved / batch-minor data layout as the API contract.** Would
  let one vector load span stripes, but forces every caller to reorganize
  its buffers; rejected — the API accepts existing layouts (§4.2).
- **D. Per-stripe `len[]`.** Rejected for v1: complicates length dispatch
  (the notail twin is only valid when `len % group_bytes == 0` — per
  stripe), chunking, and validation, for a use case served by grouping
  equal-length stripes into separate calls.
- **E. Per-stripe op (e.g. mixed failed sets in one recover call).**
  Rejected: `rj_jit` is compiled for one op; a mixed batch would need
  either re-dispatch per stripe (defeats the purpose) or multiple compiled
  units (the caller can already do that).

---

## 11. Open questions for Carlo

Decisions needed before Lane 5 implementation:

1. **Q1 — S fixed at compile time or per call?** Design recommends per
   call (§4.6): S-at-compile-time multiplies `rj_compile` cost and breaks
   `rj_jit*` reuse. Confirm, or state the case for compile-time S (e.g. a
   deep kernel that unrolls over a constant S).
2. **Q2 — Ship the checked wrapper in v1?** Design proposes both
   `rj_fn_batched` (void, unchecked) and `rj_batch_call` (int, validated).
   Alternative: void-only, like `rj_fn` today. The wrapper is thin but it
   is new API surface to maintain.
3. **Q3 — Canonical pointer layout.** Design proposes flat row-major
   `in[S*nbufs_in]` / `out[S*nbufs_out]`. Alternative: array of S
   per-stripe pointer arrays. Flat is recommended (single block, no double
   indirection).
4. **Q4 — Tier 1 only, or Tier 2 deep kernel in Lane 5?** Design scopes v1
   to the Tier-1 driver (no codegen) and defers Tier 2 until Tier-1 numbers
   show the driver loop is the bottleneck. Since #9 is billed as the
   "biggest system-level lever", confirm that driver-only is an acceptable
   first increment.
5. **Q5 — S_MAX value and policy.** Proposed: checked wrapper rejects
   `S > 2^20`; fast path documents S as sane-but-unbounded with internal
   chunking. Alternatives: lower cap, hard-fail vs clamp, or no cap at all.
6. **Q6 — Batch-aware benchmarking?** v1 inherits the single-stripe
   benchmark winner. Should Lane 5 also time candidates with S stripes in
   flight (could flip NT/prefetch twin selection)? Recommended: defer until
   Tier-1 data exists.
7. **Q7 — Expose NT-ness in `rj_info`?** The post-call fence contract
   (§9.3) currently has no machine-readable signal (only `desc`). A tiny
   `int nt_used` field would let batched (and single) callers fence
   conditionally. Adjacent scope — include or defer?
8. **Q8 — Naming.** `rj_fn_batched` / `rj_get_fn_batched` /
   `rj_batch_call`, or a different family name (e.g. `rj_encode_batch`)?
   Note `rj_encode`/`rj_recover` as named in the queue item don't exist as
   such today — the entry points are `rj_get_fn` + `rj_fn`.

---

## 12. References

- `include/raidjit.h` — public API; `rj_fn` contract, `rj_options`,
  topology/operation constructors.
- `src/api.h`, `src/jit.h` — compile pipeline; `version_t` twins.
- `docs/limitations.md` — thread-safety, no code caching, NT fence contract.
- `hidden_files/FEATURES_CONSOLIDATED.md` §2, item 9 — catalogue entry.
- `hidden_files/EXECUTION_QUEUE.md` — Lane 2 item 7 (this design), Lane 5
  (#9 implementation after design review).
