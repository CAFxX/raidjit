# CRC Phase 2 REDUX — interleaved CRC-fold spec (128-bit per-stream fold)

Status: SPEC. No implementation. This document supersedes the Phase 2
design in `hidden_files/crc_impl_SPEC.md` (§R11/R11a and every
"4-lane / every-4th-chunk / Barrett-per-lane + shift-XOR" passage) and
the stale in-code comments that describe it (`src/fused.h:37`
"nstreams*64 bytes, one zmm per stream", `src/fused.c`
`fold_state[... * 64]` + 4-lane init, `src/x86_64/emit.c`
`em_vpclmulqdq`'s "crc_fold twin" note as far as the fold is concerned).

Notation: S = 128-bit fold state (little-endian integer, bits
[127:0]); C = 16-byte message chunk; K1 = 0xae689191, K2 = 0xccaa009e;
QREF = 0x1DB710641 (reflected CRC-32 polynomial). All CRCs are
reflected (zlib) CRC-32.

---

## 1. Why the v1 design was wrong

### 1.1 The specified construction

v1 (`crc_impl_SPEC.md` R11) specified, per stream: one zmm = 4×
128-bit lanes; lane k folds every 4th 16B chunk (chunks k, k+4, k+8,
…); at epilogue, "per-stripe lane reduction": Barrett-reduce each
128-bit lane to 32 bits, then shift-XOR the four 32-bit values into one
CRC ("Barrett-per-lane + shift-XOR", per the original §4.2).

### 1.2 The counterexample (fatal)

A worker exhibited two 8-chunk (128-byte) messages whose four 128-bit
lane states are BIT-IDENTICAL, but whose CRC32s differ:

- message A → CRC32 `0x51eb4347`
- message B → CRC32 `0x24abc5d9`

identical lane states, different CRCs. Therefore NO finalize function
of `(fold_state, nstreams, total_len)` — however clever — can produce
the correct CRC32 for messages longer than 4 chunks. The v1 finalize
was not merely buggy; the lane state does not determine the answer.
This is recorded in `src/x86_64/crc_fold.c`'s header comment.

Why it fails, precisely: the lane transition is affine over GF(2), so
each lane's final state is an affine function of its chunk
subsequence. For an M-chunk message the map is
L: {0,1}^{128M} → {0,1}^{512} with a kernel of dimension ≥ 128M−512;
at M=8 the worker found an explicit collision pair with different
CRCs. Concretely, the two chunks inside one lane (e.g. chunks k and
k+4) need DIFFERENT positional shifts when the message is
reassembled, and a single per-lane shift — or any function of the four
lane states alone — cannot supply both. Reducing each lane to 32 bits
first (Barrett-per-lane) discards the intra-lane positional
information before combination; the specified shift-XOR of the reduced
values is a different operation from a correct combination and is
wrong.

### 1.3 The zlib-ng misreading

v1 cited the lane assignment as "(zlib-ng structure)". The
resemblance is superficial: zlib-ng does deal consecutive 16B chunks
across lanes, but its correctness rests on its SPECIFIC combine step —
folding the UNREDUCED 128-bit lane states into each other with
distance-correct sequencing BEFORE any reduction. v1 replaced that
with per-lane Barrett reduction followed by shift-XOR of four 32-bit
values. A 32-bit reduced lane value is "the CRC of the subsequence as
if it started at position 0"; shift-XORing four such values cannot
reconstruct the interleaved message's CRC, because the subsequences'
bytes sit at different message positions and one shift per lane cannot
encode that. The citation was wrong; the structure was not zlib-ng's.

### 1.4 Consequence for REDUX

REDUX eliminates cross-lane combination entirely. Each stream keeps
ONE 128-bit fold state advanced by the single-stream fold whose
transition was designed (K1 = x^{−128} mod QREF, K2 = x^{−64} mod
QREF) so that the 128-bit state remains a sufficient statistic for the
CRC — verified bit-exact vs zlib (see `src/x86_64/crc_fold.c` header).
There is no combination step, so the §1.2 impossibility does not
apply.

---

## 2. Corrected formulation: one 128-bit fold per stream

### 2.1 The fold (exact math)

Per stream, the fold state is a single 128-bit value S held in one xmm
register for the whole kernel invocation.

- **Init:** S ← `0x000000000000000000000000FFFFFFFF` (low 32 bits =
  0xFFFFFFFF — reflected init; upper 96 bits zero).
- **Per 16-byte chunk C** (chunk bytes little-endian into a 128-bit
  integer):
  ```
  W  = S XOR C
  S' = CLMUL(W[63:0], K1) XOR CLMUL(W[127:64], K2)
  ```
  with K1 = 0xae689191, K2 = 0xccaa009e.
- **Finalize:** CRC32(stream) = red128(S) XOR 0xFFFFFFFF, where red128
  reduces the 128-bit value mod QREF = 0x1DB710641 to < 2^32. This is
  EXACTLY what the existing `rj_crc_fold_finalize_x86()` in
  `src/x86_64/crc_fold.c` implements — reused unchanged. Its
  `total_len` parameter stays unused (the single-stream fold state
  fully determines the CRC; no length adjustment).

### 2.2 Where the fold executes

The encode kernel already holds each stream's 64-byte vector in a zmm
(input vectors after load; parity vectors after computation, before
store). The fold consumes that vector in place — zero extra memory
traffic. Per 64B vector, per stream, the emitter produces four 16B
folds:

```
; xs = stream's 128-bit fold state (xmm, persistent across the tile loop)
; zdata = the stream's 64B vector (zmm, already live)
; xc, xa, xb = temp xmms (shared across all streams/chunks)
; xk1, xk2 = constant xmms, both 64-bit lanes = K1 / K2 (hoisted, §4)
VEXTRACTI32X4 xc, zdata, 0      ; chunk 0 into xc
VPXOR         xc, xc, xs        ; W = C XOR S
PCLMULQDQ     xa, xc, xk1, 0x00 ; A = CLMUL(W[63:0], K1)
PCLMULQDQ     xb, xc, xk2, 0x11 ; B = CLMUL(W[127:64], K2)
VPXOR         xs, xa, xb        ; S' = A XOR B
... repeat with imm 1, 2, 3 for chunks 1..3 ...
```

5 uops per 16B chunk → 20 uops per 64B per stream, of which
8 × PCLMULQDQ, 4 × VEXTRACTI32X4, 8 × VPXOR. Constants: xk1/xk2 are
loaded ONCE at kernel entry (each holding the 64-bit constant in both
lanes, so imm=0x00 and imm=0x11 both read the right half).

**Fold points (emitter requirement):** for input streams, fold
immediately after the 64B vector load; for parity streams, fold
immediately after the 64B parity vector is computed and BEFORE it is
stored or the zmm is reused. The fold must observe the exact bytes the
encode consumes/produces — folding a vector that is later modified, or
missing a vector, is a correctness bug, caught by the §8 differential
tests.

### 2.3 Rejected alternative: VPCLMULQDQ with lanes = streams

Considered and REJECTED: packing four streams' 16B chunks into one
zmm and folding with 2 × VPCLMULQDQ (the "lanes = streams" execution
model from the pre-correction draft). Data-flow analysis:

- Per 16B position across 4 streams: 4 × VEXTRACTI32X4 + 3 ×
  VINSERTI32X4 (pack) + 1 × VPTERNLOG (XOR states) + 2 × VPCLMULQDQ +
  1 × XOR products ≈ 10 uops per 16B position → **≈40 uops per 64B
  for 4 streams**, dominated by packing overhead.
- 128-bit per-stream fold: **8 PCLMULQDQs per 64B per stream** (= 32
  for 4 streams) + 4 extracts + 8 XORs — no packing, no cross-stream
  choreography in the hot loop.

The packing overhead is prohibitive: it burns shuffle-port uops on the
same ports the encode needs, to save nothing (the per-chunk multiply
count is identical — 2 per 16B per stream either way). The 128-bit
fold is also UNIFORM in N (see §5): no 4-stream grouping, no masking,
no remainder logic. The existing `em_vpclmulqdq` emitter stays in the
tree for other consumers; REDUX does not use it.

**Superseded:** the pre-correction "pack 4 streams' states into one
zmm for storage" idea is dropped too — states live in N independent
xmms (§4). There is no packed representation anywhere in REDUX.

### 2.4 New emitter required

No 128-bit PCLMULQDQ emitter exists in `src/x86_64/emit.c` today
(only the 512-bit `em_vpclmulqdq`). REDUX needs:

```
void em_pclmulqdq(em_t *e, int vd, int vs1, int vs2, unsigned imm);
; VEX.128.66.0F3A 44 /r ib : PCLMULQDQ xmm1, xmm2, xmm3/m128, imm8
; xmm1(dst) = CLMUL(xmm2, xmm3) selected by imm8 (0x00/0x11 used)
; Non-destructive 3-operand: vd=xmm1(ModRM.reg), vs1=xmm2(vvvv), vs2=xmm3(r/m)
```

Derived per the AGENTS.md emitter rules: clear ALL operand fields
when deriving the base (rd/rn/rm/imm — the 2026-10-01 lesson), GAS
cross-check across randomized operands (even/odd, low/high regs)
BEFORE use, exact-equality uop-model test. `em_vpternlog`,
`em_vextracti32x4` already exist and are reused.

---

## 3. Kernel/driver contract

### 3.1 Call shape

The 4-argument kernel type already exists and is kept:

```c
/* src/fused.h:37 */
typedef void (*rj_fn_crc_fold)(const void *in[], void *out[], size_t len,
                               void *fold_state);
```

- `in`/`out`: the target op's buffer-pointer arrays (same as `rj_fn`).
- `len`: bytes per stream for THIS call (the tile length; §3.4).
- `fold_state`: driver-owned, PER-CALL (stack array, never per-handle
  → concurrent `rj_fused_call` is safe), `nstreams * 16` bytes,
  **at least 16-byte aligned** (driver keeps its `alignas(64)` stack
  array; the contract requires 16).

**CHANGED vs the tree:** `src/fused.h:37` and `src/fused.c` currently
say/allocate `nstreams * 64` bytes ("one zmm per stream") with a 4-lane
init. That is the v1 layout and is STALE — the driver must be updated
to `nstreams * 16` with the §3.2 init. (Spec only; the code change is
not this document.)

### 3.2 State layout and init convention

- `fold_state`: `nstreams` consecutive 16-byte little-endian 128-bit
  states, stream s at bytes `[s*16, s*16+16)`. Stream order = target
  op's buffer order: inputs 0..nin−1, then parities.
- **Init (driver, per `rj_fused_call`, before the tile loop):**
  every stream's 16 bytes ← `FF FF FF FF 00 ×12`
  (128-bit integer 0xFFFFFFFF = reflected init in the low 32 bits).
- **Kernel entry:** VMOVDQA-load the N states into the N state xmms.
  **Kernel exit:** store the N states back. Between entry and exit the
  states live ONLY in xmms (no memory traffic per tile).

### 3.3 Finalize convention

After the tile loop + tail (§6), the driver calls the EXISTING,
unchanged:

```c
void rj_crc_fold_finalize_x86(const void *fold_state, int nstreams,
                              size_t total_len, uint32_t *crc);
```

`total_len` remains unused (documented in the code). Output: standard
zlib CRC-32 per stream.

### 3.4 Per-call byte coverage (the kernel's tail rule)

The kernel folds, per stream per call, exactly the first
`floor(tl/16)*16` bytes of its (already tile-offset) pointers, where
`tl` is the `len` argument of that call. Bytes beyond that in the call
range are IGNORED by the fold (the encode still processes them
normally). Because tiles are contiguous and each call folds a
16-multiple prefix, the union of folded ranges over the whole
`rj_fused_call` is exactly `[0, floor(len/16)*16)` — the unfolded
suffix is precisely the final `len % 16` bytes, handled by §6. This
holds for ANY tile size; no `tile % 16 == 0` constraint is needed.

`len = 0` is unreachable (`fused_driver_call` asserts `len >= 1`;
carried over from v1 R13).

### 3.5 Scope

REDUX covers ENCODE targets only (v1 R13 scope, retained). A CRC op
targeting PARITY_UPDATE, or any path where the gate (§3.6) is closed,
uses the existing scalar path (`rj_crc32_fast` over the hot tile
buffers) — byte-identical behavior to today.

### 3.6 µarch gate (conservative)

`fused_crc_interleave_ok()` (and the planner's `rj_crc_fold_viable`
twin-gating) returns 1 ONLY when ALL hold:

- `defined(__x86_64__)`,
- `rj_uarch_current() == RJ_UARCH_INTEL_SAPPHIRERAPIDS` (**SPR-only**;
  v1's ICL-SP/Zen4 are NOT carried over — re-opening them needs
  per-µarch throughput data, cf. §7),
- `cpu->pclmulqdq` (VEX 128-bit carryless multiply),
- `cpu->avx512f` (fold twins are AVX-512 candidates; VEXTRACTI32X4).

CHANGED vs v1/tree: `cpu->vpclmulqdq` is NO LONGER required (the fold
uses 128-bit PCLMULQDQ, not VPCLMULQDQ). AArch64 and RISC-V gates stay
CLOSED (scalar fallback), as in the tree. Unknown/new µarchs → 0
(fail closed). The gate is pure in (uarch tag, feature flags);
cpuinfo parsing stays in `rj_cpu_uarch()` exactly once (v1 R10
Finding 7, retained).

---

## 4. Register allocation

Per `crc_fold=1` kernel, the fold needs N + 5 128-bit lanes:

| Use | Count | Lifetime |
|---|---|---|
| Fold states (one xmm per stream) | N | whole kernel invocation (loaded at entry, stored at exit) |
| Temps: chunk/W (`xc`), product A (`xa`), product B (`xb`) | 3 | per 16B chunk; shared across all streams/chunks |
| Constants: K1-broadcast (`xk1`), K2-broadcast (`xk2`) | 2 | whole kernel; loaded once at entry |

**Disjointness (hard requirement):** the N+5 lanes must never alias a
zmm whose upper lanes hold live encode data — EVEX 128-bit
destinations ZERO bits [511:128], which would corrupt encode state.
Mechanism (required, deterministic): the planner reserves the TOP N+5
zmm registers (`zmm_{32-(N+5)} … zmm31`) as encode-free in
`crc_fold=1` kernels; the fold emitter addresses their low 128-bit
lanes. `plan_need_regs` accounts the fold pool EXACTLY
(the v1-R12 "formulas exact for the new shape" rule, retained):

```
fold_regs = N + 5            # 128-bit lanes
enumerated iff encode_zmm_need + (N + 5) <= 32
```

If the encode's zmm demand exceeds `32 − (N+5)`, the `crc_fold=1`
twin is NOT enumerated — fail closed to the scalar CRC path (the
existing `rj_compile_want_crc` → NULL → plain-compile fallback in
`src/fused.c`). In particular this caps foldable topologies at
**N ≤ 27** streams; wider topologies always take the scalar path.
The planner's `twin_dominates` pins the `crc_fold` axis IN THE SAME
CHANGE that introduces the new twin shapes (AGENTS.md lesson,
retained from v1 R9).

NT-store note (v1 R9 Finding 5, adapted): fold states live in xmms
across the tile loop with ONE 16B load + ONE 16B store per stream per
`rj_fused_call` — entry/exit traffic is negligible; no NT consideration
applies. Parity DATA stores keep their existing NT policy.

---

## 5. Remainder handling (N % 4)

**None required — eliminated by the design.** The fold is per-stream
and 128-bit: stream s uses state xmm s, with no grouping, no
cross-stream packing, no masked lanes. N = 1 and N = 27 fold with the
identical per-stream recipe; the driver loop over `nstreams` 16-byte
states is uniform. The ONLY N-dependent condition is the §4 register
budget (fail-closed scalar fallback beyond N = 27). This is a
deliberate simplification vs both v1 (4-lane combination) and the
rejected VPCLMULQDQ packing (§2.3), which would each have needed
masking/remainder paths.

---

## 6. Tail handling

The kernel NEVER folds partial 16B chunks (§3.4). The final
`tail_len = len % 16` bytes of each stream are folded by the DRIVER,
once per stream per `rj_fused_call`, after the tile loop and before
finalize, via a small pure-C helper (new; lives next to
`rj_crc_fold_finalize_x86`):

```c
/* Advance one 128-bit fold state by tail_len (0..15) bytes at p.
 * s: 16-byte little-endian state (updated in place). */
void rj_crc_fold_tail(uint8_t s[16], const uint8_t *p, size_t tail_len);
```

Semantics (exact):
```
R  = red128(le128(s))              # 128 -> 32, same shift-XOR reduction
                                   # as crc_fold_red128 in crc_fold.c
R  = rj_crc32_update_scalar(R, p, tail_len)
s  = le128(R)                      # zero-extended to 128 bits
```

Correctness argument: the fold invariant is
`red128(S) = CRC_register(bytes folded so far)`. After the helper,
`red128(s') = R = CRC_register(all len bytes)` by the scalar updater's
chaining contract (`include/raidjit.h:491`:
`rj_crc32_fast(c,d,n) == rj_crc32_update(c,d,n)`, same for
`_scalar`). Finalize then yields the standard CRC-32. Any 128-bit
value with the right reduction finalizes correctly, so re-extending
the 32-bit register is sound. `tail_len = 0` → driver skips the call
(no-op). `len < 16` → the kernel folds nothing; the whole stream is
the scalar tail applied to the init state (red128(0xFFFFFFFF) =
0xFFFFFFFF ✓).

This REPLACES v1's in-kernel `x^{-8r}`/`x^{64-8r}` tail constants:
no extra constants, the kernel hot loop stays uniform (whole chunks
only), and the tail cost — one red128 + ≤15 scalar bytes — is paid
once per stream per call, i.e. negligible. The helper is pure C,
unit-testable in isolation (§8.1).

---

## 7. Cost model: interleaved vs separate (Carlo's principle)

### 7.1 Instruction counts (exact, derived)

For D total bytes across all streams (tail excluded — negligible):

- PCLMULQDQ: D/16 chunks × 2 = **D/8** instructions
  (NOT D/32 — that was the rejected VPCLMULQDQ-packing arithmetic).
- VEXTRACTI32X4: D/16; VPXOR (xmm): D/8.

Per 64B per stream: 8 × PCLMULQDQ + 4 × VEXTRACTI32X4 + 8 × VPXOR.

### 7.2 Port model (SPR; analytic — see §7.4 for the empirical gate)

Ground truth (FINAL_SYNTHESIS.md, Agner Fog / Cawley): PCLMULQDQ =
1 uop, throughput 1/cycle, **port 5**; VEXTRACTI32X4 = 1 uop, port 5
(shuffle); VPXOR xmm = 1 uop, ports 0/1/5. Per 64B per stream the fold
adds ≈ 12 p5-uops + 8 p0/p1/p5-uops. The encode's XOR/parity traffic
is p0/p1-bound.

Carlo's principle: `cost(encode+CRC) ≈ max(cost(encode), cost(CRC))`
when the two compete for disjoint resources. First-order condition
for the multiplies to hide: the encode supplies ≥ ~8 p5-idle cycles
per 64B per stream on non-p5 ports (the parent's analysis phrases it
per 64B of stripe as ≥8N cycles on non-p5 ports for N streams). The
extracts/XORs add real pressure on p0/p1/p5 — hiding is PARTIAL, not
total, and degrades if the encode is itself shuffle-heavy.

### 7.3 The right baseline: interleaved vs separate

"Almost free vs nothing" is the wrong frame. The alternative to the
fold is not "no CRC" but a SEPARATE CRC pass: re-reading D bytes
(they are cold by then for real stripe sizes — DRAM traffic) PLUS the
same D/8 multiplies. Prior measurement (MEMORY.md, CRC Phase 2 entry):
single-lane 128-bit fold ≈ 3× vs scalar CRC ≈ 10× — i.e. the
interleaved fold was already measured well cheaper than a separate
scalar pass BEFORE counting the memory traffic it saves.

REDUX wins on two independent axes, either sufficient:

1. **Memory:** the fold consumes vectors already in registers — ZERO
   extra memory traffic vs a full D-byte re-read for the separate
   pass.
2. **Compute overlap:** the D/8 multiplies sit on p5 while the encode
   works p0/p1 (§7.2); a separate pass serializes them.

So the claim is: `cost(interleaved) ≤ cost(encode) + ε`, with ε small
and bounded by the §7.2 port analysis — and strictly below
`cost(encode) + cost(separate CRC pass)` by the memory traffic alone.
The old "fold_cycles/byte < dram_cycles/byte" test
(FINAL_SYNTHESIS.md:108) is RETIRED as a ship gate for REDUX; it
tested the wrong baseline.

### 7.4 Empirical gate (numbers stay measured, never analytic)

No cycle count in this spec is a promise. The done-gate is a
benchmark on SPR hardware, interleaved vs separate-pass CRC over the
same fused op sequence, across the topology corpus (§8): ship iff
interleaved ≤ separate − margin, with the margin recorded. Per the
standing rule (v1 Assumptions §5), no analytic figure may be
presented as measured in code comments, docs, or commit messages.

---

## 8. Test matrix

### 8.1 Unit: the math (no JIT needed)

- **C-model differential:** a C model of (§2.1 fold + §6 tail +
  finalize) vs `rj_crc32_update` vs zlib `crc32`, randomized:
  lengths 1..2^20 (all residues mod 16 — the tail classes),
  unaligned buffers, random data, ≥10k vectors. Bit-identical, all
  three.
- **Tail helper isolation:** `rj_crc_fold_tail` vs full-message
  `rj_crc32_update` for tail_len 0..15 on random prefixes; also
  `tail_len=0` is a byte-exact no-op on the state.
- **Red-green (mutation):** each of these MUST fail the suite —
  K1/K2 low bit flipped; init low-32 = 0 instead of 0xFFFFFFFF;
  tail helper skipped (`tail_len>0` ignored); finalize missing the
  `^ 0xFFFFFFFF`; one state xmm not restored at kernel exit
  (simulated in the C model by dropping a chunk). A new test that
  cannot be made to fail is rejected (tautology rule).

### 8.2 Kernel differential (the real correctness gate; no timing)

- Randomized topologies (raid5, raid6, rs(6,3), rs(10,4), custom),
  lengths 1..2^20, all tail classes, unaligned, random data: CRCs from
  the fold kernel MUST be bit-identical to `rj_crc32_fast` (Phase 1)
  and zlib, for EVERY stream (inputs AND parities). Any mismatch =
  ship-blocker. (v1 R13, retained.)
- **Exercise guarantee** (v1 R13 Finding 10, retained): tests run with
  `benchmark=0` and bypass selection — enumerate candidates, build
  each `crc_fold=1` version directly via `jit_build_version`, assert
  the fold path executed (plan-ID "crcfold" token present), then
  differential-test. Closes the heuristic-may-not-pick hole and keeps
  timing out of correctness tests.
- **N sweep:** N = 1..27 foldable (register budget edge at 27/28),
  plus N = 28 → asserts scalar fallback taken and results identical.

### 8.3 Combination tests (interop, not isolation)

- Fold × {nt=0/1, prefetch on/off} twins: CRCs identical across all
  twins; parity DATA bytes identical (fold must not perturb encode
  output).
- Fold × tail classes × unaligned × {in-place parities}.
- PARITY_UPDATE-targeted CRC op → scalar path (scope §3.5); results
  identical to ENCODE-targeted fold on the same bytes.
- Concurrent `rj_fused_call` on one handle from threads (per-call
  fold_state; TSan or helgrind if available).

### 8.4 Fallback tests

- Gate closed (uarch override to non-SPR; feature-flag injection
  clearing `pclmulqdq`): scalar path taken, CRCs bit-identical to
  gate-open run.
- `rj_compile_want_crc` failure → plain compile; driver runs 3-arg
  kernels; identical CRCs.
- QEMU aarch64 + riscv64: gates closed → scalar path proven (v1 R13,
  retained). `command -v qemu-aarch64` first; never claim absence.

### 8.5 Byte-identity / anti-leak (v1 R14, retained)

- Regen `make kernels`; diff every section against
  `hidden_files/crc_fold_baseline_sha256.txt`: all non-CRC sections
  identical. Any difference in a non-CRC kernel = ship-blocker.
- Anti-leak (v1 Finding 9): with `want_crc=0`, ZERO `crc_fold=1`
  candidates enumerated (plan-ID token absent in `rj_list_variants`)
  AND emitted bytes identical pre/post REDUX (A/B via two worktrees,
  never git-stash).

### 8.6 Hygiene gates

- `make test` AND `make test-asan` green (the A5 lesson: EVERY build
  config a done-gate names — verify, don't infer).
- `grep -rn "TODO\|FIXME\|XXX\|not implemented" src/` clean at
  delivery; docs consistent (spec ↔ code ↔ README).
- Mutation/red-green evidence for every NEW test (the Phase 1 R4
  corruption-harness pattern).
- Self-attack pass per SKILL.md (malformed topologies, 1-byte
  stripes, N=1/27/28, resource limits, concurrent compiles) — report
  what was tried.
- ≥2 independent peer reviews, none by the implementer; all findings
  remediated and re-tested.

---

## 9. Reused from Phase 2 v1 vs changed in REDUX

### Reused (v1 → REDUX, unchanged or adapted)

1. **Planner plumbing:** `plan_t.want_crc` (plan.h:195),
   `cand_t.crc_fold` twin enumeration gated by the viable() check,
   `rj_compile_want_crc` / `rj_jit_crc_fold` /
   `rj_jit_handle_crc_fold` (src/api.h:52–66), twin_dominates pinning
   discipline, "crcfold" plan-ID token mechanics.
2. **Driver skeleton:** `src/fused.c` `fused_driver_call` — 4-arg
   dispatch through `rj_fn_crc_fold` (src/fused.h:37) when
   `crc_fold=1`, per-call stack `fold_state`, scalar fallback when the
   gate is closed or `want_crc` compile fails.
3. **Finalize:** `rj_crc_fold_finalize_x86()` — already implements the
   single-128-bit-state-per-stream 128→32 reduction; reused VERBATIM.
4. **Fold math:** formula `W = S XOR C;
   S' = CLMUL(W[63:0],K1) XOR CLMUL(W[127:64],K2)`, K1/K2 values,
   reflected init `0xFFFFFFFF` — as documented in
   `src/x86_64/crc_fold.c`'s corrected header comment.
5. **Emitters:** `em_vpternlog`, `em_vextracti32x4` reused.
   `em_vpclmulqdq` remains for other consumers (unused by REDUX).
6. **Test doctrine:** differential-vs-zlib-vs-`rj_crc32_fast`,
   exercise guarantee, byte-identity regen, anti-leak, red-green
   mutation, peer-review bars (§8).

### Changed (v1 → REDUX)

1. **Fold execution:** 128-bit `PCLMULQDQ` per stream (§2.2) instead of
   VPCLMULQDQ lanes; NEW emitter `em_pclmulqdq` (VEX.128.66.0F3A.44)
   required — derived under the AGENTS.md emitter rules.
2. **Register allocation:** N xmm states + 3 temps + 2 consts (§4),
   top-down reservation, exact `plan_need_regs` accounting,
   fail-closed at N > 27. (v1: 1 zmm = 4 lanes per stream.)
3. **fold_state layout:** `nstreams × 16` bytes, single-state init
   (§3.1–§3.2). The tree's `nstreams × 64` + 4-lane init is STALE and
   must be updated with the implementation.
4. **Remainder:** no N%4 handling — eliminated (§5).
5. **Tail:** driver-side scalar tail via `rj_crc_fold_tail` (§6);
   v1's in-kernel `x^{-8r}` tail constants dropped.
6. **µarch gate:** SPR-only + `pclmulqdq` + `avx512f`; `vpclmulqdq`
   feature NO LONGER required (§3.6). AArch64/RISC-V stay closed.
7. **Cost baseline:** interleaved-vs-separate (§7.3); D/8 PCLMULQDQs
   per D bytes (§7.1). The old fold-vs-DRAM test is retired.
8. **Scope:** ENCODE targets only (retained from v1 R13; stated here
   as §3.5).

---

## Assumptions (declared for veto)

1. `len ≥ 1` on every `rj_fused_call` (existing assert; fold kernels
   unreachable at len 0).
2. SPR-only gate: ICL-SP/Zen4 re-open only with per-µarch throughput
   data and a spec amendment.
3. `benchmark=1` selection machinery unchanged; the µarch gate
   controls enumeration only; tests use `benchmark=0`.
4. All cycle/uop figures in §7 are analytic (Agner Fog / Cawley via
   FINAL_SYNTHESIS.md); the empirical benchmark in §7.4 is the gate —
   no analytic number may be presented as measured.
5. The kernel entry/exit state load/store uses VMOVDQA (aligned);
   the driver guarantees ≥16B alignment of `fold_state`.
