# CRC Phase 2: interleaved per-microarchitecture CRC folding (Option A)

Owner approval: 2026-10-08 ("Implement A + C"). Standing constraints: NO
timing benchmarks, NO perf guard, no push, all work under `nice`,
non-CRC codegen byte-identical (baseline:
`hidden_files/crc_fold_baseline_sha256.txt`, 150 sections).

Design inputs (read before implementing):
- `hidden_files/crc_optimum_research/FINAL_SYNTHESIS.md` — Option A
  construction (§Phase 2), per-µarch verdict table. THE spec basis.
- `hidden_files/design_crc_fold.md` §11 — verified PCLMULQDQ fold
  formulation, constants K1=0xae689191 / K2=0xccaa009e, bit-exact vs zlib.
- `hidden_files/crc_impl_SPEC.md` — Phase 1 (Option C) spec; the
  standalone `rj_crc32_fast` API is the differential oracle and the
  scalar-fallback path.

Contract: `~/workspace/skills/world-class-implementation/SKILL.md` (all 8
done gates). AGENTS.md load-bearing lesson: any new `cand_t` axis is pinned
in `prune.c::twin_dominates` IN THE SAME CHANGE.

Branch: `crc-fold` from Phase-1 master (d77fdb9). Work ONLY in the
worktree. Never commit to main; integration is by ff-only merge.

## 1. What Phase 2 builds

Interleaved CRC-32/ISO-HDLC folding INSIDE the JIT encode kernel: while the
kernel computes RAID parity, it folds every input vector (after load) and
every output vector (before store) through a carryless-multiply fold,
accumulating per-stream 128-bit fold state in dedicated vector registers.
Zero extra memory traffic by construction. The driver finalizes per-stream
state to 32-bit CRCs at the end of the fused call.

This is Option A from FINAL_SYNTHESIS.md, with per-microarchitecture gating:
interleave ONLY where the vector unit is strong enough that the fold hides
under DRAM bandwidth in the real-RAID regime. Everywhere else, the fused
path uses the Phase-1 standalone API (scalar pass over the tile buffers —
the current `rj_fuse_compile([ENCODE, CRC])` behavior, which stays).

## 2. Per-µarch gating (conservative)

From FINAL_SYNTHESIS.md §Phase 2 "almost free" test (fold_cycles/byte <
dram_cycles/byte @ 0.06):

INTERLEAVE (fold in-kernel):
- x86-64: AVX-512 + VPCLMULQDQ on SPR (fold 0.028 cyc/B < 0.06, 53% margin).
  Conservative: gate on `cpu->vpclmulqdq && cpu->avx512f` AND
  `u == RJ_UARCH_INTEL_SAPPHIRERAPIDS`. The allowlist is the conservative
  part: we do NOT trust feature bits alone (a weak VPCLMULQDQ implementation
  would silently be 2-4x, not "almost free").
  
  Zen4 (0.047, 21% margin) and ICL-SP (0.055, 8% margin) are EXCLUDED from
  the Phase 2 allowlist. FINAL_SYNTHESIS.md labels both "borderline" and
  states they "deserve empirical checking first" when benchmarks resume.
  The prior spike's analytic VPCLMULQDQ throughput estimate for ICL-SP was
  ~3x optimistic — analytic optimism on exactly these numbers is a
  demonstrated failure mode. They join the allowlist AFTER measurement
  confirms the bound, not before. (This is the conservative gate the
  "almost free" safety claim rests on.)
  
  Note: `RJ_UARCH_AMD_ZEN5` exists and is not allowlisted — deliberate
  conservative default (no data), not an oversight.
- AArch64: ARMv8.2+SHA3+crypto with the Cawley fusion structure — but
  Option A uses ONLY the vector unit (the scalar crc32 unit needs GPRs;
  extracting from vector regs costs more than it saves — synthesis §Phase 2
  "Only the VECTOR unit is usable"). M1-class (24 B/c vector-only) qualifies
  (0.041 < 0.06). HOWEVER: Phase 2 does NOT implement AArch64 interleave.
  The codebase has no Apple-silicon µarch detection (`rj_cpu_uarch` returns
  RJ_UARCH_UNKNOWN on AArch64), making a conservative allowlist
  unimplementable. The gate is therefore CLOSED on AArch64 for Phase 2
  (scalar fallback always). AArch64 interleave is deferred to a future phase
  that adds the µarch detection. This is a deliberate conservative choice,
  not an oversight.
- RISC-V: UNKNOWN (no published vclmul throughput). Conservative: NEVER
  interleave on RISC-V in Phase 2. Always scalar fallback. The gate can be
  revisited when core data exists.

SCALAR FALLBACK (current behavior, Phase-1 `rj_crc32_fast` over tiles):
- x86-64: SKX/CLX, Zen2/3, any AVX-512 without VPCLMULQDQ, non-AVX-512.
- AArch64: N1 and anything not on the allowlist.
- RISC-V: always.
- Unknown/new µarchs: ALWAYS fallback (conservative default).

The gate lives in ONE place: `src/fused.c`, a function
`fused_crc_interleave_ok(const rj_cpu_info *cpu)` returning 1/0. The
planner's `plan_candidates` enumerates `crc_fold=1` ONLY when
`pl->want_crc && fused_crc_interleave_ok(cpu)`. The backend ALSO
re-checks (defense in depth): `jit_build_version` rejects `crc_fold=1`
if the gate is closed, returning a clean build error, never wrong code.

## 3. Planner integration

### 3.1 `cand_t.crc_fold` (new axis)

```c
int crc_fold; /* 1 = fold CRC-32/ISO-HDLC over all n_in input streams and
               * n_out output streams while computing them (interleaved
               * Option A). Enumerated by plan_candidates ONLY when
               * pl->want_crc && fused_crc_interleave_ok(cpu); otherwise
               * every candidate has crc_fold=0. Pinned in
               * prune.c::twin_dominates like every other axis (SAME
               * change). The fold state lives in dedicated vector
               * registers (never spilled in the steady state); the
               * backend rejects (clean error) if the register budget
               * does not fit. Byte-exact vs rj_crc32_update. */
```

### 3.2 `plan_t.want_crc` (new field)

Set ONLY via the internal fused path (`fused.c` → internal `rj_compile`
variant declared in `src/api.h`, NOT public). When `want_crc=1`:
- `plan_candidates` enumerates ONLY `crc_fold=1` candidates (no plain).
  Rationale (from design_crc_fold.md §3): a crc_fold kernel does
  encode+CRC in one pass; the scalar alternative is encode + a full extra
  pass. Selection by construction. The normal duel still picks the fastest
  AMONG crc_fold candidates (apples-to-apples).
- If the want_crc compile yields no buildable kernel (gate closed,
  register pressure, unsupported combo), `fused.c` recompiles with
  `want_crc=0` and uses the scalar driver path. Never wrong code, never
  a hard failure.

### 3.3 `prune.c::twin_dominates`

Pin `crc_fold` in the field-by-field comparison IN THE SAME COMMIT as the
axis (AGENTS.md lesson). A `(crc_fold=1,ternlog=1)/(crc_fold=0,ternlog=0)`
pair must never satisfy the twin relation.

### 3.4 `plan_need_regs`

Account fold-state registers when `crc_fold=1`: `nstreams` vector registers
(`n_in + n_out`), one per stream. The existing `need > nregs` filter in
`plan_candidates` then rejects unencodable (U, topology) combos before the
backend sees them. Per-ISA: x86-64 = 1 zmm per stream (4 lanes packed);
AArch64 = 1 qreg per stream; RISC-V = N/A (never interleaved).

REGISTER BUDGET ANALYSIS (Major 6, spec deliverable): Before the x86-64
backend is complete, the implementer MUST produce a per-(topology, U)
budget table showing which combinations fit in 32 zmm registers:
`encode_regs(U, topology) + nstreams <= 32`. For RS(10,4) (14 streams),
14 + ~20 > 32, so large-U combos will be rejected and the driver will
silently fall back. The table must state the expected coverage (which
topologies actually get interleave at which U). Silent fallback is "never
wrong code" but must be VISIBLE — the table is the visibility mechanism.
If a flagship topology (e.g. RS(10,4)) never qualifies, that is a
product-level finding to report, not a silent non-event.

### 3.5 `planid.c`

Encode a `crc_fold` token in the plan ID so stable plan IDs reproduce the
twin. Format follows existing twin tokens (e.g. `+crc_fold`).

### 3.6 Stage-2 tail twins

`notail`/`predtail`/`maskedtail` are built from the winning candidate
post-selection; they copy the winner's `cand`, so `crc_fold` rides along.

CRITICAL: For masked/predicated tails, ONLY the valid tail bytes may be
folded. The AVX-512 masked tail (k1) stores only r bytes of a 64B vector;
folding the full 64B vector would fold garbage from masked-off lanes →
wrong CRC. The backend MUST EITHER:
(a) zero the masked-off lanes before folding (e.g. via the mask register),
OR
(b) use the scalar `rj_crc32_update` tail path for the tail bytes (composed
via shift-XOR), not the vector fold.

Same for SVE predtail (predicate governs which lanes are valid). A backend
implementer following "fold the tail bytes" literally without masking ships
wrong CRCs on tail residues whenever the maskedtail/predtail twin wins.
Covered in the combination test matrix (§7.4) with tail-residue-specific
differential tests.

## 4. Kernel/driver contract

### 4.1 Kernel signature

The crc_fold kernel is built for a plan with `n_in` inputs, `n_out`
outputs. The driver calls it as `fn(in, out, len, fold_state)` where:
- `in`: `n_in` input pointers (as today).
- `out`: `n_out` output pointers (as today).
- `len`: byte length (as today).
- `fold_state`: pointer to `nstreams * 64` bytes, 64-byte aligned,
  `nstreams = n_in + n_out`. Streams `0..n_in-1` = input streams,
  `n_in..n_in+n_out-1` = output streams. Each stream's 64 bytes hold
  one zmm (4x 128-bit lanes, every-4th 16B chunk per §4.2). Owned by
  the driver for the whole fused call (persists across tiles).

CALLING CONVENTION SAFETY (blocker fix): A crc_fold kernel uses a 4-argument
calling convention, DIFFERENT from the standard 3-argument `rj_fn`
(`fn(in, out, len)`). A crc_fold kernel MUST NEVER be installed in the
normal dispatch table or called via the 3-arg path — doing so would pass
a garbage `fold_state` pointer (wrong code, potential crash).

Enforcement:
- `rj_compile` (public) NEVER sets `want_crc`; only the internal fused
  path does. Therefore no crc_fold kernel can enter the public dispatch
  tables.
- The fused driver (`fused.c`) is the SOLE caller of crc_fold kernels.
  Before each call it checks `rj_jit_crc_fold(v) == 1`; if 0, it uses the
  3-arg call (scalar fallback). If 1, it uses the 4-arg call.
- The backend's `jit_build_version` MUST reject (clean error, never
  wrong code) any attempt to build a crc_fold kernel when the uarch gate
  is closed, even if `cand.crc_fold=1` somehow arrives.
- Test: `test_crc_fold_callconv.c` verifies that a crc_fold version
  is never returned by `rj_compile` (public path) and that the fused
  driver uses the correct arity.

Kernel behavior:
- Load fold state at entry (one vector load per stream, or GPR init per
  backend convention — backend's choice, documented).
- Fold every input vector AFTER load, every output vector BEFORE store.
- Cover ALL `len` bytes INCLUDING the scalar tail (the tail path folds
  too; no byte left behind).
- Store fold state at exit.

### 4.2 Fold formulation (x86-64; AArch64/RISC-V per worker spec)

From design_crc_fold.md §11 (verified bit-exact vs zlib):
- State S: 128 bits per stream, init S_0 = 0xFFFFFFFF (low 32 bits; i.e.
  the reflected init).
- Per 16B chunk C (LE): `W = S XOR C`; `S' = CLMUL(W[63:0], K1) XOR
  CLMUL(W[127:64], K2)` with K1=0xae689191, K2=0xccaa009e.
- With VPCLMULQDQ (the interleave path): 4 lanes per zmm, each lane an
  independent 128-bit fold state for every-4th 16B chunk (zlib-ng
  structure). Data XOR fused via VPTERNLOGQ (no shuffle tax).
- Tail (r bytes, 1<=r<=15): fold per the §11 tail formulation, or scalar
  `rj_crc32_update` over the tail bytes composed via the shift-XOR
  rule (worker chooses; property tests are the check).
- Finalize: `CRC = red_QREF(S) XOR 0xFFFFFFFF`, then combine the 4 lanes
  per stream via the shift-XOR composition rule
  (`CRC(A||B) = shift(CRC(A),|B|) XOR CRC(B)`).

AArch64: PMULL-based fold per FINAL_SYNTHESIS.md ARM rows (M1-class).
RISC-V: N/A (never interleave in Phase 2).

### 4.3 Driver (`src/fused.c`)

- Allocates `nstreams * 64` bytes per fused call (64-byte aligned, one
  zmm per stream), initializes per backend convention.
- Passes `fold_state` to each tile's kernel call.
- After all tiles: finalizes 128-bit → 32-bit per stream (backend-defined;
  x86-64: driver-side Barrett reduction of the 4 lanes then XOR per
  §4.2; backends with 32-bit state finalize trivially).
- Writes final CRCs to the caller's `crc[]` array.
- Fallback: if `fused_crc_interleave_ok(cpu) == 0`, use the Phase-1
  scalar path (`rj_crc32_fast` over tile buffers) — current behavior,
  untouched.

### 4.4 New `src/api.h` accessor (internal)

`int rj_jit_crc_fold(const rj_jit *j, int *nstreams);`
- 1 = in-kernel CRC active (kernel expects `fold_state`), `nstreams` set.
- 0 = scalar fallback (kernel is a plain encode kernel).
Declared in `src/api.h` (NOT public `include/raidjit.h`).

## 5. Backend mandates

Common:
- Fold state in dedicated vector registers across the inner loop (never
  spilled in the steady state).
- Every fold emitter increments `cg.uops` exactly (model exact by
  construction).
- Unsupported (kind, twin-combo, ISA) → clean build rejection (return -1
  with message), never wrong code, never abort. Extends the
  `test_twin_audit` reachable sweep (must stay green).
- Emitters GAS cross-checked across randomized operands (AGENTS.md
  emitter lesson: clear ALL operand fields when deriving bases).
- `cg.uops` accounting: the fold's uops must be EXACTLY countable via
  disassembly (test: count fold instructions in emitted code, assert
  `uops_per_byte` delta vs non-fold twin matches).

x86-64 (coordinator):
- VPCLMULQDQ 4-lane fold per §4.2. Requires `cpu->vpclmulqdq &&
  cpu->avx512f` AND µarch allowlist (SPR/Zen4/ICL-SP).
- New `rj_cpu_info` fields if needed (check existing flags first).
- Scalar tail folds per §4.2.

AArch64 (worker):
- PMULL-based fold per FINAL_SYNTHESIS.md (M1-class allowlist).
- NEON/SVE register-budget analysis in the worker's spec addendum.

RISC-V (worker):
- NO interleave in Phase 2 (gate always closed). Worker verifies the
  scalar fallback path and the planner never enumerates `crc_fold=1`
  for RISC-V. (Documents the "unknown" as a deliberate conservative
  choice.)

## 6. Uop model

Fold emitters count `cg.uops`. Non-tautological test: disassemble the
emitted crc_fold kernel, count fold instructions (e.g. `VPCLMULQDQ`
occurrences), assert the `uops_per_byte` delta vs the non-fold twin
equals exactly `count * uops_per_insn / group_bytes`.

## 7. Test matrix

### 7.1 Correctness (property tests)
- Random topologies: RAID-5/6, RS(6,3), RS(10,4), custom.
- Random lengths: 0..4MiB including every tail residue class (0..63).
- Random alignments (0..63 byte offset).
- Random data.
- Assert: per-stream kernel CRC == `rj_crc32_fast` (Phase-1 oracle) ==
  `rj_crc32_update` (scalar reference).
- Run under ASan/UBSan; run under QEMU for AArch64 (even though AArch64
  interleave is allowlisted, the fallback path must be correct) and
  RISC-V (fallback only).

EXECUTION ENVIRONMENT (Major 8): The x86-64 VPCLMULQDQ fold path requires
SPR silicon (or QEMU x86-64 with `-cpu` exposing VPCLMULQDQ+AVX512F and
reporting SapphireRapids µarch). If neither is available in CI, the fold
path cannot be tested — this must be a LOUD skip (not silent), and the
gate-closed fallback path must still be tested. The spec requires the
environment to be named in the test report.

MULTI-TILE (Major 9): `fold_state` persists across per-tile kernel calls.
Tests MUST include:
- Single-tile vs multi-tile differential: same data, different tile sizes,
  identical CRCs.
- Stale-state red-green: skip the state reload at tile entry → test MUST fail
  (proves the hand-off is load-bearing, not tautological).
- Tile-boundary residues: lengths that put chunk/lane boundaries at tile
  edges.

### 7.2 Red-green
- Flip a fold constant (K1/K2) → tests fail.
- Drop a fold (skip one stream) → tests fail.
- Gate closed (synthetic cpu without allowlisted µarch) → clean fallback,
  correct CRCs via scalar path.

### 7.3 Byte-identity (non-CRC untouched)
- sha256 over all 150 kernels-doc sections in
  `hidden_files/crc_fold_baseline_sha256.txt` before vs after: ALL match.
- `crc_fold` unreachable from `make kernels` (want_crc never set there).

### 7.4 Combination
- `crc_fold` × {nt, prefetch, notail, maskedtail/predtail}:
  builds or rejects cleanly (no abort, no wrong code).
- `test_twin_audit` reachable sweep green.

### 7.5 Fallback
- Synthetic `rj_cpu_info` with gate closed → scalar path, correct CRCs.
- want_crc=1 with no buildable kernel → fused.c recompiles want_crc=0,
  correct CRCs.

### 7.6 Full suite
- `make test` green.
- `make test-asan` / `make test-ubsan` green (all build configs per
  AGENTS.md lesson).
- QEMU aarch64 + riscv64 suites green.

## 8. "Almost free" — what we claim

Structural only (benchmarks postponed):
- Zero extra memory traffic by construction (fold consumes vectors
  already in registers).
- Fold uops bounded and reported by the static model.
- The per-µarch gate ensures we only interleave where the analytic
  bound says the fold hides under DRAM bandwidth.
- NO measured numbers presented. The "almost free" claim is
  explicitly analytic/gated, not empirical.

REGIME CONDITION (Major 5): The "almost free" claim is a DRAM-bound
regime claim. In the cache-resident microbenchmark regime, the fold does
NOT hide (e.g. SPR: fold at ~108 GB/s vs encode at 100+ GB/s — the fold
is comparable, not hidden). The gate is static and regime-blind; it
will interleave even for cache-resident stripes where the overhead is
higher. This is accepted for Phase 2 because:
(a) real RAID is DRAM-bound (the target regime),
(b) the fallback (Phase-1 `rj_crc32_fast` extra pass) is itself fast,
so the worst case is bounded.
When benchmarks resume, the regime-specific behavior must be measured.

THRESHOLD ASSUMPTION: The 0.06 cyc/B threshold assumes 50 GB/s DRAM
bandwidth @ 3 GHz. At higher per-core bandwidth, the threshold tightens
and the safety margin shrinks. §2's allowlist must be re-evaluated if
the bandwidth assumption changes.

## 9. Docs

- This spec.
- Code comments at contract points (`cand_t.crc_fold`,
  `plan_t.want_crc`, `fused.c` gate, `api.h` accessor).
- `docs/` update if a CRC doc exists (check `docs/crc.md`).

## 10. Done gates (world-class-implementation)

1. Spec exists (this file) + 2 peer reviews of the spec.
2. No placeholders; full implementation.
3. Adversarial test matrix (§7) incl. combination tests.
4. Tests never tautological (red-green §7.2).
5. Full suite green (all build configs).
6. 2+ independent peer reviews, all findings remediated + re-tested.
7. Self-attack pass.
8. Byte-identity proof (§7.3).

## 11. Work split

- Coordinator (this worktree): spec, planner integration (§3),
  x86-64 backend (§5), fused.c driver (§4.3), verification
  orchestration, reviews, self-attack.
- Worker A (AArch64 backend): separate worktree. Phase 2 scope is
  FALLBACK-ONLY verification: confirm the planner never enumerates
  `crc_fold=1` for AArch64, the scalar path is correct under QEMU,
  and the gate stays closed. (Interleave implementation deferred —
  see §2.)
- Worker B (RISC-V): separate worktree, verify fallback-only path
  (no interleave), planner gate test.
