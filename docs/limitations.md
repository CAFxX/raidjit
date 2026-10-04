# raidjit — limitations and sharp edges

This document expands the "Known limitations" section of the README. Every
item here is a real, verified property of the implementation — not a guess.
Failure modes are stated precisely: "clean failure" means `rj_compile`
returns NULL with an error message; anything else is called out explicitly.

## 1. Topology limits

### 1.1 Default RS matrix is MDS only inside a small envelope
`rj_topo_raid6()` and `rj_topo_rs()` use `M[j][i] = 2^(j*(ndata-1-i))`,
emitted by the plan-time generator-matrix search (`docs/matrix_search.md`).
Since `ord(2) = 51` in GF(2⁸)/0x11B, the matrix is a true Reed-Solomon
(every ≤ npar erasure set recoverable) only for **npar ≤ 3 and ndata ≤ 51**
(`npar ≤ 1` has no basis restriction). The constructors return NULL outside
this envelope. Every emitted matrix is proven MDS by exact enumeration of
all C(ndata+npar, npar) minors — no sampling. Verified failure examples:
`rs(52,2)` fails `{0,51}`; `rs(20,4)` fails `{0,17,21,22}`.

Wider MDS topologies are available via `rj_topo_custom()` — e.g. Cauchy
matrices are MDS for `ndata + npar ≤ 255`. Note the tradeoff: non-Vandermonde
matrices lose the Horner fast path and fall back to MULACC (correct, slower).

The default matrix was deliberately *not* changed to a wider-MDS construction:
doing so would change the on-disk parity format (wire-format break) for no
correctness gain inside the envelope.

### 1.2 Register-pressure parity ceilings
The JIT keeps all accumulators in vector registers (no spilling), so beyond a
parity count there is no encodable candidate and `rj_compile` fails cleanly
with "no candidates for this CPU/plan" — never wrong code. Ceilings at
unroll U=1 (`nregs` = 16 on SSE/AVX2, 32 on AVX-512):

| plan × method      | formula              | SSE/AVX2 | AVX-512 |
|--------------------|----------------------|----------|---------|
| Horner / nibble    | 3·n+6 ≤ nregs        | n ≤ 3    | n ≤ 8   |
| Horner / GFNI      | 2·n+2 ≤ nregs        | n ≤ 7    | n ≤ 15  |
| MULACC / nibble    | n+7 ≤ nregs          | n ≤ 9    | n ≤ 25  |
| MULACC / GFNI      | n+3 ≤ nregs          | n ≤ 13   | n ≤ 29  |
| XOR                | n·U+1 ≤ nregs        | n ≤ 15   | n ≤ 31  |

For recovery, `n` is the number of *failed* disks, not `npar`. Since the
named constructors are capped at `npar ≤ 3`, the binding constraint for
named topologies is the envelope (§1.1), not this table; the table bites
only for custom matrices with high parity counts. There is no accumulator
spilling or output-row blocking fallback — this is the main functional gap
versus "arbitrary legal topologies", and it fails safe.

### 1.3 Custom matrices: MDS is the caller's responsibility
`rj_topo_custom()` accepts any matrix. Encode always works. Recovery of a
failure set whose columns are rank-deficient fails `rj_compile` with
"singular system" — safe, but per-failure-set: some subsets may compile while
others fail. There is no whole-matrix MDS pre-check. If you need a guarantee,
verify the MDS property of your matrix yourself (all square submatrices of
the parity-check matrix nonsingular) or use a construction that is MDS by
proof (Cauchy with distinct nonzero x, disjoint y).

### 1.4 Hard caps
`ndata + npar ≤ 256` (`RJ_MAX_DISKS`); `ndata ≥ 1`; `npar ≥ 0`. `rj_topo_raid5`
needs `ndata ≥ 2`, `rj_topo_raid6` needs `3 ≤ ndata ≤ 51`. Recovery with
`nfailed > npar` fails cleanly.

## 2. Caller contracts (unchecked — violating them silently computes wrong data)

- **Buffer alignment:** every `in[]`/`out[]` buffer must be 64-byte aligned.
  The kernels assume alignment; misaligned buffers are not detected.
- **No buffer overlap:** no two buffers may overlap or alias each other
  (`in[]` vs `out[]`, or outputs vs each other). The kernels stream
  load-compute-store with no aliasing analysis.
- **Recover input order:** `in[]` must be surviving-disk pointers in
  *increasing disk-index order*; `out[]` one pointer per failed disk in
  *increasing failed-index order*. The planner builds its survivor list
  internally in increasing order and matches coefficients positionally —
  passing survivors out of order is not detected and rebuilds wrong data.
- **Pointers/length:** every pointer must be valid for the full `len` bytes.
  `len = 0` is legal (no-op). `len` is arbitrary, need not be a vector
  multiple; correctness never depends on `len`, only speed does.
- **Op/topology lifetime:** the `rj_op` and `rj_topology` must outlive the
  `rj_compile` call that consumes them (compile copies what it needs); the
  returned `rj_jit` is independent afterwards.

## 3. Codegen and platform scope

- **x86-64, AArch64, RISC-V.** Three backends share the planner and
  API; only the emitter/JIT/CPU-detection differ (`src/`, `src/aarch64/`,
  `src/riscv/`).
- **AArch64** (added 2026-10-01): NEON + SVE/SVE2. Requires
  `getauxval(AT_HWCAP)`; SVE vector length via `PR_SVE_GET_VL` (disabled
  if reported VL < 16 bytes, as happens under qemu-user). NEON tested
  under qemu-aarch64 (27417 matrix runs, 0 failures, 2026-10-02). SVE/SVE2 codegen is
  VL-agnostic (runtime `cntb`, never hardcoded width) and execution-tested
  under qemu-aarch64 8.2.2 `-cpu max,sve-default-vector-length=128/256`
  (128k+ matrix runs each, 0 failures).
- **RISC-V** (added 2026-10-01): RVV backend (`src/riscv/`), LMUL=1,
  `vsetvli` strip-mining, nibble-shuffle multiply via `vrgather`.
  Tested under qemu-riscv64 `-cpu max` (10326 matrix runs, 0 failures).
  The P-extension (packed SIMD) is explicitly not implemented.
- **x86-64 details.** Requires CPUID; ISA levels are
  gated on hardware *and* OS state (XCR0 for AVX/AVX-512, OSXSAVE-gated
  XGETBV). Tested on AMD EPYC (AVX-512/GFNI) and Intel via QEMU TCG.
- **AVX-512 without BW is supported** (loads/stores use `VMOVDQU32`,
  AVX512F-only), but the nibble multiply path needs AVX512BW; without it,
  only XOR plans (and GFNI plans, if GFNI is present) have AVX-512
  candidates — multiplies fall back to AVX2/SSE.
- **QEMU 8.2 TCG cannot execute AVX-512/GFNI**; those paths are covered by
  native execution only. QEMU models exercise detection + AVX2-and-below
  execution.
- **Generated code makes no calls** and uses only unaligned vector memory
  ops. It is thread-safe (no mutable shared state; concurrent `rj_compile`
  is safe) but **not async-signal-safe**.
- **Prologue leaves `rsp % 16` as 0 or 8** depending on the push count
  (RBX [+R15] [+R12–R14 when pointer hoisting is active]: 1, 2, 4 or 5
  pushes → `rsp%16` is 0, 8, 8 or 0). Safe today —
  no calls, no aligned spills — but a latent hazard: if generated code ever
  calls out, `sub rsp,8` must be added (noted in `src/x86_64/jit.c`).
- **No code caching:** each `rj_compile` re-emits and re-benchmarks from
  scratch. Compiling many distinct ops has linear cost; cache the `rj_jit*`
  yourself if you compile in a hot path.
- **`VZEROUPPER`:** AVX/AVX-2/AVX-512 epilogues emit `VZEROUPPER` (`C5 F8 77`)
  before `ret` so callers using legacy SSE afterwards pay no AVX–SSE
  transition penalty. The SSE backend (legacy-encoded) emits none.
- **Memory:** code is `mmap`ed RW, filled, then `mprotect`ed RX — no RWX
  window at any point. Constant pools are small (hundreds of bytes);
  offsets are `int`-cast, fine for realistic sizes.

## 4. Benchmark methodology scope

- With `benchmark = 1` (default), **every candidate is validated against an
  independent scalar reference before timing** (scalar-only, vector-boundary,
  multi-group, and unaligned-offset lengths). A candidate that computes wrong
  results fails compilation before it can win on speed; a dedicated test
  corrupts a candidate's pool and requires rejection.
- Timing: several working-set sizes (¼×/1×/4× `bench_bytes`, clamped to
  4 KiB–16 MiB, default 1 MiB), warm-up, interleaved rounds; score = sum of
  per-size median times. Eligibility window = `max(1.5%, pooled measurement
  noise)` with noise = pooled relative IQR. Ties within noise resolve
  **deterministically** by lower µops/byte, then memory traffic/byte, then
  code size.
- Consequences:
  - A reported GB/s means exactly: *best among the generated legal
    candidates under this protocol on this CPU*. Not portable, not absolute,
    not a claim about other implementations. The header documents this.
  - Within-noise ties are deliberately resolved toward *efficiency*
    (µop-cache, memory bandwidth), not raw peak speed. A candidate that is
    0.5% faster but measurably more wasteful will lose by design.
  - With `benchmark = 0` the pick is heuristic (ISA level + unroll order)
    and `rj_info.gbps` is an estimate, not a measurement.
  - Compile-time cost of `benchmark = 1` is real (candidates × sizes ×
    iters); use `benchmark = 0` for one-shot compiles where it matters.

## 5. Robustness notes

- **Out-of-memory inside the JIT aborts** (`assert` in pool/emitter growth
  paths) — fail-fast, never silent corruption, but not a clean error return.
  Full error propagation is future work.
- `bench_iters <= 0` defaults to 11 and `bench_bytes = 0` defaults to 1 MiB
  (documented in the header); negative `bench_iters` is *not* rejected, just
  defaulted.
- `loop_align` must be 0 (auto) or a power of two; anything else is rejected
  by `rj_compile`.
- `emit_shadow_stack` requires OS-enabled CET; on non-CET hardware the
  emitted `setssbsy` raises `#UD` (verified under QEMU). `emit_endbr` is a
  NOP on non-CET CPUs and always safe. After `ret` every backend emits an
  unconditional trap sled (`ud2` on x86-64, `udf #0` on AArch64, `unimp`
  on RISC-V) and the dead space before the const pool is trap-filled, so a
  fall-through past `ret` faults immediately instead of sliding through
  NOPs into the pool.
- Determinism: same CPU + topology + op ⇒ same winner (deterministic
  tie-breaks). Across different CPUs the winner may differ — expected.

## 6. Deliberately not addressed

- No accumulator spilling / row-blocking fallback for §1.2 (fails safe;
  spilling would complicate the register allocator for marginal gain inside
  the named-topology envelope).
- No non-GF(2⁸) fields, no 32-bit x86, no streaming API.
  Partial-stripe parity update IS supported via rj_op_parity_update().
- The `rj_info` benchmark noise/window internals are not exposed
  programmatically (visible with `verbose = 1` on stderr).

## 7. Physical vs architectural registers

The `need ≤ nregs` filter in `plan_candidates` counts *architectural*
vector registers. That is the correct feasibility criterion: on x86-64,
encodability depends only on architectural register numbers, and renaming
to physical registers is transparent — physical pressure can never cause
*wrong code*, only slower code. Concretely, worst-case live values stay
small: e.g. AVX-512 Horner/nibble at the filter edge (`n_out=8, U=1`) holds
8 accumulators + 16 table registers + ~7 temps ≈ 31 architectural values,
far below typical physical vector files (~160–224 entries).

Physical-machine effects — PRF size, ROB/scheduler capacity, load/store
buffers, µop cache, move elimination — are optimized **empirically, not
modeled**: with `benchmark = 1` (default) the timed selection measures the
real kernel on the real CPU, so unroll factors that overflow the OOO window
or thrash the µop cache simply lose. This is stronger than any static
physical-register model, and it is why the candidate filter is allowed to
be purely architectural.

Codegen follows renamer-friendly idioms throughout: accumulators are
initialized with the dependency-breaking `pxor`/`vpxor` zero idiom (which
the renamer eliminates), 3-operand VEX/EVEX forms avoid register-to-register
moves (the `vd == s1` elision covers the legacy-SSE paths), GPRs are zeroed
with `xor r32,r32`, and temporaries are reused across independent chains so
renaming — not program order — determines physical assignment.

Caveat: with `benchmark = 0` the heuristic pick uses a roofline estimate
(`max(uops/3.5, mem/64)`) with no physical-register term. x86-64 does not
expose PRF/ROB sizes via CPUID, so a static term would be guesswork; the
documented guidance is to leave benchmarking on whenever the choice matters.
The same measure-don't-model principle will apply to the ARM/RISC-V ports
(§8).

## 8. Portability roadmap

- **Larger topologies:** `RJ_MAX_DISKS` is 256; named RS topologies remain
  capped by the MDS envelope (§1.1). The register-pressure ceilings (§1.2)
  are unchanged — they are per-kernel, not per-topology-count.
- **RVV `M` extension:** the RVV backend assumes the `M` extension is present
  alongside `V`. The F6 counted-loop prologue uses `mul`/`divu` for unroll
  factors U∈{3,5} (U∈{1,2,4} use shifts); a V-without-M core would fault on
  those kernels. In practice every RVV implementation ships `M` (RVA
  profiles mandate it; QEMU `-cpu max` has it), so this is a portability
  note, not a live constraint.
- **AArch64 (NEON, SVE/SVE2): implemented 2026-10-01** in `src/aarch64/`
  (emitter, JIT, CPU detection). NEON's `tbl` is the direct counterpart
  of `pshufb` for nibble multiplies; SVE/SVE2 use a length-agnostic loop
  structure (runtime `cntb`, predicated `whilelo`, single predicate p0).
  On CPUs reporting SVE2 (`HWCAP2_SVE2`), the SVE backend folds the
  nibble-multiply reduction into one EOR3 per multiply (see
  `docs/sve2_audit.md`); the desc string gains a `+eor3` tag. The
  baseline SVE path is unchanged for non-SVE2 machines.
  (Fixed 2026-10-02: P_DELTA now seeds its accumulator from the old-parity
  input (`in[2*nc+j]`); see §12.)
- **RISC-V (RVV, packed SIMD):** planned.
  RISC-V has two vector stories: RVV (scalable, like SVE) and the P extension
  (packed SIMD, fixed 32/64-bit subword operations across RV32/RV64, with
  versioned instruction subsets). Detection via `getauxval(AT_HWCAP)` /
  `hwprobe`/`riscv_hwprobe`; the candidate/selection machinery (plan →
  candidates → benchmark → winner) is ISA-agnostic by design.
- **VLMAX principle:** scalable-vector backends (SVE/SVE2, RVV) must use the
  full locally-available vector length (`VLMAX`), never a hardcoded assumed
  width. The JIT queries the hardware vector length at startup and sizes
  unroll factors, register budgets, and loop strides from the measured value.
  Fixed-width assumptions (e.g. "SVE is 256 bits") are a correctness bug on
  hardware with a different implementation choice.
- **RISC-V P-extension versions:** the P extension is not ratified (draft
  0.9.10 as of 2026; the task group reached Development Milestone in
  September 2026 and is heading to Internal Review). Multiple incompatible
  draft versions exist in the wild (0.5, 0.9.x, 0.18/0.20), with different
  encodings and mnemonics — toolchains ship separate branches per version
  (e.g. riscv-gcc `p-dev` for v0.20 vs a separate branch for v0.9.11).
  The packed-SIMD backend must therefore detect and handle the specific
  draft version present; the emitter must not assume one draft's encoding
  works on another.

## 9. Non-temporal stores

- `rj_options.nt_threshold = 0` (default) disables NT stores entirely: no
  NT candidate is generated and the emitted code contains no `MOVNTDQ` /
  `VMOVNTDQ`.
- Setting `nt_threshold > 0` adds an NT twin of every candidate
  (`MOVNTDQ` on SSE, `VMOVNTDQ` on AVX2/AVX-512). The twin participates
  in the normal benchmark protocol and wins only if it measures faster —
  NT is *never* enabled on static-model faith.
- **Fence contract:** NT stores are weakly ordered and the jitted code
  emits **no fence instruction**. A caller that reads the NT-written
  output buffers (on any CPU, including the calling thread) — or that
  needs the data visible to other agents/devices — **must execute
  `SFENCE` itself after the jitted function returns**, before touching
  the outputs. Same-thread, same-address reads need no fence (coherence
  and store-to-load forwarding cover them — cf. glibc's NT `memmove`,
  which deliberately omits `SFENCE`); the fence is required when another
  thread, device, or DMA engine must observe the NT-written data. On AArch64, the jitted code likewise emits **no `DMB`**
  instruction; callers that need ordering of non-temporal (`STNP`/
  `STNT1B`) stores must execute `DMB ISHST` themselves after the jitted
  function returns. Omitting the fence is a correctness bug in the caller,
  not in the library: without it, a subsequent load may observe stale
  data. Correctness tests (`tests/test_nt.c`) verify the shape (0×
  SFENCE, ≥1 NT store, temporal twins contain no NT store) plus output
  equality against the scalar reference for encode and recover.

### Empirical selection policy (measured 2026-10-01, AMD EPYC 9D25, 32MB L3)

`tests/study_nt.c`, RAID-6(8) AVX-512+GFNI:

| scenario | temporal | NT | winner |
|---|---|---|---|
| cold streaming, 128MB buffers (>> LLC), write-once | 12.75 GB/s | 25.91 GB/s | **NT, 2.03x** |
| immediate readback, 64KB buffers, write+read | 63.08 GB/s | 2.82 GB/s | **temporal, 22x** |

Policy: enable NT (`nt_threshold > 0` + `benchmark = 1`) when the
workload streams parity/DIF data that will not be read back soon —
large sequential writes beyond LLC, e.g. full-stripe RAID rebuilds or
backup writes. Keep temporal (default) when outputs are immediately
reused — small buffers, read-after-write patterns, or when the data
fits in cache. The benchmark protocol measures both and picks the
winner for the actual buffer sizes used at compile time, so the safest
choice is to enable the twin and let measurement decide.
- NT stores require the same 64-byte alignment the buffer contract
  already guarantees (§2); on x86/AArch64 misaligned NT stores fault
  like any aligned vector move (on RISC-V the annotated `vse8.v`
  unit-stride store never faults on misalignment — it is only slower —
  but the 64-byte contract makes the point moot everywhere).
- **RISC-V:** the NT twin prefixes every output `vse8.v` with the
  ratified Zihintntl `NTL.ALL` hint (`0x00500033`, an `ADD x0,x0,x5`
  HINT NOP — spec: "The NTL instructions do not change architectural
  state, nor do they alter the architecturally visible effects of the
  target instruction"). Two consequences differ from x86/AArch64:
  (1) no feature gating is needed — a HINT is an architectural NOP on
  every implementation, so RVV NT twins are enumerated unconditionally;
  (2) **no caller fence is needed** — the hint does not weaken store
  ordering the way `MOVNTDQ`/`STNP` do. Details and rejected
  alternatives (Zicbom, Zicboz, vendor extensions) in
  `docs/rvv_nt_prefetch.md`.
- **Recovery outputs:** in a recover op the NT-written outputs are the
  reconstructed *failed disks* (not the parities); the contract above
  applies unchanged — x86 callers `SFENCE` (AArch64: `DMB ISHST`)
  after the call before another agent reads the rebuilt disks, and the
  jitted code itself still emits no fence. NT recovery twins are
  covered like encode twins: `tests/test_nt.c` proves output equality
  vs the scalar reference and survivor-disks-untouched for rs(6,3)
  single/double/all-parity/data+parity failures, rs(12,3) and
  raid6(8) double failures, a 2 MiB large-length recovery (the NT
  regime), and prefetch×NT combination twins — all with the per-ISA
  fence-discipline shape checks (0 fences, ≥1 NT store marker,
  temporal twins clean).
- When benchmarking is disabled (`benchmark = 0`), NT twins are not
  generated at all: on the static model they would tie with their
  temporal originals and lose the deterministic tie-break, so
  `nt_threshold` is inert without benchmarking. (The RVV hint is not
  counted as a uop for the same reason: on implementations that ignore
  it, it executes as zero uops.)

## 10. Register allocation limits

- **Pointer-array reloads:** `in[]` input pointers are hoisted into GPRs
  on x86-64: the prologue loads the first 5 `in[]` pointers once into
  RSI/RDI/R12-R14 (all loop-invariant), and the vector body uses them
  directly instead of `mov r10, [r8+i*8]` per group iteration
  (`src/x86_64/jit.c`: `ptr_hoist_count`/`body_inptr`). R12-R14 are
  callee-saved (pushed/popped when used); RSI/RDI are caller-saved and
  dead in the vector body. P_DELTA is excluded (its `[old,new,oldpar]`
  pointer layout needs its own design); output pointers still reload
  per iteration (n_out loads, diminishing returns). The static model
  accounts the surviving per-iteration pointer loads (1 uop + 8 mem
  bytes each) for the hoisted kinds; P_DELTA's per-iteration pointer
  loads are unaccounted (0 uops / 0 mem in the model). Measured on RAID-6(5) Horner AVX-512+GFNI U=8: +2.2%
  cache-resident (core-bound; 5/6 paired runs), ~0% when DRAM-bound
  (pointer loads are L1 hits), never negative.
- **Table reloads:** P_MULACC/P_DELTA reload nibble/GFNI constants per
  unroll lane. The tables are invariant across `u`, but hoisting them
  competes with accumulator registers; the current loop order trades
  register pressure against data traffic. Candidate-specific budgeting
  could improve specific cases.

## 11. Software prefetch (evaluated, default off)

- `rj_options.prefetch_distance = 0` (default) disables software
  prefetch. Setting it > 0 adds a prefetch twin of every candidate
  (PREFETCHT0 for each input stream, this many bytes ahead of the
  current group, once per group iteration); the benchmark picks
  empirically whether the twin wins. The distance is a `size_t`.
  x86-64 consumes it as a 32-bit displacement (values ≥ 2³¹ are
  truncated — prefetch hints never fault, so the only effect of an
  absurd value is a useless prefetch). AArch64 validates it against
  the PRFM immediate range (0..32760, multiple of 8) and fails the
  twin build cleanly ("not encodable") on anything else — never
  silently wrapped (fixed 2026-10-03; previously distances ≥ 32768
  wrapped, e.g. 1 MB → offset 0). RVV validates 32..2016 (multiple
  of 32) and fails the build otherwise.
- **Evaluation (2026-10-01, AMD EPYC 9D25):** `tests/study_prefetch.c`
  compared distances 0/256/512/1024/2048/4096/8192 on RAID-6(8)
  AVX-512+GFNI for cold streaming (128MB, fresh buffers per rep) and
  in-cache (256KB) workloads. No distance showed a consistent win:
  small distances hurt (extra uops, cache pollution), large distances
  were neutral within noise. The hardware prefetcher already tracks
  the sequential streams well. Prefetch therefore stays opt-in and
  benchmark-gated; it is not enabled by default and no distance is
  recommended as "optimal" — measurement on the target CPU decides.
- **B4 auto-tuning rejection (2026-10-03, AMD EPYC 9D25, SOTA item
  18):** making the distance benchmark-picked per (ISA, plan
  kind, op) was evaluated and **rejected with evidence**. The
  decisive experiment ran `bench_select` itself (3 reps) on
  {0, 16K, 64K, 256K} for raid6(8) encode, raid5(8) encode and
  rs(6,3) double-failure recovery: raid6 → winners 64K, 0, 0;
  raid5 → 64K, 0, 0; rs(6,3) rec2 → 16K, 64K, 256K — a *different*
  distance every rep, i.e. no stable argmin (candidates tied within
  the noise floor). No distance clears the ">2% consistent win
  across 3 runs" gate on any class, so no auto-tuning ladder ships;
  the manual `prefetch_distance` opt-in above is unchanged. The
  per-stream variant was validated too (shared vs even-only /
  staggered / ramp schemes): no consistent win — all input streams
  advance in lockstep, so one shared distance is the right model;
  the experiment hook was removed. Re-test trigger: a CPU whose HW
  prefetcher does not track 7+ sequential streams, or a quiet-box
  measurement showing a stable >2% win. Design note and logs:
  `~/workspace/raidjit-tracks/Bx86/B4_DESIGN.md`,
  `B4_EVIDENCE.md`, `B4_exp_bs_run1.log`.
- **RISC-V:** the prefetch twin uses Zicbop `prefetch.r` (read
  prefetch; outputs are writes and are covered by `NTL.ALL`, so no
  `.w`/`.i`), one per input stream per group iteration,
  `prefetch_distance` bytes ahead. Enumerated only when the CPU reports
  Zicbop (`rj_cpu_info.rvv_zicbop`, detected from the `isa:` line of
  `/proc/cpuinfo` — no hwprobe key exposes it). Encodable range is
  32..2016 bytes in 32-byte steps (GAS-verified); other distances fail
  the twin build cleanly ("not encodable"), never silently rounded. A
  force-built candidate without Zicbop degrades to temporal (never
  wrong code). Like x86/AArch64: opt-in, benchmark-gated, no distance
  recommended — measurement decides. Full survey in
  `docs/rvv_nt_prefetch.md`.
- **AArch64:** the prefetch twin uses `PRFM PLDL1STRM` (streaming
  prefetch, the ARM counterpart of x86's `PREFETCHT0` for
  write-once/read-once RAID streams). Encodable distances are
  0..32760 bytes in 8-byte steps (PRFM imm12 × 8, GAS-verified);
  anything else fails the twin build cleanly ("not encodable"),
  never silently wrapped. Like x86, it is opt-in
  (`prefetch_distance > 0`) and benchmark-gated: the twin wins only if
  it measures faster on the target CPU. No AArch64-specific prefetch
  distance is recommended; the hardware prefetcher handles sequential
  RAID streams well, and measurement decides. The B4 past-end
  self-attack (`tests/test_b4_prefetch.c`) uses the encodable maximum
  32760 there, and scans the emitted code to prove every PRFM word
  carries the full offset.

## 12. AArch64 backend known issues (2026-10-01)
- **P_DELTA accumulator seeding (FIXED 2026-10-02):** the AArch64 JIT
  used to seed the P_DELTA accumulator from the output buffer (`out[j]`)
  instead of the old-parity input (`in[2*nc+j]`), producing wrong results
  for parity-update operations (`test_matrix`: `neon raid5(4) pdelta
  len=128 par0 mismatch [neon delta+nibble U=8]`). The codegen now loads
  the seed from `in[2*nc+j]` (`src/aarch64/jit.c`: "Seed accumulators
  from old parity"); verified by `tests/test_pdelta_edge.c` under QEMU
  aarch64 (out-of-place and in-place, all OK).
- **SVE under qemu-user:** `PR_SVE_GET_VL` reports 8 bytes while the true
  VL is 64 bytes. The backend disables SVE if the reported VL < 16 bytes
  and falls back to NEON. On real hardware with a correct `PR_SVE_GET_VL`,
  SVE/SVE2 candidates are generated and benchmarked normally.
- **Emitter:** all instruction encodings were verified against
  `aarch64-linux-gnu-as` (80 checks in `tests/test_aarch64_emit.c`, all
  passing). Three bugs were found and fixed during verification (NEON
  USHR base + shift encoding, SVE TBL base bit 17, NEON TBL base).
- **SVE `test_matrix` segfault under QEMU (NOT REPRODUCIBLE 2026-10-02;
  original attribution REFUTED):** a Makefile note claimed `test_matrix`
  segfaulted with SVE enabled under QEMU and excluded it from
  `test-qemu-aarch64-sve`. Re-verification on 2026-10-02 (qemu-aarch64
  8.2.2, `-cpu max,sve-default-vector-length=` 128/256/512) shows no
  segfault: the full matrix passes at all three VLs (128457 / 128233 /
  128233 runs, 0 failures; deterministic seed, so not flaky luck) — an
  independent peer reviewer reproduced the clean run (128457/0).
  The crash was first attributed to the `em_sve_lsr_unpred` emitter-base
  bug (base derived clearing only the imm field, corrupting zn), but a
  peer-review experiment **refuted** this: reintroducing the exact old
  base (`0x04209630`) into the current tree yields clean data-mismatch
  failures (`sve raid6(4) encode len=512 par1 mismatch`, exit 1) — zero
  crashes. Mechanism: on the SVE path `V(cg,r)=16+r`
  (`src/aarch64/jit.c:77`), so the corrupted zn' = 17|zn is always a
  valid z-register; the instruction decodes legally and only reads the
  wrong source. A wrong register cannot segfault. The emitter bug is
  real and its fix is real (pinned by the multi-register-combo shape
  tests at `tests/test_aarch64_emit.c:137-151`, among the binary's 90
  checks), but it never crashed anything — the AGENTS.md record itself
  says "caught by QEMU test *failure*".
  The historical segfault's **true root cause is unknown**. Leading
  hypothesis: a flawed observation — e.g. a misreported SIGILL from
  synthetic SVE variants executing on an SVE-less QEMU host (the
  host-gating in `tests/test_matrix.c:1457` exists precisely because
  un-gated synthetic SVE "crashes instead of testing"), a stale binary,
  or a QEMU flake. This fits the note's "other tests pass" (only
  `test_matrix` uses synthetic ISA variants). It cannot be confirmed:
  the tree has no git history to recover the segfault-era state.
  Residual risk is assessed low: 128k-run matrices x 3 VLs clean,
  guard-page over-read probes clean, and the reviewer found no remaining
  SVE fault path (runtime `cntb` VL, `p0=whilelo` set once, scalar tail,
  16-byte-aligned stack). `test_matrix` was restored to the SVE QEMU
  target (VL128/256; VL512 verified ad-hoc) on the strength of
  non-reproduction, not of a root cause.

## 13. P_XOR VPTERNLOGD twin (AVX-512, evaluated 2026-10-01)

- For AVX-512 `P_XOR` plans with `n_out >= 2`, the planner enumerates a
  VPTERNLOGD twin of every temporal candidate (desc tag `+ternlog`).
  Input columns are processed in pairs `(a,b) = (2i,2i+1)`; `in[a]` is
  loaded once into a shared temp register and each row using both
  columns folds `acc ^= in[a] ^ in[b]` into a single
  `VPTERNLOGD $0x96, in[b](mem), %zmm_t, %zmm_acc` (imm `0x96` is the
  three-way XOR truth table: output bits 1, 2, 4, 7 set, matching the
  Intel SDM; XOR's symmetry makes operand order irrelevant). Rows using
  one column of a pair take a plain XOR; a leftover odd column degrades
  to a plain step. The twin is benchmark-gated like every other twin: it
  is appended after its temporal original and `bench_select` picks the
  winner empirically on the target CPU. Never forced.
- **Gating:** no twin is enumerated for `n_out == 1` (e.g. RAID-5).
  With a single row the twin saves no uops (one uop per column either
  way) but pays `n_out*U` VPXOR-zero uops to clear the accumulators up
  front, so it is statically worse (plain 0.1406 uops/byte on a dense
  8+1 plan; the twin would add the vzero cost on top). With `n_out >= 2`
  the shared `in[a]` load amortizes across rows and each row saves one
  XOR uop per column pair (dense 8+2: 0.2500 vs 0.4062 uops/byte;
  dense 4+3: 0.2188 vs 0.2969; dense 12+2: 0.3438 vs 0.5938; all from
  the static model, U-invariant).
  **Pre-SOTA-batch figures:** the uops/byte numbers above are static-model
  values measured 2026-10-01, before the SOTA batch changed the planner
  (XOR-schedule twins, pruning, length dispatch). A 2026-10-02 review
  flagged them as stale against newer measurements (0.1992 vs 0.2148 on
  the re-measured config); re-measurement is owned by the A7
  reconsideration audit. Treat the numbers above as historical, not as
  current-model predictions.
- **Evaluation:** the emitter (`em_vpternlog_mem`) was cross-checked
  byte-for-byte against GAS on 24 randomized operand combinations
  (register classes, SIB scales, disp8/disp32 forms). Correctness is
  covered by `tests/test_pxor_ternlog.c` (differential vs the plain
  candidate and a scalar reference: npar 1..3, ndata 1..8, dense /
  checkerboard / seeded-random patterns (the random ones also force a
  zero column and a zero row), U 1..8, 13 main/tail-boundary lengths
  each, plus an NT-store twin and a prefetch twin) and by
  `tests/test_codegen_shape.c` (`test_tl`: exactly 3 VPTERNLOGD on a
  4x2 plan, none on the plain twin, planner gating to AVX-512 only,
  no twin for `n_out == 1`; red-green verified by mutation).
- **Selection:** the twin is appended after its temporal original.
  Under `heuristic_select` (benchmark off) the plain candidate wins
  outright on dense shapes (lower `max(uops/3.5, mem/64)`); under
  `bench_select` the noise tie-break prefers the lower uops/byte,
  which is the twin — so on a noisy host a within-noise "win" may
  reflect the tie-break rather than a real measurement. Only a clear
  measured win is trustworthy.
- **No empirical win was measurable on the dev machine** (shared 2-vCPU
  VM, load average ~16): the benchmark noise floor on identical
  candidates spans 0.6x-1.9x, so no twin-vs-plain comparison there can
  resolve a real difference. The twin therefore ships as a
  benchmark-gated candidate only; on a quiet machine `bench_select`
  will pick it exactly where it measures faster. Revisit with hardware
  numbers when available.

## 14. P_XOR XOR-schedule twin (cross-row CSE, evaluated 2026-10-02)

- For `P_XOR` plans, the planner builds a greedy cross-row CSE schedule
  at plan time (`src/xorsched.c`): identical rows are deduplicated, then
  the most co-occurring input pair across rows is repeatedly extracted
  into a temp `t_k = in_a ^ in_b` (flat pairs only, never nested). The
  schedule is attached to the plan (`plan_t.xor_sched`); a
  schedule-driven body exists in all three backends (desc tag `+xsched`).
  See `docs/xor_schedule.md` for the full design.
- **Gating:** the twin is enumerated per (ISA, U) base candidate only
  when `xorsched_version_uops` predicts a STRICT uop win under the
  emitter's own counting convention, and only if the extra temp register
  fits (`xorsched_need_regs`: +0 on x86-AVX, +1 on x86-SSE/AArch64/RVV).
  Never combined with `+ternlog`; NT/prefetch twins of it are produced
  by the existing twin loops. The benchmark picks empirically.
- **Dominance-pruning safety:** the strict gate means the naive twin can
  never dominate the scheduled twin on `(uops, mem)`; both versions'
  static fields come from the same emitter machinery. Verified by
  `tests/test_xorsched.c` (strict inequality on built versions).
- **Evaluation:** `tests/test_xorsched.c` (differential scheduled vs
  naive vs scalar reference: 6 topologies incl. duplicate/dead rows,
  13 lengths, U 1/2/4, all ISAs; model-exactness: predicted uops ==
  emitter uops on every backend, incl. the +nt/+pf combos; gate checks;
  red-green verified by mutation). AArch64/RVV executed under QEMU
  2026-10-02, all green: NEON 234 iterations (130 full checks), 3
  twins; SVE 468 × 2 VLs (260 full each), 3 twins each; RVV 234 × 3
  VLENs (156 full each), 4 twins each; 0 failures everywhere. The
  shipped-path e2e (`test_xorsched_e2e.c`) ran on every QEMU config,
  0 failures.
- **No conclusive empirical win measured:** on the dev machine (AMD
  EPYC 9D25, 2 vCPU), a sharing-heavy 20-in/4-out config shows a 20.1%
  static uop win (0.3711 vs 0.4648 uops/B, model-exact, AVX-512 U=4).
  A quiet-system duel (taskset-pinned, 12 runs/arm through
  `bench_select`, `tools/duel_xorsched.c`): at 64 KiB the twin is
  +0.8% median (83.66 vs 82.98 GB/s) with overlapping IQRs — faster in
  10/12 paired runs but within noise; at 1 MiB the shared VM's
  memory-bandwidth noise dominates (4x within-arm spread) and the duel
  is inconclusive. The earlier loaded-system "slower or tied"
  observation does not reproduce under pinning. Honest status: no
  conclusive measurement either way; the twin ships benchmark-gated
  and `bench_select` picks it exactly where it measures faster.
  Full detail in `docs/xor_schedule.md` §7.

## 15. XOR array codes (RDP/EVENODD/Liberation, merged 2026-10-02)

- `rj_topo_rdp(p)` / `rj_topo_evenodd(p)` / `rj_topo_liberation(k)` are
  binary (XOR-only) MDS array codes. They lower to expanded row-model
  topologies: with `k` data disks of `R` rows, `ndata = k*R` row-disks
  and `npar = 2*R` parity rows; `in[d*R + r]` is row `r` of data disk
  `d`, `out[r]` / `out[R + r]` are the P / Q parity rows. Every parity
  row is a pure XOR of input rows, so these compile through the existing
  `P_XOR` plan path. `rj_xorcode_rows(t)` reports `R` (0 for non-XOR
  topologies). Full construction: `docs/xor_codes.md`.
- **Validity envelopes vs constructor caps (deliberate):** the
  `xorcode_*_valid()` predicates in `src/xorcode.h` admit a wider
  envelope than the constructors — e.g. `xorcode_rdp_valid(257) == 1`
  (257 is prime) but `rj_topo_rdp(257)` returns NULL via the 256-disk
  (`RJ_MAX_DISKS`) cap. The predicates answer the math question ("is
  this a valid code parameter?"); the constructors additionally enforce
  the engineering envelope. Documented here so the asymmetry is not
  mistaken for a bug.
- **Default-table status:** `rj_topo_raid6(ndata)` dispatches on
  `docs/xor_code_default.md` (machine-readable winner table; the C table
  in `src/xorcode.c` is transcribed from it and `test_xorcode.c` checks
  they agree). The table's status line records the latest merged-tree
  re-run outcome (see `docs/xor_code_default.md`); while it is all-RS,
  `rj_topo_raid6()` returns RS topologies exactly as before. If a
  re-run flips an entry, the return format changes to the expanded row
  model for that `ndata` (a semantic API consequence; the README
  row-model section covers the caller contract).
- **Recovery sampling rationale:** `tests/test_xorcode.c` recovers every
  2-data-disk pair exhaustively through the same `solve_recovery` the
  JIT uses (plus data+P, data+Q, P+Q); the parity-involved extras beyond
  that are spot-sampled by design, and `check_mds` is exhaustive over
  all whole-disk pairs. Rationale: the parity rows are pure XORs of
  input rows, so the solver path for parity-involved failures is the
  same linear machinery exercised by the exhaustive data-disk pairs;
  the sampled extras guard against wiring mistakes in the expanded
  index mapping, not against solver unsoundness.
