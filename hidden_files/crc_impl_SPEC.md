# CRC fast paths: implementation spec (Option A interleaved fold + Option C standalone API)

Owner approval: 2026-10-08. Standing constraints: NO timing benchmarks, NO perf
guard, no push, all work under `nice`, non-CRC codegen byte-identical
(baseline: `hidden_files/crc_fold_baseline_sha256.txt`, 151 sections).

Design inputs (read before implementing):
- `hidden_files/crc_optimum_research/FINAL_SYNTHESIS.md` — per-ISA optimum
  table, Option A construction, per-µarch verdict table. THE spec basis.
- `hidden_files/design_crc_fold.md` §11 — verified PCLMULQDQ fold formulation,
  constants K1=0xae689191 / K2=0xccaa009e, tail handling, bit-exact vs zlib.
- `hidden_files/crc_optimum_research/{verify_crc,fold_verify,compose_verify}.py`
  — reuse the verification approach (Python models as oracles where useful).

Contract: `~/workspace/skills/world-class-implementation/SKILL.md` (all 8 done
gates). AGENTS.md lesson (load-bearing): any new `cand_t` axis is pinned in
`prune.c::twin_dominates` IN THE SAME CHANGE.

Branch plan: Phase 1 on branch `crc-fast` (from main), merge at Phase 1
sign-off after all gates. Phase 2 on branch `crc-fold` (from new main), merge
at Phase 2 sign-off. Never commit directly to main; workers `git pull --rebase`
before pushing to the branch.

WASM-track coordination (sibling track, active): changes here are additive to
`src/`. New `rj_cpu_info` fields must not break `tools/wasm` (its
`cpu_wasm_*.c` stubs construct the struct — verify the wasm build still
compiles; do NOT edit `tools/wasm/*` yourself, flag breakage to the
coordinator). The new planner axis is source-level; if `tools/wasm/rjw_api.c`
enumerates twin flags explicitly it will want the new flag — flag to
coordinator, do not touch.

---

## Phase 1 — Option C: standalone fast-CRC API (do first; zero codegen risk)

### R1 Public API (immutable)
`uint32_t rj_crc32_fast(uint32_t crc, const void *data, size_t len);`
declared in `include/raidjit.h`, documented there. Semantics: IDENTICAL
chaining contract to `rj_crc32_update` (takes running CRC, returns updated
running CRC; init/final via existing `rj_crc32_init`/`rj_crc32_final`). Must
be a drop-in replacement: for all inputs,
`rj_crc32_fast(c,d,n) == rj_crc32_update(c,d,n)`.

### R2 Dispatch
- First-call selection, thread-safe (match existing `pthread_once` style in
  `src/crc.c`), cached function pointer. Unknown/absent features degrade to
  a slower CORRECT variant — never wrong, never SIGILL.
- x86-64 chain (best first):
  1. `vpclmulqdq && avx512f` → Cawley-style fusion (scalar crc32q lanes +
     VPCLMULQDQ vector lanes), per FINAL_SYNTHESIS §Phase 1 x86 rows.
     (Review R1-6: if the fusion's lane management needs byte shuffles,
     `avx512bw` — already detected — joins this condition; worker
     justifies the exact flag set from the source.)
  2. `vpclmulqdq && !avx512f` → scalar+vector fusion with 256-bit
     VPCLMULQDQ lanes (Zen3 row: 31.76 GB/s reported, 2.3x over the
     128-bit fold). (Review R1-2/Finding 13.)
  3. `pclmulqdq` (no vpclmulqdq) → 128-bit fold, zlib-ng/Chromium
     structure (synthesis table optimum for Intel SKL→CLX: 24.05 GB/s
     reported; AMD Zen: 12.9 GB/s). (Finding 13: the table — not fusion —
     is the arbiter here.)
  4. `sse42` → hw `crc32q` N-way interleave (22.5–27.3 GB/s reported).
  5. else → existing slice-by-8 (`rj_crc32_update` body, reused not
     duplicated).
- AArch64 chain:
  1. `pmull && sha3` → Cawley fusion (`crc32_armv8_pmull_eor3.c` structure).
     Lane structure follows the synthesis PER-CORE variants (M1 v9s3x2e_s3
     vs N1 v3s4x2e_v2 — Finding 13): select via MIDR implementer
     (0x61 → M1 variant, else N1 variant; same parse as R10) or document
     why a single variant suffices (reviewer adjudicates).
  2. `crc32` → scalar `crc32x` N-way interleave (the worker justifies the
     lane count from the N1/M1 numbers in the synthesis). NOTE (review R1-1):
     scalar goes BEFORE the pure PMULL fold because of the documented N1
     inversion (pure PMULL fold 7.3 B/c < scalar crc32x 7.9 B/c on N1);
     Apple cores always take tier 1, so no Apple regression.
  3. `pmull` (no CRC32 hw) → pure PMULL fold (dougallj structure).
  4. else → slice-by-8.
- RISC-V chain:
  1. `rvv_zvbc` → Zvbc vector fold (SEW=64, streams packed as elements).
  2. `zbc` (new flag) → scalar clmul fold (zlib-ng `crc32_zbc.c` structure).
  3. else → slice-by-8.
- WASM / unknown: slice-by-8 (the wasm stubs leave new flags 0).
  (Finding 17c: under `__EMSCRIPTEN__` the top-level `rj_crc32_fast`
  calls `rj_crc32_update` directly — no separate WASM backend entry.)

### R2.5 Pinned variant entry points (cross-worker contract — exact names)
Each backend compiles its file unconditionally (variant bodies use function
`target` attributes where needed so the TU builds on any host). Declarations
live in the backend's private header (`src/<isa>/crc_fast.h`), non-static:

x86-64 (`src/x86_64/crc_fast.h`):
  `uint32_t rj_crc32_fast_x86(uint32_t crc, const void *d, size_t n);`
    best available x86 variant (internal runtime selection).
  `uint32_t rj_crc32_fast_x86_vpclmul(uint32_t crc, const void *d, size_t n);`
  `uint32_t rj_crc32_fast_x86_pclmul(uint32_t crc, const void *d, size_t n);`
  `uint32_t rj_crc32_fast_x86_crc32hw(uint32_t crc, const void *d, size_t n);`
  `int rj_crc32_fast_x86_class(void);`  /* 0=scalar 1=crc32hw 2=pclmul128
    3=vpclmul (either fusion flavor: rung 1 or rung 2 — the
    rj_crc32_fast_x86_vpclmul entry point selects internally) */

AArch64 (`src/aarch64/crc_fast.h`): same shape —
  `rj_crc32_fast_arm`, `rj_crc32_fast_arm_fusion`,
  `rj_crc32_fast_arm_crc32hw`, `rj_crc32_fast_arm_pmull`,
  `rj_crc32_fast_arm_class` (0=scalar 1=crc32hw 2=pmull 3=fusion).

RISC-V (`src/riscv/crc_fast.h`): same shape —
  `rj_crc32_fast_riscv`, `rj_crc32_fast_riscv_zvbc`,
  `rj_crc32_fast_riscv_zbc`, `rj_crc32_fast_riscv_class`
  (0=scalar 1=zbc 2=zvbc).

Top-level (`src/crc_fast.c`, W4): `rj_crc32_fast` picks the backend by
compile-time ISA and calls its best-variant entry; the `*_class`
functions let tests assert selection matches detected features. W1–W3
implement EXACTLY these names; W4 tests against them.

### R3 New `rj_cpu_info` flags (all in `include/raidjit.h`, all documented)
- x86: `pclmulqdq` (CPUID.1:ECX[1]), `vpclmulqdq` (CPUID.7:ECX[10]),
  `sse42` (CPUID.1:ECX[20]). XCR0-gate AVX-512-dependent claims exactly as
  the existing code does; pclmulqdq/vpclmulqdq/sse42 need no XCR0 state.
- AArch64: `crc32` (HWCAP_CRC32 = 1UL<<7), `pmull` (HWCAP_PMULL = 1UL<<4),
  `sha3` (HWCAP_SHA3 = 1UL<<17). All in AT_HWCAP (not HWCAP2) — verified
  against `/usr/aarch64-linux-gnu/include/asm/hwcap.h`. For portability
  also probe HWCAP2_SHA3 under `#ifdef`, mirroring zlib-ng
  `arch/arm/arm_features.c` (review R1-8). Follow the existing
  local-`#ifndef`-define pattern in `src/aarch64/cpu.c`.
- RISC-V: `zbc` (scalar Zbc carryless multiply; detection follows the
  existing `rvv_zvbc` pattern in `src/riscv/cpu.c`).
- Every flag has a test-only override path if the file already has one;
  otherwise tests exercise real detection + forced-fallback unit tests of
  each variant entry point directly (entry points are non-static for tests,
  declared in a private header).

### R4 Bit-exactness (hard gate)
Every variant is bit-exact vs zlib `crc32` AND vs `rj_crc32_update_scalar`
on: lengths 0..256 exhaustive, all tail classes around 16B/64B boundaries,
lengths {1000, 10000, 1<<20}, unaligned buffers (all misalignments 0..15,
head and tail), ≥3 PRNG seeds. KAT: `CRC32("123456789") == 0xCBF43926`
(already asserted in `tests/test_crc.c`; keep).
Red-green: each test fails when run against a deliberately broken constant
(flip K1/K2 low bit; flip one fusion lane-shift constant) — or an
equivalent mutation per variant for constant-free variants (drop an
interleave lane, corrupt the table polynomial) (Finding 16). The test
harness supports an env/flag-injected corruption, or a separate
mutation-run documents the kill. A test that cannot fail proves nothing.
Split-point chaining (review R1-3, load-bearing for the fused driver's
per-tile accumulation): `f(c,A||B) == f(f(c,A),B)` at every split offset
mod 64 over a range of lengths and seeds — a variant correct on whole
buffers but wrong across chunk boundaries must fail here.

### R5 Fuzz / adversarial
- len 0 with NULL and non-NULL data; len 0 with garbage crc.
- Unaligned data at every offset mod 16, lengths crossing page-ish
  boundaries (use mmap'd guard pages: 1 byte before unmapped page must not
  over-read — vector variants must prove no over-read via masking/tail
  handling, tested with ASan + guard pages).
- Concurrent calls from N threads (dispatch init race).
- `size_t` extremes: no pointer wraparound and no 32-bit truncation in
  block-count arithmetic — test lengths around 2^32 (2^32±16) for
  truncation (review R1-9; a SIZE_MAX execution test would hang, so the
  requirement is the truncation test plus a code-review assertion of no
  wraparound). Document the edge-case contract in `raidjit.h` (public API
  edge — Finding 17b): chunk arithmetic is wraparound-free by
  construction; absurd lengths are a caller bug, handled without UB.

### R6 Integration into the driver
`src/fused.c` switches its post-tile accumulation from `rj_crc32_update` to
`rj_crc32_fast`. This is the immediate payoff (replaces the 1–3 GB/s scalar
path everywhere — Finding 17a: cite the synthesis's figure, not "<10") with zero codegen change. `tests/test_crc.c` /
`tests/test_fused.c` extended accordingly; existing expectations unchanged.
`rj_crc32()` (one-shot, currently caller-free) routes through
`rj_crc32_fast` as well (init + fast + final) — review R1-5, API coherence.

### R7 File layout (disjoint per worker; no cross-edits)
- `src/x86_64/crc_fast.c` + private header: x86 variants + x86 selector.
- `src/aarch64/crc_fast.c` + private header: ARM variants + selector.
- `src/riscv/crc_fast.c` + private header: RISC-V variants + selector.
- `src/crc_fast.c`: `rj_cpu_info` flag additions live in
  `include/raidjit.h` (owned by THIS file's worker), top-level
  `rj_crc32_fast` + dispatch glue + per-ISA selector interface definition.
- `tests/test_crc_fast.c`: the R4/R5 matrix.
- Internal contract (fixed by this spec, not negotiable downstream):
  each backend exposes
  `uint32_t rj_crc32_fast_<isa>(uint32_t crc, const void *d, size_t n);`
  (its BEST available variant, selected internally by the backend) and the
  top-level dispatch chooses the backend by compile-time ISA. Forcing a
  specific fallback in tests goes through the backend's own test hook.
- Phase 1 uses C+intrinsics (`<wmmintrin.h>` for PCLMULQDQ/VPCLMULQDQ,
  `<arm_neon.h>` for PMULL/CRC32/EOR3; RISC-V uses vector intrinsics only
  if the toolchain supports them, else plain C). Never raw `asm`:
  intrinsics keep the compiler's ordering guarantees; bit-exactness is
  proven by R4 testing, not by construction.

### R8 Docs (same change)
- `include/raidjit.h` documents `rj_crc32_fast` + the dispatch order +
  the analytic-only performance note (never present analytic GB/s as
  measured).
- `tools/kernels-README.md` or new `docs/crc.md`: per-ISA variant table,
  fallback chains, where each variant wins. Honest about what is analytic.

---

## Phase 2 — Option A: interleaved CRC-fold planner twin (after Phase 1 sign-off)

### R9 Planner plumbing
- `cand_t.crc_fold` (int; 1 = this candidate folds per-stream CRC-32 of
  its inputs and computed parities into the kernel).
- `plan_t.want_crc` (int). DERIVED, not a new public option (Finding 2):
  `rj_fuse_compile` scans its op sequence; any ENCODE op targeted by a CRC
  op gets want_crc=1. Both `rj_compile_crc` ([ENCODE,CRC]) and a direct
  `rj_fuse_compile([ENCODE,CRC])` behave identically — no new public
  field, no behavioral asymmetry. Only CRC-targeted ops (k == crc_tgt)
  get it; other ops in the sequence never do.
- `want_crc` is set on the plan post-`plan_build` at EVERY
  plan-construction site: the normal path and the A5 survivor path in
  api.c (Finding 2c). `plan_build` takes no options, so the setter is a
  small helper (e.g. `plan_set_want_crc`) — the "set everywhere or
  nowhere" invariant stays reviewable in one place.
- `prune.c::twin_dominates` pins `crc_fold` IN THE SAME CHANGE (AGENTS.md
  lesson). `tests/test_prune.c` extended: the 176-check suite still passes
  identically (behavior-preserving for existing axes) + new pin tests
  following the `unit_twin_axes_pinned` pattern (test_prune.c:1269).
- Candidate buffers (Finding 3): api.c:218 warns "a future axis must
  increase the buffer or fail closed." The fold axis doubles the worst
  case under want_crc && viable. REQUIRED, both: (a) increase the
  `cands[512]` buffers at all six call sites (api.c ×4, rjw_api.c,
  gen_kernels.c) with headroom justified by the worst-case count;
  (b) `plan_candidates` returns -1 with err on would-be truncation instead
  of silently truncating (fail closed).
- Plan IDs (Finding 4): new twin token for crc_fold in planid.c — follow
  the existing token list/parse/validate pattern (suggested token
  "crcfold"). Without it, fold/non-fold candidates share an ID →
  `rj_compile_pinned` ambiguity and `rj_list_variants` duplicates.
  Required: token + parse + shape validation + synth_desc parity
  (api.c:1221) + visibility in `rj_list_variants`. The wasm JSON surface
  is coordinator-owned (flagged already — do not touch).
- nt/prefetch twins (Finding 5): YES, enumerated for fold candidates
  (uniformity — plan.h: twins are made "of every candidate"). R12's
  register accounting covers the fold state under nt/prefetch shapes, and
  on nt=1 fold twins the fold-state stores are TEMPORAL (states are
  re-read next tile; NT stores would be pessimal) while parity data
  stores stay NT.
- Skip fold twins when n_out==0 (Finding 14): a fold twin on a no-parity
  encode interleaves with nothing — driver-level CRC suffices. Documented.

### R10 µarch gate (conservative; analytic verdicts from FINAL_SYNTHESIS)
New PURE function `int rj_crc_fold_viable(rj_uarch_t u,
const rj_cpu_info *cpu, int isa)` — semantics fixed:
- x86-64: 1 iff `u ∈ {RJ_UARCH_INTEL_SAPPHIRERAPIDS,
  RJ_UARCH_INTEL_ICELAKE, RJ_UARCH_AMD_ZEN4}` AND `cpu->vpclmulqdq` AND
  `cpu->avx512f`. Zen5 excluded (no data — fail closed, documented).
  SKX/Zen3 excluded (synthesis: 2–4x, NOT free).
- AArch64: 1 iff `u == RJ_UARCH_ARM_APPLE` (NEW enumerator — Finding 6)
  AND `cpu->pmull && cpu->sha3`.
- RISC-V: 0 always (synthesis: throughput unknown → conservative off,
  documented).
- WASM/other: 0.
- PURITY (Finding 7): the /proc/cpuinfo → tag mapping lives in
  `rj_cpu_uarch()` EXACTLY ONCE (cached, like the x86 CPUID path);
  `viable()` is pure in (tag, flags, isa) and never parses cpuinfo
  itself. Tests inject via `rj_uarch_override` (exists, src/cpu.h:39) +
  flag injection — so the ARM fold is differential-testable under QEMU
  aarch64 (`-cpu max`; worker verifies max exposes PMULL+SHA3 HWCAPs
  first, else the test reports SKIP with a reason, never a silent pass).
- ARM tag work (Finding 6): add `RJ_UARCH_ARM_APPLE` to `rj_uarch_t`;
  detect in `src/aarch64/cpu.c`'s `rj_cpu_uarch()` via the MIDR
  implementer parse (0x61; part numbers documented from Asahi Linux or
  equivalent, cited in a code comment); add the cost-model weights-table
  entry (values justified from the synthesis M1 numbers, or explicitly
  marked conservative — reviewer checks).
- `plan_candidates` enumerates `crc_fold=1` twins ONLY when
  `want_crc && rj_crc_fold_viable(...)`. Otherwise the CRC request keeps
  today's driver-level behavior (now accelerated by Phase 1's
  `rj_crc32_fast`).

### R11 Emitter construction (per FINAL_SYNTHESIS Option A)
- x86-64 (the primary target): for each stream, 1 zmm of fold state = 4
  independent 128-bit lanes; lane k folds every 4th 16B chunk. Per 64B of
  stream data already in registers: data-XOR fused via VPTERNLOGQ, then
  2× VPCLMULQDQ. Per-stripe lane reduction at epilogue; tail via §11
  `x^{-8r}`/`x^{64-8r}` constants; final `red(S)^0xFFFFFFFF`.
  Enumerated only for AVX-512 candidates (the fold needs zmm).
- AArch64 (Apple-gated): vector-only PMULL fold (NOT the scalar+vector
  fusion — synthesis: the scalar unit needs GPRs; extracting from vector
  regs costs more than it saves). EOR3 fuses the data XOR where SHA3.
- New emitters derived per AGENTS.md lesson: clear ALL operand fields when
  deriving bases; GAS cross-check across randomized operands (even/odd,
  low/high regs) BEFORE use (binutils ≥ 2.31 for VPCLMULQDQ — worker
  verifies; Finding 17e); exact-equality uop-model test per (ISA,U,
  topology) for the new twin (the "twin cost models in lockstep" lesson).

#### R11a Fold-state kernel/driver ABI (Finding 1, BLOCKER)
`rj_fn` is fixed as `(in[], out[], len)` — it cannot carry per-call state,
and changing it would break every kernel, caller, and the WASM surface.
Fold kernels therefore use an EXTENDED signature; non-fold kernels keep
`rj_fn` untouched:
  `typedef void (*rj_fn_crc)(const void *in[], void *out[], size_t len,
                             void *crc_states);`
  (name TBD by worker; this shape or equivalent.)
- `crc_states`: driver-owned, PER-CALL (stack array like the driver's
  `acc[]`, never per-handle) → concurrent `rj_fused_call` is safe.
  Points to `nstreams * state_bytes` bytes; `state_bytes` comes from the
  kernel's info struct (new field `rj_info.crc_state_bytes`, 0 for
  non-fold kernels). The state FORMAT is kernel-private — the driver
  never interprets it.
- The fused driver: after selection, if the winning candidate has
  crc_fold=1 it calls through `rj_fn_crc` with its per-call state array;
  otherwise the existing `rj_fn` path (byte-identical behavior).
- `rj_compile_pinned` on a fold token (Finding 4) yields an `rj_fn_crc`
  — documented: the caller must use the extended signature.
  `rj_list_variants` shows the token.
- Length-dispatch composition (Finding 8): tail twins built from a fold
  winner PRESERVE the extended ABI. `notail` is ALLOWED on fold winners
  (the dispatcher guarantees len%group==0 → no tail bytes exist → the
  fold stays correct). `predtail`/`maskedtail` are EXCLUDED on fold
  winners: predicated/masked lanes read as zero and folding zeros
  corrupts the CRC (the 4f1e69c pipe+predtail class). The fold kernel
  handles all lengths with its own scalar tail (R11's tail story).

#### R11b Ordering (Finding 12)
R12 is decided BEFORE R11: R11's emitter construction and the
`plan_need_regs` formulas follow the R12 decision, not the reverse.

### R12 Register pressure (the flagged risk) — DECIDED BEFORE R11
14 streams × 1 zmm fold state + the encode's ~20 live zmm exceeds 32.
The worker picks ONE mitigation with RECORDED arithmetic: (a) planner
reserves registers: `crc_fold` twins use reduced unroll U so total live ≤
the derived budget (quantify the U cost); (b) spill fold states to L1
(quantify load/store uop cost per byte); (c) hybrid. The arithmetic starts
from current `plan_need_regs` + exactly nstreams×1 zmm; the old "≤28"
budget must be DERIVED (32 architectural zmm minus explicitly accounted
headroom), not asserted. Composes with nt/prefetch twins per the R9
Finding-5 rule, or the worker records why a twin is excluded (reviewer
adjudicates). `plan_need_regs`-style accounting covers the fold state
exactly (the I8 lesson: formulas exact for the new shape). The reviewer
checks the arithmetic.

### R13 Differential proof (the real correctness gate; no timing allowed)
Randomized topologies (raid5, raid6, rs(6,3), rs(10,4), custom), lengths
1..2^20 (Finding 11: len=0 EXCLUDED — `fused_driver_call` asserts len>=1,
so fold kernels are unreachable via any public path at len 0; documented),
all tail classes, unaligned, random data: CRCs from the interleaved kernel
MUST be bit-identical to `rj_crc32_fast` (Phase 1) and to zlib, for every
stream (inputs AND parities). Any mismatch = ship-blocker.
SCOPE (Finding 17d): ENCODE only — `rj_compile_crc` emits [ENCODE,CRC];
a fused CRC op targeting PARITY_UPDATE is out of scope (stated).
EXERCISE GUARANTEE (Finding 10): tests use benchmark=0 AND bypass
selection — enumerate candidates, build each `crc_fold=1` version directly
via `jit_build_version`, assert the fold path executed (via the Finding-4
plan-ID token), then differential-test. This closes both the
heuristic-may-not-pick hole and timing inside correctness tests.
Runs natively AND under QEMU for aarch64/riscv64 (where the gate is off →
tests prove the driver-level path, not the fold).

### R14 Byte-identical non-CRC proof (HARD gate)
After Phase 2, regenerate `make kernels` and diff every section against
`hidden_files/crc_fold_baseline_sha256.txt`: all 151 sections identical.
Name every doc artifact in the proof: `raidjit_kernels.md`,
`raidjit_kernels.html`, the manifest. R16's added CRC sections are
explicitly allow-listed.
ANTI-LEAK TEST (Finding 9 — the kernels doc deliberately contains no CRC
sections, so the regen diff can NEVER observe a want_crc leak; R14 as
written would pass vacuously): a dedicated test on a fixed topology
corpus × ISAs asserts that with want_crc=0, ZERO `crc_fold=1` candidates
are enumerated (via the plan-ID token's absence in `rj_list_variants`)
AND emitted bytes are identical pre/post Phase 2. A/B via two worktrees,
never git-stash.
Any difference in a non-CRC kernel = ship-blocker, no exceptions.

### R15 Twin audit
`tests/test_twin_audit.c` extended: the reachable sweep includes
`want_crc=1` × viable-uarch combos. Mechanics (Finding 15):
(a) the sweep's synthetic CPU (`fill_cpu`) sets the new R3 flags
(vpclmulqdq; pmull/sha3/crc32 on aarch64) AND the ARM tag — else
viable() never fires and the sweep tests nothing new;
(b) new sweep counts recorded AS EXPECTED-COUNT ASSERTIONS in the test
file (not stdout);
(c) the (crc_fold=1)/(crc_fold=0) pin test follows `unit_twin_axes_pinned`
(test_prune.c:1269);
(d) the sweep covers the Finding-8 cross product: fold winner ×
(notail allowed / predtail+maskedtail excluded) — the exclusion is
asserted, not just documented;
(e) gen_kernels.c's R16 recipe uses the same trick (want_crc=1 via the
post-plan_build setter + uarch override + synthetic CPU flags) — blessed
here explicitly since plan_t is internal.
0 aborts. New counts recorded.

### R16 Kernels doc
Representative CRC-fold kernel sections are added to the doc
(`tools/gen_kernels.c` + regen), generated with the uarch override forced
to a viable µarch, clearly marked with the µarch gate and the analytic-only
note. SCOPE (review R1-7): x86_64 sections are REQUIRED (the dev host has
VPCLMULQDQ — verified — so execution-verification works); AArch64 sections
only if the generator can execute-verify under QEMU aarch64, else omit with
a "Deliberately not represented" note in `tools/kernels-README.md`.
If the worker finds a principled reason to exclude even the x86_64 sections,
the reason is written in `tools/kernels-README.md` — exclusion needs
reviewer sign-off, it is not the default.

### R17 Docs (same change)
`docs/crc.md` (new) or `tools/kernels-README.md`: the per-µarch dispatch
table from the synthesis, WHERE IT IS NOT FREE (SKX/CLX, Zen2/3, N1, RV),
the analytic-only disclaimer, and the fallback behavior. No contradiction
with the synthesis doc.

---

## Test & gate summary (both phases)

- `make test` (native), `make test-asan` (ASan/UBSan — note the A5 lesson:
  verify EVERY build config a done-gate names), QEMU aarch64 + riscv64
  suites green. (`command -v qemu-aarch64` first; never claim absence.)
- Mutation/red-green evidence for every new test (the Phase 1 R4
  corruption harness; Phase 2 differential test must fail if the fold
  constants are corrupted).
- Self-attack pass per SKILL.md (malformed topologies, 0-length stripes,
  resource limits, concurrent compiles) — report what was tried.
- Independent peer review: ≥2 reviewers per phase, none of whom implemented
  the reviewed code; all findings remediated and re-tested.
- Pre-delivery: `grep -rn "TODO\|FIXME\|XXX\|not implemented" src/`
  clean; docs consistent (spec ↔ code ↔ README).

## Assumptions (declared for veto)
1. `rj_crc32_fast` becomes public API in `include/raidjit.h` (unreleased
   project; API may change freely).
2. ARM M1-gate via /proc/cpuinfo implementer 0x61 (all Apple silicon ≥ M1
   class); part-number granularity NOT required.
3. Zen5 excluded from the interleave gate until throughput data exists.
4. Compile-time `benchmark=1` selection machinery is UNCHANGED; the µarch
   gate controls enumeration only. Tests use `benchmark=0` for
   determinism unless a test specifically targets selection.
5. The "almost free" claim stays analytic until the owner resumes
   benchmarks; no number in code comments, docs, or commit messages may
   present an analytic/reported figure as measured.
