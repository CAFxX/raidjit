# SPEC: raidjit WASM live kernel explorer

Status: DRAFT (pending peer review) — 2026-10-08
Owner request: compile raidjit + a disassembler to WebAssembly so a web page
compiles ANY option combination live in-browser, replacing the static
150-kernel explorer page.

## S1. Goals (numbered, checkable)

1. A web page `raidjit_kernels.html` that, for an arbitrary user-chosen option
   set (target/ISA+features, topology+params, operation+params, multiply
   method, unroll, twin flags, codegen options), compiles the kernel
   **live in the browser** via a WASM module and shows the config table +
   disassembly.
2. The WASM build emits **byte-identical machine code** to the native build
   for the same option combination (automated parity matrix, §S9).
3. The native build's emitted code is **unchanged** by this work: `make kernels`
   output byte-identical before/after (proof required, §S9).
4. The page works from `file://` (no server, no threads, no
   SharedArrayBuffer, no COOP/COEP).
5. Optional A/B comparison mode: two independent option sets compiled live,
   shown side by side, deep-linkable.
6. The static 150-kernel HTML explorer is **superseded** by this page (same
   deliverable filename). `tools/gen_kernels_page.py` and the `.md` reference
   stay in the repo; the `.md`'s 150 kernels remain a verification corpus.

Non-goals: executing generated code in-browser; benchmarking in-browser;
interleaved CRC (cancelled track — `crc_fold` stays hidden); changing planner
heuristics; pushing anything anywhere.

## S2. Standing constraints (from the owner)

- C1. No benchmarks, no perf guard. (Benchmarks postponed 2026-10-06.)
- C2. No push. All work local.
- C3. Do NOT restart the interleaved-CRC redesign; build on the tree as-is.
- C4. Additive only: any new API layer must not alter native codegen. New
      `#ifdef RJ_WASM` blocks are allowed in existing files ONLY if the
      native build never defines the macro; the before/after `make kernels`
      proof (§S9.3) is the enforcement.
- C5. Follow the world-class-implementation contract: spec → red-green tests →
      combination tests → independent peer reviews → self-attack → docs in the
      same change.

## S3. Architecture

Three compile modules (mirrors the native per-ARCH build exactly — this is
deliberate: it avoids touching `plan.c`'s `#if defined(__aarch64__)/__riscv`
ISA-enumeration branches, which would risk native codegen):

- `raidjit_x86.wasm` — COMMON_SRCS + src/x86_64/{emit,jit}.c + wasm cpu stub
- `raidjit_aarch64.wasm` — COMMON_SRCS + src/aarch64/{emit,jit}.c + wasm cpu stub
- `raidjit_riscv.wasm` — COMMON_SRCS + src/riscv/{emit,jit}.c + wasm cpu stub

One disassembler module:

- `disasm.wasm` — Capstone 6.0.0-Alpha11+ built from source with emcc,
  `CAPSTONE_X86_SUPPORT + CAPSTONE_AARCH64_SUPPORT + CAPSTONE_RISCV_SUPPORT`
  only. (Stable v5 has NO RVV — the v6 line is mandatory. Verified in source:
  `RISCV_INS_VCLMUL_VV/VX`, `VCLMULH_VV/VX` present; upstream MC tests cover
  `vclmul.vv`.)

The page lazy-selects the compile module by ISA and uses the single disasm
module for all three. All modules built with `-sSINGLE_FILE=1` (wasm inlined
as base64 in the JS glue) and inlined into one self-contained HTML by the
page generator, so `file://` works with zero network fetches.

### S3.1 Why not one module

`plan_candidates` (src/plan.c:722-947) enumerates ISA candidates under
compile-time `#if defined(__aarch64__) / #elif defined(__riscv) / #else`
(x86-64). A single wasm32 module would need a source refactor of the
planner or `-D__aarch64__` hacks that perturb emscripten's libc headers.
Per-ISA modules reuse the exact native build shape; the byte-identity
argument is then "same sources, different compiler", which the parity
matrix (§S9) checks directly.

## S4. Compile-only API (C, exported from each module)

New file `tools/wasm/rjw_api.c` (additive; not linked into native builds):

```c
/* Compiles ONE kernel from a JSON option document. Never executes code,
 * never benchmarks, never probes hardware. Deterministic: identical JSON
 * in -> identical bytes + metadata out. */
char *rjw_compile(const char *json_opts);   /* caller frees (free()) */

/* Access to the last compiled code/const blobs (valid until next call). */
const uint8_t *rjw_code_bytes(size_t *len_out);
const uint8_t *rjw_const_bytes(size_t *len_out);
```

Semantics of `rjw_compile`:

1. Parse JSON (bundled single-file JSON parser, e.g. a vendored minimal
   parser under tools/wasm/vendor/ — no new runtime deps).
2. Build `rj_topology*` from `topology` (constructor name + params; custom
   matrix as hex string). Build `rj_op` from `op` (encode | recover{failed[]}
   | parity_update{changed[]}).
3. Fill `rj_cpu_info` from the `target` feature bundle (§S5.1). Never call
   the native `rj_cpu_detect` (replaced by per-arch stub, §S6).
4. `plan_build` (or `plan_build_recover_rows` when `op=recover` and an
   explicit equation-row subset is given).
5. Candidate: if `cand` is `"auto"` → `plan_candidates` filtered by ISA, then
   the static `heuristic_select` winner (bench.c) — the same rule native
   uses at `benchmark=0`. Else construct `cand_t` directly from the pinned
   axes (the `gen_kernels.c` pattern) and validate ISA-applicability with the
   same eligibility predicates the planner uses; on violation return
   `{ok:false, err}` (never wrong code).
6. `jit_build_version` → on success return metadata JSON; on failure
   `{ok:false, err}` with the backend's error string.
7. **Skip** `validate_versions` (it executes the kernel — bench.c:205,234)
   and all of stage 2 (`o.benchmark` is forced 0). Consequence, recorded in
   metadata: `"exec_validated": false`. Correctness proof is the byte-identity
   parity matrix (§S9), since the native build CI executes these kernels.

Result JSON fields: `ok`, `err?`, `desc`, `plan_id` (rj_plan_id), `target`,
`topology`, `operation`, `strategy` (derived, read-only), `isa`, `mul`,
`unroll`, `lmul`, `twins[]`, `code_size`, `const_size`, `uops_per_byte`,
`mem_per_byte` (labeled static-model estimates), `code_sha256`,
`const_sha256`, `isa_marker`, `selection` (`heuristic`|`pinned`),
`exec_validated:false`, `nominal_vl_bytes`.

Disassembler module API (`tools/wasm/dis_api.c` over Capstone):

```c
/* arch: "x86_64"|"aarch64"|"riscv64". Returns JSON array of
 * {addr, size, bytes(hex), mnemonic, op_str}. x86 uses CS_OPT_SYNTAX_ATT;
 * riscv opens CS_ARCH_RISCV with CS_MODE_RISCV64|CS_MODE_RISCV_V. */
char *dj_disasm(const char *arch, const uint8_t *bytes, size_t len,
                uint64_t base_addr);
```

JS-side normalization (~30 lines, tested): strip objdump size suffixes
(`vpxord`→`vpxor`), comma spacing, hex case — so output is comparable to the
`.md` reference. `tidy_disasm.py`'s lone-byte folding is NOT needed
(Capstone never splits instructions across lines). Byte equality is the
invariant, not asm text.

## S5. Option surface

### S5.1 Target bundles (synthetic `rj_cpu_info`, the `fill_cpu` pattern)

Presets: `avx512-gfni`, `avx2`, `sse`, `sve2`, `neon`, `rvv`, `rvv-zvbc32e`,
`rvv-zicbop`. Expert mode exposes the codegen-gating flags individually:
`gfni vbmi avx512bw sve2 rvv_zvbc32e rvv_zicbop` (+ `sve_vl`/`rvv_vlen`,
default 16; VL-agnostic codegen — affects U selection and nominal metadata,
not emitted bytes). Flags with no backend consumers today (`avx512vl`,
`bitalg`, `rvv_zvbb`, `rvv_zicond`, `p_ext`) are accepted and ignored, and
documented as such. `ISA_PEXT` is excluded (no backend). `crc_fold` is
hidden (no backend consumption; pending owner decision).

### S5.2 Topology

All `rj_topo_*` constructors with their params and validity envelopes
(raid0/raid1/raid5/raid6/raid6_rs/rdp/evenodd/liberation/rs/lrc/lrc_azure/
lrc_optimal/custom-hex-matrix). Invalid params → constructor-NULL →
`{ok:false, err}`. XOR-code whole-disk selection maps to expanded row
indices via `rj_xorcode_rows()`.

### S5.3 Operation

`encode` | `recover{failed:[...], rows?:[...]}` | `parity_update{changed:[...]}`.
`recover` uses A5 survivor machinery (codegen-only, no execution — verified
in recon). A4 closed-form is JIT-internal and bit-identical — not an axis.

### S5.4 Candidate axes (pinned or `auto`)

`mul` (nibble|gfni|zvbc), `U` (1..8), `lmul` (rvv only), `nt`, `prefetch_dist`,
`ternlog`, `remat`, `notail`, `predtail`, `maskedtail`, `permi2b`, `xorsched`,
`xorbit`, `mshift`, `pipe`. `strategy` is **derived read-only**
(`plan_select_strategy`; no override exists — forcing it would be a codegen
change, out of scope).

Body-twin mutual exclusion (enforced client-side by greying out, server-side
by the backend's asserts + `{ok:false,err}` backstop): `xorsched`/`xorbit`/
`permi2b`/`mshift`/`pipe`/`remat` pairwise exclusive; `ternlog` composes only
with `mshift`; tail twins (`notail`/`predtail`/`maskedtail`) mutually
exclusive; `maskedtail` needs P_XOR+AVX512; `predtail` SVE-only and not with
`pipe`.

### S5.5 Codegen options

`loop_align` (auto|16|32|64), `emit_endbr`, `emit_shadow_stack`. Inert at
compile-only: `benchmark`, `nt_threshold`, `prefetch_distance` (page drives
`cand_t` directly), `want_crc` (internal-only).

### S5.6 uarch

`rj_cpu_uarch()` returns `RJ_UARCH_UNKNOWN` under wasm32 (its cpuid asm is
`#if defined(__x86_64__)`-guarded out), and peer review 2026-10-08 verified
`rj_uarch_current` has zero callers in the tree — the uarch tag is fully
inert in the compile-only path. No preset needed; the wasm build is
deterministic without it.

## S6. Native-touchpoint stubs (all `#ifdef RJ_WASM`, wasm build only)

Per recon (exact lines in the report):

1. **cpu.c replaced** per arch by `tools/wasm/cpu_wasm_<arch>.c` providing
   `rj_cpu_detect` that fills the struct from a preset bundle passed in
   (never cpuid/getauxval/`/proc/cpuinfo`/prctl/ecall).
2. **mmap/mprotect pivot** in each `src/<arch>/jit.c` `jit_build_version` /
   `jit_free_version`: under `RJ_WASM`, after `em_resolve`, copy
   `cg.em.len` bytes to a fresh `malloc` block; `v->mem`/`v->memsz`/`v->fn`
   point at it; free path uses `free`. The `__builtin___clear_cache`
   (aarch64, riscv only) and `sysconf` lines live inside the replaced
   block, so they never compile under emcc.
2b. **VL/VLEN reads**: `src/aarch64/jit.c:1566` (`__asm__ volatile("cntb")`
   overriding `cg.vl` from hardware) and `src/riscv/jit.c:151-155,1402`
   (`read_vlenb()` CSR read into `cg.vlenb`) are inline asm OUTSIDE the
   S6.2 replaced block and would hard-fail under emcc. Each gets its own
   `#ifdef RJ_WASM` that sources the value from the preset bundle instead
   (`cpu->sve_vl` / `cpu->rvv_vlen`, §S5.1). (Found by peer review
   2026-10-08.)
3. **pthread**: emscripten's musl libc already provides working
   `pthread_mutex_*` in single-threaded builds without `-pthread` — define
   NO stubs (duplicates would risk link errors). pthread users in the
   compile path: none contend (mutexes guard the batched/CRC runtime
   trampoline pools only). NOTE: `-sSINGLE_FILE=1` is incompatible with
   `-pthread` by design — the no-threads requirement is enforced by the
   link, which fails loudly if `-pthread` is ever added. (Found by peer
   review 2026-10-08.)
4. `getenv` calls are inert (NULL under emscripten) — no change.
5. `rjw_api.c` forces `o.benchmark=0`, skips `validate_versions`, skips stage
   2, never calls `rj_init` (prefetch probe).

Module source lists (the Makefile's COMMON_SRCS, auditable here):

```
COMMON_WASM_SRCS = src/gf.c src/plan.c src/topo.c src/bench.c src/ref.c \
  src/api.c src/prune.c src/dispatch.c src/xorsched.c src/matrix_search.c \
  src/xorcode.c src/survivor.c src/prefetch_calib.c src/planid.c src/crc.c \
  src/striping.c src/api_crc.c src/fused.c src/api_batched.c
  # PLUS tools/wasm/rjw_api.c, tools/wasm/cpu_wasm_<arch>.c
  # MINUS src/<arch>/cpu.c (replaced by the stub, §S6.1)
ARCH_WASM_SRCS(x86_64)  = src/x86_64/emit.c src/x86_64/jit.c
ARCH_WASM_SRCS(aarch64) = src/aarch64/emit.c src/aarch64/jit.c
ARCH_WASM_SRCS(riscv64) = src/riscv/emit.c src/riscv/jit.c
```

Build flags (both sides, pinned for determinism): `-O2 -std=c11
-D_POSIX_C_SOURCE=200809L -ffp-contract=off`. The last flag closes a
latent hazard found by peer review 2026-10-08: `plan_uarch_cost`
(plan.c) computes `sqrt(sqrt(...))`, an fma-contractable pattern feeding
the `1e-9` tie-breaks in `heuristic_select`; today neither side contracts,
and the flag pins it permanently.

Determinism (verified in recon): position-independent codegen on all ISAs,
no `rand`/`time`/`getpid` in src, total-order qsort comparators, FNV-1a-64
plan IDs over raw matrix bytes, strict `< sb - 1e-9` heuristic tie-breaks.
ASLR/pointers never enter bytes or metadata.

## S7. Page design (`tools/gen_live_kernels_page.py` → `raidjit_kernels.html`)

- Single self-contained HTML (emscripten `-sSINGLE_FILE=1` outputs inlined
  as base64 by the generator; pin `-sSINGLE_FILE_BINARY_ENCODE=0` since
  emscripten ≥ 4.0.18 changed the default embedding to a UTF-8 string —
  base64 is the battle-tested path; declare `<meta charset="utf-8">`
  regardless). Works from `file://` with no server, no fetch, no threads.
- **Single view (default):** option panel (target preset + expert flags,
  topology + params, operation + params, candidate axes with auto/pin,
  codegen options), Compile button (also auto-compiles on change, debounced),
  config table (same fields as the `.md` + plan ID + `exec_validated:false`
  note + static-model labeling), disassembly pane.
- **A/B mode (toggle):** two independent option panels side by side; each
  pane compiles independently; config tables aligned row-by-row; disassembly
  panes side by side (diff view optional, implementation detail).
- **Deep links:** `#a=<base64url(JSON option set)>` for single view,
  `#a=...&b=...` for comparison. The JSON option set is the full `rjw_compile`
  input document (§S4), so links are exact and forward-compatible.
- Invalid combinations: greyed out via the eligibility matrix (§S5.4);
  backend `{ok:false, err}` surfaced verbatim as the backstop.
- Search box over the option space is out of scope (the space is now
  infinite); keep a "copy link" button per pane.
- The page must never execute generated bytes (no `fn` call, no executable
  memory); the API has no execution entry point by construction.

## S8. File layout

```
tools/wasm/
  SPEC.md              (this file)
  README.md            (build/run/test instructions; page usage)
  Makefile.wasm        (builds the 4 modules; SINGLE_FILE=1; no native impact)
  rjw_api.c            (compile-only JSON API, §S4; pthread no-op stubs)
  cpu_wasm_x86.c / cpu_wasm_aarch64.c / cpu_wasm_riscv.c  (rj_cpu_detect stubs)
  dis_api.c            (Capstone wrapper, §S4)
  vendor/              (capstone-6.0.0-Alpha11+ source or submodule ref;
                        minimal JSON parser)
  parity/              (byte-identity harness: parity_driver.c,
                        parity_matrix.json, run_parity.sh, node runner)
tools/gen_live_kernels_page.py   (embeds the 4 SINGLE_FILE js outputs into
                                  raidjit_kernels.html)
~/workspace/your_files/raidjit/raidjit_kernels.html  (the deliverable; mirrors
                                  the live page, replacing the static one)
```

Native tree changes: only `#ifdef RJ_WASM` blocks in the three `jit.c`
files (§S6.2). Everything else additive under `tools/wasm/`.

## S9. Tests & acceptance gates

- S9.1 **Feasibility spike (kill-or-commit):** minimal emcc build of
  planner+emitter for x86-64, JS-callable, returning bytes for one topology.
  Kill triggers: compile path fundamentally requires native execution or
  un-portable syscalls beyond the §S6 stub list. (Recon says no; spike proves.)
  **VERDICT 2026-10-08: PASS.** Scratch-tree spike
  (`hidden_files/wasm-spike-tree/`): emcc 6.0.11 built COMMON_SRCS +
  x86_64 emit/jit with `-DRJ_WASM` (malloc pivot in jit.c, synthetic-CPU
  stub replacing cpu.c), ran under node for raid5(4) encode on the
  avx512-gfni bundle. WASM code sha256
  `ce8faa39fe23325e3d9c09f64f1e20559a870a574bc9730e51dbff8e8df9ff92`
  == native reference byte-for-byte; desc/sizes/uops/winner identical.
  No kill trigger fired. (Spike used `-O0` for the final byte comparison;
  `-O2` compile also proven; emitted bytes are independent of host -O.)
- S9.2 **Parity matrix (the master gate):** `tools/wasm/parity/` — native
  per-ARCH driver emits `{options_hash, code_sha256, const_sha256, metadata}`
  JSON lines; node runner feeds the same option documents to the wasm
  modules and compares. Matrix covers: 3 ISAs × {raid5, raid6, raid6_rs,
  rdp, rs(6,3), lrc} × {encode, recover, parity_update} × {auto} ∪ {each
  forcible twin × its ISA} ∪ {custom matrix → P_MULACC} ∪ {U pins 1..8} ∪
  {lmul 1,2,4,8} ∪ {prefetch on/off} ∪ {loop_align, endbr}. Every byte must
  match; every metadata field must match except none (all deterministic).
  Red-green: the harness must fail if a single byte differs (verified by
  flipping one emitter byte in a scratch build).
- S9.3 **Native no-regression proof:** `make kernels` before/after the change;
  the 150 `.md` sections' disassembly + sha256 must be byte-identical.
  (Guards C4.)
- S9.4 **Disassembly differential:** decode every kernel in the parity corpus
  with `disasm.wasm`; diff normalized output against the objdump reference
  (`.md` + `tidy_disasm.py`). Zero unknown/` .byte` decodes; zero
  mis-decodes on the emitted subset. Kill triggers per recon: per-ISA
  fallback (riscv/aarch64 hand-written decoder; x86 iced-wasm).
- S9.5 **Native suite green:** `make test` + `make test-asan` (correctness
  only); `test-qemu-aarch64`/`test-qemu-riscv64` if quick.
- S9.6 **Page smoke test:** node-based DOM-stub test (as done for the static
  page) exercising single + A/B + deep-link round-trip; plus a real-browser
  check delegated to the parent (subagents have no live browser).
- S9.7 **Peer reviews:** ≥2 independent reviewers on the spec (this
  document), then ≥2 on the implementation; all findings remediated and
  re-tested.
- S9.8 **Self-attack:** malformed JSON, unknown constructors, out-of-envelope
  params, huge topologies (ndata=256), failed-disk index garbage, custom
  matrix garbage, 10k-option fuzz loop in node (no crash/hang/OOM beyond a
  documented cap; document the cap).
- S9.9 **Docs in the same change:** `tools/wasm/README.md`,
  `tools/kernels-README.md` update (static page superseded; `.md` archived
  reference), regen wiring (`finalize_kernels.sh` / `mirror_your_files.sh`
  mirror the live page, not the static one).

Acceptance = S9.2 + S9.3 + S9.4 + S9.5 + S9.6 + S9.7 + S9.8 + S9.9 all green.
No benchmarks anywhere in this track (C1).

## S10. Rollout / regen wiring

- `make kernels` continues to regenerate the `.md` (archived reference).
- New target `make kernels-live` (or a flag) builds the wasm modules +
  regenerates `raidjit_kernels.html` via `tools/gen_live_kernels_page.py`.
- `tools/mirror_your_files.sh` mirrors the live `raidjit_kernels.html`
  (replacing the static file at the same path).
- `tools/gen_kernels_page.py` stays in the repo, runnable, but is removed
  from the default regen pipeline (superseded deliverable).

## S11. Open decisions (coordinator calls, recorded)

- D1. `rjw_compile` JSON schema versioning: `"v":1` field, additive evolution.
- D2. Default U/mul in the page: `auto` (heuristic winner) — pinned axes are
      opt-in. (Matches gen_kernels.c's winner-then-twins structure.)
- D3. Eligibility matrix lives client-side in JS (grey-out) with backend
      `err` as backstop (not server-side only).
- D4. Deep-link encoding: base64url(JSON option document) — exact, no
      separate compact scheme to maintain.
- D5. `exec_validated:false` shown in the page's config table with a
      one-line explanation (byte-identity vs the CI-validated native build
      is the correctness argument).
