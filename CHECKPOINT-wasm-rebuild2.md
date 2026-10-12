# CHECKPOINT — raidjit WASM explorer rebuild + republish lane (wasm-rebuild2)

Branch: `wt/raidjit/wasm-rebuild2` (worktree `~/workspace/worktrees/raidjit/wasm-rebuild2`)
Base: crc-fast `4e63a82`. Registry id: `raidjit-wasm-rebuild2`.
Started: 2026-10-12 ~12:05 CEST.

## Why
Published gh-pages explorer (df08fef8, built from main@7203cf1) predates three
integrations that change user-visible behavior: L5 remat-phantom twin effect
markers (08e7fc2+1459ada — remat on all-trivial plans now reports taken instead
of phantom), L2 const-pool zero-fill (15d1611 — invisible, gap bytes only), and
the plan.c RJW_AARCH64/RJW_RISCV dispatch-clause restore (4e63a82). Standing
order: WASM build + page must reflect current main after such integrations.

## Build tree layout (this worktree)
- `src/`, `include/` — crc-fast @4e63a82, untouched.
- `tools/wasm/` glue copied read-only from branch `wasm-explorer` @ec2349e
  (the maintained versions — main's `tools/wasm/Makefile.wasm` and
  `tools/gen_live_kernels_page.py` are stale leftovers: main's Makefile lacks
  `ui_meta.c`/`_rjw_meta_schema,_rjw_meta_valid`/WASM_BIGINT and references
  `rjw_api.c` which is not on main; main's page generator (535 lines) predates
  the metadata-driven rewrite (898 lines on wasm-explorer)):
  `Makefile.wasm`, `rjw_api.c`, `rjw_wasm.h`, `dis_api.c`,
  `cpu_wasm_{x86,aarch64,riscv}.c`, `vendor/cJSON.{c,h}`, `parity/` harness
  (no `work/` artifacts), `tools/gen_live_kernels_page.py` (metadata-driven:
  validity queried from the library via `_rjw_meta_valid`, never reimplemented).
- `tools/wasm/vendor/capstone/` — full 6.0.0-Alpha11 tree, copied from the
  wasm-republish lane worktree. Build dependency only, NOT committed.
- `tools/wasm/SPEC.md` — main's version (from HEAD; not part of glue).

## In-main prerequisites (verified, no lane patches needed)
1. `src/plan.c` lines 838/898 carry the `#if defined(__aarch64__) ||
   defined(RJW_AARCH64)` / `#elif defined(__riscv) || defined(RJW_RISCV)`
   dispatch clauses (commit 4e63a82). The previous lane's worktree patch is
   superseded — nothing to carry.
2. L2 const-pool zero-fill (15d1611) is in main, unconditional, in all three
   backends' `pool_get` (`src/{x86_64,aarch64,riscv}/jit.c`). The previous
   lane's `#ifdef RJ_WASM` memset is superseded — NOT re-added.
3. `cand_t` unchanged between 7203cf1 and 4e63a82 (only `crc_fold`, already
   re-synced in the glue, below).

## Glue re-sync applied (from the wasm-republish lane, verified identical base)
`tools/wasm/rjw_api.c` — BUG-COMBO-STATIC-F16 crc_fold re-sync (4 hunks):
parse `crc_fold` in `build_pinned_cand` (fail closed); `feature_gates` rejects
`crc_fold=1` ("crc_fold needs the fused CRC path (want_crc), not available in
the explorer"); BUG-COMBO-E50 noop guard gains `|| pinned.crc_fold`; result
`twins[]` list gains the `crc_fold` entry (exhaustive axis report).
`tools/wasm/parity/run_parity.sh` — COMMON gains `src/crc_fast.c`,
`src/ui_meta.c`, `src/ports.c`; x86 driver gains `src/x86_64/crc_fast.c` +
`src/x86_64/crc_fold.c`; aarch64/riscv drivers gain `src/<arch>/crc_fast.c`.
`tools/wasm/parity/parity_matrix.json` — `x86/twin/xorbit` expectation
`ok`→`fail` (c1199a9 bans xorbit on x86; both sides fail identically).

## Build
`make -f tools/wasm/Makefile.wasm`, emcc 6.0.11 (`~/tools/emsdk`), from
crc-fast@4e63a82. No errors (only the known WASM_BIGINT deprecation warning).
- `tools/wasm/out/raidjit_x86.js` — 246,171 B
- `tools/wasm/out/raidjit_aarch64.js` — 213,904 B
- `tools/wasm/out/raidjit_riscv.js` — 217,167 B
- `tools/wasm/out/disasm.js` — 3,831,326 B

## Parity
`tools/wasm/parity/run_parity.sh` → **54 passed, 0 failed**
(x86 33/33, aarch64 10/10, riscv 11/11). Native drivers via gcc +
aarch64/riscv cross compilers; qemu-user 8.2.2 already present (no reinstall).

## Page
`python3 tools/gen_live_kernels_page.py` →
`tools/wasm/out/raidjit_kernels.html` (4,546,849 B,
md5 `b67474c30cff8d0f95d29c9564ae9e1c`).

## Headless verification (node; `.tmp/headless_verify.js`)
79/79 checks pass. The page is one giant `<script>` block (4 bundles +
PAGE_JS); the script extracts each bundle from the GENERATED HTML, splits on
the factory-assignment markers, and byte-checks each against the `out/` file
(modulo the generator's `</script`→`<\/script` escaping — all 4 identical),
then instantiates the page's exact factories (`RaidjitX86`, `RaidjitAArch64`,
`RaidjitRiscv`, `Disasm`) in `vm` contexts and drives them through the page's
exact `compileKernel`/`disassemble` wrappers:
- 4/4 factories instantiate with all exports present.
- ctor×ISA matrix: 13 ctors × 3 ISAs = 39/39 compile `ok:true` with non-empty
  code bytes (page's per-ctor defaults; targets avx512-gfni / sve2 / rvv).
- disasm round-trip per ISA: kernel code bytes → `_dj_disasm` → non-empty
  JSON insn array, all 3 ISAs.
- `node --check` on the extracted PAGE_JS: OK.
- L5 sanity (built x86 module): pinned `remat=1` on a P_HORNER plan (custom
  Vandermonde rs(6,3), nibble, U=8 — the L5 test's topology) → `ok:true`,
  `twins=["remat"]`, strategy=horner: the twin is reported APPLIED, not a
  phantom. Reachability note (proven from `plan_select_strategy`): an
  all-trivial *planner* plan is always P_XOR, and the planid gate correctly
  rejects remat there ("remat is P_HORNER-only"); the white-box all-trivial
  P_HORNER case is covered natively by `test_remat` (passed at L5 integration).
  The module-level observable is: remat requested → reported applied.
- Phantom fail-closed control: pinned `ternlog=1` on tripless raid1 plan →
  clean `ok:false` "ternlog twin requested but no VPTERNLOGD emitted (plan has
  no foldable triples)". The F16 postcondition is intact after L5.
- crc_fold guard: pinned `crc_fold=1` → clean "crc_fold needs the fused CRC
  path (want_crc), not available in the explorer".

## Publish
Copied html to `~/workspace/raidjit-wasm/tools/wasm/out/raidjit_kernels.html`;
ran `python3 ~/workspace/skills/github-pat/bin/publish_explorer.py`:
gh-pages **df08fef8 → 14b68006** (2026-10-12T10:35:40Z), index.html blob
`39b4ac90`, verified via GitHub API byte-identical to the local page.
Pages API 403 = designed behavior (GitHub Pages enablement is his manual step;
not attempted).

## Local copy
`~/workspace/your_files/raidjit/raidjit_kernels.html` refreshed with the
regenerated page (md5 match).

## Follow-ups / nits (not fixed in this lane)
- `tools/wasm/parity/run_parity.sh` header comment says `-DRJ_WASM` "activates
  only the const-pool gap memset" — stale since L2 made the memset
  unconditional in main. Harmless (deterministic either way); comment-only fix
  for the harness owner.
- Main's `tools/wasm/Makefile.wasm` + `tools/gen_live_kernels_page.py` are
  stale leftovers (see "Build tree layout"); the wasm-explorer branch carries
  the maintained versions. If main ever re-adopts these files, they need a
  re-sync from the branch, not the reverse.
- FLAGGED: the L2 integration commit 15d1611 accidentally committed lane
  working files to crc-fast: root `CHECKPOINT.md` (the L2 lane's, not this
  lane's), `l2_repro/` (incl. a 293 KB binary `repro_l2` + logs), `Makefile`
  hunk. The coordinator should clean these from the branch before they ride
  to main.

## Committed on this branch
`tools/wasm/out/{raidjit_x86,raidjit_aarch64,raidjit_riscv,disasm}.js`,
`tools/wasm/out/raidjit_kernels.html`, this CHECKPOINT-wasm-rebuild2.md.
NOT committed: `tools/wasm/vendor/capstone/`, `tools/wasm/parity/work/`,
`.tmp/` (build logs, verification scripts), glue files (reproducible from
wasm-explorer@ec2349e + the re-sync/harness hunks documented above).
Nothing landed on main; no other pushes (gh-pages publish only, via the
authorized API script).
