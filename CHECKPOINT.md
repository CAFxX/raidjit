# CHECKPOINT — raidjit WASM explorer rebuild + republish lane

Branch: `wt/raidjit/wasm-republish` (worktree `~/workspace/worktrees/raidjit/wasm-republish`)
Base: main `7203cf1` (crc-fast). Registry id: `raidjit-wasm-republish`.
Started: 2026-10-12 ~07:05 CEST (05:05 UTC).

## Why
Published gh-pages explorer (commit a24625a1f, 2026-10-11 07:20Z) predates the
phantom-twin fixes (1179cee etc.), the blurb/tooltip rewrite, and the crc_fold
axis. Standing order: WASM build + page must reflect current main after
integrations changing user-visible options.

## Build tree layout (this worktree)
- `src/`, `include/` — main @7203cf1 (untouched; the phantom-twin fixes,
  STATIC-F03/04/05/F16, plan_need_regs arity, GFNI ban, all present).
- `tools/wasm/` glue copied read-only from branch `wasm-explorer` @ec2349e:
  `Makefile.wasm` (the maintained one — main's references deleted
  `src/crc_fast.c`), `rjw_api.c`, `rjw_wasm.h`, `dis_api.c`,
  `cpu_wasm_{x86,aarch64,riscv}.c`, `vendor/cJSON.{c,h}`,
  `parity/` harness (no `work/` artifacts), `tools/gen_live_kernels_page.py`
  (metadata-driven page: blurb rewrite + reviewer-3 fixes + stale-link
  normalization + disasm table).
- `tools/wasm/vendor/capstone/` — full 6.0.0-Alpha11 tree (the old worktree's
  `vendor/capstone` was an empty husk; real tree was `capstone-6.0.0-Alpha11`).
  NOT for commit (build dependency only).

## Re-sync found needed (BUG-COMBO-STATIC-F16)
Main's `cand_t` grew `crc_fold` (REDUX interleaved CRC). The explorer glue
(ec2349e) predates it: rjw_api.c parses every other axis but not crc_fold, and
carries an explicit "needs a re-sync from main" comment. The twin is NOT in
`rj_twin_id_t` (hidden from the page schema by design — want_crc is
internal-only via `rj_compile_want_crc`, never set on the WASM path; native
pinned path rejects it via the membership check against plan_candidates).
Re-sync applied in `tools/wasm/rjw_api.c`:
1. `build_pinned_cand`: parse `crc_fold` (fail closed, never silently drop).
2. `feature_gates`: reject `crc_fold=1` with a clean error (explorer has no
   fused-CRC path; native would fail the membership check).
3. BUG-COMBO-E50 noop guard: added `|| pinned.crc_fold` (mirrors api.c).
4. Result `twins[]` list: added the crc_fold entry (exhaustive axis report).

## Re-sync found needed (WASM-only ifdefs lost from main)
Main's src lost two WASM-build ifdefs the explorer branch carried:
1. `src/plan.c` `plan_candidates`: the old WASM src had
   `#if defined(__aarch64__) || defined(RJW_AARCH64)` /
   `#elif defined(__riscv) || defined(RJW_RISCV)`; main has only the
   compiler-arch macros. Without them, the aarch64/riscv WASM modules
   (built by emcc, where __aarch64__/__riscv are NOT defined) fall into
   the x86-64 enumeration branch and find zero candidates. Restored both
   clauses in the lane worktree (no-ops for native builds).
2. `src/{x86_64,aarch64,riscv}/jit.c` `pool_get`: the old WASM src had an
   `#ifdef RJ_WASM` memset zeroing the const-pool alignment gap
   (22f98cc); main lost it, causing nondeterministic const-blob bytes
   (WASM vs native sha mismatch with identical code). Restored in all
   three backends (WASM-only; native output byte-identical).
Both are no-ops for native builds and should go back to main (flagged in
the handoff); the lane carries them so the modules build correctly.

## Harness fixes (tools/wasm/parity/run_parity.sh)
The harness predates the metadata-driven page and the CRC/ports work:
1. Added `src/ui_meta.c` to COMMON (rjw_api.c calls rj_ui_twin_* now).
2. Added `src/crc_fast.c` to COMMON + per-arch `src/<arch>/crc_fast.c`
   (+ `src/x86_64/crc_fold.c` for x86) to the driver builds (fused.c refs).
3. Added `src/ports.c` to COMMON (plan.c's plan_uarch_cost refs; dead code
   on the benchmark=0 path but gcc needs the symbols to link).
4. `parity_matrix.json`: `x86/twin/xorbit` expectation `ok`→`fail`
   (c1199a9 deliberately bans xorbit on x86, BUG_t1_529; both sides now
   fail identically with the clean error — parity holds, expectation was
   stale).

## Status / next (updated 2026-10-12 ~08:55 UTC, post-reboot resume)
- [x] Build tree assembled in lane worktree; all Makefile.wasm srcs verified present.
- [x] crc_fold re-sync edits (compile-checked via the built modules).
- [x] `make -f tools/wasm/Makefile.wasm` — all 3 ISA modules built with emcc
      6.0.11 pre-reboot (x86 246,087 B; aarch64 214,772 B; riscv 217,399 B).
      Build artifacts survived the reboot on workspace disk; base 7203cf1
      still current for user-visible options (main moved to a14ecae, docs-only:
      README/docs link fixes — no rebuild needed).
- [x] Parity: `parity/run_parity.sh` → x86 33/33, aarch64 10/10, riscv 11/11
      (0 failed). Cross drivers were cached pre-reboot; qemu-user reinstalled
      post-reboot to run them.
- [x] crc_fold rejection verified live in the built x86 module: pinned
      `crc_fold:1` → clean error "crc_fold needs the fused CRC path
      (want_crc), not available in the explorer"; `auto` path never sets it.
- [x] Regenerate page: `python3 tools/gen_live_kernels_page.py
      tools/wasm/out/raidjit_kernels.html` — byte-identical to pre-reboot
      artifact (deterministic; md5 b9127b9660f45b85b86a2a1223227d7c).
- [x] Headless-Chromium verification: no usable chromium on the fresh VM
      (noble's chromium/chromium-browser are snap stubs). Ran a
      node-based headless-equivalent instead (/tmp/headless_verify.js):
      the page's exact module factories (RaidjitX86/RaidjitAArch64/
      RaidjitRiscv/Disasm) instantiated from the generated HTML's embedded
      bundles; ctor×ISA matrix (13 ctors × 3 ISAs) all compile OK;
      disasm round-trip (512 B → 101 insns); PAGE_JS syntax-checked with
      `node --check`. 40/40 passed.
- [x] Copied html to `~/workspace/raidjit-wasm/tools/wasm/out/raidjit_kernels.html`
      and ran `python3 ~/workspace/skills/github-pat/bin/publish_explorer.py`:
      gh-pages a24625a1 → df08fef8 (2026-10-12T08:51:48Z), index.html blob
      a006b9ca, verified via GitHub API. Pages stays disabled (his manual
      step; the script's 403 path is the designed behavior).
- [ ] Commit `tools/wasm/out/{raidjit_x86,raidjit_aarch64,raidjit_riscv,disasm}.js`,
      `tools/wasm/out/raidjit_kernels.html`, glue files, WASM-only src
      ifdefs, CHECKPOINT.md to the lane branch with explicit pathspecs.
      Do NOT commit `vendor/capstone`, `parity/work/`, or build logs.
- [ ] Registry: heartbeat hourly + after each major step; mark done at the end.

## Resume notes (if daemon-restarted)
- Worktree and branch persist; CHECKPOINT.md is here. Glue is already copied.
- Re-run backoff check before any heavy build: `~/workspace/system/bin/load-shed.sh backoff`.
- Do NOT touch `~/workspace/raidjit` (main tree) except read-only; do NOT touch
  `~/workspace/raidjit-wasm` (old wasm-explorer worktree) except reading.
- Never push to CAFxX/raidjit main; gh-pages publish via the API script only.
- Open item: the WASM-only ifdefs restored in src/plan.c and
  src/{x86_64,aarch64,riscv}/jit.c are no-ops for native builds and should
  go back to main (see "WASM-only ifdefs lost from main" above).
