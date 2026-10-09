# WASM Explorer Resume State (worktree ~/workspace/raidjit-wasm, branch wasm-explorer)

## Verified good (do not redo)
- x86: 33/33 parity (clean worktree driver, -DRJ_WASM deterministic const gaps)
- riscv: 11/11 parity (first clean run)
- aarch64: 9/10 parity (predtail case failed; fix committed, module rebuild pending)

## Commits on wasm-explorer
- 22f98cc: RJ_WASM const-pool gap memset + plan.c ifdefs (from main tree)
- run_parity.sh: -DRJ_WASM for native driver (deterministic const gaps)
- tools/gen_live_kernels_page.py (from main tree)
- rjw_api.c: mask stage-2 tail flags for planid_cand_valid (predtail pinnable)

## Fixes applied
1. run_parity.sh: native driver built with -DRJ_WASM (fixes flaky const-blob gaps)
2. rjw_api.c: predtail/notail/maskedtail masked for planid_cand_valid (both in
   build_pinned_cand and the U-scan loop); feature_gates + backend validate them
3. dis_api.c: Capstone 6 CS_OPT_MEM setup before cs_open (fixes CS_ERR_MEMSETUP);
   verified natively: cs_open OK, disassembles correctly

## In progress
- disasm.js rebuild (with mem_setup fix) — running
- aarch64.js rebuild (with predtail fix) — pending after disasm

## Remaining
1. aarch64 parity 10/10 (after rebuild)
2. disasm.js smoke test (after rebuild)
3. S9.4 disasm differential vs objdump
4. Live page: python3 tools/gen_live_kernels_page.py
5. S9.3 native no-regression (make kernels before/after)
6. S9.5 native suite, S9.6 page smoke, S9.7 peer reviews, S9.8 self-attack, S9.9 docs

## Key lessons
- Parity driver MUST be rebuilt from worktree sources; cached main-tree driver is stale
- Capstone 6.0.0-Alpha11 requires cs_option(NULL, CS_OPT_MEM, ...) before cs_open
