# WASM Orphan Triage Report

**Date:** 2026-10-10
**Worktree:** `~/workspace/worktrees/raidjit/wasm-orphan` (branch `wt/wasm-orphan/wasm-orphan`, commit `1e2d349`)
**Source:** 15 files, 104 insertions, 43 deletions — found as uncommitted modifications in main tree (`~/workspace/raidjit`), dated 2026-10-08, no active owner.

## Summary of Changes

The orphaned changes fall into three categories:

### 1. WASM Source Changes (src/) — DUPLICATES, discard

| File | Change | Status in raidjit-wasm |
|------|--------|------------------------|
| `src/plan.c` | `RJW_AARCH64`/`RJW_RISCV` ifdefs for WASM target selection | **Already present** (lines 722, 782) |
| `src/x86_64/jit.c` | `RJ_WASM`-gated `memset` zeroing const pool alignment gap | **Already present** (line 225) |
| `src/aarch64/jit.c` | Same `RJ_WASM` pool zeroing | **Already present** (line 144) |
| `src/riscv/jit.c` | Same `RJ_WASM` pool zeroing | **Already present** (line 153) |

**Verdict:** These are already in the `raidjit-wasm` repo (branch `wasm-explorer`). They were likely copied from the WASM repo to the main tree during development, then the main tree copy was abandoned when the WASM work moved to its own repo. **Safe to discard.**

### 2. Test API Changes (tests/) — INCONSISTENT, discard

The test files were updated to call functions with fewer parameters:
- `tests/test_emit.c`: `em_mov_rm` 6→5 args, `em_movzx_rm8` 6→5 args
- `tests/test_matrix.c`: `mkcpu` 3→2 args
- `tests/test_prefetch_calib.c`: `dist_for` 6→5 args
- `tests/test_xorsched.c`: `xorsched_version_uops` 7→6 args
- `tests/test_prune.c`, `test_pxor_ternlog.c`, `test_rvv_lmul.c`, `test_xorcode.c`: similar

**Critical:** The source functions were NOT updated. Current source still has the old signatures:
- `em_mov_rm(em_t *e, int d, int b, int x, int sc, int disp)` — 6 params
- `xorsched_version_uops(..., int sched, int nt, int notail)` — 7 params

**Verdict:** These changes would **break the build** if applied. They appear to be from an incomplete API refactoring (removing unused parameters) that was started but never finished — the tests were updated but the source wasn't. **Discard.** If the API cleanup is desired, it should be done properly as a separate task with source+tests updated together.

### 3. Documentation Changes (hidden_files/, tools/) — VALUABLE, integrate to main

| File | Change | Recommendation |
|------|--------|----------------|
| `hidden_files/design_autotune_db.md` | Added "SUPERSEDED 2026-10-06" notice (Carlo rejected persistent DB in favor of stable plan IDs) | **Integrate** — important status update |
| `hidden_files/design_batched_api.md` | Added §13 "Decisions (Carlo, 2026-10-06)" with Q1-Q8 resolutions | **Integrate** — records Carlo's decisions |
| `tools/wasm/SPEC.md` | Added §2c documenting const-pool alignment padding fix | **Already in raidjit-wasm** — check if main needs it |

## Triage Verdict

**Recommendation: (a) partial integrate + (c) discard remainder**

1. **Integrate to main:** The two documentation updates (`design_autotune_db.md` SUPERSEDED notice, `design_batched_api.md` Q1-Q8 decisions). These are valuable records of Carlo's decisions that belong in the main repo.

2. **Discard:** 
   - WASM source changes (already in raidjit-wasm repo)
   - Test API changes (inconsistent, would break build)
   - SPEC.md addition (already in raidjit-wasm; main's tools/wasm/ may not need it)

## Action Items

- [ ] Cherry-pick the two doc updates to main (or manually apply)
- [ ] Abandon this worktree after doc integration
- [ ] Optional: If API parameter cleanup is desired, dispatch as separate task (source + tests together)

## Notes

The WASM changes confirm that the `raidjit-wasm` repo is the canonical home for WASM-specific code. The main repo should not accumulate WASM ifdefs — they belong in the WASM repo's copy of the source.
