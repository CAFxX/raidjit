# CRC Phase 2 Coordinator Checkpoint (2026-10-09 ~22:00 CEST)

## Status
- Worktree: ~/workspace/worktrees/raidjit/crc-fold (branch wt/eaee5117-d142-437a-bbac-a6bab55b1ddb/crc-fold)
- Base: d77fdb9 (Phase 1 complete)
- Commits:
  - 6fe699c: Phase 2 spec
  - 7433cc3: planner integration (cand_t.crc_fold + prune.c pin, want_crc, gate, plan_need_regs)
  - c0eb807: planid token, api.h accessor
  - 6673bcc: spec blocker fixes (callconv safety, AArch64 gate closed)
  - 04ab26d: fused driver (want_crc compile, 4-arg call, fold_state)
  - 05c2118: spec major fixes (SPR-only gate, masked tail, regime, budget, tests)

## Spec reviews (both done)
Reviewer 1: 2 blockers + 9 majors + 12 minors. FIXED:
  - Blocker 1: §4 fold_state size contradiction (16 vs 64 bytes) → now consistently nstreams*64
  - Blocker 2: AArch64 gate unimplementable → gate CLOSED, documented as deliberate
  - Major 3: uarch plumbing → already using rj_uarch_current() (override-aware)
  - Major 4: Borderline µarchs (Zen4/ICL-SP) → SPR-ONLY allowlist (conservative)
  - Major 5: Cache-resident regime → documented as DRAM-regime claim with conditions
  - Major 6: Register budget → per-(topology,U) table required as spec deliverable
  - Major 7: Masked tail folding → MUST mask/zero masked-off lanes or use scalar tail
  - Major 8: x86 test env → must name SPR silicon or QEMU config; loud skip if unavailable
  - Major 9: Multi-tile tests → single-vs-multi differential + stale-state red-green required

Reviewer 2: 3 blockers (full report pending delivery). Preview indicates §4.2 fold
construction issues. Awaiting full text.

## Workers
- x86-64 backend implementation (running; notified of fold_state=64B/stream and finalize requirement)
- AArch64 fallback verification (running)
- RISC-V fallback verification (running)

## Test suite
- `make test` running in background (proc_1b4e1e6da5e6), ~4 min so far

## Remaining
- x86-64 backend (worker) + finalize function
- Reviewer 2 blockers (awaiting full report)
- Test suite completion
- Byte-identity proof vs baseline (150 sections)
- 2+ implementation peer reviews
- Self-attack pass
- ASan/UBSan, QEMU

## Key design decisions (updated)
- fold_state: nstreams * 64 bytes (one zmm per stream), 64-byte aligned
- 4-arg kernel calling convention, ONLY via fused driver (callconv safety enforced)
- x86-64 gate: SPR ONLY (Zen4/ICL-SP excluded pending empirical confirmation)
- AArch64/RISC-V gates CLOSED for Phase 2 (conservative)
- "Almost free" is DRAM-regime claim with stated assumptions (50GB/s @ 3GHz)
