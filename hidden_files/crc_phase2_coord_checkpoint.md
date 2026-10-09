# CRC Phase 2 Coordinator Checkpoint (2026-10-09 ~21:45 CEST)

## Status
- Worktree: ~/workspace/worktrees/raidjit/crc-fold (branch wt/eaee5117-d142-437a-bbac-a6bab55b1ddb/crc-fold)
- Base: d77fdb9 (Phase 1 complete)
- Commits:
  - 6fe699c: Phase 2 spec
  - 7433cc3: planner integration (cand_t.crc_fold + prune.c pin, want_crc, gate, plan_need_regs)
  - c0eb807: planid token, api.h accessor
  - 6673bcc: spec blocker fixes (callconv safety, AArch64 gate closed)
  - 04ab26d: fused driver (want_crc compile, 4-arg call, fold_state)

## Spec reviews
- Reviewer 1 (done): Found 2 blockers (inferred from preview):
  1. §4 contract contradiction (fold_state size: 16 vs 64 bytes) — FIXED (spec now says nstreams*64)
  2. AArch64 gate unimplementable — FIXED (gate closed, documented as deliberate)
- Reviewer 2 (running)

## Workers
- x86-64 backend implementation (running)
- AArch64 fallback verification (running)
- RISC-V fallback verification (running)

## Remaining
- x86-64 backend (worker)
- Finalize function in fused.c driver (placeholder)
- Test suite (running in background)
- Byte-identity proof vs baseline
- 2+ implementation peer reviews
- Self-attack pass
- ASan/UBSan, QEMU

## Key design decisions
- fold_state: nstreams * 64 bytes (one zmm per stream), 64-byte aligned
- 4-arg kernel calling convention, ONLY via fused driver
- AArch64/RISC-V gates CLOSED for Phase 2 (conservative)
- x86-64 gate: vpclmulqdq && avx512f && (SPR|Zen4|ICL-SP)
