# Single-Erasure Coordinator Checkpoint

## Status (2026-10-10, updated)

### Completed
1. **Spec**: `hidden_files/single_erasure_SPEC.md` — honest win-region analysis.
2. **Planner core** (commit 43b2677): rec_cf axis, prune pin, cf_data plumbing.
3. **Twin enumeration** (commit 566153b): plan_candidates gates, backend stubs.
4. **Backend emitters** (commit 016f58f):
   - x86_64: 664/664 differential-clean
   - aarch64: 104/104 NEON + 104/104 SVE differential-clean (QEMU)
   - riscv: 180/180 differential-clean (QEMU, VLEN 128/256/512)
5. **Tests**: test_recover_closedform ✅, test_twin_audit ✅ (1152 builds)

### In flight
- Peer review 1 (general): ef291005
- Peer review 2 (adversarial): 304094a1
- test_correct running in background (proc_90fa6aa387f3)

### Remaining
1. Peer reviews → remediate findings
2. Self-attack pass
3. ASan/UBSan (deferred due to load; do before integration)
4. Integration via wt.py

### Notes
- The RISC-V worker flagged a transient x86_64 e=2 issue; re-ran differential test: 664/664 clean. Was a transient during development.
- AArch64 worker noted QEMU SVE VL quirk (sve-default-vector-length=128 means 128 BYTES); test recommendation documented.
- NO benchmarks, NO perf guard, NO push.
