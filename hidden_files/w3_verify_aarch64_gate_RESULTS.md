# W3 verification results: CRC Phase 2 AArch64 gate CLOSED, fallback correct

Date: 2026-10-09. Tree: branch `wt/eaee5117-d142-437a-bbac-a6bab55b1ddb/crc-fold`
at 5fece87 (verified against HEAD; rebuilt from clean HEAD state).
Verifier: W3 subagent (AArch64 worker).

Spec basis: `hidden_files/crc_fold_phase2_SPEC.md` §2 (AArch64 gate CLOSED
for Phase 2 — no Apple-silicon µarch detection exists, so a conservative
allowlist is unimplementable; scalar fallback always) and §7.1/§7.5
(fallback correctness).

## Verdict

1. **Gate closed? YES.** `fused_crc_interleave_ok()` returns 0 on AArch64
   (src/fused.c `#elif defined(__aarch64__)` branch: unconditional
   `return 0`). Confirmed at runtime under QEMU aarch64 `-cpu max`.
2. **Planner never enumerates crc_fold=1 on AArch64? YES.**
   `plan_candidates` with `pl.want_crc=1` returns 0 candidates (the gate
   check at src/plan.c:1259 fires; caller falls back per spec §3.2).
   With `want_crc=0`: 3 candidates enumerated, all with `crc_fold == 0`.
3. **Scalar fallback correct? YES.** End-to-end fused `[ENCODE, CRC->0]`
   (Phase-1 `rj_crc32_fast` over tile buffers — the gate-closed driver
   path) produces per-stream CRCs bit-exact vs an independent bitwise
   oracle, vs `rj_crc32_fast`, and vs `rj_crc32_update`, across
   raid5(4)/raid6_rs(6)/rs(6,3) × lengths {1,7,63,64,65,1000,4096,65537}.
4. **QEMU aarch64 runs? YES** (qemu-aarch64 8.2.2, `-cpu max`,
   QEMU_LD_PREFIX=/usr/aarch64-linux-gnu, aarch64-linux-gnu-gcc 13.3.0).

## Test evidence (all under QEMU aarch64, tree at 5fece87)

| Test | Result |
|---|---|
| `hidden_files/w3_verify_aarch64_gate.c` (new gate/planner/fallback test) | PASS: `gate: fused_crc_interleave_ok() = 0`; `want_crc=1 -> 0 candidates`; `want_crc=0 -> 3 candidates` (all crc_fold=0); fused CRCs match oracle/fast/update on all topologies×lengths |
| `test_crc_fast` (RJ_CRC_SKIP_HUGE=1) | PASS (failures=0 deferred=0) |
| `test_crc` | all OK |
| `test_fused` | all OK (rejections, encode_crc_equals_38 ×3, pupdate_crc, encode_pupdate_pair, two_encodes_and_single, benchmark_path) |
| `test_prune` | all OK (253 checks) |
| `test_planid` | CRASH — pre-existing, out of scope (see below) |

## Bugs found

1. **FIXED (build breaker):** `src/aarch64/jit.c:1520`, in `compute_regs()`,
   passed `cand->crc_fold` where no `cand` is in scope (the candidate copy
   lives at `cg->cand`; the same expression already uses `cg->cand.remat`
   / `cg->cand.pipe`). The aarch64 cross-build was broken at HEAD by this
   (`-Werror=implicit-function... 'cand' undeclared`). One-token fix to
   `cg->cand.crc_fold`. The x86-64 native build never saw it (aarch64-only
   TU) — the coordinator does not cross-build. Landed via the coordinator's
   sweep in 8ec019b (identical to the verifier's working-tree fix).
2. **REPORTED, NOT FIXED (pre-existing, orthogonal):** `test_planid` aborts
   under QEMU aarch64 with glibc `*** buffer overflow detected ***`. Root
   cause is in the TEST, not the library: Part 1 splices a `+pipe` token by
   `strstr(id1, ":u8:-:")`, an x86-64 ID-format assumption. On AArch64 the
   ID is `...:enc:neon:mulacc:nibble:u2:pipe:scalar`, so `strstr` returns
   NULL, `pre` wraps, and `memcpy(pid, id1, pre)` trips fortify. The test
   file was last touched by 4aca664 (pre-crc-fold); the crc-fold planid
   change only adds a `+crc_fold` token when `crc_fold=1`, which never
   occurs with the gate closed (no token appears in the aarch64 IDs).
   The Makefile's `_test-qemu-aarch64` target does not run test_planid.
   Suggested fix: make the splice arch-aware (locate the twins axis
   generically instead of hard-coding `:u8:-:`). Left for the coordinator —
   out of this verification's scope.

## Notes

- No AArch64 interleave was implemented (per task constraints; deferred to
  a future phase per spec §2).
- No benchmarks or timing code were added or run (per task constraints).
- The verifier's test file (`hidden_files/w3_verify_aarch64_gate.c`) follows
  the `w3_verify_riscv.c` precedent: verification evidence in hidden_files,
  not part of the shipped test suite.
- During verification the coordinator committed concurrently in the same
  worktree (04ab26d swept the verifier's in-flight test file; 8ec019b
  swept the identical one-line fix). The final battery above was re-run
  from a clean HEAD build (19:54) to attribute results to 5fece87 exactly.
