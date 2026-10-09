# RISC-V fallback verification — CRC Phase 2 (interleaved CRC folding)

Date: 2026-10-09. Worktree: `wt/eaee5117-d142-437a-bbac-a6bab55b1ddb/crc-fold`.
Verified against HEAD `54f6989` (plus the two uncommitted backend hunks
this report commits). No RISC-V interleave implemented — gate closed by
design, per spec §2. No benchmarks run (none requested; task forbids them).

## Verdict

1. **Gate closed on RISC-V: CONFIRMED.** `fused_crc_interleave_ok()`
   returns 0 on RISC-V, empirically and by inspection.
2. **Planner never enumerates `crc_fold=1` on RISC-V: CONFIRMED.**
   `plan_candidates` with `want_crc=1` returns 0 candidates (fused driver
   recompiles `want_crc=0`); with `want_crc=0` all candidates have
   `crc_fold=0`.
3. **Scalar fallback correct: CONFIRMED.** Fused `[ENCODE, CRC]` CRCs match
   the Phase-1 oracle (`rj_crc32_fast`) and an independent bitwise oracle
   over 3 topologies × 16 lengths × all input+output streams.
4. **QEMU riscv64: relevant suites green** (test_fused, test_crc,
   test_crc_fast; test_planid/test_prune green natively on x86_64;
   test_twin_audit reachable sweep green).
5. **Bug found and fixed:** the spec-mandated backend defense-in-depth
   rejection of `crc_fold=1` when the gate is closed was missing —
   a hand-crafted `+crc_fold` pinned plan ID aborted the RISC-V backend
   on an assertion. All three backends now reject cleanly.
6. **Pre-existing issue (out of scope, not fixed):** `test_planid` aborts
   with a fortify "*** buffer overflow detected ***" under qemu-riscv64 in
   Part 1 — reproduces on the Phase-1 base commit `d77fdb9` (no Phase-2
   code). Not in the project's `_test-qemu-riscv64` target list.

## Task 1 — gate returns 0 on RISC-V

`src/fused.c::fused_crc_interleave_ok`: RISC-V (`__riscv`) falls into the
`#else` branch → `(void)cpu; return 0;`. Empirical, under
`qemu-riscv64 -cpu max,zvbc=false,zvkg=false`:
`fused_crc_interleave_ok(NULL) == 0`, `fused_crc_interleave_ok(live cpu) == 0`.

## Task 2 — planner never enumerates crc_fold=1

Code: the only `crc_fold = 1` assignment in `plan_candidates`
(`src/plan.c`) is inside `if (pl->want_crc)`, guarded by
`if (!fused_crc_interleave_ok(cpu)) return 0;`. Base candidates are built
with designated initializers, so `crc_fold` defaults to 0. Empirical
(QEMU riscv64), topologies raid5(4), raid6_rs(6), rs(6,3):
- `want_crc=1` → 0 candidates (26/19/14 candidates with `want_crc=0`,
  every one `crc_fold=0`).

Note: `rj_topo_rs(10,4)` returns NULL by design (outside the public
envelope, `include/raidjit.h`); it is not a usable test topology.

## Task 3 — scalar fallback correctness

`fused_driver_call` (`src/fused.c`) with `f->crc_fold == 0` (always true
on RISC-V: `rj_compile_want_crc` is never attempted when the gate is
closed) runs the unchanged Phase-1 path: `rj_crc32_init` /
`rj_crc32_fast` over each tile buffer / `rj_crc32_final`.

Empirical (QEMU riscv64, scratch test `/tmp/verify_riscv_gate.c`, not
committed): fused `[ENCODE, CRC->0]` for raid5(4), raid6_rs(6), rs(6,3);
16 lengths each (1..524289, covering every tail residue mod 64 and tile
boundaries); per-stream CRCs asserted against BOTH `rj_crc32_fast`
(the exact oracle the driver uses) and an independent bitwise
CRC-32/ISO-HDLC reimplementation. All checks passed.

Existing suites under QEMU riscv64:
- `test_fused`: all OK (fused `[ENCODE,CRC]` bit-exact vs `rj_compile_crc`
  and vs bitwise oracle).
- `test_crc`: all OK.
- `test_crc_fast` (class 1, `RJ_CRC_SKIP_HUGE=1`): PASS, failures=0.

## Bug found and fixed — backend gate re-check (spec §2, §4.1)

The spec mandates: "`jit_build_version` rejects `crc_fold=1` if the gate
is closed, returning a clean build error, never wrong code." No backend
implemented this. Reachable channel: `rj_compile_pinned` with a
hand-crafted `+crc_fold` plan ID — `planid_parse` sets `crc_fold=1`
(`src/planid.c`), and `cand_spec_equal` (`src/api.c`) does NOT compare
`crc_fold`, so the membership check passes against a `crc_fold=0`
candidate.

Reproducer (QEMU riscv64, `/tmp/g4_pinned.c`): pinned ID
`rj2:raid5(4)#07e41ddc8aef8c9c:enc:rvv:xor:nibble:u8:lmul1+crc_fold:scalar`
→ before fix: **SIGABRT**,
`src/riscv/jit.c:1477: jit_build_version: Assertion 'vneed == want_regs * L'
failed` (register budget accounts fold regs the backend never allocates;
a release build would instead silently build a non-folding kernel whose
version falsely claims `rj_jit_crc_fold() == 1`).

Fix (this commit): each backend's `jit_build_version` rejects
`cand->crc_fold && !fused_crc_interleave_ok(cpu)` with a clean build
error (`-1` + message), per spec. After fix, the reproducer fails
cleanly: "pinned plan ID: candidate build failed: riscv backend:
crc_fold=1 but the interleaved-CRC uarch gate is closed on this CPU".

Note: the coordinator's concurrent commit `54f6989` independently added
the identical x86_64 hunk (verbatim); this commit adds only the
riscv + aarch64 hunks. The x86_64 working-tree hunk was therefore
dropped as redundant.

Regression evidence for the fix:
- riscv64 lib + tests build clean (`-Werror`); reproducer now rejects;
  gate/planner/fallback verification still passes (final tree).
- x86_64 native: `test_twin_audit` ok (reachable sweep: 1152 built,
  0 gracefully rejected, 0 aborted — spec §7.4); `test_prune` all OK
  (351 checks); `test_planid` all OK; `test_fused` all OK.
- aarch64 backend cross-compiles clean (`-Werror`).

## Pre-existing issue — test_planid fortify abort under qemu-riscv64

`test_planid` aborts almost immediately (`*** buffer overflow detected
***: terminated`) under qemu-riscv64. Bisected to the Phase-1 base
`d77fdb9` (built in a scratch dir from `git archive`; no Phase-2 code) —
**pre-existing, unrelated to this work**. Localized to Part 1, in the
`strcmp(id1, id2)` CHECK region after two successful `rj_plan_id` calls
for raid6(8) (`id1=rj2:raid6(8)#8db4e55841a28f38:enc:rvv:mulacc:nibble:u3:lmul2:scalar`).
Not investigated further: out of scope, and `test_planid` is not in the
project's `_test-qemu-riscv64` target list (which is why it was never
caught). Flagged for coordinator triage.

## Notes

- The coordinator committed to this branch during verification
  (`04ab26d`, `05c2118`, `e16467b`, `54f6989`); the x86-64 allowlist is
  now SPR-only per the updated spec — out of scope for this report.
- All commands run under `nice -n 10`; backoff check was CLEAR before
  heavy work.
- No code changes beyond the two backend rejection hunks; no
  RISC-V interleave implemented (gate closed by design).
