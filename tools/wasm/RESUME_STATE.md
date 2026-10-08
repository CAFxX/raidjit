# WASM live explorer — resume state (written 2026-10-09 ~01:05 CEST by main agent)

Two coordinators died in daemon restarts (2026-10-08 ~23:00, 2026-10-09 ~00:54).
Respawns paused until the runtime stabilizes; all state below is on disk.

## What the original coordinator completed (handoff received)
- x86 module: built AND verified 33/33 parity vs native.
- Spec: `tools/wasm/SPEC.md` (reviewed). Incorporated: stub-list fixes
  (two `#ifdef RJ_WASM` VL fixes + pthread stub removal), `-ffp-contract=off`
  determinism pin, SINGLE_FILE + base64 for file://.
- The six errored workers were daemon-restart losses, already accounted for.

## On disk, ready to use
- `tools/wasm/out/raidjit_x86.js` (built 2026-10-08 19:00)
- `tools/wasm/out/raidjit_aarch64.js` (built 2026-10-08 19:53)
- Parity harness: `tools/wasm/parity/` (was 33 passed / 0 failed / 54 x86 cases)
- Capstone 6.0.0-Alpha11 in `tools/wasm/vendor/`
- `tools/wasm/dis_api.c`, `rjw_api.c`, `rjw_wasm.h`,
  `cpu_wasm_{x86,aarch64,riscv}.c`
- Pending-build fixes synced to `~/workspace/tmp/wasm-s93/`

## Tree tangle (MUST resolve first)
- `~/workspace/raidjit` is on branch `crc-fast`, NOT master, and carries
  UNCOMMITTED WASM changes (the two VL fixes in src/*/jit.c, test
  expectation updates, SPEC.md +7). The WASM pivots commit `2c9e8cc` is
  already an ancestor of crc-fast.
- The original coordinator reported "native builds blocked on the CRC
  track's merge conflicts" — this is the same tangle.
- RECOMMENDATION: do WASM work in a dedicated worktree branched from a
  clean base (or commit the WASM changes to their own branch first).
  Never mix WASM and CRC uncommitted changes. Coordinate with the CRC
  track; do not fight over the working tree.

## Still to do
1. Verify the x86 33/33 claim independently (don't trust the handoff).
2. aarch64/riscv/disasm builds from the wasm-s93 fixes.
3. Capstone-wasm disassembly wiring.
4. The live page (any combo compiled live; single + side-by-side A/B with
   deep links; replaces the static page).
5. Byte-identical wasm-vs-native proof, peer reviews, self-attack.

## Constraints (repeat every resume)
No timing benchmarks. No perf guard. No push. `nice` everything.
`TMPDIR=~/workspace/tmp`. Additive `#ifdef RJ_WASM` only.
