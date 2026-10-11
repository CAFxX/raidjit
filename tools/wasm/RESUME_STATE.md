# WASM Explorer Resume State (worktree ~/workspace/raidjit-wasm, branch wasm-explorer)

## Verified good (do not redo)
- x86: 33/33 parity (clean worktree driver, -DRJ_WASM deterministic const gaps)
- riscv: 11/11 parity (first clean run)
- aarch64: 9/10 parity (predtail case failed; fix committed, module rebuild pending)

## Commits on wasm-explorer
- 22f98cc: RJ_WASM const-pool gap memset + [plan.c](../../src/plan.c) ifdefs (from main tree)
- [run_parity.sh](parity/run_parity.sh): -DRJ_WASM for native driver (deterministic const gaps)
- [tools/gen_live_kernels_page.py](../gen_live_kernels_page.py) (from main tree)
- [rjw_api.c](rjw_api.c): mask stage-2 tail flags for planid_cand_valid (predtail pinnable)

## Fixes applied
1. [run_parity.sh](parity/run_parity.sh): native driver built with -DRJ_WASM (fixes flaky const-blob gaps)
2. [rjw_api.c](rjw_api.c): predtail/notail/maskedtail masked for planid_cand_valid (both in
   build_pinned_cand and the U-scan loop); feature_gates + backend validate them
3. [dis_api.c](dis_api.c): Capstone 6 CS_OPT_MEM setup before cs_open (fixes CS_ERR_MEMSETUP);
   verified natively: cs_open OK, disassembles correctly

## In progress
- [disasm.js](out/disasm.js) + [aarch64.js](out/raidjit_aarch64.js) rebuilds — OFFLOADED TO GHA 2026-10-09 05:25 UTC
  (VM load ~37, local emcc hangs). Tenant `wasm-build` in CAFxX/fanout:
  commit 39e6fd99 (Dockerfile, shard-aware driver.sh, README, slim
  source tarball from wasm-explorer @6095f11, Capstone .c list verified
  identical to [Makefile.wasm](Makefile.wasm)'s find). Image
  ghcr.io/cafxx/fanout-wasm-build:latest auto-building via build-image.yml.
  Job `wasm-rebuild-20261009a` (2 shards: 0=disasm, 1=aarch64) to be
  submitted once the image is fresh. Results commit back to
  results/wasm-build/wasm-rebuild-20261009a/ — then download
  shard-0/disasm.js + shard-1/raidjit_aarch64.js into [tools/wasm/out/](out/)
  and re-run parity/smoke LOCALLY (GHA outputs are leads, not verdicts).
- 2026-10-09 05:35 UTC: job `wasm-rebuild-20261009a` FAILED both shards —
  root cause: tenant Dockerfile pinned emsdk 4.0.11, which predates
  `-sSINGLE_FILE_BINARY_ENCODE` (needs >= 4.0.18; local toolchain is
  6.0.11). Diagnosed from workflow logs (PAT-authenticated download).
  Fix pushed (2197b4be): Dockerfile -> emsdk:6.0.11 (matches local
  verified toolchain), driver.sh tees build log to $OUT_DIR.
  Rebuild image in progress; job `wasm-rebuild-20261009b` queued next.
- 2026-10-09 05:55 UTC: job B partial — aarch64 OK, disasm failed:
  `cs.c` includes every arch's `*Module.h` unconditionally; slim tree
  lacked them. Fixed tarball (added 20 self-contained `*Module.h`);
  job C rebuilt disasm OK. But aarch64 parity still 9/10: the predtail
  case failed on BOTH sides with "plan ID did not fit" — root cause in
  [rjw_api.c](rjw_api.c) (not the previous coordinator's bug): planid_format rejects
  tail flags in cand by design; they must go in the planid_tail_t arg.
  Fixed (171c6b4): tail flags moved to the <tail> axis; native driver
  now returns ok:true with plan_id
  `rj2:raid5(4)#07e41ddc8aef8c9c:enc:sve:xor:nibble:u1:-:predtail`.
  Also hardened [run_parity.sh](parity/run_parity.sh) (both-failed-but-expect-ok reported
  cleanly instead of KeyError). Job D (aarch64 rebuild with fix)
  submitted; then re-run parity -> target 10/10.
- 2026-10-09 06:10 UTC: **aarch64 parity 10/10** (job D module). ALL-ISA
  parity complete: x86 33/33, riscv 11/11, aarch64 10/10.
- 2026-10-09 06:30-07:30 UTC: S9.4 disasm differential. Fixed test bugs:
  BigInt base_addr, stringToNewUTF8 for arch, objdump -D for binaries,
  data16-prefix stripping, norm() len>3, 6690 xchg->nop idiom,
  b.hs/b.cs b.lo/b.cc aliases, trailing-zero trim. x86+aarch64: full
  match vs objdump. riscv: objdump cannot decode RVV in binary mode
  (binutils 2.42 limitation); riscv differential uses a native
  [dis_api.c](dis_api.c) reference build instead (in progress).
- GHA tenant: wasm-build (Dockerfile emsdk:6.0.11, shard driver,
  push_paths.py with ref-race retry). Jobs: A failed (emsdk 4.0.11),
  B partial (disasm missing Module.h), C disasm OK, D aarch64 OK.

## Remaining
1. aarch64 parity 10/10 (after rebuild)
2. [disasm.js](out/disasm.js) smoke test (after rebuild)
3. S9.4 disasm differential vs objdump
4. Live page: python3 tools/gen_live_kernels_page.py
5. S9.3 native no-regression (make kernels before/after)
6. S9.5 native suite, S9.6 page smoke, S9.7 peer reviews, S9.8 self-attack, S9.9 docs

## Key lessons
- Parity driver MUST be rebuilt from worktree sources; cached main-tree driver is stale
- Capstone 6.0.0-Alpha11 requires cs_option(NULL, CS_OPT_MEM, ...) before cs_open

- 2026-10-09 08:15 UTC: **S9.4 disasm differential COMPLETE (48/48)**.
  x86+aarch64 (37 blobs): mnemonic sequences match objdump exactly
  (documented normalizations: 6690 xchg->nop idiom, b.hs/b.cs b.lo/b.cc
  aliases, trailing-zero trim, AT&T suffix strip). riscv (11 blobs):
  objdump cannot decode RVV in binary mode (binutils 2.42); validated
  WASM [disasm.js](out/disasm.js) against a native build of the same [dis_api.c](dis_api.c)
  (parity/native_ref/disasm_native_ref, gitignored): 11/11 match.
- 2026-10-09 08:05 UTC: x86 module rebuilt on GHA (job E) from fixed
  source for consistency; **x86 parity re-verified 33/33**.

- 2026-10-09 08:20 UTC: **First working page**. [tools/gen_live_kernels_page.py](../gen_live_kernels_page.py)
  fixed (OUT_DIR -> [tools/wasm/out/](out/), dj_disasm JS bindings), generated
  [tools/wasm/out/raidjit_kernels.html](out/raidjit_kernels.html) (4.3MB, self-contained, file://).
  S9.6 smoke PASS in Node: compile case_1 -> 512 bytes, disassemble ->
  101 insns.

- 2026-10-09 08:45 UTC: **S9.3 native no-regression PROOF**.
  x86 runtime before(2c9e8cc^)/after(HEAD), pure native (no RJ_WASM):
  29/29 comparable cases byte-identical code blobs; 4/4 expect=fail
  cases fail identically on both. (Also 33/33 identical with -DRJ_WASM.)
  Preprocessor: [plan.c](../../src/plan.c) IDENTICAL under __x86_64__/__aarch64__/__riscv;
  the three jit.c differ only in #ifdef whitespace, braces around
  single-statement ifs, and __LINE__ in asserts (semantically null;
  x86 .o differs only in assert line-number immediates).

- 2026-10-09 10:00 UTC: **S9.7 peer review (3 reviewers)**. Reviewer 1 found
  three pinned tail-flag combinations that aborted the WASM module via
  backend assert instead of clean error: x86 maskedtail+notail,
  x86 maskedtail+non-AVX512 ISA, aarch64 notail+predtail. Fixed in
  feature_gates (commit 13f053d); modules rebuilt on GHA (jobs F);
  verified all three now return clean errors; x86 parity 33/33
  re-verified; page regenerated.

- 2026-10-09 11:30 UTC: **S9.7 peer reviews complete (3/3)**. Reviewer 2:
  [dis_api.c](dis_api.c) correct; [disasm_diff.py](parity/disasm_diff.py) riscv path fixed (now points to
  native-ref); -sWASM_BIGINT=1 pinned. Reviewer 3: page JS robustness
  fixes (JSON.parse guards, disassemble error paths, hash validation).
  All remediations committed; page regenerated.

- 2026-10-09 12:30 UTC: **S9.8 self-attack 102/102**
  (tools/wasm/self_attack/attack.js + SELF_ATTACK.md): 29 adversarial
  compile docs (+29 state-recovery probes), 20 deep-link hash attacks
  (incl. prototype-pollution, 5MB hash), 12 disasm attacks. No traps,
  no hangs, no state corruption. No findings requiring code changes.

- 2026-10-09 12:40 UTC: **S9.5 native `make test` GREEN**. Full run
  08:46-12:35 UTC progressed sequentially through test_batched (make
  stops on first failure); tail re-run explicitly: test_batched OK,
  test_lrc 2733 checks 0 failures, test_twin_audit ok.

- 2026-10-09 12:45 UTC: **S9.9 docs final**: [SPEC.md](SPEC.md) status FINAL +
  §S12 verification record + Capstone 6 mem_setup requirement + GHA
  build flow; [README.md](README.md) expanded (Page section: any-combination UI,
  A/B mode, deep links; Build flow: local + GHA wasm-build tenant;
  Capstone 6 note); [tools/kernels-README.md](../kernels-README.md) notes the static page is
  superseded. [out/disasm.js](out/disasm.js) + [out/raidjit_riscv.js](out/raidjit_riscv.js) committed (verified
  modules, consistent with the committed x86/aarch64 modules).

## Deliverable
- Live page: [tools/wasm/out/raidjit_kernels.html](out/raidjit_kernels.html) (4,450,464 bytes,
  self-contained, file://-safe). Regeneration is deterministic:
  re-running [tools/gen_live_kernels_page.py](../gen_live_kernels_page.py) reproduces it byte-for-byte.
- S9.6 re-smoke on the final page modules (2026-10-09 12:50 UTC):
  compile raid5(4)/avx512-gfni → 512 bytes, disassemble → 101 insns. PASS.
- Swap: ~/workspace/your_files/raidjit/raidjit_kernels.html REPLACED
  with the live page (static explorer omitted per owner order);
  raidjit_kernels.md stays as the archived verification corpus.
