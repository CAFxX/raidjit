# QEMU full matrix results — 2026-10-02

QEMU 8.2.2 (Debian 1:8.2.2+ds-0ubuntu1.18), cross-built with
`aarch64-linux-gnu-gcc` / `riscv64-linux-gnu-gcc` from the tree at
`~/workspace/raidjit` (post sweep-remediation + VPTERNLOGD/hoisting work).
Private build dirs `/tmp/qemubuild-aa`, `/tmp/qemubuild-rv`; the tree's
`build-*` dirs untouched.

## AArch64

| binary | NEON (`-cpu max`) | SVE VL128 | SVE VL256 |
|---|---|---|---|
| test_aarch64_emit | PASS (90 checks) | PASS (90 checks) | PASS (90 checks) |
| test_nt | PASS | — | — |
| test_pdelta | PASS | — | — |
| test_pdelta_edge | PASS | PASS | PASS |
| test_boundaries | PASS | PASS | PASS |
| test_large_mds | PASS | — | — |
| test_select | PASS | — | — |
| test_benchreject | PASS | — | — |
| test_correct | PASS | PASS | PASS |
| test_matrix | PASS (27417 runs) | PASS (128457 runs) | PASS (128233 runs) |
| test_aarch64_remat | PASS | PASS | PASS |
| test_sve_pdelta | — | PASS | PASS |
| test_sve_eor3 | — | PASS | PASS |
| **config total** | **11/11** | **8/8** | **8/8** |

"—" = not in that make target.
`test_matrix` at SVE VL512 was also run once: 128233 runs, 0 failures, PASS.

## RISC-V (RVV)

| binary | VLEN 128 (`-cpu max`) | VLEN 256 | VLEN 512 |
|---|---|---|---|
| test_nt | PASS | — | — |
| test_pdelta | PASS | — | — |
| test_pdelta_edge | PASS | — | — |
| test_boundaries | PASS | PASS | PASS |
| test_large_mds | PASS | — | — |
| test_correct | PASS | PASS | PASS |
| test_matrix | PASS (17198 runs) | PASS (17198 runs) | PASS (17198 runs) |
| test_riscv_remat | PASS | PASS | PASS |
| test_riscv_prefetch | PASS | PASS | PASS |
| test_rvv_f6 | PASS | PASS | PASS |
| **config total** | **10/10** | **6/6** | **6/6** |

**Grand total: 49/49 PASS, 0 failures.**

## Notes

- The historical "test_matrix segfaults with SVE enabled" (Makefile note,
  now removed) does **not** reproduce. It was first attributed to a real
  codegen bug in `em_sve_lsr_unpred`'s emitter base, but a peer-review
  experiment refuted that (the old bug yields data mismatches, never a
  crash); the true root cause of the historical report is unknown —
  see `docs/limitations.md` §12 for the full corrected record.
  `test_matrix` was restored to the `test-qemu-aarch64-sve` make target
  on the strength of non-reproduction.
- `test_pdelta_edge` logs contain `qemu: uncaught target signal 11`
  notices under QEMU: expected — the test forks children that deliberately
  die in PROT_NONE guard pages (`verify_guard_active`); the parent checks
  for the expected SIGSEGV death and the test exits 0 / "all OK". Not a
  failure.
- `test_matrix` uses a fixed RNG seed (`0x243f6a8885a308d3`), so runs are
  deterministic; the three clean SVE runs are not flaky luck.
- `test_pressure` (portable — in `PORTABLE_TESTS`, run by the native
  `test` target on all arches) and ASAN builds are not covered under
  QEMU; neither appears in the QEMU make targets.

## A1 track re-validation (candidate-pruning hardening, 2026-10-02)

Re-ran all four QEMU suites for real on the A1 tree after the
pruning-hardening changes (kind guard, remat axis, test-matrix
extensions). QEMU 8.2.2, cross-compilers reinstalled post-VM-reboot
(`apt --fix-broken install` was needed: the reboot left
`gcc-13-aarch64-linux-gnu` half-configured and the aarch64 sysroot
without libc headers).

- `make test-qemu`: exit 0 — `test_matrix` "all OK" under all 4
  qemu-x86_64 CPU models (qemu64,+ssse3,-avx / Haswell /
  Skylake-Server / Icelake-Server).
- `make test-qemu-aarch64`: exit 0 — all 12 binaries pass under
  qemu-aarch64 (`-cpu max`).
- `make test-qemu-aarch64-sve`: exit 0 — all 18 runs pass
  (test_matrix: 128457 runs VL128 + 128233 runs VL256, 0 failures).
- `make test-qemu-riscv64`: exit 0 — all binaries pass, incl.
  test_matrix at vlen=128/256/512 (17198 runs each, 0 failures).

Latent bug fixed to make this possible: `tests/test_prune.c` defined
x86-64-only topology/op helpers unconditionally, which fails the
cross builds with `-Werror=unused-function` (the pre-existing file
had the same issue; it was never exposed because the cross targets
were not actually run). The x86-only helpers and
`compile_capture_stderr` are now guarded by `#if defined(__x86_64__)`.

## Integration 1 re-validation (2026-10-02, evening)

After the A1/A2/A3/A4/A6 merge, two test-infrastructure bugs were found
and fixed; the QEMU matrix was re-run to validate.

### SVE VL source (test_dispatch failure)

`test_dispatch` failed under QEMU SVE at all VLs: the notail HORNER twin
produced zero output bytes at every admissible length. Root cause: the
merge had (briefly) reverted A2's `cntb` build-time VL hunk in
`src/aarch64/jit.c` on the misdiagnosis that "cntb is 8x wrong". Measured
ground truth: qemu-user 8.2.2 interprets
`-cpu max,sve-default-vector-length=N` as N *bytes* (not bits) — the true
emulated VL is N bytes, verified by an SVE `ld1b`/`st1b` round-trip under
`whilelo(cntb)` moving exactly `cntb` bytes (128 for N=128). `prctl` and
`rdvl` under-report 8x. The hunk was restored; the companion fix was in
`tests/test_xorsched.c`, whose prctl-based `uop_divisor` was deleted —
all call sites now use each version's `group_bytes` (RVV uses U*16,
matching its NOMVL-normalized `uops_per_byte`). See
`docs/length_dispatch.md` §11.0.

### Re-validation results

- `test_xorsched` + `test_dispatch`: PASS at SVE VL128/256/512 and RVV
  vlen=128/256/512 (post-fix, post-rebuild).
- Native `make test`: exit 0 (25/25 binaries).
- `make test-qemu-riscv64`: in progress; `test_xorsched` passes at all
  vlens with the RVV divisor fix.
- Full `make test-qemu-aarch64-sve` re-run: critical tests pass;
  complete suite pending (long benchmark tests under emulation).
