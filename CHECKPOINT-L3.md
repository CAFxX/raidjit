# LEAD L3 — AArch64 direct-caller backend gates (CHECKPOINT)

Claim: defensive AArch64 backend gates for direct `jit_build_version` callers may
be missing for x86-only `permi2b`/`mshift` twin pins — a direct caller passing
permi2b=1/mshift=1 to the aarch64 backend might get a silently wrong build instead
of a clean rejection. Status: OPEN, explicitly "unverified".

- Worktree: ~/workspace/worktrees/raidjit/lead-l3-aarch64-gates (branch
  wt/raidjit/lead-l3-aarch64-gates), base 7203cf1e (crc-fast@7203cf1)
- Registry: ~/workspace/system/work-registry/raidjit-lead-l3-aarch64-gates.json

## Log
- 2026-10-12 08:00 CEST: started. Backoff CLEAR (load=6.62). Worktree claimed.
- 2026-10-12 ~08:20 CEST: RED phase. Wrote tests/test_aarch64_phantom_gates.c
  (wired into Makefile PORTABLE_TESTS + qemu run list). Cross-built with
  ARCH=aarch64 CROSS_COMPILE=aarch64-linux-gnu- (note: plain
  `make ARCH=aarch64` without CROSS_COMPILE misbuilds with the x86
  assembler — stale x86_64 objects in build-aarch64/ had to be rm -rf'd).
  Under qemu-aarch64: all 6 phantom pins (permi2b/mshift/ternlog/maskedtail/
  mul=GFNI/mul=ZVBC) returned rc=0 silently — CONFIRMED, test fails red as
  designed (12 FAIL lines). planid_cand_valid rejects all 4 twin pins
  (planner path fine). Control plain-neon build succeeds.
- 2026-10-12 ~08:35 CEST: FIX. src/aarch64/jit.c: five fail-closed gates
  after the pipe gate (permi2b/mshift/ternlog/maskedtail/mul!=MUL_NIBBLE),
  mirroring x86_64/riscv message style; permi2b+mshift added to the F16
  postcondition twins[] table (defense in depth). Rebuilt cross, test
  GREEN: all 6 pins rejected with clean errors, planid path still rejects,
  control builds.
- 2026-10-12 ~09:00 CEST: full native suite (keep-going runner,
  run_full_suite.sh): test_remat FAILS on "rs(6,3) all-trivial nibble U=8
  build: remat twin requested but its code path was not taken (phantom
  twin)" — VERIFIED PRE-EXISTING on pristine base 7203cf1 (stash test:
  identical failure with my changes stashed; none of my files feed the
  x86_64 build). Out of scope for this lane; recorded as NEW LEAD for the
  coordinator (x86_64 backend: remat phantom twin on all-trivial coeffs).
  Rest of suite still running.

## Gate surface map (base 7203cf1, src/aarch64/jit.c::jit_build_version)

Backend-level gates PRESENT (fail closed): remat (P_HORNER only), pipe
(P_MULACC/P_HORNER nibble, non-remat, no xorsched), rec_cf (full body-twin
exclusion incl. permi2b/mshift), nt+NEON+U<2, prefetch_dist encodability,
lmul (RVV-only), predtail (SVE-only), pipe+predtail, SVE VL>0.

Backend-level gates MISSING (phantom pins, silently ignored by codegen;
grep shows ZERO references to these in src/aarch64/):
- permi2b, mshift (only excluded inside the rec_cf combo gate, line 2108)
- ternlog (x86-64 VPTERNLOGD twin)
- maskedtail (x86 opmask / RVV v0 twin; api.c duels it only on
  ISA_AVX512/ISA_RVV, never aarch64 — gate is safe)
- cand->mul != MUL_NIBBLE (MUL_GFNI x86-only, MUL_ZVBC riscv-only;
  planner enumerates MUL_NIBBLE only for NEON/SVE — plan.c:1006-1018)

BUG-COMBO-STATIC-F16 postcondition: aarch64 twins[] table (line ~2398)
lists nt/prefetch/remat/xorsched/xorbit/pipe/rec_cf but NOT permi2b/mshift
(x86_64's table DOES list them, x86_64/jit.c:3227-3228).

Planner level is fine: api.c build_pinned_cand + planid.c planid_cand_valid
SHAPE_BAD permi2b/mshift/ternlog/maskedtail for non-AVX512 ISA; ui_meta
twin-enable delegates to planid_cand_valid. The gap is exactly the DIRECT
jit_build_version caller boundary — LEAD L3 CONFIRMED (no wrong code: twins
are perf-identical, but the pin is silently dropped and v->cand records a
false twin — the phantom-twin class F16 was built to kill).

Out of scope (noted for coordinator): riscv backend has the same missing
standalone gates (permi2b/mshift/ternlog only inside rec_cf at riscv/jit.c:2111;
its F16 table also omits permi2b/mshift/ternlog/maskedtail) — candidate for a
follow-up lane.
