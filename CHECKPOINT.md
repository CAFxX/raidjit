# Lead L2 (const-pool alignment-gap determinism) — checkpoint

Worktree: /home/hatch/workspace/worktrees/raidjit/lead-l2-constpool
Branch: wt/raidjit/lead-l2-constpool @ crc-fast 7203cf1 (2026-10-12)
Registry: ~/workspace/system/work-registry/raidjit-lead-l2-constpool.json
Status: VERIFICATION (source-level confirmation done; empirical reproducer next)

## 2026-10-12 ~08:05 CEST — source inspection CONFIRMS the mechanism on all 3 backends

- x86_64/jit.c pool_get: `off = (cg->pool_len + 63) & ~63`; gaps [pool_len, off) never written;
  pool buffer grows via realloc (heap memory). Emission copies `cg.pool_len` bytes verbatim
  into the image (`em_byte(&cg.em, cg.pool[i])` loop). Gaps land in the emitted image.
- aarch64/jit.c: identical pattern (64B align, same verbatim copy loop).
- riscv/jit.c: identical pattern (16B align, same verbatim copy loop).
- Commit 7bece42 trap-fills only the dead space between `ret` and the pool base
  (`em_trap_align`) — it does NOT touch inter-entry gaps inside the pool image. L2 is a
  distinct region; no overlap.
- Gap-prone entries (x86): CK_TAILMASK (520B, 56B gap each), CK_NIBLO/HI at vl=16/32,
  CK_MASK/CK_ONE/CK_MSHIFT at vl=16/32, CK_CRCFOLD (16B, 48B gap each).
  First entry has no gap (pool starts at 0); gaps appear from the 2nd entry onward.

## Next: empirical reproducer (native, hostile heap history between two identical builds)

## 2026-10-12 ~08:20 CEST — VERDICT: CONFIRMED (empirical, airtight)

Reproducer `l2_repro/repro_l2.c`: builds the same raid6_rs(6) candidate twice with
hostile heap churn (512x malloc(8192)+free filled 0xA5, then 0x5A, plus 4096-class)
between builds; compares FULL image (code_size + const_size bytes).
Full log: l2_repro/repro_unfixed.log (exit 1).

- 8/32 candidates DIFFER, all in the pool region only; code region always identical.
- First diff offsets are exactly gap starts (e.g. cand17 isa=1/AVX2 vl=32: code=1600,
  first diff at 1632 = code_size+32 = start of [32,64) gap after first 32B entry).
- Differing bytes are the churn patterns (0xB4/0x4B = 0xA5^0x11 / 0x5A^0x11 from the
  4096-class churn, reaching the pool buffer via free-chunk coalescing) — proving the
  bytes are uninitialized heap memory copied into the image.
- 24/32 candidates (all AVX512, vl=64, every entry a 64B multiple) identical — no gaps
  exist there, consistent with the mechanism.
- No other pool-like verbatim copies exist (crc_fold.c/crc_fast.c clean).

7bece42 overlap: NONE — trap-fill covers ret-to-pool padding only, not inter-entry gaps.

## Fix plan (root cause, all three backends)
Zero-fill the gap [pool_len, off) in pool_get immediately after the capacity
realloc and before the entry memcpy. Gaps are never executed/read, so no
executed/read byte changes; 64B (x86/aarch64) / 16B (riscv) alignment contract
preserved (offsets untouched). riscv is gap-free by construction today
(rep is always a 16-multiple); the memset is a no-op there, kept for uniformity.

## 2026-10-12 ~08:45 CEST — fix applied + red-green proven

Fix: zero-fill of the [pool_len, off) gap in pool_get, all three backends
(src/x86_64/jit.c, src/aarch64/jit.c, src/riscv/jit.c), immediately after the
capacity realloc and before the entry write. riscv gap is empty by construction
(rep always a 16-multiple); memset is a no-op there, kept for uniformity.

Red-green (test_constpool_determinism, new in tests/ + Makefile PORTABLE_TESTS):
- RED: fix stashed -> test exits 1, 8/32 candidates (all SSE/AVX2) differ in pool
  region, bytes 0xb4 vs 0x4b (the churn patterns). Log: l2_repro/test_red_unfixed.log
- GREEN: fix applied -> exit 0, 32/32 byte-identical. Reproducer agrees
  (l2_repro/repro_fixed.log vs l2_repro/repro_unfixed.log).

Full native suite: DEFERRED — box load 14.02 >= 12 backoff gate at 08:47 CEST.
Running targeted codegen-shape subset now; full `make test` queued for when the
gate clears.

## 2026-10-12 ~09:10 CEST — full suite: 1 PRE-EXISTING failure, unrelated to L2

`make test` stops at test_remat: "rs(6,3) all-trivial nibble U=8 build: remat twin
requested but its code path was not taken (phantom twin)". Proven pre-existing:
stashed ALL lane changes, rebuilt pristine 7203cf1 lib, test_remat fails
identically. Not caused by the L2 fix (the memset provably cannot affect codegen
path selection; failure is byte-identical with/without the fix). Out of scope for
this lane — reported for the coordinator as a separate bug.
Re-running full suite with `make -k test` to verify everything else is green.
