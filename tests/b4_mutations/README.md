# B4 mutation evidence (M1–M3)

Each `M<N>.log` records a real mutation run on 2026-10-03: the exact
source diff applied, the rebuild, and the failing test output. The
pristine sources were restored after each run (verified with `diff`).

Reproduce any of them, e.g. M1:

```
cp src/plan.c /tmp/plan.c.orig
# apply the diff shown at the top of M1.log to src/plan.c
make build-x86_64/test_b4_prefetch
./build-x86_64/test_b4_prefetch
cp /tmp/plan.c.orig src/plan.c   # restore
make build-x86_64/test_b4_prefetch
```

## What each mutation proves

- **[M1](M1.log)** (`plan_candidates` drops all prefetch twins):
  `test_enumeration` FAILS (`want one prefetch twin per base (0 vs
  22)`). Proves the enumeration-count assertion is live: a planner
  regression that silently stops emitting prefetch twins is caught.
- **[M2](M2.log)** (`em_prefetch` emits UD2 `0F 0B` instead of PREFETCHT0
  `0F 18`): `test_past_end` dies with SIGILL (exit 132). Proves the
  prefetch-past-end self-attack actually executes the emitted prefetch
  path at tiny lengths — a faulting prefetch emission would be caught,
  not silently skipped.
- **[M3](M3.log)** (`plan_candidates` drops all NT twins):
  `test_nt_prefetch_combo` FAILS (`combo: missing NT twin`). Proves
  the four-combo enumeration assertions are live; also exercises the
  `ntested > 0` guards in the new large-recovery test (without them,
  `test_one`-style loops over zero NT twins would pass vacuously).
