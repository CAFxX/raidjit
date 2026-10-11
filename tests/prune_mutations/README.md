# Prune mutation evidence (M1–M4)

Each `M<N>.log` records a real mutation run against [src/prune.c](../../src/prune.c) on
2026-10-02: the exact source diff applied, the rebuild, and the full
`test_prune` output. The pristine [src/prune.c](../../src/prune.c) was restored after each
run (verified with `diff`).

Reproduce any of them, e.g. M1:

```
cp src/prune.c /tmp/prune.c.orig
# apply the diff shown at the top of M1.log to src/prune.c
make build-x86_64/test_prune
./build-x86_64/test_prune
cp /tmp/prune.c.orig src/prune.c   # restore
make build-x86_64/test_prune
```

## What each mutation proves

- **[M1](M1.log)** (`twin_dominates` without the `U` equality -- cross-U
  pruning): `unit_cross_u_twin_no_prune` FAILS (the dedicated
  P_XOR-like U-invariant catcher: the U=1 twin pareto-dominates the
  U=8 plain on the triple and prunes it). On the real corpus, the
  Part-3 **tripwire fires** on `delta/64K` (`unpruned argmin-time
  't pupd 0+2 delta avx512+nibble U=8+nt' was pruned`) and winners flip
  at the 3%/6% noise windows. `unit_u_variants_survive` does NOT fail
  under M1 -- that fixture's code sizes grow with U, so no cross-U
  dominance exists there; this corrects the earlier (wrong) claim that
  M1 breaks it.
- **[M2](M2.log)** (`twin_dominates` symmetric in `tl` -- prune twin in favor of
  plain): `unit_mulacc_plain_dominant_twin_survives` FAILS. The P_XOR
  direction is contained by the kind guard (`unit_pxor_twin_survives`
  still passes); on the pre-kind-guard code this mutation instead broke
  `unit_pxor_twin_survives` and flipped `xor2/1M` in Part 3.
- **[M3](M3.log)** (`twin_dominates` without the triple gate -- prune plain
  whenever a same-class twin exists): 3 failures --
  `unit_twin_not_pareto_survives`, `unit_eps_ties_survive` (eps ties
  get pruned), and `unit_mulacc_plain_dominant_twin_survives`.
- **[M4](M4.log)** (`prune_versions` keeps the plain, prunes the twin --
  inverted keep): 12 unit checks FAIL, starting with
  `unit_twin_prunes_plain`. Part 3 shows no flips on this corpus (the
  replay winners are GFNI, not nibble twins) -- the unit tests are the
  M4 detector.

M5 (api.c hook removed) is not a correctness mutation and has no log
here: Part 2 still passes by design (pruning only removes), but
`hook_wiring` (Part 2b) fails because the verbose prune line
disappears -- the production-path wiring is pinned by that test.
