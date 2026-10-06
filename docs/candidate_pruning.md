# Candidate pruning by static dominance (track A1, item 34)

## Problem

`rj_compile` benchmarks every legal codegen candidate (`bench_select`:
validation at 6 lengths + warm-up + interleaved timing rounds at 3 working
sets per candidate). On x86-64 a P_MULACC config enumerates up to 64
candidates; the benchmark dominates compile (selection) wall time. Some
enumerated candidates are *statically dominated*: a twin candidate does
the same work with fewer uops, no more memory traffic, and smaller code.
Benchmarking the dominated twin cannot change the winner; it only burns
wall time.

## Design

New module `src/prune.c` / `src/prune.h`, called from `rj_compile`
(`src/api.c`) after all versions are built and **before** `bench_select`:

```
plan_candidates -> jit_build_version x N -> prune_versions -> bench_select
```

### Dominance rule (narrowed 2026-10-02 after a rejection; kind guard added 2026-10-02)

Version B (plain, `ternlog=0`) is pruned iff there exists a version A
(`ternlog=1`) such that **all** hold:

1. **Twin pair**: `A.cand.ternlog == 1`, `B.cand.ternlog == 0`, and
   `A`/`B` agree on every other candidate axis: plan kind, `isa`,
   `mul`, `U`, `nt`, `prefetch_dist`, `remat`. Compared field-by-field
   (not `memcmp`: struct padding is not guaranteed zeroed). The rule
   NEVER fires across plan kinds, across U, across NT, across ISA/mul,
   across prefetch distance, across remat, or in the reverse direction
   (plain pruning a twin).
2. **Plan-kind guard**: the pair's plan kind is P_MULACC or P_DELTA --
   the only kinds where the twin direction is validated. Enforced by
   code in `twin_dominates()`, not by the triple values: if the model
   ever claims a P_XOR (or other-kind) twin dominates, pruning is
   refused, with a verbose log when `PRUNE_VERBOSE` is set.
3. **Triple dominance** on the static cost model:
   `A.uops_per_byte <= B.uops_per_byte + eps`,
   `A.mem_per_byte  <= B.mem_per_byte + eps`,
   `A.code_size     <= B.code_size`,
   with at least one strict (`eps = 1e-9`, float-comparison dust -- NOT
   the 1.5% `select_by_time` noise floor).

Survivors are compacted to `vs[0..ret-1]` preserving original relative
order; the return value is the new count. Pruned versions are finalized
inside the call (`jit_free_version`, releasing their executable
mappings); the caller owns only the survivors. A defensive fallback
keeps `vs[0]` if the survivor set were ever empty (unreachable: the
twin itself is never pruned by this rule).

### Why this narrow (rejection of the broader rule)

A broader rule was implemented first: triple dominance within the same
(isa, mul) class, across U and in both twin directions. The adversarial
test matrix (`tests/test_prune.c`, Parts 2+3) REJECTED it on 2026-10-02
— it pruned real benchmark winners:

- **P_XOR U-invariance**: the model reports identical uops/B and mem/B
  for every U (0.1250/6.0 for dense 4+2 recover), so U=1 "dominated"
  U=8..U=2 on code size alone — yet U=4/U=5 win the benchmarks.
  Unrolling benefits are invisible to the model; cross-U dominance is
  unsound.
- **P_XOR twin competitiveness**: the model ranks plain above the
  `+ternlog` twin (0.1250/6.0 vs 0.1406/7.0), but the twin WON the
  1 MiB benchmark. Static uop counting does not reflect VPTERNLOGD's
  true cost in the zero-up-front P_XOR codegen. Pruning twin-in-favor-
  of-plain is unsound.
- **P_DELTA AVX2**: an AVX2 nibble winner was pruned by the broader
  rule's cross-U logic.

The surviving rule fires only where codegen is structurally identical
except for the multiply instruction (the twin pair), and only in the
direction where the twin is a pareto improvement (fewer uops, smaller
code, same mem). The zero-flip empirical gate below is what qualifies
this direction; the P_XOR direction is excluded because the twin is
empirically competitive there.

### What it prunes today (x86-64)

- **P_MULACC, AVX-512 nibble**: the `+ternlog` twin folds the nibble
  multiply into VPTERNLOGD — strictly fewer uops, same memory traffic,
  smaller code (rs(6,3) rec U=8: 1.346 vs 1.502 uops/B, 32.016 vs
  32.016 mem/B, 5440 vs 5824 code). The **plain** candidate is pruned;
  the twin survives. 64 -> 48 candidates on rs(6,3) recover.
- **P_DELTA, AVX-512 nibble**: the twin reports an IDENTICAL triple to
  plain (0.1250/6.0, same code) — no strict dominance, so the triple
  gate correctly yields no pruning. (The multiply is trivial in
  P_DELTA; the twin is not a pareto improvement.)
- **P_XOR**: the twin is never pruned (rule direction); plain is never
  pruned by the twin (twin is not pareto-better). No-op, correctly:
  the twin wins at 1 MiB.
- **P_HORNER**: no ternlog twins enumerated; no-op.
- **AArch64/RVV**: no ternlog twins; no-op.

Deliberately **not** pruned (all verified against the rule):

- U variants (rule requires identical U; see rejection above).
- NT twins (rule requires identical nt; NT-ness is invisible to the
  model and NT twins win benchmarks).
- Cross-ISA / cross-`mul` (rule requires identical isa+mul; GFNI
  strictly dominates nibble on the raw triple yet nibble wins
  benchmarks — incommensurable uop kinds).
- Prefetch variants (rule requires identical prefetch_dist; the model
  does not capture latency-hiding benefit).
- P_XOR twins in either direction (see rejection above).
- **`+permi2b` twins (track B2, SOTA item 15)**. The twin axis is
  `permi2b`, not `ternlog`: `twin_dominates()` compares versions
  field-by-field and requires them to agree on every axis except the
  ternlog flag — `permi2b` is pinned like every other axis (added by
  the B2 escape fix; before that a (permi2b=1,ternlog=1) /
  (permi2b=0,ternlog=0) pair slipped the twin relation and only the
  triple gate's values stopped a cross-axis misprune), so a permi2b
  twin and its plain twin never form a pruning pair: the rule is
  inert on this axis by construction. This is deliberate: the pruner
  only ever prunes plain-in-favor-of-twin where the twin direction is
  validated (P_MULACC/P_DELTA +ternlog), and a permi2b twin is never
  statically better (6-7 vs 5-6 uops per (i,j,u); extra vmov for
  vpermi2b's index clobber), so no dominance argument applies. The
  twin is CPUID-gated on VBMI
  (`src/x86_64/cpu.c`: leaf 7 ECX bit 1 + avx512f + ZMM OS state) and
  left entirely to `bench_select`'s empirical noise rule.
  Measured on AMD EPYC 9D25 (track Bx86, `B2_duel.log`): the twin
  loses in all 6 duels — cache-fit gaps -0.7%/-1.9%/-10%
  (custom(4,2) U=1/U=8, custom(5,3) U=8 at 64K; bench_select picks
  the twin 0/12 in every clean duel), streaming duels noise-dominated
  — verdict REJECTED-WITH-EVIDENCE on this host. Since the 2026-10-03 B2
  follow-up the twin is no longer enumerated in stage-1 (that cost +50%
  stage-1 builds: 192 vs 128 candidates on custom(4,2) encode):
  `src/api.c` builds it from the stage-1 winner and duels it with the
  standard `bench_select` protocol (+1 build + 1 duel, only when the
  winner is an AVX-512 MUL_NIBBLE MULACC/DELTA candidate with a
  nontrivial coefficient on a VBMI host), preserving the per-host
  empirical re-decision at a fraction of the compile cost; nothing
  about selection prefers it.

### Why "provably" dominated (conservativeness argument)

The pruned candidate B (plain) has a surviving twin A that is identical
in every candidate axis (plan kind, isa, mul, U, nt, prefetch_dist,
remat) and differs only in the multiply instruction (nibble sequence vs
one VPTERNLOGD), with no-worse uops/byte, no-worse mem/byte, no-worse
code size, and strictly better somewhere. B can only win the benchmark
by measuring strictly better than A by more than the noise window
despite A doing strictly less work with the same structure.

### Safety precondition (explicit)

The argument above is a plausibility argument, not a proof: pruning is
safe **iff no pruned version would be the argmin-time candidate**.
Removing a pruned argmin re-centers `select_by_time`'s noise window and
can flip the winner (attacker result, 2026-10-02: 19,453 flips / 6M
fuzz trials, 100% of flips involving a pruned argmin). The zero-flip
replay gate is therefore corpus-bound: it shows no winner change on the
validated corpus/host, it does not prove there never is one.

Two mechanisms make future violations LOUD instead of silent:

1. **The kind guard is code, not triple values.** The twin direction is
   validated only for P_MULACC / P_DELTA; `twin_dominates()` refuses to
   prune any other plan kind even if the model claims triple dominance
   (with a verbose log under `PRUNE_VERBOSE`). A model/codegen change
   that makes a P_XOR twin look dominant cannot silently start pruning
   it.
2. **The Part-3 argmin tripwire.** `tests/test_prune.c` Part 3 asserts,
   on the UNPRUNED run, that the argmin-time candidate is not in the
   pruned set -- in addition to winner equality. Any future
   model/codegen change that makes a pruned version the argmin fails
   this test instead of flipping winners silently.

The `RJ_NO_PRUNE` switch is the escape hatch for CPUs where the model
misranks a twin pair (as it does for P_XOR, which is why that direction
is excluded).

### Per-microarchitecture re-validation requirement (2026-10-06)

The tripwire is host-specific by construction, and MUST be re-run per
microarchitecture family -- not just per code change:

1. The safety precondition ("no pruned version would be the argmin-time
   candidate") is a statement about measured times on one host. Static
   uop counts do not capture microarchitectural costs: the P_XOR twin
   rejection exists precisely because VPTERNLOGD's true cost differs
   from its static count, and that gap is microarchitecture-dependent
   (port pressure, uop-cache behavior, and memory-subsystem timing all
   vary by family). A twin pair that is safely prunable on AMD EPYC 9D25
   can have a different argmin on Intel Ice Lake, AMD Zen 5, or an ARM
   Neoverse core.
2. `tests/test_prune.c` Part 3 therefore qualifies the rule for the
   host it ran on, not for the rule in the abstract. Deploying raidjit
   on a new microarchitecture family (different vendor, or a new
   microarchitecture from the same vendor -- e.g. Zen 4 -> Zen 5)
   REQUIRES re-running Part 3 there and confirming (a) zero winner
   flips and (b) the argmin tripwire stays quiet. A tripwire fire on
   the new host means the prune rule is unsound there: ship with
   `RJ_NO_PRUNE=1` on that family until the rule is re-validated, and
   record the family in the Limitations section below.
3. Code changes that alter any backend's emission, the static cost
   model (`uops_per_byte` / `mem_per_byte` / `code_size`), or the twin
   relation in `twin_dominates()` re-arm the tripwire on EVERY
   supported host: re-run Part 3 per family before shipping.
4. QEMU results do not qualify a microarchitecture: QEMU's timing is
   not the silicon's timing. Cross-ISA functional tests under QEMU
   validate correctness only; the prune tripwire needs real hardware
   (or a cycle-accurate model of the target family).

This is a MUST, not a SHOULD: a silent argmin flip on an unvalidated
family is exactly the failure mode the tripwire exists to catch, and
skipping the per-family run converts it back into a silent one.

### Disable switch

`prune_versions(vs, nv, flags)` with `flags & PRUNE_DISABLE`, or the
environment variable `RJ_NO_PRUNE` set to any non-empty value other
than exactly `"0"` (checked once per call inside `prune_versions`, so
later consumers of the API get it for free). The contract is precisely
"non-empty AND != `"0"`": `"1"`, `"false"`, `"00"`, `"0 "` (trailing
space) all disable pruning; only `""` (empty/unset) and `"0"` leave it
enabled. Either mechanism disables pruning for that call (returns `nv`
unchanged). Tested in `tests/test_prune.c` (including the
`"false"`/`"00"`/`"0 "` cases).

### API (for the survivor-set track)

```c
/* src/prune.h */
#define PRUNE_DISABLE 1u
#define PRUNE_VERBOSE 2u   /* log kind-guard refusals to stderr */
#define PRUNE_EPS 1e-9
int prune_versions(version_t *vs, int nv, unsigned flags);
```

Takes built versions (the cost model lives on `version_t`), compacts
survivors in place preserving order, returns the new count. `nv == 0`
returns 0. Pruned versions are finalized inside the call
(`jit_free_version`); the caller must not touch them afterwards. This
is intentionally the whole API: the survivor-set enumeration track
calls it on its version list before benchmarking.

## Falsifiable ship criterion

1. **Zero flips + argmin tripwire**: `tests/test_prune.c` Part 3 runs
   pruned vs unpruned `select_by_time` over a topology x length matrix
   (P_XOR n_out=1, P_XOR n_out=2 with the VPTERNLOGD twin, P_HORNER,
   P_MULACC, P_DELTA, a prefetch_dist!=0 config, and large recovery
   configs rs(10,3), rs(20,3), rs(10,4)-Cauchy; bench lengths 64 KiB
   and 1 MiB) on real measured times and asserts (a) identical winners
   across noise windows {1.5%, 3%, 6%}, (b) the full run's winner
   survives pruning, and (c) the TRIPWIRE: the unpruned run's
   argmin-time candidate is not in the pruned set. (c) is the loud
   detector for the safety precondition above: any future model or
   codegen change that makes a pruned version the argmin fails the
   test instead of flipping winners silently. Part 1 unit-tests the
   rule on synthetic triples (including the kind guard and the
   U-invariance M1 catcher); Part 2 smoke-tests `rj_compile`
   with/without pruning and checks each winner against the scalar
   reference; Part 2b pins the api.c hook wiring via the verbose prune
   line. Any flip or tripwire fire fails the test and rejects the
   rule. (Winner equality across two *independent* benchmark runs is
   not asserted: noise alone flips those even with pruning disabled.)
   "No winner change" is therefore a statement about the validated
   corpus/host, not a theorem; `RJ_NO_PRUNE` is the escape hatch.
2. **Measured compile-time reduction**: `tools/perf_guard
   --compare` over the 18-config corpus: base = `RJ_NO_PRUNE=1` run,
   new = pruning-enabled run (both committed under
   `tools/perf_guard/`, methodology in
   `tools/perf_guard/COMPILE_TIME_NOTE.md`). Per-size GB/s medians
   must stay within the base [min,max] bands (throughput parity --
   pruning must not change what wins); per-config compile_s medians
   must not exceed the base band top. The load-independent structural
   fact: 25% fewer candidates benchmarked on P_MULACC recovery
   (64 -> 48 on rs(6,3) recover; 128 -> 96 with prefetch twins).
   The 2026-10-02 wall-clock runs were conducted under heavy
   shared-VM contention and their total (-3.5%) is noise-dominated;
   no specific wall-clock % is claimed -- see the note. Earlier
   uncommitted claims ("-15.6%", "149.8s vs 97.9s") are withdrawn.
3. **Switch tested**: `RJ_NO_PRUNE=1` and `PRUNE_DISABLE` both yield zero
   pruning (unit-tested, including the exact "non-empty and != "0""
   contract).

## Limitations

- The dominance claim is relative to the static cost model and is
  validated host-specifically (AMD EPYC 9D25 + QEMU). The P_XOR twin
  rejection shows the model can misrank; a CPU where it misranks a
  P_MULACC twin pair would need `RJ_NO_PRUNE`.
- Pruning runs after `jit_build_version`, so it saves benchmark wall
  time, not codegen time.
- No-op for P_HORNER, P_XOR, P_DELTA, AArch64, RVV — the win
  concentrates on P_MULACC on x86-64 (AVX-512 nibble twin pairs).
