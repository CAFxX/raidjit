# Length-specialized runtime dispatch (track A2, item 33)

## 1. Goal

Today every `rj_fn` is one code version: a vector loop plus a tail path for
`len % group != 0` (scalar byte tail on x86-64/AArch64; predicated
strip-mining tail on RISC-V). Lengths that are exact multiples of the vector
group still pay for the tail's existence: the exit branch chain, the dead
tail code in icache, the scalar-tail register saves, and the log/exp
constant tables that exist only for the scalar tail.

This track asks one empirical question: **does a "notail" twin of the
winning candidate — identical vector body, no tail code at all — beat the
general version through the compile-time benchmark?** If yes, `rj_compile`
keeps both versions and `rj_get_fn` returns a dispatcher that routes each
call to the fastest variant admitting that call's length. If no, zero
behavior change: the general version is returned exactly as today.

Nothing here is decided by static-model appeal. The notail twin is a
candidate like any other: it must validate against the scalar reference and
win under the noise-aware selection protocol (bench.h) to ship.

## 2. Specialization set

| Variant | Tail policy | Admits | Codegen delta vs general |
|---|---|---|---|
| GENERAL | `TAIL_SCALAR` (x86/AArch64) / predicated strip-mining (RISC-V) | every `len` | — (status quo) |
| NOTAIL | `TAIL_NONE` | only `len % group == 0` | tail branch chain, tail body, scalar-tail register saves, and log/exp pool tables omitted |
| PREDTAIL | `TAIL_MASKED` (SVE `whilelt`, track B1sve) | only tail lengths (`len % group != 0`) | scalar byte tail replaced by a `whilelt`-predicated vector chunk loop (`gen_pred_tail`); disjoint from NOTAIL |
| MASKEDTAIL | `TAIL_MASKED` (AVX-512 k1, track B1x86) | only tail lengths (`len % group != 0`) | scalar byte tail replaced by a k1-masked vector chunk loop (`gen_tail_masked`); disjoint from NOTAIL |

`group` = bytes per full vector-loop iteration = `U * VL` (x86-64:
`U*16/32/64`; AArch64: `U*16` NEON, `U*sve_vl` SVE; RISC-V: `U*VLMAX` with
VLMAX read from `vlenb` at compile time). U is not necessarily a power of
two (planner emits 1..8), so `group` is not necessarily a power of two
either; the admission test is `len % group == 0`, strength-reduced to
`len & (group-1)` when `group` is a power of two. The pow2-ness is computed
once at dispatcher-build time (plan/JIT time), never per call — per the
invariant that nothing is computed at runtime that could be done earlier.

## 3. Falsifiable ship criterion

The dispatcher ships **iff all of these hold**, otherwise the change is a
documented rejection with evidence:

1. The notail twin builds cleanly for the winning candidate (a build
   failure degrades silently to general-only).
2. The twin validates bit-exact against the scalar reference at
   group-multiple lengths (same gate as every candidate: a wrong twin
   fails compilation).
3. In a head-to-head `bench_select` of `{general, notail}` under the
   standard protocol (correctness first, interleaved rounds, several
   working-set sizes, noise-aware `select_by_time`), the notail twin wins.
   A tie inside the noise window resolves by the existing tie-break
   (uops/byte, mem/byte, code size), which the smaller notail version wins
   on code size.
4. `tools/perf_guard` on the 18-config corpus is REPORTED, not gating.
   The 2026-10-02 post-change run (`tools/perf_guard/
   post_2026-10-02_A2remediation.json`, comparison script
   `compare_m2.py`, verdicts in `m2_comparison_2026-10-02.log`) shows
   the baseline pass-to-pass bands are dominated by winner-selection
   variance — different passes pick different winners, so bands span up
   to ~3x (e.g. raid6(8+2)/enc 64 KiB baseline [50.27,145.77] GB/s) and
   band membership cannot gate a change. Post-change deviations go in
   both directions (5 below-band, 6 above-band across the 36 gated
   (config, size) cells), consistent with noise; no mechanism in this
   remediation can change steady-state throughput (validation buffer
   sizing and the group_bytes==0 reject are compile-time-only, the
   dispatch_call fallback removal deletes unreachable code, and the
   ATK-1 trampoline assert is one predictable branch, compiled out under
   NDEBUG). One 4 KiB cell (rs(20,3)/enc, 0.69x of the baseline minimum)
   trips the large-drop rule on the VM-noisy size; the same config's
   64 KiB / 1 MiB cells split below-band / in-band — noise, not a
   regression signature.
5. `make test`, `make test-asan`, `make test-qemu` green, including the new
   `tests/test_dispatch.c` with mutation/red-green evidence.

Rejection (twin loses or ties-but-loses-tie-break) is a complete outcome:
report the measured numbers and stop.

## 4. Portable C dispatcher (src/dispatch.c, src/dispatch.h)

`rj_fn` carries no context pointer, so a per-`rj_jit` C dispatcher cannot be
a single static function closing over its variants. The portable solution
is a small pool of static trampolines (macro-generated, one per slot),
each bound to a slot holding that `rj_jit`'s variant table:

```c
typedef enum {
    TAIL_SCALAR = 0,  /* byte scalar tail (x86/AArch64); admits every len
                        * when no TAIL_MASKED twin is installed, else only
                        * multiples (derived to TAIL_SCALAR_MULT at
                        * create time). RISC-V general variant uses
                        * predicated strip-mining but the same contract,
                        * so it reuses this policy. */
    TAIL_NONE   = 1,  /* no tail code; admits only len % group == 0 */
    TAIL_MASKED = 2,  /* fully predicated/masked vector tail (SVE whilelt
                        * twin, track B1sve; AVX-512 k1 twin, track
                        * B1x86): admits only len % group != 0. A
                        * distinct policy so routing and diagnostics can
                        * tell the codegen apart. */
    TAIL_SCALAR_MULT = 3, /* derived by rj_dispatcher_create only
                            * (callers must not set): TAIL_SCALAR
                            * deferred to the installed masked twin for
                            * tails; admits only len % group == 0. */
} rj_tail_policy;

typedef struct {
    rj_fn fn;             /* variant entry point, rj_fn ABI */
    rj_tail_policy tail;  /* tail policy (the plug-in hook) */
    size_t group;         /* full-iteration bytes; TAIL_NONE admission */
    int group_pow2;       /* precomputed at build time */
    size_t group_mask;    /* group-1 when group_pow2 */
    double gbps;          /* measured under the compile-time protocol */
} rj_variant;
```

Admission is the per-variant tail-policy interface the later track plugs
into:

```c
static inline int rj_variant_admits(const rj_variant *v, size_t len);
```

The dispatcher stores variants **sorted fastest-first** (by measured
`gbps` — the shootout for general/notail, the tail duel for a tail twin)
and calls the first variant admitting `len`. The
general variant admits everything when no masked twin is installed, so it
is the always-valid fallback; when a masked twin IS installed, the
general variant defers tail lengths to it (derived `TAIL_SCALAR_MULT`,
§12) and the masked twin is the only tail admitter. A new ISA's
masked/predicated tail plugs in by installing a `TAIL_MASKED` variant
with no dispatcher change. `rj_dispatcher_pick()` exposes the selection
for white-box testing.

Slot pool: 32 slots, mutex-guarded. Exhaustion degrades cleanly to
general-only (no dispatcher). The per-call cost of the dispatcher is one
`len % group` (or `&`) test plus one indirect call — measured and reported
in §7; it applies to every call, including ones routed to the general
variant.

**Lifetime footgun (ATK-1):** the dispatched `rj_fn` shares the `rj_jit`'s
lifetime — `rj_free` destroys the dispatcher and releases its slot, after
which the entry point is dangling. Never cache a dispatched `rj_fn` across
`rj_free`: the slot may be reused by a newer dispatcher, turning a loud
segfault into *silent wrong output* (the stale entry would route into
another plan's variants). Always re-fetch via `rj_get_fn` after
re-compiling. Debug builds abort loudly on use-after-free-before-reuse
(assert in the trampoline, zero cost under NDEBUG); slot reuse itself is
NOT detectable — the contract in `src/dispatch.h` is the only protection.

**Why not a JIT stub?** A few bytes of arch-specific dispatch code would be
faster (no extra call), but the track explicitly requires a portable C
dispatcher, and the stub would triple the arch-specific surface for a
follow-up to maintain. The C dispatcher keeps all dispatch policy in one
portable place.

## 5. Per-variant tail policy by measurement

The policy attached to each variant is not inherited from old
single-kernel numbers — it is whatever the shootout measured:

- The twin's `TAIL_NONE` policy is *defined* by its codegen (no tail code
  exists), but whether the dispatcher *routes* to it is decided by the
  benchmark: only a winning twin gets installed.
- `TAIL_MASKED` (track B1sve, SVE `whilelt` tail; track B1x86, AVX-512 k1
  tail): the dispatcher does not assume it beats scalar — the tail twin
  must win its own tail gate to be installed: `bench_tail_duel`, the
  ISA-agnostic tail duel (29 tail-heavy lengths, strict win outside the
  noise window, plus a large-size no-regression gate), for both the
  B1x86 masked twin and the B1sve predtail twin. A large-size shootout
  win is not sufficient: under length-aware routing (§12) an installed
  `TAIL_MASKED` twin deterministically owns 100% of tail lengths, so a
  twin slower than general at tail lengths would be selected for all of
  them — fastest-first routing cannot save it, because the general
  variant defers tail lengths to the installed twin. The twin admits
  only tail lengths once installed (see §12). Its rank in the
  fastest-first order still comes from measured `gbps`, not from the
  policy name.

## 6. Codegen changes (supporting edits)

`cand_t` gains `int notail` (a twin dimension like `nt`/`remat`; the
benchmark picks the twin empirically); `jit_build_version()` threads it
into each backend's `cg_t`, which omits, when set:

- x86-64 (`src/x86_64/jit.c`): the `ja tail_lab` becomes `ja done_lab`;
  the `tail_lab` bind / `cmp` / `gen_tail` sequence is not emitted;
  `need_logexp` is forced to 0 (log/exp tables exist only for the scalar
  tail). The R15 pool-base setup is unaffected: `need_mask` stays 1 for
  the vector body, so `need_r15` still holds.
- AArch64 (`src/aarch64/jit.c`): the counted loop's `cbz xN, tail` becomes
  `cbz xN, done`; tail bind/cmp/`gen_tail` omitted; the x19 save/restore
  for the P_HORNER scalar tail is skipped. `need_logexp` is deliberately
  NOT gated: it doubles as the "constant pool is needed" signal for
  HORNER/DELTA (`need_pool`), and the vector body loads nibble tables via
  the pool base — gating it broke the pool setup (caught by QEMU; the
  tables allocate lazily and are never emitted when `gen_tail` doesn't
  run, so keeping the flag costs nothing).
- RISC-V (`src/riscv/jit.c`): after the counted full-group loop, the
  remainder computation, strip-mining loop and single-chunk tail are
  omitted — control falls through to `done_lab`. (There is no scalar tail
  or log/exp pool on this backend.)

### 6.1 Predicated-tail twin (SVE, track B1sve)

`cand_t`/`version_t` gain `int predtail`, disjoint from `notail`
(`jit_build_version` asserts `!(notail && predtail)`). When set on SVE,
`gen_function` emits `gen_pred_tail` instead of the scalar tail:

```
x13 = len - x11                     // rem in 1..group-1
chunk:
    whilelt p1.b, xzr, x13          // new emitter em_sve_whilelt
    <vector body, cg->pred = 1>      // one lane per chunk (ulanes = 1)
    x11 += x12;  subs x13, x13, x12
    b.gt chunk                      // signed rem > 0
```

Loads/stores/arithmetic run under `p1`; `p0` (main loop) is untouched.
Lane-independent ops (AND, LSR, TBL, MOV, EOR3) stay unpredicated — sound
because every body op is lane-independent and stores are predicated, so
inactive lanes can never affect active lanes or memory (see
`~/workspace/raidjit-tracks/Bsve/DESIGN.md` §3.3). Two bugs found by the
new tests are documented there (§6): predicated `c_and3`/`c_lsr4` dropped
their source operand, and `gen_const_setup` was skipped for `len < group`
(the vector tail needs the mask/tables; the scalar tail doesn't).

`version_t` gains `int notail` and `size_t group_bytes` (true full-group
bytes: `U*VL` on x86/AArch64; `U * vlenb` on RISC-V, read once at build
time). On AArch64/SVE the VL is read via `cntb` at JIT-build time, not
from `prctl(PR_SVE_GET_VL)` — the latter is unreliable under qemu-user
and the generated code uses `cntb` at runtime, so they must agree; this
was latent (the old `cg.group` was unused by the SVE loop) until
`group_bytes` made it load-bearing. The loop bodies are byte-identical
between twins — only tail handling differs, so any divergence is a
codegen bug by construction.

No-op plans (`pl->noop`) never get a twin: the emitted function is already
just prologue/epilogue.

### 6.2 Masked-tail twin (AVX-512, track B1x86)

`cand_t`/`version_t` gain `int maskedtail`, disjoint from `notail` and
`predtail` (`jit_build_version` asserts all three pairwise). When set on
AVX-512, `gen_function` emits `gen_tail_masked` instead of the scalar
tail. New emitters (`src/x86_64/emit.c`):

- `em_kmovq_km` — `KMOVQ k, m64` (VEX.0F.W1 90 /r), loads the 64-bit tail
  mask from the constant pool into an opmask register.
- `em_vmovdqu8_ld_masked` / `em_vmovdqu8_st_masked` — byte-granular
  masked load/store. Loads use `{k1}{z}` (zeroing) so masked-off lanes
  read as zero; stores use `{k1}` (merging) so bytes past the tail are
  never written. The `VMOVDQU8` (F2) form is mandatory: `VMOVDQA64`
  masks at 64-bit granularity, which is wrong for byte tails (caught by
  `tests/test_maskedtail.c` during development).
- `em_vpshufb_masked` — masked nibble shuffle `{k1}{z}` (for the future
  nibble/GFNI masked-tail gates; implemented, byte-exact vs GAS 2.42,
  not yet wired into codegen).

New pool kind `CK_TAILMASK`: 65 qwords, `mask[i] = (1ULL << i) - 1` for
`i < 64`, `mask[64] = ~0ULL`. The chunk loop (entered with `r11` =
group-aligned byte offset, `rcx` = len):

```
r10 = rcx - r11                        // tail bytes, 1..group-1
chunk:
    rdx = min(r10, 64)
    kmovq k1, [r15 + maskoff + rdx*8]   // k1 = low rdx bits set
    zero the u=0 accumulator slots      // masked-off lanes stay zero
    per input: vmovdqu8 t{k1}{z}, [in]; acc ^= t   // unmasked XORs
    per output: vmovdqu8 [out]{k1}, acc            // merging stores
    r11 += rdx; r10 -= rdx; jnz chunk
```

Chunk starts are always 64-byte aligned; the unaligned `vmovdqu8` forms
are used (no penalty at aligned addresses), and masked-off lanes never
fault (verified empirically with a guard-page probe on AMD EPYC 9D25,
`tests/test_maskedtail.c`).
`need_r15` is ORed with `maskedtail` (the mask table needs the pool
base; P_XOR otherwise never uses r15 — a missing OR segfaulted during
development). `need_logexp` is forced to 0 for the twin (no scalar
tail), like `notail`. Only P_XOR is implemented; nibble/GFNI masked
tails are separate benchmark gates (not implemented).

## 7. Benchmark integration (src/bench.c, src/api.c)

Two-stage, all inside `rj_compile` when `o.benchmark` is on:

1. **Stage 1 (unchanged):** build all candidates, `bench_select` picks the
   winning candidate as today.
2. **Stage 2 (twins):** build the tail twins of the winning candidate —
   the notail twin on every ISA (as before), plus the predicated-tail
   twin for SVE winners (track B1sve) and the masked-tail twin for
   eligible AVX-512 winners (track B1x86) — and run `bench_select` over
   `{general, notail}`. The predicated/masked tail twins are NOT judged
   by the shootout (at its large sizes the tail is a negligible
   fraction, so the standard protocol can neither install nor reject
   them meaningfully); each is gated by its own tail duel
   (`bench_tail_duel`, ISA-agnostic: tail-heavy lengths, a strict win
   outside the noise window, plus a large-size no-regression gate). A
   duel-winning tail twin is installed for tail lengths; a duel loss
   drops it (tail traffic falls back to the scalar tail). A twin that
   builds but fails validation fails compilation; a twin that fails to
   build degrades silently to general-only. Supporting changes:
   - *Validation* becomes per-version: a `notail` version is validated
     only at group-multiple lengths (`group`, `2*group`,
     `round_down(vmax, group)`, plus an offset-start case at a
     group-multiple length). A general version keeps the existing
     `{1, group-1, group, group+1, vmax}` + offset-start coverage.
     A `predtail` version validates at every length (it admits all).
     Scratch buffers are sized for `max(vmax, 2*group_bytes)` across
     versions (M1: the old fixed-`vmax` sizing was a heap over-read /
     over-write whenever `2*gb > vmax`).
   - *Timing* becomes per-version: a `notail` version is timed at
     `round_down(size, group)`; per-version byte totals keep each
     version's reported GB/s honest. Honest caveat: the selection score
     is the RAW total time (un-normalized), so when `group` does not
     divide the nominal size the twin times slightly fewer bytes and
     gets a systematic edge of less than one group's worth of work
     (~0.04% at 1 MiB with a 512 B group — negligible, but real and not
     hand-waved away). It is exact whenever the group divides the size
     (the common case: bench sizes are powers of two ≥ 4096 and groups
     are small); the reported GB/s numbers are unaffected either way
     because they divide by each version's own byte total.
   - If a twin wins its gate, `rj_compile` keeps the general version plus
     every installed twin and installs the dispatcher (fastest-first);
     routing picks the fastest *admitting* variant. Each twin is judged
     where it serves — the notail twin at large sizes, the tail twins at
     tail-heavy lengths — so no twin is installed on the strength of a
     measurement that never exercised its code. Otherwise the twins are
     freed and behavior is bit-identical to today. The two-variant case
     behaves exactly as before.

With `o.benchmark == 0` (heuristic select) no twin is built: there is no
measurement to gate it on. Compile-time benchmark cost roughly doubles for
the winning candidate only — documented, not hidden.

`rj_get_info` reports the winning candidate's info as today; `desc` gains
`" +notail"` / `" +predtail"` suffixes for each installed twin, so the
dispatch decision is observable.

One implementation subtlety: `vs[]` skips failed candidate builds, so the
stage-1 winner index must never be reused on `cands[]`; the twin is built
from `vs[winner].cand` (the winner's own candidate travels with the
version). A `NOTE` comment guards the site and `test_twin_source_mapping`
pins the pairing invariant (twin.desc == winner.desc + " notail").

## 8. Measured results (2026-10-02, AMD EPYC 9D25 VM)

Shootout (repo benchmark machinery: interleaved rounds, per-size medians,
noise-windowed selection, code-size tie-break), stage 2 on
`raid6(6) horner avx512+gfni U=8`, `bench_bytes=64 KiB`, 3 iters:

- general: 57.62 GB/s (noise 7.6%)
- notail twin: 57.53 GB/s (noise 4.1%), code 1536 vs 1920 bytes
- selection window: 7.6% → twin 0.16% slower is a **tie** → twin wins on
  the smaller-code tie-break → **dispatcher ships** (per §3's criterion).

Honest characterization: the twin does not make the loop faster (loop
bodies are byte-identical; uops/B and mem/B are equal). The win is the
tie-break, not a measured speedup. The tie-break win is deterministic,
not luck: the scalar tail emits raw `em_` calls with zero uops/mem
accounting, so the twins' uops/B and mem/B are exactly equal by
construction and the comparison always falls through to code size,
which the twin wins by the 384 B of omitted log/exp tables (1536 vs
1920 bytes). The dispatcher was installed in all
4 benchmarked e2e configs (encode, recover, parity-update, NT).

Dispatcher overhead: one admission test + one indirect call per `rj_fn`
invocation. Measured per-call delta vs a direct call is a few ns,
unmeasurable above this VM's noise floor (observed deltas: +73.5 ns,
+28.6 ns, −12.4 ns across runs — all noise). It is negligible next to
stripe times at the guarded sizes (64 KiB ≈ microseconds). (A future
optimization: JIT the dispatch into the prologue — rejected for now as
triplicated arch surface; see §4.)

## 9. Tests (tests/test_dispatch.c)

- **White-box selection** (no JIT): mock variants with recording stubs;
  assert `rj_dispatcher_pick` routes `len % group == 0` (incl. `len = 0`)
  to the notail variant and every other length to general; assert a
  dispatcher with only a general variant routes everything to it;
  assert slot exhaustion (33rd create) fails cleanly; assert
  `bench_select` rejects a notail version with `group_bytes == 0`
  (R1-m13: the timing loop would otherwise record garbage at nominal
  sizes for a twin the dispatch predicate admits nothing for).
- **Integration matrix** (JIT): topologies × {encode, recover(1 disk),
  parity-update} × lengths {0, 1, group-1, group, group+1, 2·group-1,
  2·group, 2·group+1, large non-multiple} × {benchmark on/off}, for up
  to 3 buildable candidates per config (distinct U variants, not just
  the first): every result bit-compared against `ref_run`. The
  `benchmark=1` cases exercise the real shootout+dispatcher path; `desc`
  suffix tells whether the dispatcher was installed. Includes the NT ×
  notail and prefetch × notail twin combinations (R2-m7a/b; the prefetch
  twin is skipped on RVV without Zicbop, where the planner emits no
  prefetch candidates).
- **M1 large-group validation** (JIT): a real notail twin with
  `group_bytes` doctored to the smallest true-group multiple ≥ 4096
  (simulating a wide-vector backend) through `bench_select`. Red-green:
  against the old fixed-`vmax` buffer sizing this trips ASan
  (heap-buffer-overflow in the validation fill loop — log in
  `tests/evidence/m1_asan_red_oldcode.log`); against the fixed sizing
  it passes.
- **Agreement**: when the dispatcher is installed, general-only `fn`
  (via a `benchmark=0` compile of the same plan) and dispatched `fn`
  agree bit-for-bit over the matrix. The suite asserts the dispatcher
  was installed in ≥ 1 benchmark config (R2-m7c), so a silent
  all-general run cannot pass as dispatch coverage.
- **Mutation evidence** (executed 2026-10-02, then reverted):
  (a) `dispatch_call` forced to always take variant 0 → fails loudly
  (exit 1, ~hundreds of failures, run-dependent — reviewer 2 observed
  240 and 305 on repeat runs; do not quote an exact count): white-box
  routing asserts fail (len=257/100/0 misrouted) and the e2e matrix
  fails at non-multiple lengths — proving the e2e test exercises
  dispatch routing, not just the general path.
  (b) notail loop-exit `CC_A` → `CC_AE` (x86: exits one group early) →
  11 failures, test exits 1: the direct twin matrix catches wrong twin
  output at group multiples, *and* the stage-2 benchmark validation
  rejects the corrupted twin ("produced wrong output at len=448"),
  failing compilation loudly — the wrong-code-must-never-ship path
  works.
  (c) red-green for the admission predicate: direct asserts on
  `rj_variant_admits` (pow2 `&` path, non-pow2 `%` path, group 0).
- **TAIL_MASKED admission**: unit tests pin that `TAIL_MASKED` admits
  ONLY tail lengths (`len % group != 0`; including `len = 0` rejected)
  for any group, that an installed `TAIL_MASKED` twin defers every
  `TAIL_SCALAR` variant to multiples (derived `TAIL_SCALAR_MULT`; see
  §12), and that `rj_dispatcher_create` accepts a variant set whose
  admit-everything variant is a lone `TAIL_SCALAR` (no masked twin).
  (`tests/test_dispatch.c:test_admits`, `test_length_aware_routing`)
- **Predtail twin source mapping**: `test_twin_source_mapping` pins
  `twin.desc == winner.desc + " predtail"` for a predtail twin built from
  the winner, mirroring the notail pin.
- **Predtail codegen** (`tests/test_sve_predtail.c`, new; QEMU SVE at
  flag values 128/256/512 — two distinct VLs, 128B and 256B, since
  qemu-user 8.2.2 clamps flag=512 to the 2048-bit max): shape tests
  (exactly one `WHILELT p1.b,xzr,x13`;
  tail LD1B/ST1B use `pg=1`; no scalar-tail STRB), exhaustive
  `1..2*group` correctness vs `ref_run` and vs the general twin for
  P_XOR/P_MULACC/P_HORNER/P_DELTA (with output canaries and input
  preservation), `x19` preservation for HORNER, and guard-page
  no-overrun tests. Red-green: corrupted WHILELT base → shape fails;
  single-iteration chunk loop → multi-chunk tails fail.

## 10. Follow-ups (not this track)

- Counted-loop notail: with `len % group == 0` guaranteed, the x86
  compare-based loop bound (`offset+group <= len` per iteration) could
  become a plain counter (`len >> lg(group)` when pow2). Not done here:
  it changes the loop shape, not just the tail, and needs its own
  benchmark gate.
- Masked/predicated tails plug into `rj_tail_policy`; they are
  duel-gated, not shootout-judged, and rank by duel-measured GB/s within
  their admission partition (`TAIL_MASKED` admits only tail lengths).
- SVE streaming-mode / SME: out of scope.

### 10.1 Integration notes for the A2+A3 merge (reviewer 2 guidance)

- Textual conflicts are expected ONLY on the three `snprintf(*desc)`
  lines (A3 appends its `"+xsched"` tag where A2 appends `" notail"` /
  `" +notail"`). Hand-merge: keep BOTH tags.
- All other hunks are adjacent but non-overlapping (A2 touches tail
  regions, A3 touches body regions); they should apply cleanly.
- Keep A2's `cntb`-VL hunk deliberately — see §11.0. Do NOT
  auto-resolve it away.
- Do NOT "harmonize" the `need_logexp` asymmetry: x86 gates it on
  `!notail`, AArch64 intentionally does NOT (it doubles as the constant-
  pool-needed signal; gating it broke the pool setup — see §6). The
  asymmetry is load-bearing, not an oversight.
- **UNTESTED COMBINATION — notail × xorsched twin:** A2's twin
  dimension and A3's xor-schedule dimension were developed in parallel
  and their combination (a notail twin of an xorsched candidate) has
  never been built or validated. Integration 1 must add a combinatorial
  test (xorsched winner → notail twin → bit-exact vs reference at group
  multiples) before shipping the merged tree.

## 11. Files touched

### 11.0 OUT-OF-BRIEF CHANGE — KEPT DELIBERATELY (integrator: read this first)

`src/aarch64/jit.c` reads the SVE vector length via `cntb` inline asm at
JIT-build time instead of `cpu->sve_vl` (from `prctl(PR_SVE_GET_VL)`).
This hunk is **out of the track's brief** (it changes ALL versions, not
just the notail twin) and is kept deliberately as a latent-bug fix.
The full story, including a wrong revert and re-revert during
Integration 1, is recorded here so nobody "simplifies" it again:

- `prctl(PR_SVE_GET_VL)` is unreliable under qemu-user. Measured
  2026-10-02 under qemu-user 8.2.2 with `-cpu
  max,sve-default-vector-length=N`: qemu interprets N as *bytes* (not
  bits), so the true emulated VL is N bytes, but `prctl` (and `rdvl`)
  report N/8 bytes. Verified ground truth: for N=128, `cntb` returns
  128 and an SVE `ld1b`/`st1b` round-trip under a `whilelo(cntb)`
  predicate moves 128 bytes. (`cnth`/`cntw`/`cntd` are likewise 8x —
  the counts come out in bits.)
- During Integration 1 the hunk was briefly reverted on the
  misdiagnosis that "cntb is 8x wrong and prctl is right". That broke
  `test_dispatch` under QEMU SVE: the notail twin was built with
  group=U*16 (wrong) while the generated code reads the true VL=128
  via `cntb` at runtime, so `niter = len/(128*U)` was 0 for
  len < 128*U and the tailless twin emitted nothing (twin mismatch at
  every admissible length). Restoring the hunk fixed it; the
  companion fix was in `tests/test_xorsched.c`, whose `uop_divisor`
  trusted prctl — it now uses each version's own `group_bytes`
  (the JIT's ground truth), the same pattern as
  `tests/test_boundaries.c:sve_vl_bytes`.
- It was latent because the old `cg.group` was unused by the SVE loop —
  `group_bytes` made the VL load-bearing for dispatch, and the twin was
  validated at wrong lengths and produced garbage (caught by QEMU).
- It therefore alters **general-path codegen under qemu-user** (the
  general version's `group_bytes`, validation lengths, and timing sizes
  now use the true VL). On native hardware `prctl` and `cntb` agree, so
  native codegen is unaffected.
- Do NOT auto-resolve this hunk away in any future merge, and do NOT
  "harmonize" it into something smaller: the whole point is that
  build-time VL == runtime VL, verified in every test environment
  (native, QEMU at all VLs). See `AGENTS.md` ("New load-bearing fields
  must re-verify their sources").
### 11.1 File list

New: `src/dispatch.c`, `src/dispatch.h`, `docs/length_dispatch.md` (this
file), `tests/test_dispatch.c`.
Modified: `src/api.c` (shootout + dispatcher wiring at the entry point),
`src/jit.h` (`version_t.notail`/`group_bytes` fields; `jit_build_version`
threads `cand_t.notail` into each backend's `cg_t` — its signature is
unchanged), `src/plan.h` (`cand_t.notail` twin dimension),
`src/bench.c`/`src/bench.h` (per-version validation lengths, per-version
timing sizes and byte totals), `src/x86_64/jit.c`, `src/aarch64/jit.c`,
`src/riscv/jit.c` (notail codegen), `Makefile` (build `dispatch.c`, new
test incl. qemu targets), `README.md` (one paragraph on length dispatch).
New: `AGENTS.md` (process lesson from the `cands[]`/`vs[]` index-confusion
catch — escape protocol).

## 12. Length-aware tail-twin routing (track B1x86, generalized for B1sve)

A masked/predicated tail twin (`TAIL_MASKED`) is installed only after
winning its tail gate (`bench_tail_duel`, the ISA-agnostic tail duel, for
both the x86-64 k1 twin and the SVE `whilelt` twin). It was measured at
tail lengths -- so it should serve tail lengths, not compete on GB/s
noise at lengths where its tail never runs.

The mechanism (in `src/dispatch.h` / `src/dispatch.c`, ISA-agnostic):

- `TAIL_MASKED` admits ONLY tail lengths (`len % group != 0`).
- When a `TAIL_MASKED` variant is installed, `rj_dispatcher_create`
  derives every `TAIL_SCALAR` variant's policy to `TAIL_SCALAR_MULT`
  (admits only `len % group == 0`): the scalar general variant defers
  tail lengths to the installed twin. Without an installed masked twin,
  `TAIL_SCALAR` admits everything (legacy behavior).
- `TAIL_NONE` admits multiples as before.

Routing consequences (deterministic, no GB/s coin flip):

| Length | Admits | Routes to |
|---|---|---|
| `len % group != 0` (tail) | masked twin only | the installed tail twin |
| `len % group == 0` (multiple) | general + notail | fastest-first among those |

The dispatcher does not know or care HOW the masked twin earned its
place (the tail duel in both tracks): the presence of a `TAIL_MASKED`
variant in the installed set is the trigger. A new ISA's
predicated/masked tail plugs in by installing a `TAIL_MASKED` variant --
no dispatcher change.

Create-time coverage check: the variant set must admit every length,
else `rj_dispatcher_create` fails cleanly (caller falls back to
general-only). A lone `TAIL_SCALAR` covers everything; otherwise the set
needs a tail-server (`TAIL_MASKED`) plus a multiple-server (`TAIL_NONE`).
`TAIL_SCALAR_MULT` is derived-only: callers must not set it.

Note for B1sve: under this routing, an installed `TAIL_MASKED` twin
receives ALL tail traffic deterministically, so its install gate must
cover tail-length performance -- a large-size-only shootout win is not
sufficient evidence, which is why the predtail twin is gated by
`bench_tail_duel` (the x86-64 twin's duel, generalized) rather than the
shootout.
