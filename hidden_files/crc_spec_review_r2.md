# CRC spec review — Reviewer 2 (adversarial integration + edge cases)

Reviewer: independent spec reviewer (2 of 2). Review-only; no implementation.
Spec: `hidden_files/crc_impl_SPEC.md`. Basis: `hidden_files/crc_optimum_research/FINAL_SYNTHESIS.md`.
Date: 2026-10-08. Status: COMPLETE.

## Sources checked
- [x] crc_impl_SPEC.md (full)
- [x] FINAL_SYNTHESIS.md (full)
- [x] src/plan.h (cand_t axes, plan_t, plan_candidates/plan_need_regs contracts)
- [x] src/prune.c (twin_dominates pinning pattern)
- [x] tests/test_twin_audit.c (reachable/targeted/full sweeps)
- [x] src/api_crc.c (rj_compile_crc entry)
- [x] src/fused.c (fused driver; per-op rj_compile; driver CRC accumulation)
- [x] include/raidjit.h (rj_options, rj_fn contract, plan-ID grammar)
- [x] src/api.c (plan_candidates call sites ×4, stage-2 twins, synth_desc, buffer comment)
- [x] tools/wasm/rjw_api.c (explicit twin-flag JSON surface: parse/validate/report)
- [x] tools/gen_kernels.c (spec declaration, 512 buffer, tail/NT twin construction)
- [x] tests/test_prune.c (unit list incl. unit_twin_axes_pinned:1269)
- [x] src/dispatch.c (variant coverage rule)
- [x] src/cpu.h (rj_uarch_t: x86-only tags; rj_uarch_override decl)
- [x] src/plan.c (plan_candidates maxout guards; rj_cpu_uarch: UNKNOWN on ARM/RV; rj_uarch_current; weights)
- [x] src/planid.c (twin token compose/parse/validate)
- [x] src/riscv/cpu.c (rvv_zvbc detection pattern)

## Findings
(numbered; severity BLOCKER / MAJOR / MINOR / NIT; spec section cited)

## Finding 1 — BLOCKER — R11: fold-state kernel/driver ABI deferred, but load-bearing
Spec: "Kernel calling convention: the driver passes per-stream fold states
(driver-owned array, persistent across tiles) via new pointer args ... The
exact ABI extension is the worker's design, reviewed."
Checked: `rj_fn` is `void (*)(const void *in[], void *out[], size_t len)`
(raidjit.h); fused.c calls `f->fns[k](&ftin[bii], ..., tl)` with that exact
signature; `struct rj_fused.fns[]` is `rj_fn[]`. There is no channel for
per-call fold-state pointers today.
Why it cannot wait: the choice determines (a) whether the uniform `rj_fn`
contract survives (fold kernels are built by `rj_compile` inside
`rj_fuse_compile` and their `rj_fn` is stored in `f->fns[k]`); (b) thread
safety — "persistent across tiles" per CALL (stack, like `acc[]` in
fused_driver_call) is safe for concurrent `rj_fused_call`, per-HANDLE
storage races; the spec's self-attack list includes "concurrent compiles"
but not concurrent CALLS of one CRC handle; (c) plan-ID / `rj_compile_pinned`
— can a fold kernel be pinned and reconstituted, and through which entry
type?; (d) length dispatch — notail/maskedtail twins built from a fold
winner (api.c:407ff builds `twinc` from `vs[winner].cand`, inheriting
`crc_fold=1`) need the same state channel; (e) R16's doc tool must invoke
or at least disassemble with the right arg layout.
Required: the spec must fix the ABI (or a closed set of options with
selection criteria) BEFORE emitter work — e.g. states-as-trailing-`out[]`
pointers keeps one signature; a wider entry type needs a typed accessor
and a pinned-compile rule. "Worker's design, reviewed" is fine for
details, not for the contract.

## Finding 2 — MAJOR — R9: want_crc threading underspecified at three joints
Spec: "`want_crc` originates in `rj_crc_options` (new field `want_crc_fold`?
NO — simpler: `rj_compile_crc` always sets it; the fused driver passes it
through). Exact struct/field threading is the worker's design."
Checked: api_crc.c builds `rj_fuse_options fo` from `rj_crc_options`
(field-by-field copy); fused.c `fuse_options_to_rj` copies
`rj_fuse_options`→`rj_options`; `rj_fuse_compile` compiles EACH data op via
`rj_compile(topo,&op,&o,...)` with one shared `rj_options o`; `plan_build`
takes no options (so `plan_t.want_crc` must be set post-build by api.c).
Gaps:
 (a) The new `rj_fuse_options` field is a PUBLIC API addition — unnamed,
     undocumented, and its semantics for DIRECT `rj_fuse_compile`
     ([ENCODE,CRC]) users is unspecified (do they get the fold? opt-in?
     silently not?). "rj_compile_crc always sets it" implies direct fused
     users never get the fold — a silent behavioral asymmetry between two
     public paths to the same op sequence. State the rule.
 (b) Per-op gating: the flag must reach ONLY the CRC target's compile
     (k == crc_tgt); other data ops in the sequence must compile WITHOUT
     want_crc. Not stated.
 (c) `plan_t.want_crc` must be applied at EVERY plan-construction site in
     the compile path — api.c has at least two (normal plan_build ~line 212
     and the A5 survivor path ~line 200, `wpl`). Not stated.
The "NEVER set for non-CRC compiles" invariant is asserted but no
mechanism is named; R14 (see Finding 9) cannot enforce it as written.

## Finding 3 — MAJOR — R9/R16: candidate-buffer overflow → silent truncation
Checked: plan_candidates guards EVERY loop with `n < maxout` (plan.c
745,754,767,...) — overflow is SILENT truncation, not an error. Six call
sites use `cands[512]` (api.c 220/961/1268/1468, rjw_api.c 840,
gen_kernels.c 848). api.c:218's comment is explicit: "a future axis must
increase the buffer or fail closed — silent truncation is unacceptable"
(94% full today). A crc_fold axis DOUBLES the worst case when
want_crc && viable.
Required: R9 must require the buffer bump (all six sites, or at least the
CRC-compile path + doc tool) or make plan_candidates fail closed
(return -1) on maxout. Note the existing precedent: nt/prefetch twins are
only enumerated when benchmarking, partly for buffer pressure — the fold
axis needs the same deliberate treatment (see also Finding 5).

## Finding 4 — MAJOR — R9: stable plan ID ignores the new axis
Checked: src/planid.c composes the stable plan ID with an explicit twin
token list (planid.c:208-209 `any_twin`, 222-245 TWIN_TOK nt/pf/ternlog/
remat/xorsched/xorbit/mshift/permi2b/pipe/lmul), parses them back
(683-694), and shape-validates (803+ SHAPE_BAD). raidjit.h documents the
ID grammar publicly ("the candidate is the contract").
The spec NEVER mentions plan IDs. Without a `crcfold` token, fold and
non-fold candidates share an ID → `rj_compile_pinned` ambiguity,
`rj_list_variants` duplicate IDs, `rj_plan_id` collisions. Required: token
+ parse + shape validation (fold ID requires the want_crc compile
context — cf. Finding 1c) + synth_desc parity (api.c:1221) + the wasm
JSON twin-flag surface (rjw_api.c:1011-1031, coordinator-owned).

## Finding 5 — MAJOR — R9/R11: nt/prefetch twin enumeration for fold candidates unspecified
Checked: plan.h documents nt/prefetch twins as "a non-temporal twin of
EVERY candidate"; plan_candidates appends them after temporal originals.
If fold candidates get nt/prefetch twins: candidate explosion on CRC
compiles and R12's register accounting must cover nt variants. If not:
the spec must say the fold twin is enumerated only on the base candidate
(and why — e.g. NT+fold is still benchmark-meaningful for large stripes,
so exclusion needs a principled reason, not silence).
Related emitter rule the spec misses: on an nt=1 fold twin, the fold
STATE stores (prologue/epilogue) must be TEMPORAL (re-read next tile)
even though parity stores are NT. Name it.

## Finding 6 — MAJOR — R10: AArch64 gate needs uarch-tag + detection work the spec never names
Checked: `rj_uarch_t` (src/cpu.h:12-24) has NO ARM tags; `rj_cpu_uarch()`
returns RJ_UARCH_UNKNOWN on AArch64/RISC-V (plan.c:1313-1318: "no
per-uarch port data is modeled"). R10's gate signature
`rj_crc_fold_viable(rj_uarch_t u, ...)` therefore cannot express the
Apple condition without NEW enumerators (e.g. RJ_UARCH_APPLE_M1),
detection in src/aarch64/cpu.c (implementer 0x61 parse), and weights
entries. The spec gives gate SEMANTICS but not this work — a reviewer
cannot verify completeness against an unnamed work item.
Required: name the tag(s), the detection site, and the weights-table
entry (neutral fallback is acceptable, stated).

## Finding 7 — MAJOR — R10: viable() must be pure; the ARM clause as worded invites impurity
Spec: "AArch64: 1 iff cpu->pmull && cpu->sha3 AND MIDR implementer == 0x61
(/proc/cpuinfo `CPU implementer` field, same parse pattern as the existing
brand reader...)".
If the implementer check re-reads /proc/cpuinfo INSIDE viable(), the
function is impure: `rj_uarch_override` (the R15 test mechanism) cannot
force the ARM fold path on non-Apple hardware, and two call sites can
disagree. Required: the cpuinfo→tag mapping lives in `rj_cpu_uarch()`
EXACTLY ONCE (per Finding 6); `rj_crc_fold_viable` is a pure function of
(uarch_tag, cpu_flags, isa). Then R15's override(RJ_UARCH_APPLE_M1) +
synthetic pmull/sha3 flags suffices. As worded, R15's ARM coverage is
untestable — this is the same testability class the codebase already
solved with rj_uarch_override.

## Finding 8 — MAJOR — R11/R15: fold × stage-2 tail twins composition unadjudicated
Checked: api.c:407ff builds notail (then predtail/maskedtail) twins from
the STAGE-1 WINNER via `twinc = vs[winner].cand` — a fold winner's twins
inherit crc_fold=1 unless gated. test_twin_audit.c's principle (from the
4f1e69c pipe+predtail SIGABRT): planner-set axis × api.c-built twin must
build or reject gracefully, never abort.
The sharp case: maskedtail REPLACES the scalar tail with a masked vector
tail; masked lanes read as ZERO. Folding zeros into the CRC lanes
corrupts the CRC unless the masked tail folds ONLY valid bytes. R11's
tail story ("tail via §11 x^{-8r}/x^{64-8r} constants") covers the
SCALAR tail only. notail+fold is sound (dispatcher guarantees
len % group == 0 → no tail bytes), but maskedtail/predtail+fold need a
verdict.
Required: the spec must EITHER (a) exclude stage-2 tail twins on fold
winners (documented exclusion like pipe+predtail, audited for graceful
skip) OR (b) specify the masked/predicated tail-fold composition with
bit-exactness proof. R15 must then cover the chosen rule in the sweep
(planner fold outputs × {notail,predtail,maskedtail,permi2b},
build-or-graceful-reject). As written R15 only pins the prune axis —
it does not cover the twin×twin cross product, repeating the 4f1e69c
class.

## Finding 9 — MAJOR — R14: byte-identical proof is vacuous for its own requirement
Spec: "regenerate `make kernels` and diff every section against baseline:
all 151 sections identical."
Checked: the kernels doc DELIBERATELY contains no CRC/fused sections
("Deliberately not represented" — driver-level compositions emitting
byte-identical machine code to their sub-kernels; tools/kernels-README.md).
So the regen+diff can NEVER observe a want_crc leak or any fold codegen
change — R14 passes trivially even if the worker breaks the "NEVER set
for non-CRC compiles" invariant (Finding 2).
Required: a companion LEAK test, e.g.: fix a corpus of (topo,op); build
every plan_candidates output with want_crc forced 0 vs 1 (viable uarch
override); assert (i) want_crc=0 emits ZERO crc_fold=1 candidates, and
(ii) the want_crc=0 emitted bytes are byte-identical pre/post change
(hashes recorded in the test or the baseline file). R14's doc-diff stays
for the non-CRC surface.
Minor: R14 should name ALL doc-build artifacts (the .md, the disassembly
files, AND the HTML explorer pages — tools/gen_kernels_page.py,
gen_live_kernels_page.py exist); state that R16's ADDED sections are
allowed and per-section hashing keeps old sections comparable;
replace "git stash-style A/B" with two-worktree A/B (merge-base vs tip)
— stash is fragile under the branch plan.

## Finding 10 — MAJOR — R13: nothing guarantees the test exercises the fold twin
Spec: "CRCs from the interleaved kernel MUST be bit-identical ..." but no
selection rule. Checked: rj_crc_options_default sets benchmark=1
(api_crc.c); A4 says tests use benchmark=0 "unless a test specifically
targets selection".
The trap: with benchmark=0, heuristic_select picks by STATIC model — the
spec never says the fold twin ranks first, so R13 may test only the
driver path. With benchmark=1 (the rj_compile_crc DEFAULT), selection is
timing-dependent → nondeterministic fold coverage (the gate can PASS
without ever running the fold — false confidence) AND the test performs
timing benchmarks, in tension with the standing NO-timing-benchmarks
constraint and A4's determinism rule. A developer running R13 with
defaults hits both.
Required: R13 must state benchmark=0 AND enumerate plan_candidates'
crc_fold=1 outputs directly, building/testing EACH via jit_build_version
(bypassing selection), asserting the fold path was exercised (e.g. via
the Finding-4 plan-ID token). The "no timing allowed" parenthetical is
then structural, not aspirational.

## Finding 11 — MAJOR — R13: length lower bound 0 is untestable as stated
Spec: "random lengths (0..1<<20, ...)". Checked: fused_driver_call asserts
len >= 1 (fused.c); fold kernels are reachable ONLY via the fused driver
(want_crc is set only inside rj_fuse_compile's per-op rj_compile —
Finding 2). So len=0 cannot go through ANY public path to a fold kernel;
a literal implementation of R13 trips the assert (debug) or tests
nothing (the driver never calls the kernel with 0).
Required: driver-level differential uses 1..2^20; len=0 EITHER tested at
kernel level via direct jit_build_version invocation (if the emitter
supports it — prologue loads states, loop skipped, epilogue stores,
reduction yields init CRC) OR explicitly excluded with rationale
(driver contract len>=1). Also cross the tail matrix with the tail-twin
actually installed (ties to Finding 8).

## Finding 12 — MAJOR — R12: defers the load-bearing decision past its consumer
Spec: "The worker MUST pick and justify one mitigation [(a) reserve regs /
(b) spill to L1 / (c) hybrid] with a uop-cost argument. The reviewer
checks the arithmetic."
The choice DETERMINES R11's emitter construction (states in zmm vs
memory) and plan_need_regs' exact formulas — R11 cannot be implemented
before R12 is decided, but the spec imposes no ordering. "Worker picks,
reviewer checks" is fine for a parameter; not for a decision its
consumer (R11) is specified against.
Required: (1) R12 decided BEFORE R11 emitter work, arithmetic recorded in
the design doc; (2) the arithmetic starts from plan_need_regs' CURRENT
output for the (isa,mul,U) and adds EXACTLY nstreams×1 zmm (n_in+n_out);
(3) derive the "≤ 28" budget — 32 architectural zmm minus WHAT?
(k-regs? temps? spill headroom?) — unexplained in the spec; (4) if
(b)/(c): quantify per-tile prologue/epilogue memop cost amortized per
byte, and forbid NT for state stores (Finding 5); (5) the decision must
compose with the winner's other twins (nt, prefetch) or explicitly
exclude them (Finding 5).

## Finding 13 — MAJOR — R2 x86 rung 2 contradicts the declared basis
Spec basis: FINAL_SYNTHESIS is "THE spec basis". R2 rung 2: "`pclmulqdq`
→ scalar+vector fusion (Cawley v-lane structure; the synthesis shows 34%
over vector-only on ICL-SP and is the only fast path on Zen2/3)."
Checked against the synthesis Phase-1 table:
 - Intel PCLMULQDQ row says the optimum is the "128-bit fold
   (zlib-ng/Chromium)" — NOT fusion (ties crc32q). R2 prescribes fusion.
 - The "34% over vector-only on ICL-SP" parenthetical cites the
   VPCLMULQDQ+AVX512 fusion row (rung 1), not the PCLMULQDQ rung —
   miscited.
 - Zen3 HAS vpclmulqdq but NOT avx512f → falls to rung 2; the synthesis's
   Zen3 row says the optimum is "fusion w/ 256-bit VPCLMULQDQ" (2.3x over
   128-bit PCLMULQDQ fold). Rung 2 as written uses 128-bit PCLMULQDQ on
   Zen3, leaving the 2.3x on the table — directly contradicting the basis.
Rung 2 conflates THREE synthesis rows (Intel-128bit-fold,
AMD-scalar-led-fusion, Zen3-256bit-fusion) into one "scalar+vector
fusion".
Required: split rung 2 (vpclmulqdq&&!avx512f → 256-bit fusion per the Zen3
row; Intel pclmulqdq → 128-bit fold per the table OR a justified fusion;
AMD → scalar-led fusion), or explicitly delegate the per-case choice to
the worker WITH the table as arbiter. (Correctness is unaffected — R4 is
the gate — but a spec that contradicts its own "THE basis" on the
declared optimum is a spec bug.)
Related MINOR: R2 AArch64 rung 1 ("pmull && sha3 → Cawley fusion") doesn't
select per core, but the synthesis has DIFFERENT fusion variants per core
(M1 v9s3x2e_s3 vs N1 v3s4x2e_v2). The "worker justifies the lane count"
delegation is stated only for rung 3 (scalar) — extend it to rung 1.

## Finding 14 — MINOR — R9: npar==0 / noop plans
A fold twin for an ENCODE with npar==0 (or plan_t.noop) interleaves the
fold with NOTHING — the "almost free" premise is vacuous and the kernel
is pure CRC (worse than rj_crc32_fast: kernel-call overhead, no
interleave). Not incorrect, just pointless; but the emitter must still
handle the degenerate body. Required: state whether fold twins are
enumerated when n_out==0 (recommend: skip — driver-level CRC suffices),
so the worker doesn't discover it mid-emitter.

## Finding 15 — MINOR — R15: sweep mechanics underspecified
"the reachable sweep includes want_crc=1 × viable-uarch combos (via
rj_uarch_override...). New sweep counts recorded; 0 aborts."
 (a) test_twin_audit.c fill_cpu sets NO new R3 flags (no vpclmulqdq;
     aarch64 sets neon/sve/sve2 but not pmull/sha3/crc32). The sweep must
     set them, or viable() never fires and the sweep tests nothing new.
     (Depends on Findings 6-7 for the ARM tag.)
 (b) "New sweep counts recorded" — WHERE? (stdout? expected-count
     assertion? a doc?). Name the artifact.
 (c) The (crc_fold=1)/(crc_fold=0) pin test matches the existing
     unit_twin_axes_pinned pattern (test_prune.c:1269) — well-grounded,
     no change needed. But the sweep must ALSO cover Finding 8's
     twin×twin cross product (or the chosen exclusion).
 (d) gen_kernels.c for R16 needs the same trick (want_crc=1 via
     post-plan_build set + uarch override + new R3 flags in its
     synthetic CPU) — R16 doesn't mention any of it; bless the
     post-plan_build want_crc set explicitly since plan_t is internal.

## Finding 16 — MINOR — R4: red-green for constant-free variants
"each test fails when run against a deliberately broken constant (flip
K1/K2 low bit; flip one fusion lane-shift constant)". The sse42 crc32q
interleave and slice-by-8 fallback have NO such constants. Required:
"or an equivalent mutation per variant (drop an interleave lane,
corrupt the table polynomial)" — a test that cannot fail proves
nothing, and the spec's own red-green doctrine demands it.

## Finding 17 — NIT — misc
 (a) R6 "replaces the <10 GB/s scalar path" — the synthesis's reported
     figure for slice-by-8 is 1–3 GB/s; cite it instead of "<10".
 (b) R5 SIZE_MAX: "return crc unchanged" is a SILENT WRONG ANSWER, not
     "fail closed" (uint32_t has no error channel). The spec already
     requires the choice be spec'd+tested; add: document the choice in
     raidjit.h (public API edge) and note the caller-bug contract.
 (c) R7: name the WASM backend entry (or state it reuses slice-by-8
     directly) — "the top-level dispatch chooses the backend by
     compile-time ISA" leaves the __wasm__ case implicit.
 (d) R13 topology list is ENCODE-only; the fused CRC op may also target
     PARITY_UPDATE (fused.c allows it) but rj_compile_crc only emits
     [ENCODE,CRC] — state the ENCODE-only scope explicitly or cover
     P_DELTA in R13.
 (e) R11 emitter scope: VERIFIED ADEQUATE — no VPCLMULQDQ emitter exists
     today (grep: none in src/), so exactly one new EVEX emitter family
     (VPCLMULQDQ zmm, imm8 ∈ {0x00,0x01,0x10,0x11} for data fold + lane
     reduction) is needed; VPTERNLOGQ/loads/stores exist. R11 correctly
     invokes the AGENTS.md emitter lesson. One caveat: the GAS
     cross-check needs binutils ≥ 2.31 for VPCLMULQDQ — confirm the
     repo's cross-check pattern covers EVEX (it did for VBMI's
     VPMULTISHIFTQB/VPERMI2B, so presumably yes; verify at implementation
     time).

## Verified sound (no finding)
 - R13 DOES cover parity streams: "for every stream (inputs and
   parities)" — matches the synthesis's "input CRC while vectors load
   and parity CRC from registers around stores". ✓
 - R6 driver swap (rj_crc32_update → rj_crc32_fast): 1:1 under R1's
   identical-chaining contract; zero codegen change; R2's thread-safe
   first-call dispatch covers concurrent driver calls. ✓
 - R3 XCR0 note ("pclmulqdq/vpclmulqdq/sse42 need no XCR0 state"):
   correct — rung 1's avx512f is XCR0-gated by existing code. HWCAP
   values (PMULL=1UL<<4, CRC32=1UL<<7, SHA3=1UL<<17, all AT_HWCAP):
   correct per kernel headers. ✓
 - R3 "detection follows the existing rvv_zvbc pattern in
   src/riscv/cpu.c": grounded — src/riscv/cpu.c:212
   `out->rvv_zvbc = has_ext_cpuinfo("zvbc") ? 1 : 0;`. ✓
 - rj_uarch_override EXISTS (src/cpu.h:39, src/plan.c:1247) — R15's
   cited mechanism is real. ✓ (x86_64/cpu.c has no override/getenv
   path, so R3's fallback — direct entry-point tests — applies on x86;
   the spec already provides for this. ✓)
 - WASM handling: rjw_api.c DOES enumerate twin flags explicitly
   (JSON→cand_t ~644-659; validation ~396-446; reporting ~1011-1031),
   so the spec's "flag to coordinator, do not touch" is the RIGHT call
   and the condition is confirmed TRUE. wasm never sets want_crc and
   manual cand_t is memset-0 → the new axis cannot leak into wasm
   builds; only feature-exposure is coordinator-owned. ✓
 - prune pin pattern: unit_twin_axes_pinned (test_prune.c:1269) is the
   established pattern for R9's "new pin tests"; twin_dominates'
   field-by-field comparison extends mechanically. ✓
 - Single-stream topologies: fold is per-stream independent; no minimum
   stream assumption in the construction. ✓
 - Concurrent rj_compile_crc: present in the spec's self-attack list
   ("concurrent compiles"). ✓ (per-CALL fold-state thread safety is
   Finding 1b.)

## Overall assessment
The spec's Phase 1 (Option C) is in good shape: dispatch chains are
concrete, R4/R5 test doctrine is strong, and the integration surface
(fused.c swap) is a 1:1 contract substitution — except R2's x86 rung 2,
which contradicts the synthesis it cites (Finding 13). Phase 2
(Option A) has the right hard gates (R13 differential, R14
byte-identical, R15 audit) but FIVE load-bearing integration decisions
are missing or deferred: the fold-state ABI (Finding 1, BLOCKER), the
want_crc threading invariants (Finding 2), candidate-buffer growth
(Finding 3), the plan-ID token (Finding 4), and the fold×tail-twin
composition rule (Finding 8). R14 as written cannot enforce its own
anti-leak requirement (Finding 9), and R13 as written may never execute
the fold (Finding 10). None of these is an implementation detail — each
changes what the worker builds. Recommend: revise R9–R11/R13–R15 to
close Findings 1–13 before Phase 2 work starts; Phase 1 can proceed
with Finding 13 fixed.
