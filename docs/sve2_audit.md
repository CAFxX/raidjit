# SVE2 audit (2026-10-01)

Question: does any SVE2 instruction profitably improve the SVE backend in
`src/aarch64/`? This is an honest study, not a mandate: the answer could
have been "nothing wins". It turned out exactly one instruction wins.

## Verdict

**EOR3 is the sole winner — implemented.** Every other surveyed SVE2
instruction is documented below with the reason it loses. NEON-side EOR3
is a follow-up (out of scope here).

## Method

For each candidate instruction:
1. **Encoding**: verified against `aarch64-linux-gnu-as` (assemble with
   `-march=armv8.2-a+sve2`, confirm rejection under plain `+sve`).
2. **Executability**: run on QEMU `neoverse-n2`/`max` (HWCAP2_SVE2=1) and
   confirm SIGILL on `neoverse-n1` (no SVE2), to prove the feature gate.
3. **Codegen delta**: count the vector instructions the backend would emit
   before/after, check register-pressure impact (the pressure formulas in
   `docs/pressure_analysis.md` must not change silently).
4. **Microarchitecture**: Neoverse V2 Software Optimization Guide,
   Table 3-21 (latency 2, throughput 1/cycle, pipe V0 for EOR3).
5. **Cost of adoption**: a new planner candidate dimension doubles SVE
   benchmark time, so a win must be structural (strictly dominates) to be
   taken as a pure codegen gate instead.

## Per-instruction results

| Instruction | What it does | Delta vs current codegen | Decision |
|---|---|---|---|
| **EOR3** | 3-input XOR, `Zdn ^= Zm ^ Za`, unpredicated, base SVE2 | HORNER nontrivial `(j,u)`: drops MOV + 1 EOR (9→7 uops remat, 7→5 vector insns hoisted); MULACC/DELTA accumulate: 2 EOR → 1 EOR3 | **WIN — implemented** |
| BCAX | `Zdn ^= ~(Zm) & Za` (bit-clear-and-xor) | Would need a `~mask` constant plus a zero vector and is destructive on acc — strictly worse than EOR3 for our shapes | reject |
| XAR | XOR then rotate | No rotate in the nibble-multiply reduction; no use | reject |
| RAX1 | 128-bit rotate-xor (SHA3) | Needs SVE_SHA3 (`HWCAP2_SVESHA3`), which we don't detect; no use for GF math | reject |
| PMULLB/T | 8→16-bit carryless multiply | Settled by `docs/pmull_study.md`: 13 insns vs 7 for the TBL nibble path — not redone here | reject |
| BDEP/BEXT | bit deposit/extract | Cannot beat AND for the nibble split; destructive on the destination | reject |
| MATCH | first-match predicate generation | Framing/predication domain, not GF arithmetic | reject |
| SM4* / AES* | crypto rounds | Wrong domain | reject |
| post-index SVE loads | — | Do not exist in SVE (that's AArch64 scalar/NEON addressing) | n/a |
| 256-byte TBL (full byte) | one TBL per 64 B of table | Needs 16 table registers per row — register-pressure impossible | reject |
| EOR3-folding in P_XOR | fold XOR chains | P_XOR is load-bound; folding needs a restructure plus a pressure-formula change | reject |

## EOR3 details

- **Base SVE2, not crypto-gated.** Verified three ways: (1) assembles with
  `-march=armv8.2-a+sve2`, rejected with plain `+sve`; (2) executes on QEMU
  `neoverse-n2`/`max` (HWCAP2_SVE2=1), SIGILLs on `neoverse-n1`;
  (3) independent source (zingaburga's SVE2 analysis gist) confirms
  EOR3/BCAX/XAR moved to base SVE2 — only RAX1 needs SVE_SHA3.
- **Detection already existed.** `src/aarch64/cpu.c` reads `HWCAP2_SVE2`
  into `rj_cpu_info.sve2`; no new hwcap work was needed.
- **Encoding** (verified vs GAS and QEMU execution):
  `EOR3 Zdn.d,Zdn.d,Zm.d,Za.d` = `0x04203800 | (Zm<<16) | (Za<<5) | Zdn`.
  Destructive (`Zdn ^= Zm ^ Za`), unpredicated, `.D` only (element size
  is irrelevant for XOR). Fixed-bits mask for code scanning: `0xFFE0FC00`.
- **Codegen deltas** (no register-pressure change — same live registers,
  so `docs/pressure_analysis.md` formulas are untouched):
  - HORNER i>0 nontrivial: `TBL acc,lo / MOV acc / TBL r2,hi /
    EOR acc,r2 / EOR acc,r1` → `TBL acc,lo / TBL r2,hi / EOR3 acc,acc,r2,r1`
    (drops the MOV and one EOR: 9→7 uops on the remat path, 7→5 vector
    insns with hoisted tables; acc is dead after the nibble split, so TBL
    can target it directly).
  - MULACC/DELTA accumulate else-branch: `EOR acc,r1 / EOR acc,r2` →
    `EOR3 acc,acc,r1,r2` (saves 1).
  - The first-column path deliberately keeps copy+xor: EOR3 folds the
    destination's *live* value in, but there acc is dead, not zero.
- **Microarch** (Neoverse V2 SOG, Table 3-21): EOR3 latency 2,
  throughput 1/cycle, pipe V0. Even under pessimistic assumptions
  (EOR3@1/cy vs EOR@2/cy) the swap is throughput-neutral and shortens the
  accumulator chain (~4 cycles/row); under realistic assumptions it is a
  clear win. Strictly-dominates argument → no new planner candidate
  dimension (avoids 2× SVE benchmark cost).
- **Gating**: pure codegen gate on `cpu->sve2`, set iff the backend will
  emit at least one EOR3 — the `+eor3` desc tag means "this build contains
  EOR3 instructions", exactly. Per kind:
  - P_HORNER: some `horner_c[j] ∉ {0,1}` and `n_in >= 2` (EOR3 sites live
    in the i>0 loop; `n_in <= 1` HORNER is unreachable from the planner,
    the bound is defensive).
  - P_MULACC: some `mat[j][i] ∉ {0,1}` with `i` past row j's first nonzero
    column (a nontrivial coefficient on the first column takes the
    copy+xor path — destination dead, not zero — and emits no EOR3).
  - P_DELTA: `need_mask` (every nontrivial coefficient takes the
    accumulate path since `first_col` is NULL); all-0/1 delta columns
    (e.g. RAID-5) correctly yield no tag.
  - P_XOR (and any future kind): never.
  `cg_t.use_eor3` is documented in `src/aarch64/jit.c`; `c_eor3()` asserts
  it as a tripwire.
- **No degenerate-tag edge**: an earlier draft of the gate used
  `need_mask` ("some nontrivial coefficient exists") for MULACC, which
  would have tagged plans that emit zero EOR3s (e.g. every nontrivial
  coefficient sitting on its row's first column). Peer review caught this;
  the gate is now exact per the rules above, and `tests/test_sve_eor3.c`
  pins the biconditional `(emitted EOR3s > 0) ⟺ ("+eor3" in desc)` on
  degenerate MULACC topologies.
- **Tests**: `tests/test_aarch64_emit.c` has byte-exact `em_sve_eor3`
  checks (GAS cross-checked); `tests/test_sve_eor3.c` force-builds all SVE
  candidates for HORNER/MULACC/P_XOR/P_DELTA, asserts `+eor3` ⟺
  (sve2 && nontrivial), scans emitted code for the EOR3 opcode, checks
  bit-exactness vs a scalar `gf_mul` reference at lengths
  {0,1,VL-1,VL,VL+1,63,64,65,1000}, and re-runs the HORNER case with
  `sve2` force-disabled (baseline path: no `+eor3`, no EOR3, still
  correct). All three mutation checks (emitter encoding, EOR3 semantics,
  gating stuck-off) were demonstrated to fail. Run under QEMU at
  VL=128 and VL=256 (`make _test-qemu-aarch64-sve`).

## Packaging note

`tests/dump_kernel.c` labels an ISA_SVE build on a synthetic `sve2=1` CPU
as "sve2"; the prebuilt kernels in `your_files/raidjit/kernels/`
(`raid{5-4,6-5}_sve2.asm`) contain zero `eor3` today — plain SVE under an
SVE2 label. After this change, rebuilt sve2 kernels genuinely use EOR3,
making the label accurate. The shipped zips are stale regardless;
refreshing them is a separate todo (not reworked here).

## Follow-ups (not done)

- **NEON EOR3**: the Advanced-SIMD 3-register SHA3 `EOR3` needs
  `+sha3` (`HWCAP_SHA3`), which the backend does not detect. Same
  strictly-dominates argument would apply to the NEON nibble path, but it
  needs a new hwcap bit plus kernel-label thought. Left for a separate
  change.
- **P_XOR EOR3-folding**: would need a load/store restructure and a
  pressure-formula change; revisit only with measurements showing P_XOR
  is ALU-bound on some target.
