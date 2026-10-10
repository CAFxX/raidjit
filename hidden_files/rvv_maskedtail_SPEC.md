# Spec: RVV masked tail twin (`maskedtail` on RISC-V)

Status: spec (2026-10-10). Implements the coordinator's Option C from the
fullmask exploration: fill RVV's empty TAIL_MASKED slot with a masked tail
twin (NOT a tailless loop).

## 1. Problem

RVV's tail is the strip-mining remainder loop (`src/riscv/jit.c:1286-1310`):
for remaining R in 1..U*VLMAX-1, it loops with `vsetvli T0, A2` (vl =
min(VLMAX, R) in hardware) and runs the single-unroll body (`gen_body(cg,1,1)`).

Dispatch taxonomy (`src/dispatch.h`):
- x86 has TAIL_MASKED (maskedtail/k1 twin).
- SVE has TAIL_MASKED (predtail/whilelt twin).
- RVV has NO TAIL_MASKED twin — only TAIL_SCALAR (the strip-mining loop)
  and TAIL_NONE (notail).

This change gives RVV a TAIL_MASKED twin so tail lengths get their own served
variant under length-aware routing, exactly like x86/SVE.

## 2. Design: masked tail twin (Shape A)

Keep the counted main loop unchanged. Replace the strip-mining remainder loop
with an **unrolled full-VL masked tail**:

```
# A2 = remaining, 1 .. U*VLMAX-1 (guaranteed by the counted loop)
vsetvli T0, x0, e8, m<lmul>, ta, ma   # vl = VLMAX, once (no vsetvli in chunks)
vid.v vd_tmp                          # vd_tmp[i] = i (0..VLMAX-1)
for u in 0..U-1 (unrolled at JIT time):
    # chunk u covers bytes [u*VLMAX, (u+1)*VLMAX)
    sub  tmp, A2, u*VLMAX             # tmp = remaining bytes for this chunk
    blez tmp -> next_chunk            # skip if chunk fully past the end
    vmsltu.vx v0, vd_tmp, tmp         # v0[i] = (i <u tmp); unsigned:
                                     # exact for VLMAX <= 255 (SEW=8
                                     # scalar truncates to 8 bits, V §11.1;
                                     # backend gate rejects wider configs)
    <body for u=1 with MASKED data loads/stores, v0.t, win=0>
next_chunk:
```

Key points:
- `vl` stays VLMAX throughout (no vl reduction). The mask selects active lanes.
- `win=0`: vl is provably VLMAX, so the vsetivli-16 table-load window is elided
  (saves 3 uops per `c_ldtab`/`c_ldtab1` call vs the strip-mining loop's win=1).
- Only DATA loads/stores are masked. Table loads stay unmasked (pool entries are
  replicated to full VLMAX; vl == VLMAX so no stale-element issue).
- Compute on masked-off lanes is don't-care.
- Fully-masked chunks (tmp <= 0, skipped via blez) execute no memory ops.
- A chunk with 0 < tmp < VLMAX executes masked loads/stores (no fault past end).

### Why not Shape B (single strip-mining loop over the whole buffer)?

Dominated: U× more loop iterations than the counted loop for large n, each
paying a vsetvli, and the u=1 body loses per-column table-load amortization.
Would fail the large-size no-regression gate. Rejected in exploration.

## 3. v0 mask register reservation

RVV masked ops (vm=0) hardwire v0 as THE mask register. The backend currently:
- Has NO masked emitters (vm hardcoded to 1 in `em_opv`, `em_vle8`, `em_vse8`).
- Allocates accumulators starting at v0 (`(j*U+u)*lmul`).

Hazard: a masked load with vd == v0 under `ma` (mask-agnostic) lets hardware
overwrite masked-off lanes of v0 with 1s, destroying the mask mid-chunk.

Fix: for `maskedtail` twins, shift the vector register map up by one LMUL-group
(reserve v0 as the dedicated mask register) in the register assignment
(`jit_build_version` ~lines 1453-1530), and account the extra group in
`plan_need_regs` so `plan_candidates`' `need > nregs` filter rejects unencodable
combos before the backend sees them.

## 4. New emitters

All in `src/riscv/emit.h`/`emit.c`, following the existing emitter pattern.
Per AGENTS.md emitter lesson: clear ALL operand fields when deriving bases;
GAS cross-check across randomized operands (multiple register combos).

1. `em_vle8_v_m` / `em_vse8_v_m`: masked vector load/store (vm=0).
   Encoding: same as `em_vle8`/`em_vse8` with vm bit = 0.
2. `em_vid_v`: `vid.v vd` — vector instruction, vd only.
   Encoding: OP-V, funct6=010100, funct3=010 (OPMVV), vs1=10001, vm=1,
   vs2 field = 00000. GAS-verified: `vid.v v9` -> `0x5208a4d7`.
3. `em_vmsltu_vx`: `vmsltu.vx vd, vs2, rs1` — set mask if vs2[i] <u rs1
   (unsigned). Encoding: OP-V, funct6=011010, funct3=100 (OPIVX), vm=1.
   Unsigned (not `vmslt.vx`=011011 signed): at SEW=8 the scalar operand
   truncates to 8 bits (V spec §11.1), so tmp >= 128 would read negative
   under the signed compare. Exact for VLMAX <= 255 (vid.v indexes mod
   256); `jit_build_version` rejects `lmul * vlenb > 255`. vd MUST be v0
   (mask register); assert this. GAS-verified: `vmsltu.vx v0,v9,a3` ->
   `0x6a96c057`. (Note: 011100 is `vmsleu.vx`, unsigned <=.)

Each gets a shape test pinning the encoding bytes.

## 5. `cand_t` / planner / prune integration

- `cand_t.maskedtail` axis ALREADY EXISTS (`src/plan.h:84`, x86's twin).
  No new axis needed.
- `prune.c::twin_dominates`: `maskedtail` is ALREADY PINNED (line 225).
  Verify the pin covers the RVV case (it compares the field; ISA-independent).
- `plan_candidates`: enumerate `maskedtail=1` for RVV. Gate conditions:
  - `isa == ISA_RVV`
  - `cpu->rvv_zvbc` or appropriate feature flag (TBD: check what RVV features
    the backend requires; at minimum the base V extension)
  - Disjoint from `notail`, `predtail` (already enforced by twin construction)
  - Register pressure: `plan_need_regs` must account the v0 reservation (see §3)
- `plan_need_regs`: +1 LMUL-group when `maskedtail && isa == ISA_RVV`.

## 6. Backend implementation (`src/riscv/jit.c`)

- `cg_t`: add `int maskedtail` field (mirrors `notail`).
- `jit_build_version`:
  - Set `cg.maskedtail = cand->maskedtail`.
  - Validate: `maskedtail` requires `isa == ISA_RVV`; reject if `notail` or
    `predtail` also set (disjointness); reject if insufficient vector registers
    after v0 reservation (defense in depth; planner should have filtered).
  - Register map: shift accumulators up by one LMUL-group when maskedtail.
- Tail emission (replacing lines 1286-1310 when `cg->maskedtail`):
  - Emit the Shape A sequence from §2.
  - Thread a "masked data memory" mode through `c_vle`/`c_vse` (the data
    load/store helpers). `gen_body`'s compute emitters are untouched.
  - `vid.v` needs a temp vector register (not v0, not an accumulator).
- Static cost model: `cg->uops` increments for `vid.v` (1), `vmsltu.vx` per
  chunk (1 each), masked loads/stores (same as unmasked). The model is exact
  by construction (each emitter increments).

## 7. `api.c` twin wiring

Mirror the x86 maskedtail block (~lines 480-510):
- After the winner is selected, if `vs[winner].cand.isa == ISA_RVV` and
  feature-gated, build `mtwinc` with `maskedtail=1`.
- Run through `api_tail_duel_gate` with the "maskedtail" name.
- The twins[]-kind assert (~line 559) must accept the RVV maskedtail twin
  and map it to `TAIL_MASKED`.
- `tail_eff_maskedtail` reporting already exists; verify it covers RVV.

Note: `bench_tail_duel` is the install gate, but benchmarks are postponed.
The twin is built and correctness-validated now; the duel decision is deferred.
Pre-register: if the duel later rejects it, that's a rejected-with-evidence
twin, not a failure.

## 8. Test matrix

- **Differential**: maskedtail twin vs general variant (strip-mining tail) on
  ALL remainders R in 1..U*VLMAX-1, especially:
  - R < 16 (sub-vector)
  - R = VLMAX-1, VLMAX, VLMAX+1 (boundary)
  - R = U*VLMAX-1 (max)
  - R = 1 (single byte)
- **Matrix**: LMUL ∈ {1,2,4,8} × U ∈ {1..8} × plan kinds {P_XOR, P_MULACC,
  P_DELTA, P_HORNER} × topologies {RAID-5, RAID-6, RS(6,3)}.
- **Red-green**: flip a mask constant / drop the v0 reservation → tests fail.
- **Byte-identity**: sha256 of all kernels-doc sections before vs after —
  maskedtail unreachable from `make kernels` specs, so all match.
- **Combination**: maskedtail × {nt, prefetch} builds or rejects cleanly.
- **QEMU**: riscv64 tests pass (masked tail is RVV-only; x86/aarch64 unaffected).
- **ASan/UBSan**: clean.

## 9. Documentation

- `src/plan.h`: update `maskedtail` doc comment to mention RVV.
- Code comments at the v0 reservation and masked tail emission.
- `tools/kernels-README.md`: document the new twin if the showcase rule applies.

## 10. Done gates

1. Spec exists (this file).
2. No placeholders; all emitters GAS cross-checked.
3. Differential tests green across the full matrix.
4. Byte-identity proof for non-maskedtail kernels.
5. Full `make test` green; ASan/UBSan clean; QEMU riscv64 green.
6. 2+ independent peer reviews, all findings remediated.
7. Self-attack pass.
8. NO benchmarks, NO push.

## References

- Exploration: `~/workspace/worktrees/raidjit/fullmask/hidden_files/fullmask_rv_explore.md`
- x86 maskedtail: `src/x86_64/jit.c:gen_tail_masked`
- RVV tail: `src/riscv/jit.c:1286-1310`
- Twin wiring: `src/api.c:480-510`
- Emitter lesson: `~/AGENTS.md` ("Deriving emitter bases")
