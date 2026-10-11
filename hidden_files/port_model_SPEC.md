# Spec: Per-Microarchitecture Port-Pressure Cost Model

**Status**: spec (2026-10-10). Addresses Carlo's "I wish we had something more solid" — replaces the weighted-average heuristic (I3) with a true port-pressure model with quantified accuracy.

## 1. Problem

The current `plan_uarch_cost()` (src/plan.c, "I3") computes a **weighted average** of per-op-class reciprocal throughputs:

```
cost = (n_tl*t_ternlog + n_shuf*t_shuffle + ...) / ntot
```

This is wrong in principle: it assumes all ops contend equally. In reality:
- VPTERNLOGD (p0/p1) and VPSHUFB (p5) execute **in parallel** — no contention.
- Two VPSHUFBs contend on p5 — the cost is **max**, not average.

The correct throughput-bound model is:

```
cycles = max over ports p ( uops_on_port[p] / port_throughput[p] )
```

Carlo: "I feel like the current code size heuristic is a bit weak... I wish we had something more solid."

"Sold" = (1) principled (port pressure, not weighted average), (2) data-driven (published port tables, not hand-tuned), (3) falsifiable (quantified prediction error, not "trust me").

## 2. Model

### 2.1 Port-pressure vector

For each microarchitecture `u`, define:
- **Ports**: named execution ports (e.g., Intel: `p0`..`p7`; AMD Zen4: `fp0`..`fp3`, `agu0`..`agu3`; ARM N1: `b0`..`b3`).
- **Port throughput**: uops/cycle per port (usually 1.0; some ports are 0.5 for 512-bit on certain uarchs).

For each instruction class `c` (see §2.2), define:
- **Port distribution**: fraction of the instruction's uops that execute on each port. E.g., VPTERNLOGD zmm on ICL-SP: `{p0: 0.5, p1: 0.5}` (1 uop, dispatched to p0 or p1). VPSHUFB zmm on ICL-SP: `{p5: 1.0}`.

For a kernel with instruction mix `{n_c}` (counts per class per byte):
```
pressure[p] = sum_c n_c * dist_c[p]      # uops on port p per byte
cycles_per_byte = max_p pressure[p] / throughput[p]
```

This is the **throughput-bound** estimate. It assumes:
- Perfect OoO scheduling (all independent uops issue as ports free).
- No frontend bottleneck (uop cache / decoder not limiting).
- No memory bottleneck (handled by the existing roofline bandwidth leg).

### 2.2 Instruction classes

Classes are per-ISA, derived from the emitter sequences. x86-64 (AVX-512) initial set:

| Class | Instructions | Notes |
|-------|--------------|-------|
| `ternlog` | VPTERNLOGD/Q zmm | 3-input logic |
| `xor` | VPXORD/Q zmm | 2-input logic |
| `shuffle` | VPSHUFB zmm | Byte shuffle |
| `gfni` | VGF2P8AFFINEQB, VGF2P8MULB zmm | GF multiply |
| `pclmul` | VPCLMULQDQ zmm | Carryless multiply (for CRC) |
| `load` | VMOVDQA64 zmm (load) | Vector load |
| `store` | VMOVDQA64 zmm (store) | Vector store |
| `gpr` | MOV, ADD, CMP, etc. | Scalar overhead |

AArch64 (NEON/SVE) and RISC-V (RVV) classes defined in Phase 2/3.

### 2.3 Uarch coverage

**Phase 1 (this spec)**: x86-64
- Intel: SKX/CLX (0x55), ICL-SP (0x6A/0x6C), SPR (0x8F)
- AMD: Zen3 (0x19), Zen4 (0x19/0x10-0x1F), Zen5 (0x1A)

**Phase 2**: AArch64
- ARM Neoverse N1, V2
- Apple Silicon (M1/M2/M4) — best-effort from published data

**Phase 3**: RISC-V
- Generic RVV (ratified extensions only)

`RJ_UARCH_UNKNOWN` falls back to the current generic behavior (no regression).

### 2.4 Data sources (in priority order)

1. **uops.info** per-microarchitecture pages (most precise, per-instruction port breakdowns).
2. **Agner Fog instruction_tables.pdf** (in `hidden_files/crc_optimum_research/thirdparty/agner_tables.pdf`) — cross-check.
3. **Vendor SOGs** (ARM Neoverse Software Optimization Guides) — for AArch64.
4. Existing `rj_uarch_weights_t` values — as sanity-check priors, NOT as primary source.

Every table entry cites its source. No "est." without a source.

## 3. Data file format

Port tables live in data files, NOT in C code. Format: simple line-based text (parseable without dependencies):

```
# uarch: INTEL_ICELAKE
# ports: p0 p1 p2 p3 p4 p5 p6 p7
# class: ports (fraction, sums to 1.0 per uop)
ternlog: p0=0.5 p1=0.5
shuffle: p5=1.0
...
```

- One file per uarch: `data/uarch/intel_icelake.ports`
- Loader: `rj_ports_load(uarch)` → port model struct. Cached.
- New uarch = new file, no code change.
- File format versioned; loader validates (fractions sum to 1.0, ports exist).

## 4. Integration

### 4.1 Where the model plugs in

- `plan_uarch_cost()` keeps its signature (no API break). Internals switch from weighted-average to port-pressure max.
- `prune.c`: candidate ordering and the twin-dominance gate use the new cost. The gate's monotonicity requirement (load-bearing for prune soundness) must be re-verified.
- Heuristic selector: ranks candidates by new cost before benchmark.

### 4.2 What does NOT change

- **Benchmark remains the final arbiter** (Carlo's invariant). The model improves ranking; it never overrides measured results.
- `RJ_UARCH_UNKNOWN` behavior unchanged (generic fallback).
- The roofline bandwidth leg (`mem_per_byte / bw_bpc`) is kept; only the compute leg changes.

### 4.3 Performance

The model runs per candidate during planning. Must be O(instructions):
- Port-pressure vector: fixed small size (≤8 ports).
- Per candidate: one pass over instruction mix (already counted), dot products with port distributions.
- Data file loaded once per process (cached).

## 5. Validation (falsifiability)

Carlo explicitly wants "something more solid." That means **quantified accuracy**, not hand-waving.

### 5.1 Static validation (now, no benchmarks needed)

1. **Source cross-check**: Every table entry verified against ≥2 sources (uops.info + Agner, or uops.info + vendor SOG). Discrepancies documented.
2. **Known-ordering positive controls**: The model MUST predict:
   - GFNI < nibble-shuffle on ICL-SP (GFNI is 1 uop vs 4+ for nibble).
   - VPTERNLOGD < 2×VPXORD on all uarchs (1 uop vs 2).
   - PCLMULQDQ fold < slice-by-8 on SPR (published 24 GB/s vs 8 GB/s).
   If any control fails, the model is wrong — fix the tables, not the controls.
3. **Published-number spot checks**: For pure kernels (e.g., all-VPTERNLOGD), model prediction vs Agner/uops.info published throughput. Report % error.

### 5.2 Dynamic validation (when benchmarks resume)

Framework ready now, populated later:
- `tools/validate_port_model.c`: takes benchmark results (candidate, uarch, measured GB/s), computes model prediction, reports:
  - **MAPE** (mean absolute % error) per uarch.
  - **Rank correlation** (Spearman) — does the model rank candidates correctly? (More important than absolute accuracy for the planner's use case.)
  - **Top-k accuracy**: is the model's #1 pick in the measured top-3?
- Target: rank correlation > 0.8, top-3 accuracy > 90%. If not met, the model is "weak" — iterate on tables.

### 5.3 Red-green

- Corrupt a port table (e.g., swap p0/p5 for shuffle) → positive controls fail.
- Zero a port throughput → model predicts infinite cycles (division guard).

## 6. Test plan

1. **Unit**: port-pressure math (max-over-ports) on synthetic mixes.
2. **Table validation**: loader rejects malformed files (fractions ≠ 1.0, unknown ports).
3. **Positive controls** (§5.1.2) — must pass.
4. **Regression**: `plan_uarch_cost()` on the existing test corpus — no candidate ordering inversions vs old model that aren't explained by the port model (document each).
5. **Prune soundness**: re-run the prune tripwire corpus; the monotonicity proof must hold for the new cost function.

## 7. Phases

- **Phase 1**: x86-64 port tables (Intel SKX/ICL/SPR, AMD Zen3/4/5) + loader + `plan_uarch_cost` rewrite + static validation.
- **Phase 2**: AArch64 tables (N1/V2, Apple best-effort) + `rj_cpu_uarch()` extension for ARM.
- **Phase 3**: RISC-V tables + `rj_cpu_uarch()` extension for RVV.
- Each phase: spec → implement → validate → peer review.

## 8. Open questions

1. Should the model include a simple critical-path (latency) component for short stripes? (Deferred: throughput-bound covers the common case; revisit if validation shows systematic error on small sizes.)
2. Frequency derating (freq512): keep the existing estimate or derive from published data? (Keep existing for now; flag as "est.")
3. How to handle uarchs with no published port data (new AMD/Intel)? (Fall back to nearest known uarch + log a warning.)

## 9. References

- uops.info (per-uarch instruction tables)
- Agner Fog, instruction_tables.pdf (thirdparty/)
- Intel Intrinsics Guide (latency/throughput, secondary)
- ARM Neoverse N1/V2 Software Optimization Guides (Phase 2)
- Current: src/plan.c `rj_uarch_weights`, `kernel_w_uop`, `plan_uarch_cost`
- Brainstorm: hidden_files/brainstorm_isa_uarch_2026-10-06.md §3
