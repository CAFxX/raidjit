# raidjit

JIT compiler for RAID encode / erasure-recovery kernels. Given a topology
(RAID-0/1/5/6, generic Reed-Solomon, or a custom parity matrix) and an
operation (encode, recover N failed disks, decode-and-encode), raidjit
generates specialized x86-64 machine code for the exact matrix at hand,
benchmarks every legal codegen candidate on the host CPU, and keeps the
winner. No external dependencies; C11, `-Wall -Wextra -Werror` clean.

## Quick start

```c
#include <raidjit.h>

rj_topology *t = rj_topo_raid6(8);          /* 8 data + 2 parity */
rj_op enc = rj_op_encode();
rj_options o; rj_options_default(&o);       /* benchmark selection on */
char err[256];
rj_jit *j = rj_compile(t, &enc, &o, err, sizeof(err));
rj_fn fn = rj_get_fn(j);
fn(in /* 8 data ptrs */, out /* 2 parity ptrs */, len /* bytes */);

/* lose disks 2 and 5, rebuild them from the survivors */
int failed[] = {2, 5};
rj_op rec = rj_op_recover(failed, 2);
rj_jit *r = rj_compile(t, &rec, &o, err, sizeof(err));
rj_get_fn(r)(survivors /* ptrs in increasing disk order */,
             rebuilt /* 2 ptrs, increasing failed order */, len);
```

`make` builds the static lib, all tests, and `examples/demo.c`.
`make test` runs the full suite. `make test-asan` rebuilds with
ASan+UBSan. `make test-qemu` runs the ISA matrix under qemu-user.

## API

- **Topologies**: `rj_topo_raid0/raid1/raid5/raid6(n)`, `rj_topo_rs(ndata,npar)`,
  `rj_topo_custom(ndata, npar, matrix)` (row-major `M[j][i]`,
  `parity[j] = XOR_i data[i]*M[j][i]` in GF(2^8)/0x11B).
  `rj_topo_raid6(n)` dispatches to the benchmark-selected default code for
  that `ndata` (recorded in `docs/xor_code_default.md`; currently RS at
  every ndata — see below); `rj_topo_raid6_rs(n)` is the explicit RS
  RAID-6, `rj_topo_rdp(p)` / `rj_topo_evenodd(p)` / `rj_topo_liberation(k)`
  are the XOR array codes (RDP, EVENODD, Liberation).
- **XOR array codes and the expanded row model**: RDP/EVENODD/Liberation
  are binary (XOR-only) MDS codes whose diagonal parities mix bytes
  *across rows* of a data disk, so each **row** is exposed as a logical
  disk. With `k` data disks of `R` rows: `ndata = k*R` row-disks,
  `npar = 2*R` parity rows; `in[d*R + r]` is row `r` of data disk `d`,
  `out[r]` is P parity row `r`, `out[R + r]` is Q parity row `r`.
  `rj_xorcode_rows(t)` reports `R` for an XOR-code topology, 0 for
  anything else — use it to map a whole-disk failure of disk `d` to the
  expanded indices `d*R .. d*R+R-1` for `rj_op_recover`. Every parity row
  is a pure XOR of input rows, so these topologies lower to the existing
  `P_XOR` plan path automatically. Full construction details:
  `docs/xor_codes.md`.
- **Ops**: `rj_op_encode()`; `rj_op_recover(failed[], n)` — failed disk
  indices `0..ndata+npar-1`; the op owns a sorted copy, free with
  `rj_op_free`. Recovery is lowered to survivor linear combinations by a
  Gauss-Jordan solver at compile time. When more survivors exist than
  strictly needed, the cheapest measured survivor set is selected at
  plan time (see `docs/survivor_sets.md`); the winning rows are recorded
  in the plan desc as `[rows 0+2]`. Set `RJ_NO_SURVIVOR_SELECT=1` to
  disable survivor-set selection and take the legacy single-subset
  (all-rows) recovery path. `rj_op_parity_update(changed[], n)` —
  partial-stripe update: `in` = `[old_0..old_{nc-1}, new_0..new_{nc-1},
  oldpar_0..oldpar_{np-1}]`, computes `new_p[j] = old_p[j] XOR
  Σ_i((old[i] XOR new[i]) * M[j][changed[i]])`. `out[j]` may alias
  `in[2*nc+j]` (in-place).
- **Compile**: `rj_compile(topo, op, opts, err, errlen)` → `rj_jit *`
  (NULL + message on failure). `rj_get_fn()` returns
  `void fn(const void *in[], void *out[], size_t len)`. `rj_get_info()`
  reports the chosen implementation, measured GB/s, and static cost
  model. `rj_free()` releases everything.
- **Calling convention of generated code**: `in[]`/`out[]` pointer arrays
  in RDI/RSI, byte length in RDX. Encode: `in` = data ptrs, `out` = parity
  ptrs (`out` may be NULL when `npar == 0`). Recover: `in` = surviving
  disk ptrs in increasing disk-index order, `out` = one ptr per failed
  disk in increasing failed-index order. Any `len`; every buffer must be
  64-byte aligned and no two buffers may overlap or alias (unchecked);
  all vector memory ops are unaligned-safe. Generated code makes no calls,
  touches no callee-saved registers except RBX/R12–R14/R15 (saved/restored), and
  never reads/writes outside `[0, len)` of the given buffers.

## Code generation

Three plan kinds, picked per matrix row by inspection:

- `P_XOR` — row is all-ones: pure XOR accumulation.
- `P_HORNER` — row is geometric `[c^{n-1}..c^0]`: Horner accumulation
  `acc = acc*c ^ in[i]` (one multiply per input, the optimal schedule).
- `P_MULACC` — general row: multiply-accumulate per input.

Multiply methods: nibble-shuffle tables (SSSE3/AVX2/AVX-512; needs
AVX512BW on the 512-bit path) or `VGF2P8AFFINEQB` (GFNI). ISA levels:
SSE (2 regs… 16), AVX2 (16), AVX-512 (32). Candidates span unroll
factors 8 down to 1, filtered by exact register-pressure formulas, so every
generated candidate is encodable. On AVX-512, `P_XOR` plans with
`n_out >= 2` additionally get a VPTERNLOGD twin that folds each column
pair's two per-row XORs into one ternary-logic uop (`+ternlog` in the
candidate desc). `P_XOR` plans with cross-row shared subexpressions get
an XOR-schedule twin (`+xsched`): a plan-time greedy CSE builds a
straight-line program of shared `t = a^b` temps, and the twin's body
computes each temp once and fans it out (see docs/xor_schedule.md).
The twin is enumerated only on a strict static uop win and the benchmark
picks empirically. With `nt_threshold > 0`, every
candidate additionally gets a non-temporal twin that stores outputs
with `MOVNTDQ`/`VMOVNTDQ` (x86), `STNP`/`STNT1B` (AArch64), or the
Zihintntl `NTL.ALL` hint before each vector store (RISC-V). The jitted
code executes no fence: a caller that reads NT-written data must fence
itself afterwards on x86 (`SFENCE`) and AArch64 (`DMB ISHST`); no
caller fence is needed on RISC-V, where the hint does not change
memory ordering. The benchmark picks the twin
empirically, so NT stores are only used where they win. Hot loops are
aligned (`loop_align`, default 64/32/16 by vector width, must be 0 or a
power of two); scalar tails handle any `len`, including 0. Constant
pools are 64-aligned, RIP-relative; code is mapped RW, filled, then
`mprotect`ed RX (no RWX window). Optional hardening: `endbr64` at entry,
shadow-stack `setssbsy`/`clrssbsy` (needs OS-enabled CET), `ud2` trap
sleds after `ret`.

Length-specialized dispatch: after the benchmark picks the winning
candidate, `rj_compile` also builds its notail twin (same loop, tail
code omitted) and benchmarks the pair head-to-head. If the twin wins,
the returned `rj_fn` is a dispatcher (shown as `+notail` in the info
desc) that routes `len % group == 0` to the twin and every other length
to the general version; if the twin loses, nothing changes. The twin is
validated at group multiples before it is allowed to win, so a wrong
twin fails compilation instead of shipping. The dispatcher costs one
admission test plus one indirect call per invocation (a few ns,
unmeasurable above VM noise). Details and the falsifiable ship
criterion: `docs/length_dispatch.md`.

## Correctness limits (hard guarantees)

- **MDS envelope of the default matrix.** `M[j][i] = 2^(j*(ndata-1-i))`
  with `ord(2) = 51`. The matrix bytes are emitted by the plan-time
  generator-matrix search (`docs/matrix_search.md`), which proves MDS
  by exact minor enumeration for every emitted matrix. The named
  constructors deliver true Reed-Solomon semantics (every ≤ npar erasure
  set recoverable) only for `npar ≤ 3` and `ndata ≤ 51` (`npar ≤ 1` has
  no basis limit). They return NULL outside this envelope. Wider MDS
  topologies (e.g. Cauchy, MDS for `ndata+npar ≤ 255`) via
  `rj_topo_custom()`. Out-of-envelope or
  singular custom matrices fail `rj_compile` with a clean error — never
  silently wrong parities.
- **Parity-count codegen ceilings.** All accumulators stay in vector
  registers; beyond the ceiling `rj_compile` fails cleanly with
  "no candidates for this CPU/plan". At U=1: Horner/nibble needs
  `3*n+6 ≤ nregs` (n ≤ 3 SSE/AVX2, ≤ 8 AVX-512), Horner/GFNI needs
  `2*n+2 ≤ nregs` (≤ 7 / ≤ 15), mulacc/nibble needs `n+7 ≤ nregs`
  (≤ 9 / ≤ 25). Recovery of `nf` failures uses `nf` in place of `n`.
- **ISA gating** is by CPUID + OS state: AVX/AVX-512 need the XCR0 bits,
  GFNI needs AVX512F hardware + OS state, AVX512BW-gated paths need the
  BW bit. 512-bit loads use `VMOVDQU32` (AVX512F-only), so AVX512F-without-BW
  CPUs (e.g. Xeon Phi) are supported.
- Thread-safe: `rj_compile` shares one piece of process-wide mutable
  state -- the generator-matrix search cache in `src/matrix_search.c`
  (one entry per (ndata, npar), written once per shape). It is guarded
  by a C11 `atomic_flag` spinlock and the search itself is deterministic,
  so concurrent first-calls compute identical bytes. Generated code
  is not async-signal-safe (it is ordinary computational code).

## Benchmark methodology and performance claims

With `benchmark = 1` (default), every candidate is first validated
against an independent scalar reference at scalar-only, vector-boundary,
multi-group, and unaligned-offset lengths — a wrong candidate fails
compilation before it can win on speed. Timing uses several working-set
sizes (`bench_bytes/4`, `bench_bytes`, `bench_bytes*4`, clamped to
4 KiB–16 MiB, default 1 MiB), warm-up, and interleaved rounds; scores
are sums of per-size median times. Eligibility window =
`max(1.5%, pooled measurement noise)` where noise is the pooled
(median-over-candidates of median-over-sizes) relative IQR; candidates
tied within noise resolve deterministically by lower µops/byte, then
memory traffic/byte, then code size.

A reported GB/s therefore means exactly: **best among the generated
legal candidates under this protocol on this CPU** — not a portable or
absolute claim. Set `benchmark = 0` for a heuristic pick without timing.

Before benchmarking, statically-dominated candidates are pruned: the
plain member of a `+ternlog` twin pair is dropped when the twin (same
plan kind, ISA, mul, unroll, NT, prefetch; only the multiply
instruction differs) is no-worse on µops/byte, memory/byte, and code
size and strictly better on at least one — and only for plan kinds
`P_MULACC`/`P_DELTA`, where the twin direction is validated (see
`docs/candidate_pruning.md`). No winner change has been observed on the
validated corpus/host; that is an empirical result, not a theorem —
the rule is safe iff no pruned version would be the argmin-time
candidate, and `tests/test_prune.c` Part 3 asserts exactly that as a
tripwire on every run, so a future model/codegen change that breaks
the precondition fails loudly instead of flipping winners silently.
Set `RJ_NO_PRUNE=1` to disable pruning entirely.

## Testing

- `test_emit`: differential emitter test — every emitted encoding checked
  byte-for-byte against GNU as reference vectors (30 vectors incl. all
  high-register EVEX permutations, compressed-displacement boundaries).
- `test_affine`: GFNI affine layout validated on hardware (all 256
  constants × 64 inputs); transposed layout as negative control.
- `test_correct`: encode/recover against independent scalar oracles,
  unaligned offsets, vector/tail boundaries, MDS-envelope pinning.
- `test_matrix`: 195,874-run exhaustive matrix — every ISA level
  (incl. simulated AVX512F-without-BW/GFNI and SSE2-only CPUs) ×
  topologies × every failure subset up to the parity count × unaligned
  offsets × guard pages × decode-and-encode chains. Zero failures.
- `test_select` (16 cases): selection unit tests incl. tie-breaks,
  adaptive-window math, NaN rejection.
- `test_benchreject`: corrupts a candidate's constant pool and requires
  the benchmark to reject it before timing.
- `test_cet`: `endbr64`/`ud2`/shadow-stack behavior, CPUID/XCR0 gating.
- `make test-asan`: full suite under ASan+UBSan. `make test-qemu`:
  matrix under qemu-user for qemu64/+ssse3, Haswell, Skylake-Server,
  Icelake-Server (QEMU 8.2 TCG lacks AVX-512/GFNI execution; those paths
  are covered natively).
- Four independent peer reviews (emitter, GF/planner, JIT backend,
  benchmark methodology) audited the implementation; all findings were
  addressed and re-tested.
- `test_codegen_shape`: generated-code shape assertions (duplicate-load
  elimination, table dedup, `vzeroupper`, VPTERNLOGD, pointer hoisting,
  coefficient-aware allocation) — every assertion red-green verified.
- `test_nt`: NT-store twins — shape (no fence in jitted code, ≥1 NT
  store), output equality, caller-side fence modeled after each call.
- `test_pdelta_edge`: P_DELTA adversarial coverage — max-width updates,
  200-disk topologies, guard pages, input immutability, aliasing, NT+fence.
- `test_boundaries`: systematic 0/1/VL−1/VL/VL+1 main/tail seam tests.
- `test_rvv_f6`, `test_pxor_ternlog`: RVV counted-loop and VPTERNLOGD
  shape + differential tests.
- `test_aarch64_emit`, `test_aarch64_remat`, `test_riscv_remat`,
  `test_riscv_prefetch`, `test_sve_eor3`, `test_sve_pdelta`,
  `test_scan_align`, `test_remat`: backend-specific emitter, rematerialization,
  prefetch, and alignment tests.

## Known limitations

- Out-of-memory inside the JIT aborts via `assert` (fail-fast, no silent
  corruption); error propagation is not yet threaded through the emitter.
- No accumulator spilling: parity counts above the codegen ceilings are
  rejected rather than compiled with spills (see limits above).
- `rj_info.gbps` with `benchmark = 0` is a heuristic estimate, not a
  measurement.

Full details, including caller contracts that are *not* checked at runtime
(survivor ordering, buffer overlap) and the exact scope of the benchmark
performance claims: [docs/limitations.md](docs/limitations.md).
