# Fast CRC-32: standalone API (Option C, Phase 1)

`uint32_t rj_crc32_fast(uint32_t crc, const void *data, size_t len);`
(declared + documented in [include/raidjit.h](../include/raidjit.h)).

Drop-in faster replacement for `rj_crc32_update` with an **identical
chaining contract**: takes the running CRC, returns the updated running
CRC. Bit-exact — `rj_crc32_fast(c,d,n) == rj_crc32_update(c,d,n)` for
every input (proven by [tests/test_crc_fast.c](../tests/test_crc_fast.c), not by assertion).
Initialize with `rj_crc32_init`, finalize with `rj_crc32_final`:

```c
uint32_t c; rj_crc32_init(&c);
c = rj_crc32_fast(c, chunk1, n1);
c = rj_crc32_fast(c, chunk2, n2);   /* chaining across calls */
uint32_t out = rj_crc32_final(c);
```

`len == 0` is legal (`data` may then be `NULL`). Thread-safe: dispatch
holds no mutable state. See [include/raidjit.h](../include/raidjit.h) for the edge-case contract
(Finding 17b).

## Dispatch

Top level ([src/crc_fast.c](../src/crc_fast.c)): compile-time ISA selects the backend's
*best-variant* entry; the backend does its own runtime feature
selection internally. WASM (`__EMSCRIPTEN__`) and unknown ISAs go
straight to slice-by-8 (spec Finding 17c). Unknown/absent features
degrade to a slower CORRECT variant — never wrong, never SIGILL.

### x86-64 (best first)

| Class | Condition | Variant |
|---|---|---|
| 3 | `vpclmulqdq && pclmulqdq && avx512f && avx512bw` (`avx512bw` joins per spec review R1-6: the 512-bit fusion's constant broadcast/lane reduction need it) | Cawley-style fusion: PCLMUL-based scalar lanes (`crc_u64_step` — the x86 `crc32` instruction is CRC-32C-only and unusable here) + VPCLMULQDQ vector lanes (rung 1); 256-bit VPCLMULQDQ fusion lanes (rung 2, Zen3: `vpclmulqdq` without `avx512f`). The `rj_crc32_fast_x86_vpclmul` entry selects the rung internally (spec Finding 13); `RJ_CRC_X86_FORCE256=1` forces rung 2 on AVX-512 hosts for test coverage |
| 2 | `pclmulqdq` (no tier-3) | 128-bit PCLMULQDQ fold, zlib-ng/Chromium structure (the synthesis table — not fusion — is the arbiter here) |
| 0 | else | slice-by-8 |

**There is no `crc32`-instruction tier on x86** (W1 finding, documented
in [src/x86_64/crc_fast.c](../src/x86_64/crc_fast.c)): the x86 `crc32` instruction implements
CRC-32C (Castagnoli, 0x1EDC6F41), **not** the IEEE CRC-32 (0x04C11DB7)
this library computes — it cannot be used here, not for a scalar lane,
not for an N-way interleave. Consequently `rj_crc32_fast_x86_crc32hw`
delegates to slice-by-8, the internal selector skips it, and class 1 is
never returned on x86 (an `sse42`-only machine reports class 0). The
spec's R2 x86 tier 4 ("hw `crc32q` N-way interleave") and the
synthesis's SSE4.2 row describe CRC-32C throughput, not IEEE CRC-32 —
flagged for spec remediation; the implementation is authoritative.
(ARM differs: ARMv8 has distinct IEEE `crc32b/h/w/x` and Castagnoli
`crc32cb/ch/cw/cx` instructions, so the ARM backend's class 1 is real.)

### AArch64 (best first)

| Class | Condition | Variant |
|---|---|---|
| 3 | `pmull && sha3 && crc32` | Cawley fusion (`crc32_armv8_pmull_eor3.c` structure) — the `crc32` conjunct is required because the fusion's lane management uses the scalar crc32 unit (spec amendment 2026-10-09, peer-review-blessed) |
| 1 | `crc32` | scalar `crc32x` N-way interleave — **before** the pure PMULL fold because of the documented N1 inversion (pure PMULL fold 7.3 B/c < scalar crc32x 7.9 B/c on N1); Apple cores always take class 3, so no Apple regression |
| 2 | `pmull` (no CRC32 hw) | pure PMULL fold (dougallj structure) |
| 0 | else | slice-by-8 |

### RISC-V (best first)

| Class | Condition | Variant |
|---|---|---|
| 2 | `rvv_zvbc` | Zvbc vector fold (SEW=64, streams packed as elements) |
| 1 | `zbc` | scalar clmul fold (zlib-ng `crc32_zbc.c` structure) |
| 0 | else | slice-by-8 |

Fallback chain on every ISA ends at slice-by-8. Class numbers are part
of the pinned cross-worker contract (spec R2.5).

## Where it is NOT the fastest (honest)

- **Slice-by-8 baseline**: the class-0 fallback everywhere. Synthesis
  figure: 1–3 GB/s class (spec Finding 17a: cite this, not "<10").
- **x86 without CLMUL**: there is no `crc32`-instruction tier for IEEE
  CRC-32 (see the x86 table above) — an `sse42`-only x86 machine gets
  slice-by-8. The Chorba scalar construction (Russell 2024, reported "on
  par with hw") would beat it and is **not implemented** — recorded as a
  future lead, not a silent gap.
- **AArch64 N1 without SHA3/PMULL**: scalar `crc32x` (class 1) is picked
  over the pure PMULL fold (class 2) deliberately — the inversion is
  load-bearing, see the table above.
- **RISC-V**: no shipping core publishes clmul/vclmul throughput, so the
  Zvbc/Zbc bounds are in instructions/byte only, honestly not convertible
  to GB/s.
- **WASM**: always slice-by-8 by construction (Finding 17c).

## Performance numbers: analytic only

Every throughput figure in this document and in
[hidden_files/crc_optimum_research/FINAL_SYNTHESIS.md](../hidden_files/crc_optimum_research/FINAL_SYNTHESIS.md) is **analytic**
(derived from instruction-throughput tables, e.g. Agner Fog) or
**reported** from the literature (Cawley et al.). **None was measured in
this repo** — timing benchmarks are out of scope by standing constraint.
No number in code comments, docs, or commit messages may present an
analytic/reported figure as measured (spec assumption 5).

## Backend contract (for W1/W2/W3)

Pinned entry points, exact names (spec R2.5):

- x86-64 ([src/x86_64/crc_fast.h](../src/x86_64/crc_fast.h)): `rj_crc32_fast_x86`,
  `rj_crc32_fast_x86_vpclmul`, `rj_crc32_fast_x86_pclmul`,
  `rj_crc32_fast_x86_crc32hw`, `rj_crc32_fast_x86_class`
- AArch64 ([src/aarch64/crc_fast.h](../src/aarch64/crc_fast.h)): `rj_crc32_fast_arm`,
  `rj_crc32_fast_arm_fusion`, `rj_crc32_fast_arm_crc32hw`,
  `rj_crc32_fast_arm_pmull`, `rj_crc32_fast_arm_class`
- RISC-V ([src/riscv/crc_fast.h](../src/riscv/crc_fast.h)): `rj_crc32_fast_riscv`,
  `rj_crc32_fast_riscv_zvbc`, `rj_crc32_fast_riscv_zbc`,
  `rj_crc32_fast_riscv_class`

Dispatch mechanics: [src/crc_fast.c](../src/crc_fast.c)'s dispatcher references the
best-variant entries with strong-undefined references so the linker
extracts a real backend TU from `libraidjit.a` for *every* consumer.
`rj_crc32_fast_test_backend_present()` (test-only, defined in
[src/crc_fast.c](../src/crc_fast.c)) reports whether the statically bound entry is a real
backend (`g_crc_fast_fn != rj_crc32_update`), so
[tests/test_crc_fast.c](../tests/test_crc_fast.c)'s dispatch assertions go strict exactly when a
backend is linked and defer (loudly, or fail under `RJ_CRC_STRICT=1`)
on the slice-by-8 fallback path. (During development a transitional
`src/crc_fast_fallback.c` provided weak slice-by-8 definitions of the
entry points until all three backend TUs landed; it has since been
deleted with its Makefile stanza, and the probe moved into the
dispatcher.)

## Testing

[tests/test_crc_fast.c](../tests/test_crc_fast.c) implements the spec R4/R5 matrix; see its header
comment for the section map. Highlights:

- KAT `CRC32("123456789") == 0xCBF43926` through every entry point.
- Bit-exactness vs three independent oracles: an in-test bitwise
  implementation (shares no code with [src/crc.c](../src/crc.c)), zlib `crc32()` when
  linked (`-DRJ_CRC_HAVE_ZLIB=1`, probed by the [Makefile](../Makefile)), and
  `rj_crc32_update_scalar`.
- Split-point chaining at every split offset mod 64 (spec R1-3).
- Guard pages + ASan for over-reads; 8-thread concurrency; 2^32±16
  truncation differentials via zero-page `mmap` (no 4GB RAM).
- Dispatch assertions compare `*_class()` against independent hardware
  detection (test-side CPUID / `getauxval` / `/proc/cpuinfo` isa parsing),
  never against the library's own `rj_cpu_detect`.

Controls:

- `RJ_CRC_SKIP_HUGE=1` skips the 2^32 test (used by the QEMU Makefile
  targets — minutes under TCG — and anywhere memory is tight).
- `RJ_CRC_STRICT=1` turns deferred backend-absent dispatch checks into
  failures (proves the deferral path is not a silent pass).

Red-green: [tests/mutate_crc.sh](../tests/mutate_crc.sh) applies a single-constant mutation in a
scratch git worktree, rebuilds `test_crc_fast` there, and expects it to
FAIL. A surviving mutant means the matrix is insufficient. Run per
backend once W1/W2/W3 land (e.g. flip K1/K2 low bit, drop an interleave
lane — spec Finding 16).

## Code-review assertions (no wraparound)

Spec R5 requires the truncation test **plus** a code-review assertion.
For W4-owned code:

- [src/crc_fast.c](../src/crc_fast.c): no arithmetic on `len` at all — pure forwarding.
- [src/crc.c](../src/crc.c) (`rj_crc32_update`, `rj_crc32_update_scalar`): all
  block/byte counts are `size_t`; loops decrement toward zero; no
  32-bit intermediates, no pointer wraparound (buffers are caller-valid
  by contract).
- [src/fused.c](../src/fused.c) accumulation: forwards per-tile lengths (`tl <= tile`);
  no new arithmetic.
- Backend block-count arithmetic is W1/W2/W3's assertion to record.

## WASM coordination

The WASM build ([`tools/wasm/*`](../tools/wasm/), coordinator-owned per the spec) wires
its own sources; the CRC contract it must honor is: (1) link
[src/crc_fast.c](../src/crc_fast.c) (or provide the `rj_crc32_fast` symbol — `fused.c`
calls it), and (2) the `rj_cpu_info` field additions are additive, so
the `cpu_wasm_*.c` stubs keep working as long as they zero-initialize
the struct. The WASM track owns this wiring — details live in
[tools/wasm/](../tools/wasm/), not here.

## Consistency

This document agrees with
[hidden_files/crc_optimum_research/FINAL_SYNTHESIS.md](../hidden_files/crc_optimum_research/FINAL_SYNTHESIS.md) (Phase 1 table)
and the remediated [hidden_files/crc_impl_SPEC.md](../hidden_files/crc_impl_SPEC.md) (R1–R8, Findings
13/16/17a–17c). The per-ISA optimum rows, the N1 inversion note, the
Zen3 256-bit-lane row, and the class enum follow the spec verbatim.
