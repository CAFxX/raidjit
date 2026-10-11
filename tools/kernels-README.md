# Kernel Dumps: RAID-5(4+1) and RAID-6(5+2)

> **Note 2026-10-09:** the static HTML explorer built from these dumps
> ([`raidjit_kernels.html`](wasm/out/raidjit_kernels.html)) has been superseded by the **live WASM kernel
> explorer** at the same deliverable path — the page now compiles any
> option combination in-browser via WebAssembly (see
> [`tools/wasm/README.md`](wasm/README.md), [SPEC](wasm/SPEC.md) §S7). The Markdown reference below and
> the shipped `.code.bin`/`.const.bin`/`.asm` files remain the archived
> verification corpus (S9.3 byte-identity proof, parity harness).

Regenerated 2026-10-07 with `make kernels` (see [`tools/gen_kernels.c`](gen_kernels.c) in the
raidjit repo). Each kernel was JIT-compiled with `benchmark=0` (heuristic
candidate selection) for a synthetic CPU matching the target ISA — never the
host CPUID, since the cross targets differ from the build host.

## Selection method

`heuristic_select` (with `benchmark=0`, so no timing): highest ISA wins;
among candidates of equal ISA, the minimum roofline estimate
`max(uops/3.5, mem/64)` cycles/byte wins; exact ties keep the earliest
candidate. Cross-ISA empirical benchmarking under QEMU is
cycle-meaningless, so no timing is used. All candidates are built and
ranked — this is heuristic selection, not first-match.

## Verification

Every kernel was executed in-process against the scalar reference
([`src/ref.c`](../src/ref.c)) at 6 lengths (0, 1, 7, 1003, 65537, 1 MiB) × 3 seeds = 18
runs, on both the live JIT image and a fresh executable mapping of the
shipped `.code.bin`/`.const.bin` bytes (proving the shipped bytes are the
tested bytes). Buffers 64-byte aligned per the API contract. A red-green
proof (const-pool byte flip + corrupted-expectation, in `/tmp`, never
shipped) confirmed the checks are not vacuous: both corruptions are
detected. (The flip test needs a const pool, so it is skipped for the
four kernels with `const_size == 0` (the three plain RAID-5 kernels plus
the raid5-4_sve2 predtail twin); the corrupted-
expectation check runs for all nine.)

An ISA marker assertion runs at generation time and fails the build if
absent (counts below are occurrences in the emitted code bytes).

## RAID-5 (4 data + 1 parity) — XOR

| ISA | Desc | Code | Const | Marker |
|-----|------|------|-------|--------|
| avx512-gfni | raid5-4 xor avx512 U=8 | 512 B | 0 B | VPXORD (EVEX.66.0F.W0 EF) ×24 |
| avx512-gfni | raid5-4 xor avx512 U=8 maskedtail | 576 B | 520 B | KMOVQ (tail-mask load into k1) ×1 |
| rvv | raid5-4 xor rvv U=5 | 784 B | 0 B | RVV vector opcodes ×37 |
| sve2 | sve xor U=5 | 512 B | 0 B | SVE codegen path ×21 (EOR3 n/a, see below) |
| sve2 | sve xor U=5 predtail | 512 B | 0 B | WHILELT (predicated-tail predicate) ×1 |

RAID-5 parity is pure XOR (no GF multiplication), so the avx512 kernel uses
`VPXORD` (not GFNI); "gfni" in the filename names the target CPU.

The `maskedtail` / `predtail` rows are the item-14 tail twins: complete
versions of the winning candidate whose scalar tail loop is replaced by the
masked AVX-512 tail (`KMOVQ` k1 from the 520-byte tail-mask pool, `{k1}{z}`
loads / `{k1}` stores) or the fully predicated SVE tail (`WHILELT`). Built
from the winner exactly as [`api.c`](../src/api.c) does; see [`tools/gen_kernels.c`](gen_kernels.c).

## RAID-6 (5 data + 2 parity) — Horner

| ISA | Desc | Code | Const | Marker |
|-----|------|------|-------|--------|
| avx512-gfni | raid6-5 horner avx512+gfni U=8 | 1664 B | 576 B | VGF2P8AFFINEQB (EVEX.66.0F3A.W1 CE) ×32 |
| rvv | raid6-5 horner rvv+nibble U=5 | 2192 B | 32 B | RVV vector opcodes ×335 |
| sve2 | sve horner+nibble U=4+eor3 | 1152 B | 640 B | EOR3 (SVE2) ×16 |
| sve2 | sve horner+nibble U=4+eor3 predtail | 1024 B | 80 B | WHILELT (predicated-tail predicate) ×1 |

RAID-6 Q parity uses Horner accumulation with GF(2⁸) multiplies:
- avx512+gfni: `VGF2P8AFFINEQB` (single-instruction GF multiply)
- rvv: nibble-shuffle via `vrgather` (32-byte table)
- sve2: nibble-shuffle via `TBL` with the lo^hi^input reduction folded
  into one SVE2 `EOR3` per multiply (640-byte table)

## On the "sve2" label

The previous kernel set (2026-10-01 12:28) was built before the SVE2
audit; its "sve2" kernels contained zero SVE2 instructions. That is fixed
here: `raid6-5_sve2` contains 16 `EOR3` instructions (the `+eor3` desc tag
is exact — it is set iff ≥1 EOR3 is emitted; see [`docs/sve2_audit.md`](../docs/sve2_audit.md)).

`raid5-4_sve2` cannot contain EOR3 by construction: P_XOR is pure XOR with
no GF multiply for EOR3 to fold ([`docs/sve2_audit.md`](../docs/sve2_audit.md)). Its marker is the
SVE codegen path itself — desc `sve xor U=5` and SVE major-opcode
instructions in the bytes. Both kernels were compiled for the SVE2 target
CPU; the one that can use SVE2 instructions does.

## Files

- `<name>.code.bin` — raw machine-code bytes (entry at offset 0)
- `<name>.const.bin` — constant pool bytes (must follow the code at
  exactly `code_size` bytes; pool references are PC-relative in all
  backends, so code+const form one relocatable image)
- `<name>.asm` — disassembly of the exact shipped bytes
- `manifest.json` — machine-readable: per kernel (target, topology, op,
  desc, strategy, unroll, sizes, file names + sha256, selection method,
  ISA marker + count) plus the entry contract

All of the above (including this README) are installed by
[`tools/finalize_kernels.sh`](finalize_kernels.sh) via `make kernels`; the README source lives at
[`tools/kernels-README.md`](kernels-README.md) in the repo.

Entry contract: call as `rj_fn(const void *in[], void *out[], size_t len)`.
Every buffer 64-byte aligned; no two buffers may overlap or alias. No
fences are emitted; the caller owns visibility per the ISA memory model.

Disassembly method: x86-64 and AArch64 via `objdump -b binary` with the
matching `-m`. RISC-V: each 32-bit word is wrapped as a `.word` directive
under an `$x` mapping label, assembled with
`riscv64-linux-gnu-as -march=rv64gcv`, and disassembled with
`riscv64-linux-gnu-objdump -d` (binutils 2.42 decodes RVV 1.0:
`vle8.v`, `vsetvli`, `vrgather.vv`, …). Raw `-b binary` cannot select the
V extension, which is why older listings showed `.insn` raw words.

## What changed vs the previous set (2026-10-05 → 2026-10-07)

New representative specs (Lane-2 capabilities):
- LRC family (A6): `lrc-azure` (12,2,2 Azure-style), `lrc-azure-rec0`
  (full recovery `[rows 0+1+2+3]`), `lrc-azure-rec0-local` (the local-
  repair kernel the A5 survivor machinery selects: `[rows 2]`, pure
  P_XOR), `lrc-opt-20-2-4` (distance-optimal), `lrc-8-2-2` (general
  mechanism). Each on all three targets with eligible twins.
- I2 VPMULTISHIFTQB twin (`_mshift`): forced mshift=1 on the first
  eligible AVX512 MUL_NIBBLE base; marker-verified in the disassembly.
- I8 software-pipelined twin (`_pipe`): forced pipe=1 on the first
  eligible NEON/SVE/RVV MUL_NIBBLE base.
- A1 xorbit twin (`_xorbit`): implemented but the strict uop-win gate
  did not fire for any spec in this set, so no `_xorbit` sections were
  emitted (the twin is conditional by design, like `_xsched`).
- X1: the `_xsched` twin is now ISA-open (was AVX512-only); the
  planner's gates did not produce an eligible RVV xorsched candidate
  for these specs, so no `_rvv_xsched` sections were emitted.

Codegen changes (Lane 2) reflected in reselected winners:
- #31 roofline cost model changed heuristic winners (e.g. `raid6-5`
  sve2 now selects the I8 pipe twin `mulacc+nibble U=4+eor3+pipe`).
- A1 bit-level XOR scheduling, A4 closed-form recovery, and the I8
  pipe twin are visible in the disassemblies where selected.

Bug fix (2026-10-07, found by this regen):
- pipe+predtail crashed the AArch64 backend (the predicated tail
  reuses the non-pipelined body whose register layout is dead in pipe
  mode). The backend now rejects the combination cleanly, and [api.c](../src/api.c) /
  [gen_kernels](gen_kernels.c) skip the predtail twin for pipe winners (the pipe twin
  keeps its scalar tail). Hence `raid6-5_sve2` (pipe winner) has no
  `_predtail` section; non-pipe SVE winners (e.g. `raid5-4_sve2`) do.
- x86: `maskedtail` added to the pipe twin's exclusion list (same
  class; was only reachable via direct candidate construction).

## Deliberately not represented (driver-level compositions)

This document showcases JIT-emitted machine code. Three Lane-2
capabilities compose planner-selected kernels at the C driver level and
emit no distinct kernel of their own, so they have no sections here;
their sub-kernels' shapes are covered by the sections above:

- #8 fused kernels (`rj_fuse_compile`, [src/fused.c](../src/fused.c)): each data op
  (ENCODE / PARITY_UPDATE) is compiled through the public `rj_compile`
  planner -- the emitted bytes are exactly the sub-op kernels' bytes.
  The "fusion" (one entry point, one plan-time tile loop, CRC
  accumulation over the hot tile) is C code in `fused_driver_call`;
  no JIT codegen is involved. A fused section would duplicate the
  sub-op sections byte-for-byte.
- #10 cache-blocked striping (`rj_compile_tiled`, [src/striping.c](../src/striping.c)):
  wraps the winning `rj_fn` in a C tile loop (`rj_tile_select` is
  plan-time). The header contract is explicit: "Bit-exact vs the
  untiled rj_fn (same kernel, disjoint tiles)."
- #38 integrated CRC (`rj_compile_crc`, [src/api_crc.c](../src/api_crc.c)): the CRC-32 is
  scalar table-driven C ([src/crc.c](../src/crc.c), slice-by-8) accumulated in the
  driver; the JIT kernel is the standard encode winner. (`rj_compile_crc`
  is `rj_fuse_compile` with the sequence [ENCODE, CRC].)

## What changed vs the 2026-10-01 set (2026-10-05)

- `raid6-5_sve2`: now contains 16 EOR3 (was: zero SVE2 instructions);
  U=2→4, code 960→1280 B (const pool unchanged at 640 B).
- `raid5-4_sve2`: U=4→5, code 384→512 B (pressure-model changes since the
  AArch64 remat work; EOR3 n/a for P_XOR by design).
- `raid5-4_rvv`: U=4→5, code 384→448 B.
- `raid6-5_rvv`: U=4→5, code 1056→1232 B (const pool unchanged at 32 B).
- avx512-gfni: unchanged selection (same descs and sizes).
- New in this set: `.code.bin`/`.const.bin` shipped (previously
  /tmp-only), `manifest.json` with sha256 digests, fully decoded RVV
  disassembly, and per-kernel ISA-marker verification at build time.
