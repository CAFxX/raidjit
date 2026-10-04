# Kernel Dumps: RAID-5(4+1) and RAID-6(5+2)

Regenerated 2026-10-01 with `make kernels` (see `tools/gen_kernels.c` in the
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
(`src/ref.c`) at 6 lengths (0, 1, 7, 1003, 65537, 1 MiB) × 3 seeds = 18
runs, on both the live JIT image and a fresh executable mapping of the
shipped `.code.bin`/`.const.bin` bytes (proving the shipped bytes are the
tested bytes). Buffers 64-byte aligned per the API contract. A red-green
proof (const-pool byte flip + corrupted-expectation, in `/tmp`, never
shipped) confirmed the checks are not vacuous: both corruptions are
detected. (The flip test needs a const pool, so it is skipped for the
three RAID-5 kernels, which have `const_size == 0`; the corrupted-
expectation check runs for all six.)

An ISA marker assertion runs at generation time and fails the build if
absent (counts below are occurrences in the emitted code bytes).

## RAID-5 (4 data + 1 parity) — XOR

| ISA | Desc | Code | Const | Marker |
|-----|------|------|-------|--------|
| avx512-gfni | raid5-4 xor avx512 U=8 | 512 B | 0 B | VPXORD (EVEX.66.0F.W0 EF) ×24 |
| rvv | raid5-4 xor rvv U=5 | 448 B | 0 B | RVV vector opcodes ×21 |
| sve2 | sve xor U=5 | 512 B | 0 B | SVE codegen path ×16 (EOR3 n/a, see below) |

RAID-5 parity is pure XOR (no GF multiplication), so the avx512 kernel uses
`VPXORD` (not GFNI); "gfni" in the filename names the target CPU.

## RAID-6 (5 data + 2 parity) — Horner

| ISA | Desc | Code | Const | Marker |
|-----|------|------|-------|--------|
| avx512-gfni | raid6-5 horner avx512+gfni U=8 | 1600 B | 576 B | VGF2P8AFFINEQB (EVEX.66.0F3A.W1 CE) ×32 |
| rvv | raid6-5 horner rvv+nibble U=5 | 1232 B | 32 B | RVV vector opcodes ×184 |
| sve2 | sve horner+nibble U=4+eor3 | 1280 B | 640 B | EOR3 (SVE2) ×16 |

RAID-6 Q parity uses Horner accumulation with GF(2⁸) multiplies:
- avx512+gfni: `VGF2P8AFFINEQB` (single-instruction GF multiply)
- rvv: nibble-shuffle via `vrgather` (32-byte table)
- sve2: nibble-shuffle via `TBL` with the lo^hi^input reduction folded
  into one SVE2 `EOR3` per multiply (640-byte table)

## On the "sve2" label

The previous kernel set (2026-10-01 12:28) was built before the SVE2
audit; its "sve2" kernels contained zero SVE2 instructions. That is fixed
here: `raid6-5_sve2` contains 16 `EOR3` instructions (the `+eor3` desc tag
is exact — it is set iff ≥1 EOR3 is emitted; see `docs/sve2_audit.md`).

`raid5-4_sve2` cannot contain EOR3 by construction: P_XOR is pure XOR with
no GF multiply for EOR3 to fold (`docs/sve2_audit.md`). Its marker is the
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
`tools/finalize_kernels.sh` via `make kernels`; the README source lives at
`tools/kernels-README.md` in the repo.

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

## What changed vs the previous set

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
