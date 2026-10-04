# raidjit native x86-64 benchmark

Date: 2026-10-01. Machine: AMD EPYC 9D25 (126-core cloud VM).
CPU features detected: sse2 ssse3 avx avx2 avx512f avx512bw avx512vl gfni.
Build: `gcc -O2 -std=c11` native (`make clean && make`, x86-64, no emulation).

## Key findings

- Fastest encode: **107.5 GB/s** (RS(6,3), 64 KiB/disk, `horner avx512+gfni U=4`); at 1 MiB/disk the fastest encode is **58.8 GB/s** (RAID-5(4+1), `xor avx512 U=1`) — the 1 MiB working set is memory-bandwidth bound on this VM.
- Fastest two-disk rebuild: **58.4 GB/s** (RAID-6(6+2), 64 KiB/disk, `mulacc avx512+gfni U=2`); at 1 MiB/disk **51.3 GB/s** (RS(10,4) Cauchy, `mulacc avx512+gfni U=1`).
- ISA selection: of 72 configs, 61 picked AVX-512 (36 with GFNI Galois multiply, 2 with nibble-shuffle, 23 plain AVX-512 mostly for pure-XOR ops), 11 picked AVX2, none picked SSE — the auto-benchmark consistently prefers the widest available ISA.
- Strategy selection matches theory: pure-XOR ops (RAID-1/5, single-disk decode) -> `xor`; Vandermonde parity encode -> `horner`; general-matrix (Cauchy) and multi-disk recovery -> `mulacc`.
- Single-data-disk recovery in RAID-6/RS compiles to pure XOR (only P is needed); the planner discovers this with no special-casing.

## Methodology

- Each (topology, mode) is compiled with default options (`benchmark=1`): the library benchmarks all legal codegen candidates (ISA x strategy x unroll — SSE/SSSE3, AVX2, AVX-512 with nibble-shuffle or GFNI Galois multiply, U=1..8) and auto-selects the fastest. The **picked variant** column shows the winner (`rj_info.desc`, abbreviated: `[rec <failed>] <strategy> <isa> U=<unroll>`).
- Throughput: steady-state repeated calls of the jitted function, 2 warmup + 11 timed iterations; **median** is the headline number, with mean and sample stddev. GB/s = total stripe bytes `((ndata+npar) * len)` per second.
- Stripe sizes: 64 KiB and 1 MiB **per disk**.
- `pick` = the compile-time benchmark score of the picked candidate (aggregate over working sets bench_bytes/4, bench_bytes, bench_bytes*4 with interleaved rounds); shown for reference, not directly comparable to the single-size steady-state measurement.
- Correctness: every config verified — encode output vs an independent scalar GF(2^8)/0x11B reference; recover output vs the original stripe. All 72 configs: **ok**.
- Caveat: shared cloud VM; turbo/frequency scaling and noisy neighbors cause run-to-run variance (see sd column). Near-tied candidates can flip the auto-selection between runs (e.g. U=1 vs U=4 for the same config).

## Modes

| Mode | Operation |
|---|---|
| encode | `rj_op_encode()` — compute parity from data |
| decode | `rj_op_recover({0})` — rebuild single data disk 0 |
| recon-data | `rj_op_recover({0,1})` — rebuild two data disks (n/a where npar<2) |
| recon-parity | `rj_op_recover({parities})` — rebuild all parity disks |

## RAID-1 (ndata=1, npar=1)

| mode | 64 KiB/disk: picked variant | med | mean ± sd | 1 MiB/disk: picked variant | med | mean ± sd |
|---|---|---|---|---|---|---|
| encode | xor avx512 U=4 | 96.2 | 96.3 ± 4.4 | xor avx512 U=4 | 53.6 | 51.4 ± 5.1 |
| decode | rec 0 xor avx512 U=1 | 70.8 | 71.8 ± 5.1 | rec 0 xor avx2 U=4 | 58.8 | 55.3 ± 7.4 |
| recon-parity | rec 100 xor avx2 U=2 | 85.6 | 86.7 ± 3.9 | rec 100 xor avx2 U=4 | 58.3 | 49.5 ± 18.1 |

## RAID-5(3+1) (ndata=3, npar=1)

| mode | 64 KiB/disk: picked variant | med | mean ± sd | 1 MiB/disk: picked variant | med | mean ± sd |
|---|---|---|---|---|---|---|
| encode | xor avx2 U=1 | 74.6 | 74.3 ± 1.2 | xor avx2 U=2 | 46.3 | 45.5 ± 16.8 |
| decode | rec 0 xor avx512 U=1 | 80.5 | 80.7 ± 2.7 | rec 0 xor avx512 U=1 | 44.2 | 37.7 ± 13.8 |
| recon-parity | rec 100 xor avx512 U=1 | 85.6 | 85.3 ± 1.9 | rec 100 xor avx512 U=2 | 58.4 | 52.6 ± 16.9 |

## RAID-5(4+1) (ndata=4, npar=1)

| mode | 64 KiB/disk: picked variant | med | mean ± sd | 1 MiB/disk: picked variant | med | mean ± sd |
|---|---|---|---|---|---|---|
| encode | xor avx2 U=2 | 89.4 | 88.9 ± 1.6 | xor avx512 U=1 | 58.8 | 46.8 ± 21.5 |
| decode | rec 0 xor avx512 U=4 | 111.3 | 110.9 ± 1.7 | rec 0 xor avx512 U=1 | 58.2 | 52.6 ± 16.7 |
| recon-parity | rec 100 xor avx512 U=1 | 75.4 | 74.9 ± 2.3 | rec 100 xor avx512 U=1 | 58.7 | 53.1 ± 16.2 |

## RAID-5(8+1) (ndata=8, npar=1)

| mode | 64 KiB/disk: picked variant | med | mean ± sd | 1 MiB/disk: picked variant | med | mean ± sd |
|---|---|---|---|---|---|---|
| encode | xor avx512 U=1 | 86.7 | 86.4 ± 2.6 | xor avx2 U=2 | 40.2 | 33.4 ± 14.6 |
| decode | rec 0 xor avx512 U=4 | 110.9 | 106.8 ± 13.2 | rec 0 xor avx512 U=4 | 43.0 | 40.3 ± 12.1 |
| recon-parity | rec 100 xor avx512 U=1 | 92.0 | 92.2 ± 1.2 | rec 100 xor avx2 U=2 | 41.1 | 38.3 ± 11.6 |

## RAID-6(4+2) (ndata=4, npar=2)

| mode | 64 KiB/disk: picked variant | med | mean ± sd | 1 MiB/disk: picked variant | med | mean ± sd |
|---|---|---|---|---|---|---|
| encode | horner avx512+gfni U=4 | 101.7 | 99.0 ± 7.2 | horner avx512+gfni U=1 | 45.8 | 40.6 ± 13.8 |
| decode | rec 0 xor avx512 U=8 | 106.4 | 114.4 ± 12.5 | rec 0 xor avx512 U=1 | 53.5 | 47.9 ± 15.1 |
| recon-data | rec 0+1 mulacc avx512+gfni U=1 | 44.9 | 44.7 ± 1.0 | rec 0+1 mulacc avx512+gfni U=1 | 38.5 | 35.5 ± 10.9 |
| recon-parity | rec 100+101 horner avx512+nibble U=4 | 88.8 | 88.3 ± 1.2 | rec 100+101 horner avx512+gfni U=2 | 45.7 | 39.9 ± 13.1 |

## RAID-6(6+2) (ndata=6, npar=2)

| mode | 64 KiB/disk: picked variant | med | mean ± sd | 1 MiB/disk: picked variant | med | mean ± sd |
|---|---|---|---|---|---|---|
| encode | horner avx512+gfni U=8 | 90.1 | 95.7 ± 12.4 | horner avx512+gfni U=1 | 55.3 | 51.5 ± 8.6 |
| decode | rec 0 xor avx512 U=1 | 106.2 | 105.1 ± 2.2 | rec 0 xor avx2 U=1 | 67.2 | 64.5 ± 4.8 |
| recon-data | rec 0+1 mulacc avx512+gfni U=2 | 58.4 | 56.8 ± 4.5 | rec 0+1 mulacc avx512+gfni U=2 | 39.0 | 32.8 ± 14.0 |
| recon-parity | rec 100+101 horner avx512+gfni U=4 | 104.5 | 104.4 ± 2.0 | rec 100+101 horner avx512+gfni U=8 | 42.9 | 35.5 ± 16.1 |

## RAID-6(8+2) (ndata=8, npar=2)

| mode | 64 KiB/disk: picked variant | med | mean ± sd | 1 MiB/disk: picked variant | med | mean ± sd |
|---|---|---|---|---|---|---|
| encode | horner avx512+gfni U=1 | 91.5 | 85.2 ± 17.9 | horner avx512+nibble U=1 | 34.4 | 29.2 ± 11.7 |
| decode | rec 0 xor avx512 U=2 | 114.8 | 112.0 ± 8.7 | rec 0 xor avx512 U=2 | 49.9 | 46.2 ± 10.4 |
| recon-data | rec 0+1 mulacc avx512+gfni U=1 | 47.9 | 47.8 ± 0.3 | rec 0+1 mulacc avx512+gfni U=1 | 33.1 | 27.4 ± 12.9 |
| recon-parity | rec 100+101 horner avx512+gfni U=4 | 91.9 | 86.4 ± 17.0 | rec 100+101 horner avx512+gfni U=1 | 57.0 | 54.3 ± 5.6 |

## RAID-6(12+2) (ndata=12, npar=2)

| mode | 64 KiB/disk: picked variant | med | mean ± sd | 1 MiB/disk: picked variant | med | mean ± sd |
|---|---|---|---|---|---|---|
| encode | horner avx512+gfni U=1 | 63.7 | 61.5 ± 7.6 | horner avx512+gfni U=2 | 37.1 | 30.6 ± 12.4 |
| decode | rec 0 xor avx2 U=1 | 69.2 | 69.8 ± 1.2 | rec 0 xor avx2 U=2 | 41.1 | 35.5 ± 14.6 |
| recon-data | rec 0+1 mulacc avx512+gfni U=1 | 45.2 | 44.1 ± 3.5 | rec 0+1 mulacc avx512+gfni U=1 | 29.1 | 22.2 ± 11.8 |
| recon-parity | rec 100+101 horner avx512+gfni U=1 | 64.1 | 63.8 ± 1.0 | rec 100+101 horner avx512+gfni U=1 | 37.0 | 26.2 ± 16.6 |

## RS(6,3) (ndata=6, npar=3)

| mode | 64 KiB/disk: picked variant | med | mean ± sd | 1 MiB/disk: picked variant | med | mean ± sd |
|---|---|---|---|---|---|---|
| encode | horner avx512+gfni U=4 | 107.5 | 106.6 ± 2.7 | horner avx512+gfni U=4 | 45.7 | 41.0 ± 12.5 |
| decode | rec 0 xor avx512 U=1 | 120.0 | 119.0 ± 2.6 | rec 0 xor avx512 U=1 | 56.4 | 50.7 ± 14.6 |
| recon-data | rec 0+1 mulacc avx512+gfni U=1 | 57.0 | 56.9 ± 0.6 | rec 0+1 mulacc avx512+gfni U=1 | 42.4 | 35.4 ± 15.1 |
| recon-parity | rec 100+101+102 horner avx512+gfni U=1 | 95.6 | 95.4 ± 1.1 | rec 100+101+102 horner avx512+gfni U=1 | 43.4 | 36.5 ± 15.2 |

## RS(10,4)C (ndata=10, npar=4)

| mode | 64 KiB/disk: picked variant | med | mean ± sd | 1 MiB/disk: picked variant | med | mean ± sd |
|---|---|---|---|---|---|---|
| encode | mulacc avx512+gfni U=1 | 35.4 | 35.3 ± 0.6 | mulacc avx512+gfni U=1 | 26.0 | 26.3 ± 2.5 |
| decode | rec 0 mulacc avx512+gfni U=2 | 96.6 | 95.2 ± 4.7 | rec 0 mulacc avx512+gfni U=1 | 45.2 | 44.3 ± 3.5 |
| recon-data | rec 0+1 mulacc avx512+gfni U=1 | 54.2 | 54.2 ± 0.7 | rec 0+1 mulacc avx512+gfni U=1 | 51.3 | 50.9 ± 1.7 |
| recon-parity | rec 100+101+102+103 mulacc avx512+gfni U=1 | 34.5 | 32.8 ± 4.1 | rec 100+101+102+103 mulacc avx512+gfni U=1 | 27.5 | 27.4 ± 0.4 |

## Notes

- **RS(10,4)**: `rj_topo_rs(10,4)` returns NULL — the default Vandermonde matrix is only guaranteed MDS for `npar<=3, ndata<=51` (g=2 has order 51 in GF(2^8)/0x11B). It is benchmarked here as `RS(10,4)C` via `rj_topo_custom()` with a Cauchy matrix M[j][i]=1/(x_i^y_j), x=0..9, y=10..13. The matrix is not in Horner order, so the planner selects `mulacc` instead of `horner`.
- **decode of one data disk** (RAID-6/RS) is compiled to pure **xor**: with P available, a single missing data disk is recovered without touching Q (or the RS equivalents) — the planner discovers this automatically.
- **recon-parity** is encode-shaped (parities rebuilt from data); the planner independently re-selects horner/nibble/GFNI per topology.
- 1 MiB/disk rows are generally slower than 64 KiB/disk: at 1 MiB x (ndata+npar) the working set exceeds cache and the op becomes memory-bandwidth bound on this VM, while 64 KiB stays cache-resident.
- Median GB/s is the robust headline figure; mean/sd expose the VM noise. Configs with sd > ~30% of median should be treated as approximate.
