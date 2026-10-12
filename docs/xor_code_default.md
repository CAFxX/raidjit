# XOR-code default selection for `rj_topo_raid6()`

`rj_topo_raid6(ndata)` dispatches to the fastest **measured** code for that
`ndata` among the applicable codes (RDP / EVENODD / Liberation / RS).
This file is the machine-readable record of the winners; the C table in
[src/xorcode.c](../src/xorcode.c) (`xorcode_default_for_ndata()`) is transcribed from it,
and [tests/test_xorcode.c](../tests/test_xorcode.c) parses the block below and checks agreement,
so the two cannot drift.

**Status: MEASURED (2026-10-02, AMD EPYC 9D25, AVX-512+VBMI+BITALG+GFNI).
Two full sweeps + an interleaved A/B; see [docs/xor_codes.md](xor_codes.md) §4 for the
numbers. Result: RS is the fastest measured code at every ndata —
no XOR code shows a stable, repeatable win (sweep 1 suggested
liberation at ndata=3,7; sweep 2 and the interleaved A/B reversed
those; at 512 KiB RS wins everywhere by 1.3–2×).**

**Re-measurement on merged tree (2026-10-02, Integration 1):** Re-ran
the §4 protocol on the merged tree (A3 xorsched/CSE active) via
[tools/duel_xorcode.c](../tools/duel_xorcode.c) (5 interleaved rounds, ndata 3..13, 512 KiB and
8 MiB). **Result: still all-RS.** At 512 KiB (cache-fit) RS wins
everywhere by 1.2–1.8×, consistent across runs. At 8 MiB (streaming)
the measurements were not stable/repeatable on the shared 2-vCPU VM
(RS at ndata=5 varied 7.97–37.30 GB/s across runs under contention) —
no flip can be confirmed. The 8M leg occasionally favored XOR codes
(Liberation at 3,5,7,9,12; RDP at 10), suggesting A3's CSE may help
streaming, but this requires quiet-system re-measurement to confirm.
No code changes; defaults unchanged.

Benchmark protocol (code-choice): each candidate topology compiled with
the standard candidate benchmark (`benchmark=1`), winner verified against
the scalar reference, then timed at fixed *total data bytes* (512 KiB and
8 MiB) so codes with different stream counts compare fairly; 2 warmup +
11 timed iterations, median data-GB/s. Host: AMD EPYC 9D25 (AVX-512+VBMI+
BITALG+GFNI).

> **Integrator note (resolved 2026-10-02, Integration 1):** The re-run
> above was performed on the merged tree with A3's xorsched/CSE active.
> No flip; defaults unchanged. The 8M streaming anomaly is flagged for
> quiet-system follow-up but does not block the all-RS default.

## Machine-readable winners

<!-- BEGIN-MACHINE-TABLE
3 rs
4 rs
5 rs
6 rs
7 rs
8 rs
9 rs
10 rs
11 rs
12 rs
13 rs
14 rs
15 rs
16 rs
17 rs
18 rs
19 rs
20 rs
21 rs
22 rs
23 rs
24 rs
25 rs
26 rs
27 rs
28 rs
29 rs
30 rs
31 rs
32 rs
33 rs
34 rs
35 rs
36 rs
37 rs
38 rs
39 rs
40 rs
41 rs
42 rs
43 rs
44 rs
45 rs
46 rs
47 rs
48 rs
49 rs
50 rs
51 rs
END-MACHINE-TABLE -->

## Human-readable table

| ndata | applicable codes            | default (measured) | evidence |
|-------|-----------------------------|--------------------|----------|
| 3     | evenodd, liberation, rs     | rs (measured)      | §4: 25.3/27.8 vs 26.4/8.6 (lib), A/B 20.8 vs 19.4/19.5 |
| 4     | rdp, liberation, rs         | rs (measured)      | §4: 34.1/30.4 vs 25.6/29.5 (rdp), 23.5/28.9 (lib) |
| 5     | evenodd, liberation, rs     | rs (measured)      | §4: 39.7/31.0 vs 22.4/25.8 (eo), 23.4/33.9 (lib); A/B 27.2 vs 20.1/21.0 |
| 6     | rdp, liberation, rs         | rs (measured)      | §4: 38.6/27.9 vs 23.5/37.1 (rdp), 30.4/30.0 (lib); A/B 22.3 vs 19.2/20.8 |
| 7     | evenodd, liberation, rs     | rs (measured)      | §4: 30.7/38.1 vs 36.3/27.1 (eo), 36.3/32.4 (lib); A/B 26.3 vs 14.9/12.7 |
| 8     | liberation, rs              | rs (measured)      | §4: 38.9/31.1 vs 29.2/24.9 |
| 9     | liberation, rs              | rs (measured)      | §4: 44.92 (RS) vs 18.60 (liberation(9), w=11); one number per code recorded, sweep unattributed |
| 10    | rdp, liberation, rs         | rs (measured)      | §4: 35.3/34.5 vs 23.4/15.1 (rdp), 19.6/19.1 (lib) |
| 11    | evenodd, liberation, rs     | rs (measured)      | §4: 31.1/33.1 vs 3.3/18.8 (eo), 26.2/28.1 (lib) |
| 12    | rdp, liberation, rs         | rs (measured)      | §4: 35.4/33.9 vs 21.5/20.8 (rdp), 19.7/22.7 (lib) |
| 13    | evenodd, liberation, rs     | rs (measured)      | §4: 45.5/41.2 vs 20.2/20.9 (eo), 21.7/18.5 (lib) |
| 14+   | rs only (expanded models exceed RJ_MAX_DISKS=256) | rs | — |

Applicability recap: RDP needs `ndata+1` prime (`ndata` in {2,4,6,10,12}
within the disk cap); EVENODD needs `ndata` prime ({3,5,7,11,13});
Liberation needs `w*(ndata+2) <= 256` with `w` = smallest prime >=
`max(ndata,3)` (`ndata` <= 13). `rj_topo_raid6()` requires `3 <= ndata <=
51` (RS MDS envelope); outside it returns NULL as before.

Table constraint (enforced by [`tests/test_xorcode.c::test_default_table`](../tests/test_xorcode.c)):
a non-RS entry must name a code whose constructor accepts the
corresponding ndata — RDP needs `ndata+1` prime, EVENODD needs `ndata`
prime, Liberation needs `w*(ndata+2) <= 256`. As defense in depth,
`rj_topo_raid6()` falls back to RS rather than returning NULL if an
entry ever violates this, so the dispatch stays total on `3..51`.
