# Integrated CRC (#38)

One call encodes the stripe AND accumulates CRC-32/ISO-HDLC over the
input data buffers and the parity output buffers
(`include/raidjit_fused.h`: `rj_compile_crc` / `rj_fn_crc`). Follow-ups
build on this foundation: #8 generalizes the driver to a fusion
planner over op sequences (encode / parity update / CRC); #10 exposes
the plan-time tiling as a first-class API for the plain path.

## 1. Design decisions

**Fusion at the driver level, not the emitter level.** Fusing CRC
accumulation into the JITed instruction stream would require touching
every backend emitter (x86-64, AArch64, RISC-V) and every plan kind --
a large, risky change for an uncertain win. The driver instead tiles
the stripe at a plan-time tile size: per tile it runs the encode
kernel once, then accumulates CRC-32 over the tile's input and freshly
written parity buffers while they are hot in cache. Every byte is read
from DRAM once; the CRC pass is cache hits. This is the same structure
production multi-buffer libraries use (e.g. ISA-L's CRC + RAID paths).

**The CRC variant rides the real planner.** `rj_compile_crc` compiles
the encode op through public `rj_compile` -- full candidate
enumeration, static pruning, benchmarking -- then wraps the winning
`rj_fn`. "Planner-selectable" means exactly this: the CRC pipeline
inherits the planner's kernel choice, including NT/prefetch twins and
the length dispatcher.

**Zero runtime cost for the tiling decision (Carlo's invariant).**
`rj_tile_select(nstreams)` (src/striping.h) is a pure function of the
stream count, evaluated once per compile and stored in the handle. The
driver loop only reads the stored tile size.

**Tile heuristic (static, est. -- NOT measured).** Assumes a modest
16 MiB LLC, budgets half for one tile's working set across all live
streams: `tile = largest pow2 <= (LLC/2)/nstreams`, clamped to
[4 KiB, 1 MiB]. Powers of two >= 4 KiB keep every tile offset 64-byte
aligned, preserving the `rj_fn` alignment contract.

**CRC-32/ISO-HDLC.** Polynomial 0x04C11DB7 reflected, init/xorout
0xFFFFFFFF. `src/crc.c` has two implementations sharing one
runtime-generated table: byte-at-a-time scalar (differential oracle)
and slice-by-8 (production path). Tables are generated from the
polynomial at first use (no magic constants); `pthread_once` guards
generation.

## 2. Correctness argument

- The encode kernel is the validated `rj_compile` winner (validated
  against the scalar GF reference before benchmarking, same bar as
  `rj_compile`).
- Tiles are disjoint and cover `[0, len)` exactly; the CRC is a linear
  fold over the concatenation, so tiled accumulation equals
  whole-buffer CRC.
- The CRC paths are proven against two independent oracles: the
  catalog check value `0xCBF43926` and a bitwise implementation
  written from the polynomial definition in the test file.
- Fused outputs are asserted bit-exact vs the unfused `rj_fn` calls
  (tests/test_crc.c).

## 3. Non-goals / limits

- No emitter-level fusion (see §1). If a future emitter fuses CRC
  into the instruction stream, it composes: the driver would just
  stop running the separate accumulation pass.
- Benchmarks postponed: the tile size is heuristic. The compile-time
  benchmark still arbitrates kernel selection.
- NT-store winners: the caller's once-per-call fence contract carries
  over unchanged (documented in `raidjit_fused.h`).
- Trampoline slots: 32 for the `rj_fn_crc` entry point, same lifetime
  rules as `rj_get_fn_batched` (documented in the header).
