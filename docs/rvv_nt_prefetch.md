# RVV non-temporal stores and software prefetch

Implemented 2026-10-01. This document records the ISA survey verdict,
the chosen mechanisms, their exact semantics and contracts, and the
evidence that the emitted code is correct.

## Verdict

Implement **both**, with ratified extensions:

| Feature | Mechanism | Extension | Status |
|---|---|---|---|
| Non-temporal output stores | `NTL.ALL` hint before each `vse8.v` | Zihintntl v1.0 (ratified) | implemented |
| Software prefetch of inputs | `prefetch.r` per input stream per group iteration | Zicbop (ratified) | implemented, opt-in |

The pre-existing claim in [src/plan.c](../src/plan.c) that "RVV v1 has no non-temporal
store primitive" was wrong: the Zihintntl extension provides exactly
this. Both mechanisms are HINT encodings, so they degrade gracefully
(architectural NOPs) on implementations that ignore them.

## Non-temporal stores: Zihintntl NTL.ALL

### Spec (primary source)

*RISC-V Instruction Set Manual, Volume I: Unprivileged Architecture*,
"Zihintntl" Extension for Non-Temporal Locality Hints, Version 1.0:

> "The NTL instructions are HINTs that indicate that the explicit
> memory accesses of the immediately subsequent instruction (henceforth
> 'target instruction') exhibit poor temporal locality of reference.
> The NTL instructions do not change architectural state, nor do they
> alter the architecturally visible effects of the target instruction."

Four variants are provided; the relevant one:

> "The NTL.ALL instruction indicates that the target instruction does
> not exhibit temporal locality within the capacity of any level of
> cache in the memory hierarchy. **NTL.ALL is encoded as ADD x0, x0,
> x5.** ... Programmers should use NTL.ALL for accesses that have no
> exploitable temporal locality."

Streaming parity writes are exactly the "no exploitable temporal
locality" case, so the backend emits NTL.ALL (not NTL.P1/PALL/S1).

### Encoding

`ADD x0, x0, x5` = `0x00500033`. Verified against GNU as
(binutils 2.42, `-march=rv64g_zihintntl`):

```
$ echo 'ntl.all' | riscv64-linux-gnu-as -march=rv64g_zihintntl -o /tmp/x.o -
$ riscv64-linux-gnu-objdump -d /tmp/x.o
   0:   00500033    ntl.all
```

(binutils 2.42 disassembles the word back to the `ntl.all` alias.)

### Why no feature gating is needed

An `ADD x0, x0, x5` is a canonical HINT: on every RISC-V
implementation it executes as a NOP. Emitting it unconditionally for
NT candidates is therefore always safe, with or without Zihintntl
support — there is no CPUID-style check to get wrong, and no illegal
instruction risk. The planner enumerates RVV NT twins unconditionally
(when `nt_threshold > 0` and benchmarking is on).

### Fence contract: none needed (differs from x86/AArch64)

The spec sentence quoted above is the whole contract: the hint "does
not change architectural state, nor ... alter the architecturally
visible effects of the target instruction". It does not weaken the
memory ordering of the store the way x86 `MOVNTDQ` (weakly ordered,
needs `SFENCE`) or AArch64 `STNP`/`STNT1B` (needs `DMB ISHST`) do.
Consequences, all enforced by tests:

- The jitted code executes **no fence** (verified: [tests/test_nt.c](../tests/test_nt.c)
  scans NT twins for any `FENCE` word and fails if one appears).
- **Callers need no fence either** on RISC-V — unlike x86 (caller must
  `SFENCE`) and AArch64 (caller must `DMB ISHST`). This asymmetry is
  documented in `rj_options.nt_threshold` ([include/raidjit.h](../include/raidjit.h)) and
  `cand_t.nt` ([src/plan.h](../src/plan.h)).

| ISA | NT mechanism | jitted code fences | caller must fence |
|---|---|---|---|
| x86-64 | `MOVNTDQ`/`VMOVNTDQ` | no | **yes** (`SFENCE`) |
| AArch64 | `STNP`/`STNT1B` | no | **yes** (`DMB ISHST`) |
| RISC-V | `NTL.ALL` hint | no | **no** (hint, ordering unchanged) |

### Codegen

All RVV output stores funnel through `c_vse()` in [src/riscv/jit.c](../src/riscv/jit.c)
(the only caller of `em_vse8`), so the NT twin is a one-line choke
point: when `cand->nt`, emit `ntl.all` immediately before each
`vse8.v`. This covers the main loop and the vector tail (RVV has no
scalar tail — strip-mining handles every length). The hint is **not**
counted as a uop in the static model: on implementations that ignore
it, it executes as zero uops, and counting it would let the NT twin
lose the static-model tie-break on fiction. The twin ties its temporal
original on the static model and loses the deterministic tie-break;
with benchmarking on, measurement decides.

The spec discourages following an NTL hint with an instruction that
does not explicitly access memory; the code complies trivially — the
hint's immediately subsequent instruction is always the `vse8.v`
store it annotates (enforced by [tests/test_riscv_prefetch.c](../tests/test_riscv_prefetch.c)'s
adjacency check).

### Rejected alternatives

- **Zicbom `cbo.flush`/`cbo.clean`**: cache *maintenance* operations,
  not store hints; wrong tool (they force writeback/eviction with
  ordering implications, and need the Zicbom extension anyway).
- **Zicboz `cbo.zero`**: zeroes a cache block — a zeroing operation,
  not a way to store computed parity data.
- **Vendor extensions**: non-portable by definition; rejected.

## Software prefetch: Zicbop prefetch.r

### Spec (primary source)

Zicbop ("Cache-Block Prefetch Operations") is ratified and in the
RVA22 application-processor profile's mandatory set. The RVA profile
text states the design intent: "As with other HINTS, the inclusion of
prefetches in the mandatory set of extensions indicates that software
should generate these instructions where they are expected to be
useful, and hardware is expected to exploit that information."

`prefetch.r`/`prefetch.w`/`prefetch.i` are `ORI x0, rs1, imm` HINT
encodings: `imm[4:0]` selects the type (`.i`=0, `.r`=1, `.w`=3),
`imm[11:5]` is the byte offset in 32-byte units from `rs1`.
Forward encodable range in practice: **0..2016, 32-byte granularity**.
The ISA field is 7 bits (architecturally 0..4064 in 32-byte steps),
but GAS rejects offsets above 2016 ("improper prefetch offset"),
so the backend caps at the GAS-verified 2016; the JIT emits raw words
rather than going through GAS, so this is a conservative,
assembler-compatible ceiling rather than an ISA limit. The backend
accepts `prefetch_dist` in **32..2016, multiple of 32**; anything else
fails the candidate build cleanly with a "not encodable" error — never
silently rounded or relabeled (a rounded distance would benchmark a
lie).

GAS-verified words (binutils 2.42, `-march=rv64gcv_zicbop`):

| assembly | word |
|---|---|
| `prefetch.r 0(a0)` | `0x00156013` |
| `prefetch.r 32(t2)` | `0x0213e013` |
| `prefetch.r 2016(t2)` | `0x7e13e013` |
| `prefetch.r 2048(t2)` | **rejected by GAS** |
| `prefetch.r -32(t2)` | `0xfe13e013` (assembles, unused) |

Only `prefetch.r` is used: the prefetched streams are the *inputs*,
which are read. There is no `.w` for outputs (NTL.ALL already covers
the streaming-write aspect) and no `.i` (no instruction stream).

### Detection: rj_cpu_info.rvv_zicbop

No `riscv_hwprobe` key exposes Zicbop, so
[src/riscv/cpu.c](../src/riscv/cpu.c) parses the `isa:` line of `/proc/cpuinfo` for an
exact `zicbop` token. Caveats:

- Under QEMU user-mode the guest sees the **host's** (non-RISC-V)
  cpuinfo, so detection reports 0 even though `qemu-riscv64 -cpu max`
  executes the prefetch hints (verified: exit 42, no SIGILL, on both
  `-cpu max` and `-cpu rv64`). Tests therefore use a synthetic
  `rvv_zicbop=1` CPU.
- On real RISC-V Linux the kernel prints the full extension list on
  the `isa:` line; the parser matches the exact token `zicbop`
  (underscore/whitespace separated), so e.g. a hypothetical
  `zicbopx` would not false-positive.

### Planner and backend gating

- `plan_candidates` enumerates RVV prefetch twins **only** when
  `cpu->rvv_zicbop` is set. Without Zicbop the backend would ignore
  `prefetch_dist`, so the twin would be a byte-identical duplicate:
  it is skipped.
- `jit_build_version` validates the distance when Zicbop is present
  (clean `-1` + "not encodable" otherwise). A force-built candidate
  with `prefetch_dist > 0` but no Zicbop **degrades to temporal**:
  still correct, just no prefetch emitted.
- **Exact gate:** the version desc carries `+pf` iff at least one
  prefetch is actually emitted (`cg->emit_pf`), mirroring the SVE2
  EOR3 lesson that desc suffixes must reflect emitted bytes, not
  requested options.

### Codegen

`c_prefetch_in()` in [src/riscv/jit.c](../src/riscv/jit.c), called immediately after each
input-stream pointer computation (`em_add(e, T2, T2, T1)`) in the
P_XOR, P_MULACC, P_HORNER, and P_DELTA (old, new, and old-parity
seeding) paths: one `prefetch.r` per input stream per group iteration,
`prefetch_dist` bytes ahead — the same placement policy as x86
`PREFETCHT0`. Each prefetch counts as **1 uop** in the static model
(mirrors x86), so the prefetch twin loses the static tie-break and
only wins via the benchmark — per the standing policy that prefetch
stays opt-in and measurement decides (see [docs/limitations.md](limitations.md) §11).

## Verification evidence (2026-10-01)

- **Assembler:** GAS accepts `ntl.all` (`-march=rv64g_zihintntl`) →
  `0x00500033`; accepts `prefetch.r` at offsets 0/32/2016
  (`-march=rv64gcv_zicbop`), rejects 2048.
- **QEMU 8.2.2** (`qemu-riscv64`, `-cpu max` and `-cpu rv64`):
  executes both hint forms, no SIGILL.
- **Unit:** [tests/test_riscv_prefetch.c](../tests/test_riscv_prefetch.c) — planner twin enumeration
  (NT always, prefetch iff Zicbop); force-built NT twin (`+nt`,
  `#NTL.ALL == #vse8.v ≥ 1`, every store immediately preceded by the
  hint, temporal twin has none); force-built prefetch twin (`+pf`,
  every `prefetch.r` carries the requested offset); exact gate
  (no Zicbop → no `+pf`, no prefetch words); clean rejections
  (100, 4096, 8192, 16 fail; 32, 2016 build); output equality vs the
  scalar reference at 18 lengths for both twins.
  Passes under QEMU at VLEN 128/256/512.
- **Shape:** [tests/test_nt.c](../tests/test_nt.c) extended with `__riscv` branches:
  counts `0x00500033` words as NT markers and scans for any `FENCE`
  word (must be zero); RVV NT twins now run the full NT
  correctness+shape matrix (raid1, raid5(8), raid6(8), rs(6,3)
  encode+recover).
- **Mutation (red-green):** corrupting `em_ntl_all`'s word, shifting
  the prefetch offset, stubbing the prefetch gate off, and removing
  the planner Zicbop gate each produce test failures (in the new test
  and in `test_nt.c`); all restored and re-greened.
- **No regressions:** native `make test` green (464,279-run matrix,
  0 failures); full QEMU RVV suite at 3 VLENs; RVV remat untouched
  ([tests/test_riscv_remat.c](../tests/test_riscv_remat.c) re-run at all VLENs).

## Files

- [src/riscv/emit.h](../src/riscv/emit.h), [src/riscv/emit.c](../src/riscv/emit.c): `em_ntl_all()`,
  `em_prefetch_r()` (with encoding asserts).
- [src/riscv/jit.c](../src/riscv/jit.c): `c_vse()` NT hint, `c_prefetch_in()`, distance
  validation, `+nt`/`+pf` desc suffixes.
- [src/riscv/cpu.c](../src/riscv/cpu.c): `rvv_zicbop` detection via `/proc/cpuinfo`.
- [src/plan.h](../src/plan.h), [src/plan.c](../src/plan.c): twin enumeration, per-ISA fence
  contract on `cand_t.nt`.
- [include/raidjit.h](../include/raidjit.h): `rvv_zicbop`, `nt_threshold`/`prefetch_distance`
  RVV notes.
- [tests/test_riscv_prefetch.c](../tests/test_riscv_prefetch.c), [tests/test_nt.c](../tests/test_nt.c) (`__riscv`
  branches), [Makefile](../Makefile) (build + 3 QEMU VLEN lines).
