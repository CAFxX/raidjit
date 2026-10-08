# raidjit WASM live kernel explorer

Compiles raidjit's planner + emitters to WebAssembly so the kernels page
compiles **any** option combination live in the browser. Full spec:
[SPEC.md](SPEC.md).

## What is built

Three per-ISA compile modules (mirroring the native per-ARCH build) plus
one Capstone disassembler module, all `-sSINGLE_FILE=1` (self-contained,
`file://`-safe, no threads):

| Module | File | Contents |
|---|---|---|
| x86-64 | `out/raidjit_x86.js` | planner + x86-64 emitter, `rjw_compile` API |
| AArch64 | `out/raidjit_aarch64.js` | planner + AArch64 emitter, `rjw_compile` API |
| RISC-V | `out/raidjit_riscv.js` | planner + RISC-V emitter, `rjw_compile` API |
| disasm | `out/disasm.js` | Capstone 6.0.0-Alpha11+ (X86+AArch64+RISCV), `dj_disasm` API |

Build: `make -f tools/wasm/Makefile.wasm` (needs emsdk on PATH;
`source ~/tools/emsdk/emsdk_env.sh`). Capstone source goes in
`tools/wasm/vendor/capstone` (6.0.0-Alpha11+ unpacked; only the X86,
AArch64 and RISCV backends are compiled).

## `rjw_compile` JSON API (schema `"v":1`)

Input option document:

```json
{
  "v": 1,
  "target": "avx512-gfni",
  "cpu": { "vbmi": 1 },
  "topology": { "ctor": "raid5", "p1": 4 },
  "op": { "kind": "encode" },
  "cand": "auto",
  "options": { "loop_align": 0, "emit_endbr": false, "emit_shadow_stack": false }
}
```

- `target`: synthetic-CPU preset. Per module:
  - x86-64: `avx512-gfni`, `avx2`, `sse`
  - AArch64: `sve2`, `neon`
  - RISC-V: `rvv`, `rvv-zvbc32e`, `rvv-zicbop`
- `cpu` (optional): expert flag overrides — only codegen-gating flags
  (`gfni`, `vbmi`, `avx512bw` / `sve2`, `sve_vl` / `rvv_zvbc32e`,
  `rvv_zicbop`, `rvv_vlen`); anything else is rejected.
- `topology`: `ctor` + params. `p1`/`p2`/`p3` are positional:
  - `raid0{ndata}`, `raid1{}`, `raid5{ndata}`, `raid6{ndata}`,
    `raid6_rs{ndata}`, `rdp{p}`, `evenodd{p}`, `liberation{k}`,
    `rs{ndata,npar}`, `lrc{k,h,r}`, `lrc_azure{}`,
    `lrc_optimal{k,h,r}`,
    `custom{ndata,npar,matrix_hex}` (npar×ndata bytes, row-major, hex).
- `op`: `{"kind":"encode"}` |
  `{"kind":"recover","failed":[...],"rows":[...]}` (rows optional:
  explicit A5 equation-row subset; default = all parity rows) |
  `{"kind":"parity_update","changed":[...]}`.
- `cand`: `"auto"` (planner enumerates, static-model `heuristic_select`
  winner — the same rule the native build uses at `benchmark=0`), or a
  pinned object:
  `{"isa":"avx512","mul":"gfni","U":8,"lmul":1,"nt":false,`
  `"prefetch_dist":0,"ternlog":false,"remat":false,"notail":false,`
  `"predtail":false,"maskedtail":false,"permi2b":false,`
  `"xorsched":false,"xorbit":false,"mshift":false,"pipe":false}`.
  `U:0` = auto-U (every U 1..8 that passes the gates is built, heuristic
  winner picked). Omitted fields take the planner-consistent defaults.
  Invalid combinations fail closed (`{ok:false, err}`) via
  `planid_cand_valid()` + CPU-feature gates — never a backend assert.
- `options`: `loop_align` (0=auto), `emit_endbr`, `emit_shadow_stack`.
  Benchmark-related options do not exist here: the module is
  compile-only, always.

Result (success):

```json
{
  "ok": true, "desc": "raid5-4 xor avx512 U=8",
  "plan_id": "rj2:raid5(4)#...:enc:avx512:xor:nibble:u8:-:scalar",
  "target": "avx512-gfni", "topology": "raid5(4)", "operation": "encode",
  "strategy": "xor", "isa": "avx512", "mul": "nibble",
  "unroll": 8, "lmul": 1, "twins": [],
  "code_size": 512, "const_size": 0,
  "uops_per_byte": 0.078125, "mem_per_byte": 5.0,
  "selection": "heuristic", "exec_validated": false,
  "nominal_vl_bytes": 64
}
```

`exec_validated` is always false: the module never executes generated
code (it cannot — no executable memory in the browser). Correctness is
argued by byte-identity with the native build (see `parity/`), which CI
executes. `uops_per_byte`/`mem_per_byte` are static-model estimates,
not measurements.

Failure: `{"ok":false,"err":"..."}`. The code/const blobs are read via
`rjw_code_bytes(&len)` / `rjw_const_bytes(&len)` after a successful call
(valid until the next `rjw_compile`); `rjw_free_string()` releases result
strings.

In JS (MODULARIZE factory):

```js
const mod = await RaidjitX86();
const p = mod.stringToNewUTF8(JSON.stringify(opts));
const res = JSON.parse(mod.UTF8ToString(mod._rjw_compile(p)));
const lenp = mod._malloc(8);
const cp = mod._rjw_code_bytes(lenp);
const code = mod.HEAPU8.slice(cp, cp + mod.getValue(lenp, 'i32'));
```

## `dj_disasm` API

`dj_disasm(arch, bytes_ptr, len, base_addr)` → JSON array of
`{addr,size,bytes,mnemonic,op_str}`. `arch` is `x86_64` (AT&T syntax),
`aarch64`, or `riscv64` (RVV 1.0 + Zvbc enabled). Undecodable bytes are
emitted as `.byte` entries (never silently dropped).

## Parity harness (`parity/`)

`parity_driver.c` links `rjw_api.c` **natively** per ARCH (without
`-DRJ_WASM`): the same JSON in → native bytes out. `run_parity.sh`
builds the three native drivers, runs `parity_matrix.json` through them
and through the wasm modules under node, and requires byte-identical
code/const plus identical metadata. This is the master correctness
gate (SPEC §S9.2).

## Page

`../gen_live_kernels_page.py` inlines the four `out/*.js` bundles into
`raidjit_kernels.html` (single + optional A/B comparison mode,
deep-linkable via `#a=<base64url(JSON)>[&b=...]`), mirrored to
`~/workspace/your_files/raidjit/`.

## Constraints (non-negotiable)

- No benchmarks/perf guard in this track.
- The wasm build must not change native codegen: only `#ifdef RJ_WASM`
  blocks in `src/*/jit.c`; proven by byte-identical `make kernels`
  output before/after.
- No push. Interleaved-CRC work stays cancelled (`crc_fold` hidden).
