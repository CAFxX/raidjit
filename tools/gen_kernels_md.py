#!/usr/bin/env python3
"""gen_kernels_md.py <dir> <output.md>

Assemble raidjit_kernels.md from a kernels directory: manifest.json for the
per-kernel configuration tables plus the full disassembly of each kernel's
<code>.asm</code> listing in a fenced block.

<dir> is the finalize_kernels.sh stage dir (or the installed
goals/.../files/raidjit/kernels/ dir — same files).
"""
import json
import os
import sys

SRCDIR = sys.argv[1]
OUT = sys.argv[2]

with open(os.path.join(SRCDIR, "manifest.json")) as f:
    man = json.load(f)

kernels = man["kernels"]

lines = []
lines.append("# raidjit kernels — disassembled reference kernels")
lines.append("")
lines.append(f"Generated {man['generated_utc']} by {man['tool']} (`make kernels`). "
             "Each kernel was JIT-compiled with `benchmark=0` (heuristic candidate "
             "selection) for a synthetic CPU matching the target ISA — never the host "
             "CPUID, since the cross targets differ from the build host.")
lines.append("")
lines.append("## How to read")
lines.append("")
lines.append("Each section below is one kernel: the JIT-emitted machine code, "
             "disassembled. The shipped bytes (`.code.bin` / `.const.bin`) were "
             "executed against the scalar reference before disassembly, and their "
             "sha256 is recorded in each table — the bytes below are the tested bytes.")
lines.append("")
lines.append("Entry contract: call as `rj_fn(const void *in[], void *out[], size_t len)`. "
             "Every buffer must be 64-byte aligned; no two buffers may overlap or alias. "
             "Code and const-pool form one relocatable image: the const pool must "
             "immediately follow the code at offset `code_size` bytes (pool references "
             "are PC-relative in all backends). No fences are emitted; caller owns "
             "visibility per the ISA memory model.")
lines.append("")
lines.append("Selection method: " + man["selection_method"])
lines.append("")

for k in kernels:
    name = k["name"]
    lines.append(f"## {name}")
    lines.append("")
    lines.append(f"{k['desc']} — {k['topology']} {k['op']}, strategy `{k['strategy']}`, "
                 f"unroll {k['unroll']}.")
    lines.append("")
    lines.append("| field | value |")
    lines.append("|-------|-------|")
    lines.append(f"| target | `{k['target']}` |")
    lines.append(f"| topology | `{k['topology']}` |")
    lines.append(f"| operation | `{k['op']}` |")
    lines.append(f"| strategy | `{k['strategy']}` |")
    lines.append(f"| unroll | {k['unroll']} |")
    lines.append(f"| nominal vector length | {k['nominal_vl_bytes']} B |")
    lines.append(f"| code size | {k['code_size']} B |")
    lines.append(f"| const-pool size | {k['const_size']} B |")
    lines.append(f"| uops / byte | {k['uops_per_byte']} |")
    lines.append(f"| mem refs / byte | {k['mem_per_byte']} |")
    lines.append(f"| ISA marker | {k['marker']} (x{k['marker_count']}) |")
    lines.append(f"| code sha256 | `{k['code_sha256']}` |")
    lines.append(f"| const sha256 | `{k['const_sha256']}` |")
    lines.append("")
    asm_path = os.path.join(SRCDIR, k["asm_file"])
    with open(asm_path) as f:
        asm = f.read().rstrip("\n")
    assert "```" not in asm, f"fence-breaking content in {asm_path}"
    lines.append("```asm")
    lines.append(asm)
    lines.append("```")
    lines.append("")

with open(OUT, "w") as f:
    f.write("\n".join(lines))
print(f"gen_kernels_md.py: wrote {OUT} ({len(kernels)} kernels)")
