#!/usr/bin/env python3
"""disasm_diff.py — S9.4: Capstone-vs-objdump differential (SPEC §S9.4).

For each (ISA, blob) in the parity corpus work dir:
  1. disassemble with the wasm disasm module (node)
  2. disassemble with objdump (native or cross)
  3. normalize both with tools/tidy_disasm.py logic
  4. require: same instruction COUNT and same mnemonic sequence
     (byte equality is already proven by the parity matrix; this checks
     the disassembler doesn't drop/misalign instructions).

Usage: ./disasm_diff.py [workdir]
"""
import json
import os
import re
import subprocess
import sys

WORK = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "work")
WASM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DISASM_JS = os.path.join(WASM_DIR, "out", "disasm.js")

ARCH = {
    # -D (not -d): blobs are raw binaries with no sections to select
    "x86": ("x86_64", "objdump", ["-D", "-M", "att"]),
    "aarch64": ("aarch64", "aarch64-linux-gnu-objdump", ["-D"]),
    "riscv": ("riscv64", "riscv64-linux-gnu-objdump", ["-D"]),
}


def capstone_disasm(arch, blob):
    # trailing zero bytes are alignment padding, not code: strip them so
    # a decoder that stops at the padding (objdump "...") and one that
    # decodes zeros as udf (capstone) agree.
    raw = open(blob, "rb").read().rstrip(b"\x00")
    js = f"""
(async () => {{
  const mod = await require('{DISASM_JS}')();
  const bytes = Buffer.from('{raw.hex()}', 'hex');
  const p = mod._malloc(bytes.length || 1);
  if (bytes.length) mod.HEAPU8.set(bytes, p);
  const ap = mod.stringToNewUTF8('{arch}');
  const rp = mod._dj_disasm(ap, p, bytes.length, 0n);
  const arr = JSON.parse(mod.UTF8ToString(rp));
  mod._dj_free_string(rp); mod._free(p); mod._free(ap);
  for (const e of arr) console.log(e.mnemonic + ' ' + e.bytes);
}})();
"""
    r = subprocess.run(["node", "-e", js], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"capstone failed: {r.stderr[:200]}")
    out = []
    for l in r.stdout.splitlines():
        if not l.strip():
            continue
        mne, b = l.split(" ", 1)
        out.append((mne, b))
    return out


def objdump_disasm(arch, blob):
    import shutil
    import tempfile
    _, prog, flags = ARCH[arch]
    if shutil.which(prog) is None:
        raise RuntimeError(f"{prog} not installed")
    # strip trailing zero padding (see capstone_disasm)
    raw = open(blob, "rb").read().rstrip(b"\x00")
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf:
        tf.write(raw)
        tpath = tf.name
    r = subprocess.run([prog] + flags + ["-b", "binary", "-m", {
        "x86": "i386:x86-64", "aarch64": "aarch64",
        "riscv": "riscv:rv64"}[arch], tpath],
        capture_output=True, text=True)
    os.unlink(tpath)
    if r.returncode != 0:
        raise RuntimeError(f"{prog} failed: {r.stderr[:200]}")
    mnems = []
    # objdump prints legacy prefixes (data16/data32/addr32/lock/rep/...)
    # as separate leading pseudo-mnemonics, sometimes stacked
    # ("data16 data16 ... nopw"); capstone folds them into the
    # instruction. Drop all leading prefix tokens, take the real one.
    prefixes = {"data16", "data32", "addr16", "addr32", "lock", "rep",
                "repe", "repne", "rex64"}
    for line in r.stdout.splitlines():
        # bytes are space-separated pairs (x86) or one contiguous word
        # (aarch64/riscv 32-bit insns); the disassembly follows.
        m = re.match(r"^\s+[0-9a-f]+:\s+([0-9a-f]+(?: [0-9a-f]+)*)\s+(.*)$",
                     line)
        if not m:
            continue
        bytehex = m.group(1).replace(" ", "")
        words = re.findall(r"[a-z][a-z0-9.]*", m.group(2))
        words = [w for w in words if w not in prefixes]
        if words:
            mnems.append((words[0], bytehex))
    return mnems


def norm(m):
    # strip x86 AT&T size suffixes for comparison. Only when len > 3:
    # unsuffixed mnemonics like "sub" must not become "su", while
    # "subq" -> "sub". (objdump sometimes omits the suffix where
    # capstone prints it.)
    return re.sub(r"[bwlq]$", "", m) if len(m) > 3 else m


def cap_norm(mnemonic, bytehex):
    return norm_idiom(mnemonic, bytehex)


def norm_idiom(mnemonic, bytehex):
    # 66 90 is xchg %ax,%ax, the 2-byte NOP idiom: objdump prints the
    # raw "xchg", capstone folds it to "nop". Same bytes, same meaning.
    if bytehex == "6690" and norm(mnemonic) == "xchg":
        return "nop"
    # AArch64 condition-code aliases: hs==cs, lo==cc. Capstone prefers
    # hs/lo, objdump prefers cs/cc (objdump even annotates
    # "b.cc // b.lo, b.ul, b.last").
    m = norm(mnemonic)
    return {"b.hs": "b.cs", "bc.hs": "bc.cs",
            "b.lo": "b.cc", "bc.lo": "bc.cc"}.get(m, m)


def main():
    if not os.path.exists(DISASM_JS):
        print(f"SKIP: {DISASM_JS} not built")
        return 0
    # map work blobs back to their ISA via the cases TSV
    cases = {}
    tsv = os.path.join(WORK, "cases.tsv")
    if os.path.exists(tsv):
        for line in open(tsv):
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                cases[parts[0]] = parts[1]
    # find code blobs
    import glob
    blobs = sorted(glob.glob(os.path.join(WORK, "case_*.code.bin")))
    if not blobs:
        print("no blobs found; run run_parity.sh first")
        return 1
    # we need the ISA per blob: use the matrix
    matrix = json.load(open(os.path.join(WORK, "..", "parity_matrix.json")))
    isa_by_idx = {}
    # cases.tsv lines are in matrix order (skipping filtered archs); rebuild
    # the mapping from the blob prefix number is fragile — instead use the
    # per-case prefix files written by run_parity.sh (case_N.opts.json order
    # == matrix order for the selected arch).
    n_ok = n_fail = 0
    for blob in blobs:
        m = re.search(r"case_(\d+)\.code\.bin$", blob)
        if not m:
            continue
        idx = int(m.group(1)) - 1
        # find the matrix entry by order among run cases: approximate by
        # reading the .opts.json next to it
        opts_p = blob.replace(".code.bin", ".opts.json")
        if not os.path.exists(opts_p):
            continue
        # ISA from the wasm module used: infer from which run produced it.
        # Simpler: try each arch's module by matching non-empty disassembly.
        # For now, use the matrix order directly.
        if idx >= len(matrix):
            continue
        mod = matrix[idx]["module"]
        arch = {"x86": "x86_64", "aarch64": "aarch64",
                "riscv": "riscv64"}[mod]
        try:
            cap = capstone_disasm(arch, blob)
        except RuntimeError as e:
            print(f"FAIL {blob}: {e}")
            n_fail += 1
            continue
        try:
            obj = objdump_disasm(mod, blob)
        except RuntimeError as e:
            print(f"SKIP {blob}: {e}")
            continue
        cap_n = [cap_norm(m, b) for m, b in cap]
        obj_n = [norm_idiom(m, b) for m, b in obj]
        if cap_n == obj_n:
            n_ok += 1
        else:
            n_fail += 1
            print(f"FAIL {os.path.basename(blob)} ({arch}): "
                  f"capstone {len(cap_n)} vs objdump {len(obj_n)} insns")
            for i, (a, b) in enumerate(zip(cap_n, obj_n)):
                if a != b:
                    print(f"  first diff at {i}: capstone={a} objdump={b}")
                    break
    print(f"disasm_diff: {n_ok} match, {n_fail} differ")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
