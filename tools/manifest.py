#!/usr/bin/env python3
"""Merge per-kernel manifest fragments from tools/gen_kernels.c into one
manifest.json, adding sha256 digests of the shipped binaries and the
top-level selection/contract metadata.

Usage: manifest.py <stagedir> [date_utc]   (writes <stagedir>/manifest.json)

Validates: every fragment in the stage dir is well-formed, every referenced
file exists, on-disk sizes match the fragment's code_size/const_size. The
optional date_utc lets the caller (tools/finalize_kernels.sh) stamp the same
timestamp used in the .asm headers; without it, UTC now is used.
"""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

# The fragment set is the source of truth: gen_kernels.c dies loudly on any
# emission or twin-build failure, so a successful run guarantees all intended
# kernels are present. The manifest validates internal consistency (names
# match, files exist, sizes match).

SELECTION_METHOD = (
    "heuristic_select (benchmark=0): highest ISA wins; among equal ISA, "
    "min roofline max(uops/3.5, mem/64); ties keep the earliest "
    "candidate. Cross-ISA empirical benchmarking under QEMU is "
    "cycle-meaningless, so no timing is used. All candidates are built "
    "and ranked — this is heuristic selection, not first-match."
)

ENTRY_CONTRACT = (
    "Entry at code base; call as rj_fn(const void *in[], void *out[], "
    "size_t len). Every buffer must be 64-byte aligned; no two buffers "
    "may overlap or alias. Code and const-pool form one relocatable "
    "image: the const pool must immediately follow the code at offset "
    "code_size bytes (pool references are PC-relative in all backends). "
    "No fences are emitted; caller owns visibility per the ISA memory "
    "model."
)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: manifest.py <stagedir> [date_utc]")
    stagedir = sys.argv[1]
    date_utc = (sys.argv[2] if len(sys.argv) > 2
                else datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    # The fragment list is the source of truth: gen_kernels.c dies loudly
    # on any emission failure, so all intended kernels are present.
    frags = sorted(f[:-5] for f in os.listdir(stagedir) if f.endswith(".json"))
    frags = [f for f in frags if f != "manifest"]
    if not frags:
        sys.exit("manifest.py: no kernel fragments in %s" % stagedir)
    kernels = []
    for name in frags:
        frag = os.path.join(stagedir, name + ".json")
        with open(frag) as f:
            k = json.load(f)
        if k.get("name") != name:
            sys.exit("manifest.py: fragment %s has name %r" % (frag,
                                                               k.get("name")))
        for key in ("code_file", "const_file", "asm_file"):
            p = os.path.join(stagedir, k[key])
            if not os.path.isfile(p):
                sys.exit("manifest.py: missing file %s" % p)
        code_p = os.path.join(stagedir, k["code_file"])
        const_p = os.path.join(stagedir, k["const_file"])
        asm_p = os.path.join(stagedir, k["asm_file"])
        if os.path.getsize(code_p) != k["code_size"]:
            sys.exit("manifest.py: size mismatch %s" % code_p)
        if os.path.getsize(const_p) != k["const_size"]:
            sys.exit("manifest.py: size mismatch %s" % const_p)
        k["code_sha256"] = sha256_of(code_p)
        k["const_sha256"] = sha256_of(const_p)
        # .asm files embed the generation timestamp, so this digest only
        # identifies the shipped listing, not a reproducible artifact.
        k["asm_sha256"] = sha256_of(asm_p)
        kernels.append(k)

    manifest = {
        "generated_utc": date_utc,
        "tool": "tools/gen_kernels.c + tools/manifest.py",
        "selection_method": SELECTION_METHOD,
        "entry_contract": ENTRY_CONTRACT,
        "kernels": kernels,
    }
    out = os.path.join(stagedir, "manifest.json")
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print("wrote %s (%d kernels)" % (out, len(kernels)))


if __name__ == "__main__":
    main()
