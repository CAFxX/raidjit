#!/bin/sh
# tools/finalize_kernels.sh <stagedir> — disassemble staged kernel bins,
# build manifest.json, install into the goal files dir
# (goals/raidjit-follow-on-extensions/files/raidjit/kernels/), build the
# disassembled-kernels reference document raidjit_kernels.md, then mirror
# into your_files/ (the Library / System Files browser does not follow
# symlinks, so your_files holds hardlinks).
# Run via `make kernels` (rvv is generated last, immediately before this step).
set -eu

STAGE=${1:?usage: finalize_kernels.sh <stagedir>}
REPO=$(cd "$(dirname "$0")/.." && pwd)
DEST=$REPO/../goals/raidjit-follow-on-extensions/files/raidjit/kernels
DATE_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)

for c in objdump aarch64-linux-gnu-objdump riscv64-linux-gnu-objdump \
    riscv64-linux-gnu-as python3; do
    command -v "$c" >/dev/null 2>&1 || {
        echo "finalize_kernels.sh: missing required tool: $c" >&2
        exit 1
    }
done

NAMES=""
for j in "$STAGE"/*.json; do
    n=$(basename "$j" .json)
    [ "$n" = "manifest" ] && continue
    NAMES="$NAMES $n"
done
[ -n "$NAMES" ] || { echo "finalize_kernels.sh: no kernel fragments in $STAGE" >&2; exit 1; }
# Strip leading space for tidy logs.
NAMES=$(echo "$NAMES" | sed 's/^ //')
echo "finalize_kernels.sh: kernels:$NAMES"

disasm() { # $1 = kernel name, $2 = objdump, rest = objdump args
    name=$1
    shift
    {
        echo "# $name.asm — disassembly of $name.code.bin"
        echo "# generated $DATE_UTC by tools/gen_kernels.c (make kernels)"
        echo "# bytes verified by execution before disassembly; see manifest.json"
        echo ""
        "$@" -D "$STAGE/$name.code.bin"
    } >"$STAGE/$name.asm"
}

disasm_rvv() { # $1 = kernel name
    # Raw-word disassembly cannot select the V extension under
    # `-b binary` (hence the old .insn-only listings). Instead, wrap each
    # 32-bit little-endian word as a .word directive under an $x mapping
    # label, assemble with -march=rv64gcv, and disassemble the ELF:
    # binutils 2.42 then decodes RVV 1.0 (vle8.v, vsetvli, vrgather.vv, ...).
    name=$1
    sfile=$STAGE/$name.s
    ofile=$STAGE/$name.o
    # Clean up scratch even if as/objdump fail below (stage is /tmp anyway).
    trap 'rm -f "$sfile" "$ofile"' EXIT
    python3 - "$STAGE/$name.code.bin" >"$sfile" <<'EOF'
import struct, sys
data = open(sys.argv[1], 'rb').read()
assert len(data) % 4 == 0, "code size not a multiple of 4"
print("$x:")
for (w,) in struct.iter_unpack("<I", data):
    print(".word 0x%08x" % w)
EOF
    riscv64-linux-gnu-as -march=rv64gcv "$sfile" -o "$ofile"
    {
        echo "# $name.asm — disassembly of $name.code.bin"
        echo "# generated $DATE_UTC by tools/gen_kernels.c (make kernels)"
        echo "# bytes verified by execution before disassembly; see manifest.json"
        echo "# method: each 32-bit word wrapped as .word under an \$x mapping"
        echo "# label, assembled with riscv64-linux-gnu-as -march=rv64gcv,"
        echo "# disassembled with riscv64-linux-gnu-objdump -d (binutils 2.42"
        echo "# decodes RVV 1.0)."
        echo ""
        riscv64-linux-gnu-objdump -d "$ofile"
    } >"$STAGE/$name.asm"
    rm -f "$sfile" "$ofile"
    trap - EXIT
    # Harden against toolchain drift: objdump must have decoded real
    # instructions, not fallen back to .word data.
    if grep -qE '^[[:space:]]+[0-9a-f]+:[[:space:]]+[0-9a-f]+[[:space:]]+\.word' \
        "$STAGE/$name.asm"; then
        echo "finalize_kernels.sh: $name.asm contains undecodable .word lines" >&2
        exit 1
    fi
}

for n in $NAMES; do
    case "$n" in
        *avx512*) disasm "$n" objdump -b binary -m i386:x86-64;;
        *sve2*)   disasm "$n" aarch64-linux-gnu-objdump -b binary -m aarch64;;
        *rvv*)    disasm_rvv "$n";;
        *) echo "finalize_kernels.sh: unknown target for $n" >&2; exit 1;;
    esac
done

# Pass our timestamp so manifest.json and the .asm headers agree.
python3 "$REPO/tools/manifest.py" "$STAGE" "$DATE_UTC"

mkdir -p "$DEST"
for n in $NAMES; do
    cp "$STAGE/$n.code.bin" "$STAGE/$n.const.bin" "$STAGE/$n.asm" "$DEST/"
done
cp "$STAGE/manifest.json" "$DEST/"
# README source lives in the repo (tools/kernels-README.md) so the pipeline
# installs it — no hand-copying from /tmp.
cp "$REPO/tools/kernels-README.md" "$DEST/README.md"

# Build the disassembled-kernels reference document: per-kernel configuration
# table plus the full disassembly of all six kernels (replaces kernels.zip as
# the shipped deliverable).
python3 "$REPO/tools/gen_kernels_md.py" "$STAGE" "$DEST/../raidjit_kernels.md"
NK=$(echo "$NAMES" | wc -w)
echo "finalize_kernels.sh: installed $NK kernels + manifest.json + README.md, built raidjit_kernels.md"
ls -la "$DEST" "$DEST/../raidjit_kernels.md"
# Refresh the Library-visible hardlink mirror (your_files/).
sh "$REPO/tools/mirror_your_files.sh"
