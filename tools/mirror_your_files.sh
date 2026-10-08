#!/bin/sh
# tools/mirror_your_files.sh — re-mirror goal deliverables into
# ~/workspace/your_files as hardlinked real files.
#
# Why this exists: the Library / System Files browser does not follow
# symlinks, so your_files/raidjit must be real directory entries. Hardlinks
# share inodes with the canonical goal files (no duplication, no drift).
#
# Canonical home (single source of truth):
#   ~/workspace/goals/raidjit-follow-on-extensions/files/
# Library-visible mirror:
#   ~/workspace/your_files/raidjit/  (+ raidjit-source.zip, + raidjit_kernels.md, + raidjit_kernels.html)
#
# Re-run after any `make kernels` or other deliverable refresh.
# `make kernels` (tools/finalize_kernels.sh) calls this automatically.
set -eu

REPO=$(cd "$(dirname "$0")/.." && pwd)
G="$REPO/../goals/raidjit-follow-on-extensions/files"
Y="$REPO/../your_files"

[ -d "$G/raidjit" ] || { echo "mirror: missing $G/raidjit" >&2; exit 1; }

rm -rf "$Y/raidjit"
mkdir -p "$Y/raidjit"
# -l = hardlink; -a would preserve symlinks, but there are none under $G.
cp -al "$G/raidjit/." "$Y/raidjit/"

rm -f "$Y/raidjit-source.zip"
ln "$G/raidjit-source.zip" "$Y/raidjit-source.zip"

# Disassembled-kernels reference doc, top-level in your_files per 2026-10-04.
rm -f "$Y/raidjit_kernels.md"
ln "$G/raidjit/raidjit_kernels.md" "$Y/raidjit_kernels.md"

# Interactive kernels explorer page (built by tools/gen_kernels_page.py during
# `make kernels`; top-level hardlink mirrors the .md treatment above).
rm -f "$Y/raidjit_kernels.html"
ln "$G/raidjit/raidjit_kernels.html" "$Y/raidjit_kernels.html"

echo "mirror: $Y/raidjit, $Y/raidjit-source.zip, $Y/raidjit_kernels.md and $Y/raidjit_kernels.html now hardlink $G"
