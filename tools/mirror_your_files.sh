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
#   ~/workspace/your_files/raidjit/  (+ raidjit-source.zip)
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

echo "mirror: $Y/raidjit and $Y/raidjit-source.zip now hardlink $G"
