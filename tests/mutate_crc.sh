#!/bin/sh
# tests/mutate_crc.sh -- red-green evidence for the CRC test matrix.
#
# Applies a single-constant mutation to a CRC source file inside a scratch
# git worktree, rebuilds test_crc_fast there, and runs it. Success for this
# script means the test suite KILLED the mutant (non-zero exit); if the
# mutant survives (exit 0), the script fails -- the matrix is insufficient.
#
# Usage:
#   tests/mutate_crc.sh --file src/crc.c \
#       --sed 's/0xEDB88320u/0xEDB88321u/' \
#       --desc 'flip CRC polynomial low bit'
#
# The worktree keeps the mutation out of the shared working tree (sibling
# workers' builds never see it). Requires the backend under test to be
# present in the worktree: for per-backend mutations (e.g. flip K1/K2 in
# src/x86_64/crc_fast.c) run after W1/W2/W3 land.
set -eu

FILE=""
SED=""
DESC=""

while [ $# -gt 0 ]; do
    case "$1" in
        --file) FILE="$2"; shift 2 ;;
        --sed)  SED="$2";  shift 2 ;;
        --desc) DESC="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done
[ -n "$FILE" ] && [ -n "$SED" ] || { echo "need --file and --sed" >&2; exit 2; }

REPO="$(cd "$(dirname "$0")/.." && pwd)"
# Worktree parent: ~/workspace/tmp (persistent) -- /tmp is pruned on this VM.
WT_PARENT="${TMPDIR:-$HOME/workspace/tmp}"
mkdir -p "$WT_PARENT"
WT="$(mktemp -d "$WT_PARENT/crc-mut.XXXXXX")"
cleanup() {
    cd "$REPO" 2>/dev/null || true
    git worktree remove --force "$WT" >/dev/null 2>&1 || true
    rm -rf "$WT"
}
trap cleanup EXIT INT TERM

echo "mutate: worktree $WT"
git -C "$REPO" worktree add --detach "$WT" HEAD >/dev/null 2>&1
cp "$WT/$FILE" "$WT/$FILE.orig"
# The --sed program should name its constant exactly (e.g.
# s/0xEDB88320u/0xEDB88321u/); the diff below shows every hunk it made.
sed -i "$SED" "$WT/$FILE"
if cmp -s "$WT/$FILE" "$WT/$FILE.orig"; then
    echo "mutate: sed expression matched nothing in $FILE" >&2
    exit 2
fi
echo "mutate: $DESC"
diff "$WT/$FILE.orig" "$WT/$FILE" | head -8 || true

ARCH="$(uname -m)"
BIN="$WT/build-$ARCH/test_crc_fast"
echo "mutate: building test_crc_fast in worktree (this takes a while)"
if ! nice -n 19 make -C "$WT" "build-$ARCH/test_crc_fast" >/tmp/crc-mut-build.log 2>&1; then
    echo "mutate: BUILD FAILED (see /tmp/crc-mut-build.log) -- counts as infra failure, not a kill" >&2
    tail -20 /tmp/crc-mut-build.log >&2
    exit 2
fi

echo "mutate: running test_crc_fast (mutant)"
if "$BIN" >/tmp/crc-mut-run.log 2>&1; then
    echo "mutate: SURVIVED -- the test suite did NOT catch: $DESC" >&2
    tail -5 /tmp/crc-mut-run.log >&2
    exit 1
else
    echo "mutate: KILLED (test failed as required): $DESC"
    grep -m3 "^FAIL" /tmp/crc-mut-run.log || true
    exit 0
fi
