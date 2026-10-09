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
#   tests/mutate_crc.sh --qemu aarch64 --file src/aarch64/crc_fast.c \
#       --sed 's/0xae689191u/0xae689190u/' --desc 'flip K1 low bit (arm)'
#
# --qemu aarch64|riscv64 cross-builds the mutant and runs it under
# qemu-user (the in-repo red-green story for the ARM/RV backends).
# QEMU runs set RJ_CRC_SKIP_HUGE=1 like the Makefile's QEMU recipes.
set -eu

FILE=""
SED=""
DESC=""
QEMU=""

while [ $# -gt 0 ]; do
    case "$1" in
        --file) FILE="$2"; shift 2 ;;
        --sed)  SED="$2";  shift 2 ;;
        --desc) DESC="$2"; shift 2 ;;
        --qemu) QEMU="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done
[ -n "$FILE" ] && [ -n "$SED" ] || { echo "need --file and --sed" >&2; exit 2; }

REPO="$(cd "$(dirname "$0")/.." && pwd)"
# Worktree parent: ~/workspace/tmp (persistent) -- /tmp is pruned on this VM.
WT_PARENT="${TMPDIR:-$HOME/workspace/tmp}"
mkdir -p "$WT_PARENT"
WT="$(mktemp -d "$WT_PARENT/crc-mut.XXXXXX")"
# Per-run log paths (mktemp): concurrent runs must not clobber each
# other's build/run logs.
BUILD_LOG="$(mktemp /tmp/crc-mut-build.XXXXXX.log)"
RUN_LOG="$(mktemp /tmp/crc-mut-run.XXXXXX.log)"
cleanup() {
    cd "$REPO" 2>/dev/null || true
    git worktree remove --force "$WT" >/dev/null 2>&1 || true
    rm -rf "$WT" "$BUILD_LOG" "$RUN_LOG"
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
MAKE_ARGS=""
RUN_ENV=""
RUN_BIN_PREFIX=""
if [ -n "$QEMU" ]; then
    case "$QEMU" in
        aarch64) QEMU_CPU="max"; QEMU_LD="/usr/aarch64-linux-gnu" ;;
        riscv64) QEMU_CPU="max,zvbc=false,zvkg=false"; QEMU_LD="/usr/riscv64-linux-gnu" ;;
        *) echo "unknown --qemu arch: $QEMU (want aarch64|riscv64)" >&2; exit 2 ;;
    esac
    command -v "qemu-$QEMU" >/dev/null 2>&1 || { echo "qemu-$QEMU not found" >&2; exit 2; }
    ARCH="$QEMU"
    MAKE_ARGS="ARCH=$QEMU CROSS_COMPILE=$QEMU-linux-gnu-"
    RUN_ENV="RJ_CRC_SKIP_HUGE=1 QEMU_LD_PREFIX=$QEMU_LD"
    RUN_BIN_PREFIX="qemu-$QEMU -cpu $QEMU_CPU"
fi
BIN="$WT/build-$ARCH/test_crc_fast"
echo "mutate: building test_crc_fast in worktree (this takes a while)"
# shellcheck disable=SC2086
if ! nice -n 19 make -C "$WT" $MAKE_ARGS "build-$ARCH/test_crc_fast" >$BUILD_LOG 2>&1; then
    echo "mutate: BUILD FAILED (see $BUILD_LOG) -- counts as infra failure, not a kill" >&2
    tail -20 $BUILD_LOG >&2
    exit 2
fi

echo "mutate: running test_crc_fast (mutant)${QEMU:+ under qemu-$QEMU}"
# shellcheck disable=SC2086
if env $RUN_ENV $RUN_BIN_PREFIX "$BIN" >$RUN_LOG 2>&1; then
    echo "mutate: SURVIVED -- the test suite did NOT catch: $DESC" >&2
    tail -5 $RUN_LOG >&2
    exit 1
else
    echo "mutate: KILLED (test failed as required): $DESC"
    grep -m3 "^FAIL" $RUN_LOG || true
    exit 0
fi
