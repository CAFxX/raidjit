#!/bin/sh
# Foreign-CPU coverage via qemu-user.
#
# Runs build-<arch>/test_matrix with RAIDJIT_TEST_CPU=detect (real CPUID detection)
# under qemu-x86_64 with emulated CPU models covering the ISA/extension
# combinations the host may not have. Each model exercises the full
# detection -> plan -> candidate -> JIT -> execute path for its feature set.
#
# Models (coverage actually achieved depends on what the installed QEMU's
# TCG implements; QEMU warns and drops unsupported features, execution
# continues with the remaining feature set):
#   qemu64,+ssse3,-avx        SSE era (pshufb nibble path, no AVX)
#   Haswell                   AVX2 era
#   Skylake-Server            AVX512F/BW/VL, no GFNI (nibble path at 512-bit)
#   Icelake-Server            AVX512 + GFNI
# NOTE: qemu-user 8.2 TCG does not implement AVX-512/GFNI, so on such
# hosts the last two models fall back to AVX2-and-below paths (the warnings
# QEMU prints confirm this). AVX-512/GFNI *execution* is covered by the
# native matrix on AVX-512 hosts; the QEMU value is genuine foreign-CPUID
# detection plus SSE/AVX2-era execution.
set -u

QEMU="${QEMU:-qemu-x86_64}"
# Default binary: arch-specific build dir (build-$(uname -m)/test_matrix).
# Override with $1 for cross builds.
ARCH_DEFAULT="$(uname -m)"
BIN="${1:-build-${ARCH_DEFAULT}/test_matrix}"

if ! command -v "$QEMU" >/dev/null 2>&1; then
    echo "qemu_matrix: $QEMU not found; skipping qemu coverage" >&2
    exit 0
fi
if [ ! -x "$BIN" ]; then
    echo "qemu_matrix: $BIN not built" >&2
    exit 1
fi

fail=0
run_model() {
    model="$1"
    echo "=== qemu -cpu $model ==="
    if ! RAIDJIT_TEST_CPU=detect "$QEMU" -cpu "$model" "$BIN"; then
        echo "FAILED under -cpu $model" >&2
        fail=1
    fi
}

run_model "qemu64,+ssse3,-avx"
run_model "Haswell"
run_model "Skylake-Server"
run_model "Icelake-Server"

if [ "$fail" -ne 0 ]; then
    echo "qemu_matrix: FAILURES" >&2
    exit 1
fi
echo "qemu_matrix: all OK"
