#!/bin/bash
# run_parity.sh — the byte-identity master gate (SPEC §S9.2).
#
# For every case in parity_matrix.json:
#   1. run the NATIVE per-ARCH parity driver (rjw_api.c without -DRJ_WASM)
#      -> result JSON + code/const blobs
#   2. run the WASM module under node (parity_compare.js)
#      -> result JSON + blob sha256
#   3. require: ok/fail agreement, identical result JSON (all fields),
#      identical code+const sha256.
#
# Usage: ./run_parity.sh [x86|aarch64|riscv]   (default: all available)
# aarch64/riscv native drivers are cross-compiled and run under qemu.
set -u
PARITY_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$PARITY_DIR/../../.." && pwd)"
WASM_DIR="$REPO/tools/wasm"
MATRIX="$PARITY_DIR/parity_matrix.json"
WORK="$PARITY_DIR/work"
mkdir -p "$WORK"

ARCHS="${1:-all}"

CFLAGS="-O2 -std=c11 -D_POSIX_C_SOURCE=200809L -ffp-contract=off \
  -DRJ_WASM \
  -I$REPO/include -I$REPO/src -I$WASM_DIR"
# NOTE: -DRJ_WASM for the driver activates only the const-pool gap memset
# (src/*/jit.c) and the malloc pivot — both byte-neutral for emitted code.
# The driver is compile-only (never executes), so this is safe, and it makes
# the native const blob deterministic: without it, gap bytes hold malloc
# garbage that varies with the driver's own allocation history, causing
# flaky blob mismatches (e.g. raid6-5/enc/auto passed with one driver build,
# failed with another). Library native builds (make kernels) are unaffected.
COMMON="$REPO/src/gf.c $REPO/src/plan.c $REPO/src/topo.c $REPO/src/bench.c \
  $REPO/src/ref.c $REPO/src/api.c $REPO/src/prune.c $REPO/src/dispatch.c \
  $REPO/src/xorsched.c $REPO/src/matrix_search.c $REPO/src/xorcode.c \
  $REPO/src/survivor.c $REPO/src/prefetch_calib.c $REPO/src/planid.c \
  $REPO/src/crc.c $REPO/src/striping.c $REPO/src/api_crc.c \
  $REPO/src/fused.c $REPO/src/api_batched.c \
  $WASM_DIR/rjw_api.c $WASM_DIR/vendor/cJSON.c $PARITY_DIR/parity_driver.c"

build_driver() { # <arch> <cc> <rjwdef> <archsrcs> <cpu stub note>
  local arch="$1" cc="$2" rjwdef="$3" archsrcs="$4"
  local out="$WORK/parity_driver_$arch"
  if [ -x "$out" ]; then echo "driver $arch: cached"; return 0; fi
  # shellcheck disable=SC2086
  $cc $CFLAGS -D$rjwdef $COMMON $archsrcs -o "$out" -lm -lpthread \
    >"$WORK/build-$arch.log" 2>&1 || {
    echo "driver $arch: BUILD FAILED (see work/build-$arch.log)"
    return 1
  }
  echo "driver $arch: built"
}

run_native() { # <arch> <case.json> <prefix> -> prints result json; writes blobs
  local arch="$1" json="$2" prefix="$3"
  case "$arch" in
  x86) "$WORK/parity_driver_x86" "$json" "$prefix" >"$prefix.native.json" 2>"$prefix.native.err"
    return $? ;;
  aarch64) QEMU_LD_PREFIX=/usr/aarch64-linux-gnu qemu-aarch64 -cpu max \
    "$WORK/parity_driver_aarch64" "$json" "$prefix" >"$prefix.native.json" 2>"$prefix.native.err"
    return $? ;;
  riscv) QEMU_LD_PREFIX=/usr/riscv64-linux-gnu qemu-riscv64 -cpu max \
    "$WORK/parity_driver_riscv" "$json" "$prefix" >"$prefix.native.json" 2>"$prefix.native.err"
    return $? ;;
  esac
}

MODULE_JS_x86="$WASM_DIR/out/raidjit_x86.js"
MODULE_JS_aarch64="$WASM_DIR/out/raidjit_aarch64.js"
MODULE_JS_riscv="$WASM_DIR/out/raidjit_riscv.js"

pass=0; fail=0; failed_cases=()

want() {
  case "$ARCHS" in all) return 0 ;; "$1") return 0 ;; *) return 1 ;; esac
}

if want x86; then
  build_driver x86 gcc RJW_X86 "$REPO/src/x86_64/emit.c $REPO/src/x86_64/jit.c $REPO/src/x86_64/cpu.c" \
    || exit 1
fi
if want aarch64; then
  build_driver aarch64 aarch64-linux-gnu-gcc RJW_AARCH64 \
    "$REPO/src/aarch64/emit.c $REPO/src/aarch64/jit.c $REPO/src/aarch64/cpu.c" || exit 1
fi
if want riscv; then
  build_driver riscv riscv64-linux-gnu-gcc RJW_RISCV \
    "$REPO/src/riscv/emit.c $REPO/src/riscv/jit.c $REPO/src/riscv/cpu.c" || exit 1
fi

ncases=$(python3 -c "import json; print(len(json.load(open('$MATRIX'))))")
echo "parity matrix: $ncases cases"

i=0
python3 - "$MATRIX" <<'EOF' >"$WORK/cases.tsv"
import json, sys
for c in json.load(open(sys.argv[1])):
    print(f"{c['name']}\t{c['module']}\t{c['expect']}\t{json.dumps(c['options'])}")
EOF

while IFS=$'\t' read -r name module expect options; do
  i=$((i + 1))
  if ! want "$module"; then continue; fi
  # module name in matrix is x86/aarch64/riscv; wasm files use x86_64-style arch map
  case "$module" in
  x86) modjs="$MODULE_JS_x86"; arch=x86 ;;
  aarch64) modjs="$MODULE_JS_aarch64"; arch=aarch64 ;;
  riscv) modjs="$MODULE_JS_riscv"; arch=riscv ;;
  esac
  if [ ! -f "$modjs" ]; then
    echo "SKIP $name: $modjs not built"; continue
  fi
  prefix="$WORK/case_$i"
  echo "$options" >"$prefix.opts.json"

  run_native "$arch" "$prefix.opts.json" "$prefix"
  nrc=$?
  wjson=$(node "$PARITY_DIR/parity_compare.js" "$modjs" "$prefix.opts.json")
  wok=$(echo "$wjson" | python3 -c "import json,sys; print(str(json.load(sys.stdin)['ok']).lower())")

  # ok/fail agreement (native driver rc!=0 means failure)
  nok=$([ $nrc -eq 0 ] && echo true || echo false)
  if [ "$nok" != "$wok" ]; then
    echo "FAIL $name: native ok=$nok vs wasm ok=$wok"
    echo "  native: $(cat "$prefix.native.json" 2>/dev/null | head -c 300)"
    echo "  wasm:   $(echo "$wjson" | head -c 300)"
    fail=$((fail + 1)); failed_cases+=("$name"); continue
  fi
  if [ "$expect" = "fail" ]; then
    # both failed: only the failure itself is compared (agreement is enough)
    pass=$((pass + 1)); continue
  fi
  # both ok: compare result JSON + blob hashes
  nres=$(cat "$prefix.native.json")
  wres=$(echo "$wjson" | python3 -c "import json,sys; print(json.dumps(json.load(sys.stdin)['result'], sort_keys=True))")
  nres_norm=$(echo "$nres" | python3 -c "import json,sys; print(json.dumps(json.load(sys.stdin), sort_keys=True))")
  if [ "$nres_norm" != "$wres" ]; then
    echo "FAIL $name: result JSON differs"
    echo "  native: $nres_norm" | head -c 500; echo
    echo "  wasm:   $wres" | head -c 500; echo
    fail=$((fail + 1)); failed_cases+=("$name"); continue
  fi
  ncode=$(sha256sum "$prefix.code.bin" | cut -d' ' -f1)
  nconst=$(sha256sum "$prefix.const.bin" | cut -d' ' -f1)
  wcode=$(echo "$wjson" | python3 -c "import json,sys; print(json.load(sys.stdin)['code_sha256'])")
  wconst=$(echo "$wjson" | python3 -c "import json,sys; print(json.load(sys.stdin)['const_sha256'])")
  if [ "$ncode" != "$wcode" ] || [ "$nconst" != "$wconst" ]; then
    echo "FAIL $name: blob sha differs (code $ncode vs $wcode)"
    fail=$((fail + 1)); failed_cases+=("$name"); continue
  fi
  pass=$((pass + 1))
done <"$WORK/cases.tsv"

echo "parity: $pass passed, $fail failed"
if [ ${#failed_cases[@]} -gt 0 ]; then
  printf 'failed: %s\n' "${failed_cases[@]}"
  exit 1
fi
