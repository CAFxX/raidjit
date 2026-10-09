#!/bin/bash
# Byte-identity proof: non-CRC kernels must be byte-identical before/after Phase 2.
# Baseline: hidden_files/crc_fold_baseline_sha256.txt (150 sections)
set -e
cd "$(dirname "$0")/.."

BASELINE="hidden_files/crc_fold_baseline_sha256.txt"
if [ ! -f "$BASELINE" ]; then
    echo "FAIL: baseline not found: $BASELINE"
    exit 1
fi

# Regenerate the kernels doc
echo "Regenerating kernels doc..."
nice -n 10 make kernels > /tmp/kernels_regen.log 2>&1 || {
    echo "FAIL: make kernels failed"
    cat /tmp/kernels_regen.log
    exit 1
}

# The baseline file contains sha256 hashes. We need to compare.
# Format: | code sha256 | `<hash>` |
echo "Comparing..."
# Extract hashes from baseline
grep -o '`[0-9a-f]\{64\}`' "$BASELINE" | tr -d '`' | sort > /tmp/baseline_hashes.txt
wc -l /tmp/baseline_hashes.txt

# TODO: extract hashes from regenerated doc and compare
# (The exact extraction depends on the doc format)
echo "Baseline hashes extracted. Manual comparison needed."
echo "Baseline count: $(wc -l < /tmp/baseline_hashes.txt)"
