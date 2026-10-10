#!/bin/bash
# Remaining make-test run list (post-reboot). test_pdelta skipped per task
# (heavy/unrelated). Logs persist in the worktree (not /tmp).
cd ~/workspace/worktrees/raidjit/crc-fold-v2 || exit 1
B=build-x86_64
LOGD=hidden_files/verify_logs
mkdir -p "$LOGD"
pass=0; fail=0; failed_tests=""
for t in test_survivor test_tailduel test_b4_prefetch test_prefetch_calib \
         test_traps test_planner_fuzz test_recover_closedform test_pipe \
         test_crc test_crc_fast test_fused test_striping test_planid \
         test_batched test_lrc test_twin_audit \
         test_crc_fold_gate; do
    # test_xorcode covered by the separate 1h background run
    # (hidden_files/verify_logs/rest_test_xorcode3.log); test_pdelta
    # skipped per task (heavy/unrelated).
    echo "=== $t ==="
    if ./$B/$t > "$LOGD/rest_$t.log" 2>&1; then
        echo "PASS $t"
        pass=$((pass+1))
    else
        echo "FAIL $t (exit $?)"
        fail=$((fail+1))
        failed_tests="$failed_tests $t"
    fi
done
echo "SUMMARY: pass=$pass fail=$fail failed:$failed_tests"
