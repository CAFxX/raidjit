#!/bin/bash
# Keep-going native suite runner: records PASS/FAIL per test binary.
cd /home/hatch/workspace/worktrees/raidjit/lead-l3-aarch64-gates || exit 1
B=build-x86_64
pass=0; fail=0; failed_list=""
run_one() {
    local t="$1"; shift
    out=$("./$B/$t" "$@" 2>&1)
    code=$?
    if [ $code -ne 0 ] || echo "$out" | grep -q "^FAIL"; then
        echo "FAIL $t (exit=$code)"
        echo "$out" | grep "^FAIL" | head -5
        fail=$((fail+1)); failed_list="$failed_list $t"
    else
        echo "PASS $t"
        pass=$((pass+1))
    fi
}
run_one test_aarch64_emit
run_one test_riscv_maskedtail_emit
run_one test_emit
run_one test_affine
run_one test_cet
run_one test_scan_align
run_one test_remat
run_one test_codegen_shape
run_one test_pxor_ternlog
run_one test_maskedtail
run_one test_permi2b
run_one test_xorbit
run_one test_mshift
run_one test_crc_fast_x86
run_one test_pinned_negative
run_one test_rec_cf_emit
run_one test_crc_fold
run_one test_crc_fold_prop open
run_one test_crc_fold_prop closed
run_one test_crc_fold_byteid
run_one test_crc_fold_combo
run_one test_crc_fold_gate
run_one test_aarch64_phantom_gates
run_one test_nt
run_one test_pdelta
run_one test_pdelta_edge
run_one test_boundaries
run_one test_large_mds
run_one test_pressure
run_one test_select
run_one test_benchreject
run_one test_correct
run_one test_matrix
run_one test_prune
run_one test_dispatch
run_one test_xorsched
run_one test_xorsched_e2e
run_one test_matrix_search
run_one test_xorcode
run_one test_notail_xorsched
run_one test_survivor
run_one test_tailduel
run_one test_b4_prefetch
run_one test_prefetch_calib
run_one test_traps
run_one test_planner_fuzz
run_one test_ports
run_one test_ui_meta
run_one test_recover_closedform
run_one test_pipe
run_one test_crc
run_one test_crc_fast
run_one test_fused
run_one test_striping
run_one test_planid
run_one test_batched
run_one test_lrc
run_one test_lrc_local
run_one test_twin_audit
run_one test_emitter_oom
echo "================ SUMMARY: pass=$pass fail=$fail"
echo "failed:$failed_list"
