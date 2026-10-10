/* test_ui_meta.c — verify rj_ui_twin_valid matches the WASM page's
 * historical gating behavior (TWIN_ISA_OK + TWIN_PLAN_OK).
 *
 * This is a parity test: if the page's behavior changes, this test must
 * be updated to match (the C library is the source of truth).
 */
#include <stdio.h>
#include <string.h>
#include "ui_meta.h"

/* Page isa strings -> isa_t */
static isa_t isa_of(const char *s)
{
    if (!strcmp(s, "avx512")) return ISA_AVX512;
    if (!strcmp(s, "avx2")) return ISA_AVX2;
    if (!strcmp(s, "sse")) return ISA_SSE;
    if (!strcmp(s, "neon")) return ISA_NEON;
    if (!strcmp(s, "sve")) return ISA_SVE;
    if (!strcmp(s, "rvv")) return ISA_RVV;
    return -1;
}

/* Page strategy strings -> plan_kind_t */
static plan_kind_t kind_of(const char *s)
{
    if (!strcmp(s, "xor")) return P_XOR;
    if (!strcmp(s, "mulacc")) return P_MULACC;
    if (!strcmp(s, "horner")) return P_HORNER;
    if (!strcmp(s, "delta")) return P_DELTA;
    return -1;
}

/* The page's historical TWIN_ISA_OK, reimplemented here for parity
 * checking. Returns 1 if the ISA gate passes. */
static int page_isa_ok(const char *twin, const char *isa)
{
    if (!strcmp(twin, "ternlog")) return !strcmp(isa, "avx512");
    if (!strcmp(twin, "permi2b")) return !strcmp(isa, "avx512");
    if (!strcmp(twin, "mshift")) return !strcmp(isa, "avx512");
    if (!strcmp(twin, "maskedtail"))
        return !strcmp(isa, "avx512") || !strcmp(isa, "rvv");
    if (!strcmp(twin, "pipe"))
        return !strcmp(isa, "neon") || !strcmp(isa, "sve") ||
               !strcmp(isa, "rvv");
    if (!strcmp(twin, "predtail")) return !strcmp(isa, "sve");
    if (!strcmp(twin, "xorbit"))
        /* BUG_t1_529: xorbit restricted to non-x86 (no fast vpshufb). */
        return !strcmp(isa, "neon") || !strcmp(isa, "sve") ||
               !strcmp(isa, "rvv");
    return 1; /* no ISA gate */
}

/* The page's historical TWIN_PLAN_OK. Returns 1 if the plan gate passes.
 * n_out and xor_sched and isRecover are the probe values. */
static int page_plan_ok(const char *twin, const char *strategy, int n_out,
                        int xor_sched, int is_recover)
{
    if (!strcmp(twin, "remat")) return !strcmp(strategy, "horner");
    if (!strcmp(twin, "xorbit"))
        return !strcmp(strategy, "mulacc") || !strcmp(strategy, "delta") ||
               !strcmp(strategy, "horner");
    if (!strcmp(twin, "xorsched"))
        return !strcmp(strategy, "xor") && xor_sched;
    if (!strcmp(twin, "permi2b"))
        return !strcmp(strategy, "mulacc") || !strcmp(strategy, "delta");
    if (!strcmp(twin, "mshift"))
        return !strcmp(strategy, "mulacc") || !strcmp(strategy, "delta") ||
               !strcmp(strategy, "horner");
    if (!strcmp(twin, "pipe"))
        return !strcmp(strategy, "mulacc") || !strcmp(strategy, "horner");
    if (!strcmp(twin, "ternlog"))
        return (!strcmp(strategy, "xor") && n_out >= 2) ||
               !strcmp(strategy, "mulacc") || !strcmp(strategy, "delta");
    if (!strcmp(twin, "rec_cf"))
        return !strcmp(strategy, "mulacc") && is_recover;
    return 1; /* no plan gate */
}

static int failures = 0;
static int checks = 0;

static void check(int cond, const char *msg)
{
    checks++;
    if (!cond) {
        failures++;
        printf("FAIL: %s\n", msg);
    }
}

int main(void)
{
    const char *twins[] = {"nt", "ternlog", "remat", "notail", "predtail",
                           "maskedtail", "permi2b", "xorsched", "xorbit",
                           "mshift", "pipe", "rec_cf"};
    const char *isas[] = {"avx512", "avx2", "sse", "neon", "sve", "rvv"};
    const char *strategies[] = {"xor", "mulacc", "horner", "delta"};

    /* Twin name <-> ID consistency */
    for (int t = 0; t < RJ_TWIN_COUNT; t++) {
        char msg[128];
        snprintf(msg, sizeof(msg), "twin %d name matches TWINS[%d]", t, t);
        check(!strcmp(rj_ui_twin_name((rj_twin_id_t)t), twins[t]), msg);
    }

    /* Exclusivity groups match the page's BODY_TWINS/TAIL_TWINS */
    for (int t = 0; t < RJ_TWIN_COUNT; t++) {
        const char *name = twins[t];
        int expect = 0;
        if (!strcmp(name, "xorsched") || !strcmp(name, "xorbit") ||
            !strcmp(name, "permi2b") || !strcmp(name, "mshift") ||
            !strcmp(name, "pipe") || !strcmp(name, "remat") ||
            !strcmp(name, "rec_cf"))
            expect = 1;
        else if (!strcmp(name, "notail") || !strcmp(name, "predtail") ||
                 !strcmp(name, "maskedtail"))
            expect = 2;
        char msg[128];
        snprintf(msg, sizeof(msg), "twin %s group=%d", name, expect);
        check(rj_ui_twin_group((rj_twin_id_t)t) == expect, msg);
    }

    /* Full cross product: C must match the page's historical gates */
    for (int ti = 0; ti < 12; ti++) {
        for (int ii = 0; ii < 6; ii++) {
            for (int si = 0; si < 4; si++) {
                /* n_out in {1, 4} covers the ternlog n_out>=2 gate;
                 * xor_sched in {0,1}; is_recover in {0,1}. */
                for (int n_out = 1; n_out <= 4; n_out += 3) {
                    for (int xs = 0; xs <= 1; xs++) {
                        for (int rc = 0; rc <= 1; rc++) {
                            int page = page_isa_ok(twins[ti], isas[ii]) &&
                                       page_plan_ok(twins[ti], strategies[si],
                                                    n_out, xs, rc);
                            int lib = rj_ui_twin_valid(
                                (rj_twin_id_t)ti, isa_of(isas[ii]),
                                kind_of(strategies[si]), n_out, xs, rc);
                            char msg[256];
                            snprintf(msg, sizeof(msg),
                                     "twin=%s isa=%s strategy=%s n_out=%d "
                                     "xor_sched=%d isRecover=%d: page=%d lib=%d",
                                     twins[ti], isas[ii], strategies[si],
                                     n_out, xs, rc, page, lib);
                            check(page == lib, msg);
                        }
                    }
                }
            }
        }
    }

    /* Field limits */
    check(rj_ui_u_min() == 0, "U min is 0");
    check(rj_ui_u_max() == 8, "U max is 8");
    check(rj_ui_lmul_valid(1) && rj_ui_lmul_valid(2) &&
          rj_ui_lmul_valid(4) && rj_ui_lmul_valid(8),
          "lmul 1/2/4/8 valid");
    check(!rj_ui_lmul_valid(3) && !rj_ui_lmul_valid(0) &&
          !rj_ui_lmul_valid(16),
          "lmul 0/3/16 invalid");
    check(rj_ui_prefetch_min() == 0, "prefetch min is 0");

    printf("%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
