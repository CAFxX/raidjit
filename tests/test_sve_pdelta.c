/* Regression test: SVE P_DELTA must not clobber the loop counter.
 *
 * The SVE prologue computes the iteration count into x14
 * (udiv x14, x2, x13; subs x14, x14, #1 at loop end). The P_DELTA body
 * used to reuse x14 as its 2nd delta-pointer temp, so after the first
 * iteration x14 held a pointer and the loop ran ~2^48 times (hang).
 * The temp is now x3 (x15 is unusable: c_ldpool clobbers it on SVE).
 *
 * This test force-builds every SVE P_DELTA candidate (all U values) via
 * jit_build_version and validates output against a scalar reference.
 * It must be run under QEMU with SVE enabled, e.g.
 *   QEMU_LD_PREFIX=/usr/aarch64-linux-gnu qemu-aarch64 \
 *       -cpu max,sve-default-vector-length=128 ./build-aarch64/test_sve_pdelta
 * Without SVE it prints SKIP and passes. With the bug present it hangs.
 */
#include "../include/raidjit.h"
#include "../src/plan.h"
#include "../src/jit.h"
#include "../src/cpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static int fails;
#define CHECK(c, ...) do {                                             \
    if (!(c)) {                                                        \
        printf("FAIL "); printf(__VA_ARGS__); printf("\n"); fails++;   \
    }                                                                  \
} while (0)

/* GF(2^8) multiply, 0x11b reduction (scalar reference only). */
static uint8_t gf_mul(uint8_t a, uint8_t b)
{
    uint8_t p = 0;
    while (b) {
        if (b & 1)
            p ^= a;
        int hi = a & 0x80;
        a <<= 1;
        if (hi)
            a ^= 0x1b;
        b >>= 1;
    }
    return p;
}

static uint64_t xrng_state = 0x123456789ABCDEFULL;
static uint64_t xrng(void)
{
    uint64_t x = xrng_state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return xrng_state = x;
}

/* Validate one built SVE P_DELTA candidate against the scalar reference.
 * pl.mat holds the n_out x nchanged delta columns M[j][changed[i]]. */
static void check_candidate(const plan_t *pl, const version_t *v,
                            const char *tag)
{
    int nc = pl->nchanged, npar = pl->n_out;
    int n_in = 2 * nc + npar;
    static const size_t lens[] = { 1, 63, 64, 65, 1000 };
    for (unsigned li = 0; li < sizeof(lens) / sizeof(lens[0]); li++) {
        size_t len = lens[li];
        uint8_t **in = malloc(sizeof(*in) * (size_t)n_in);
        uint8_t **out = malloc(sizeof(*out) * (size_t)npar);
        uint8_t **want = malloc(sizeof(*want) * (size_t)npar);
        for (int i = 0; i < n_in; i++) {
            in[i] = malloc(len ? len : 1);
            for (size_t b = 0; b < len; b++)
                in[i][b] = (uint8_t)(xrng() >> 33);
        }
        for (int j = 0; j < npar; j++) {
            out[j] = malloc(len ? len : 1);
            want[j] = malloc(len ? len : 1);
            /* reference: new_p[j] = old_p[j] ^
             *            sum_i (old[i]^new[i]) * mat[j*nc+i] */
            for (size_t b = 0; b < len; b++) {
                uint8_t acc = in[2 * nc + j][b];
                for (int i = 0; i < nc; i++) {
                    uint8_t d = in[i][b] ^ in[nc + i][b];
                    acc ^= gf_mul(d, pl->mat[(size_t)j * (size_t)nc + (size_t)i]);
                }
                want[j][b] = acc;
            }
        }
        const void **din = malloc(sizeof(*din) * (size_t)n_in);
        void **dout = malloc(sizeof(*dout) * (size_t)npar);
        for (int i = 0; i < n_in; i++)
            din[i] = in[i];
        for (int j = 0; j < npar; j++)
            dout[j] = out[j];
        v->fn(din, dout, len);
        for (int j = 0; j < npar; j++)
            CHECK(memcmp(out[j], want[j], len) == 0,
                  "%s %s len=%zu par%d mismatch", tag, v->desc, len, j);
        for (int i = 0; i < n_in; i++)
            free(in[i]);
        for (int j = 0; j < npar; j++) {
            free(out[j]);
            free(want[j]);
        }
        free(in); free(out); free(want); free(din); free(dout);
    }
}

static void test_one(rj_topology *t, const char *tname, const int *changed,
                     int nchanged, const rj_cpu_info *cpu)
{
    char err[256], tag[128];
    topo_int_t ti;
    if (topo_int_build(&ti, t, err, sizeof(err))) {
        printf("FAIL %s: topo_int_build: %s\n", tname, err);
        fails++;
        return;
    }
    rj_op op = rj_op_parity_update(changed, nchanged);
    plan_t pl;
    if (plan_build(&ti, &op, tname, &pl, err, sizeof(err))) {
        printf("FAIL %s: plan_build: %s\n", tname, err);
        fails++;
        topo_int_free(&ti);
        rj_op_free(&op);
        return;
    }
    CHECK(pl.kind == P_DELTA, "%s: expected P_DELTA plan, got kind=%d",
          tname, pl.kind);

    cand_t cands[64];
    int nc = plan_candidates(&pl, cpu, cands, 64, 0, 0);
    int n_sve = 0;
    rj_options o;
    rj_options_default(&o);
    for (int i = 0; i < nc; i++) {
        if (cands[i].isa != ISA_SVE || cands[i].nt || cands[i].prefetch_dist)
            continue;
        n_sve++;
        version_t v;
        if (jit_build_version(&pl, &cands[i], cpu, &o, &v, err, sizeof(err))) {
            printf("FAIL %s: jit_build_version: %s\n", tname, err);
            fails++;
            continue;
        }
        snprintf(tag, sizeof(tag), "%s ch=%d", tname, nchanged);
        check_candidate(&pl, &v, tag);
        jit_free_version(&v);
    }
    /* Guard against a vacuous pass: with SVE detected there must be
     * SVE P_DELTA candidates. */
    CHECK(n_sve > 0, "%s: no SVE P_DELTA candidates enumerated", tname);
    printf("%s ch=%d: %d SVE candidates checked\n", tname, nchanged, n_sve);

    plan_free(&pl);
    topo_int_free(&ti);
    rj_op_free(&op);
}

int main(void)
{
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    if (!cpu.sve) {
        printf("test_sve_pdelta: SKIP (no SVE)\n");
        return 0;
    }
    printf("test_sve_pdelta: SVE detected, vl=%d\n", cpu.sve_vl);

    int ch1[1] = { 0 };
    int ch2[2] = { 0, 1 };
    rj_topology *t;

    t = rj_topo_raid5(4);
    test_one(t, "raid5(4)", ch1, 1, &cpu);
    test_one(t, "raid5(4)", ch2, 2, &cpu);
    rj_topo_free(t);

    t = rj_topo_raid6(4);
    test_one(t, "raid6(4)", ch1, 1, &cpu);
    test_one(t, "raid6(4)", ch2, 2, &cpu);
    rj_topo_free(t);

    if (fails == 0)
        printf("test_sve_pdelta: all OK\n");
    else
        printf("test_sve_pdelta: %d FAILURES\n", fails);
    return fails != 0;
}
