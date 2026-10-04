/* SVE2 EOR3 codegen test.
 *
 * On CPUs reporting SVE2 via hwcap, the SVE backend folds the nibble
 * multiply reduction into a single EOR3 per multiply:
 *   HORNER i>0  : TBL lo -> acc, TBL hi -> r2, EOR3 acc,acc,r2,r1
 *                 (was: TBL r2, MOV acc,r2, TBL r2, EOR acc,r2, EOR acc,r1)
 *   MULACC/DELTA: EOR3 acc,acc,r1,r2 on the accumulate path
 *                 (was: EOR acc,r1, EOR acc,r2)
 * The first-column path of mulacc_column keeps copy+xor: EOR3 folds the
 * destination's *live* value in (Zdn ^= Zm ^ Za), but there acc is merely
 * dead, not zero.
 *
 * This test force-builds every SVE candidate (all U, incl. remat twins)
 * for HORNER / MULACC / P_XOR / P_DELTA plans and checks:
 *  1. desc carries "+eor3" iff (cpu.sve2 && plan has a nontrivial multiply),
 *  2. the emitted code contains the EOR3 opcode pattern iff "+eor3" is in
 *     the desc (proves the new path is really taken / really absent),
 *  3. output is bit-exact vs a scalar reference at many lengths, including
 *     vector-boundary lengths (0, 1, VL-1, VL, VL+1).
 *
 * Run under QEMU:
 *   QEMU_LD_PREFIX=/usr/aarch64-linux-gnu qemu-aarch64 \
 *       -cpu max,sve-default-vector-length=128 ./build-aarch64/test_sve_eor3
 *       -cpu max,sve-default-vector-length=256 ./build-aarch64/test_sve_eor3
 * The non-SVE2 baseline path is covered deterministically inside the test
 * via a synthetic cpu with sve2 force-disabled (QEMU's neoverse-n1 model
 * does not advertise SVE hwcaps, so it cannot host the baseline run).
 * Without SVE it prints SKIP and passes.
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

/* EOR3 <Zdn>.D,<Zdn>.D,<Zm>.D,<Za>.D = 0x04203800 | Zm<<16 | Za<<5 | Zdn.
 * Fixed-bits mask (Zdn/Za/Zm fields cleared). */
#define EOR3_FIXED 0x04203800u
#define EOR3_MASK  0xFFE0FC00u

static int count_eor3(const version_t *v)
{
    const uint32_t *w = (const uint32_t *)v->mem;
    size_t n = v->code_size / 4;
    int cnt = 0;
    for (size_t i = 0; i < n; i++)
        if ((w[i] & EOR3_MASK) == EOR3_FIXED)
            cnt++;
    return cnt;
}

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

static uint64_t xrng_state = 0x243F6A8885A308D3ULL;
static uint64_t xrng(void)
{
    uint64_t x = xrng_state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return xrng_state = x;
}

static void *alloc64(size_t len)
{
    void *p = aligned_alloc(64, len ? ((len + 63) & ~(size_t)63) : 64);
    if (!p) {
        fprintf(stderr, "oom\n");
        exit(1);
    }
    return p;
}

/* Reference: out[j][b] = sum_i in[i][b] * mat[j][i] (P_XOR/P_MULACC/P_HORNER
 * share this form; HORNER plans carry the expanded Vandermonde matrix in
 * pl->mat). P_DELTA handled separately below. */
static void check_version(const plan_t *pl, const version_t *v,
                          const char *tag, int vl, int expect_eor3)
{
    int n_in = pl->n_in, n_out = pl->n_out;
    int has_eor3_tag = strstr(v->desc, "+eor3") != NULL;
    CHECK(has_eor3_tag == expect_eor3,
          "%s %s: desc %s \"+eor3\", expected %s", tag, v->desc,
          has_eor3_tag ? "has" : "lacks", expect_eor3 ? "present" : "absent");
    int neor3 = count_eor3(v);
    CHECK((neor3 > 0) == (has_eor3_tag != 0),
          "%s %s: found %d EOR3 insns but desc %s \"+eor3\"", tag, v->desc,
          neor3, has_eor3_tag ? "has" : "lacks");
    /* Nontrivial plans on the EOR3 path must actually emit EOR3s (one
     * per nontrivial multiply site, so >> 0); the scan pattern's false
     * positives are controlled by the expect_eor3==0 cases below. */
    if (expect_eor3)
        CHECK(neor3 > 0, "%s %s: expected EOR3 instructions, found none",
              tag, v->desc);

    size_t lens[] = {0, 1, (size_t)vl - 1, (size_t)vl, (size_t)vl + 1,
                     63, 64, 65, 1000};
    int ncols = (pl->kind == P_DELTA) ? pl->nchanged : pl->n_in;
    int n_in_eff = (pl->kind == P_DELTA) ? 2 * pl->nchanged + n_out : n_in;
    for (unsigned li = 0; li < sizeof(lens) / sizeof(lens[0]); li++) {
        size_t len = lens[li];
        uint8_t **in = malloc(sizeof(*in) * (size_t)n_in_eff);
        uint8_t **out = malloc(sizeof(*out) * (size_t)n_out);
        uint8_t **want = malloc(sizeof(*want) * (size_t)n_out);
        for (int i = 0; i < n_in_eff; i++) {
            in[i] = alloc64(len);
            for (size_t b = 0; b < len; b++)
                in[i][b] = (uint8_t)(xrng() >> 33);
        }
        for (int j = 0; j < n_out; j++) {
            out[j] = alloc64(len);
            want[j] = alloc64(len);
            memset(out[j], 0xA5, len ? len : 1);
            for (size_t b = 0; b < len; b++) {
                uint8_t acc;
                if (pl->kind == P_DELTA) {
                    int nc = pl->nchanged;
                    acc = in[2 * nc + j][b]; /* seed: old parity */
                    for (int i = 0; i < nc; i++) {
                        uint8_t d = in[i][b] ^ in[nc + i][b];
                        acc ^= gf_mul(d,
                            pl->mat[(size_t)j * (size_t)nc + (size_t)i]);
                    }
                } else {
                    acc = 0;
                    for (int i = 0; i < ncols; i++)
                        acc ^= gf_mul(in[i][b],
                            pl->mat[(size_t)j * (size_t)ncols + (size_t)i]);
                }
                want[j][b] = acc;
            }
        }
        const void **din = malloc(sizeof(*din) * (size_t)n_in_eff);
        void **dout = malloc(sizeof(*dout) * (size_t)n_out);
        for (int i = 0; i < n_in_eff; i++)
            din[i] = in[i];
        for (int j = 0; j < n_out; j++)
            dout[j] = out[j];
        v->fn(din, dout, len);
        for (int j = 0; j < n_out; j++)
            CHECK(memcmp(out[j], want[j], len) == 0,
                  "%s %s len=%zu par%d mismatch", tag, v->desc, len, j);
        for (int i = 0; i < n_in_eff; i++)
            free(in[i]);
        for (int j = 0; j < n_out; j++) {
            free(out[j]);
            free(want[j]);
        }
        free(in); free(out); free(want); free(din); free(dout);
    }
}

/* Build every SVE candidate for (topo, op); check desc/code/correctness. */
static void test_one(rj_topology *t, const rj_op *op, const char *tname,
                     plan_kind_t want_kind, int expect_eor3,
                     const rj_cpu_info *cpu)
{
    char err[256], tag[128];
    topo_int_t ti;
    if (topo_int_build(&ti, t, err, sizeof(err))) {
        printf("FAIL %s: topo_int_build: %s\n", tname, err);
        fails++;
        return;
    }
    plan_t pl;
    if (plan_build(&ti, op, tname, &pl, err, sizeof(err))) {
        printf("FAIL %s: plan_build: %s\n", tname, err);
        fails++;
        topo_int_free(&ti);
        return;
    }
    CHECK(pl.kind == want_kind, "%s: expected plan kind %d, got %d",
          tname, want_kind, pl.kind);

    /* The EOR3 expectation only applies on SVE2 machines; without SVE2
     * the baseline path must be used even for nontrivial plans. */
    int exp = expect_eor3 && cpu->sve2;

    cand_t cands[64];
    int nc = plan_candidates(&pl, cpu, cands, 64, 0, 0);
    int n_sve = 0;
    rj_options o;
    rj_options_default(&o);
    for (int i = 0; i < nc; i++) {
        /* All ISA_SVE candidates, including nt/prefetch/remat twins: the
         * EOR3 gate does not depend on those dimensions, and the scan
         * mask is collision-free across every emitter, so the checks
         * apply to every twin. */
        if (cands[i].isa != ISA_SVE)
            continue;
        n_sve++;
        version_t v;
        if (jit_build_version(&pl, &cands[i], cpu, &o, &v, err, sizeof(err))) {
            printf("FAIL %s: jit_build_version: %s\n", tname, err);
            fails++;
            continue;
        }
        snprintf(tag, sizeof(tag), "%s", tname);
        check_version(&pl, &v, tag, cpu->sve_vl, exp);
        jit_free_version(&v);
    }
    /* Guard against a vacuous pass. */
    CHECK(n_sve > 0, "%s: no SVE candidates enumerated", tname);
    printf("%s: %d SVE candidates checked (sve2=%d, expect_eor3=%d)\n",
           tname, n_sve, cpu->sve2, exp);

    plan_free(&pl);
    topo_int_free(&ti);
}

int main(void)
{
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    if (!cpu.sve) {
        printf("test_sve_eor3: SKIP (no SVE)\n");
        return 0;
    }
    printf("test_sve_eor3: SVE detected, vl=%d sve2=%d\n",
           cpu.sve_vl, cpu.sve2);

    rj_topology *t;
    rj_op op;

    /* HORNER with a nontrivial row (raid6 Q): EOR3 expected on SVE2. */
    t = rj_topo_raid6(6);
    op = rj_op_encode();
    test_one(t, &op, "raid6(6) enc", P_HORNER, 1, &cpu);
    rj_op_free(&op);
    rj_topo_free(t);

    /* P_XOR: no multiplies at all, never EOR3 (also exercises the
     * desc/scan negative path on SVE2 machines). */
    t = rj_topo_raid5(4);
    op = rj_op_encode();
    test_one(t, &op, "raid5(4) enc", P_XOR, 0, &cpu);
    rj_op_free(&op);
    rj_topo_free(t);

    /* MULACC via recovery: rs(6,3) losing disks 0,1 reduces to a general
     * linear combination. */
    {
        int failed[2] = {0, 1};
        t = rj_topo_rs(6, 3);
        if (t) {
            op = rj_op_recover(failed, 2);
            test_one(t, &op, "rs(6,3) rec d0+d1", P_MULACC, 1, &cpu);
            rj_op_free(&op);
            rj_topo_free(t);
        } else {
            printf("SKIP rs(6,3): topo_rs unavailable\n");
        }
    }

    /* Degenerate MULACC: every nontrivial coefficient sits on its row's
     * first nonzero column, so all multiplies take the copy+xor
     * first-column path and no EOR3 is emitted. The exact gate must leave
     * the "+eor3" tag off — this pins the biconditional
     * (emitted EOR3s > 0) ⟺ ("+eor3" in desc). Rows [5,1],[7,0] are not
     * Vandermonde ([7,0] != [c,1]), so the plan is P_MULACC. */
    {
        uint8_t m[4] = {5, 1, 7, 0};
        t = rj_topo_custom(2, 2, m);
        op = rj_op_encode();
        test_one(t, &op, "custom 2x2 [[5,1],[7,0]]", P_MULACC, 0, &cpu);
        rj_op_free(&op);
        rj_topo_free(t);
    }

    /* Degenerate 1x1 MULACC ([2] is not all-0/1 and not [1], so P_MULACC):
     * single multiply on the first-column path, no EOR3, no tag. */
    {
        uint8_t m[1] = {2};
        t = rj_topo_custom(1, 1, m);
        op = rj_op_encode();
        test_one(t, &op, "custom 1x1 [2]", P_MULACC, 0, &cpu);
        rj_op_free(&op);
        rj_topo_free(t);
    }

    /* P_DELTA with a nontrivial coefficient. */
    {
        int ch[1] = {0};
        t = rj_topo_raid6(4);
        op = rj_op_parity_update(ch, 1);
        test_one(t, &op, "raid6(4) dupd ch0", P_DELTA, 1, &cpu);
        rj_op_free(&op);
        rj_topo_free(t);
    }

    /* Baseline SVE path with SVE2 force-disabled (synthetic cpu): this is
     * what a non-SVE2 machine generates. The EOR3 path must stay off
     * (no "+eor3", no EOR3 opcode) and the code must still be bit-exact.
     * The baseline code is plain SVE so it executes fine here. */
    {
        rj_cpu_info base = cpu;
        base.sve2 = 0;
        t = rj_topo_raid6(6);
        op = rj_op_encode();
        test_one(t, &op, "raid6(6) enc (no-sve2)", P_HORNER, 0, &base);
        rj_op_free(&op);
        rj_topo_free(t);
    }

    if (fails == 0)
        printf("test_sve_eor3: all OK\n");
    else
        printf("test_sve_eor3: %d FAILURES\n", fails);
    return fails != 0;
}
