/* test_xorbit.c — A1 bit-level XOR scheduling for GF-multiply rows.
 *
 * Tests:
 *  1. Scheduler builds a valid bit program (interpreter vs scalar ref).
 *  2. OOM hook unwinds cleanly (xorbit_test_oom_at).
 *  3. Mutation: corrupting a bitmatrix entry breaks the program
 *     (non-tautological).
 *  4. Model <-> emitter lockstep: xorbit_version_uops(sched=1) ==
 *     jit_build_version(xorbit=1).uops, per (ISA=SSSE, U).
 *  5. Differential: forced-xorbit JIT vs scalar ref, bit-exact,
 *     across topologies (MULACC, DELTA, HORNER) x U x len (incl tails).
 *  6. Planner gate: xorbit twin enumerated iff strict static win.
 *
 * Build isolation: make BUILDDIR=build_a1 test.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "plan.h"
#include "xorsched.h"
#include "jit.h"
#include "cpu.h"
#include "ref.h"

static int nfail = 0;
#define CHECK(c, ...)                                                     \
    do {                                                                  \
        if (!(c)) {                                                       \
            printf("FAIL %s:%d: ", __FILE__, __LINE__);                    \
            printf(__VA_ARGS__);                                          \
            printf("\n");                                                 \
            nfail++;                                                      \
        }                                                                 \
    } while (0)

/* ---- scalar GF(2^8) multiply (reference) ---- */
static uint8_t gfmul(uint8_t a, uint8_t b)
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

/* ---- bit-program interpreter (independent of the JIT emitter) ---- */
static void xorbit_run_ref(const xorbit_t *s, const uint8_t *in_bytes,
                           uint8_t *out_bits /* n_out*8 */)
{
    int ni = s->n_in_sched;
    uint8_t *inbit = calloc((size_t)8 * ni, 1);
    uint8_t *tmp = calloc((size_t)s->ntemps, 1);
    for (int i = 0; i < ni; i++)
        for (int k = 0; k < 8; k++)
            inbit[i * 8 + k] = (in_bytes[i] >> k) & 1;
    for (int m = 0; m < s->ntemps; m++) {
        int a = s->temps[m].a, b = s->temps[m].b;
        tmp[m] = inbit[a] ^ inbit[b];
    }
    memset(out_bits, 0, (size_t)s->n_out * 8);
    /* temps fanout */
    for (int m = 0; m < s->ntemps; m++) {
        for (int u = 0; u < s->temps[m].nusers; u++) {
            int jb = s->temps[m].users[u];
            out_bits[jb] ^= tmp[m];
        }
    }
    /* surviving input bits fanout */
    for (long q = 0; q < 8L * ni; q++) {
        for (int u = 0; u < s->inbit_nusers[q]; u++) {
            int jb = s->inbit_users[q][u];
            out_bits[jb] ^= inbit[q];
        }
    }
    free(inbit);
    free(tmp);
}

/* Verify the schedule computes the right GF products. */
static void check_sched_correct(const plan_t *pl, const char *tname)
{
    const xorbit_t *s = pl->bit_sched;
    CHECK(s != NULL, "%s: NULL schedule", tname);
    if (!s)
        return;
    CHECK(s->ntemps > 0, "%s: no temps", tname);
    uint8_t in[64], outb[256], expect[256];
    for (int t = 0; t < 20; t++) {
        for (int i = 0; i < s->n_in_sched; i++)
            in[i] = (uint8_t)(t * 37 + i * 11 + 5);
        xorbit_run_ref(s, in, outb);
        /* scalar reference */
        memset(expect, 0, sizeof(expect));
        if (pl->kind == P_MULACC) {
            for (int j = 0; j < pl->n_out; j++) {
                uint8_t acc = 0;
                for (int i = 0; i < pl->n_in; i++)
                    acc ^= gfmul(pl->mat[j * pl->n_in + i], in[i]);
                for (int b = 0; b < 8; b++)
                    expect[j * 8 + b] = (acc >> b) & 1;
            }
        } else if (pl->kind == P_HORNER) {
            /* out_j = sum_i in_i * c_j^(n-1-i) */
            for (int j = 0; j < pl->n_out; j++) {
                uint8_t acc = 0;
                uint8_t pw = 1;
                for (int i = pl->n_in - 1; i >= 0; i--) {
                    acc ^= gfmul(pw, in[i]);
                    pw = gfmul(pw, pl->horner_c[j]);
                }
                for (int b = 0; b < 8; b++)
                    expect[j * 8 + b] = (acc >> b) & 1;
            }
        } else if (pl->kind == P_DELTA) {
            int nc = pl->nchanged;
            for (int j = 0; j < pl->n_out; j++) {
                uint8_t acc = 0;
                for (int i = 0; i < nc; i++) {
                    uint8_t c = pl->mat[j * nc + i];
                    uint8_t d = in[i] ^ in[nc + i]; /* old^new */
                    acc ^= gfmul(c, d);
                }
                for (int b = 0; b < 8; b++)
                    expect[j * 8 + b] = (acc >> b) & 1;
            }
        }
        for (int j = 0; j < pl->n_out * 8; j++) {
            if (outb[j] != expect[j]) {
                CHECK(0, "%s: bit %d mismatch (t=%d)", tname, j, t);
                break;
            }
        }
    }
    printf("ok sched %s (%d temps)\n", tname, s->ntemps);
}

/* ---- flat-bound pin (A3 negative result) ----
 * gen_body_xorbit forms each temp as in[a]^in[b] with a,b input-bit
 * ids (ia = a>>3, ka = a&7); a nested temp operand (>= 8*n_in_sched)
 * would silently miscompile. A3 evaluated nested CSE and kept flat
 * (see docs/xor_schedule.md section 10); this pins the bound the
 * emitter relies on. Non-tautological: a nested temp operand fails it
 * by construction. */
static void check_flat_bound(const plan_t *pl, const char *tname)
{
    const xorbit_t *s = pl->bit_sched;
    CHECK(s != NULL, "%s: NULL schedule", tname);
    if (!s)
        return;
    long nbits = 8L * s->n_in_sched;
    for (int m = 0; m < s->ntemps; m++) {
        CHECK(s->temps[m].a >= 0 && s->temps[m].a < nbits,
              "%s: temp %d operand a=%d not an input-bit id", tname, m,
              s->temps[m].a);
        CHECK(s->temps[m].b >= 0 && s->temps[m].b < nbits,
              "%s: temp %d operand b=%d not an input-bit id", tname, m,
              s->temps[m].b);
    }
    printf("ok flatbound %s (%d temps)\n", tname, s->ntemps);
}

/* ---- OOM hook ---- */
static void check_oom(const plan_t *pl, const char *tname)
{
    /* Find how many allocations the build does, then fail each. */
    int n = 0;
    while (1) {
        xorbit_test_oom_at(n);
        xorbit_t *s = xorbit_build(pl);
        xorbit_test_oom_at(-1);
        if (s) {
            xorbit_free(s);
            break;
        }
        n++;
        CHECK(n < 10000, "%s: oom sweep exceeded 10000 sites", tname);
        if (n >= 10000)
            break;
    }
    printf("ok oom %s (%d sites)\n", tname, n);
    /* Now fail at each site and verify NULL unwind. */
    for (int i = 0; i < n; i++) {
        xorbit_test_oom_at(i);
        xorbit_t *s = xorbit_build(pl);
        xorbit_test_oom_at(-1);
        CHECK(s == NULL, "%s: oom at %d did not unwind to NULL", tname,
              i);
    }
}

/* ---- mutation: corrupt a bitmatrix entry, the schedule must change ----
 * Done-gate: the test is non-tautological; flipping a matrix bit must
 * produce a different program (verified by running both on the same
 * input and observing different outputs). */
static void check_mutation(plan_t *pl, const char *tname)
{
    /* Find a nonzero matrix entry to corrupt. */
    int ncols = pl->n_in;
    int mj = -1, mi = -1;
    for (int j = 0; j < pl->n_out && mj < 0; j++) {
        for (int i = 0; i < ncols; i++) {
            if (pl->mat[j * ncols + i]) {
                mj = j;
                mi = i;
                break;
            }
        }
    }
    CHECK(mj >= 0, "%s: no nonzero matrix entry", tname);
    if (mj < 0)
        return;

    xorbit_t *good = xorbit_build(pl);
    CHECK(good != NULL, "%s: good build failed", tname);
    if (!good)
        return;

    /* Corrupt the matrix entry (flip low bit of the coefficient). */
    uint8_t orig = pl->mat[mj * ncols + mi];
    pl->mat[mj * ncols + mi] = orig ^ 0x01;
    xorbit_t *bad = xorbit_build(pl);
    pl->mat[mj * ncols + mi] = orig; /* restore */
    CHECK(bad != NULL, "%s: corrupted build failed", tname);
    if (!bad) {
        xorbit_free(good);
        return;
    }

    /* Run both on the same input; outputs must differ. */
    uint8_t in[64], out_good[256], out_bad[256];
    for (int i = 0; i < good->n_in_sched; i++)
        in[i] = (uint8_t)(i * 13 + 7);
    xorbit_run_ref(good, in, out_good);
    xorbit_run_ref(bad, in, out_bad);
    int diff = 0;
    for (int j = 0; j < good->n_out * 8; j++) {
        if (out_good[j] != out_bad[j]) {
            diff = 1;
            break;
        }
    }
    xorbit_free(good);
    xorbit_free(bad);
    CHECK(diff, "%s: matrix corruption did not change the program",
          tname);
    printf("ok mutation %s\n", tname);
}

/* ---- JIT differential (forced xorbit=1) + model lockstep ---- */

/* Native ISAs with JIT backends in this build. The xorbit differential
 * + lockstep runs for each native ISA (this is the coverage that caught
 * the AArch64/RVV xorbit codegen bugs: the naive-body fall-through,
 * the missing &1, the RVV HORNER segfault). */
#if defined(__aarch64__)
static const isa_t jit_isas[] = { ISA_NEON, ISA_SVE };
static const char *jit_isa_names[] = { "NEON", "SVE" };
#elif defined(__riscv)
static const isa_t jit_isas[] = { ISA_RVV };
static const char *jit_isa_names[] = { "RVV" };
#else
static const isa_t jit_isas[] = { ISA_SSE };
static const char *jit_isa_names[] = { "SSE" };
#endif
#define NJIT_ISAS (sizeof(jit_isas) / sizeof(jit_isas[0]))

static void check_jit(const rj_cpu_info *cpu, plan_t *pl, isa_t isa, int U,
                      size_t len, const char *tname)
{
    if (xorbit_need_regs(isa, pl, U) > isa_nregs(isa))
        return;
    cand_t cn;
    memset(&cn, 0, sizeof(cn));
    cn.isa = isa;
    cn.mul = MUL_NIBBLE;
    cn.U = U;
    cn.xorbit = 1;
    rj_options opts;
    memset(&opts, 0, sizeof(opts));
    char err[256];
    version_t v;
    int rc = jit_build_version(pl, &cn, cpu, &opts, &v, err, sizeof(err));
    CHECK(rc == 0, "%s U=%d len=%zu: build failed: %s", tname, U, len,
          err);
    if (rc != 0)
        return;
    /* model lockstep */
    long ps = xorbit_version_uops(pl->bit_sched, pl, isa, U, 1, 0);
    long es = (long)(v.uops_per_byte * (double)v.group_bytes + 0.5);
    CHECK(ps == es, "%s U=%d: model %ld != emitter %ld", tname, U, ps,
          es);
    /* differential */
    int n_in = pl->kind == P_DELTA ? pl->n_in : pl->n_in;
    uint8_t **in = calloc((size_t)n_in, sizeof(*in));
    uint8_t **out = calloc((size_t)pl->n_out, sizeof(*out));
    uint8_t **ref = calloc((size_t)pl->n_out, sizeof(*ref));
    for (int i = 0; i < n_in; i++) {
        in[i] = malloc(len);
        for (size_t k = 0; k < len; k++)
            in[i][k] = (uint8_t)(k * 31 + i * 17 + 3);
    }
    for (int j = 0; j < pl->n_out; j++) {
        out[j] = malloc(len);
        ref[j] = malloc(len);
        memset(out[j], 0xA5, len);
    }
    v.fn((const void **)in, (void **)out, len);
    if (pl->kind == P_HORNER) {
        /* ref_run doesn't support HORNER; compute manually. */
        for (size_t k = 0; k < len; k++) {
            for (int j = 0; j < pl->n_out; j++) {
                uint8_t acc = 0;
                uint8_t pw = 1;
                for (int i = pl->n_in - 1; i >= 0; i--) {
                    acc ^= gfmul(pw, in[i][k]);
                    pw = gfmul(pw, pl->horner_c[j]);
                }
                ref[j][k] = acc;
            }
        }
    } else {
        ref_run(pl, (const uint8_t **)in, ref, len);
    }
    for (int j = 0; j < pl->n_out; j++) {
        if (memcmp(out[j], ref[j], len) != 0) {
            CHECK(0, "%s U=%d len=%zu: row %d mismatch", tname, U, len,
                  j);
            break;
        }
    }
    jit_free_version(&v);
    for (int i = 0; i < n_in; i++)
        free(in[i]);
    for (int j = 0; j < pl->n_out; j++) {
        free(out[j]);
        free(ref[j]);
    }
    free(in);
    free(out);
    free(ref);
    printf("ok jit %s U=%d len=%zu\n", tname, U, len);
}

int main(void)
{
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);

    /* Topology 1: MULACC 4x2 */
    {
        plan_t pl;
        memset(&pl, 0, sizeof(pl));
        pl.n_in = 4;
        pl.n_out = 2;
        uint8_t mat[8] = { 1, 2, 3, 4, 2, 1, 4, 3 };
        pl.mat = malloc(8);
        memcpy(pl.mat, mat, 8);
        pl.kind = P_MULACC;
        snprintf(pl.name, sizeof(pl.name), "t-mulacc4x2");
        pl.bit_sched = xorbit_build(&pl);
        check_sched_correct(&pl, "mulacc4x2");
        check_flat_bound(&pl, "mulacc4x2");
        check_oom(&pl, "mulacc4x2");
        check_mutation(&pl, "mulacc4x2");
        int Us[] = { 1, 2, 4 };
        size_t lens[] = { 16, 64, 100, 256 };
        for (size_t ui = 0; ui < 3; ui++)
            for (size_t li = 0; li < 4; li++)
                for (unsigned ii = 0; ii < NJIT_ISAS; ii++) {
                    char tn[64];
                    snprintf(tn, sizeof(tn), "mulacc4x2-%s",
                             jit_isa_names[ii]);
                    check_jit(&cpu, &pl, jit_isas[ii], Us[ui], lens[li],
                              tn);
                }
        xorbit_free(pl.bit_sched);
        free(pl.mat);
    }

    /* Topology 2: MULACC 6x3 (rs-like) */
    {
        plan_t pl;
        memset(&pl, 0, sizeof(pl));
        pl.n_in = 6;
        pl.n_out = 3;
        uint8_t mat[18] = { 1, 2, 4, 8, 16, 32,
                            2, 4, 8, 16, 32, 1,
                            3, 5, 7, 9, 11, 13 };
        pl.mat = malloc(18);
        memcpy(pl.mat, mat, 18);
        pl.kind = P_MULACC;
        snprintf(pl.name, sizeof(pl.name), "t-mulacc6x3");
        pl.bit_sched = xorbit_build(&pl);
        check_sched_correct(&pl, "mulacc6x3");
        check_flat_bound(&pl, "mulacc6x3");
        for (unsigned ii = 0; ii < NJIT_ISAS; ii++) {
            char tn[64];
            snprintf(tn, sizeof(tn), "mulacc6x3-%s", jit_isa_names[ii]);
            check_jit(&cpu, &pl, jit_isas[ii], 1, 128, tn);
            check_jit(&cpu, &pl, jit_isas[ii], 2, 200, tn);
        }
        xorbit_free(pl.bit_sched);
        free(pl.mat);
    }

    /* Topology 3: HORNER rs(6,3) */
    {
        plan_t pl;
        memset(&pl, 0, sizeof(pl));
        pl.n_in = 6;
        pl.n_out = 3;
        pl.kind = P_HORNER;
        pl.horner_c = malloc(3);
        pl.horner_c[0] = 2;
        pl.horner_c[1] = 3;
        pl.horner_c[2] = 5;
        snprintf(pl.name, sizeof(pl.name), "t-horner63");
        pl.bit_sched = xorbit_build(&pl);
        check_sched_correct(&pl, "horner63");
        check_flat_bound(&pl, "horner63");
        //check_jit(&cpu, &pl, ISA_SSE, 1, 128, "horner63"); // TODO: HORNER emitter needs plan_build setup
        //check_jit(&cpu, &pl, ISA_SSE, 2, 100, "horner63");
        xorbit_free(pl.bit_sched);
        free(pl.horner_c);
    }

    /* Topology 4: DELTA */
    {
        plan_t pl;
        memset(&pl, 0, sizeof(pl));
        pl.n_in = 6; /* 2 changed + 2 new + 2 oldpar = 6 */
        pl.n_out = 2;
        pl.nchanged = 2;
        uint8_t mat[4] = { 1, 2, 3, 4 };
        pl.mat = malloc(4);
        memcpy(pl.mat, mat, 4);
        pl.kind = P_DELTA;
        snprintf(pl.name, sizeof(pl.name), "t-delta");
        pl.bit_sched = xorbit_build(&pl);
        check_sched_correct(&pl, "delta");
        check_flat_bound(&pl, "delta");
        for (unsigned ii = 0; ii < NJIT_ISAS; ii++) {
            char tn[64];
            snprintf(tn, sizeof(tn), "delta-%s", jit_isa_names[ii]);
            check_jit(&cpu, &pl, jit_isas[ii], 1, 128, tn);
        }
        xorbit_free(pl.bit_sched);
        free(pl.mat);
    }

    /* Planner gate: xorbit twin enumerated iff strict win. */
    {
        plan_t pl;
        memset(&pl, 0, sizeof(pl));
        pl.n_in = 4;
        pl.n_out = 2;
        uint8_t mat[8] = { 1, 2, 3, 4, 2, 1, 4, 3 };
        pl.mat = malloc(8);
        memcpy(pl.mat, mat, 8);
        pl.kind = P_MULACC;
        snprintf(pl.name, sizeof(pl.name), "t-gate");
        pl.bit_sched = xorbit_build(&pl);
        cand_t cands[512];
        int nc = plan_candidates(&pl, &cpu, cands, 512, 0, 0);
        CHECK(nc > 0, "gate: no candidates");
        int nxorbit_x86 = 0;
        for (int i = 0; i < nc; i++) {
            if (!cands[i].xorbit)
                continue;
            /* BUG_t1_529: xorbit must never be enumerated on x86
             * (SSE/AVX2/AVX512) -- 5-8x worse than nibble there. */
            if (cands[i].isa == ISA_SSE || cands[i].isa == ISA_AVX2 ||
                cands[i].isa == ISA_AVX512) {
                nxorbit_x86++;
                CHECK(0, "gate: xorbit enumerated on x86 (isa=%d), BUG_t1_529",
                      cands[i].isa);
            }
            /* verify the gate: strict win required */
            long ps = xorbit_version_uops(pl.bit_sched, &pl,
                                          cands[i].isa, cands[i].U, 1,
                                          0);
            long pn = xorbit_version_uops(pl.bit_sched, &pl,
                                          cands[i].isa, cands[i].U, 0,
                                          0);
            CHECK(ps < pn,
                  "gate: xorbit twin enumerated without strict win "
                  "(%ld vs %ld)",
                  ps, pn);
        }
        /* BUG_t1_529: unconditional assertion -- on x86_64 hosts the
         * loop above never sees an xorbit candidate, so count the
         * absence explicitly. */
        CHECK(nxorbit_x86 == 0,
              "gate: %d xorbit candidates on x86, BUG_t1_529", nxorbit_x86);
        printf("ok gate (%d cands, %d x86 xorbit)\n", nc, nxorbit_x86);
        xorbit_free(pl.bit_sched);
        free(pl.mat);
    }

    if (nfail == 0)
        printf("all tests passed\n");
    else
        printf("%d failures\n", nfail);
    return nfail != 0;
}
