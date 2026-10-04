/* RVV counted full-group loop (F6) codegen tests.
 *
 * F6 hoisted the per-iteration divu+mul (U not in {1,2,4}) / srli+slli
 * (U in {1,2,4}) out of the strip-mining loop: the new loop is a counted
 * loop over full groups of U*VLMAX bytes with vl pinned at VLMAX, driven
 * by a single prologue divu (niter = len / group) plus one mul/shift for
 * the group size. The old strip-mining loop survives only as the
 * partial-group remainder path (it runs at most once: remaining < group
 * implies remaining/U < VLMAX).
 *
 * Shape checks (all must fail on the pre-F6 codegen):
 *  A. U=3 (divu path): exactly 2 divu + 2 mul words. One divu is the
 *     prologue trip-count divu (rd=x14/A4, rs1=x12/A2=len,
 *     rs2=x13/A3=group); the other is the remainder loop's
 *     (rd=rs2=x30/T5). The prologue divu must sit before the first
 *     backward jal (the counted loop's back edge) and the remainder
 *     divu/mul after it: a per-iteration divu inside the counted loop
 *     is functionally correct but the F6 defect, so shape must pin
 *     both sides. Pre-F6 code has 1 divu and no rd=x14 divu at all.
 *  B. U=4 (shift path): exactly 2 slli + 1 srli. One slli is the
 *     prologue group-size shift (rd=x13/A3, rs1=x5/T0=VLMAX, shamt=2);
 *     it too must precede the first backward jal, with the remainder
 *     shifts after it. Pre-F6 code has 1 slli.
 * Correctness:
 *  C. JIT kernels for U in 1..8 match the scalar reference at
 *     group-boundary lengths for VLEN 128/256/512 (group = U*VLMAX;
 *     VLMAX = VLEN/8 for e8/m1): g-1, g, g+1, g+2, g+3, 2g-1, 2g, 2g+1
 *     for every plausible group g, plus small lengths incl. 0.
 *     U=1..5 run on rs(6,3); U=6..8 on rs(4,1) (register pressure).
 * Table-load windows (post-F6 cleanup):
 *  D. The counted full-group loop runs with vl pinned at VLMAX >= 16,
 *     so its table loads elide the vsetivli-16 save/restore window;
 *     the remainder/tail bodies keep it (their vl can be < 16). The
 *     window's vl-save (addi t5,t0,0) is the unique marker: none may
 *     precede the counted loop's back edge, and at least one must
 *     follow it (remainder/tail). Checked on a P_MULACC plan (c_ldtab
 *     windows) and a P_HORNER remat plan (c_ldtab1 windows).
 *     Pre-elision codegen has saves on both sides.
 *
 * RVV-only: the JIT parts need RVV to execute; the shape scan only
 * needs the riscv64 build. On other hosts the binary builds and prints
 * SKIP.
 *
 * Run under QEMU at several VLENs, e.g.
 *   QEMU_LD_PREFIX=/usr/riscv64-linux-gnu qemu-riscv64 -cpu max \
 *       ./build-riscv64/test_rvv_f6
 *   ... -cpu max,vlen=256 ... ; ... -cpu max,vlen=512 ...
 */
#include "../include/raidjit.h"
#include "../src/plan.h"
#include "../src/jit.h"
#include "../src/cpu.h"
#include "../src/gf.h"
#include "../src/ref.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, ...)                                                   \
    do {                                                                   \
        if (!(cond)) {                                                     \
            printf("FAIL %s:%d: ", __func__, __LINE__);                    \
            printf(__VA_ARGS__);                                           \
            printf("\n");                                                  \
            failures++;                                                    \
        }                                                                  \
    } while (0)

#ifdef __riscv

static uint64_t rng_state = 0x9e3779b97f4a7c15ULL;
static uint64_t xrng(void)
{
    uint64_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return rng_state = x;
}

static void *abuf(size_t n)
{
    void *p = NULL;
    if (posix_memalign(&p, 64, n ? n : 64)) {
        fprintf(stderr, "oom\n");
        exit(1);
    }
    return p;
}

static void randfill(void *p, size_t n)
{
    uint8_t *b = p;
    for (size_t i = 0; i < n; i++)
        b[i] = (uint8_t)xrng();
}

static uint32_t word_at(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

/* R-type M-extension: funct7=0000001, opcode=0110011.
 * divu: funct3=101; mul: funct3=000. */
static int is_divu(uint32_t w)
{
    return (w & 0xfe00707fu) == 0x02005033u;
}
static int is_mul(uint32_t w)
{
    return (w & 0xfe00707fu) == 0x02000033u;
}
static unsigned r_rd(uint32_t w) { return (w >> 7) & 0x1fu; }
static unsigned r_rs1(uint32_t w) { return (w >> 15) & 0x1fu; }
static unsigned r_rs2(uint32_t w) { return (w >> 20) & 0x1fu; }

/* I-type shifts, opcode=0010011, imm[11:5]=0000000.
 * slli: funct3=001; srli: funct3=101. */
static int is_slli(uint32_t w)
{
    return (w & 0xfe00707fu) == 0x00001013u;
}
static int is_srli(uint32_t w)
{
    return (w & 0xfe00707fu) == 0x00005013u;
}
static unsigned i_rd(uint32_t w) { return (w >> 7) & 0x1fu; }
static unsigned i_rs1(uint32_t w) { return (w >> 15) & 0x1fu; }

/* J-type jal, opcode=1101111. */
static int is_jal(uint32_t w)
{
    return (w & 0x7fu) == 0x6fu;
}
static int32_t jal_off(uint32_t w)
{
    int32_t o = (int32_t)(((w >> 31) & 1u) << 20 |
                          ((w >> 21) & 0x3ffu) << 1 |
                          ((w >> 20) & 1u) << 11 |
                          ((w >> 12) & 0xffu) << 12);
    return (o << 11) >> 11; /* sign-extend bit 20 */
}

/* Build one RVV candidate directly, bypassing the planner's twin
 * policy (mirrors tests/test_riscv_prefetch.c). Temporal, no
 * prefetch: the loop shape is twin-independent. remat != 0
 * force-builds a Horner rematerialization twin (plan must be
 * P_HORNER). */
static int build_cand(const plan_t *pl, const rj_cpu_info *cpu, int U,
                      int remat, version_t *v, char *err, size_t errlen)
{
    rj_options o;
    rj_options_default(&o);
    o.benchmark = 0;
    cand_t cand;
    memset(&cand, 0, sizeof(cand));
    cand.isa = ISA_RVV;
    cand.mul = MUL_NIBBLE;
    cand.U = U;
    cand.remat = remat;
    memset(v, 0, sizeof(*v));
    return jit_build_version(pl, &cand, cpu, &o, v, err, errlen);
}

/* Index of the first backward jal (a loop back edge), or -1. */
static long first_backward_jal(const version_t *v)
{
    for (size_t i = 0; i + 4 <= v->code_size; i += 4) {
        uint32_t w = word_at(v->mem + i);
        if (is_jal(w) && jal_off(w) < 0)
            return (long)(i / 4);
    }
    return -1;
}

/* addi x30, x5, 0: the vl-save opening every table-load vsetivli-16
 * window. (Encoding: imm=0, rs1=x5/T0, funct3=000, rd=x30/T5,
 * opcode=0010011.) The only emitters are c_ldtab/c_ldtab1 (no other
 * em_addi(T5, T0, 0) site exists in the RVV backend), so each
 * occurrence marks one windowed table load. */
static int is_vl_save(uint32_t w)
{
    return w == 0x00028f13u;
}

/* D. table-load window elision: the counted full-group loop runs with
 * vl pinned at VLMAX >= 16, so its table loads elide the vsetivli-16
 * save/restore window; the remainder/tail bodies keep it (their vl
 * can be < 16). No vl-save may precede the counted loop's back edge,
 * and at least one must follow it (remainder/tail). Pre-elision
 * codegen has saves on both sides, so both halves fail red. */
static void test_shape_tabwin(const plan_t *pl, const rj_cpu_info *cpu,
                              int U, int remat, const char *what)
{
    char err[256];
    version_t v;
    CHECK(build_cand(pl, cpu, U, remat, &v, err, sizeof(err)) == 0,
          "tabwin %s U=%d: build: %s", what, U, err);
    if (failures)
        return;
    long back = first_backward_jal(&v);
    CHECK(back >= 0, "tabwin %s U=%d [%s]: no backward jal", what, U,
          v.desc);
    long before = 0, after = 0;
    for (size_t i = 0; i + 4 <= v.code_size; i += 4) {
        if (!is_vl_save(word_at(v.mem + i)))
            continue;
        if ((long)(i / 4) < back)
            before++;
        else
            after++;
    }
    CHECK(before == 0,
          "tabwin %s U=%d [%s]: %ld vl-save(s) inside the counted "
          "loop body (window not elided)", what, U, v.desc, before);
    CHECK(after > 0,
          "tabwin %s U=%d [%s]: no vl-save in remainder/tail "
          "(window lost where vl can be < 16)", what, U, v.desc);
    jit_free_version(&v);
}

/* A. divu path (U=3): the trip-count divu is hoisted to the prologue. */
static void test_shape_divu_path(const plan_t *pl, const rj_cpu_info *cpu)
{
    char err[256];
    version_t v;
    CHECK(build_cand(pl, cpu, 3, 0, &v, err, sizeof(err)) == 0,
          "U=3 build: %s", err);
    if (failures)
        return;
    int n_divu = 0, n_mul = 0;
    long prologue_divu = -1, remainder_divu = -1;
    long prologue_mul = -1, remainder_mul = -1;
    for (size_t i = 0; i + 4 <= v.code_size; i += 4) {
        uint32_t w = word_at(v.mem + i);
        long idx = (long)(i / 4);
        if (is_divu(w)) {
            n_divu++;
            /* prologue: divu A4, A2, A3 (niter = len / group) */
            if (r_rd(w) == 14 && r_rs1(w) == 12 && r_rs2(w) == 13)
                prologue_divu = idx;
            /* remainder loop: divu T5, A2, T5 */
            if (r_rd(w) == 30 && r_rs1(w) == 12 && r_rs2(w) == 30)
                remainder_divu = idx;
        }
        if (is_mul(w)) {
            n_mul++;
            /* prologue: mul A3, T0, A3 (group = VLMAX * U) */
            if (r_rd(w) == 13 && r_rs1(w) == 5 && r_rs2(w) == 13)
                prologue_mul = idx;
            /* remainder loop: mul T5, T0, T5 */
            if (r_rd(w) == 30 && r_rs1(w) == 5 && r_rs2(w) == 30)
                remainder_mul = idx;
        }
    }
    CHECK(n_divu == 2, "U=3 [%s]: n_divu=%d, want 2", v.desc, n_divu);
    CHECK(n_mul == 2, "U=3 [%s]: n_mul=%d, want 2", v.desc, n_mul);
    CHECK(prologue_divu >= 0,
          "U=3 [%s]: no prologue divu A4,A2,A3", v.desc);
    CHECK(remainder_divu >= 0,
          "U=3 [%s]: no remainder divu T5,A2,T5", v.desc);
    CHECK(prologue_mul >= 0,
          "U=3 [%s]: no prologue mul A3,T0,A3", v.desc);
    CHECK(remainder_mul >= 0,
          "U=3 [%s]: no remainder mul T5,T0,T5", v.desc);
    long back = first_backward_jal(&v);
    CHECK(back >= 0, "U=3 [%s]: no backward jal", v.desc);
    if (prologue_divu >= 0 && back >= 0)
        CHECK(prologue_divu < back,
              "U=3 [%s]: prologue divu at %ld not before first back "
              "edge at %ld", v.desc, prologue_divu, back);
    /* the remainder loop's divu/mul must sit after the counted loop's
     * back edge: a per-iteration divu with the remainder register
     * pattern inside the counted loop would otherwise pass shape
     * (functionally correct, just slow -- the F6 defect). */
    if (remainder_divu >= 0 && back >= 0)
        CHECK(remainder_divu > back,
              "U=3 [%s]: remainder divu at %ld not after back edge "
              "at %ld", v.desc, remainder_divu, back);
    if (remainder_mul >= 0 && back >= 0)
        CHECK(remainder_mul > back,
              "U=3 [%s]: remainder mul at %ld not after back edge "
              "at %ld", v.desc, remainder_mul, back);
    jit_free_version(&v);
}

/* B. shift path (U=4): the group-size shift is hoisted to the prologue. */
static void test_shape_shift_path(const plan_t *pl,
                                  const rj_cpu_info *cpu)
{
    char err[256];
    version_t v;
    CHECK(build_cand(pl, cpu, 4, 0, &v, err, sizeof(err)) == 0,
          "U=4 build: %s", err);
    if (failures)
        return;
    int n_slli = 0, n_srli = 0;
    long prologue_slli = -1, remainder_slli = -1, remainder_srli = -1;
    for (size_t i = 0; i + 4 <= v.code_size; i += 4) {
        uint32_t w = word_at(v.mem + i);
        long idx = (long)(i / 4);
        if (is_slli(w)) {
            n_slli++;
            /* prologue: slli A3, T0, lgu (group = VLMAX * U);
             * U=4 -> lgu=2. */
            if (i_rd(w) == 13 && i_rs1(w) == 5) {
                prologue_slli = idx;
                CHECK(((w >> 20) & 0x3fu) == 2,
                      "U=4 [%s]: prologue slli shamt=%u, want 2",
                      v.desc, (w >> 20) & 0x3fu);
            }
            /* remainder loop: slli T5, T0, lgu */
            if (i_rd(w) == 30 && i_rs1(w) == 5)
                remainder_slli = idx;
        }
        if (is_srli(w)) {
            n_srli++;
            /* remainder loop: srli T5, A2, lgu */
            if (i_rd(w) == 30 && i_rs1(w) == 12)
                remainder_srli = idx;
        }
    }
    CHECK(n_slli == 2, "U=4 [%s]: n_slli=%d, want 2", v.desc, n_slli);
    CHECK(n_srli == 1, "U=4 [%s]: n_srli=%d, want 1", v.desc, n_srli);
    CHECK(prologue_slli >= 0,
          "U=4 [%s]: no prologue slli A3,T0", v.desc);
    CHECK(remainder_slli >= 0,
          "U=4 [%s]: no remainder slli T5,T0", v.desc);
    CHECK(remainder_srli >= 0,
          "U=4 [%s]: no remainder srli T5,A2", v.desc);
    long back = first_backward_jal(&v);
    CHECK(back >= 0, "U=4 [%s]: no backward jal", v.desc);
    if (prologue_slli >= 0 && back >= 0)
        CHECK(prologue_slli < back,
              "U=4 [%s]: prologue slli at %ld not before first back "
              "edge at %ld", v.desc, prologue_slli, back);
    /* same anti-regression as the divu path: the remainder loop's
     * shifts must sit after the counted loop's back edge. */
    if (remainder_slli >= 0 && back >= 0)
        CHECK(remainder_slli > back,
              "U=4 [%s]: remainder slli at %ld not after back edge "
              "at %ld", v.desc, remainder_slli, back);
    if (remainder_srli >= 0 && back >= 0)
        CHECK(remainder_srli > back,
              "U=4 [%s]: remainder srli at %ld not after back edge "
              "at %ld", v.desc, remainder_srli, back);
    jit_free_version(&v);
}

/* C. boundary correctness: every plausible group g = U*VLMAX for
 * VLEN in {128,256,512} contributes g-1, g, g+1, g+2, g+3, 2g-1, 2g,
 * 2g+1 (multi-byte remainders at the counted->remainder handoff),
 * for U in 1..8 (the divu/mul path's trigger values 5..8 included):
 * U=1..5 on rs(6,3) (register pressure caps it at 5), U=6..8 on
 * rs(4,1). */
static void test_correct_boundaries(const plan_t *pl, const topo_int_t *ti,
                                    const rj_cpu_info *cpu, int U)
{
    static const size_t groups[] = { 16, 32, 48, 64, 96, 128, 192, 256 };
    static const size_t small[] = { 0, 1, 2, 3, 5, 7, 8, 15, 17, 31,
                                    33, 100, 1000 };
    char err[256];
    version_t v;
    CHECK(build_cand(pl, cpu, U, 0, &v, err, sizeof(err)) == 0,
          "U=%d build: %s", U, err);
    if (failures)
        return;
    int nd = ti->ndata, np = ti->npar;
    size_t lens[128];
    int nl = 0;
    for (size_t k = 0; k < sizeof(small) / sizeof(small[0]); k++)
        lens[nl++] = small[k];
    for (size_t k = 0; k < sizeof(groups) / sizeof(groups[0]); k++) {
        size_t g = groups[k];
        size_t b[] = { g - 1, g, g + 1, g + 2, g + 3,
                       2 * g - 1, 2 * g, 2 * g + 1 };
        for (size_t q = 0; q < 8; q++)
            lens[nl++] = b[q];
    }
    for (int li = 0; li < nl && !failures; li++) {
        size_t len = lens[li];
        size_t alen = len ? len : 64;
        uint8_t *in[64], *out[64], *exp[64];
        for (int i = 0; i < nd; i++) {
            in[i] = abuf(alen);
            randfill(in[i], len);
        }
        for (int j = 0; j < np; j++) {
            out[j] = abuf(alen);
            exp[j] = abuf(alen);
            memset(out[j], 0x5A, alen);
        }
        v.fn((const void **)in, (void **)out, len);
        ref_run(pl, (const uint8_t **)in, exp, len);
        for (int j = 0; j < np; j++)
            CHECK(memcmp(out[j], exp[j], len) == 0,
                  "U=%d [%s] len=%zu parity%d mismatch", U, v.desc,
                  len, j);
        for (int i = 0; i < nd; i++)
            free(in[i]);
        for (int j = 0; j < np; j++) {
            free(out[j]);
            free(exp[j]);
        }
    }
    jit_free_version(&v);
}

#endif /* __riscv */

int main(void)
{
#ifdef __riscv
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    if (!cpu.rvv) {
        printf("test_rvv_f6: SKIP (no RVV)\n");
        return failures != 0;
    }
    rj_topology *t = rj_topo_rs(6, 3);
    CHECK(t != NULL, "rj_topo_rs(6,3)");
    if (t) {
        char err[256];
        topo_int_t ti;
        if (topo_int_build(&ti, t, err, sizeof(err)) != 0) {
            CHECK(0, "topo: %s", err);
        } else {
            rj_op op = rj_op_encode();
            plan_t pl;
            if (plan_build(&ti, &op, "rs63", &pl, err, sizeof(err)) != 0) {
                CHECK(0, "plan: %s", err);
            } else {
                test_shape_divu_path(&pl, &cpu);
                test_shape_shift_path(&pl, &cpu);
                /* D (horner-remat half): c_ldtab1 windows elided in
                 * the counted loop, kept in remainder/tail. */
                if (!failures) {
                    CHECK(pl.kind == P_HORNER,
                          "rs63: kind=%d, want P_HORNER", pl.kind);
                    test_shape_tabwin(&pl, &cpu, 2, 1, "horner-remat");
                }
                /* rs(6,3): register pressure caps U at 5; U=6..8 run
                 * on rs(4,1) where they fit. */
                for (int U = 1; U <= 5 && !failures; U++)
                    test_correct_boundaries(&pl, &ti, &cpu, U);
                plan_free(&pl);
            }
            rj_op_free(&op);
            topo_int_free(&ti);
        }
        rj_topo_free(t);
    }
    /* D (mulacc half): c_ldtab windows elided in the counted loop,
     * kept in remainder/tail. Custom 2x2 [[5,1],[7,0]] is not
     * Vandermonde, so the plan is P_MULACC with nontrivial
     * coefficients in column 0. */
    if (!failures) {
        uint8_t m[4] = { 5, 1, 7, 0 };
        rj_topology *t = rj_topo_custom(2, 2, m);
        CHECK(t != NULL, "rj_topo_custom(2,2)");
        if (t) {
            char err[256];
            topo_int_t ti;
            if (topo_int_build(&ti, t, err, sizeof(err)) != 0) {
                CHECK(0, "topo: %s", err);
            } else {
                rj_op op = rj_op_encode();
                plan_t pl;
                if (plan_build(&ti, &op, "mulacc22", &pl, err,
                               sizeof(err)) != 0) {
                    CHECK(0, "plan: %s", err);
                } else {
                    CHECK(pl.kind == P_MULACC,
                          "custom 2x2: kind=%d, want P_MULACC", pl.kind);
                    test_shape_tabwin(&pl, &cpu, 2, 0, "mulacc");
                    plan_free(&pl);
                }
                rj_op_free(&op);
                topo_int_free(&ti);
            }
            rj_topo_free(t);
        }
    }
    /* U=6..8 boundary correctness on a topology where they fit. */
    if (!failures) {
        rj_topology *t = rj_topo_rs(4, 1);
        CHECK(t != NULL, "rj_topo_rs(4,1)");
        if (t) {
            char err[256];
            topo_int_t ti;
            if (topo_int_build(&ti, t, err, sizeof(err)) != 0) {
                CHECK(0, "topo: %s", err);
            } else {
                rj_op op = rj_op_encode();
                plan_t pl;
                if (plan_build(&ti, &op, "rs41", &pl, err,
                               sizeof(err)) != 0) {
                    CHECK(0, "plan: %s", err);
                } else {
                    for (int U = 6; U <= 8 && !failures; U++)
                        test_correct_boundaries(&pl, &ti, &cpu, U);
                    plan_free(&pl);
                }
                rj_op_free(&op);
                topo_int_free(&ti);
            }
            rj_topo_free(t);
        }
    }
    if (failures == 0)
        printf("test_rvv_f6: OK\n");
#else
    printf("test_rvv_f6: SKIP (not riscv)\n");
#endif
    return failures != 0;
}
