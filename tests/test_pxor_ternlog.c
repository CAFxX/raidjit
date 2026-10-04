/* test_pxor_ternlog: differential correctness of the P_XOR AVX-512
 * VPTERNLOGD twin.
 *
 * For npar in 1..3, ndata in 1..8, several 0/1 matrix patterns (dense,
 * checkerboard, seeded random, and an adversarial pattern with zero
 * columns/rows), and U in 1..8, the test builds the plain AVX-512
 * candidate and the VPTERNLOGD twin (plus one NT-store twin to cover
 * the NT path) and runs each on random inputs at lengths hitting
 * main/tail boundaries (0, 1, VL-1, VL, VL+1, group-1, group,
 * group+1, 2*group-1, 2*group, 2*group+1, 4096, 65536). Every output
 * byte is compared against both the plain candidate's output and a
 * scalar reference.
 *
 * Fail-capable: the R8-base bug (mem operand pointing at the in[]
 * pointer array instead of the input buffer) was caught by exactly
 * this differential during development.
 */
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include "../src/jit.h"
#include "../src/plan.h"
#include "../include/raidjit.h"

static int failures = 0;
static int n_executed = 0; /* (config,U) pairs actually differentially run */
static int n_skipped = 0;  /* (config,U) pairs skipped: twin not enumerated */
#define CHECK(c, ...) do {                                             \
    if (!(c)) {                                                        \
        printf("FAIL %s:%d: ", __func__, __LINE__);                     \
        printf(__VA_ARGS__); printf("\n"); failures++;                 \
    }                                                                  \
} while (0)

static uint64_t rng_state = 0x243f6a8885a308d3ULL;
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

/* Scalar P_XOR reference: out[j][k] = XOR over i with mat[j*nd+i] of in[i][k]. */
static void ref_xor(const uint8_t *mat, int nd, int np, uint8_t **d,
                    uint8_t **p, size_t len)
{
    for (int j = 0; j < np; j++) {
        uint8_t *o = p[j];
        memset(o, 0, len);
        for (int i = 0; i < nd; i++) {
            if (!mat[(size_t)j * (size_t)nd + (size_t)i])
                continue;
            uint8_t *di = d[i];
            for (size_t k = 0; k < len; k++)
                o[k] ^= di[k];
        }
    }
}

static rj_cpu_info avx512_cpu(void)
{
    rj_cpu_info cpu;
    memset(&cpu, 0, sizeof cpu);
    cpu.avx = cpu.avx2 = 1;
    cpu.avx512f = cpu.avx512bw = cpu.avx512vl = 1;
    return cpu;
}

/* Index into cands[] of the candidate matching (U, ternlog, nt), or -1. */
static int find_cand(const plan_t *pl, const rj_cpu_info *cpu,
                     const cand_t *cands, int nc, int U, int ternlog, int nt)
{
    (void)pl;
    (void)cpu;
    for (int i = 0; i < nc; i++) {
        if (cands[i].isa == ISA_AVX512 && cands[i].U == U &&
            !!cands[i].ternlog == !!ternlog && !!cands[i].nt == !!nt)
            return i;
    }
    return -1;
}

static int build_ver(const plan_t *pl, const cand_t *cand,
                   const rj_cpu_info *cpu, version_t *v)
{
    rj_options opts;
    rj_options_default(&opts);
    opts.benchmark = 0;
    char err[256];
    if (jit_build_version(pl, cand, cpu, &opts, v, err, sizeof err)) {
        printf("FAIL build isa=%d U=%d ternlog=%d nt=%d: %s\n", cand->isa,
               cand->U, cand->ternlog, cand->nt, err);
        failures++;
        return 0;
    }
    return 1;
}

static void run_ver(const version_t *v, uint8_t **in, uint8_t **out, int nd,
                    int np, size_t len)
{
    const void *vin[8];
    void *vout[3];
    for (int i = 0; i < nd; i++)
        vin[i] = in[i];
    for (int i = nd; i < 8; i++)
        vin[i] = in[0]; /* unused columns: valid pointer, never read */
    for (int j = 0; j < np; j++)
        vout[j] = out[j];
    v->fn(vin, vout, len);
}

/* Differential check for one (np, nd, mat) configuration. */
static void check_config(int np, int nd, const uint8_t *mat, const char *pname,
                         int with_nt, size_t prefetch_dist)
{
    rj_cpu_info cpu = avx512_cpu();
    rj_topology *topo = rj_topo_custom(nd, np, mat);
    rj_op op = rj_op_encode();
    topo_int_t ti;
    char err[256];
    if (topo_int_build(&ti, topo, err, sizeof err)) {
        printf("FAIL tl-diff %s %dx%d: topo: %s\n", pname, nd, np, err);
        failures++;
        return;
    }
    plan_t pl;
    if (plan_build(&ti, &op, "enc", &pl, err, sizeof err)) {
        printf("FAIL tl-diff %s %dx%d: plan: %s\n", pname, nd, np, err);
        failures++;
        topo_int_free(&ti);
        return;
    }
    if (pl.kind != P_XOR) {
        printf("FAIL tl-diff %s %dx%d: kind %d, want P_XOR\n", pname, nd, np,
               pl.kind);
        failures++;
        plan_free(&pl);
        topo_int_free(&ti);
        rj_topo_free(topo);
        rj_op_free(&op);
        return;
    }

    cand_t cands[160];
    int nc = plan_candidates(&pl, &cpu, cands, 160, 1, prefetch_dist);

    size_t maxlen = 65536;
    uint8_t *in[8], *out_plain[3], *out_tl[3], *out_ref[3], *out_nt[3];
    for (int i = 0; i < 8; i++)
        in[i] = abuf(maxlen);
    for (int j = 0; j < 3; j++) {
        out_plain[j] = abuf(maxlen);
        out_tl[j] = abuf(maxlen);
        out_ref[j] = abuf(maxlen);
        out_nt[j] = abuf(maxlen);
    }

    for (int U = 1; U <= 8; U++) {
        int ip = find_cand(&pl, &cpu, cands, nc, U, 0, 0);
        int it = find_cand(&pl, &cpu, cands, nc, U, 1, 0);
        if (ip < 0 || it < 0) {
            /* np==1: no twin by design (n_out>=2 gating); U beyond the
             * pressure limit is likewise not enumerated. Either way the
             * differential cannot run here -- count it as skipped. */
            n_skipped++;
            continue;
        }
        n_executed++;
        int in_ = find_cand(&pl, &cpu, cands, nc, U, 1, 1);
        version_t vp, vt, vn;
        int have_nt = 0;
        if (!build_ver(&pl, &cands[ip], &cpu, &vp))
            continue;
        if (!build_ver(&pl, &cands[it], &cpu, &vt)) {
            jit_free_version(&vp);
            continue;
        }
        if (with_nt && in_ >= 0)
            have_nt = build_ver(&pl, &cands[in_], &cpu, &vn);
        size_t g = (size_t)U * 64;
        size_t lens[] = { 0, 1, 63, 64, 65, g - 1, g, g + 1,
                          2 * g - 1, 2 * g, 2 * g + 1, 4096, 65536 };
        int nlens = (int)(sizeof lens / sizeof lens[0]);
        for (int li = 0; li < nlens; li++) {
            size_t len = lens[li];
            if (len > maxlen)
                continue;
            for (int i = 0; i < nd; i++)
                randfill(in[i], len);
            ref_xor(mat, nd, np, in, out_ref, len);
            run_ver(&vp, in, out_plain, nd, np, len);
            run_ver(&vt, in, out_tl, nd, np, len);
            for (int j = 0; j < np; j++) {
                if (memcmp(out_plain[j], out_ref[j], len)) {
                    printf("FAIL tl-diff %s %dx%d U=%d len=%zu: plain != ref (row %d)\n",
                           pname, nd, np, U, len, j);
                    failures++;
                }
                if (memcmp(out_tl[j], out_ref[j], len)) {
                    printf("FAIL tl-diff %s %dx%d U=%d len=%zu: ternlog != ref (row %d)\n",
                           pname, nd, np, U, len, j);
                    failures++;
                }
            }
            if (have_nt) {
                run_ver(&vn, in, out_nt, nd, np, len);
                for (int j = 0; j < np; j++) {
                    if (memcmp(out_nt[j], out_ref[j], len)) {
                        printf("FAIL tl-diff %s %dx%d U=%d len=%zu: ternlog-nt != ref (row %d)\n",
                               pname, nd, np, U, len, j);
                        failures++;
                    }
                }
            }
        }
        jit_free_version(&vp);
        jit_free_version(&vt);
        if (have_nt)
            jit_free_version(&vn);
    }

    for (int i = 0; i < 8; i++)
        free(in[i]);
    for (int j = 0; j < 3; j++) {
        free(out_plain[j]);
        free(out_tl[j]);
        free(out_ref[j]);
        free(out_nt[j]);
    }
    plan_free(&pl);
    topo_int_free(&ti);
    rj_topo_free(topo);
    rj_op_free(&op);
}

/* One recovery op through the twin path: fail data disk 0 and parity
 * disk 0 of a dense 4+2 topology. The recovery plan has n_out=2, so it
 * gets a ternlog twin; the reference is the plan's own recovery matrix. */
static void check_recover(void)
{
    rj_cpu_info cpu = avx512_cpu();
    uint8_t mat[8];
    memset(mat, 1, sizeof mat);
    rj_topology *topo = rj_topo_custom(4, 2, mat);
    int failed[2] = { 0, 4 };
    rj_op op = rj_op_recover(failed, 2);
    topo_int_t ti;
    plan_t pl;
    char err[256];
    if (topo_int_build(&ti, topo, err, sizeof err) ||
        plan_build(&ti, &op, "rec", &pl, err, sizeof err)) {
        printf("FAIL tl-diff recover: setup: %s\n", err);
        failures++;
        rj_op_free(&op);
        rj_topo_free(topo);
        return;
    }
    if (pl.kind != P_XOR || pl.n_out != 2) {
        printf("FAIL tl-diff recover: kind=%d n_out=%d, want P_XOR/2\n",
               pl.kind, pl.n_out);
        failures++;
        plan_free(&pl);
        topo_int_free(&ti);
        rj_topo_free(topo);
        rj_op_free(&op);
        return;
    }
    cand_t cands[160];
    int nc = plan_candidates(&pl, &cpu, cands, 160, 0, 0);
    int it = -1;
    for (int i = 0; i < nc; i++)
        if (cands[i].isa == ISA_AVX512 && cands[i].U == 2 &&
            cands[i].ternlog && !cands[i].nt) { it = i; break; }
    if (it < 0) {
        printf("FAIL tl-diff recover: no ternlog twin enumerated\n");
        failures++;
    } else {
        version_t vt;
        if (build_ver(&pl, &cands[it], &cpu, &vt)) {
            uint8_t *in[4], *out_tl[2], *out_ref[2];
            size_t len = 1000;
            for (int i = 0; i < 4; i++) {
                in[i] = abuf(len);
                randfill(in[i], len);
            }
            for (int j = 0; j < 2; j++) {
                out_tl[j] = abuf(len);
                out_ref[j] = abuf(len);
            }
            ref_xor(pl.mat, pl.n_in, pl.n_out, in, out_ref, len);
            run_ver(&vt, in, out_tl, pl.n_in, pl.n_out, len);
            n_executed++;
            for (int j = 0; j < 2; j++)
                CHECK(memcmp(out_tl[j], out_ref[j], len) == 0,
                      "recover: ternlog != ref (row %d)", j);
            jit_free_version(&vt);
            for (int i = 0; i < 4; i++)
                free(in[i]);
            for (int j = 0; j < 2; j++) {
                free(out_tl[j]);
                free(out_ref[j]);
            }
        }
    }
    plan_free(&pl);
    topo_int_free(&ti);
    rj_topo_free(topo);
    rj_op_free(&op);
}

int main(void)
{
    /* Matrix patterns over ndata=1..8, npar=1..3. */
    for (int np = 1; np <= 3; np++) {
        for (int nd = 1; nd <= 8; nd++) {
            uint8_t mat[24];
            /* dense */
            memset(mat, 1, sizeof mat);
            check_config(np, nd, mat, "dense", 0, 0);
            /* checkerboard */
            for (int j = 0; j < np; j++)
                for (int i = 0; i < nd; i++)
                    mat[j * nd + i] = (uint8_t)((i + j) & 1);
            check_config(np, nd, mat, "checker", 0, 0);
            /* seeded random, two seeds */
            for (int s = 0; s < 2; s++) {
                rng_state = 0x12345678ULL + (uint64_t)s * 0x9e3779b9ULL +
                            (uint64_t)np * 131ULL + (uint64_t)nd * 17ULL;
                for (int j = 0; j < np; j++)
                    for (int i = 0; i < nd; i++)
                        mat[j * nd + i] = (uint8_t)(xrng() & 1);
                /* adversarial: force a zero column and (if np>1) a zero row */
                mat[0] = 0;
                if (np > 1)
                    memset(mat + (np - 1) * nd, 0, (size_t)nd);
                check_config(np, nd, mat, s ? "rand1+z" : "rand0+z",
                             np == 2 && nd == 4 && s == 0, 0);
            }
        }
    }
    /* prefetch twin: one dense config with prefetch_dist > 0 exercises
     * the c_prefetch_in(cg, RDX) path in the twin. */
    {
        uint8_t mat[16];
        memset(mat, 1, sizeof mat);
        check_config(2, 8, mat, "dense-pf", 0, 4096);
    }
    /* recovery op through the twin path. */
    check_recover();
    /* The suite must actually execute differentials, not skip them all. */
    CHECK(n_executed > 0, "no (config,U) differential executed (all skipped)");
    if (failures == 0)
        printf("test_pxor_ternlog: all OK (executed=%d skipped=%d)\n",
               n_executed, n_skipped);
    else
        printf("test_pxor_ternlog: %d FAILURES (executed=%d skipped=%d)\n",
               failures, n_executed, n_skipped);
    return failures != 0;
}
