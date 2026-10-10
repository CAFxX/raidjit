/* REDUX byte-identity test (spec §8.5 anti-leak).
 *
 * A crc_fold=0 kernel must be byte-identical whether it is built through
 * the want_crc=0 path or the want_crc=1 path (gate open): the fold
 * machinery must not perturb non-CRC codegen.
 *
 * For each topology: enumerate want_crc=1 candidates (gate forced open),
 * take a fold candidate fc; build fc (fold), fc0 (same struct, crc_fold
 * cleared) twice (determinism guard), and the matching candidate from a
 * want_crc=0 enumeration (matched field-by-field on every codegen-relevant
 * cand_t field except crc_fold). Assert:
 *   1. build(fc0) is byte-identical across two builds (determinism),
 *   2. build(fc0) == build(want_crc=0 match) (the actual requirement),
 *   3. build(fc) != build(fc0) (anti-vacuity: the fold does change code).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/plan.h"
#include "../src/jit.h"
#include "../src/fused.h"
#include "../src/cpu.h"
#include "../include/raidjit.h"

static int fails = 0;
#define CHECK(cond, ...)                                                  \
    do {                                                                  \
        if (!(cond)) {                                                    \
            printf("FAIL %s:%d: ", __func__, __LINE__);                    \
            printf(__VA_ARGS__);                                          \
            printf("\n");                                                 \
            fails++;                                                      \
        }                                                                 \
    } while (0)

/* Codegen-relevant candidate equality, ignoring crc_fold (and mat, which
 * is a plan-shared pointer, and the cost-model doubles, which never
 * affect emission). */
static int cand_same_shape(const cand_t *a, const cand_t *b)
{
    return a->isa == b->isa && a->mul == b->mul &&
           a->U == b->U && a->nt == b->nt &&
           a->prefetch_dist == b->prefetch_dist &&
           a->ternlog == b->ternlog && a->remat == b->remat &&
           a->notail == b->notail && a->predtail == b->predtail &&
           a->maskedtail == b->maskedtail && a->xorsched == b->xorsched &&
           a->xorbit == b->xorbit && a->mshift == b->mshift &&
           a->lmul == b->lmul && a->pipe == b->pipe;
}

static int build_ok(const plan_t *pl, const cand_t *c, const rj_cpu_info *cpu,
                    const rj_options *opts, version_t *v, const char *tag)
{
    char err[256] = { 0 };
    memset(v, 0, sizeof(*v));
    if (jit_build_version(pl, c, cpu, opts, v, err, sizeof(err)) != 0) {
        printf("FAIL build %s: %s\n", tag, err);
        fails++;
        return 0;
    }
    return 1;
}

static int code_equal(const version_t *a, const version_t *b)
{
    return a->code_size == b->code_size &&
           memcmp(a->mem, b->mem, a->code_size) == 0;
}

static void one_topo(rj_topology *topo, const char *name,
                     const rj_cpu_info *cpu)
{
    topo_int_t ti;
    plan_t pl;
    char err[256];
    rj_op op = rj_op_encode();
    cand_t cands[512];
    int n;

    if (topo_int_build(&ti, topo, err, sizeof(err)) != 0) {
        printf("FAIL %s: topo_int_build: %s\n", name, err);
        fails++;
        return;
    }
    if (plan_build(&ti, &op, name, &pl, err, sizeof(err)) != 0) {
        printf("FAIL %s: plan_build: %s\n", name, err);
        fails++;
        topo_int_free(&ti);
        return;
    }

    /* want_crc=1, gate open: find an AVX-512 fold candidate that BUILDS.
     * Some enumerated fold candidates are cleanly rejected by the
     * backend (register pressure: the 5 VEX temps must sit in xmm0-15);
     * that fail-closed rejection is legitimate, so skip to the first
     * buildable one. */
    pl.want_crc = 1;
    n = plan_candidates(&pl, cpu, cands, 512, 0, 0);
    rj_options bopts;
    rj_options_default(&bopts);
    bopts.benchmark = 0;
    cand_t fcval;
    int found = 0, nclean_reject = 0;
    for (int i = 0; i < n && !found; i++) {
        if (!cands[i].crc_fold || cands[i].isa != ISA_AVX512)
            continue;
        version_t vtry;
        char e2[256] = { 0 };
        memset(&vtry, 0, sizeof(vtry));
        if (jit_build_version(&pl, &cands[i], cpu, &bopts, &vtry, e2,
                              sizeof(e2)) == 0) {
            fcval = cands[i];
            found = 1;
            jit_free_version(&vtry);
        } else {
            CHECK(e2[0] != '\0',
                  "%s: fold candidate rejected without error text", name);
            nclean_reject++;
        }
    }
    /* Copy the fold candidate: the want_crc=0 enumeration below reuses
     * the cands buffer, which would clobber fc->crc_fold. */
    const cand_t *fc = found ? &fcval : NULL;
    if (!fc) {
        printf("SKIP %s: no buildable fold candidate (%d clean rejects)\n",
               name, nclean_reject);
        plan_free(&pl);
        topo_int_free(&ti);
        return;
    }
    if (nclean_reject)
        printf("%s: note: %d fold candidates clean-rejected by backend\n",
               name, nclean_reject);
    cand_t fc0 = *fc;
    fc0.crc_fold = 0;

    /* want_crc=0: find the same-shape candidate (crc_fold is 0 there). */
    pl.want_crc = 0;
    int n0 = plan_candidates(&pl, cpu, cands, 512, 0, 0);
    const cand_t *wc = NULL;
    for (int i = 0; i < n0; i++)
        if (!cands[i].crc_fold && cand_same_shape(&cands[i], &fc0)) {
            wc = &cands[i];
            break;
        }
    CHECK(wc != NULL, "%s: no same-shape want_crc=0 candidate", name);

    rj_options opts;
    rj_options_default(&opts);
    opts.benchmark = 0;
    version_t v_fold, v_a, v_b, v_c;
    int ok = 1;
    ok &= build_ok(&pl, fc, cpu, &opts, &v_fold, "fold");
    ok &= build_ok(&pl, &fc0, cpu, &opts, &v_a, "nofold-a");
    ok &= build_ok(&pl, &fc0, cpu, &opts, &v_b, "nofold-b");
    if (wc)
        ok &= build_ok(&pl, wc, cpu, &opts, &v_c, "plain");
    if (!ok) {
        plan_free(&pl);
        topo_int_free(&ti);
        return;
    }

    /* 1. determinism guard */
    CHECK(code_equal(&v_a, &v_b),
          "%s: same candidate built twice differs (%zu vs %zu bytes)",
          name, v_a.code_size, v_b.code_size);
    /* 2. the requirement: fold machinery must not affect non-CRC codegen */
    if (wc)
        CHECK(code_equal(&v_a, &v_c),
              "%s: crc_fold=0-via-want_crc=1 differs from plain build "
              "(%zu vs %zu bytes)",
              name, v_a.code_size, v_c.code_size);
    /* 3. anti-vacuity: the fold kernel really is different code */
    CHECK(!code_equal(&v_fold, &v_a),
          "%s: fold kernel byte-identical to non-fold (vacuous?)", name);

    printf("ok %s: fold=%zuB nofold=%zuB plain=%s\n", name, v_fold.code_size,
           v_a.code_size, wc ? "match" : "MISSING");
    jit_free_version(&v_fold);
    jit_free_version(&v_a);
    jit_free_version(&v_b);
    if (wc)
        jit_free_version(&v_c);
    plan_free(&pl);
    topo_int_free(&ti);
}

int main(void)
{
#if !defined(__x86_64__)
    printf("SKIP: x86-64 only\n");
    return 0;
#else
    setvbuf(stdout, NULL, _IONBF, 0);
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    if (!cpu.avx512f || !cpu.pclmulqdq) {
        printf("SKIP: host lacks avx512f/pclmulqdq\n");
        return 0;
    }
    /* Force the gate open: byte-identity needs the want_crc=1 path. */
    rj_uarch_override(RJ_UARCH_INTEL_SAPPHIRERAPIDS);

    struct {
        rj_topology *t;
        const char *name;
    } ts[] = {
        { rj_topo_raid5(4), "raid5(4)" },
        { rj_topo_raid6_rs(6), "raid6_rs(6)" },
        { rj_topo_rs(6, 3), "rs(6,3)" },
    };
    for (size_t i = 0; i < sizeof(ts) / sizeof(ts[0]); i++) {
        one_topo(ts[i].t, ts[i].name, &cpu);
        rj_topo_free(ts[i].t);
    }
    if (fails == 0)
        printf("PASS\n");
    return fails != 0;
#endif
}
