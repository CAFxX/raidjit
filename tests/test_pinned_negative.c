/* Pinned-candidate negative test (crash-fix regression).
 *
 * The WASM pinned-candidate API lets users set arbitrary twin
 * combinations, bypassing src/api.c's planner gates. The backend
 * (jit_build_version) is the last line of defense: it must reject
 * invalid combinations gracefully (rc != 0), never abort via assert().
 *
 * This test feeds invalid pinned candidates directly to
 * jit_build_version, fork-isolated per case. An abort (signal) fails
 * the test; a clean rejection passes.
 *
 * Non-tautological: reverting the crash-fix gates (commit c5c1b4f)
 * makes these cases abort instead of rejecting.
 */
#include "../include/raidjit.h"
#include "../src/plan.h"
#include "../src/jit.h"
#include "../src/cpu.h"
#include "../src/gf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

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

enum { OC_BUILT = 0, OC_REJECTED = 1, OC_ABORTED = 2 };

/* Synthetic fully-featured CPU (same pattern as test_twin_audit.c). */
static void fill_cpu(rj_cpu_info *cpu)
{
    memset(cpu, 0, sizeof(*cpu));
#if defined(__x86_64__)
    cpu->sse2 = cpu->ssse3 = 1;
    cpu->avx = cpu->avx2 = 1;
    cpu->avx512f = cpu->avx512bw = cpu->avx512vl = 1;
    cpu->gfni = 1;
    cpu->vbmi = 1;
    snprintf(cpu->brand, sizeof(cpu->brand), "pinnedneg-avx512");
#elif defined(__aarch64__)
    cpu->neon = 1;
    cpu->sve = 1;
    cpu->sve2 = 1;
    cpu->sve_vl = 16;
    snprintf(cpu->brand, sizeof(cpu->brand), "pinnedneg-sve2");
#elif defined(__riscv)
    cpu->rvv = 1;
    cpu->rvv_vlenb = 16;
    cpu->zvbc = 1;
    snprintf(cpu->brand, sizeof(cpu->brand), "pinnedneg-rvv");
#endif
}

/* Fork-isolated single build. Returns OC_BUILT / OC_REJECTED /
 * OC_ABORTED. The child never returns: it _exits. */
static int build_isolated(const plan_t *pl, const cand_t *cand,
                          const rj_cpu_info *cpu)
{
    fflush(stdout);
    fflush(stderr);
    pid_t pid = fork();
    if (pid < 0) {
        printf("FAIL: fork failed\n");
        return OC_ABORTED;
    }
    if (pid == 0) {
        rj_options o;
        rj_options_default(&o);
        o.benchmark = 0;
        version_t v;
        memset(&v, 0, sizeof(v));
        char err[256];
        int rc = jit_build_version(pl, cand, cpu, &o, &v, err, sizeof(err));
        if (rc == 0)
            jit_free_version(&v);
        _exit(rc == 0 ? OC_BUILT : OC_REJECTED);
    }
    int status = 0;
    while (waitpid(pid, &status, 0) < 0)
        ;
    if (WIFSIGNALED(status))
        return OC_ABORTED;
    if (WIFEXITED(status))
        return WEXITSTATUS(status);
    return OC_ABORTED;
}

static const char *oc_name(int oc)
{
    switch (oc) {
    case OC_BUILT: return "BUILT";
    case OC_REJECTED: return "REJECTED";
    case OC_ABORTED: return "ABORTED";
    default: return "???";
    }
}

/* A base valid candidate for the given plan kind, to be mutated with
 * invalid twin flags. Uses the first plan_candidates output as the
 * template (guaranteed valid). */
static int get_base_cand(const plan_t *pl, const rj_cpu_info *cpu,
                         cand_t *out)
{
    cand_t cands[64];
    int n = plan_candidates(pl, cpu, cands, 64, 1, 64);
    if (n <= 0)
        return -1;
    *out = cands[0];
    return 0;
}

/* Run one negative case: build with the mutated candidate, expect
 * REJECTED (graceful error), fail on ABORTED (assert trip) or BUILT
 * (gate missing -- invalid combo was accepted). */
static void negative_case(const plan_t *pl, const cand_t *base,
                          const rj_cpu_info *cpu, const char *desc,
                          void (*mutate)(cand_t *))
{
    cand_t c = *base;
    mutate(&c);
    int oc = build_isolated(pl, &c, cpu);
    if (oc == OC_ABORTED) {
        printf("FAIL: %-55s -> ABORTED (assert tripped!)\n", desc);
        failures++;
    } else if (oc == OC_BUILT) {
        printf("FAIL: %-55s -> BUILT (invalid combo accepted!)\n", desc);
        failures++;
    } else {
        printf("ok:   %-55s -> REJECTED\n", desc);
    }
    (void)oc_name; /* silence unused in release */
}

/* --- Mutators: each sets one invalid twin combination --- */

#if defined(__x86_64__)
static void mut_maskedtail_mulacc_nibble(cand_t *c)
{
    /* maskedtail on P_MULACC/nibble: needs P_XOR plain or GFNI. */
    c->maskedtail = 1;
    c->isa = ISA_AVX512;
    c->mul = MUL_NIBBLE;
    c->ternlog = 0;
    c->xorsched = 0;
}
static void mut_ternlog_avx2(cand_t *c)
{
    /* ternlog on AVX2: needs AVX-512 (CRITICAL-1). */
    c->ternlog = 1;
    c->isa = ISA_AVX2;
}
static void mut_ternlog_sse(cand_t *c)
{
    c->ternlog = 1;
    c->isa = ISA_SSE;
}
static void mut_remat_mulacc(cand_t *c)
{
    /* remat on non-P_HORNER. */
    c->remat = 1;
}
static void mut_maskedtail_notail(cand_t *c)
{
    c->maskedtail = 1;
    c->notail = 1;
    c->isa = ISA_AVX512;
}
static void mut_maskedtail_avx2(cand_t *c)
{
    c->maskedtail = 1;
    c->isa = ISA_AVX2;
}
static void mut_permi2b_avx2(cand_t *c)
{
    c->permi2b = 1;
    c->isa = ISA_AVX2;
    c->mul = MUL_NIBBLE;
}
static void mut_mshift_avx2(cand_t *c)
{
    c->mshift = 1;
    c->isa = ISA_AVX2;
    c->mul = MUL_NIBBLE;
}
static void mut_pipe_xor(cand_t *c)
{
    /* pipe on P_XOR: needs P_MULACC/P_HORNER. Set kind via plan;
     * here just force the pipe flag on a P_XOR base. */
    c->pipe = 1;
    c->mul = MUL_NIBBLE;
}
static void mut_xorbit_xor(cand_t *c)
{
    /* xorbit on P_XOR: needs GF-multiply rows. */
    c->xorbit = 1;
    c->mul = MUL_NIBBLE;
}
static void mut_xorbit_twin_conflict(cand_t *c)
{
    /* xorbit combined with ternlog: excluded. */
    c->xorbit = 1;
    c->ternlog = 1;
    c->mul = MUL_NIBBLE;
    c->isa = ISA_AVX512;
}
#endif /* __x86_64__ */

int main(void)
{
    char err[512];
    rj_cpu_info cpu;
    fill_cpu(&cpu);

    /* Build one plan per kind. */
    struct {
        const rj_topology *topo;
        const char *name;
    } plan_cases[] = {
        { rj_topo_raid5(4), "P_XOR raid5(4)" },
        { NULL, NULL }, /* placeholder for custom P_MULACC */
        { NULL, NULL }, /* placeholder for vandermonde P_HORNER */
    };
    /* P_MULACC custom(4,2) */
    uint8_t m[8] = {5, 7, 3, 9, 2, 4, 6, 8};
    plan_cases[1].topo = rj_topo_custom(4, 2, m);
    plan_cases[1].name = "P_MULACC custom(4,2)";
    /* P_HORNER via vandermonde not directly available; use raid6 RS
     * which typically yields P_HORNER or P_MULACC -- check kind. */

    int nplans = 0;
    plan_t plans[3];
    topo_int_t tis[3];
    const char *pnames[3];

    for (int i = 0; i < 3; i++) {
        if (!plan_cases[i].topo)
            continue;
        memset(&tis[nplans], 0, sizeof(tis[nplans]));
        memset(&plans[nplans], 0, sizeof(plans[nplans]));
        if (topo_int_build(&tis[nplans], plan_cases[i].topo,
                           err, sizeof(err)) != 0) {
            printf("FAIL: topo build %s: %s\n", plan_cases[i].name, err);
            failures++;
            continue;
        }
        rj_op op = rj_op_encode();
        if (plan_build(&tis[nplans], &op, plan_cases[i].name,
                       &plans[nplans], err, sizeof(err)) != 0) {
            printf("FAIL: plan build %s: %s\n", plan_cases[i].name, err);
            topo_int_free(&tis[nplans]);
            failures++;
            continue;
        }
        pnames[nplans] = plan_cases[i].name;
        nplans++;
    }

    if (nplans == 0) {
        printf("FAIL: no plans built\n");
        return 1;
    }

    /* For each plan, get a base candidate and run the negative cases
     * that apply to its kind. */
    for (int p = 0; p < nplans; p++) {
        cand_t base;
        if (get_base_cand(&plans[p], &cpu, &base) != 0) {
            printf("SKIP: no candidates for %s\n", pnames[p]);
            continue;
        }
        printf("== plan %s (kind=%d) ==\n", pnames[p], plans[p].kind);

#if defined(__x86_64__)
        /* maskedtail on P_MULACC/nibble: only invalid when the plan
         * kind is not P_XOR-plain and not GFNI-multiply. */
        if (plans[p].kind == P_MULACC)
            negative_case(&plans[p], &base, &cpu,
                          "maskedtail on P_MULACC/nibble",
                          mut_maskedtail_mulacc_nibble);

        /* ternlog on non-AVX512: invalid for every plan kind. */
        negative_case(&plans[p], &base, &cpu,
                      "ternlog on AVX2",
                      mut_ternlog_avx2);
        negative_case(&plans[p], &base, &cpu,
                      "ternlog on SSE",
                      mut_ternlog_sse);

        /* remat on non-P_HORNER. */
        if (plans[p].kind != P_HORNER)
            negative_case(&plans[p], &base, &cpu,
                          "remat on non-P_HORNER",
                          mut_remat_mulacc);

        /* maskedtail+notail: disjoint. */
        negative_case(&plans[p], &base, &cpu,
                      "maskedtail+notail",
                      mut_maskedtail_notail);

        /* maskedtail on AVX2: needs AVX-512. */
        negative_case(&plans[p], &base, &cpu,
                      "maskedtail on AVX2",
                      mut_maskedtail_avx2);

        /* permi2b on AVX2: needs AVX-512. */
        if (plans[p].kind == P_MULACC || plans[p].kind == P_DELTA)
            negative_case(&plans[p], &base, &cpu,
                          "permi2b on AVX2",
                          mut_permi2b_avx2);

        /* mshift on AVX2: needs AVX-512. */
        if (plans[p].kind == P_MULACC || plans[p].kind == P_DELTA ||
            plans[p].kind == P_HORNER)
            negative_case(&plans[p], &base, &cpu,
                          "mshift on AVX2",
                          mut_mshift_avx2);

        /* pipe on P_XOR: needs P_MULACC/P_HORNER. */
        if (plans[p].kind == P_XOR)
            negative_case(&plans[p], &base, &cpu,
                          "pipe on P_XOR",
                          mut_pipe_xor);

        /* xorbit on P_XOR: needs GF-multiply rows. */
        if (plans[p].kind == P_XOR)
            negative_case(&plans[p], &base, &cpu,
                          "xorbit on P_XOR",
                          mut_xorbit_xor);

        /* xorbit+ternlog: excluded combination. */
        if (plans[p].kind == P_MULACC || plans[p].kind == P_DELTA ||
            plans[p].kind == P_HORNER)
            negative_case(&plans[p], &base, &cpu,
                          "xorbit+ternlog",
                          mut_xorbit_twin_conflict);
#endif /* __x86_64__ */
    }

    for (int p = 0; p < nplans; p++) {
        plan_free(&plans[p]);
        topo_int_free(&tis[p]);
    }

    if (failures == 0)
        printf("\nPASS: all invalid pinned candidates rejected cleanly\n");
    else
        printf("\nFAIL: %d case(s) aborted or accepted\n", failures);
    return failures != 0;
}
