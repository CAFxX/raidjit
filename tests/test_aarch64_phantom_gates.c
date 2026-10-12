/* Regression test for LEAD L3: x86-only twin pins must fail closed at the
 * AArch64 backend boundary.
 *
 * src/aarch64/jit.c never references cand->permi2b, cand->mshift,
 * cand->ternlog, cand->maskedtail, or cand->mul (MUL_GFNI/MUL_ZVBC) -- a
 * direct jit_build_version() caller pinning any of them got a successful
 * build of the plain nibble body with the pin silently dropped (a phantom
 * twin: v->cand records a twin whose code path was never taken). The
 * planner-level gates (planid_cand_valid, api.c permi2b duel eligibility)
 * already reject these for non-AVX512 ISAs; this test pins the boundary
 * the planner does not cover: the direct backend call.
 *
 * Cases: permi2b=1, mshift=1, ternlog=1, maskedtail=1, mul=MUL_GFNI,
 * mul=MUL_ZVBC must each return -1 with a non-empty error. Control: the
 * plain candidate must build (rc 0). planid_cand_valid must reject
 * permi2b/mshift/ternlog/maskedtail on ISA_NEON (public plan-ID entry
 * path).
 *
 * Runs under QEMU, e.g.
 *   QEMU_LD_PREFIX=/usr/aarch64-linux-gnu qemu-aarch64 -cpu max \
 *       ./build-aarch64/test_aarch64_phantom_gates
 * The rejection cases return before any ISA-specific code, so they are
 * valid without NEON/SVE; the control build needs NEON and SKIPs without
 * it.
 */
#include "../include/raidjit.h"
#include "../src/plan.h"
#include "../src/jit.h"
#include "../src/cpu.h"
#include "../src/planid.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__aarch64__)

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

/* One phantom-pin case: set the pin on a copy of the base candidate and
 * demand a clean backend rejection. */
static void expect_reject(const plan_t *pl, const cand_t *base,
                          const rj_cpu_info *cpu, const char *pin,
                          void (*set_pin)(cand_t *))
{
    cand_t c = *base;
    set_pin(&c);
    rj_options o;
    rj_options_default(&o);
    o.benchmark = 0;
    version_t v;
    memset(&v, 0, sizeof(v));
    char err[256];
    memset(err, 0, sizeof(err));
    int rc = jit_build_version(pl, &c, cpu, &o, &v, err, sizeof(err));
    CHECK(rc == -1, "%s: expected clean rejection (rc -1), got rc=%d%s%s",
          pin, rc, err[0] ? " err=" : "", err);
    CHECK(err[0] != '\0', "%s: rejection carried no error message", pin);
    if (rc == 0)
        jit_free_version(&v);
    else
        printf("ok phantom %s rejected: %s\n", pin, err);
}

static void set_permi2b(cand_t *c) { c->permi2b = 1; }
static void set_mshift(cand_t *c) { c->mshift = 1; }
static void set_ternlog(cand_t *c) { c->ternlog = 1; }
static void set_maskedtail(cand_t *c) { c->maskedtail = 1; }
static void set_gfni(cand_t *c) { c->mul = MUL_GFNI; }
static void set_zvbc(cand_t *c) { c->mul = MUL_ZVBC; }

/* planid_cand_valid must reject the same pins on ISA_NEON (the public
 * pinned-plan-ID entry path). */
static void expect_planid_reject(const plan_t *pl, const cand_t *base,
                                 const char *pin, void (*set_pin)(cand_t *))
{
    cand_t c = *base;
    set_pin(&c);
    char err[256];
    int rc = planid_cand_valid(pl, &c, err, sizeof(err));
    CHECK(rc != 0, "planid %s: expected rejection, accepted", pin);
    if (rc != 0)
        printf("ok planid %s rejected: %s\n", pin, err);
}

int main(void)
{
    /* custom(4,2) encode: P_MULACC (the mshift/permi2b twin's home kind). */
    uint8_t cmat[8] = { 1, 1, 1, 1, 1, 2, 4, 8 };
    rj_topology *t = rj_topo_custom(4, 2, cmat);
    CHECK(t != NULL, "rj_topo_custom failed");
    if (!t)
        return 1;
    rj_op op = rj_op_encode();
    topo_int_t ti;
    char err[256];
    if (topo_int_build(&ti, t, err, sizeof(err))) {
        printf("FAIL topo: %s\n", err);
        return 1;
    }
    plan_t pl;
    if (plan_build(&ti, &op, "phantom", &pl, err, sizeof(err))) {
        printf("FAIL plan: %s\n", err);
        return 1;
    }
    CHECK(pl.kind == P_MULACC, "expected P_MULACC, got kind %d", pl.kind);

    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);

    cand_t base;
    memset(&base, 0, sizeof(base));
    base.isa = ISA_NEON;
    base.mul = MUL_NIBBLE;
    base.U = 2;

    /* Direct-backend entry path: every x86/RVV-only pin fails closed. */
    expect_reject(&pl, &base, &cpu, "permi2b", set_permi2b);
    expect_reject(&pl, &base, &cpu, "mshift", set_mshift);
    expect_reject(&pl, &base, &cpu, "ternlog", set_ternlog);
    expect_reject(&pl, &base, &cpu, "maskedtail", set_maskedtail);
    expect_reject(&pl, &base, &cpu, "mul=GFNI", set_gfni);
    expect_reject(&pl, &base, &cpu, "mul=ZVBC", set_zvbc);

    /* Public plan-ID entry path: planid_cand_valid rejects the same pins. */
    expect_planid_reject(&pl, &base, "permi2b", set_permi2b);
    expect_planid_reject(&pl, &base, "mshift", set_mshift);
    expect_planid_reject(&pl, &base, "ternlog", set_ternlog);
    expect_planid_reject(&pl, &base, "maskedtail", set_maskedtail);

    /* Control: the plain candidate must build. */
    if (cpu.neon) {
        rj_options o;
        rj_options_default(&o);
        o.benchmark = 0;
        version_t v;
        memset(&v, 0, sizeof(v));
        int rc = jit_build_version(&pl, &base, &cpu, &o, &v, err,
                                   sizeof(err));
        CHECK(rc == 0, "control build failed: %s", err);
        if (rc == 0) {
            printf("ok control plain-neon build [%s]\n", v.desc);
            jit_free_version(&v);
        }
    } else {
        printf("SKIP control build: no NEON\n");
    }

    plan_free(&pl);
    topo_int_free(&ti);
    rj_op_free(&op);
    rj_topo_free(t);

    if (!failures)
        printf("PASS test_aarch64_phantom_gates\n");
    return failures != 0;
}

#else /* !defined(__aarch64__) */

int main(void)
{
    printf("SKIP test_aarch64_phantom_gates: not an aarch64 build\n");
    return 0;
}

#endif
