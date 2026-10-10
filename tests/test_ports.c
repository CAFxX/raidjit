/* test_ports: direct unit tests for the per-uarch port-pressure model.
 *
 * Tests rj_ports_bottleneck() on synthetic mixes with known outcomes,
 * rj_ports_for_uarch() lookup, and rj_ports_find_class(). These are
 * the missing direct unit tests flagged by peer review (spec §6.1).
 *
 * Fail-capable: every assertion pins a specific numeric outcome derived
 * from the .ports data files and uops.info primary-source values.
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "ports.h"
#include "cpu.h"

static int n_checks = 0;
static int n_failures = 0;

#define CHECK(cond, fmt, ...) do { \
    n_checks++; \
    if (!(cond)) { \
        n_failures++; \
        printf("FAIL [%d]: " fmt "\n", n_checks, ##__VA_ARGS__); \
    } \
} while (0)

#define CHECK_CLOSE(got, want, tol, name) do { \
    n_checks++; \
    double _g = (got), _w = (want); \
    if (fabs(_g - _w) > (tol)) { \
        n_failures++; \
        printf("FAIL [%d]: %s: got %.6f, want %.6f\n", n_checks, name, _g, _w); \
    } \
} while (0)

/* Helper: bottleneck for a single class with mix = 1 uop worth */
static double bottleneck_1uop(const rj_port_model_t *m, const rj_pm_class_t *c) {
    const rj_pm_class_t *cls[1] = { c };
    double mix[1] = { 1.0 / c->uops_per_instr };
    return rj_ports_bottleneck(m, cls, mix, 1);
}

int main(void) {
    /* --- Lookup tests --- */
    const rj_port_model_t *icl = rj_ports_for_uarch(RJ_UARCH_INTEL_ICELAKE);
    CHECK(icl != NULL, "ICL model exists");
    CHECK(icl->n_ports > 0, "ICL has ports");
    CHECK(icl->n_classes > 0, "ICL has classes");
    
    const rj_port_model_t *zen5 = rj_ports_for_uarch(RJ_UARCH_AMD_ZEN5);
    CHECK(zen5 != NULL, "Zen5 model exists");
    
    const rj_port_model_t *generic = rj_ports_for_uarch(RJ_UARCH_UNKNOWN);
    CHECK(generic != NULL, "generic fallback exists");
    
    /* --- Class lookup --- */
    const rj_pm_class_t *tl = rj_ports_find_class(icl, "ternlog");
    CHECK(tl != NULL, "ICL ternlog class found");
    CHECK(rj_ports_find_class(icl, "nonexistent") == NULL,
          "missing class returns NULL");
    
    /* --- ICL: uops.info verified values --- */
    /* VPTERNLOGD: 1*p05, TP 0.50 -> 0.5 cycles/uop */
    CHECK_CLOSE(bottleneck_1uop(icl, tl), 0.5, 0.01, "ICL ternlog p05");
    
    /* VGF2P8AFFINEQB/MULB: 1*p0, TP 1.00 -> 1.0 cycles/uop */
    const rj_pm_class_t *gf = rj_ports_find_class(icl, "gfni");
    CHECK(gf != NULL, "ICL gfni class found");
    CHECK_CLOSE(bottleneck_1uop(icl, gf), 1.0, 0.01, "ICL gfni p0");
    
    /* VPXORD: 1*p05 like ternlog */
    const rj_pm_class_t *xr = rj_ports_find_class(icl, "xor");
    CHECK(xr != NULL, "ICL xor class found");
    CHECK_CLOSE(bottleneck_1uop(icl, xr), 0.5, 0.01, "ICL xor p05");
    
    /* VPSHUFB: 1*p5, TP 1.00 -> 1.0 cycles/uop */
    const rj_pm_class_t *sh = rj_ports_find_class(icl, "shuffle");
    CHECK(sh != NULL, "ICL shuffle class found");
    CHECK_CLOSE(bottleneck_1uop(icl, sh), 1.0, 0.01, "ICL shuffle p5");
    
    /* --- Zen5: uops.info verified values --- */
    /* VPTERNLOGD: 1*FP0123, TP 0.25 -> 0.25 cycles/uop */
    tl = rj_ports_find_class(zen5, "ternlog");
    CHECK(tl != NULL, "Zen5 ternlog class found");
    CHECK_CLOSE(bottleneck_1uop(zen5, tl), 0.25, 0.01, "Zen5 ternlog FP0123");
    
    /* VPSHUFB: 1*FP12, TP 0.50 -> 0.5 cycles/uop */
    sh = rj_ports_find_class(zen5, "shuffle");
    CHECK(sh != NULL, "Zen5 shuffle class found");
    CHECK_CLOSE(bottleneck_1uop(zen5, sh), 0.5, 0.01, "Zen5 shuffle FP12");
    
    /* --- ICL pclmul (load-bearing for CRC folding) --- */
    /* VPCLMULQDQ zmm: 1*p5, TP 1.00 -> 1.0 cycles/uop */
    const rj_pm_class_t *pc = rj_ports_find_class(icl, "pclmul");
    CHECK(pc != NULL, "ICL pclmul class found");
    CHECK_CLOSE(bottleneck_1uop(icl, pc), 1.0, 0.01, "ICL pclmul p5");

    /* --- ICL load/store classes --- */
    const rj_pm_class_t *ld = rj_ports_find_class(icl, "load");
    CHECK(ld != NULL, "ICL load class found");
    CHECK_CLOSE(bottleneck_1uop(icl, ld), 0.5, 0.01, "ICL load p2/p3");
    const rj_pm_class_t *st = rj_ports_find_class(icl, "store");
    CHECK(st != NULL, "ICL store class found");
    CHECK_CLOSE(bottleneck_1uop(icl, st), 0.5, 0.01, "ICL store p4/p7");

    /* --- Zen4: uops.info verified values --- */
    const rj_port_model_t *zen4 = rj_ports_for_uarch(RJ_UARCH_AMD_ZEN4);
    CHECK(zen4 != NULL, "Zen4 model exists");

    /* Zen4 512-bit is double-pumped: uops_per_instr = 2.0 */
    tl = rj_ports_find_class(zen4, "ternlog");
    CHECK(tl != NULL, "Zen4 ternlog class found");
    CHECK_CLOSE(tl->uops_per_instr, 2.0, 0.01, "Zen4 ternlog 2 uops/instr");

    /* Zen4 gfni: 2 uops on FP0/FP1 (uops.info: 1*FP01) -> 0.5 cycles/uop.
     * Verifies the MAJOR-1 resolution (p0, not p01, on Intel; FP01 on Zen4). */
    gf = rj_ports_find_class(zen4, "gfni");
    CHECK(gf != NULL, "Zen4 gfni class found");
    CHECK_CLOSE(bottleneck_1uop(zen4, gf), 0.5, 0.01, "Zen4 gfni FP01");

    /* --- Zen5 gfni (MAJOR-1 resolution) --- */
    /* Zen5 gfni: 1 uop on FP1/FP2 (uops.info: 1*FP12) -> 0.5 cycles/uop */
    gf = rj_ports_find_class(zen5, "gfni");
    CHECK(gf != NULL, "Zen5 gfni class found");
    CHECK_CLOSE(bottleneck_1uop(zen5, gf), 0.5, 0.01, "Zen5 gfni FP12");

    /* --- SPR smoke test --- */
    const rj_port_model_t *spr = rj_ports_for_uarch(RJ_UARCH_INTEL_SAPPHIRERAPIDS);
    CHECK(spr != NULL, "SPR model exists");
    tl = rj_ports_find_class(spr, "ternlog");
    CHECK(tl != NULL, "SPR ternlog class found");
    CHECK_CLOSE(bottleneck_1uop(spr, tl), 0.5, 0.01, "SPR ternlog p05");

    /* --- SKX smoke test --- */
    const rj_port_model_t *skx = rj_ports_for_uarch(RJ_UARCH_INTEL_SKYLAKEX);
    CHECK(skx != NULL, "SKX model exists");
    sh = rj_ports_find_class(skx, "shuffle");
    CHECK(sh != NULL, "SKX shuffle class found");
    CHECK_CLOSE(bottleneck_1uop(skx, sh), 1.0, 0.01, "SKX shuffle p5");

    /* --- Edge cases --- */
    /* Empty mix -> 0 cycles */
    {
        const rj_pm_class_t *cls[1] = { tl };
        double mix[1] = { 0.0 };
        CHECK_CLOSE(rj_ports_bottleneck(icl, cls, mix, 1), 0.0, 1e-9,
                    "empty mix -> 0");
    }
    
    /* NULL model -> 0 (defensive) */
    {
        const rj_pm_class_t *cls[1] = { tl };
        double mix[1] = { 1.0 };
        CHECK_CLOSE(rj_ports_bottleneck(NULL, cls, mix, 1), 0.0, 1e-9,
                    "NULL model -> 0");
    }
    
    printf("\n%d checks, %d failures\n", n_checks, n_failures);
    return n_failures != 0;
}
