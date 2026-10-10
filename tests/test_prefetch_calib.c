/* I4: calibrated prefetch distance tests.
 *
 * 1. Probe sanity: the real one-time probe returns valid, positive
 *    latency and bandwidth. (Runs first: the inject override below is
 *    sticky for process lifetime.)
 * 2. Distance formula: hand-computed cases against an injected
 *    calibration (L=100ns, B=20GB/s -> 2000 bytes in flight), covering
 *    the base formula, the iters clamp [1,64], SVE/RVV vector-length
 *    handling, RVV encodability rounding + below-minimum skip, the
 *    fail-closed invalid-calibration path, and degenerate geometry.
 * 3. Option resolution: benchmark off -> 0, manual override wins,
 *    unset+benchmark -> RJ_PREFETCH_AUTO.
 * 4. Planner wiring: RJ_PREFETCH_AUTO enumerates one prefetch twin per
 *    base candidate with structurally sound distances; the distance-0
 *    base twin is still enumerated first; a literal 0 still means no
 *    twins; a manual distance still overrides every twin.
 * 5. rj_init is idempotent and leaves a valid calibration behind.
 */
#include "../include/raidjit.h"
#include "../src/plan.h"
#include "../src/cpu.h"
#include "../src/prefetch_calib.h"

#include <stdio.h>
#include <string.h>

static int fails;
#define CHECK(c, ...) do {                                             \
    if (!(c)) {                                                        \
        printf("FAIL "); printf(__VA_ARGS__); printf("\n"); fails++;   \
    }                                                                  \
} while (0)

/* in_flight for the injected calibration: 100 ns * 20 GB/s = 2000 B. */
#define INJ_LAT 100.0
#define INJ_BW 20.0

static void test_probe_sane(void)
{
    const prefetch_calib_t *c = prefetch_calib_ensure();
    CHECK(c != NULL, "ensure returned NULL");
    CHECK(c->valid, "probe reported invalid on this machine");
    CHECK(c->latency_ns >= 1.0 && c->latency_ns <= 1e5,
          "latency_ns=%f outside [1ns, 100us]", c->latency_ns);
    CHECK(c->bandwidth_gbs > 0.0 && c->bandwidth_gbs <= 1e4,
          "bandwidth_gbs=%f outside (0, 10TB/s]", c->bandwidth_gbs);
    /* Second call: cached, same pointer, no re-probe. */
    CHECK(prefetch_calib_ensure() == c, "ensure not idempotent");
    printf("note: probe latency=%.1fns bandwidth=%.1fGB/s\n",
           c->latency_ns, c->bandwidth_gbs);
}

static size_t dist_for(isa_t isa, int U, int lmul, int n_in,
                       int sve_vl, int rvv_vlen)
{
    plan_t pl;
    memset(&pl, 0, sizeof(pl));
    pl.n_in = n_in;
    cand_t cand;
    memset(&cand, 0, sizeof(cand));
    cand.isa = isa;
    cand.U = U;
    cand.lmul = lmul;
    rj_cpu_info cpu;
    memset(&cpu, 0, sizeof(cpu));
    cpu.sve_vl = sve_vl;
    cpu.rvv_vlen = rvv_vlen;
    return prefetch_calib_distance_for(&pl, &cand, &cpu);
}

static void test_formula(void)
{
    prefetch_calib_inject(INJ_LAT, INJ_BW, 1);

    /* Base: stride = 4*64*6 = 1536 B/iter; iters = ceil(2000/1536) = 2;
     * dist = 2*4*64 = 512. */
    CHECK(dist_for(ISA_AVX512, 4, 0, 6, 0) == 512,
          "base formula: got %zu, want 512",
          dist_for(ISA_AVX512, 4, 0, 6, 0));

    /* Iters clamp: L=100us -> 2e6 B in flight; ceil(2e6/1536) = 1303 ->
     * 64; dist = 64*4*64 = 16384. */
    prefetch_calib_inject(100000.0, 20.0, 1);
    CHECK(dist_for(ISA_AVX512, 4, 0, 6, 0) == 16384,
          "iters clamp: got %zu, want 16384",
          dist_for(ISA_AVX512, 4, 0, 6, 0));

    /* Iters floor: L=1ns,B=1GB/s -> 1 B; ceil(1/1536) = 1;
     * dist = 1*4*64 = 256. */
    prefetch_calib_inject(1.0, 1.0, 1);
    CHECK(dist_for(ISA_AVX512, 4, 0, 6, 0) == 256,
          "iters floor: got %zu, want 256",
          dist_for(ISA_AVX512, 4, 0, 6, 0));

    prefetch_calib_inject(INJ_LAT, INJ_BW, 1);

    /* SVE takes VL from cpu->sve_vl: stride = 2*32*4 = 256;
     * iters = ceil(2000/256) = 8; dist = 8*2*32 = 512. */
    CHECK(dist_for(ISA_SVE, 2, 0, 4, 32, 0) == 512,
          "sve vl: got %zu, want 512",
          dist_for(ISA_SVE, 2, 0, 4, 32, 0));

    /* SVE with sve_vl=0 falls back to 16: stride = 4*16*6 = 384;
     * iters = ceil(2000/384) = 6; dist = 6*4*16 = 384. */
    CHECK(dist_for(ISA_SVE, 4, 0, 6, 0) == 384,
          "sve vl fallback: got %zu, want 384",
          dist_for(ISA_SVE, 4, 0, 6, 0));

    /* RVV: vl = lmul*vlen = 16; stride = 1*16*6 = 96;
     * iters = ceil(2000/96) = 21; dist = 21*16 = 336 ->
     * round down to 32 -> 320. */
    CHECK(dist_for(ISA_RVV, 1, 1, 6, 0, 16) == 320,
          "rvv rounding: got %zu, want 320",
          dist_for(ISA_RVV, 1, 1, 6, 0, 16));

    /* RVV lmul scales vl: vl = 32; stride = 192;
     * iters = ceil(2000/192) = 11; dist = 11*32 = 352 (exact). */
    CHECK(dist_for(ISA_RVV, 1, 2, 6, 0, 16) == 352,
          "rvv lmul: got %zu, want 352",
          dist_for(ISA_RVV, 1, 2, 6, 0, 16));

    /* RVV below the 32-byte encodable minimum declines the twin. */
    prefetch_calib_inject(1.0, 1.0, 1);
    CHECK(dist_for(ISA_RVV, 1, 1, 6, 0, 16) == 0,
          "rvv below-minimum: got %zu, want 0",
          dist_for(ISA_RVV, 1, 1, 6, 0, 16));
    prefetch_calib_inject(INJ_LAT, INJ_BW, 1);

    /* Degenerate geometry declines the twin. */
    CHECK(dist_for(ISA_AVX512, 4, 0, 0, 0) == 0,
          "n_in=0: got %zu, want 0",
          dist_for(ISA_AVX512, 4, 0, 0, 0));
    CHECK(dist_for(ISA_AVX512, 0, 0, 6, 0) == 0,
          "U=0: got %zu, want 0", dist_for(ISA_AVX512, 0, 0, 6, 0));

    /* Invalid calibration fails closed: no twin. */
    prefetch_calib_inject(0.0, 0.0, 0);
    CHECK(dist_for(ISA_AVX512, 4, 0, 6, 0) == 0,
          "invalid calib: got %zu, want 0",
          dist_for(ISA_AVX512, 4, 0, 6, 0));

    /* Restore a valid fake for the enumeration test below. */
    prefetch_calib_inject(INJ_LAT, INJ_BW, 1);
}

static void test_dist_arg(void)
{
    rj_options o;
    rj_options_default(&o); /* benchmark=1, prefetch_distance=0 */
    CHECK(prefetch_calib_dist_arg(&o) == RJ_PREFETCH_AUTO,
          "unset+benchmark: want RJ_PREFETCH_AUTO, got %zu",
          prefetch_calib_dist_arg(&o));
    o.prefetch_distance = 4096;
    CHECK(prefetch_calib_dist_arg(&o) == 4096,
          "manual override: want 4096, got %zu",
          prefetch_calib_dist_arg(&o));
    o.benchmark = 0;
    CHECK(prefetch_calib_dist_arg(&o) == 0,
          "benchmark off: want 0, got %zu", prefetch_calib_dist_arg(&o));
    o.prefetch_distance = 0;
    CHECK(prefetch_calib_dist_arg(&o) == 0,
          "benchmark off, unset: want 0, got %zu",
          prefetch_calib_dist_arg(&o));
}

static void test_auto_enumeration(const rj_cpu_info *cpu)
{
    char err[256];
    rj_topology *t = rj_topo_rs(6, 3);
    topo_int_t ti;
    CHECK(topo_int_build(&ti, t, err, sizeof(err)) == 0, "topo: %s", err);
    rj_op op = rj_op_encode();
    plan_t pl;
    CHECK(plan_build(&ti, &op, "rs(6,3)", &pl, err, sizeof(err)) == 0,
          "plan: %s", err);
    if (fails)
        goto out;

    cand_t c0[512], ca[512], cm[512];
    int n0 = plan_candidates(&pl, cpu, c0, 512, 0, 0);
    int na = plan_candidates(&pl, cpu, ca, 512, 0, RJ_PREFETCH_AUTO);
    int nm = plan_candidates(&pl, cpu, cm, 512, 0, 4096);
    CHECK(n0 > 0 && na > 0 && nm > 0, "counts %d/%d/%d", n0, na, nm);

    /* Literal 0 still means no prefetch twins (existing contract). */
    for (int i = 0; i < n0; i++)
        CHECK(c0[i].prefetch_dist == 0, "dist 0 enumerated a twin");

    /* AUTO: one prefetch twin per base candidate (x86: every distance
     * encodable, so no declines), appended after the base list. */
    int pfa = 0, pfm = 0;
    for (int i = 0; i < na; i++)
        if (ca[i].prefetch_dist)
            pfa++;
    for (int i = 0; i < nm; i++)
        if (cm[i].prefetch_dist)
            pfm++;
#if defined(__riscv)
    /* RVV without Zicbop declines every twin; only assert structure. */
    (void)pfa;
#else
    CHECK(pfa == n0, "AUTO: want one twin per base (%d vs %d)", pfa, n0);
#endif
    CHECK(pfm == n0, "manual: want one twin per base (%d vs %d)", pfm, n0);

    /* Distance 0 stays a candidate: the base twins come first. */
    CHECK(ca[0].prefetch_dist == 0, "AUTO: first candidate not distance-0");
    for (int i = 0; i < n0; i++)
        CHECK(ca[i].prefetch_dist == 0, "AUTO: base twin %d not distance-0",
              i);

    /* Every AUTO twin matches a base twin on all other axes, carries a
     * structurally sound analytical distance (multiple of the
     * candidate's per-stream stride, iters in [1,64]), and every
     * manual twin carries exactly the override. */
    for (int i = n0; i < na && !fails; i++) {
        if (!ca[i].prefetch_dist)
            continue;
        int found = 0;
        for (int k = 0; k < n0; k++) {
            if (c0[k].isa == ca[i].isa && c0[k].mul == ca[i].mul &&
                c0[k].U == ca[i].U && c0[k].nt == ca[i].nt &&
                c0[k].ternlog == ca[i].ternlog &&
                c0[k].remat == ca[i].remat) {
                found = 1;
                break;
            }
        }
        CHECK(found, "AUTO twin %d has no base twin", i);
        int vl = isa_vl(ca[i].isa);
        long unit = (long)ca[i].U * vl; /* per-stream stride */
        CHECK(unit > 0 && ca[i].prefetch_dist % (size_t)unit == 0,
              "AUTO twin %d: dist %zu not a multiple of stride %ld", i,
              ca[i].prefetch_dist, unit);
        long iters = (long)(ca[i].prefetch_dist / (size_t)unit);
        CHECK(iters >= 1 && iters <= 64,
              "AUTO twin %d: iters %ld outside [1,64]", i, iters);
    }
    for (int i = 0; i < nm; i++)
        if (cm[i].prefetch_dist)
            CHECK(cm[i].prefetch_dist == 4096,
                  "manual twin %d: want 4096, got %zu", i,
                  cm[i].prefetch_dist);

out:
    plan_free(&pl);
    rj_op_free(&op);
    topo_int_free(&ti);
    rj_topo_free(t);
}

static void test_rj_init(void)
{
    rj_init();
    rj_init();
    /* Injected calibration still served (rj_init must not clobber it). */
    const prefetch_calib_t *c = prefetch_calib_ensure();
    CHECK(c->valid && c->latency_ns == INJ_LAT && c->bandwidth_gbs == INJ_BW,
          "rj_init disturbed the calibration");
}

int main(void)
{
    /* Real-probe checks first: injection below is process-sticky. */
    test_probe_sane();
    test_formula();
    test_dist_arg();

    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    test_auto_enumeration(&cpu);

    test_rj_init();

    if (!fails)
        printf("prefetch_calib: all OK\n");
    return fails != 0;
}
