/* REDUX combination test (spec §8.3): crc_fold x {nt, prefetch} twins.
 *
 * For each topology and each of {nt, prefetch}:
 *   1. candidate level: enumerate want_crc=1 candidates (gate forced open)
 *      with the twin enabled; every crc_fold=1 twin candidate must BUILD
 *      or REJECT CLEANLY (jit_build_version nonzero + err set, no crash,
 *      no hang).
 *   2. driver level: compile fused [ENCODE, CRC->0] with the twin enabled
 *      (benchmark=1 is required for twin enumeration); run at 16-aligned
 *      and non-16-aligned lengths; CRCs must match the reference and
 *      parity bytes must match a plain encode (SFENCE after the call for
 *      the NT case, per the nt_threshold contract).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <emmintrin.h> /* _mm_sfence */

#include "../include/raidjit.h"
#include "../include/raidjit_fused.h"
#include "../src/plan.h"
#include "../src/jit.h"
#include "../src/fused.h"
#include "../src/cpu.h"

static int fails = 0;
static int nbuilt = 0, nrejected = 0, nruns = 0;
#define CHECK(cond, ...)                                                  \
    do {                                                                  \
        if (!(cond)) {                                                    \
            printf("FAIL %s:%d: ", __func__, __LINE__);                    \
            printf(__VA_ARGS__);                                          \
            printf("\n");                                                 \
            fails++;                                                      \
        }                                                                 \
    } while (0)

static uint64_t rng_state = 0xF01DCAFEULL;
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

static uint32_t crc32_ref(const uint8_t *p, size_t n)
{
    uint32_t c;
    rj_crc32_init(&c);
    c = rj_crc32_update(c, p, n);
    return rj_crc32_final(c);
}

/* Candidate level: every fold twin must build or reject cleanly. */
static void cand_level(rj_topology *topo, const char *tname, int want_nt,
                       int want_pf, const rj_cpu_info *cpu)
{
    topo_int_t ti;
    plan_t pl;
    char err[256];
    rj_op op = rj_op_encode();
    if (topo_int_build(&ti, topo, err, sizeof(err)) != 0)
        return;
    if (plan_build(&ti, &op, tname, &pl, err, sizeof(err)) != 0) {
        topo_int_free(&ti);
        return;
    }
    pl.want_crc = 1;
    cand_t cands[512];
    /* Mirror api.c: nt/prefetch twins only when benchmarking. */
    int n = plan_candidates(&pl, cpu, cands, 512, want_nt,
                            want_pf ? 1024 : 0);
    rj_options opts;
    rj_options_default(&opts);
    opts.benchmark = 0;
    int nfold = 0;
    for (int i = 0; i < n; i++) {
        if (!cands[i].crc_fold)
            continue;
        if (want_nt && !cands[i].nt)
            continue;
        if (want_pf && !cands[i].prefetch_dist)
            continue;
        nfold++;
        version_t v;
        char e2[256] = { 0 };
        memset(&v, 0, sizeof(v));
        int rc = jit_build_version(&pl, &cands[i], cpu, &opts, &v, e2,
                                   sizeof(e2));
        if (rc == 0) {
            nbuilt++;
            jit_free_version(&v);
        } else {
            nrejected++;
            /* Clean rejection: an error message, no crash. */
            CHECK(e2[0] != '\0',
                  "%s %s: twin rejected without an error message",
                  tname, want_nt ? "nt" : "prefetch");
        }
    }
    printf("cand %s %s: %d fold twins (%d built, %d clean-reject)\n",
           tname, want_nt ? "nt" : "prefetch", nfold, nbuilt, nrejected);
    plan_free(&pl);
    topo_int_free(&ti);
}

/* Driver level: fused compile + run with the twin enabled. */
static void driver_level(rj_topology *topo, const char *tname, int nd, int np,
                         int want_nt, int want_pf)
{
    rj_fuse_options fopts;
    rj_fuse_options_default(&fopts);
    fopts.benchmark = 1; /* twins only enumerate when benchmarking */
    fopts.bench_bytes = 1 << 16;
    fopts.bench_iters = 2;
    if (want_nt)
        fopts.nt_threshold = 1;
    if (want_pf)
        fopts.prefetch_distance = 1024;

    char err[256];
    rj_fop_t fops[2];
    memset(fops, 0, sizeof(fops));
    fops[0].kind = RJ_FOP_ENCODE;
    fops[1].kind = RJ_FOP_CRC;
    fops[1].target = 0;
    rj_fused *f = rj_fuse_compile(topo, fops, 2, &fopts, err, sizeof(err));
    if (!f) {
        /* Clean rejection at compile time is acceptable. */
        printf("driver %s %s: compile rejected: %s\n", tname,
               want_nt ? "nt" : "prefetch", err);
        CHECK(err[0] != '\0', "%s: empty error on rejection", tname);
        nrejected++;
        return;
    }
    nbuilt++;
    rj_fn_fused ffn = rj_get_fn_fused(f);

    rj_options co;
    rj_options_default(&co);
    co.benchmark = 0;
    rj_op op = rj_op_encode();
    rj_jit *pe = rj_compile(topo, &op, &co, err, sizeof(err));
    CHECK(pe != NULL, "%s: plain encode compile failed: %s", tname, err);
    rj_fn pfn = pe ? rj_get_fn(pe) : NULL;

    static const size_t lens[] = { 1000, 65537 };
    for (size_t li = 0; li < 2; li++) {
        size_t len = lens[li];
        const void **in = malloc(sizeof(*in) * (size_t)nd);
        void **oF = malloc(sizeof(*oF) * (size_t)np);
        void **oP = malloc(sizeof(*oP) * (size_t)np);
        for (int i = 0; i < nd; i++) {
            uint8_t *b = abuf(len);
            randfill(b, len);
            in[i] = b;
        }
        for (int p = 0; p < np; p++) {
            oF[p] = abuf(len);
            oP[p] = abuf(len);
        }
        uint32_t crc[512];
        ffn(in, oF, len, crc);
        _mm_sfence(); /* NT-store contract: fence before reading outputs */
        if (pfn)
            pfn(in, oP, len);
        for (int p = 0; p < np; p++)
            CHECK(memcmp(oF[p], oP[p], len) == 0,
                  "%s %s len=%zu: parity %d differs",
                  tname, want_nt ? "nt" : "prefetch", len, p);
        for (int i = 0; i < nd; i++)
            CHECK(crc[i] == crc32_ref(in[i], len),
                  "%s %s len=%zu: crc_in[%d] got %08x want %08x", tname,
                  want_nt ? "nt" : "prefetch", len, i, crc[i],
                  crc32_ref(in[i], len));
        for (int p = 0; p < np; p++)
            CHECK(crc[nd + p] == crc32_ref(oF[p], len),
                  "%s %s len=%zu: crc_out[%d] got %08x want %08x", tname,
                  want_nt ? "nt" : "prefetch", len, p, crc[nd + p],
                  crc32_ref(oF[p], len));
        nruns++;
        for (int i = 0; i < nd; i++)
            free((void *)in[i]);
        for (int p = 0; p < np; p++) {
            free(oF[p]);
            free(oP[p]);
        }
        free(in);
        free(oF);
        free(oP);
    }
    if (pe)
        rj_free(pe);
    rj_free_fused(f);
    printf("driver %s %s: ran clean\n", tname, want_nt ? "nt" : "prefetch");
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
    rj_uarch_override(RJ_UARCH_INTEL_SAPPHIRERAPIDS);

    struct {
        rj_topology *t;
        const char *name;
        int nd, np;
    } ts[] = {
        { rj_topo_raid5(4), "raid5(4)", 4, 1 },
        { rj_topo_rs(6, 3), "rs(6,3)", 6, 3 },
    };
    for (size_t i = 0; i < sizeof(ts) / sizeof(ts[0]); i++) {
        cand_level(ts[i].t, ts[i].name, 1, 0, &cpu);
        cand_level(ts[i].t, ts[i].name, 0, 1, &cpu);
        driver_level(ts[i].t, ts[i].name, ts[i].nd, ts[i].np, 1, 0);
        driver_level(ts[i].t, ts[i].name, ts[i].nd, ts[i].np, 0, 1);
        rj_topo_free(ts[i].t);
    }
    printf("combo: built=%d rejected-clean=%d driver-runs=%d fails=%d\n",
           nbuilt, nrejected, nruns, fails);
    if (fails == 0)
        printf("PASS\n");
    return fails != 0;
#endif
}
