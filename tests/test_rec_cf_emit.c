/* Phase-2 emitter differential: rec_cf twin vs the generic MULACC path.
 *
 * For every (topology, failure set) where the planner emits a rec_cf
 * candidate twin, build the twin AND its rec_cf=0 base from the same
 * plan, run both on random data at vector/tail/mixed lengths, and
 * require bit-identical outputs. The generic path is validated by the
 * existing suite (test_correct, test_recover_closedform e2e); this
 * test proves the closed-form emitter computes the same function.
 *
 * Also red-green: flipping one byte of the cf_data must change the
 * twin's output (proves the emitter actually reads the closed-form
 * data and the differential is not vacuous).
 */
#include "../include/raidjit.h"
#include "../src/plan.h"
#include "../src/jit.h"
#include "../src/cpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, ...)                                                   \
    do {                                                                   \
        if (!(cond)) {                                                     \
            printf("FAIL %s:%d: ", __func__, __LINE__);                     \
            printf(__VA_ARGS__);                                           \
            printf("\n");                                                  \
            failures++;                                                    \
        }                                                                  \
    } while (0)

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

static rj_cpu_info cpu;
static rj_options opts;
static long n_twins_tested = 0, n_e1 = 0, n_e2 = 0;
static long n_e1_finalmul = 0; /* e=1 twins with cinv != 1 (final multiply
                                * exercised) */
static long n_e2_b0 = 0, n_e2_c0 = 0; /* e=2 k-trick edge cases */

/* Run one (twin, base) pair on random data at several lengths; outputs
 * must be bit-identical. Returns 0 on success. */
static int diff_pair(const plan_t *pl, const cand_t *twin, const char *tname,
                     const int *F, int nf)
{
    char err[256];
    cand_t base = *twin;
    base.rec_cf = 0;
    version_t vt, vb;
    memset(&vt, 0, sizeof(vt));
    memset(&vb, 0, sizeof(vb));
    if (jit_build_version(pl, twin, &cpu, &opts, &vt, err, sizeof(err))) {
        printf("FAIL build twin %s: %s\n", twin->rec_cf ? "" : "", err);
        CHECK(0, "%s F={%d,%d} nf=%d: twin build failed: %s", tname,
              F[0], nf > 1 ? F[1] : -1, nf, err);
        return -1;
    }
    if (jit_build_version(pl, &base, &cpu, &opts, &vb, err, sizeof(err))) {
        CHECK(0, "%s F={%d,%d} nf=%d: base build failed: %s", tname,
              F[0], nf > 1 ? F[1] : -1, nf, err);
        jit_free_version(&vt);
        return -1;
    }
    /* The twin must be marked in its description (and differ from base). */
    CHECK(strstr(vt.desc, "+reccf") != NULL, "%s: twin desc lacks +reccf: %s",
          tname, vt.desc);
    CHECK(strstr(vb.desc, "+reccf") == NULL, "%s: base desc has +reccf: %s",
          tname, vb.desc);

    int ns = pl->n_in;
    static const size_t lens[] = { 1, 63, 64, 65, 1000, 4096, 5000 };
    for (size_t li = 0; li < sizeof(lens) / sizeof(lens[0]); li++) {
        size_t len = lens[li];
        uint8_t **in = malloc(sizeof(*in) * (size_t)ns);
        /* n_out outputs; allocate nf buffers each */
        uint8_t **ot = malloc(sizeof(*ot) * (size_t)nf);
        uint8_t **ob = malloc(sizeof(*ob) * (size_t)nf);
        for (int i = 0; i < ns; i++) {
            in[i] = abuf(len);
            randfill(in[i], len);
        }
        for (int j = 0; j < nf; j++) {
            ot[j] = abuf(len);
            ob[j] = abuf(len);
            memset(ot[j], 0xA5, len);
            memset(ob[j], 0xA5, len);
        }
        vt.fn((const void **)in, (void **)ot, len);
        vb.fn((const void **)in, (void **)ob, len);
        for (int j = 0; j < nf; j++)
            CHECK(memcmp(ot[j], ob[j], len) == 0,
                  "%s F={%d,%d} nf=%d twin=%s len=%zu: output %d differs",
                  tname, F[0], nf > 1 ? F[1] : -1, nf, vt.desc, len, j);
        for (int i = 0; i < ns; i++)
            free(in[i]);
        for (int j = 0; j < nf; j++) {
            free(ot[j]);
            free(ob[j]);
        }
        free(in);
        free(ot);
        free(ob);
        if (failures)
            break;
    }
    /* notail twin of the rec_cf twin: valid for len % group == 0. */
    {
        cand_t ntc = *twin;
        ntc.notail = 1;
        version_t vn;
        memset(&vn, 0, sizeof(vn));
        if (jit_build_version(pl, &ntc, &cpu, &opts, &vn, err,
                              sizeof(err)) == 0) {
            size_t len = vn.group_bytes * 3;
            uint8_t **in = malloc(sizeof(*in) * (size_t)ns);
            uint8_t **ot = malloc(sizeof(*ot) * (size_t)nf);
            uint8_t **ob = malloc(sizeof(*ob) * (size_t)nf);
            for (int i = 0; i < ns; i++) {
                in[i] = abuf(len);
                randfill(in[i], len);
            }
            for (int j = 0; j < nf; j++) {
                ot[j] = abuf(len);
                ob[j] = abuf(len);
            }
            vn.fn((const void **)in, (void **)ot, len);
            vb.fn((const void **)in, (void **)ob, len);
            for (int j = 0; j < nf; j++)
                CHECK(memcmp(ot[j], ob[j], len) == 0,
                      "%s F={%d,%d} nf=%d notail twin output %d differs",
                      tname, F[0], nf > 1 ? F[1] : -1, nf, j);
            for (int i = 0; i < ns; i++)
                free(in[i]);
            for (int j = 0; j < nf; j++) {
                free(ot[j]);
                free(ob[j]);
            }
            free(in);
            free(ot);
            free(ob);
            jit_free_version(&vn);
        } else if (opts.verbose) {
            printf("note: notail twin build failed (ok if unsupported): %s\n",
                   err);
        }
    }
    jit_free_version(&vt);
    jit_free_version(&vb);
    n_twins_tested++;
    if (nf == 1) {
        n_e1++;
        if (pl->rec_cf->cinv != 1)
            n_e1_finalmul++;
    } else {
        n_e2++;
        if (pl->rec_cf->b == 0)
            n_e2_b0++;
        if (pl->rec_cf->c == 0)
            n_e2_c0++;
    }
    return failures ? -1 : 0;
}

/* Red-green: flip one cf_data coefficient byte; the twin's output must
 * change vs the unflipped twin (proves the emitter reads cf_data). */
static void redgreen_case(const plan_t *pl, const cand_t *twin,
                          const char *tname, const int *F, int nf)
{
    char err[256];
    /* duplicate the plan's cf_data so the flip doesn't disturb pl */
    struct rec_cf_data *cfd = pl->rec_cf;
    struct rec_cf_data mut = *cfd;
    uint8_t *ecopy = malloc((size_t)cfd->ns * 2);
    memcpy(ecopy, nf == 1 ? cfd->ecoef : cfd->ecoef1, (size_t)cfd->ns);
    if (nf == 2)
        memcpy(ecopy + cfd->ns, cfd->ecoef2, (size_t)cfd->ns);
    if (nf == 1)
        mut.ecoef = ecopy;
    else {
        mut.ecoef1 = ecopy;
        mut.ecoef2 = ecopy + cfd->ns;
    }
    /* flip a nonzero coefficient (or the first byte if all zero) */
    int fi = 0;
    for (int s = 0; s < cfd->ns; s++)
        if (ecopy[s]) {
            fi = s;
            break;
        }
    ecopy[fi] ^= 0xFF;
    if (ecopy[fi] == 0)
        ecopy[fi] = 0x5A; /* keep it nonzero so the flip is observable */

    plan_t plm = *pl;
    plm.rec_cf = &mut;
    version_t v0, v1;
    memset(&v0, 0, sizeof(v0));
    memset(&v1, 0, sizeof(v1));
    CHECK(jit_build_version(pl, twin, &cpu, &opts, &v0, err, sizeof(err)) == 0,
          "%s redgreen: orig build: %s", tname, err);
    CHECK(jit_build_version(&plm, twin, &cpu, &opts, &v1, err, sizeof(err)) ==
              0,
          "%s redgreen: mutated build: %s", tname, err);
    if (failures) {
        free(ecopy);
        return;
    }
    int ns = pl->n_in;
    size_t len = 1000;
    uint8_t **in = malloc(sizeof(*in) * (size_t)ns);
    uint8_t **o0 = malloc(sizeof(*o0) * (size_t)nf);
    uint8_t **o1 = malloc(sizeof(*o1) * (size_t)nf);
    for (int i = 0; i < ns; i++) {
        in[i] = abuf(len);
        randfill(in[i], len);
    }
    for (int j = 0; j < nf; j++) {
        o0[j] = abuf(len);
        o1[j] = abuf(len);
    }
    v0.fn((const void **)in, (void **)o0, len);
    v1.fn((const void **)in, (void **)o1, len);
    int diff = 0;
    for (int j = 0; j < nf; j++)
        if (memcmp(o0[j], o1[j], len) != 0)
            diff = 1;
    CHECK(diff, "%s F={%d,%d} nf=%d: cf_data flip did not change output "
          "(vacuous red-green)",
          tname, F[0], nf > 1 ? F[1] : -1, nf);
    for (int i = 0; i < ns; i++)
        free(in[i]);
    for (int j = 0; j < nf; j++) {
        free(o0[j]);
        free(o1[j]);
    }
    free(in);
    free(o0);
    free(o1);
    jit_free_version(&v0);
    jit_free_version(&v1);
    free(ecopy);
}

/* One (topology, failure set, row set): build the recovery plan, and
 * if a rec_cf twin is enumerated, differentially test every twin. */
static void topo_case_rows(rj_topology *t, const char *tname, const int *F,
                           int nf, const int *rows, int nrows)
{
    char err[256];
    topo_int_t ti;
    CHECK(topo_int_build(&ti, t, err, sizeof(err)) == 0, "%s topo: %s", tname,
          err);
    if (failures)
        return;
    plan_t pl;
    memset(&pl, 0, sizeof(pl));
    int rc = plan_build_recover_rows(&ti, F, nf, rows, nrows, tname, &pl,
                                    err, sizeof(err));
    CHECK(rc == 0, "%s F={%d,%d} nf=%d: plan build: %s", tname, F[0],
          nf > 1 ? F[1] : -1, nf, err);
    if (rc == 0 && pl.rec_cf && pl.kind == P_MULACC) {
        cand_t cands[256];
        int nc = plan_candidates(&pl, &cpu, cands, 256, 0, 0);
        int ntw = 0;
        for (int i = 0; i < nc; i++) {
            if (!cands[i].rec_cf)
                continue;
            ntw++;
            if (diff_pair(&pl, &cands[i], tname, F, nf))
                break;
            /* red-green on the first twin of each shape only */
            if (ntw == 1)
                redgreen_case(&pl, &cands[i], tname, F, nf);
            if (failures)
                break;
        }
        if (ntw)
            printf("ok %s F={%d,%d} nf=%d: %d rec_cf twins differential-clean\n",
                   tname, F[0], nf > 1 ? F[1] : -1, nf, ntw);
    }
    plan_free(&pl);
    topo_int_free(&ti);
}

static void topo_case(rj_topology *t, const char *tname, const int *F, int nf)
{
    char err[256];
    topo_int_t ti;
    if (topo_int_build(&ti, t, err, sizeof(err)) != 0)
        return;
    int np = ti.npar;
    int *rows = malloc(sizeof(int) * (size_t)np);
    for (int r = 0; r < np; r++)
        rows[r] = r;
    topo_int_free(&ti);
    topo_case_rows(t, tname, F, nf, rows, np);
    free(rows);
}

int main(void)
{
    rj_cpu_detect(&cpu);
    printf("cpu: %s\n", cpu.brand);
    rj_options_default(&opts);
    opts.benchmark = 0;

    /* e=1 scan */
    {
        rj_topology *t = rj_topo_raid6(8);
        for (int f = 0; f < 10; f++) {
            int F[1] = { f };
            topo_case(t, "raid6(8)", F, 1);
        }
        /* row subsets yielding cinv != 1 (exercises the e=1 final
         * multiply): raid6 F={0} rows={1} -> cinv=2, F={1} -> cinv=4 */
        {
            int F0[1] = { 0 }, F1[1] = { 1 }, r1[1] = { 1 };
            topo_case_rows(t, "raid6(8)", F0, 1, r1, 1);
            topo_case_rows(t, "raid6(8)", F1, 1, r1, 1);
        }
        rj_topo_free(t);
    }
    {
        rj_topology *t = rj_topo_rs(6, 3);
        for (int f = 0; f < 9; f++) {
            int F[1] = { f };
            topo_case(t, "rs(6,3)", F, 1);
        }
        /* cinv != 1 cases: F={0} rows={2} -> cinv=2, F={1} rows={2} ->
         * cinv=4 */
        {
            int F0[1] = { 0 }, F1[1] = { 1 }, r2[1] = { 2 };
            topo_case_rows(t, "rs(6,3)", F0, 1, r2, 1);
            topo_case_rows(t, "rs(6,3)", F1, 1, r2, 1);
        }
        rj_topo_free(t);
    }
    {
        uint8_t cmat[8] = { 1, 1, 1, 1, 1, 2, 4, 8 };
        rj_topology *t = rj_topo_custom(4, 2, cmat);
        for (int f = 0; f < 6; f++) {
            int F[1] = { f };
            topo_case(t, "custom(4,2)", F, 1);
        }
        rj_topo_free(t);
    }
    /* e=2 scan */
    {
        rj_topology *t = rj_topo_raid6(8);
        int pairs[][2] = { { 0, 1 }, { 1, 5 }, { 2, 8 }, { 8, 9 }, { 0, 9 } };
        for (size_t k = 0; k < sizeof(pairs) / sizeof(pairs[0]); k++)
            topo_case(t, "raid6(8)", pairs[k], 2);
        rj_topo_free(t);
    }
    {
        rj_topology *t = rj_topo_rs(6, 3);
        int pairs[][2] = { { 0, 4 }, { 3, 6 }, { 6, 8 }, { 1, 7 } };
        for (size_t k = 0; k < sizeof(pairs) / sizeof(pairs[0]); k++)
            topo_case(t, "rs(6,3)", pairs[k], 2);
        rj_topo_free(t);
    }
    {
        uint8_t cmat[8] = { 1, 1, 1, 1, 1, 2, 4, 8 };
        rj_topology *t = rj_topo_custom(4, 2, cmat);
        int pairs[][2] = { { 1, 3 }, { 0, 5 }, { 4, 5 } };
        for (size_t k = 0; k < sizeof(pairs) / sizeof(pairs[0]); k++)
            topo_case(t, "custom(4,2)", pairs[k], 2);
        rj_topo_free(t);
    }

    printf("rec_cf twins tested: %ld (e=1: %ld, e=2: %ld)\n", n_twins_tested,
           n_e1, n_e2);
    printf("e=1 final-multiply exercised: %ld; e=2 b==0: %ld, c==0: %ld\n",
           n_e1_finalmul, n_e2_b0, n_e2_c0);
    CHECK(n_twins_tested > 0, "no rec_cf twins enumerated (vacuous test)");
    CHECK(n_e1 > 0, "no e=1 rec_cf twin tested (vacuous)");
    CHECK(n_e2 > 0, "no e=2 rec_cf twin tested (vacuous)");
    CHECK(n_e1_finalmul > 0,
          "e=1 final multiply (cinv != 1) never exercised (vacuous)");
    if (failures) {
        printf("%d FAILURES\n", failures);
        return 1;
    }
    printf("all tests passed\n");
    return 0;
}
