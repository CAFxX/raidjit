/* Tests for the RAID-6 XOR array codes (RDP, EVENODD, Liberation).
 *
 * Coverage:
 *  1. validity boundaries: prime/non-prime p, k/w envelope, the
 *     RJ_MAX_DISKS expanded-size cap, rj_xorcode_rows, rj_topo_raid6_rs
 *     preserving the old RS RAID-6 behavior (incl. P_HORNER plan kind).
 *  2. matrix unit tests: binary entries, Liberation Q ones-count
 *     (kw+k-1, Plank FAST'08 3.3), exhaustive 2-disk MDS check via
 *     solve_recovery (the exact solver plan_build uses).
 *  3. encode via the public rj_compile API vs the scalar ref_run oracle,
 *     across codes x params x boundary lengths x {benchmark, static}
 *     selection.
 *  4. recovery via rj_compile + rj_op_recover: every 2-data-disk pair,
 *     data+P, data+Q, P+Q, at two lengths; recovered bytes must equal the
 *     originals bit-exactly. (The parity-involved extras are spot samples
 *     by design: check_mds is exhaustive over all whole-disk pairs
 *     through the same solver plan_build uses.)
 *  4b. max-config JIT smoke: rdp(13)/evenodd(13)/liberation(13) (up to 26
 *     outputs) — encode + one 2-disk recovery end to end vs ref_run;
 *     CPUs that cannot encode a config must fail cleanly (NULL from
 *     rj_compile), asserted via the plan_candidates oracle.
 *  5. combination matrix (synthetic CPUs x every offered candidate x
 *     boundary lengths), JITing each candidate directly like
 *     test_matrix.c does: ISA x topology x length x candidate.
 *  6. default-table agreement: docs/xor_code_default.md's machine table
 *     vs xorcode_default_for_ndata(), and rj_topo_raid6() dispatch
 *     matching the table for every ndata.
 *
 * Portable: runs natively on x86-64 and under QEMU for aarch64/riscv64.
 * The rj_topo_raid6() default dispatch is benchmark-selected (see
 * docs/xor_code_default.md); tests that need the RS code path use the
 * explicit rj_topo_raid6_rs() opt-in.
 */
#include "../include/raidjit.h"
#include "../src/xorcode.h"
#include "../src/plan.h"
#include "../src/jit.h"
#include "../src/cpu.h"
#include "../src/ref.h"
#include "../src/gf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

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

static uint64_t rng_state = 0x51ab3f9c77d2e4a1ULL;
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
        printf("FAIL: oom\n");
        exit(1);
    }
    return p;
}

/* ---------------- 1. validity boundaries ---------------- */

static void test_validity(void)
{
    rj_topology *t;

    /* RDP: p prime >= 3; expanded (p-1)^2+2(p-1) = p^2-1 <= 256 */
    CHECK(rj_topo_rdp(2) == NULL, "rdp(2) must be NULL");
    CHECK(rj_topo_rdp(4) == NULL, "rdp(4) must be NULL");
    CHECK(rj_topo_rdp(9) == NULL, "rdp(9) must be NULL");
    CHECK(rj_topo_rdp(1) == NULL, "rdp(1) must be NULL");
    CHECK(rj_topo_rdp(-5) == NULL, "rdp(-5) must be NULL");
    t = rj_topo_rdp(3);
    CHECK(t != NULL, "rdp(3)");
    if (t) {
        CHECK(rj_topo_ndata(t) == 4 && rj_topo_npar(t) == 4,
              "rdp(3) dims %d+%d", rj_topo_ndata(t), rj_topo_npar(t));
        CHECK(rj_xorcode_rows(t) == 2, "rdp(3) rows");
        rj_topo_free(t);
    }
    t = rj_topo_rdp(13); /* 13^2-1 = 168 <= 256 */
    CHECK(t != NULL, "rdp(13)");
    if (t) {
        CHECK(rj_topo_ndata(t) == 144 && rj_topo_npar(t) == 24,
              "rdp(13) dims");
        CHECK(rj_xorcode_rows(t) == 12, "rdp(13) rows");
        rj_topo_free(t);
    }
    CHECK(rj_topo_rdp(17) == NULL, "rdp(17): 288 > 256 disks");

    /* EVENODD: p prime >= 3; expanded (p+2)(p-1) <= 256 */
    CHECK(rj_topo_evenodd(2) == NULL, "evenodd(2) must be NULL");
    CHECK(rj_topo_evenodd(4) == NULL, "evenodd(4) must be NULL");
    CHECK(rj_topo_evenodd(6) == NULL, "evenodd(6) must be NULL");
    t = rj_topo_evenodd(3);
    CHECK(t != NULL, "evenodd(3)");
    if (t) {
        CHECK(rj_topo_ndata(t) == 6 && rj_topo_npar(t) == 4,
              "evenodd(3) dims %d+%d", rj_topo_ndata(t), rj_topo_npar(t));
        CHECK(rj_xorcode_rows(t) == 2, "evenodd(3) rows");
        rj_topo_free(t);
    }
    t = rj_topo_evenodd(13); /* 15*12 = 180 <= 256 */
    CHECK(t != NULL, "evenodd(13)");
    if (t) {
        CHECK(rj_topo_ndata(t) == 156 && rj_topo_npar(t) == 24,
              "evenodd(13) dims");
        rj_topo_free(t);
    }
    CHECK(rj_topo_evenodd(17) == NULL, "evenodd(17): 304 > 256 disks");

    /* Liberation: k >= 2, w = smallest prime >= max(k,3);
     * expanded w*(k+2) <= 256 */
    CHECK(rj_topo_liberation(0) == NULL, "liberation(0) must be NULL");
    CHECK(rj_topo_liberation(1) == NULL, "liberation(1) must be NULL");
    CHECK(rj_topo_liberation(-3) == NULL, "liberation(-3) must be NULL");
    t = rj_topo_liberation(2); /* w=3 */
    CHECK(t != NULL, "liberation(2)");
    if (t) {
        CHECK(rj_topo_ndata(t) == 6 && rj_topo_npar(t) == 6,
              "liberation(2) dims %d+%d", rj_topo_ndata(t),
              rj_topo_npar(t));
        CHECK(rj_xorcode_rows(t) == 3, "liberation(2) rows");
        rj_topo_free(t);
    }
    t = rj_topo_liberation(4); /* w=5 */
    CHECK(t != NULL, "liberation(4)");
    if (t) {
        CHECK(rj_xorcode_rows(t) == 5, "liberation(4) rows");
        rj_topo_free(t);
    }
    t = rj_topo_liberation(13); /* w=13, 13*15=195 <= 256 */
    CHECK(t != NULL, "liberation(13)");
    if (t)
        rj_topo_free(t);
    CHECK(rj_topo_liberation(14) == NULL, "liberation(14): 272 > 256");
    CHECK(rj_topo_liberation(16) == NULL, "liberation(16): 306 > 256");

    /* rj_topo_raid6_rs: the old RS RAID-6, unchanged semantics */
    CHECK(rj_topo_raid6_rs(2) == NULL, "raid6_rs(2) must be NULL");
    CHECK(rj_topo_raid6_rs(52) == NULL, "raid6_rs(52) must be NULL");
    t = rj_topo_raid6_rs(8);
    CHECK(t != NULL, "raid6_rs(8)");
    if (t) {
        CHECK(rj_topo_ndata(t) == 8 && rj_topo_npar(t) == 2,
              "raid6_rs(8) dims");
        CHECK(rj_xorcode_rows(t) == 0, "raid6_rs is not an xorcode");
        /* old behavior: RS matrix -> P_HORNER plan */
        topo_int_t ti;
        char err[128];
        if (topo_int_build(&ti, t, err, sizeof(err)) == 0) {
            plan_t pl;
            rj_op op = rj_op_encode();
            if (plan_build(&ti, &op, "t", &pl, err, sizeof(err)) == 0) {
                CHECK(pl.kind == P_HORNER, "raid6_rs plan kind %d",
                      (int)pl.kind);
                plan_free(&pl);
            } else {
                CHECK(0, "raid6_rs plan_build: %s", err);
            }
            rj_op_free(&op);
            topo_int_free(&ti);
        }
        rj_topo_free(t);
    }

    /* rj_xorcode_rows on non-xorcode topologies */
    t = rj_topo_raid5(4);
    CHECK(t && rj_xorcode_rows(t) == 0, "raid5 rows==0");
    rj_topo_free(t);
    CHECK(rj_xorcode_rows(NULL) == 0, "rows(NULL)==0");

    /* rj_topo_raid6 bounds preserved */
    CHECK(rj_topo_raid6(2) == NULL, "raid6(2) must be NULL");
    CHECK(rj_topo_raid6(52) == NULL, "raid6(52) must be NULL");
    t = rj_topo_raid6(3);
    CHECK(t != NULL, "raid6(3)");
    rj_topo_free(t);
}

/* ---------------- 2. matrix unit tests ---------------- */

/* Exhaustive 2-disk MDS check through solve_recovery (the exact solver
 * plan_build uses for recover ops). Disks 0..k-1 are data, k is P,
 * k+1 is Q; each disk = R expanded indices. */
static void check_mds(const char *name, const uint8_t *m, int k, int R)
{
    int nd = k * R, np = 2 * R, nall = nd + np, ndisks = k + 2;
    for (int a = 0; a < ndisks; a++) {
        for (int b = a + 1; b < ndisks; b++) {
            int F[64], S[1024];
            uint8_t Rsol[64 * 1024];
            int nf = 0;
            for (int d = a; d <= b; d += (b - a)) {
                for (int r = 0; r < R; r++) {
                    int idx =
                        d < k ? d * R + r : nd + (d - k) * R + r;
                    F[nf++] = idx;
                }
                if (a == b)
                    break;
            }
            /* F is already increasing (a < b) */
            int ns = 0, fi = 0;
            for (int dsk = 0; dsk < nall; dsk++) {
                if (fi < nf && F[fi] == dsk) {
                    fi++;
                    continue;
                }
                S[ns++] = dsk;
            }
            int rc =
                solve_recovery(m, nd, np, F, nf, S, ns, Rsol);
            CHECK(rc == 0, "%s: disks %d+%d not recoverable", name, a, b);
        }
    }
}

static void test_matrices(void)
{
    /* RDP p=3,5,7 */
    int ps[] = {3, 5, 7};
    for (size_t pi = 0; pi < 3; pi++) {
        int p = ps[pi], R = p - 1, k = p - 1;
        uint8_t *m = abuf((size_t)2 * R * k * R);
        CHECK(xorcode_rdp_matrix(p, m) == 0, "rdp_matrix(%d)", p);
        for (int i = 0; i < 2 * R * k * R; i++)
            CHECK(m[i] == 0 || m[i] == 1, "rdp(%d) binary", p);
        /* P rows are exact row parity: k ones at (i, j) */
        for (int j = 0; j < R; j++) {
            int ones = 0;
            for (int i = 0; i < k * R; i++)
                ones += m[(size_t)j * k * R + i];
            CHECK(ones == k, "rdp(%d) P row %d ones=%d", p, j, ones);
        }
        check_mds("rdp", m, k, R);
        free(m);
        CHECK(xorcode_rdp_matrix(4, NULL) == -1, "rdp_matrix guards");
    }
    /* EVENODD p=3,5,7 */
    for (size_t pi = 0; pi < 3; pi++) {
        int p = ps[pi], R = p - 1, k = p;
        uint8_t *m = abuf((size_t)2 * R * k * R);
        CHECK(xorcode_evenodd_matrix(p, m) == 0, "evenodd_matrix(%d)",
              p);
        for (int i = 0; i < 2 * R * k * R; i++)
            CHECK(m[i] == 0 || m[i] == 1, "evenodd(%d) binary", p);
        for (int j = 0; j < R; j++) {
            int ones = 0;
            for (int i = 0; i < k * R; i++)
                ones += m[(size_t)j * k * R + i];
            CHECK(ones == k, "evenodd(%d) P row %d ones=%d", p, j, ones);
        }
        check_mds("evenodd", m, k, R);
        free(m);
    }
    /* Liberation (k,w): (2,3),(3,3),(4,5),(5,5),(6,7),(7,7) */
    int kws[][2] = {{2, 3}, {3, 3}, {4, 5}, {5, 5}, {6, 7}, {7, 7}};
    for (size_t ti = 0; ti < 6; ti++) {
        int k = kws[ti][0], w = kws[ti][1];
        uint8_t *m = abuf((size_t)2 * w * k * w);
        CHECK(xorcode_liberation_matrix(k, w, m) == 0,
              "liberation_matrix(%d,%d)", k, w);
        for (int i = 0; i < 2 * w * k * w; i++)
            CHECK(m[i] == 0 || m[i] == 1, "liberation binary");
        /* FAST'08 3.3: Q bitmatrix has kw + k - 1 ones */
        int ones = 0;
        for (int i = 0; i < w * k * w; i++)
            ones += m[(size_t)w * k * w + i];
        CHECK(ones == k * w + k - 1, "liberation(%d,%d) Q ones %d",
              k, w, ones);
        check_mds("liberation", m, k, w);
        free(m);
        CHECK(xorcode_liberation_matrix(k, 4, NULL) == -1,
              "liberation_matrix guards");
    }
    /* constructor/matrix agreement: rj_topo_rdp's matrix == builder's */
    {
        rj_topology *t = rj_topo_rdp(5);
        CHECK(t != NULL, "rdp(5)");
        if (t) {
            topo_int_t ti;
            char err[128];
            CHECK(topo_int_build(&ti, t, err, sizeof(err)) == 0, "%s",
                                err);
            uint8_t *m = abuf((size_t)ti.npar * ti.ndata);
            CHECK(xorcode_rdp_matrix(5, m) == 0, "rebuild");
            CHECK(memcmp(m, ti.mat, (size_t)ti.npar * ti.ndata) == 0,
                  "rdp(5) topo matrix == builder matrix");
            free(m);
            topo_int_free(&ti);
            rj_topo_free(t);
        }
    }
}

/* ---------------- helpers for JIT tests ---------------- */

static size_t length_set[] = {0, 1, 63, 64, 65, 1000, 4096};
#define NLEN (sizeof(length_set) / sizeof(length_set[0]))

/* Compile (topo, op) with the public API and run it; verify outputs
 * against ref_run on the same plan. Returns 1 on success. */
static int run_vs_ref(rj_topology *t, rj_op *op, const char *tag,
                      int benchmark, size_t *lens, int nlen)
{
    char err[256];
    rj_options opts;
    rj_options_default(&opts);
    opts.benchmark = benchmark;
    rj_jit *j = rj_compile(t, op, &opts, err, sizeof(err));
    CHECK(j != NULL, "%s: compile: %s", tag, err);
    if (!j)
        return 0;
    rj_fn fn = rj_get_fn(j);

    topo_int_t ti;
    CHECK(topo_int_build(&ti, t, err, sizeof(err)) == 0, "%s topo: %s",
          tag, err);
    plan_t pl;
    CHECK(plan_build(&ti, op, tag, &pl, err, sizeof(err)) == 0,
          "%s plan: %s", tag, err);
    int nd = ti.ndata, np = ti.npar;

    for (int li = 0; li < nlen; li++) {
        size_t len = lens[li];
        int nin = op->is_recover ? nd + np - op->nfailed : nd;
        int nbuf = nd + np;
        uint8_t **b = malloc(sizeof(*b) * (size_t)(nbuf + np));
        for (int i = 0; i < nbuf + np; i++) {
            b[i] = abuf(len);
            for (size_t k = 0; k < len; k++)
                b[i][k] = (uint8_t)xrng();
        }
        const void **in = malloc(sizeof(*in) * (size_t)nin);
        void **out = malloc(sizeof(*out) * (size_t)np);
        uint8_t **want = malloc(sizeof(*want) * (size_t)np);
        for (int i = 0; i < np; i++)
            want[i] = abuf(len);
        /* snapshot every buffer that must survive the call: encode
         * inputs, and recovery survivors (failed disks are clobbered
         * on purpose and excluded). */
        uint8_t **snap = NULL;
        int nsnap = 0;

        if (!op->is_recover) {
            for (int i = 0; i < nd; i++)
                in[i] = b[i];
            for (int j = 0; j < np; j++) {
                memset(b[nd + j], 0xA5, len);
                out[j] = b[nd + j];
            }
            nsnap = nd;
            snap = malloc(sizeof(*snap) * (size_t)nsnap);
            for (int i = 0; i < nd; i++) {
                snap[i] = abuf(len);
                memcpy(snap[i], b[i], len);
            }
            const uint8_t **rin = malloc(sizeof(*rin) * (size_t)nd);
            for (int i = 0; i < nd; i++)
                rin[i] = b[i];
            ref_run(&pl, rin, want, len);
            free(rin);
            fn(in, out, len);
            for (int j = 0; j < np; j++)
                CHECK(memcmp(b[nd + j], want[j], len) == 0,
                      "%s len=%zu par%d mismatch (bench=%d)", tag, len,
                      j, benchmark);
            for (int i = 0; i < nd; i++)
                CHECK(memcmp(b[i], snap[i], len) == 0,
                      "%s len=%zu input %d clobbered", tag, len, i);
        } else {
            /* recover: first encode the reference parity (surviving
             * parity disks must be consistent with the data), then
             * clobber the failed disks, feed survivors, compare out */
            int nf = op->nfailed;
            int *isfailed = calloc((size_t)(nd + np), sizeof(int));
            for (int f = 0; f < nf; f++)
                isfailed[op->failed[f]] = 1;
            {
                plan_t epl;
                rj_op eop = rj_op_encode();
                CHECK(plan_build(&ti, &eop, tag, &epl, err,
                                sizeof(err)) == 0,
                      "%s enc plan: %s", tag, err);
                const uint8_t **ein =
                    malloc(sizeof(*ein) * (size_t)nd);
                for (int i = 0; i < nd; i++)
                    ein[i] = b[i];
                uint8_t **eout =
                    malloc(sizeof(*eout) * (size_t)np);
                for (int j = 0; j < np; j++)
                    eout[j] = b[nd + j];
                ref_run(&epl, ein, eout, len);
                free(ein);
                free(eout);
                plan_free(&epl);
                rj_op_free(&eop);
            }
            uint8_t **orig = malloc(sizeof(*orig) * (size_t)nf);
            int si = 0, fi = 0;
            nsnap = nd + np - nf;
            snap = malloc(sizeof(*snap) * (size_t)nsnap);
            int sni = 0;
            for (int d = 0; d < nd + np; d++) {
                if (isfailed[d]) {
                    orig[fi] = abuf(len);
                    memcpy(orig[fi], b[d], len);
                    memset(b[d], 0x5A, len);
                    out[fi] = b[d];
                    fi++;
                } else {
                    in[si++] = b[d];
                    snap[sni] = abuf(len);
                    memcpy(snap[sni], b[d], len);
                    sni++;
                }
            }
            fn(in, out, len);
            fi = 0;
            sni = 0;
            for (int d = 0; d < nd + np; d++) {
                if (isfailed[d]) {
                    CHECK(memcmp(b[d], orig[fi], len) == 0,
                          "%s len=%zu disk%d recover mismatch (bench=%d)",
                          tag, len, d, benchmark);
                    fi++;
                } else {
                    CHECK(memcmp(b[d], snap[sni], len) == 0,
                          "%s len=%zu survivor disk%d clobbered",
                          tag, len, d);
                    sni++;
                }
            }
            for (int f = 0; f < nf; f++)
                free(orig[f]);
            free(orig);
            free(isfailed);
        }
        for (int i = 0; i < nsnap; i++)
            free(snap[i]);
        free(snap);
        for (int i = 0; i < nbuf + np; i++)
            free(b[i]);
        for (int i = 0; i < np; i++)
            free(want[i]);
        free(b);
        free(in);
        free(out);
        free(want);
        if (failures)
            break;
    }
    plan_free(&pl);
    topo_int_free(&ti);
    rj_free(j);
    return failures == 0;
}

/* ---------------- 3. encode vs ref ---------------- */

typedef struct {
    const char *name;
    rj_topology *(*mk)(void);
} topo_case_t;

static rj_topology *mk_rdp3(void) { return rj_topo_rdp(3); }
static rj_topology *mk_rdp5(void) { return rj_topo_rdp(5); }
static rj_topology *mk_eo3(void) { return rj_topo_evenodd(3); }
static rj_topology *mk_eo5(void) { return rj_topo_evenodd(5); }
static rj_topology *mk_lib2(void) { return rj_topo_liberation(2); }
static rj_topology *mk_lib4(void) { return rj_topo_liberation(4); }
static rj_topology *mk_lib6(void) { return rj_topo_liberation(6); }
static rj_topology *mk_r6rs(void) { return rj_topo_raid6_rs(4); }

static void test_encode(void)
{
    topo_case_t cases[] = {
        {"rdp(3)", mk_rdp3},       {"rdp(5)", mk_rdp5},
        {"evenodd(3)", mk_eo3},    {"evenodd(5)", mk_eo5},
        {"liberation(2)", mk_lib2}, {"liberation(4)", mk_lib4},
        {"liberation(6)", mk_lib6}, {"raid6_rs(4)", mk_r6rs},
    };
    for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ci++) {
        for (int bench = 0; bench <= 1; bench++) {
            rj_topology *t = cases[ci].mk();
            CHECK(t != NULL, "%s construct", cases[ci].name);
            if (!t)
                continue;
            /* the expanded matrix must lower to P_XOR (except RS) */
            {
                topo_int_t ti;
                char err[128];
                if (topo_int_build(&ti, t, err, sizeof(err)) == 0) {
                    plan_t pl;
                    rj_op op0 = rj_op_encode();
                    if (plan_build(&ti, &op0, "t", &pl, err,
                                   sizeof(err)) == 0) {
                        int is_rs = strcmp(cases[ci].name, "raid6_rs(4)") == 0;
                        CHECK((pl.kind == P_XOR) != is_rs,
                              "%s plan kind %d", cases[ci].name,
                              (int)pl.kind);
                        plan_free(&pl);
                    }
                    rj_op_free(&op0);
                    topo_int_free(&ti);
                }
            }
            rj_op op = rj_op_encode();
            char tag[96];
            snprintf(tag, sizeof(tag), "%s enc", cases[ci].name);
            run_vs_ref(t, &op, tag, bench, length_set, NLEN);
            rj_op_free(&op);
            rj_topo_free(t);
            if (failures)
                return;
        }
    }
}

/* ---------------- 4. recovery ---------------- */

/* Build the expanded failed-disk list for whole original disks.
 * disks[] holds original disk ids: 0..k-1 data, k = P, k+1 = Q. */
static int *expand_failed(rj_topology *t, const int *disks, int ndisks,
                          int *out_nf)
{
    int R = rj_xorcode_rows(t);
    int k = (rj_topo_ndata(t) - 0) / (R ? R : 1);
    int *F = malloc(sizeof(int) * (size_t)ndisks * (R ? R : 1));
    int nf = 0;
    for (int di = 0; di < ndisks; di++) {
        int d = disks[di];
        if (R == 0) {
            F[nf++] = d;
        } else if (d < k) {
            for (int r = 0; r < R; r++)
                F[nf++] = d * R + r;
        } else {
            int base = rj_topo_ndata(t) + (d - k) * R;
            for (int r = 0; r < R; r++)
                F[nf++] = base + r;
        }
    }
    *out_nf = nf;
    return F;
}

static void test_recover(void)
{
    struct {
        const char *name;
        rj_topology *(*mk)(void);
        int k; /* data disks */
    } cases[] = {
        {"rdp(5)", mk_rdp5, 4},
        {"evenodd(5)", mk_eo5, 5},
        {"liberation(4)", mk_lib4, 4},
        {"liberation(6)", mk_lib6, 6},
    };
    size_t lens[] = {1000, 65536};
    for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ci++) {
        rj_topology *t = cases[ci].mk();
        CHECK(t != NULL, "%s construct", cases[ci].name);
        if (!t)
            continue;
        int k = cases[ci].k;
        /* every 2-data-disk pair */
        for (int a = 0; a < k && !failures; a++) {
            for (int b = a + 1; b < k && !failures; b++) {
                int disks[2] = {a, b};
                int nf = 0;
                int *F = expand_failed(t, disks, 2, &nf);
                rj_op op = rj_op_recover(F, nf);
                free(F);
                char tag[96];
                snprintf(tag, sizeof(tag), "%s rec d%d+d%d",
                         cases[ci].name, a, b);
                run_vs_ref(t, &op, tag, 1, lens, 2);
                rj_op_free(&op);
            }
        }
        /* data+P, data+Q, P+Q, single data disk.
         *
         * Sampling rationale (deliberate, not an oversight): the data+data
         * pairs above ARE exhaustive at JIT level, and check_mds() in §2
         * already proves EVERY whole-disk pair (data+data, data+P,
         * data+Q, P+Q) recoverable through solve_recovery — the exact
         * solver plan_build() uses for recover ops. The JIT-level extras
         * below therefore only spot-sample one data+P, one data+Q, the
         * P+Q pair and one single-disk failure per topology, to catch
         * JIT/codegen issues in the parity-involved paths rather than
         * re-proving the constructions. */
        int extra[][2] = {{0, k}, {1, k + 1}, {k, k + 1}, {2, 2}};
        for (size_t ei = 0; ei < 4 && !failures; ei++) {
            int nd = (ei == 3) ? 1 : 2;
            int nf = 0;
            int *F = expand_failed(t, extra[ei], nd, &nf);
            rj_op op = rj_op_recover(F, nf);
            free(F);
            char tag[96];
            snprintf(tag, sizeof(tag), "%s rec %d+%d", cases[ci].name,
                     extra[ei][0], extra[ei][1]);
            run_vs_ref(t, &op, tag, 1, lens, 2);
            rj_op_free(&op);
        }
        rj_topo_free(t);
    }
}

/* ---------------- max-config JIT smoke ----------------
 *
 * rdp(13)/evenodd(13)/liberation(13) are the largest constructible
 * configs (up to 26 parity outputs; larger params hit the RJ_MAX_DISKS
 * cap). Validity + matrix-MDS already cover them, but nothing JITed
 * them end to end. For each config this JITs encode and one 2-disk
 * recovery through the public API and checks both against ref_run.
 *
 * CPUs that cannot encode a config (register pressure: P_XOR needs
 * n_out*U+1 <= nregs, so 24-26 outputs don't fit 16-register SSE/AVX2/
 * NEON/SVE backends) must fail CLEANLY -- rj_compile returns NULL with
 * "no candidates" -- never miscompile. That clean failure is asserted
 * here, not silently skipped: the planner's own candidate count
 * (plan_candidates, the exact oracle rj_compile uses internally)
 * decides which behavior to expect, so the failure mode is loud in
 * both directions by construction. Expected split: 6 JIT-verified on
 * 32-register ISAs (AVX-512, RVV), 6 clean no-candidate on 16-register
 * ISAs (SSE/AVX2, NEON, SVE). Static selection (benchmark=0): the smoke
 * target is "it JITs correctly / fails cleanly", not kernel choice. */
static rj_topology *mk_rdp13(void) { return rj_topo_rdp(13); }
static rj_topology *mk_eo13(void) { return rj_topo_evenodd(13); }
static rj_topology *mk_lib13(void) { return rj_topo_liberation(13); }

static void test_max_smoke(void)
{
    struct {
        const char *name;
        rj_topology *(*mk)(void);
    } cases[] = {
        {"rdp(13)", mk_rdp13},
        {"evenodd(13)", mk_eo13},
        {"liberation(13)", mk_lib13},
    };
    size_t lens[] = {1000, 65536};
    rj_cpu_info host;
    rj_cpu_detect(&host);
    int n_verified = 0, n_cleanfail = 0;
    const int n_ops_total = 6; /* 3 configs x {encode, recover} */
    for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]) && !failures;
         ci++) {
        rj_topology *t = cases[ci].mk();
        CHECK(t != NULL, "%s construct", cases[ci].name);
        if (!t)
            continue;
        for (int oi = 0; oi < 2 && !failures; oi++) {
            rj_op op;
            char tag[96];
            if (oi == 0) {
                op = rj_op_encode();
                snprintf(tag, sizeof(tag), "%s max enc", cases[ci].name);
            } else {
                int disks[2] = {0, 1}, nf = 0;
                int *F = expand_failed(t, disks, 2, &nf);
                op = rj_op_recover(F, nf);
                free(F);
                snprintf(tag, sizeof(tag), "%s max rec d0+d1",
                         cases[ci].name);
            }
            /* Encodability oracle: the same call rj_compile makes. */
            int nc = -1;
            {
                topo_int_t ti;
                char err[128];
                plan_t pl;
                if (topo_int_build(&ti, t, err, sizeof(err)) == 0 &&
                    plan_build(&ti, &op, tag, &pl, err, sizeof(err)) == 0) {
                    cand_t cands[256];
                    nc = plan_candidates(&pl, &host, cands, 256, 0, 0);
                    plan_free(&pl);
                } else {
                    CHECK(0, "%s: plan: %s", tag, err);
                }
                topo_int_free(&ti);
            }
            if (failures) {
                rj_op_free(&op);
                break;
            }
            if (nc == 0) {
                /* Unencodable on this CPU: public API must fail
                 * cleanly (NULL), not produce a bad kernel. */
                rj_options opts;
                rj_options_default(&opts);
                opts.benchmark = 0;
                char err[256];
                rj_jit *j = rj_compile(t, &op, &opts, err, sizeof(err));
                CHECK(j == NULL,
                      "%s: expected clean no-candidate failure, got JIT",
                      tag);
                if (j) {
                    rj_free(j);
                } else {
                    n_cleanfail++;
                    printf("%s: clean no-candidate (%s)\n", tag, err);
                }
            } else if (nc > 0) {
                run_vs_ref(t, &op, tag, 0, lens, 2);
                if (!failures)
                    n_verified++;
            } else {
                CHECK(0, "%s: plan_candidates failed", tag);
            }
            rj_op_free(&op);
        }
        rj_topo_free(t);
    }
    printf("max-config smoke: %d JIT-verified, %d clean no-candidate\n",
           n_verified, n_cleanfail);
    CHECK(n_verified + n_cleanfail == n_ops_total,
          "every op either verified or cleanly unencodable (got %d+%d)",
          n_verified, n_cleanfail);
}

/* ---------------- 5. combination matrix ---------------- */

#if defined(__aarch64__)
static rj_cpu_info mkcpu(int neon, int sve, int sve2)
{
    rj_cpu_info c;
    memset(&c, 0, sizeof(c));
    c.neon = neon;
    c.sve = sve;
    c.sve2 = sve2;
    c.sve_vl = sve ? 16 : 0;
    snprintf(c.brand, sizeof(c.brand), "synthetic");
    return c;
}
#elif defined(__riscv)
static rj_cpu_info mkcpu(int rvv)
{
    rj_cpu_info c;
    memset(&c, 0, sizeof(c));
    c.rvv = rvv;
    c.rvv_vlen = rvv ? 16 : 0;
    snprintf(c.brand, sizeof(c.brand), "synthetic");
    return c;
}
#else
static rj_cpu_info mkcpu(int sse2, int ssse3, int avx, int avx2, int f,
                         int bw, int vl, int gfni)
{
    rj_cpu_info c;
    memset(&c, 0, sizeof(c));
    c.sse2 = sse2;
    c.ssse3 = ssse3;
    c.avx = avx;
    c.avx2 = avx2;
    c.avx512f = f;
    c.avx512bw = bw;
    c.avx512vl = vl;
    c.gfni = gfni;
    snprintf(c.brand, sizeof(c.brand), "synthetic");
    return c;
}
#endif

typedef struct {
    const char *name;
    rj_cpu_info cpu;
    int host_gated; /* only include if host has the feature */
} cpu_variant_t;

/* JIT every candidate the planner offers for (topo, cpu) and verify
 * encode vs ref_run at boundary lengths. This is the ISA x topology x
 * length x candidate combination matrix. */
static void test_combinations(void)
{
    topo_case_t cases[] = {
        {"rdp(3)", mk_rdp3},       {"rdp(5)", mk_rdp5},
        {"evenodd(3)", mk_eo3},    {"evenodd(5)", mk_eo5},
        {"liberation(2)", mk_lib2}, {"liberation(4)", mk_lib4},
    };
    size_t ncases = sizeof(cases) / sizeof(cases[0]);

    cpu_variant_t variants[8];
    int nvar = 0;
    const char *test_cpu = getenv("RAIDJIT_TEST_CPU");
    if (test_cpu && strcmp(test_cpu, "detect") == 0) {
        /* qemu path (tests/qemu_matrix.sh): the detected CPU may be an
         * emulated lesser model; synthetic variants would emit
         * instructions it cannot execute (SIGILL). Use detection only,
         * exactly like tests/test_matrix.c. */
        rj_cpu_info host;
        rj_cpu_detect(&host);
        variants[nvar].name = "detected";
        variants[nvar].cpu = host;
        nvar = 1;
        printf("cpu variant: detected (%s)\n", host.brand);
    } else {
#if defined(__aarch64__)
    variants[nvar].name = "neon";
    variants[nvar++].cpu = mkcpu(1, 0, 0);
    {
        rj_cpu_info host;
        rj_cpu_detect(&host);
        if (host.sve) {
            variants[nvar].name = "sve";
            variants[nvar].cpu = mkcpu(1, 1, 0);
            variants[nvar].cpu.sve_vl = host.sve_vl;
            variants[nvar].host_gated = 1;
            nvar++;
        }
    }
#elif defined(__riscv)
    {
        rj_cpu_info host;
        rj_cpu_detect(&host);
        if (host.rvv) {
            variants[nvar].name = "rvv";
            variants[nvar].cpu = mkcpu(1);
            variants[nvar].host_gated = 1;
            nvar++;
        }
    }
#else
    {
        rj_cpu_info host;
        rj_cpu_detect(&host);
        /* Host-gate: emitting AVX2/AVX-512 for a lesser host is SIGILL,
         * not a test. The sse2-only fallback keeps nvar > 0 on any
         * x86-64. */
        if (host.ssse3) {
            variants[nvar].name = "sse3";
            variants[nvar++].cpu = mkcpu(1, 1, 0, 0, 0, 0, 0, 0);
        } else {
            variants[nvar].name = "sse2-only";
            variants[nvar++].cpu = mkcpu(1, 0, 0, 0, 0, 0, 0, 0);
        }
        if (host.avx2) {
            variants[nvar].name = "avx2";
            variants[nvar++].cpu = mkcpu(1, 1, 1, 1, 0, 0, 0, 0);
        }
        if (host.avx512f && host.avx512bw && host.avx512vl) {
            variants[nvar].name = "avx512-nogfni";
            variants[nvar++].cpu = mkcpu(1, 1, 1, 1, 1, 1, 1, 0);
        }
        if (host.avx512f && host.gfni) {
            variants[nvar].name = "avx512-gfni";
            variants[nvar++].cpu = mkcpu(1, 1, 1, 1, 1, 1, 1, 1);
        }
    }
#endif
    }
    CHECK(nvar > 0, "no cpu variants (host lacks vector ISA?)");
    if (nvar == 0)
        return;

    rj_options opts;
    rj_options_default(&opts);
    opts.benchmark = 0;

    int total_cands = 0;
    for (size_t ci = 0; ci < ncases && !failures; ci++) {
        rj_topology *t = cases[ci].mk();
        CHECK(t != NULL, "%s construct", cases[ci].name);
        if (!t)
            continue;
        topo_int_t ti;
        char err[256];
        CHECK(topo_int_build(&ti, t, err, sizeof(err)) == 0, "%s",
              err);
        rj_op op = rj_op_encode();
        plan_t pl;
        CHECK(plan_build(&ti, &op, "t", &pl, err, sizeof(err)) == 0,
              "%s", err);
        int nd = ti.ndata, np = ti.npar;
        for (int vi = 0; vi < nvar && !failures; vi++) {
            cand_t cands[256];
            int nc =
                plan_candidates(&pl, &variants[vi].cpu, cands, 256, 0, 0);
            CHECK(nc > 0, "%s %s: no candidates", cases[ci].name,
                  variants[vi].name);
            for (int candi = 0; candi < nc && !failures; candi++) {
                version_t v;
                memset(&v, 0, sizeof(v));
                if (jit_build_version(&pl, &cands[candi],
                                      &variants[vi].cpu, &opts, &v, err,
                                      sizeof(err)) != 0) {
                    CHECK(0, "%s %s cand %d: build: %s",
                          cases[ci].name, variants[vi].name, candi,
                          err);
                    continue;
                }
                total_cands++;
                for (size_t li = 0; li < NLEN && !failures; li++) {
                    size_t len = length_set[li];
                    uint8_t **b =
                        malloc(sizeof(*b) * (size_t)(nd + np));
                    for (int i = 0; i < nd + np; i++) {
                        b[i] = abuf(len);
                        for (size_t k = 0; k < len; k++)
                            b[i][k] = (uint8_t)xrng();
                    }
                    const void **in =
                        malloc(sizeof(*in) * (size_t)nd);
                    void **out = malloc(sizeof(*out) * (size_t)np);
                    uint8_t **want =
                        malloc(sizeof(*want) * (size_t)np);
                    for (int i = 0; i < nd; i++)
                        in[i] = b[i];
                    for (int j = 0; j < np; j++) {
                        want[j] = abuf(len);
                        memset(b[nd + j], 0xA5, len);
                        out[j] = b[nd + j];
                    }
                    const uint8_t **rin =
                        malloc(sizeof(*rin) * (size_t)nd);
                    for (int i = 0; i < nd; i++)
                        rin[i] = b[i];
                    ref_run(&pl, rin, want, len);
                    free(rin);
                    v.fn(in, out, len);
                    for (int j = 0; j < np; j++)
                        CHECK(memcmp(b[nd + j], want[j], len) == 0,
                              "%s %s cand%d len=%zu par%d mismatch",
                              cases[ci].name, variants[vi].name,
                              candi, len, j);
                    /* inputs not clobbered */
                    for (int i = 0; i < nd + np; i++)
                        free(b[i]);
                    for (int j = 0; j < np; j++)
                        free(want[j]);
                    free(b);
                    free(in);
                    free(out);
                    free(want);
                }
                jit_free_version(&v);
            }
        }
        plan_free(&pl);
        rj_op_free(&op);
        topo_int_free(&ti);
        rj_topo_free(t);
    }
    printf("combination matrix: %d candidates JITed x %d lengths\n",
           total_cands, (int)NLEN);
    CHECK(total_cands > 0, "combination matrix was vacuous");
}

/* ---------------- 6. default-table agreement ---------------- */

/* Parse docs/xor_code_default.md's machine block:
 *   BEGIN-MACHINE-TABLE
 *   <ndata> <rs|rdp|evenodd|liberation>
 *   END-MACHINE-TABLE
 */
static int parse_default_table(xorcode_choice_t out[52])
{
    for (int i = 0; i < 52; i++)
        out[i] = (xorcode_choice_t)-1;
    FILE *f = fopen("docs/xor_code_default.md", "r");
    if (!f)
        f = fopen("../docs/xor_code_default.md", "r");
    CHECK(f != NULL, "cannot open docs/xor_code_default.md");
    if (!f)
        return -1;
    char line[256];
    int in_block = 0, n = 0;
    while (fgets(line, sizeof(line), f)) {
        if (strstr(line, "BEGIN-MACHINE-TABLE")) {
            in_block = 1;
            continue;
        }
        if (strstr(line, "END-MACHINE-TABLE"))
            break;
        if (!in_block)
            continue;
        int nd;
        char choice[32];
        if (sscanf(line, "%d %31s", &nd, choice) != 2)
            continue;
        CHECK(nd >= 3 && nd <= 51, "table ndata %d in range", nd);
        xorcode_choice_t c;
        if (strcmp(choice, "rs") == 0)
            c = XORCODE_RS;
        else if (strcmp(choice, "rdp") == 0)
            c = XORCODE_RDP;
        else if (strcmp(choice, "evenodd") == 0)
            c = XORCODE_EVENODD;
        else if (strcmp(choice, "liberation") == 0)
            c = XORCODE_LIBERATION;
        else {
            CHECK(0, "table: unknown choice '%s'", choice);
            continue;
        }
        out[nd] = c;
        n++;
    }
    fclose(f);
    return n;
}

static void test_default_table(void)
{
    xorcode_choice_t tab[52];
    int n = parse_default_table(tab);
    CHECK(n > 0, "parsed %d table entries", n);
    if (n <= 0)
        return;
    for (int nd = 3; nd <= 51; nd++) {
        CHECK(tab[nd] != (xorcode_choice_t)-1,
              "table covers ndata=%d", nd);
        CHECK(xorcode_default_for_ndata(nd) == tab[nd],
              "ndata=%d: C table %d != md %d", nd,
              (int)xorcode_default_for_ndata(nd), (int)tab[nd]);
        /* every non-RS entry must construct (table <-> validity) */
        rj_topology *t = NULL;
        int expect_rows = 0;
        switch (tab[nd]) {
        case XORCODE_RDP:
            t = rj_topo_rdp(nd + 1);
            expect_rows = nd;
            break;
        case XORCODE_EVENODD:
            t = rj_topo_evenodd(nd);
            expect_rows = nd - 1;
            break;
        case XORCODE_LIBERATION:
            t = rj_topo_liberation(nd);
            break;
        case XORCODE_RS:
            break;
        }
        if (tab[nd] != XORCODE_RS) {
            CHECK(t != NULL, "ndata=%d: table choice constructs", nd);
            if (t) {
                if (tab[nd] == XORCODE_LIBERATION)
                    expect_rows = rj_xorcode_rows(t);
                CHECK(rj_xorcode_rows(t) == expect_rows,
                      "ndata=%d rows", nd);
                rj_topo_free(t);
            }
        }
        /* rj_topo_raid6 dispatch matches the table */
        t = rj_topo_raid6(nd);
        CHECK(t != NULL, "raid6(%d) non-NULL", nd);
        if (t) {
            int is_xor = rj_xorcode_rows(t) != 0;
            CHECK(is_xor == (tab[nd] != XORCODE_RS),
                  "raid6(%d) dispatch matches table", nd);
            rj_topo_free(t);
        }
    }
}

int main(void)
{
    printf("== validity ==\n");
    test_validity();
    printf("== matrices ==\n");
    test_matrices();
    printf("== encode ==\n");
    test_encode();
    printf("== recover ==\n");
    test_recover();
    printf("== max-config smoke ==\n");
    test_max_smoke();
    printf("== combinations ==\n");
    test_combinations();
    printf("== default table ==\n");
    test_default_table();
    if (failures == 0)
        printf("XORCODE-ALL-OK\n");
    else
        printf("XORCODE-FAILURES: %d\n", failures);
    return failures != 0;
}
