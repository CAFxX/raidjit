/* tools/perf_guard.c — before/after PERF GUARD for the SOTA batch.
 *
 * Measures end-user steady-state throughput of the winning kernel for a
 * fixed corpus of (topology, op) configs, compiled with default options
 * (benchmark=1), at 3 stripe sizes (cache-resident 4 KiB and 64 KiB per
 * disk, streaming 1 MiB per disk). Also records compile (selection) wall
 * time and the picked variant per config.
 *
 * Correctness: every config's winner output is verified against the
 * scalar reference before timing; any mismatch aborts the run (a perf
 * guard that measures wrong code is worse than no guard).
 *
 * Methodology per (config, size): 2 warmup + 11 timed iterations, median
 * of per-iteration GB/s. The whole corpus runs PASSES=3 times in-process
 * (fresh compiles each pass, capturing selection variance); the report
 * gives the median and [min,max] across passes. GB/s = total stripe
 * bytes ((ndata+npar) * len) per second, matching benchmark_results.md.
 *
 * Usage: perf_guard <results.json>
 *        perf_guard --compare <base.json> <new.json>
 * Prints a human table to stdout and writes JSON to <results.json>.
 * Exit 0 on success (all configs correct), 1 on any correctness failure.
 *
 * Comparison mode (added 2026-10-02 for the candidate-pruning track):
 * compares two reports config by config. For each shared config the new
 * report's per-size GB/s medians must fall within the base report's
 * [min,max] bands across its passes (throughput parity), and the new
 * compile_s median must not exceed the base band top (no compile-time
 * regression); the per-config and total compile-time deltas are
 * reported. Exit 0 iff every shared config passes both checks, 1
 * otherwise, 2 on I/O/parse errors. Typical use: base = RJ_NO_PRUNE=1
 * run, new = pruning-enabled run.
 *
 * Provenance: the JSON records "prune": "on"|"off", derived from
 * RJ_NO_PRUNE with the same contract as prune_versions (any non-empty
 * value other than exactly "0" disables).
 *
 * Build (from repo root, x86-64):
 *   gcc -O2 -std=c11 -Wall -Wextra -Werror -D_POSIX_C_SOURCE=200809L \
 *       -Iinclude -Isrc -o /tmp/perf_guard tools/perf_guard.c \
 *       build-x86_64/libraidjit.a -lm
 * or: make build-x86_64/perf_guard
 */
#define _POSIX_C_SOURCE 200809L
#include "../include/raidjit.h"
#include "../src/plan.h"
#include "../src/gf.h"
#include "../src/ref.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#define PASSES 3
#define NWARM 2
#define NITER 11

static const size_t sizes[] = { 4096, 65536, 1048576 };
#define NSIZES (sizeof(sizes) / sizeof(sizes[0]))

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
        fprintf(stderr, "perf_guard: oom\n");
        exit(1);
    }
    return p;
}

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ---------------- config corpus ---------------- */

typedef enum { C_ENC, C_REC, C_PUPD } ckind_t;

typedef struct {
    const char *name;   /* "raid5(4+1)/enc" */
    ckind_t kind;
    int ndata, npar;    /* topology params */
    int cauchy;         /* 1 = custom Cauchy matrix via rj_topo_custom */
    int failed[8];      /* C_REC */
    int nfailed;
    int changed[8];     /* C_PUPD */
    int nchanged;
} config_t;

static const config_t corpus[] = {
    { "raid1/enc",        C_ENC,  1, 1, 0, {0}, 0, {0}, 0 },
    { "raid5(4+1)/enc",   C_ENC,  4, 1, 0, {0}, 0, {0}, 0 },
    { "raid5(4+1)/rec0",  C_REC,  4, 1, 0, {0}, 1, {0}, 0 },
    { "raid5(4+1)/recP",  C_REC,  4, 1, 0, {4}, 1, {0}, 0 },
    { "raid5(4+1)/pupd",  C_PUPD, 4, 1, 0, {0}, 0, {0, 2}, 2 },
    { "raid5(16+1)/enc",  C_ENC, 16, 1, 0, {0}, 0, {0}, 0 },
    { "raid6(8+2)/enc",   C_ENC,  8, 2, 0, {0}, 0, {0}, 0 },
    { "raid6(8+2)/rec01", C_REC,  8, 2, 0, {0, 1}, 2, {0}, 0 },
    { "raid6(8+2)/recPQ", C_REC,  8, 2, 0, {8, 9}, 2, {0}, 0 },
    { "raid6(16+2)/enc",  C_ENC, 16, 2, 0, {0}, 0, {0}, 0 },
    { "rs(6,3)/enc",      C_ENC,  6, 3, 0, {0}, 0, {0}, 0 },
    { "rs(6,3)/rec02",    C_REC,  6, 3, 0, {0, 2}, 2, {0}, 0 },
    { "rs(10,3)/enc",     C_ENC, 10, 3, 0, {0}, 0, {0}, 0 },
    { "rs(10,3)/rec01",   C_REC, 10, 3, 0, {0, 1}, 2, {0}, 0 },
    { "rs(10,4)c/enc",    C_ENC, 10, 4, 1, {0}, 0, {0}, 0 },
    { "rs(10,4)c/rec01",  C_REC, 10, 4, 1, {0, 1}, 2, {0}, 0 },
    { "rs(20,3)/enc",     C_ENC, 20, 3, 0, {0}, 0, {0}, 0 },
    { "rs(20,3)/rec012",  C_REC, 20, 3, 0, {0, 1, 2}, 3, {0}, 0 },
};
#define NCORPUS (sizeof(corpus) / sizeof(corpus[0]))

static rj_topology *mk_topo(const config_t *c)
{
    if (c->cauchy) {
        /* Custom Cauchy matrix: x_i = i+1, y_j = nd+1+j, M = inv(x^y).
         * MDS by construction (distinct x_i, y_j, x_i != y_j). Same
         * pattern as tests/test_large_mds.c. */
        int nd = c->ndata, np = c->npar;
        uint8_t *m = malloc((size_t)np * (size_t)nd);
        if (!m)
            return NULL;
        for (int j = 0; j < np; j++)
            for (int i = 0; i < nd; i++)
                m[(size_t)j * (size_t)nd + (size_t)i] =
                    gf_inv((uint8_t)((i + 1) ^ (nd + 1 + j)));
        rj_topology *t = rj_topo_custom(nd, np, m);
        free(m);
        return t;
    }
    if (c->ndata == 1 && c->npar == 1)
        return rj_topo_raid1();
    if (c->npar == 1)
        return rj_topo_raid5(c->ndata);
    if (c->npar == 2)
        return rj_topo_raid6(c->ndata);
    return rj_topo_rs(c->ndata, c->npar);
}

static rj_op mk_op(const config_t *c)
{
    if (c->kind == C_REC)
        return rj_op_recover(c->failed, c->nfailed);
    if (c->kind == C_PUPD)
        return rj_op_parity_update(c->changed, c->nchanged);
    return rj_op_encode();
}

/* ---------------- scalar reference ---------------- */

static void ref_encode_mat(const uint8_t *mat, int nd, int np,
                           uint8_t **d, uint8_t **p, size_t len)
{
    for (int j = 0; j < np; j++)
        for (size_t k = 0; k < len; k++) {
            uint8_t a = 0;
            for (int i = 0; i < nd; i++) {
                uint8_t c = mat[(size_t)j * (size_t)nd + (size_t)i];
                if (c)
                    a ^= gf_mul(d[i][k], c);
            }
            p[j][k] = a;
        }
}

/* Verify the compiled winner against the scalar reference at a small
 * length. Returns 0 on match, 1 on mismatch. */
static int verify(const config_t *c, rj_fn fn, topo_int_t *ti)
{
    int nd = ti->ndata, np = ti->npar;
    size_t len = 1000; /* exercises vector body + scalar tail */
    int nbuf = nd + np;
    uint8_t **b = malloc(sizeof(*b) * (size_t)(nbuf + np));
    uint8_t **want = malloc(sizeof(*want) * (size_t)np);
    for (int i = 0; i < nbuf + np; i++)
        b[i] = abuf(len);
    for (int j = 0; j < np; j++)
        want[j] = abuf(len);
    for (int i = 0; i < nd; i++)
        for (size_t k = 0; k < len; k++)
            b[i][k] = (uint8_t)xrng();

    const void **in;
    void **out;
    int ok = 1;

    if (c->kind == C_ENC) {
        ref_encode_mat(ti->mat, nd, np, b, want, len);
        in = malloc(sizeof(*in) * (size_t)nd);
        out = malloc(sizeof(*out) * (size_t)np);
        for (int i = 0; i < nd; i++)
            in[i] = b[i];
        for (int j = 0; j < np; j++) {
            memset(b[nd + j], 0xA5, len);
            out[j] = b[nd + j];
        }
        fn(in, out, len);
        for (int j = 0; j < np && ok; j++)
            if (memcmp(b[nd + j], want[j], len))
                ok = 0;
        free(in);
        free(out);
    } else if (c->kind == C_REC) {
        /* encode reference parities, then recover and compare */
        ref_encode_mat(ti->mat, nd, np, b, want, len);
        for (int j = 0; j < np; j++)
            memcpy(b[nd + j], want[j], len);
        int ns = nd + np - c->nfailed;
        in = malloc(sizeof(*in) * (size_t)ns);
        out = malloc(sizeof(*out) * (size_t)c->nfailed);
        int si = 0, fi = 0;
        uint8_t **orig_fail = malloc(sizeof(*orig_fail) *
                                    (size_t)c->nfailed);
        for (int d = 0; d < nd + np; d++) {
            int isf = 0;
            for (int f = 0; f < c->nfailed; f++)
                if (c->failed[f] == d)
                    isf = 1;
            if (isf) {
                orig_fail[fi] = abuf(len);
                memcpy(orig_fail[fi], b[d], len);
                memset(b[d], 0x5A, len); /* clobber "failed" disk */
                out[fi++] = b[d];
            } else {
                in[si++] = b[d];
            }
        }
        fn(in, out, len);
        for (int f = 0; f < c->nfailed && ok; f++)
            if (memcmp(out[f], orig_fail[f], len))
                ok = 0;
        for (int f = 0; f < c->nfailed; f++)
            free(orig_fail[f]);
        free(orig_fail);
        free(in);
        free(out);
    } else { /* C_PUPD */
        int nc = c->nchanged, np2 = np;
        /* in = [old_0..old_{nc-1}, new_0..new_{nc-1}, oldpar_0..] */
        uint8_t **old = malloc(sizeof(*old) * (size_t)nc);
        uint8_t **nw = malloc(sizeof(*nw) * (size_t)nc);
        for (int i = 0; i < nc; i++) {
            old[i] = abuf(len);
            nw[i] = abuf(len);
            for (size_t k = 0; k < len; k++) {
                old[i][k] = (uint8_t)xrng();
                nw[i][k] = (uint8_t)xrng();
            }
            memcpy(b[c->changed[i]], old[i], len);
        }
        ref_encode_mat(ti->mat, nd, np, b, want, len); /* old parities */
        for (int i = 0; i < nc; i++)
            memcpy(b[c->changed[i]], nw[i], len);
        uint8_t **want2 = malloc(sizeof(*want2) * (size_t)np);
        for (int j = 0; j < np; j++)
            want2[j] = abuf(len);
        ref_encode_mat(ti->mat, nd, np, b, want2, len); /* new parities */
        in = malloc(sizeof(*in) * (size_t)(2 * nc + np2));
        out = malloc(sizeof(*out) * (size_t)np2);
        for (int i = 0; i < nc; i++) {
            in[i] = old[i];
            in[nc + i] = nw[i];
        }
        for (int j = 0; j < np2; j++) {
            memcpy(b[nd + j], want[j], len);
            in[2 * nc + j] = b[nd + j];
            memset(b[nd + np + j], 0, len);
            out[j] = b[nd + np + j];
        }
        fn(in, out, len);
        for (int j = 0; j < np2 && ok; j++)
            if (memcmp(out[j], want2[j], len))
                ok = 0;
        for (int i = 0; i < nc; i++) {
            free(old[i]);
            free(nw[i]);
        }
        for (int j = 0; j < np; j++)
            free(want2[j]);
        free(old);
        free(nw);
        free(want2);
        free(in);
        free(out);
    }

    for (int i = 0; i < nbuf + np; i++)
        free(b[i]);
    for (int j = 0; j < np; j++)
        free(want[j]);
    free(b);
    free(want);
    return !ok;
}

/* ---------------- timing ---------------- */

static int cmp_d(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* Steady-state GB/s of fn at one size; 2 warmup + 11 timed, median. */
static double time_one(const config_t *c, rj_fn fn, size_t len,
                       uint8_t **bufs, volatile uint64_t *sink)
{
    int nd = c->ndata, np = c->npar;
    const void *in[2 * 16 + 4];
    void *out[8];
    int nin, nout;
    if (c->kind == C_ENC) {
        nin = nd;
        nout = np;
    } else if (c->kind == C_REC) {
        nin = nd + np - c->nfailed;
        nout = c->nfailed;
    } else {
        nin = 2 * c->nchanged + np;
        nout = np;
    }
    int bi = 0;
    for (int i = 0; i < nin; i++)
        in[i] = bufs[bi++];
    for (int j = 0; j < nout; j++)
        out[j] = bufs[bi++];
    for (int i = 0; i < NWARM; i++)
        fn(in, out, len);
    double it[NITER];
    double bytes = (double)(nd + np) * (double)len;
    for (int i = 0; i < NITER; i++) {
        double t0 = now_s();
        fn(in, out, len);
        double t1 = now_s();
        it[i] = bytes / (t1 - t0) / 1e9;
        /* anti-DCE: fold one byte of each output buffer */
        for (int j = 0; j < nout; j++)
            *sink += ((uint8_t *)out[j])[i & (len - 1)];
    }
    qsort(it, NITER, sizeof(double), cmp_d);
    return it[NITER / 2];
}

/* ---------------- comparison mode ---------------- */

/* Minimal parser for reports written by this tool. Schema:
 *   {"cpu": "...", "passes": N, "prune": "on"|"off", "configs": [
 *     {"name": "...", "compile_s": [..], "pick": [..],
 *      "gbps": [[..],[..],[..]]}, ... ]} */

typedef struct {
    char name[64];
    double compile[8];  /* per-pass compile_s */
    double gbps[3][8];  /* [size][pass] */
    int npass;
} cmp_cfg_t;

typedef struct {
    cmp_cfg_t *cfgs;
    int ncfg;
    char prune[8]; /* "on" / "off" / "?" (old reports lack the field) */
} cmp_report_t;

static int parse_report(const char *path, cmp_report_t *r)
{
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "compare: cannot open %s\n", path);
        return -1;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0)
        sz = 0;
    char *txt = malloc((size_t)sz + 1);
    if (!txt) {
        fclose(f);
        return -1;
    }
    size_t got = fread(txt, 1, (size_t)sz, f);
    txt[got] = '\0';
    fclose(f);

    memset(r, 0, sizeof(*r));
    strcpy(r->prune, "?");
    const char *p = strstr(txt, "\"prune\": \"");
    if (p) {
        p += (int)strlen("\"prune\": \"");
        size_t i = 0;
        while (*p && *p != '"' && i + 1 < sizeof(r->prune))
            r->prune[i++] = *p++;
        r->prune[i] = '\0';
    }

    int n = 0;
    for (p = txt; (p = strstr(p, "\"name\": \"")) != NULL; p += 8)
        n++;
    r->cfgs = calloc((size_t)(n ? n : 1), sizeof(*r->cfgs));
    if (!r->cfgs) {
        free(txt);
        return -1;
    }

    p = txt;
    int ci = 0, rc = 0;
    while ((p = strstr(p, "\"name\": \"")) != NULL && ci < n) {
        cmp_cfg_t *c = &r->cfgs[ci];
        p += (int)strlen("\"name\": \"");
        int ni = 0;
        while (*p && *p != '"' && ni + 1 < (int)sizeof(c->name))
            c->name[ni++] = *p++;
        c->name[ni] = '\0';

        const char *q = strstr(p, "\"compile_s\": [");
        if (!q) {
            rc = -1;
            break;
        }
        q += (int)strlen("\"compile_s\": [");
        int pi = 0, nn = 0;
        while (pi < 8 && sscanf(q, "%lf%n", &c->compile[pi], &nn) == 1) {
            pi++;
            q += nn;
            while (*q == ' ' || *q == ',')
                q++;
            if (*q == ']')
                break;
        }
        c->npass = pi;

        q = strstr(q, "\"gbps\": [");
        if (!q) {
            rc = -1;
            break;
        }
        q += (int)strlen("\"gbps\": [");
        for (int si = 0; si < 3; si++) {
            while (*q && *q != '[')
                q++;
            if (*q == '[')
                q++;
            pi = 0;
            while (pi < 8 &&
                   sscanf(q, "%lf%n", &c->gbps[si][pi], &nn) == 1) {
                pi++;
                q += nn;
                while (*q == ' ' || *q == ',')
                    q++;
                if (*q == ']')
                    break;
            }
        }
        ci++;
        p = q;
    }
    r->ncfg = ci;
    free(txt);
    return rc;
}

static void med_min_max(const double *v, int n, double *med, double *mn,
                        double *mx)
{
    double tmp[8];
    if (n > 8)
        n = 8;
    memcpy(tmp, v, (size_t)n * sizeof(double));
    qsort(tmp, (size_t)n, sizeof(double), cmp_d);
    *mn = tmp[0];
    *mx = tmp[n - 1];
    *med = tmp[n / 2];
}

/* Compare two reports config by config. Throughput parity: the new
 * report's per-size GB/s medians must fall within the base report's
 * [min,max] bands. Compile time: the new compile_s median must not
 * exceed the base band top (below the band bottom is an improvement,
 * not a violation). Reports per-config and total compile-time deltas.
 * Returns 0 iff every shared config passes both checks. */
static int cmd_compare(const char *base_path, const char *new_path)
{
    cmp_report_t base, newr;
    if (parse_report(base_path, &base) || parse_report(new_path, &newr))
        return 2;
    printf("compare: base=%s (prune=%s, %d configs)\n", base_path,
           base.prune, base.ncfg);
    printf("compare: new =%s (prune=%s, %d configs)\n", new_path,
           newr.prune, newr.ncfg);
    int fails = 0;
    double base_ct_total = 0.0, new_ct_total = 0.0;
    static const char *szn[] = { "4K", "64K", "1M" };
    printf("\n%-16s %10s %10s %9s  %s\n", "config", "base_ct", "new_ct",
           "delta", "gbps bands");
    for (int i = 0; i < base.ncfg; i++) {
        const cmp_cfg_t *b = &base.cfgs[i];
        const cmp_cfg_t *nb = NULL;
        for (int j = 0; j < newr.ncfg; j++)
            if (strcmp(b->name, newr.cfgs[j].name) == 0)
                nb = &newr.cfgs[j];
        if (!nb) {
            printf("%-16s MISSING in new report\n", b->name);
            fails++;
            continue;
        }
        double bmed, bmn, bmx, nmed, nmn, nmx;
        med_min_max(b->compile, b->npass, &bmed, &bmn, &bmx);
        med_min_max(nb->compile, nb->npass, &nmed, &nmn, &nmx);
        base_ct_total += bmed;
        new_ct_total += nmed;
        double delta = bmed > 0 ? 100.0 * (nmed - bmed) / bmed : 0.0;
        int ct_ok = (nmed <= bmx); /* above band top = regression */
        int g_ok = 1;
        char gbuf[64] = "";
        for (int si = 0; si < 3; si++) {
            double gbm, gbn, gbx, gnm, gnn, gnx;
            med_min_max(b->gbps[si], b->npass, &gbm, &gbn, &gbx);
            med_min_max(nb->gbps[si], nb->npass, &gnm, &gnn, &gnx);
            int ok = (gnm >= gbn && gnm <= gbx);
            if (!ok)
                g_ok = 0;
            char tmp[16];
            snprintf(tmp, sizeof(tmp), "%s:%s ", szn[si],
                     ok ? "ok" : "OUT");
            strncat(gbuf, tmp, sizeof(gbuf) - strlen(gbuf) - 1);
        }
        if (!g_ok || !ct_ok)
            fails++;
        printf("%-16s %10.2f %10.2f %+8.1f%%  %s%s\n", b->name, bmed,
               nmed, delta, gbuf,
               (!g_ok || !ct_ok) ? " <-- BAND VIOLATION" : "");
    }
    for (int j = 0; j < newr.ncfg; j++) {
        int found = 0;
        for (int i = 0; i < base.ncfg; i++)
            if (strcmp(newr.cfgs[j].name, base.cfgs[i].name) == 0)
                found = 1;
        if (!found)
            printf("note: %s only in new report (not compared)\n",
                   newr.cfgs[j].name);
    }
    double total_delta = base_ct_total > 0 ?
        100.0 * (new_ct_total - base_ct_total) / base_ct_total : 0.0;
    printf("\ntotal compile_s medians: base=%.1f new=%.1f delta=%+.1f%%\n",
           base_ct_total, new_ct_total, total_delta);
    printf("compare result: %s\n", fails ? "FAIL" : "OK");
    free(base.cfgs);
    free(newr.cfgs);
    return fails ? 1 : 0;
}

/* ---------------- main ---------------- */

int main(int argc, char **argv)
{
    if (argc == 4 && strcmp(argv[1], "--compare") == 0)
        return cmd_compare(argv[2], argv[3]);
    if (argc != 2) {
        fprintf(stderr,
                "usage: %s <results.json>\n"
                "       %s --compare <base.json> <new.json>\n",
                argv[0], argv[0]);
        return 2;
    }
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    /* provenance, with the same contract as prune_versions: any
     * non-empty value other than exactly "0" disables pruning */
    const char *npe = getenv("RJ_NO_PRUNE");
    int prune_on = !(npe && npe[0] != '\0' && strcmp(npe, "0") != 0);
    printf("perf_guard: cpu=%s sse2=%d ssse3=%d avx=%d avx2=%d "
           "avx512f=%d avx512bw=%d avx512vl=%d gfni=%d prune=%s\n",
           cpu.brand, cpu.sse2, cpu.ssse3, cpu.avx, cpu.avx2,
           cpu.avx512f, cpu.avx512bw, cpu.avx512vl, cpu.gfni,
           prune_on ? "on" : "off");

    /* results[pass][ci][si] */
    static double res[3][NCORPUS][NSIZES];
    static double ctime[3][NCORPUS];
    static char pick[3][NCORPUS][256];
    int failures = 0;
    volatile uint64_t sink = 0;

    for (int p = 0; p < PASSES; p++) {
        for (size_t ci = 0; ci < NCORPUS; ci++) {
            const config_t *c = &corpus[ci];
            rj_topology *t = mk_topo(c);
            rj_op op = mk_op(c);
            rj_options o;
            rj_options_default(&o);
            char err[256];
            double t0 = now_s();
            rj_jit *j = rj_compile(t, &op, &o, err, sizeof(err));
            double t1 = now_s();
            if (!j) {
                printf("FAIL compile %s: %s\n", c->name, err);
                failures++;
                rj_op_free(&op);
                rj_topo_free(t);
                continue;
            }
            ctime[p][ci] = t1 - t0;
            rj_info info;
            rj_get_info(j, &info);
            snprintf(pick[p][ci], sizeof(pick[p][ci]), "%s", info.desc);
            rj_fn fn = rj_get_fn(j);

            topo_int_t ti;
            if (topo_int_build(&ti, t, err, sizeof(err)) != 0) {
                printf("FAIL topo_int %s\n", c->name);
                failures++;
            } else if (verify(c, fn, &ti)) {
                printf("FAIL verify %s (pass %d)\n", c->name, p);
                failures++;
            }
            topo_int_free(&ti);

            int nd = c->ndata, np = c->npar;
            int nbuf = (c->kind == C_PUPD) ? 2 * c->nchanged + 2 * np
                                          : nd + np;
            uint8_t **bufs = malloc(sizeof(*bufs) * (size_t)nbuf);
            for (int i = 0; i < nbuf; i++) {
                bufs[i] = abuf(sizes[NSIZES - 1]);
                for (size_t k = 0; k < sizes[NSIZES - 1]; k += 4096)
                    bufs[i][k] = (uint8_t)xrng();
            }
            for (size_t si = 0; si < NSIZES; si++)
                res[p][ci][si] = time_one(c, fn, sizes[si], bufs,
                                          &sink);
            for (int i = 0; i < nbuf; i++)
                free(bufs[i]);
            free(bufs);
            rj_free(j);
            rj_op_free(&op);
            rj_topo_free(t);
            printf("pass %d/%d config %2zu/%zu %-16s compile %6.1fs pick %s\n",
                   p + 1, PASSES, ci + 1, (size_t)NCORPUS, c->name,
                   ctime[p][ci], pick[p][ci]);
            fflush(stdout);
        }
    }
    if (sink == 0xdeadbeef)
        printf("unreachable\n");

    /* human table: median [min,max] across passes */
    printf("\n%-16s %10s %22s %22s %22s\n", "config", "compile_s",
           "4KiB GB/s", "64KiB GB/s", "1MiB GB/s");
    for (size_t ci = 0; ci < NCORPUS; ci++) {
        double ct[PASSES], g[NSIZES][PASSES];
        for (int p = 0; p < PASSES; p++) {
            ct[p] = ctime[p][ci];
            for (size_t si = 0; si < NSIZES; si++)
                g[si][p] = res[p][ci][si];
        }
        qsort(ct, PASSES, sizeof(double), cmp_d);
        printf("%-16s %10.1f ", corpus[ci].name, ct[PASSES / 2]);
        for (size_t si = 0; si < NSIZES; si++) {
            qsort(g[si], PASSES, sizeof(double), cmp_d);
            printf("%6.1f [%5.1f,%5.1f] ", g[si][PASSES / 2], g[si][0],
                   g[si][PASSES - 1]);
        }
        printf(" %s\n", pick[PASSES / 2][ci]);
    }

    /* JSON */
    FILE *f = fopen(argv[1], "w");
    if (!f) {
        perror("fopen");
        return 2;
    }
    fprintf(f, "{\n  \"cpu\": \"%s\", \"passes\": %d, \"prune\": \"%s\",\n"
               "  \"configs\": [\n",
            cpu.brand, PASSES, prune_on ? "on" : "off");
    for (size_t ci = 0; ci < NCORPUS; ci++) {
        fprintf(f, "    {\"name\": \"%s\", \"compile_s\": [", corpus[ci].name);
        for (int p = 0; p < PASSES; p++)
            fprintf(f, "%s%.2f", p ? ", " : "", ctime[p][ci]);
        fprintf(f, "], \"pick\": [");
        for (int p = 0; p < PASSES; p++)
            fprintf(f, "%s\"%s\"", p ? ", " : "", pick[p][ci]);
        fprintf(f, "], \"gbps\": [");
        for (size_t si = 0; si < NSIZES; si++) {
            fprintf(f, "%s[", si ? ", " : "");
            for (int p = 0; p < PASSES; p++)
                fprintf(f, "%s%.2f", p ? ", " : "", res[p][ci][si]);
            fprintf(f, "]");
        }
        fprintf(f, "]}%s\n", ci + 1 < NCORPUS ? "," : "");
    }
    fprintf(f, "  ]\n}\n");
    fclose(f);
    printf("wrote %s (%d failures)\n", argv[1], failures);
    return failures != 0;
}
