/* tests/test_planner_fuzz.c
 *
 * Planner fuzzing harness (Lane 2 item #45): systematic randomized
 * validation of the raidjit planner (src/plan.c, src/prune.c) through
 * the public API (src/api.c, include/raidjit.h).
 *
 * What it does
 * -------------
 * Each iteration deterministically generates a random valid topology
 * (RS, RAID-5/6, RAID-0/1, the RDP/EVENODD/Liberation XOR array codes,
 * and custom matrices covering the P_XOR / P_MULACC / P_HORNER plan
 * paths) and a random op (encode, recover with a random -- possibly
 * invalid -- failed-disk list, parity update with a random changed
 * list). For every (topology, op) it then checks:
 *
 *   (a) no crash / no garbage: plan_build and rj_compile either succeed
 *       or fail cleanly with a non-empty err string. Inputs that are
 *       invalid by contract (more failures than parities, out-of-range
 *       disk index, empty failed list, recover on a parity-less
 *       topology, mathematically singular recovery systems) must be
 *       rejected cleanly -- never crash, never return a NULL err, never
 *       return a non-NULL jit for an invalid op.
 *   (b) differential correctness: the compiled function's output must
 *       match the scalar reference (src/ref.c, via plan_build+ref_run)
 *       AND an independent scalar GF(2^8) oracle written in this file
 *       (matrix multiply for encode, original disk contents for
 *       recover, the delta formula for parity update) on random data.
 *   (c) plan invariants: plan kind in range, sane n_in/n_out, non-NULL
 *       mat (n_out > 0), non-NULL horner_c for P_HORNER, nchanged >= 1
 *       for P_DELTA, op-specific layout checks (encode: n_in=ndata,
 *       n_out=npar; recover: n_in=survivors, n_out=nfailed; parity
 *       update: P_DELTA with n_in=2*nchanged+npar, n_out=npar), non-NULL
 *       function pointer from rj_get_fn (stable across calls), and sane
 *       rj_info fields (strategy string matches the plan kind,
 *       unroll >= 1, lmul in {1,2,4,8}, code_size > 0).
 *
 * Usage
 * -----
 *   ./test_planner_fuzz [iters] [seed]
 *   RJ_FUZZ_SEED=1234 ./test_planner_fuzz 500
 *   RJ_FUZZ_ITERS=500 ./test_planner_fuzz
 *
 * The seed comes from argv[2], else $RJ_FUZZ_SEED, else a fixed default
 * (deterministic across runs). The iteration count comes from argv[1],
 * else $RJ_FUZZ_ITERS, else 200. The PRNG is a single xorshift64* stream
 * seeded once, so (seed, iters) fully determines the run: a failure
 * prints a REPLAY line that reproduces it exactly.
 *
 * Non-tautology proof
 * -------------------
 * RJ_FUZZ_SABOTAGE=1 flips one byte of the JIT output buffer before the
 * differential comparison. The harness MUST then report FAIL and exit
 * nonzero; if it ever passes with sabotage on, the differential check is
 * dead. (Checked by the author: `RJ_FUZZ_SABOTAGE=1 ./test_planner_fuzz`
 * fails with a differential mismatch and exits nonzero as required.)
 *
 * Failure output
 * --------------
 * On the first failure the harness prints the seed, iteration index,
 * topology family + parameters (+ matrix for custom topologies), the
 * op (failed/changed lists), the stripe length, what was being checked,
 * and a REPLAY command. Exit status is 1; 0 means all iterations passed.
 *
 * Cost control: small topologies (ndata <= 8; npar <= 3 for the
 * RS/RAID/custom families; XOR codes with R <= 5 rows per disk), two
 * short stripe lengths per case, compile with benchmark=0
 * (correctness only -- no benchmarking anywhere in this file). The
 * default 200 iterations finish in well under two minutes.
 *
 * One cost carve-out, documented so it is not mistaken for a gap:
 * recover on XOR-code topologies compiles with RJ_NO_SURVIVOR_SELECT=1
 * (the legacy all-rows path). The A5 survivor-set selection enumerates
 * parity-row subsets and builds every candidate per subset -- on an
 * expanded XOR topology (npar = 8..10) a single whole-disk recovery
 * costs ~17s, which would blow the iteration budget. Survivor selection
 * itself is still fuzzed on every small (npar <= 3) topology, where the
 * enumeration is trivially cheap, and it has its own dedicated test
 * (tests/test_survivor.c).
 *
 * Slow-case reporting: RJ_FUZZ_SLOW_MS=N prints any case taking longer
 * than N milliseconds (0 = off). Useful to spot planner pathologies;
 * never affects pass/fail.
 */

#include "../include/raidjit.h"
#include "../src/plan.h"
#include "../src/gf.h"
#include "../src/ref.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ------------------------------------------------------------------ */
/* deterministic PRNG: xorshift64*                                     */
/* ------------------------------------------------------------------ */

static uint64_t rng_state;

static uint64_t xrng(void)
{
    uint64_t x = rng_state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rng_state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

/* uniform in [0, n) */
static uint64_t rng_below(uint64_t n)
{
    return xrng() % n;
}

/* ------------------------------------------------------------------ */
/* failure reporting                                                   */
/* ------------------------------------------------------------------ */

static unsigned long g_seed;
static long g_iter; /* current iteration, for the replay line */
static int g_sabotage;
static long g_slow_ms; /* RJ_FUZZ_SLOW_MS: report cases slower than this */

static uint64_t now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000 + (uint64_t)ts.tv_nsec / 1000000;
}

static const char *topo_fam_name(int fam)
{
    switch (fam) {
    case 0: return "raid1";
    case 1: return "raid5";
    case 2: return "raid6_rs";
    case 3: return "rs";
    case 4: return "rdp";
    case 5: return "evenodd";
    case 6: return "liberation";
    case 7: return "custom_xor";
    case 8: return "custom_vand";
    case 9: return "raid0";
    default: return "?";
    }
}

typedef struct {
    int topo_fam; /* 0..9, see topo_fam_name */
    int p1, p2;   /* family params (ndata / npar / prime / k) */
    uint8_t cmat[3 * 8]; /* custom matrix bytes (families 7, 8) */
    int op_kind;  /* 0 = encode, 1 = recover, 2 = parity_update */
    int failed[RJ_MAX_DISKS];
    int nfailed;
    int null_failed; /* pass NULL as the failed list (still nfailed > 0) */
    int changed[RJ_MAX_DISKS];
    int nchanged;
    size_t lens[2];
    int nlens;
} fuzz_case_t;

static void describe_case(const fuzz_case_t *c)
{
    printf("  topo: %s", topo_fam_name(c->topo_fam));
    switch (c->topo_fam) {
    case 1:
    case 2:
    case 9:
        printf(" ndata=%d", c->p1);
        break;
    case 3:
        printf(" ndata=%d npar=%d", c->p1, c->p2);
        break;
    case 4:
    case 5:
        printf(" p=%d", c->p1);
        break;
    case 6:
        printf(" k=%d", c->p1);
        break;
    case 7:
    case 8:
        printf(" ndata=%d npar=%d mat=", c->p1, c->p2);
        for (int j = 0; j < c->p2; j++) {
            printf("%s[", j ? " " : "");
            for (int i = 0; i < c->p1; i++)
                printf("%02x%s", c->cmat[j * c->p1 + i],
                       i + 1 < c->p1 ? "," : "]");
        }
        break;
    default:
        break;
    }
    printf("\n  op: %s", c->op_kind == 0   ? "encode"
                       : c->op_kind == 1 ? "recover"
                                         : "parity_update");
    if (c->op_kind == 1) {
        if (c->null_failed)
            printf(" failed=NULL nfailed=%d", c->nfailed);
        else {
            printf(" failed={");
            for (int i = 0; i < c->nfailed; i++)
                printf("%s%d", i ? "," : "", c->failed[i]);
            printf("}");
        }
    } else if (c->op_kind == 2) {
        printf(" changed={");
        for (int i = 0; i < c->nchanged; i++)
            printf("%s%d", i ? "," : "", c->changed[i]);
        printf("}");
    }
    printf("\n  lens:");
    for (int i = 0; i < c->nlens; i++)
        printf(" %zu", c->lens[i]);
    printf("\n");
}

#define FAIL_AT(c, fmt, ...)                                              \
    do {                                                                  \
        printf("FAIL iter=%ld seed=%lu\n", g_iter, g_seed);                \
        describe_case(c);                                                 \
        printf("  check: " fmt "\n", ##__VA_ARGS__);                      \
        printf("REPLAY: RJ_FUZZ_SEED=%lu ./test_planner_fuzz 200 %lu\n",   \
               g_seed, g_seed);                                           \
        fflush(stdout);                                                   \
        exit(1);                                                          \
    } while (0)

/* ------------------------------------------------------------------ */
/* small helpers                                                       */
/* ------------------------------------------------------------------ */

static void *abuf(size_t n)
{
    void *p = NULL;
    if (posix_memalign(&p, 64, n ? n : 64)) {
        fprintf(stderr, "fuzz: out of memory\n");
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

static int int_cmp(const void *a, const void *b)
{
    return *(const int *)a - *(const int *)b;
}

/* sort + dedupe in place; returns the deduped count */
static int sort_dedupe(int *v, int n)
{
    qsort(v, (size_t)n, sizeof(int), int_cmp);
    int w = 0;
    for (int i = 0; i < n; i++)
        if (i == 0 || v[i] != v[i - 1])
            v[w++] = v[i];
    return w;
}

/* independent scalar oracle: par[j][k] = sum_i d[i][k]*mat[j*nd+i].
 * Written in this file on purpose -- independent of src/ref.c, so the
 * two oracles cross-check each other as well as the JIT. */
static void scalar_encode(const uint8_t *mat, int nd, int np,
                          uint8_t **d, uint8_t **p, size_t len)
{
    for (int j = 0; j < np; j++)
        for (size_t k = 0; k < len; k++) {
            uint8_t a = 0;
            for (int i = 0; i < nd; i++) {
                uint8_t c = mat[j * nd + i];
                if (c)
                    a ^= gf_mul(d[i][k], c);
            }
            p[j][k] = a;
        }
}

/* ------------------------------------------------------------------ */
/* case generation                                                     */
/* ------------------------------------------------------------------ */

static const int primes_small[] = { 3, 5 }; /* R <= 5 keeps the expanded
                                              * row model small: np <= 8,
                                              * compile stays cheap */

static void gen_topology(fuzz_case_t *c)
{
    uint64_t r = rng_below(100);
    if (r < 5) {
        c->topo_fam = 0; /* raid1 */
    } else if (r < 20) {
        c->topo_fam = 1; /* raid5 */
        c->p1 = 2 + (int)rng_below(7);
    } else if (r < 35) {
        c->topo_fam = 2; /* raid6_rs */
        c->p1 = 3 + (int)rng_below(6);
    } else if (r < 50) {
        c->topo_fam = 3; /* rs */
        c->p1 = 2 + (int)rng_below(7);
        c->p2 = 1 + (int)rng_below(3);
    } else if (r < 56) {
        c->topo_fam = 4; /* rdp */
        c->p1 = primes_small[rng_below(2)];
    } else if (r < 62) {
        c->topo_fam = 5; /* evenodd */
        c->p1 = primes_small[rng_below(2)];
    } else if (r < 66) {
        c->topo_fam = 6; /* liberation */
        c->p1 = 2 + (int)rng_below(3); /* k <= 4: w <= 5, R <= 5 */
    } else if (r < 77) {
        c->topo_fam = 7; /* custom 0/1 matrix -> P_XOR */
        c->p1 = 2 + (int)rng_below(7);
        c->p2 = 1 + (int)rng_below(2);
        for (int j = 0; j < c->p2; j++)
            for (int i = 0; i < c->p1; i++)
                c->cmat[j * c->p1 + i] = (uint8_t)rng_below(2);
    } else if (r < 92) {
        c->topo_fam = 8; /* custom Vandermonde -> P_HORNER */
        c->p1 = 2 + (int)rng_below(7);
        c->p2 = 1 + (int)rng_below(3);
        /* same Horner-order Vandermonde form as the RS default, so the
         * matrix is MDS inside the (ndata<=51, npar<=3) envelope */
        for (int j = 0; j < c->p2; j++)
            for (int i = 0; i < c->p1; i++)
                c->cmat[j * c->p1 + i] =
                    gf_pow(2, (unsigned)(j * (c->p1 - 1 - i)));
    } else {
        c->topo_fam = 9; /* raid0 */
        c->p1 = 2 + (int)rng_below(5);
    }
}

static rj_topology *build_topo(const fuzz_case_t *c)
{
    switch (c->topo_fam) {
    case 0:
        return rj_topo_raid1();
    case 1:
        return rj_topo_raid5(c->p1);
    case 2:
        return rj_topo_raid6_rs(c->p1);
    case 3:
        return rj_topo_rs(c->p1, c->p2);
    case 4:
        return rj_topo_rdp(c->p1);
    case 5:
        return rj_topo_evenodd(c->p1);
    case 6:
        return rj_topo_liberation(c->p1);
    case 7:
    case 8:
        return rj_topo_custom(c->p1, c->p2, c->cmat);
    case 9:
        return rj_topo_raid0(c->p1);
    default:
        return NULL;
    }
}

/* pick `cnt` distinct values in [0, bound); result unsorted */
static void pick_distinct(int *out, int cnt, int bound)
{
    int n = 0;
    while (n < cnt) {
        int v = (int)rng_below((uint64_t)bound);
        int dup = 0;
        for (int i = 0; i < n; i++)
            if (out[i] == v)
                dup = 1;
        if (!dup)
            out[n++] = v;
    }
}

static void gen_op(fuzz_case_t *c, int nd, int np)
{
    int N = nd + np;
    uint64_t r = rng_below(100);
    if (r < 40) {
        c->op_kind = 0; /* encode */
        return;
    }
    if (r < 80 && np > 0) {
        /* recover */
        c->op_kind = 1;
        c->null_failed = 0;
        int R = 0; /* XOR-code rows per disk, 0 for the rest */
        /* set by the caller via c->p fields? No -- recompute here would
         * need the topology; instead the caller patches xor R after
         * building the topology. Gen non-xor here, patch below. */
        (void)R;
        uint64_t s = rng_below(100);
        int maxnf = np < 3 ? np : 3;
        if (s < 60) {
            /* valid count, valid indices (maybe unsorted/dupes) */
            c->nfailed = 1 + (int)rng_below((uint64_t)maxnf);
            for (int i = 0; i < c->nfailed; i++)
                c->failed[i] = (int)rng_below((uint64_t)N);
        } else if (s < 72) {
            /* too many failures: clean rejection expected */
            c->nfailed = np + 1;
            pick_distinct(c->failed, c->nfailed, N);
        } else if (s < 82) {
            /* out-of-range index: clean rejection expected */
            c->nfailed = 1 + (int)rng_below((uint64_t)maxnf);
            pick_distinct(c->failed, c->nfailed, N);
            c->failed[(int)rng_below((uint64_t)c->nfailed)] = N;
        } else if (s < 88) {
            /* empty failed list: clean rejection expected */
            c->nfailed = 0;
        } else if (s < 94) {
            /* NULL list with nonzero count: clean rejection expected */
            c->nfailed = 1 + (int)rng_below((uint64_t)maxnf);
            c->null_failed = 1;
        } else {
            /* duplicates: dedupe makes it valid */
            c->nfailed = 2;
            c->failed[0] = c->failed[1] = (int)rng_below((uint64_t)N);
        }
        return;
    }
    /* parity update (or recover fallback when np == 0 handled above) */
    if (np == 0) {
        c->op_kind = 0; /* raid0: only encode is meaningful */
        return;
    }
    c->op_kind = 2;
    c->nchanged = 1 + (int)rng_below((uint64_t)(nd < 3 ? nd : 3));
    pick_distinct(c->changed, c->nchanged, nd);
}

/* For XOR-code topologies, whole-disk failures must map to the expanded
 * row model: data disk d -> rows d*R..d*R+R-1; parity "disks" P and Q ->
 * rows nd+r and nd+R+r. Rewrites a recover case's failed list when the
 * topology is an XOR code. Called after the topology is built. */
static void patch_xor_recover(fuzz_case_t *c, const rj_topology *t, int nd)
{
    int R = rj_xorcode_rows(t);
    if (R == 0 || c->op_kind != 1 || c->null_failed || c->nfailed == 0)
        return;
    int k = nd / R; /* data disks */
    int out[RJ_MAX_DISKS];
    int nout = 0;
    /* map each requested failed disk index (interpreted as a whole-disk
     * index in [0, k+2)) to its expanded row range */
    int seen[RJ_MAX_DISKS];
    int nseen = 0;
    for (int i = 0; i < c->nfailed && nout + R <= RJ_MAX_DISKS; i++) {
        int d = c->failed[i] % (k + 2); /* keep in whole-disk range */
        int dup = 0;
        for (int s = 0; s < nseen; s++)
            if (seen[s] == d)
                dup = 1;
        if (dup)
            continue;
        seen[nseen++] = d;
        int base = d < k ? d * R : nd + (d - k) * R;
        for (int rr = 0; rr < R; rr++)
            out[nout++] = base + rr;
    }
    if (nout == 0) { /* degenerate: fall back to a single valid row */
        out[nout++] = 0;
    }
    memcpy(c->failed, out, sizeof(int) * (size_t)nout);
    c->nfailed = nout;
}

static rj_op build_op(const fuzz_case_t *c)
{
    switch (c->op_kind) {
    case 1:
        return rj_op_recover(c->null_failed ? NULL : c->failed, c->nfailed);
    case 2:
        return rj_op_parity_update(c->changed, c->nchanged);
    default:
        return rj_op_encode();
    }
}

/* ------------------------------------------------------------------ */
/* plan invariant checks                                               */
/* ------------------------------------------------------------------ */

static void check_plan(const fuzz_case_t *c, const plan_t *pl, int nd, int np,
                       int nfd, int nc)
{
    if (pl->kind < P_XOR || pl->kind > P_DELTA)
        FAIL_AT(c, "plan kind %d out of range", (int)pl->kind);
    if (pl->n_in <= 0 || pl->n_in > RJ_MAX_DISKS)
        FAIL_AT(c, "plan n_in=%d insane", pl->n_in);
    if (pl->n_out < 0 || pl->n_out > RJ_MAX_DISKS)
        FAIL_AT(c, "plan n_out=%d insane", pl->n_out);
    if (pl->n_out > 0 && pl->mat == NULL)
        FAIL_AT(c, "plan n_out=%d but mat is NULL", pl->n_out);
    if (pl->kind == P_HORNER && pl->horner_c == NULL)
        FAIL_AT(c, "P_HORNER plan with NULL horner_c");
    if (pl->kind == P_DELTA && pl->nchanged < 1)
        FAIL_AT(c, "P_DELTA plan with nchanged=%d", pl->nchanged);
    if (pl->name[0] == '\0')
        FAIL_AT(c, "plan name empty");
    switch (c->op_kind) {
    case 0: /* encode */
        if (pl->n_in != nd || pl->n_out != np)
            FAIL_AT(c, "encode plan layout n_in=%d n_out=%d, want %d/%d",
                    pl->n_in, pl->n_out, nd, np);
        if (pl->kind == P_DELTA)
            FAIL_AT(c, "encode plan is P_DELTA");
        break;
    case 1: /* recover */
        if (pl->n_in != nd + np - nfd || pl->n_out != nfd)
            FAIL_AT(c, "recover plan layout n_in=%d n_out=%d, want %d/%d",
                    pl->n_in, pl->n_out, nd + np - nfd, nfd);
        if (pl->kind == P_DELTA)
            FAIL_AT(c, "recover plan is P_DELTA");
        break;
    case 2: /* parity update */
        if (pl->kind != P_DELTA)
            FAIL_AT(c, "parity_update plan kind=%d, want P_DELTA",
                    (int)pl->kind);
        if (pl->n_in != 2 * nc + np || pl->n_out != np)
            FAIL_AT(c, "delta plan layout n_in=%d n_out=%d, want %d/%d",
                    pl->n_in, pl->n_out, 2 * nc + np, np);
        break;
    }
}

/* ------------------------------------------------------------------ */
/* one fuzz case                                                       */
/* ------------------------------------------------------------------ */

static const size_t len_pool[] = { 0, 1, 7, 16, 17, 63, 64, 65, 100, 257,
                                   1000 };
static const int nlen_pool = sizeof(len_pool) / sizeof(len_pool[0]);

static void run_case(fuzz_case_t *c, rj_options *opts,
                     unsigned *nrun, unsigned *nskipped,
                     unsigned *nrejected)
{
    rj_topology *t = build_topo(c);
    if (!t) {
        /* constructor refused: valid (out-of-envelope params) */
        (*nskipped)++;
        return;
    }
    int nd = rj_topo_ndata(t);
    int np = rj_topo_npar(t);

    gen_op(c, nd, np);
    patch_xor_recover(c, t, nd);

    /* canonical deduped/sorted failed list, mirroring
     * plan_recover_failed_list */
    int Fd[RJ_MAX_DISKS];
    int nfd = 0;
    if (c->op_kind == 1 && !c->null_failed && c->nfailed > 0) {
        memcpy(Fd, c->failed, sizeof(int) * (size_t)c->nfailed);
        nfd = sort_dedupe(Fd, c->nfailed);
    }

    /* expectation: is this op invalid by contract? */
    int expect_fail = 0;
    if (c->op_kind == 1) {
        if (c->null_failed || c->nfailed < 1)
            expect_fail = 1;
        else if (nfd < 1 || np == 0 || nfd > np)
            expect_fail = 1;
        else
            for (int i = 0; i < nfd; i++)
                if (Fd[i] < 0 || Fd[i] >= nd + np)
                    expect_fail = 1;
    }

    rj_op op = build_op(c);
    if (op.oom)
        FAIL_AT(c, "op constructor OOM flag set");

    topo_int_t ti;
    char err[256];
    err[0] = '\0';
    if (topo_int_build(&ti, t, err, sizeof(err)) != 0)
        FAIL_AT(c, "topo_int_build failed on constructed topo: %s", err);

    /* plan_build (internal path, same as the library uses): either
     * succeeds with a sane plan or fails cleanly with err set. */
    plan_t pl;
    memset(&pl, 0, sizeof(pl));
    int plan_ok =
        plan_build(&ti, &op, topo_internal_name(t), &pl, err, sizeof(err)) ==
        0;
    if (plan_ok) {
        if (err[0] != '\0') {
            /* garbage: success must not leave a stale error */
            plan_free(&pl);
            FAIL_AT(c, "plan_build succeeded but err was set: %s", err);
        }
        check_plan(c, &pl, nd, np, nfd, c->nchanged);
    } else {
        if (err[0] == '\0')
            FAIL_AT(c, "plan_build failed with empty err (garbage)");
    }

    /* public API compile.
     *
     * Cost carve-out (see header): XOR-code recover uses the legacy
     * all-rows path -- the A5 survivor enumeration over npar = 8..10
     * expanded parity rows costs ~17s per whole-disk recovery and
     * would blow the iteration budget. Survivor selection is still
     * fuzzed on every small (npar <= 3) topology and covered by
     * tests/test_survivor.c. */
    int no_survivor = rj_xorcode_rows(t) > 0 && c->op_kind == 1;
    if (no_survivor)
        setenv("RJ_NO_SURVIVOR_SELECT", "1", 1);
    err[0] = '\0';
    rj_jit *j = rj_compile(t, &op, opts, err, sizeof(err));
    if (no_survivor)
        unsetenv("RJ_NO_SURVIVOR_SELECT");
    if (expect_fail) {
        if (j != NULL)
            FAIL_AT(c, "rj_compile accepted an invalid op");
        if (err[0] == '\0')
            FAIL_AT(c, "rj_compile rejected invalid op with empty err");
        (*nrejected)++;
        if (plan_ok)
            plan_free(&pl);
        topo_int_free(&ti);
        rj_op_free(&op);
        rj_topo_free(t);
        return;
    }
    if (j == NULL) {
        /* clean failure on a valid op: acceptable only with err set
         * (singular recovery system, no encodable candidate, ...);
         * an empty err is garbage. */
        if (err[0] == '\0')
            FAIL_AT(c, "rj_compile failed with empty err on valid op");
        (*nrejected)++;
        if (plan_ok)
            plan_free(&pl);
        topo_int_free(&ti);
        rj_op_free(&op);
        rj_topo_free(t);
        return;
    }
    if (!plan_ok) {
        /* compile succeeded where plan_build failed: the only legal
         * path is the A5 survivor subset selection picking a solvable
         * row subset of a singular all-rows system. The semantic
         * checks below still fully validate the compiled function. */
    }

    /* (c) function-pointer / info invariants */
    rj_fn fn = rj_get_fn(j);
    if (fn == NULL)
        FAIL_AT(c, "rj_get_fn returned NULL on success");
    if (rj_get_fn(j) != fn)
        FAIL_AT(c, "rj_get_fn unstable across calls");
    rj_info info;
    rj_get_info(j, &info);
    if (info.desc[0] == '\0')
        FAIL_AT(c, "rj_info desc empty");
    if (info.isa_name[0] == '\0')
        FAIL_AT(c, "rj_info isa_name empty");
    const char *want_strategy =
        pl.kind == P_XOR      ? "xor"
        : pl.kind == P_HORNER ? "horner"
        : pl.kind == P_DELTA  ? "delta"
                              : "mulacc";
    if (plan_ok && strcmp(info.strategy, want_strategy) != 0)
        FAIL_AT(c, "info.strategy=%s but plan kind=%s", info.strategy,
                want_strategy);
    if (strcmp(info.strategy, "xor") != 0 &&
        strcmp(info.strategy, "horner") != 0 &&
        strcmp(info.strategy, "mulacc") != 0 &&
        strcmp(info.strategy, "delta") != 0)
        FAIL_AT(c, "info.strategy=%s not a known strategy",
                info.strategy);
    if (info.unroll < 1)
        FAIL_AT(c, "info.unroll=%d insane", info.unroll);
    if (info.lmul != 1 && info.lmul != 2 && info.lmul != 4 && info.lmul != 8)
        FAIL_AT(c, "info.lmul=%d insane", info.lmul);
    if (info.code_size == 0)
        FAIL_AT(c, "info.code_size == 0");

    /* (b) differential correctness, per stripe length */
    int N = nd + np;
    for (int li = 0; li < c->nlens; li++) {
        size_t len = c->lens[li];
        uint8_t **all = malloc(sizeof(*all) * (size_t)N);
        uint8_t **keep = malloc(sizeof(*keep) * (size_t)N);
        if (!all || !keep)
            FAIL_AT(c, "harness OOM");
        for (int i = 0; i < N; i++) {
            all[i] = abuf(len);
            keep[i] = abuf(len);
            if (i < nd)
                randfill(all[i], len);
        }
        scalar_encode(ti.mat, nd, np, all, all + nd, len);
        for (int i = 0; i < N; i++)
            memcpy(keep[i], all[i], len);

        /* canonical changed list for parity update (constructor sorts) */
        int ch[RJ_MAX_DISKS];
        int nch = 0;
        if (c->op_kind == 2 && op.changed) {
            nch = op.nchanged;
            memcpy(ch, op.changed, sizeof(int) * (size_t)nch);
        }

        /* input/output pointer layout per the rj_fn caller contract */
        const void **in = NULL;
        void **out = NULL;
        uint8_t **ref = NULL; /* ref.c oracle outputs */
        int nin = 0, nout = 0;
        if (c->op_kind == 0) {
            nin = nd;
            nout = np;
        } else if (c->op_kind == 1) {
            nin = N - nfd;
            nout = nfd;
        } else {
            nin = 2 * nch + np;
            nout = np;
        }
        in = malloc(sizeof(*in) * (size_t)(nin ? nin : 1));
        out = malloc(sizeof(*out) * (size_t)(nout ? nout : 1));
        ref = malloc(sizeof(*ref) * (size_t)(nout ? nout : 1));
        if (!in || !out || !ref)
            FAIL_AT(c, "harness OOM");
        for (int i = 0; i < nout; i++) {
            out[i] = abuf(len);
            ref[i] = abuf(len);
            memset(out[i], 0xA5, len); /* catch missing writes */
        }

        if (c->op_kind == 0) {
            for (int i = 0; i < nd; i++)
                in[i] = all[i];
            for (int i = 0; i < np; i++) {
                free(out[i]); /* prologue abuf unused here */
                out[i] = all[nd + i]; /* parity buffers get overwritten */
            }
            fn(in, out, len);
            if (g_sabotage && nout > 0 && len > 0)
                ((uint8_t *)out[0])[0] ^= 0xFF;
            /* oracle 1: src/ref.c on the built plan */
            if (plan_ok) {
                ref_run(&pl, (const uint8_t **)in,
                        (uint8_t **)ref, len);
                for (int i = 0; i < np; i++)
                    if (memcmp(out[i], ref[i], len) != 0)
                        FAIL_AT(c,
                                "encode len=%zu par%d: jit != ref.c "
                                "(%s)",
                                len, i, info.desc);
            }
            /* oracle 2: independent scalar encode */
            {
                uint8_t **exp = malloc(sizeof(*exp) * (size_t)(np ? np : 1));
                for (int i = 0; i < np; i++)
                    exp[i] = abuf(len);
                scalar_encode(ti.mat, nd, np, all, exp, len);
                for (int i = 0; i < np; i++)
                    if (memcmp(out[i], exp[i], len) != 0)
                        FAIL_AT(c,
                                "encode len=%zu par%d: jit != scalar "
                                "oracle (%s)",
                                len, i, info.desc);
                for (int i = 0; i < np; i++)
                    free(exp[i]);
                free(exp);
            }
            /* data disks must be untouched */
            for (int i = 0; i < nd; i++)
                if (memcmp(all[i], keep[i], len) != 0)
                    FAIL_AT(c, "encode len=%zu: data disk %d clobbered",
                            len, i);
        } else if (c->op_kind == 1) {
            /* wipe failed disks, then rebuild the survivor/fail lists */
            for (int f = 0; f < nfd; f++)
                memset(all[Fd[f]], 0x5A, len);
            int ni = 0, oi = 0, fi = 0;
            for (int dsk = 0; dsk < N; dsk++) {
                if (fi < nfd && Fd[fi] == dsk) {
                    free(out[oi]); /* prologue abuf unused here */
                    out[oi] = all[dsk];
                    if (plan_ok) {
                        free(ref[oi]); /* prologue abuf, replaced */
                        ref[oi] = abuf(len);
                    }
                    oi++;
                    fi++;
                } else {
                    in[ni++] = all[dsk];
                }
            }
            fn(in, out, len);
            if (g_sabotage && nout > 0 && len > 0)
                ((uint8_t *)out[0])[0] ^= 0xFF;
            if (plan_ok) {
                uint8_t **refin =
                    malloc(sizeof(*refin) * (size_t)(nin ? nin : 1));
                for (int i = 0; i < nin; i++)
                    refin[i] = abuf(len);
                for (int i = 0; i < nin; i++)
                    memcpy(refin[i], in[i], len);
                ref_run(&pl, (const uint8_t **)refin, ref, len);
                for (int i = 0; i < nfd; i++)
                    if (memcmp(out[i], ref[i], len) != 0)
                        FAIL_AT(c,
                                "recover len=%zu disk%d: jit != ref.c "
                                "(%s)",
                                len, Fd[i], info.desc);
                for (int i = 0; i < nin; i++)
                    free(refin[i]);
                free(refin);
            }
            /* semantic oracle: reconstructed == original contents */
            for (int i = 0; i < nfd; i++)
                if (memcmp(out[i], keep[Fd[i]], len) != 0)
                    FAIL_AT(c,
                            "recover len=%zu disk%d: jit != original "
                            "data (%s)",
                            len, Fd[i], info.desc);
            /* survivors must be untouched */
            {
                int si = 0, fi2 = 0;
                for (int dsk = 0; dsk < N; dsk++) {
                    if (fi2 < nfd && Fd[fi2] == dsk) {
                        fi2++;
                    } else {
                        if (memcmp(in[si], keep[dsk], len) != 0)
                            FAIL_AT(c,
                                    "recover len=%zu: survivor disk %d "
                                    "clobbered",
                                    len, dsk);
                        si++;
                    }
                }
            }
        } else {
            /* parity update: in = [old nc, new nc, oldpar np] */
            uint8_t **old = malloc(sizeof(*old) * (size_t)(nch ? nch : 1));
            uint8_t **newd = malloc(sizeof(*newd) * (size_t)(nch ? nch : 1));
            for (int i = 0; i < nch; i++) {
                old[i] = abuf(len);
                newd[i] = abuf(len);
                randfill(old[i], len);
                randfill(newd[i], len);
            }
            for (int i = 0; i < nch; i++)
                in[i] = old[i];
            for (int i = 0; i < nch; i++)
                in[nch + i] = newd[i];
            for (int j = 0; j < np; j++)
                in[2 * nch + j] = all[nd + j]; /* old parities */
            /* out[] already holds 0xA5-filled standalone buffers from the
             * prologue -- the "missing write" tripwire. */
            fn(in, out, len);
            if (g_sabotage && nout > 0 && len > 0)
                ((uint8_t *)out[0])[0] ^= 0xFF;
            /* oracle 1: src/ref.c on the built plan */
            if (plan_ok) {
                uint8_t **refin =
                    malloc(sizeof(*refin) * (size_t)(nin ? nin : 1));
                for (int i = 0; i < nin; i++) {
                    refin[i] = abuf(len);
                    memcpy(refin[i], in[i], len);
                }
                ref_run(&pl, (const uint8_t **)refin, ref, len);
                for (int j = 0; j < np; j++)
                    if (memcmp(out[j], ref[j], len) != 0)
                        FAIL_AT(c,
                                "parity_update len=%zu par%d: jit != "
                                "ref.c (%s)",
                                len, j, info.desc);
                for (int i = 0; i < nin; i++)
                    free(refin[i]);
                free(refin);
            }
            /* oracle 2: independent scalar delta formula */
            for (int j = 0; j < np; j++) {
                for (size_t k = 0; k < len; k++) {
                    uint8_t a = all[nd + j][k];
                    for (int t = 0; t < nch; t++) {
                        uint8_t delta = old[t][k] ^ newd[t][k];
                        if (delta)
                            a ^= gf_mul(delta,
                                        ti.mat[j * nd + ch[t]]);
                    }
                    if (((uint8_t *)out[j])[k] != a)
                        FAIL_AT(c,
                                "parity_update len=%zu par%d byte%zu: "
                                "jit != scalar delta oracle (%s)",
                                len, j, k, info.desc);
                }
            }
            for (int i = 0; i < nch; i++) {
                free(old[i]);
                free(newd[i]);
            }
            free(old);
            free(newd);
        }

        for (int i = 0; i < N; i++) {
            free(all[i]);
            free(keep[i]);
        }
        free(all);
        free(keep);
        free(in);
        /* out[] ownership: only parity_update holds standalone abufs;
         * encode and recover alias all[] (freed above). ref[] is always
         * standalone. */
        for (int i = 0; i < nout; i++) {
            if (c->op_kind == 2)
                free(out[i]);
            free(ref[i]);
        }
        free(out);
        free(ref);
    }

    (*nrun)++;
    rj_free(j);
    if (plan_ok)
        plan_free(&pl);
    topo_int_free(&ti);
    rj_op_free(&op);
    rj_topo_free(t);
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */

static unsigned long parse_ulong_env(const char *name, int *present)
{
    const char *s = getenv(name);
    *present = s && s[0];
    return *present ? strtoul(s, NULL, 0) : 0;
}

int main(int argc, char **argv)
{
    long iters = 200;
    int have;
    unsigned long env_iters = parse_ulong_env("RJ_FUZZ_ITERS", &have);
    if (have && env_iters > 0)
        iters = (long)env_iters;
    if (argc > 1)
        iters = strtol(argv[1], NULL, 0);

    g_seed = 0xF0755EED12345678ULL; /* fixed default: reproducible */
    unsigned long env_seed = parse_ulong_env("RJ_FUZZ_SEED", &have);
    if (have)
        g_seed = env_seed;
    if (argc > 2)
        g_seed = strtoul(argv[2], NULL, 0);
    rng_state = g_seed ? g_seed : 0x9E3779B97F4A7C15ULL;

    g_sabotage = 0;
    {
        const char *s = getenv("RJ_FUZZ_SABOTAGE");
        if (s && s[0] && strcmp(s, "0") != 0)
            g_sabotage = 1;
    }
    g_slow_ms = 0;
    {
        int have2;
        unsigned long v = parse_ulong_env("RJ_FUZZ_SLOW_MS", &have2);
        if (have2)
            g_slow_ms = (long)v;
    }

    if (iters <= 0) {
        fprintf(stderr, "iters must be > 0\n");
        return 2;
    }

    rj_cpu_info ci;
    rj_cpu_detect(&ci);
    printf("planner fuzz: cpu=%s iters=%ld seed=%lu%s\n", ci.brand, iters,
           g_seed, g_sabotage ? " SABOTAGE=on" : "");

    rj_options opts;
    rj_options_default(&opts);
    opts.benchmark = 0; /* correctness only; never benchmark here */
    opts.verbose = 0;

    unsigned nrun = 0, nskipped = 0, nrejected = 0;
    for (long i = 0; i < iters; i++) {
        g_iter = i;
        fuzz_case_t c;
        memset(&c, 0, sizeof(c));
        gen_topology(&c);
        /* two stripe lengths per case, distinct, tail-heavy */
        c.nlens = 2;
        c.lens[0] = len_pool[rng_below((uint64_t)nlen_pool)];
        do {
            c.lens[1] = len_pool[rng_below((uint64_t)nlen_pool)];
        } while (c.lens[1] == c.lens[0]);
        uint64_t t0 = g_slow_ms > 0 ? now_ms() : 0;
        run_case(&c, &opts, &nrun, &nskipped, &nrejected);
        if (g_slow_ms > 0) {
            uint64_t dt = now_ms() - t0;
            if ((long)dt > g_slow_ms) {
                printf("slow iter=%ld %llums\n", i,
                       (unsigned long long)dt);
                describe_case(&c);
            }
        }
        if ((i + 1) % 50 == 0)
            printf("  ... %ld/%ld\n", i + 1, iters);
    }

    printf("planner fuzz: %u cases ran, %u skipped (ctor refused), "
           "%u clean-rejected, 0 failures\n",
           nrun, nskipped, nrejected);
    return 0;
}
