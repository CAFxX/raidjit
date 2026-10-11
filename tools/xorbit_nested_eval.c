/* A3 evaluation: flat vs nested greedy pair-CSE on the bit-level xorbit IR.
 *
 * STANDALONE PROTOTYPE -- not part of the build, not wired into the
 * Makefile. Evaluates whether nested temp extraction (temp-input and
 * temp-temp pairs, i.e. temps of temps) beats A1's flat input-pair-only
 * greedy in src/xorsched.c, under the static bit-level gate-count
 * objective: G = ntemps + sum_ob max(0, natoms(ob)-1), the XOR gate
 * count of the scheduled DAG.
 *
 * Method (mirrors tools/nested_cse_experiment.c from the byte-level
 * Wave-1 evaluation, commit 03056d6):
 *   flat   = faithful replica of xorbit_build's greedy (input-bit pairs
 *            only). Fidelity is CHECKED against the real xorbit_build on
 *            every corpus matrix (ntemps, per-temp operands, G must
 *            match exactly); any divergence aborts.
 *   nested = flat, then further greedy rounds over the atom universe
 *            (input bits + already-formed temps), extracting any pair
 *            with >= 1 temp co-occurring in >= 2 output-bit rows.
 *            Each extraction provably reduces G by (nusers-1) >= 1, so
 *            G_nested <= G_flat monotonically; the question is the
 *            margin and the pressure cost (max simultaneously-live
 *            temps, which a production nested emitter would have to
 *            keep in registers -- the flat emitter fans each temp out
 *            immediately and keeps none).
 *
 * Correctness: every nested schedule is verified by a nested-aware
 * interpreter against a scalar GF(2^8) reference on 20 deterministic
 * inputs per matrix. Non-tautological: a mutation check corrupts one
 * temp operand and requires the interpreter to disagree with the
 * reference.
 *
 * Build: gcc -O2 -std=c11 -Wall -Wextra -Iinclude -Isrc \
 *            -o /tmp/xorbit_nested_eval tools/xorbit_nested_eval.c \
 *            build_a3/libraidjit.a
 * Run: /tmp/xorbit_nested_eval   (deterministic; no arguments)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>

#include "plan.h"
#include "xorsched.h"

#define XB_MAX_TEMPS 1024

/* ---- scalar GF(2^8) (reference; same as tests/test_xorbit.c) ---- */
static uint8_t gfmul(uint8_t a, uint8_t b)
{
    uint8_t p = 0;
    while (b) {
        if (b & 1)
            p ^= a;
        int hi = a & 0x80;
        a <<= 1;
        if (hi)
            a ^= 0x1b;
        b >>= 1;
    }
    return p;
}

static uint8_t gf_pow(uint8_t base, unsigned e)
{
    uint8_t r = 1;
    while (e) {
        if (e & 1)
            r = gfmul(r, base);
        base = gfmul(base, base);
        e >>= 1;
    }
    return r;
}

/* ---- corpus matrix descriptor ---- */
typedef struct {
    const char *name;
    int n_in, n_out;   /* plan dims (DELTA: n_in = 2*nc + n_out) */
    plan_kind_t kind;
    int nchanged;      /* DELTA only */
    uint8_t *mat;      /* n_out x ncols */
    uint8_t *horner_c; /* HORNER only */
    int real_shape;    /* 1 = real workload shape, 0 = synthetic stress */
} mat_t;

/* scheduler-visible coefficient, mirroring xb_coeff in src/xorsched.c */
static uint8_t mc_coeff(const mat_t *m, int j, int i)
{
    if (m->kind == P_DELTA)
        return m->mat[(size_t)j * (size_t)m->nchanged +
                      (size_t)(i % m->nchanged)];
    if (m->kind == P_HORNER)
        return gf_pow(m->horner_c[j], (unsigned)(m->n_in - 1 - i));
    return m->mat[(size_t)j * (size_t)m->n_in + (size_t)i];
}

static int mc_ni_sched(const mat_t *m)
{
    return (m->kind == P_DELTA) ? 2 * m->nchanged : m->n_in;
}

/* ---- greedy working state (replica of xorbit_build's internals) ---- */
typedef struct {
    int *atoms;
    int n, cap;
} erow_t;

typedef struct {
    long nbits_in, nbits_out;
    erow_t *rows;      /* [nbits_out] current atom lists */
    uint8_t *has;      /* [nbits_out*nbits_in] input-bit membership */
    long *init_natoms; /* [nbits_out] pre-CSE atom counts (for G_naive) */
    int *counts;       /* [nbits_in*nbits_in] flat pair counts */
    int *ta, *tb;      /* [XB_MAX_TEMPS] temp operands (atom ids) */
    int ntemps;
} estate_t;

static void die(const char *msg)
{
    fprintf(stderr, "FATAL: %s\n", msg);
    exit(1);
}

static void *xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p)
        die("out of memory");
    return p;
}

static void *xcalloc(size_t n, size_t s)
{
    void *p = calloc(n ? n : 1, s ? s : 1);
    if (!p)
        die("out of memory");
    return p;
}

static void erow_add(erow_t *r, int atom)
{
    if (r->n == r->cap) {
        int ncap = r->cap ? r->cap * 2 : 8;
        int *na = realloc(r->atoms, (size_t)ncap * sizeof(int));
        if (!na)
            die("out of memory");
        r->atoms = na;
        r->cap = ncap;
    }
    r->atoms[r->n++] = atom;
}

static void erow_remove(erow_t *r, int atom)
{
    for (int i = 0; i < r->n; i++) {
        if (r->atoms[i] == atom) {
            r->atoms[i] = r->atoms[--r->n];
            return;
        }
    }
    die("erow_remove: atom not present");
}

static int erow_has(const erow_t *r, int atom)
{
    for (int i = 0; i < r->n; i++)
        if (r->atoms[i] == atom)
            return 1;
    return 0;
}

/* Step 1 replica: incidence sets + initial pair counts. Loop order and
 * canonical (min,max) count slots match xorbit_build exactly. */
static void estate_init(estate_t *e, const mat_t *m)
{
    int ni_sched = mc_ni_sched(m);
    long nbits_in = 8L * ni_sched;
    long nbits_out = 8L * m->n_out;
    e->nbits_in = nbits_in;
    e->nbits_out = nbits_out;
    e->rows = xcalloc((size_t)nbits_out, sizeof(erow_t));
    e->has = xcalloc((size_t)nbits_out * (size_t)nbits_in, 1);
    e->init_natoms = xcalloc((size_t)nbits_out, sizeof(long));
    e->counts = xcalloc((size_t)nbits_in * (size_t)nbits_in, sizeof(int));
    e->ta = xmalloc(XB_MAX_TEMPS * sizeof(int));
    e->tb = xmalloc(XB_MAX_TEMPS * sizeof(int));
    e->ntemps = 0;

    for (int j = 0; j < m->n_out; j++) {
        for (int b = 0; b < 8; b++) {
            long ob = 8L * (long)j + (long)b;
            for (int i = 0; i < ni_sched; i++) {
                uint8_t c = mc_coeff(m, j, i);
                if (!c)
                    continue;
                /* columns of the bitmatrix of x -> c*x */
                uint8_t col[8];
                for (int k = 0; k < 8; k++)
                    col[k] = gfmul(c, (uint8_t)(1u << k));
                for (int k = 0; k < 8; k++) {
                    if (!((col[k] >> b) & 1u))
                        continue;
                    int q = 8 * i + k;
                    erow_add(&e->rows[ob], q);
                    e->has[ob * nbits_in + q] = 1;
                }
            }
            e->init_natoms[ob] = e->rows[ob].n;
            for (int x = 0; x < e->rows[ob].n; x++)
                for (int y = x + 1; y < e->rows[ob].n; y++) {
                    int qa = e->rows[ob].atoms[x], qb = e->rows[ob].atoms[y];
                    int p = qa < qb ? qa : qb, q = qa < qb ? qb : qa;
                    e->counts[(size_t)p * (size_t)nbits_in + (size_t)q]++;
                }
        }
    }
}

static void estate_free(estate_t *e)
{
    if (!e)
        return;
    for (long ob = 0; ob < e->nbits_out; ob++)
        free(e->rows[ob].atoms);
    free(e->rows);
    free(e->has);
    free(e->init_natoms);
    free(e->counts);
    free(e->ta);
    free(e->tb);
    memset(e, 0, sizeof(*e));
}

/* Step 2 replica: greedy flat pair extraction. Verbatim logic of
 * xorbit_build's step 2 (same scan order, same strictly-greater
 * tie-break, same count maintenance). */
static void estate_flat(estate_t *e)
{
    long nbits_in = e->nbits_in, nbits_out = e->nbits_out;
    for (;;) {
        long ba = -1, bb = -1;
        int best = 0;
        for (long a = 0; a < nbits_in; a++)
            for (long b = a + 1; b < nbits_in; b++) {
                int cc = e->counts[a * nbits_in + b];
                if (cc > best) {
                    best = cc;
                    ba = a;
                    bb = b;
                }
            }
        if (best < 2 || e->ntemps >= XB_MAX_TEMPS)
            break;
        e->ta[e->ntemps] = (int)ba;
        e->tb[e->ntemps] = (int)bb;
        int tatom = (int)nbits_in + e->ntemps;
        e->ntemps++;
        for (long ob = 0; ob < nbits_out; ob++) {
            uint8_t *h = e->has + ob * nbits_in;
            if (!h[ba] || !h[bb])
                continue;
            /* the row loses pairs (ba,bb), (ba,x), (bb,x) */
            e->counts[ba * nbits_in + bb]--;
            for (long x = 0; x < nbits_in; x++) {
                if (!h[x] || x == ba || x == bb)
                    continue;
                long p = x < ba ? x : ba, q = x < ba ? ba : x;
                e->counts[p * nbits_in + q]--;
                p = x < bb ? x : bb, q = x < bb ? bb : x;
                e->counts[p * nbits_in + q]--;
            }
            h[ba] = h[bb] = 0;
            erow_remove(&e->rows[ob], (int)ba);
            erow_remove(&e->rows[ob], (int)bb);
            erow_add(&e->rows[ob], tatom);
        }
    }
}

/* Gate count of the current program state:
 * G = ntemps + sum_ob max(0, natoms(ob)-1). */
static long estate_gates(const estate_t *e)
{
    long g = e->ntemps;
    for (long ob = 0; ob < e->nbits_out; ob++) {
        int n = e->rows[ob].n;
        if (n > 1)
            g += n - 1;
    }
    return g;
}

static long estate_gates_naive(const estate_t *e)
{
    long g = 0;
    for (long ob = 0; ob < e->nbits_out; ob++) {
        long n = e->init_natoms[ob];
        if (n > 1)
            g += n - 1;
    }
    return g;
}

/* max simultaneously-live temps under immediate fanout: temp m is live
 * on [m, lastuse(m)], where lastuse(m) is the newest temp referencing
 * m (output-bit users are fanned out at formation m itself). */
static int estate_maxlive(const estate_t *e)
{
    int nt = e->ntemps;
    long N = e->nbits_in;
    if (nt == 0)
        return 0;
    int *lastuse = xmalloc((size_t)nt * sizeof(int));
    for (int m = 0; m < nt; m++)
        lastuse[m] = m;
    for (int m = 0; m < nt; m++) {
        int a = e->ta[m], b = e->tb[m];
        if (a >= N) {
            int am = (int)(a - N);
            assert(am < m);
            if (m > lastuse[am])
                lastuse[am] = m;
        }
        if (b >= N) {
            int bm = (int)(b - N);
            assert(bm < m);
            if (m > lastuse[bm])
                lastuse[bm] = m;
        }
    }
    int ml = 0;
    for (int p = 0; p < nt; p++) {
        int live = 0;
        for (int m = 0; m < nt; m++)
            if (m <= p && p <= lastuse[m])
                live++;
        if (live > ml)
            ml = live;
    }
    free(lastuse);
    return ml;
}

/* Nested greedy rounds over the full atom universe (input bits +
 * already-formed temps). maxlive_cap < 0 = uncapped; otherwise an
 * extraction that would push maxlive above the cap is skipped for the
 * round (next-best pair is tried). */
static void estate_nested(estate_t *e, int maxlive_cap)
{
    long N = e->nbits_in, nbits_out = e->nbits_out;
    long Amax = N + XB_MAX_TEMPS;
    int *pc = xcalloc((size_t)Amax * (size_t)Amax, sizeof(int));

    for (;;) {
        long A = N + e->ntemps;
        memset(pc, 0, (size_t)A * (size_t)A * sizeof(int));
        /* co-occurrence over pairs with >= 1 temp */
        for (long ob = 0; ob < nbits_out; ob++) {
            const erow_t *r = &e->rows[ob];
            for (int x = 0; x < r->n; x++)
                for (int y = x + 1; y < r->n; y++) {
                    int xa = r->atoms[x], ya = r->atoms[y];
                    if (xa >= N || ya >= N) {
                        int p = xa < ya ? xa : ya;
                        int q = xa < ya ? ya : xa;
                        pc[(size_t)p * (size_t)A + (size_t)q]++;
                    }
                }
        }
        /* best pair with strictly-greater tie-break (flat convention);
         * skip pairs that would violate the maxlive cap */
        int bxa = -1, bxb = -1, best = 0;
        for (long x = 0; x < A; x++) {
            for (long y = x + 1; y < A; y++) {
                if (x < N && y < N)
                    continue;
                int cc = pc[(size_t)x * (size_t)A + (size_t)y];
                if (cc > best) {
                    if (maxlive_cap >= 0) {
                        /* tentative maxlive with this extraction:
                         * append tentatively, measure exactly,
                         * roll back. */
                        int mnew = e->ntemps;
                        e->ta[mnew] = (int)x;
                        e->tb[mnew] = (int)y;
                        e->ntemps = mnew + 1;
                        int ml = estate_maxlive(e);
                        e->ntemps = mnew;
                        if (ml > maxlive_cap)
                            continue;
                    }
                    best = cc;
                    bxa = (int)x;
                    bxb = (int)y;
                }
            }
        }
        if (best < 2 || e->ntemps >= XB_MAX_TEMPS)
            break;
        e->ta[e->ntemps] = bxa;
        e->tb[e->ntemps] = bxb;
        int tatom = (int)N + e->ntemps;
        e->ntemps++;
        for (long ob = 0; ob < nbits_out; ob++) {
            erow_t *r = &e->rows[ob];
            if (!erow_has(r, bxa) || !erow_has(r, bxb))
                continue;
            erow_remove(r, bxa);
            erow_remove(r, bxb);
            erow_add(r, tatom);
        }
    }
    free(pc);
}

/* ---- nested-aware interpreter (independent of any emitter) ---- */
static void estate_run(const estate_t *e, const uint8_t *in_bytes,
                       uint8_t *out_bits /* [nbits_out] */)
{
    long N = e->nbits_in;
    long ni = N / 8;
    uint8_t *inbit = xcalloc((size_t)N, 1);
    uint8_t *tmp = xcalloc((size_t)(e->ntemps ? e->ntemps : 1), 1);
    for (long i = 0; i < ni; i++)
        for (int k = 0; k < 8; k++)
            inbit[i * 8 + k] = (in_bytes[i] >> k) & 1;
    for (int m = 0; m < e->ntemps; m++) {
        int a = e->ta[m], b = e->tb[m];
        uint8_t va = (a < N) ? inbit[a] : tmp[a - N];
        uint8_t vb = (b < N) ? inbit[b] : tmp[b - N];
        tmp[m] = va ^ vb;
    }
    memset(out_bits, 0, (size_t)e->nbits_out);
    for (long ob = 0; ob < e->nbits_out; ob++) {
        const erow_t *r = &e->rows[ob];
        uint8_t acc = 0;
        for (int i = 0; i < r->n; i++) {
            int a = r->atoms[i];
            acc ^= (a < N) ? inbit[a] : tmp[a - N];
        }
        out_bits[ob] = acc;
    }
    free(inbit);
    free(tmp);
}

/* scalar GF(2^8) reference for one input vector */
static void scalar_ref(const mat_t *m, const uint8_t *in,
                       uint8_t *out_bits /* [8*n_out] */)
{
    memset(out_bits, 0, (size_t)8 * (size_t)m->n_out);
    if (m->kind == P_MULACC) {
        for (int j = 0; j < m->n_out; j++) {
            uint8_t acc = 0;
            for (int i = 0; i < m->n_in; i++)
                acc ^= gfmul(m->mat[(size_t)j * (size_t)m->n_in +
                                    (size_t)i],
                             in[i]);
            for (int b = 0; b < 8; b++)
                out_bits[j * 8 + b] = (acc >> b) & 1;
        }
    } else if (m->kind == P_HORNER) {
        for (int j = 0; j < m->n_out; j++) {
            uint8_t acc = 0, pw = 1;
            for (int i = m->n_in - 1; i >= 0; i--) {
                acc ^= gfmul(pw, in[i]);
                pw = gfmul(pw, m->horner_c[j]);
            }
            for (int b = 0; b < 8; b++)
                out_bits[j * 8 + b] = (acc >> b) & 1;
        }
    } else { /* P_DELTA */
        int nc = m->nchanged;
        for (int j = 0; j < m->n_out; j++) {
            uint8_t acc = 0;
            for (int i = 0; i < nc; i++) {
                uint8_t c = m->mat[(size_t)j * (size_t)nc + (size_t)i];
                uint8_t d = in[i] ^ in[nc + i];
                acc ^= gfmul(c, d);
            }
            for (int b = 0; b < 8; b++)
                out_bits[j * 8 + b] = (acc >> b) & 1;
        }
    }
}

/* deterministic PRNG for test inputs (fixed seed) */
static uint64_t rng_state = 0x123456789abcdefULL;
static uint8_t rng_byte(void)
{
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (uint8_t)(rng_state >> 33);
}

/* verify the estate program against the scalar reference */
static void check_correct(const estate_t *e, const mat_t *m,
                          const char *what)
{
    int ni = mc_ni_sched(m);
    uint8_t *in = xmalloc((size_t)ni);
    uint8_t *got = xmalloc((size_t)8 * (size_t)m->n_out);
    uint8_t *want = xmalloc((size_t)8 * (size_t)m->n_out);
    for (int t = 0; t < 20; t++) {
        for (int i = 0; i < ni; i++)
            in[i] = rng_byte();
        estate_run(e, in, got);
        scalar_ref(m, in, want);
        for (int j = 0; j < 8 * m->n_out; j++) {
            if (got[j] != want[j]) {
                fprintf(stderr,
                        "FATAL: %s schedule wrong on %s (bit %d, t=%d)\n",
                        what, m->name, j, t);
                exit(1);
            }
        }
    }
    free(in);
    free(got);
    free(want);
}

/* mutation: corrupt one temp operand; the interpreter must then
 * disagree with the reference (proves the check is non-tautological) */
static void check_mutation(estate_t *e, const mat_t *m)
{
    if (e->ntemps == 0)
        return;
    int ni = mc_ni_sched(m);
    int last = e->ntemps - 1;
    int orig = e->ta[last];
    long A = e->nbits_in + e->ntemps;
    int bad = (int)((orig + 1) % A);
    if (bad == e->tb[last])
        bad = (int)((orig + 2) % A);
    e->ta[last] = bad;
    uint8_t *in = xmalloc((size_t)ni);
    uint8_t *got = xmalloc((size_t)8 * (size_t)m->n_out);
    uint8_t *want = xmalloc((size_t)8 * (size_t)m->n_out);
    int diff = 0;
    for (int t = 0; t < 50 && !diff; t++) {
        for (int i = 0; i < ni; i++)
            in[i] = rng_byte();
        estate_run(e, in, got);
        scalar_ref(m, in, want);
        for (int j = 0; j < 8 * m->n_out; j++) {
            if (got[j] != want[j]) {
                diff = 1;
                break;
            }
        }
    }
    e->ta[last] = orig; /* restore */
    free(in);
    free(got);
    free(want);
    if (!diff) {
        fprintf(stderr,
                "FATAL: mutation check failed on %s (corrupted temp not "
                "detected -- interpreter may be tautological)\n",
                m->name);
        exit(1);
    }
}

/* fidelity: the tool's flat replica must match the real xorbit_build
 * exactly (ntemps, per-temp operands, gate count) */
static void check_fidelity(const estate_t *e, const xorbit_t *real,
                           const mat_t *m)
{
    long g_real, g_flat;
    int nt_real = 0;
    if (!real) {
        /* nothing to share: replica must agree on zero temps */
        if (e->ntemps != 0) {
            fprintf(stderr,
                    "FATAL: fidelity divergence on %s: real=NULL but "
                    "replica found %d temps\n",
                    m->name, e->ntemps);
            exit(1);
        }
        return;
    }
    nt_real = real->ntemps;
    g_real = real->ntemps;
    for (long ob = 0; ob < 8L * m->n_out; ob++) {
        int n = real->bit_natoms[ob];
        if (n > 1)
            g_real += n - 1;
    }
    g_flat = estate_gates(e);
    if (nt_real != e->ntemps || g_real != g_flat) {
        fprintf(stderr,
                "FATAL: fidelity divergence on %s: real(nt=%d,G=%ld) vs "
                "replica(nt=%d,G=%ld)\n",
                m->name, nt_real, g_real, e->ntemps, g_flat);
        exit(1);
    }
    for (int mm = 0; mm < nt_real; mm++) {
        if (real->temps[mm].a != e->ta[mm] ||
            real->temps[mm].b != e->tb[mm]) {
            fprintf(stderr,
                    "FATAL: fidelity divergence on %s: temp %d differs "
                    "real=(%d,%d) replica=(%d,%d)\n",
                    m->name, mm, real->temps[mm].a, real->temps[mm].b,
                    e->ta[mm], e->tb[mm]);
            exit(1);
        }
    }
}

/* ---- corpus construction ---- */
static uint32_t lcg_next(uint32_t *s)
{
    *s = *s * 1103515245u + 12345u;
    return (*s >> 16) & 0x7fff;
}

static mat_t *corpus_make(int *nm)
{
    /* 16 fixed matrices: real workload shapes + synthetic stress */
    mat_t *c = xcalloc(16, sizeof(mat_t));
    int n = 0;

    /* 0: t-mulacc4x2 (verbatim from tests/test_xorbit.c) */
    c[n].name = "t-mulacc4x2";
    c[n].n_in = 4;
    c[n].n_out = 2;
    c[n].kind = P_MULACC;
    c[n].real_shape = 1;
    c[n].mat = xmalloc(8);
    uint8_t m0[8] = { 1, 2, 3, 4, 2, 1, 4, 3 };
    memcpy(c[n].mat, m0, 8);
    n++;

    /* 1: t-mulacc6x3 (verbatim) */
    c[n].name = "t-mulacc6x3";
    c[n].n_in = 6;
    c[n].n_out = 3;
    c[n].kind = P_MULACC;
    c[n].real_shape = 1;
    c[n].mat = xmalloc(18);
    uint8_t m1[18] = { 1, 2, 4, 8, 16, 32,
                       2, 4, 8, 16, 32, 1,
                       3, 5, 7, 9, 11, 13 };
    memcpy(c[n].mat, m1, 18);
    n++;

    /* 2: raid6-8p2: 8 data + P(all-1) + Q(powers of 2) */
    c[n].name = "raid6-8p2";
    c[n].n_in = 8;
    c[n].n_out = 2;
    c[n].kind = P_MULACC;
    c[n].real_shape = 1;
    c[n].mat = xmalloc(16);
    for (int i = 0; i < 8; i++) {
        c[n].mat[i] = 1;
        c[n].mat[8 + i] = (uint8_t)(1u << i);
    }
    n++;

    /* 3: rs63-vdm: RS(6,3) Vandermonde parity, c[j][i] = 2^(i*j) */
    c[n].name = "rs63-vdm";
    c[n].n_in = 6;
    c[n].n_out = 3;
    c[n].kind = P_MULACC;
    c[n].real_shape = 1;
    c[n].mat = xmalloc(18);
    for (int j = 0; j < 3; j++)
        for (int i = 0; i < 6; i++)
            c[n].mat[j * 6 + i] = gf_pow(2, (unsigned)(i * j));
    n++;

    /* 4: rs104-vdm */
    c[n].name = "rs104-vdm";
    c[n].n_in = 10;
    c[n].n_out = 4;
    c[n].kind = P_MULACC;
    c[n].real_shape = 1;
    c[n].mat = xmalloc(40);
    for (int j = 0; j < 4; j++)
        for (int i = 0; i < 10; i++)
            c[n].mat[j * 10 + i] = gf_pow(2, (unsigned)(i * j));
    n++;

    /* 5: rs144-vdm */
    c[n].name = "rs144-vdm";
    c[n].n_in = 14;
    c[n].n_out = 4;
    c[n].kind = P_MULACC;
    c[n].real_shape = 1;
    c[n].mat = xmalloc(56);
    for (int j = 0; j < 4; j++)
        for (int i = 0; i < 14; i++)
            c[n].mat[j * 14 + i] = gf_pow(2, (unsigned)(i * j));
    n++;

    /* 6: dense8x8 structured */
    c[n].name = "dense8x8";
    c[n].n_in = 8;
    c[n].n_out = 8;
    c[n].kind = P_MULACC;
    c[n].real_shape = 0;
    c[n].mat = xmalloc(64);
    for (int j = 0; j < 8; j++)
        for (int i = 0; i < 8; i++)
            c[n].mat[j * 8 + i] = (uint8_t)(((i * 3 + j * 5 + 1) % 255) + 1);
    n++;

    /* 7: rand10x5-s7 */
    c[n].name = "rand10x5-s7";
    c[n].n_in = 10;
    c[n].n_out = 5;
    c[n].kind = P_MULACC;
    c[n].real_shape = 0;
    c[n].mat = xmalloc(50);
    {
        uint32_t s = 7;
        for (int k = 0; k < 50; k++)
            c[n].mat[k] = (uint8_t)((lcg_next(&s) % 255) + 1);
    }
    n++;

    /* 8: rand12x6-s42 */
    c[n].name = "rand12x6-s42";
    c[n].n_in = 12;
    c[n].n_out = 6;
    c[n].kind = P_MULACC;
    c[n].real_shape = 0;
    c[n].mat = xmalloc(72);
    {
        uint32_t s = 42;
        for (int k = 0; k < 72; k++)
            c[n].mat[k] = (uint8_t)((lcg_next(&s) % 255) + 1);
    }
    n++;

    /* 9: chain23: alternating {2,3} -- heavy xtime subexpression sharing */
    c[n].name = "chain23";
    c[n].n_in = 6;
    c[n].n_out = 4;
    c[n].kind = P_MULACC;
    c[n].real_shape = 0;
    c[n].mat = xmalloc(24);
    for (int j = 0; j < 4; j++)
        for (int i = 0; i < 6; i++)
            c[n].mat[j * 6 + i] = (uint8_t)(((i + j) % 2) ? 3 : 2);
    n++;

    /* 10: deepshare: 4x3 block of c=3 (deep xtime sharing) + dense noise */
    c[n].name = "deepshare";
    c[n].n_in = 6;
    c[n].n_out = 4;
    c[n].kind = P_MULACC;
    c[n].real_shape = 0;
    c[n].mat = xmalloc(24);
    for (int j = 0; j < 4; j++)
        for (int i = 0; i < 6; i++) {
            if (j < 4 && i < 3)
                c[n].mat[j * 6 + i] = 3;
            else
                c[n].mat[j * 6 + i] =
                    (uint8_t)(((i * 11 + j * 7) % 251) + 1);
        }
    n++;

    /* 11: sparse20x4: recovery-like sparse (6 ones/row) -- expect no CSE */
    c[n].name = "sparse20x4";
    c[n].n_in = 20;
    c[n].n_out = 4;
    c[n].kind = P_MULACC;
    c[n].real_shape = 1;
    c[n].mat = xcalloc(80, 1);
    for (int j = 0; j < 4; j++)
        for (int t = 0; t < 6; t++)
            c[n].mat[j * 20 + (j * 5 + t * 3) % 20] = 1;
    n++;

    /* 12: t-horner63 (verbatim) */
    c[n].name = "t-horner63";
    c[n].n_in = 6;
    c[n].n_out = 3;
    c[n].kind = P_HORNER;
    c[n].real_shape = 1;
    c[n].horner_c = xmalloc(3);
    c[n].horner_c[0] = 2;
    c[n].horner_c[1] = 3;
    c[n].horner_c[2] = 5;
    n++;

    /* 13: horner10x4 */
    c[n].name = "horner10x4";
    c[n].n_in = 10;
    c[n].n_out = 4;
    c[n].kind = P_HORNER;
    c[n].real_shape = 1;
    c[n].horner_c = xmalloc(4);
    c[n].horner_c[0] = 2;
    c[n].horner_c[1] = 3;
    c[n].horner_c[2] = 5;
    c[n].horner_c[3] = 7;
    n++;

    /* 14: t-delta (verbatim) */
    c[n].name = "t-delta";
    c[n].n_in = 6;
    c[n].n_out = 2;
    c[n].kind = P_DELTA;
    c[n].nchanged = 2;
    c[n].real_shape = 1;
    c[n].mat = xmalloc(4);
    uint8_t m14[4] = { 1, 2, 3, 4 };
    memcpy(c[n].mat, m14, 4);
    n++;

    /* 15: delta4x3 */
    c[n].name = "delta4x3";
    c[n].n_in = 11; /* 2*4 changed + 3 oldpar */
    c[n].n_out = 3;
    c[n].kind = P_DELTA;
    c[n].nchanged = 4;
    c[n].real_shape = 1;
    c[n].mat = xmalloc(12);
    for (int j = 0; j < 3; j++)
        for (int i = 0; i < 4; i++)
            c[n].mat[j * 4 + i] = (uint8_t)(((i * 7 + j * 3 + 1) % 255) + 1);
    n++;

    *nm = n;
    return c;
}

static void corpus_free(mat_t *c, int n)
{
    for (int i = 0; i < n; i++) {
        free(c[i].mat);
        free(c[i].horner_c);
    }
    free(c);
}

/* build a plan_t for the real xorbit_build */
static void make_plan(const mat_t *m, plan_t *pl)
{
    memset(pl, 0, sizeof(*pl));
    pl->n_in = m->n_in;
    pl->n_out = m->n_out;
    pl->kind = m->kind;
    pl->nchanged = m->nchanged;
    int ncols = (m->kind == P_HORNER)   ? 0
                : (m->kind == P_DELTA)  ? m->nchanged
                                        : m->n_in;
    if (ncols) {
        pl->mat = malloc((size_t)m->n_out * (size_t)ncols);
        if (!pl->mat)
            die("out of memory");
        memcpy(pl->mat, m->mat, (size_t)m->n_out * (size_t)ncols);
    }
    if (m->kind == P_HORNER) {
        pl->horner_c = malloc((size_t)m->n_out);
        if (!pl->horner_c)
            die("out of memory");
        memcpy(pl->horner_c, m->horner_c, (size_t)m->n_out);
    }
    snprintf(pl->name, sizeof(pl->name), "eval-%s", m->name);
}

int main(int argc, char **argv)
{
    int cap = -1;
    if (argc > 1)
        cap = atoi(argv[1]);
    int nm = 0;
    mat_t *c = corpus_make(&nm);

    printf("cap=%d\n", cap);
    printf("%-14s %7s %7s %7s %7s %6s %7s %4s\n", "matrix", "G_naive",
           "G_flat", "G_nest", "dG", "dG%", "ml_nest", "real");
    printf("%-14s %7s %7s %7s %7s %6s %7s %4s\n", "--------------",
           "-------", "-------", "-------", "-------", "------", "-------",
           "----");

    long tot_naive = 0, tot_flat = 0, tot_nest = 0;
    long r_naive = 0, r_flat = 0, r_nest = 0; /* real-shape subset */
    int nwins = 0, nrwins = 0, maxml = 0;

    for (int mi = 0; mi < nm; mi++) {
        const mat_t *m = &c[mi];

        /* real scheduler (fidelity target) */
        plan_t pl;
        make_plan(m, &pl);
        xorbit_t *real = xorbit_build(&pl);

        /* tool replica: flat */
        estate_t e;
        estate_init(&e, m);
        long g_naive = estate_gates_naive(&e);
        estate_flat(&e);
        long g_flat = estate_gates(&e);
        int nt_flat = e.ntemps;
        check_fidelity(&e, real, m);
        if (nt_flat > 0)
            check_correct(&e, m, "flat");

        /* nested (maxlive-capped) */
        estate_nested(&e, cap);
        long g_nest = estate_gates(&e);
        int nt_nest = e.ntemps;
        int ml = estate_maxlive(&e);
        if (nt_nest > 0) {
            check_correct(&e, m, "nested");
            check_mutation(&e, m);
        }
        if (ml > maxml)
            maxml = ml;

        long dg = g_flat - g_nest;
        double pct = g_flat ? 100.0 * (double)dg / (double)g_flat : 0.0;
        if (dg > 0) {
            nwins++;
            if (m->real_shape)
                nrwins++;
        }
        tot_naive += g_naive;
        tot_flat += g_flat;
        tot_nest += g_nest;
        if (m->real_shape) {
            r_naive += g_naive;
            r_flat += g_flat;
            r_nest += g_nest;
        }

        if (real)
            printf("%-14s %7ld %7ld %7ld %+7ld %5.1f%% %7d %4s\n",
                   m->name, g_naive, g_flat, g_nest, dg, pct, ml,
                   m->real_shape ? "yes" : "syn");
        else
            printf("%-14s %7ld %7s %7s %7s %6s %7s %4s  (no temps: "
                   "flat+real agree on NULL)\n",
                   m->name, g_naive, "-", "-", "-", "-", "-",
                   m->real_shape ? "yes" : "syn");

        xorbit_free(real);
        free(pl.mat);
        free(pl.horner_c);
        estate_free(&e);
    }

    printf("\nall-corpus : G_naive=%ld G_flat=%ld G_nest=%ld dG=%ld "
           "(%.1f%% of flat)\n",
           tot_naive, tot_flat, tot_nest, tot_flat - tot_nest,
           tot_flat ? 100.0 * (double)(tot_flat - tot_nest) / (double)tot_flat
                    : 0.0);
    printf("real-shape : G_naive=%ld G_flat=%ld G_nest=%ld dG=%ld "
           "(%.1f%% of flat)\n",
           r_naive, r_flat, r_nest, r_flat - r_nest,
           r_flat ? 100.0 * (double)(r_flat - r_nest) / (double)r_flat
                  : 0.0);
    printf("matrices with dG>0: %d/%d (real-shape: %d)\n", nwins, nm,
           nrwins);
    printf("max nested maxlive over corpus: %d\n", maxml);
    printf("fidelity: replica == xorbit_build on all %d matrices\n", nm);
    printf("correctness: nested schedules verified vs scalar GF ref\n");
    printf("mutation: corrupted-temp schedules detected\n");

    corpus_free(c, nm);
    return 0;
}
