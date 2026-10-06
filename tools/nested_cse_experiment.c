/* Item-3 experiment (2026-10-06): flat vs nested greedy pair-CSE.
 *
 * STANDALONE PROTOTYPE -- not part of the build, not wired into the
 * Makefile. Evaluates whether nested temp extraction (temp-input and
 * temp-temp pairs) beats the flat input-pair-only greedy in
 * src/xorsched.c, under honest AVX-convention static uop counting.
 *
 * Build: gcc -O2 -std=c11 -Wall -o /tmp/nested_proto tools/nested_cse_experiment.c
 * Run: /tmp/nested_proto
 *
 * Verdict (see docs/xor_schedule.md section 6): nested wins 11-21% on
 * synthetic dense plans but needs up to 6 simultaneously-live temps
 * (vs 2); on the repo's xorsched differential-test topologies (t1-t6)
 * it saves 0-1 uop/lane. Deliberate bound: flat only.
 */
/* Item 3 experiment: flat vs nested greedy pair-CSE for P_XOR plans.
 *
 * Compares honest AVX-convention static uop counts:
 *   flat   = current xorsched.c algorithm (input-pair temps only)
 *   nested = extended greedy allowing temp-input and temp-temp pairs
 *
 * Nested counting assumes an emitter that keeps temps in dedicated
 * registers (formation: 2 uops if any input operand [ld+xor], 1 uop if
 * both operands are temps [reg-reg xor]; fan-out 1 uop/row; row atoms
 * 1 uop each after the first). Liveness/max-live-temps is computed to
 * check the register-pressure side.
 *
 * Correctness: every schedule is evaluated scalarly against the matrix.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define MAXN_IN 64
#define MAXN_OUT 16
#define MAXTEMPS 256

typedef struct {
    int n_in, n_out;
    uint8_t mat[MAXN_OUT][MAXN_IN]; /* row-major 0/1 */
} plan_t;

typedef struct {
    int a, b;          /* atom ids: < n_in input, >= n_in temp (n_in+k) */
    int nusers;
    int users[MAXN_OUT];
} temp_t;

typedef struct {
    int ntemps;
    temp_t temps[MAXTEMPS];
    /* per unique row: final atom list */
    int nunique;
    int row_natoms[MAXN_OUT];
    int row_atoms[MAXN_OUT][MAXN_IN + MAXTEMPS];
    int row_first[MAXN_OUT]; /* first atom or -1 */
    int in_nusers[MAXN_IN];
} sched_t;

/* ---- greedy extraction ---- */

static int row_has(const int *atoms, int n, int atom)
{
    for (int i = 0; i < n; i++)
        if (atoms[i] == atom)
            return 1;
    return 0;
}

static void row_remove(int *atoms, int *n, int atom)
{
    for (int i = 0; i < *n; i++)
        if (atoms[i] == atom) {
            atoms[i] = atoms[--(*n)];
            return;
        }
}

/* Build schedule. nested=0: input pairs only (current). nested=1: also
 * temp-input and temp-temp pairs (temps pair in formation order). */
static void build_sched(const plan_t *pl, int nested, sched_t *s)
{
    int n_in = pl->n_in, n_out = pl->n_out;
    memset(s, 0, sizeof(*s));
    /* work rows: atom lists */
    int rows[MAXN_OUT][MAXN_IN + MAXTEMPS];
    int rn[MAXN_OUT];
    /* pair counts over the atom universe (inputs + formed temps).
     * counts are maintained only for pairs where both atoms are
     * "available": inputs always; temps once formed. */
    static int counts[MAXN_IN + MAXTEMPS][MAXN_IN + MAXTEMPS];
    memset(counts, 0, sizeof(counts));

    for (int j = 0; j < n_out; j++) {
        rn[j] = 0;
        for (int i = 0; i < n_in; i++)
            if (pl->mat[j][i]) {
                rows[j][rn[j]++] = i;
                /* has-bit for pair init */
            }
        for (int x = 0; x < rn[j]; x++)
            for (int y = x + 1; y < rn[j]; y++) {
                int a = rows[j][x], b = rows[j][y];
                if (a > b) { int t = a; a = b; b = t; }
                counts[a][b]++;
            }
    }
    s->nunique = n_out;

    int ntemps = 0;
    for (;;) {
        int ba = -1, bb = -1, best = 0;
        int natoms = n_in + ntemps;
        for (int a = 0; a < natoms; a++)
            for (int b = a + 1; b < natoms; b++) {
                /* flat mode: inputs only */
                if (!nested && (a >= n_in || b >= n_in))
                    continue;
                int c = counts[a][b];
                if (c > best) {
                    best = c;
                    ba = a;
                    bb = b;
                }
            }
        if (best < 2 || ntemps >= MAXTEMPS)
            break;
        int tatom = n_in + ntemps;
        s->temps[ntemps].a = ba;
        s->temps[ntemps].b = bb;
        ntemps++;
        /* replace (ba,bb) by tatom in every row containing both */
        for (int j = 0; j < n_out; j++) {
            if (!row_has(rows[j], rn[j], ba) ||
                !row_has(rows[j], rn[j], bb))
                continue;
            /* row loses pairs (ba,bb),(ba,x),(bb,x); gains (tatom,x) */
            counts[ba < bb ? ba : bb][ba < bb ? bb : ba]--;
            for (int xi = 0; xi < rn[j]; xi++) {
                int x = rows[j][xi];
                if (x == ba || x == bb)
                    continue;
                int p = x < ba ? x : ba, q = x < ba ? ba : x;
                counts[p][q]--;
                p = x < bb ? x : bb, q = x < bb ? bb : x;
                counts[p][q]--;
            }
            row_remove(rows[j], &rn[j], ba);
            row_remove(rows[j], &rn[j], bb);
            /* add tatom; it pairs with every remaining atom */
            for (int xi = 0; xi < rn[j]; xi++) {
                int x = rows[j][xi];
                int p = x < tatom ? x : tatom, q = x < tatom ? tatom : x;
                counts[p][q]++;
            }
            rows[j][rn[j]++] = tatom;
        }
    }
    s->ntemps = ntemps;
    /* finalize rows: temps in formation order, then inputs */
    for (int j = 0; j < n_out; j++) {
        int w = 0;
        for (int k = 0; k < ntemps; k++)
            if (row_has(rows[j], rn[j], n_in + k))
                s->row_atoms[j][w++] = k; /* temp k */
        for (int i = 0; i < n_in; i++)
            if (row_has(rows[j], rn[j], i))
                s->row_atoms[j][w++] = ntemps + i;
        s->row_natoms[j] = w;
        s->row_first[j] = w ? s->row_atoms[j][0] : -1;
    }
    /* temp users */
    for (int k = 0; k < ntemps; k++) {
        int nu = 0;
        for (int j = 0; j < n_out; j++)
            if (row_has(rows[j], rn[j], n_in + k))
                s->temps[k].users[nu++] = j;
        s->temps[k].nusers = nu;
    }
    for (int i = 0; i < n_in; i++) {
        int nu = 0;
        for (int j = 0; j < n_out; j++)
            if (row_has(rows[j], rn[j], i))
                nu++;
        s->in_nusers[i] = nu;
    }
}

/* ---- honest AVX uop counter (per u-lane; stores excluded, both sides) ---- */

/* naive body, mirroring xs_body_naive x86 non-SSE branch */
static long count_naive(const plan_t *pl)
{
    int n_in = pl->n_in, n_out = pl->n_out;
    long uops = 0;
    int first_in[MAXN_OUT];
    for (int j = 0; j < n_out; j++) {
        first_in[j] = -1;
        for (int i = 0; i < n_in; i++)
            if (pl->mat[j][i]) {
                first_in[j] = i;
                break;
            }
    }
    for (int j = 0; j < n_out; j++)
        if (first_in[j] < 0)
            uops += 1; /* dead-row zeroing */
    for (int i = 0; i < n_in; i++) {
        int nusers = 0, uj = -1;
        for (int j = 0; j < n_out; j++)
            if (pl->mat[j][i]) {
                nusers++;
                uj = j;
            }
        if (!nusers)
            continue;
        if (nusers == 1)
            uops += (i == first_in[uj]) ? 1 : 1; /* ld-acc or mem-xor */
        else
            uops += 1 + nusers;
    }
    return uops;
}

/* scheduled body. flat: temps formed ld+mem-xor (2 uops). nested: temp
 * operands already in regs: 2 uops if any input operand, else 1. */
static long count_sched(const plan_t *pl, const sched_t *s, int *maxlive_out)
{
    int n_in = pl->n_in;
    long uops = 0;
    /* liveness: temp k live from formation until last use (by a later
     * temp or by a row). Compute last-use temp index per temp. */
    int lastuse[MAXTEMPS];
    for (int k = 0; k < s->ntemps; k++)
        lastuse[k] = -1;
    for (int k = 0; k < s->ntemps; k++) {
        int a = s->temps[k].a, b = s->temps[k].b;
        if (a >= n_in)
            lastuse[a - n_in] = k;
        if (b >= n_in)
            lastuse[b - n_in] = k;
    }
    /* Immediate fan-out (like the real emitter): a temp with only row
     * users dies right after its own formation+fanout (lastuse = k).
     * A temp used by a later temp lives until that temp forms. */
    for (int k = 0; k < s->ntemps; k++)
        if (s->temps[k].nusers > 0 && lastuse[k] < k)
            lastuse[k] = k; /* dies after immediate fan-out */
    int maxlive = 0;
    for (int k = 0; k < s->ntemps; k++) {
        int live = 0;
        for (int t = 0; t <= k; t++)
            if (lastuse[t] >= k)
                live++;
        if (live > maxlive)
            maxlive = live;
    }
    if (maxlive_out)
        *maxlive_out = maxlive;

    for (int j = 0; j < pl->n_out; j++)
        if (s->row_first[j] < 0)
            uops += 1; /* dead row */
    for (int k = 0; k < s->ntemps; k++) {
        int a = s->temps[k].a, b = s->temps[k].b;
        int ain = a < n_in, bin = b < n_in;
        /* formation: ld (1) if any input operand + xor (1) */
        uops += (ain || bin) ? 2 : 1;
        uops += s->temps[k].nusers; /* fan-out */
    }
    for (int i = 0; i < n_in; i++) {
        int nusers = s->in_nusers[i];
        if (!nusers)
            continue;
        uops += 1 + nusers;
    }
    return uops;
}

/* ---- scalar evaluator: check schedule == matrix ---- */
static void eval_sched(const plan_t *pl, const sched_t *s,
                       const uint8_t *in, uint8_t *out)
{
    int n_in = pl->n_in, n_out = pl->n_out;
    uint8_t tvals[MAXTEMPS];
    /* temp values in formation order */
    for (int k = 0; k < s->ntemps; k++) {
        int a = s->temps[k].a, b = s->temps[k].b;
        uint8_t va = (a < n_in) ? in[a] : tvals[a - n_in];
        uint8_t vb = (b < n_in) ? in[b] : tvals[b - n_in];
        tvals[k] = va ^ vb;
    }
    for (int j = 0; j < n_out; j++) {
        uint8_t acc = 0;
        int first = 1;
        for (int w = 0; w < s->row_natoms[j]; w++) {
            int atom = s->row_atoms[j][w];
            uint8_t v = (atom < s->ntemps) ? tvals[atom]
                                           : in[atom - s->ntemps];
            if (first) {
                acc = v;
                first = 0;
            } else {
                acc ^= v;
            }
        }
        out[j] = acc;
    }
}

static int check_correct(const plan_t *pl, const sched_t *s)
{
    uint8_t in[MAXN_IN], out[MAXN_OUT], ref[MAXN_OUT];
    /* deterministic pseudo-random */
    uint64_t st = 0x12345;
    for (int trial = 0; trial < 20; trial++) {
        for (int i = 0; i < pl->n_in; i++) {
            st = st * 6364136223846793005ULL + 1442695040888963407ULL;
            in[i] = (uint8_t)(st >> 33);
        }
        eval_sched(pl, s, in, out);
        for (int j = 0; j < pl->n_out; j++) {
            uint8_t r = 0;
            for (int i = 0; i < pl->n_in; i++)
                if (pl->mat[j][i])
                    r ^= in[i];
            ref[j] = r;
            if (out[j] != ref[j]) {
                printf("  CORRECTNESS FAIL row %d\n", j);
                return 0;
            }
        }
    }
    return 1;
}

static void run_case(const char *name, const plan_t *pl)
{
    sched_t sf, sn;
    build_sched(pl, 0, &sf);
    build_sched(pl, 1, &sn);
    long naive = count_naive(pl);
    int mlf = 0, mln = 0;
    long flat = count_sched(pl, &sf, &mlf);
    long nest = count_sched(pl, &sn, &mln);
    int okf = check_correct(pl, &sf);
    int okn = check_correct(pl, &sn);
    /* bounded nesting: fall back to flat when maxlive > 2 (the
     * existing t/t2 temp budget) */
    long bnest = (mln <= 2) ? nest : flat;
    printf("%-28s naive=%3ld flat=%3ld (%d temps, ml %d) nest=%3ld "
           "(%d temps, ml %d) d=%+ld bounded=%3ld(d=%+ld) %s %s\n",
           name, naive, flat, sf.ntemps, mlf, nest, sn.ntemps, mln,
           nest - flat, bnest, bnest - flat,
           okf ? "ok" : "BROKEN", okn ? "ok" : "BROKEN");
}

int main(void)
{
    plan_t pl;
    /* T1: deep sharing: 4 rows share a^b^c, differ in 4th input */
    memset(&pl, 0, sizeof(pl));
    pl.n_in = 7;
    pl.n_out = 4;
    /* rows: {0,1,2,3}, {0,1,2,4}, {0,1,2,5}, {0,1,2,6} */
    for (int j = 0; j < 4; j++) {
        pl.mat[j][0] = pl.mat[j][1] = pl.mat[j][2] = 1;
        pl.mat[j][3 + j] = 1;
    }
    run_case("deep-share-4x7", &pl);

    /* T2: chain structure: rows {0,1,2,3},{0,1,2,4},{0,1,4,5},{0,1,5,6} */
    memset(&pl, 0, sizeof(pl));
    pl.n_in = 7;
    pl.n_out = 4;
    uint8_t t2[4][7] = {{1,1,1,1,0,0,0},
                        {1,1,1,0,1,0,0},
                        {1,1,0,0,1,1,0},
                        {1,1,0,0,0,1,1}};
    for (int j = 0; j < 4; j++) for (int i = 0; i < 7; i++) pl.mat[j][i] = t2[j][i];
    run_case("chain-4x7", &pl);

    /* T3: the repo's t1 smoke topology (4x20) */
    memset(&pl, 0, sizeof(pl));
    pl.n_in = 20;
    pl.n_out = 4;
    static const uint8_t t1[4][20] = {
        {1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {1,1,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0},
        {1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0},
        {0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0},
    };
    for (int j = 0; j < 4; j++) for (int i = 0; i < 20; i++) pl.mat[j][i] = t1[j][i];
    run_case("repo-t1-4x20", &pl);

    /* T4: dense 6x12 (repo t6) */
    memset(&pl, 0, sizeof(pl));
    pl.n_in = 12;
    pl.n_out = 6;
    static const uint8_t t6[6][12] = {
        {1,1,1,1,0,0,0,0,0,0,0,0},
        {1,1,0,0,1,1,0,0,0,0,0,0},
        {0,0,1,1,1,1,0,0,0,0,0,0},
        {0,0,0,0,0,0,1,1,1,1,0,0},
        {0,0,0,0,0,0,1,1,0,0,1,1},
        {1,0,1,0,1,0,1,0,1,0,1,0},
    };
    for (int j = 0; j < 6; j++) for (int i = 0; i < 12; i++) pl.mat[j][i] = t6[j][i];
    run_case("repo-t6-6x12", &pl);

    /* T5: RS(6,3)-like Cauchy-ish: rows share large subsets.
     * Simulate: 3 parity rows over 6 data, each row dense. */
    memset(&pl, 0, sizeof(pl));
    pl.n_in = 9;
    pl.n_out = 3;
    /* rows: {0..5},{0,1,2,6,7,8},{3,4,5,6,7,8} -- pairwise share 3 */
    for (int i = 0; i < 6; i++)
        pl.mat[0][i] = 1;
    pl.mat[1][0] = pl.mat[1][1] = pl.mat[1][2] = 1;
    pl.mat[1][6] = pl.mat[1][7] = pl.mat[1][8] = 1;
    pl.mat[2][3] = pl.mat[2][4] = pl.mat[2][5] = 1;
    pl.mat[2][6] = pl.mat[2][7] = pl.mat[2][8] = 1;
    run_case("rs-like-3x9", &pl);

    /* T6: 8 rows, deep common core of 4 inputs + 2 unique each */
    memset(&pl, 0, sizeof(pl));
    pl.n_in = 20;
    pl.n_out = 8;
    for (int j = 0; j < 8; j++) {
        pl.mat[j][0] = pl.mat[j][1] = pl.mat[j][2] = pl.mat[j][3] = 1;
        pl.mat[j][4 + 2 * j] = 1;
        pl.mat[j][5 + 2 * j] = 1;
    }
    run_case("deep-core-8x20", &pl);

    /* T7: dense random 0/1 (recovery-like), 6x14, density 0.5 */
    {
        memset(&pl, 0, sizeof(pl));
        pl.n_in = 14; pl.n_out = 6;
        uint64_t st = 0xabcdef;
        for (int j = 0; j < 6; j++)
            for (int i = 0; i < 14; i++) {
                st = st * 6364136223846793005ULL + 1442695040888963407ULL;
                pl.mat[j][i] = (st >> 33) & 1;
            }
        /* ensure no empty rows */
        for (int j = 0; j < 6; j++) {
            int any = 0;
            for (int i = 0; i < 14; i++) any |= pl.mat[j][i];
            if (!any) pl.mat[j][0] = 1;
        }
        run_case("dense-rand-6x14", &pl);
    }
    /* T8: RDP-like (p=5): 4 data+ p parity rows with diagonal structure */
    {
        memset(&pl, 0, sizeof(pl));
        pl.n_in = 20; pl.n_out = 6;
        /* row parity rows: {0..4},{5..9},{10..14},{15..19} */
        for (int j = 0; j < 4; j++)
            for (int i = 0; i < 5; i++) pl.mat[j][5*j+i] = 1;
        /* diagonal parities */
        for (int d = 0; d < 2; d++)
            for (int k = 0; k < 5; k++) {
                int i = (k + d) % 5 + 5 * (k % 4);
                if (i < 20) pl.mat[4+d][i] = 1;
            }
        run_case("rdp-like-6x20", &pl);
    }
    /* T9: dense random 0/1, 4x20, density 0.7 (heavy sharing) */
    {
        memset(&pl, 0, sizeof(pl));
        pl.n_in = 20; pl.n_out = 4;
        uint64_t st = 0x1234567;
        for (int j = 0; j < 4; j++)
            for (int i = 0; i < 20; i++) {
                st = st * 6364136223846793005ULL + 1442695040888963407ULL;
                pl.mat[j][i] = ((st >> 33) % 10) < 7;
            }
        run_case("dense70-4x20", &pl);
    }

    /* repo t2/t3/t5 */
    {
        static const uint8_t t2m[3][6] = {{1,1,1,1,0,0},{1,1,0,0,1,1},{1,1,1,1,0,0}};
        memset(&pl, 0, sizeof(pl)); pl.n_in = 6; pl.n_out = 3;
        for (int j = 0; j < 3; j++) for (int i = 0; i < 6; i++) pl.mat[j][i] = t2m[j][i];
        run_case("repo-t2-3x6", &pl);
        static const uint8_t t3m[3][6] = {{1,1,0,0,1,0},{0,0,0,0,0,0},{1,1,1,1,0,0}};
        memset(&pl, 0, sizeof(pl)); pl.n_in = 6; pl.n_out = 3;
        for (int j = 0; j < 3; j++) for (int i = 0; i < 6; i++) pl.mat[j][i] = t3m[j][i];
        run_case("repo-t3-3x6", &pl);
        static const uint8_t t5m[4][8] = {{1,1,0,0,0,0,0,0},{0,0,1,1,0,0,0,0},{0,0,0,0,1,1,0,0},{0,0,0,0,0,0,1,1}};
        memset(&pl, 0, sizeof(pl)); pl.n_in = 8; pl.n_out = 4;
        for (int j = 0; j < 4; j++) for (int i = 0; i < 8; i++) pl.mat[j][i] = t5m[j][i];
        run_case("repo-t5-4x8", &pl);
    }
    return 0;
}
