/* XOR-schedule / cross-row CSE (track A3) + bit-level scheduling for
 * GF-multiply rows (track A1). See xorsched.h for the designs. */
#include "xorsched.h"
#include "gf.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

/* JIT-time guards: the greedy pair table is n_in*n_in ints; beyond these
 * sizes no schedule is built and the naive emitter is used. */
#define XS_MAX_TEMPS 1024
#define XS_MAX_DIMS 16384 /* n_in * n_out */
#define XS_MAX_NIN 256

/* ---- internal greedy state ---- */

typedef struct {
    int *atoms; /* input ids 0..n_in-1, temps n_in+k */
    int n, cap;
} xs_row_t;

/* Test hook: fail the Nth allocation inside xorsched_build (counting from
 * 0), to exercise the OOM unwind path. -1 (default) disables. Armed via
 * xorsched_test_oom_at(); inert otherwise. Not thread-safe: tests only. */
static int xs_oom_fail_at = -1;

void xorsched_test_oom_at(int n)
{
    xs_oom_fail_at = n;
}

/* Allocation wrapper for xorsched_build: when the test hook is armed,
 * counts down and returns NULL at the armed site instead of allocating. */
static void *xs_oom_realloc(void *p, size_t n)
{
    if (xs_oom_fail_at >= 0) {
        if (xs_oom_fail_at-- == 0)
            return NULL;
    }
    return realloc(p, n);
}

static void *xs_oom_calloc(size_t nmemb, size_t size)
{
    if (xs_oom_fail_at >= 0) {
        if (xs_oom_fail_at-- == 0)
            return NULL;
    }
    return calloc(nmemb, size);
}

/* All xorsched_build allocations go through these so the test hook can
 * inject a failure at any site. */
#define XS_MALLOC(n) xs_oom_realloc(NULL, (n))
#define XS_CALLOC(nmemb, size) xs_oom_calloc((nmemb), (size))
#define XS_REALLOC(p, n) xs_oom_realloc((p), (n))

/* Returns 0 on success, -1 on allocation failure (fail-stop: the caller
 * unwinds and xorsched_build returns NULL). */
static int xs_row_add(xs_row_t *r, int atom)
{
    if (r->n == r->cap) {
        int ncap = r->cap ? r->cap * 2 : 8;
        int *na = XS_REALLOC(r->atoms, (size_t)ncap * sizeof(int));
        if (!na)
            return -1;
        r->atoms = na;
        r->cap = ncap;
    }
    r->atoms[r->n++] = atom;
    return 0;
}

/* remove first occurrence of atom; rows hold each atom at most once */
static void xs_row_remove(xs_row_t *r, int atom)
{
    for (int i = 0; i < r->n; i++) {
        if (r->atoms[i] == atom) {
            r->atoms[i] = r->atoms[--r->n];
            return;
        }
    }
    assert(!"xs_row_remove: atom not present");
}

static int xs_row_has_atom(const xs_row_t *r, int atom)
{
    for (int i = 0; i < r->n; i++)
        if (r->atoms[i] == atom)
            return 1;
    return 0;
}

static int xs_rows_equal_mat(const plan_t *pl, int n_in, int j0, int j1)
{
    return memcmp(pl->mat + (size_t)j0 * (size_t)n_in,
                  pl->mat + (size_t)j1 * (size_t)n_in, (size_t)n_in) == 0;
}


static void xs_free_greedy(xs_row_t *rows, int nunique, uint8_t *has,
                           int *counts, int *ta, int *tb)
{
    if (rows) {
        for (int u = 0; u < nunique; u++)
            free(rows[u].atoms);
        free(rows);
    }
    free(has);
    free(counts);
    free(ta);
    free(tb);
}

xorsched_t *xorsched_build(const plan_t *pl)
{
    if (!pl || pl->kind != P_XOR || !pl->mat)
        return NULL;
    int n_in = pl->n_in, n_out = pl->n_out;
    if (n_out < 2 || n_in < 2)
        return NULL; /* need >= 2 rows for cross-row sharing */
    if ((long)n_in * (long)n_out > XS_MAX_DIMS || n_in > XS_MAX_NIN)
        return NULL;

    /* 1. deduplicate identical rows */
    int *urow_of = XS_MALLOC((size_t)n_out * sizeof(int));
    int *first_row = XS_MALLOC((size_t)n_out * sizeof(int));
    int *mult = XS_MALLOC((size_t)n_out * sizeof(int));
    if (!urow_of || !first_row || !mult) {
        free(urow_of);
        free(first_row);
        free(mult);
        return NULL;
    }
    int nunique = 0;
    for (int j = 0; j < n_out; j++) {
        int found = -1;
        for (int u = 0; u < nunique; u++) {
            if (xs_rows_equal_mat(pl, n_in, j, first_row[u])) {
                found = u;
                break;
            }
        }
        if (found < 0) {
            first_row[nunique] = j;
            mult[nunique] = 1;
            urow_of[j] = nunique;
            nunique++;
        } else {
            mult[found]++;
            urow_of[j] = found;
        }
    }

    /* 2. init rows: atoms + input bitsets + pair counts */
    xs_row_t *rows = XS_CALLOC((size_t)nunique, sizeof(xs_row_t));
    uint8_t *has = XS_CALLOC((size_t)nunique * (size_t)n_in, 1);
    int *counts = XS_CALLOC((size_t)n_in * (size_t)n_in, sizeof(int));
    int *ta = XS_MALLOC(XS_MAX_TEMPS * sizeof(int));
    int *tb = XS_MALLOC(XS_MAX_TEMPS * sizeof(int));
    if (!rows || !has || !counts || !ta || !tb) {
        xs_free_greedy(rows, nunique, has, counts, ta, tb);
        free(urow_of);
        free(first_row);
        free(mult);
        return NULL;
    }
    int oom = 0; /* set on any later allocation failure: unwind, return NULL */
    int ntemps = 0;
    xorsched_t *s = NULL;
    for (int u = 0; u < nunique && !oom; u++) {
        int j = first_row[u];
        for (int i = 0; i < n_in; i++) {
            if (!pl->mat[(size_t)j * (size_t)n_in + (size_t)i])
                continue;
            if (xs_row_add(&rows[u], i) != 0) {
                oom = 1;
                break;
            }
            has[(size_t)u * (size_t)n_in + (size_t)i] = 1;
        }
        for (int x = 0; x < rows[u].n; x++)
            for (int y = x + 1; y < rows[u].n; y++) {
                int a = rows[u].atoms[x], b = rows[u].atoms[y];
                counts[a * n_in + b]++;
            }
    }
    if (oom)
        goto done;

    /* 3. greedy extraction (flat input pairs only) */
    for (;;) {
        int ba = -1, bb = -1, best = 0;
        for (int a = 0; a < n_in; a++)
            for (int b = a + 1; b < n_in; b++) {
                int c = counts[a * n_in + b];
                if (c > best) {
                    best = c;
                    ba = a;
                    bb = b;
                }
            }
        if (best < 2 || ntemps >= XS_MAX_TEMPS)
            break;
        ta[ntemps] = ba;
        tb[ntemps] = bb;
        int tatom = n_in + ntemps;
        ntemps++;
        for (int u = 0; u < nunique; u++) {
            uint8_t *h = has + (size_t)u * (size_t)n_in;
            if (!h[ba] || !h[bb])
                continue;
            /* the row loses pairs (ba,bb), (ba,x), (bb,x) */
            counts[ba * n_in + bb]--;
            for (int x = 0; x < n_in; x++) {
                if (!h[x] || x == ba || x == bb)
                    continue;
                int p = x < ba ? x : ba, q = x < ba ? ba : x;
                counts[p * n_in + q]--;
                p = x < bb ? x : bb, q = x < bb ? bb : x;
                counts[p * n_in + q]--;
            }
            h[ba] = h[bb] = 0;
            xs_row_remove(&rows[u], ba);
            xs_row_remove(&rows[u], bb);
            if (xs_row_add(&rows[u], tatom) != 0) {
                oom = 1;
                break;
            }
        }
        if (oom)
            break;
    }
    if (oom)
        goto done;

    if (ntemps > 0 && !oom) {
        s = XS_CALLOC(1, sizeof(*s));
        if (!s) {
            oom = 1;
            goto done;
        }
        s->n_in = n_in;
        s->n_out = n_out;
        s->ntemps = ntemps;
        s->nunique = nunique;
        s->urow_of = urow_of;
        s->first_row = first_row;
        s->mult = mult;
        urow_of = first_row = mult = NULL; /* owned by s now */
        s->temps = XS_CALLOC((size_t)ntemps, sizeof(xorsched_temp_t));
        s->row_natoms = XS_CALLOC((size_t)nunique, sizeof(int));
        s->row_atoms = XS_CALLOC((size_t)nunique, sizeof(int *));
        s->row_first = XS_CALLOC((size_t)nunique, sizeof(int));
        s->in_nusers = XS_CALLOC((size_t)n_in, sizeof(int));
        s->in_users = XS_CALLOC((size_t)n_in, sizeof(int *));
        if (!s->temps || !s->row_natoms || !s->row_atoms || !s->row_first ||
            !s->in_nusers || !s->in_users) {
            oom = 1;
            goto done;
        }
        /* temps */
        for (int k = 0; k < ntemps; k++) {
            s->temps[k].a = ta[k];
            s->temps[k].b = tb[k];
            /* users: rows whose atoms contain temp n_in+k */
            int nu = 0;
            for (int u = 0; u < nunique; u++)
                if (xs_row_has_atom(&rows[u], n_in + k))
                    nu++;
            s->temps[k].nusers = nu;
            s->temps[k].users = XS_MALLOC((size_t)nu * sizeof(int));
            if (!s->temps[k].users && nu > 0) {
                oom = 1;
                goto done;
            }
            nu = 0;
            for (int u = 0; u < nunique; u++)
                if (xs_row_has_atom(&rows[u], n_in + k))
                    s->temps[k].users[nu++] = u;
        }
        /* rows: temps in formation order, then inputs ascending */
        for (int u = 0; u < nunique; u++) {
            int nt = 0, ni = 0;
            for (int i = 0; i < rows[u].n; i++) {
                if (rows[u].atoms[i] >= n_in)
                    nt++;
                else
                    ni++;
            }
            s->row_natoms[u] = nt + ni;
            s->row_atoms[u] = XS_MALLOC((size_t)(nt + ni) * sizeof(int));
            if (!s->row_atoms[u] && nt + ni > 0) {
                oom = 1;
                goto done;
            }
            int w = 0;
            for (int k = 0; k < ntemps; k++)
                if (xs_row_has_atom(&rows[u], n_in + k))
                    s->row_atoms[u][w++] = k; /* temp k -> atom k */
            for (int i = 0; i < n_in; i++)
                if (has[(size_t)u * (size_t)n_in + (size_t)i])
                    s->row_atoms[u][w++] = ntemps + i;
            assert(w == nt + ni);
            s->row_first[u] = w > 0 ? s->row_atoms[u][0] : -1;
        }
        /* per-input direct users */
        for (int i = 0; i < n_in; i++) {
            int nu = 0;
            for (int u = 0; u < nunique; u++)
                if (has[(size_t)u * (size_t)n_in + (size_t)i])
                    nu++;
            s->in_nusers[i] = nu;
            s->in_users[i] = XS_MALLOC((size_t)nu * sizeof(int));
            if (!s->in_users[i] && nu > 0) {
                oom = 1;
                goto done;
            }
            nu = 0;
            for (int u = 0; u < nunique; u++)
                if (has[(size_t)u * (size_t)n_in + (size_t)i])
                    s->in_users[i][nu++] = u;
        }
        /* dead plan rows and live duplicates */
        s->ndead_plan = 0;
        s->ndup_live = 0;
        for (int j = 0; j < n_out; j++) {
            int u = s->urow_of[j];
            if (s->row_first[u] < 0)
                s->ndead_plan++;
            else if (j != s->first_row[u])
                s->ndup_live++;
        }
    }

done:
    xs_free_greedy(rows, nunique, has, counts, ta, tb);
    free(urow_of);
    free(first_row);
    free(mult);
    if (oom) {
        /* OOM mid-construction: s is partial or NULL. Unwind with
         * NULL-tolerant frees; never return a corrupt schedule. */
        if (s) {
            if (s->temps) {
                for (int k = 0; k < ntemps; k++)
                    free(s->temps[k].users);
                free(s->temps);
            }
            if (s->row_atoms) {
                for (int u = 0; u < nunique; u++)
                    free(s->row_atoms[u]);
                free(s->row_atoms);
            }
            free(s->row_natoms);
            free(s->row_first);
            if (s->in_users) {
                for (int i = 0; i < n_in; i++)
                    free(s->in_users[i]);
                free(s->in_users);
            }
            free(s->in_nusers);
            free(s->urow_of);
            free(s->first_row);
            free(s->mult);
            free(s);
        }
        return NULL;
    }
    if (ntemps == 0 || !s) {
        xorsched_free(s);
        return NULL;
    }
    return s;
}

void xorsched_free(xorsched_t *s)
{
    if (!s)
        return;
    for (int k = 0; k < s->ntemps; k++)
        free(s->temps[k].users);
    free(s->temps);
    for (int u = 0; u < s->nunique; u++)
        free(s->row_atoms[u]);
    free(s->row_natoms);
    free(s->row_atoms);
    free(s->row_first);
    for (int i = 0; i < s->n_in; i++)
        free(s->in_users[i]);
    free(s->in_nusers);
    free(s->in_users);
    free(s->urow_of);
    free(s->first_row);
    free(s->mult);
    free(s);
}

/* ---- static cost model ---- */

/* first contributing input per plan row (-1 if none), from the matrix */
static void xs_first_in(const plan_t *pl, int *first_in)
{
    int n_in = pl->n_in, n_out = pl->n_out;
    for (int j = 0; j < n_out; j++) {
        first_in[j] = -1;
        for (int i = 0; i < n_in; i++) {
            if (pl->mat[(size_t)j * (size_t)n_in + (size_t)i]) {
                first_in[j] = i;
                break;
            }
        }
    }
}

/* naive body uops for nu lanes. Mirrors each backend's P_XOR column loop:
 * x86: body_inptr mov when i >= nhoist; folded mem-XOR is 1 uop on every
 * x86 ISA (SSE uses the pxor-m128 form). AArch64/RVV: pointer setup
 * uncounted; always load-into-temp + fan-out (RVV additionally folds
 * single-user-first into a direct acc load). */
static long xs_body_naive(const plan_t *pl, isa_t isa, int nu,
                          int nhoist, const int *first_in)
{
    int n_in = pl->n_in, n_out = pl->n_out;
    int is_x86 = (isa == ISA_SSE || isa == ISA_AVX2 || isa == ISA_AVX512);
    long uops = 0;
    for (int j = 0; j < n_out; j++)
        if (first_in[j] < 0)
            uops += nu; /* dead-row zeroing */
    for (int i = 0; i < n_in; i++) {
        int nusers = 0;
        for (int j = 0; j < n_out; j++)
            if (pl->mat[(size_t)j * (size_t)n_in + (size_t)i])
                nusers++;
        if (!nusers)
            continue;
        if (is_x86 && i >= nhoist)
            uops += 1; /* body_inptr mov */
        for (int u = 0; u < nu; u++) {
            if (is_x86 && nusers == 1) {
                /* need the single user's row to test first-ness */
                int uj = -1;
                for (int j = 0; j < n_out; j++)
                    if (pl->mat[(size_t)j * (size_t)n_in + (size_t)i]) {
                        uj = j;
                        break;
                    }
                if (i == first_in[uj])
                    uops += 1; /* direct load into acc */
                else
                    uops += 1; /* c_vpxor_mem: folded pxor-mem (1 uop)
                                * on every x86 ISA, incl. SSE */
            } else if (nusers == 1) {
                int uj = -1;
                for (int j = 0; j < n_out; j++)
                    if (pl->mat[(size_t)j * (size_t)n_in + (size_t)i]) {
                        uj = j;
                        break;
                    }
                /* AArch64 (NEON/SVE) and RVV fold first->acc with a direct
                 * load (1 uop); otherwise load t + one move/xor (2). */
                if (!is_x86 &&
                    (isa == ISA_RVV || isa == ISA_NEON ||
                     isa == ISA_SVE) &&
                    i == first_in[uj])
                    uops += 1;
                else
                    uops += 2; /* load t + one move/xor */
            } else {
                uops += 1 + nusers; /* load t + fan-out */
            }
        }
    }
    return uops;
}

/* scheduled body uops for nu lanes. Mirrors the schedule-driven emitters:
 * phase 1 forms each temp (2 uops on every x86 ISA: ld + folded mem-xor;
 * 3 uops on AArch64/RVV: 2 ld + xor) and fans out; phase 2 streams
 * surviving inputs; duplicates are moves; dead rows are zeroed. */
static long xs_body_sched(const xorsched_t *s, isa_t isa, int nu,
                          int nhoist)
{
    int is_x86 = (isa == ISA_SSE || isa == ISA_AVX2 || isa == ISA_AVX512);
    long uops = 0;
    uops += (long)s->ndead_plan * nu;
    for (int k = 0; k < s->ntemps; k++) {
        if (is_x86) {
            if (s->temps[k].a >= nhoist)
                uops += 1;
            if (s->temps[k].b >= nhoist)
                uops += 1;
        }
        /* formation: 2 uops on every x86 ISA (ld + folded mem-xor;
         * SSE uses the pxor-m128 form, 1 uop); 3 elsewhere (2 ld + xor) */
        uops += (long)(is_x86 ? 2 : 3) * nu;
        uops += (long)s->temps[k].nusers * nu; /* fan-out */
    }
    for (int i = 0; i < s->n_in; i++) {
        int nusers = s->in_nusers[i];
        if (!nusers)
            continue;
        if (is_x86 && i >= nhoist)
            uops += 1;
        int uw = s->in_users[i][0];
        int is_first = (nusers == 1 && s->row_first[uw] == s->ntemps + i);
        for (int u = 0; u < nu; u++) {
            if (is_x86) {
                if (is_first)
                    uops += 1; /* direct load into dead acc */
                else if (nusers == 1)
                    uops += 1; /* folded mem-xor (1 uop every x86 ISA) */
                else
                    uops += 1 + nusers; /* load t + fan-out */
            } else if (isa == ISA_RVV) {
                if (is_first)
                    uops += 1; /* direct vle into dead acc */
                else
                    uops += 1 + nusers;
            } else {
                /* AArch64: always load-into-t + fan-out (copy/xor) */
                uops += 1 + nusers;
            }
        }
    }
    uops += (long)s->ndup_live * nu;
    return uops;
}

long xorsched_version_uops(const xorsched_t *s, const plan_t *pl,
                           isa_t isa, int U, int sched, int nt, int notail)
{
    int n_in = pl->n_in;
    int nhoist = (n_in < 2) ? 0 : (n_in < 5 ? n_in : 5);
    int first_in[RJ_MAX_DISKS];
    xs_first_in(pl, first_in);
    (void)notail; /* RVV: remainder excluded either way; x86/AArch64: notail
                   * only omits the uncounted scalar/predicated tail. */

    /* Store uops: NEON pairs adjacent lanes with STNP on the nt path
     * (one uop per pair, trailing lane falls back to STR); every other
     * ISA emits one uop per vector on both paths (RVV nt is an uncounted
     * NTL.ALL hint). */
    long stU = (nt && isa == ISA_NEON) ? (long)pl->n_out * ((U + 1) / 2)
                                       : (long)pl->n_out * U;

    if (isa == ISA_RVV) {
        /* Full version: gen_body is emitted once with nu=U in the
         * counted full-group loop (F6). The remainder loop is
         * steady-state-excluded (like AArch64's predicated tail and
         * the x86 scalar tail): its body and vsetvli are not counted.
         * Stores are inside gen_body; 1 counted vsetvli (the F6
         * prologue). Prologue const setup is uncounted, and all other
         * prologue instructions use raw em_* calls. Notail twin: the
         * counted loop is the whole version, so only the single nu=U
         * body is emitted, plus the one prologue vsetvli. */
        long bU = sched ? xs_body_sched(s, isa, U, nhoist)
                        : xs_body_naive(pl, isa, U, nhoist, first_in);
        return (bU + stU) + 1;
    }
    /* x86/AArch64: the notail twin omits only the scalar tail, which
     * emits no counted uops, so the notail prediction is identical. */
    long bU = sched ? xs_body_sched(s, isa, U, nhoist)
                    : xs_body_naive(pl, isa, U, nhoist, first_in);
    return bU + stU; /* stores */
}

int xorsched_need_regs(isa_t isa, const plan_t *pl, int U)
{
    int base = plan_need_regs(isa, MUL_NIBBLE, pl, U, 0, 0, 0,
                              pl->want_crc);
    if (pl->kind != P_XOR)
        return base;
    /* AVX folds the temp formation into a memory XOR (one live temp);
     * SSE/AArch64/RVV need a second temp register. */
    if (isa == ISA_AVX2 || isa == ISA_AVX512)
        return base;
    return base + 1;
}

/* ---- bit-level (gate-level) XOR scheduling for GF-multiply rows ----
 * (track A1). See xorsched.h for the bit-program format.
 *
 * Guards: the greedy pair table is nbits_in*nbits_in ints; beyond these
 * sizes no schedule is built and the naive emitter is used. ni_sched is
 * capped at 64 (512 input bits -> 1MB pair table); larger plans keep the
 * nibble path.
 */
#define XB_MAX_TEMPS 1024
#define XB_MAX_DIMS 131072 /* (8*ni_sched) * (8*n_out) */
#define XB_MAX_NIN 64

/* Test hook: fail the Nth allocation inside xorbit_build (counting from
 * 0), to exercise the OOM unwind path. -1 (default) disables. Armed via
 * xorbit_test_oom_at(); inert otherwise. Not thread-safe: tests only. */
static int xb_oom_fail_at = -1;

void xorbit_test_oom_at(int n)
{
    xb_oom_fail_at = n;
}

static void *xb_oom_realloc(void *p, size_t n)
{
    if (xb_oom_fail_at >= 0) {
        if (xb_oom_fail_at-- == 0)
            return NULL;
    }
    return realloc(p, n);
}

static void *xb_oom_calloc(size_t nmemb, size_t size)
{
    if (xb_oom_fail_at >= 0) {
        if (xb_oom_fail_at-- == 0)
            return NULL;
    }
    return calloc(nmemb, size);
}

/* All xorbit_build allocations go through these so the test hook can
 * inject a failure at any site. */
#define XB_MALLOC(n) xb_oom_realloc(NULL, (n))
#define XB_CALLOC(nmemb, size) xb_oom_calloc((nmemb), (size))
#define XB_REALLOC(p, n) xb_oom_realloc((p), (n))

/* Scheduler-visible coefficient of output row j, scheduler input i:
 * the raw matrix for P_MULACC, the delta columns duplicated over
 * old/new for P_DELTA, the Horner expansion c_j^(n-1-i) for P_HORNER. */
static uint8_t xb_coeff(const plan_t *pl, int j, int i)
{
    if (pl->kind == P_DELTA)
        return pl->mat[(size_t)j * (size_t)pl->nchanged +
                       (size_t)(i % pl->nchanged)];
    if (pl->kind == P_HORNER)
        return gf_pow(pl->horner_c[j], (unsigned)(pl->n_in - 1 - i));
    return pl->mat[(size_t)j * (size_t)pl->n_in + (size_t)i];
}

typedef struct {
    int *atoms; /* input-bit ids 0..nbits_in-1, temps nbits_in+m */
    int n, cap;
} xb_row_t;

/* Returns 0 on success, -1 on allocation failure. */
static int xb_row_add(xb_row_t *r, int atom)
{
    if (r->n == r->cap) {
        int ncap = r->cap ? r->cap * 2 : 8;
        int *na = XB_REALLOC(r->atoms, (size_t)ncap * sizeof(int));
        if (!na)
            return -1;
        r->atoms = na;
        r->cap = ncap;
    }
    r->atoms[r->n++] = atom;
    return 0;
}

/* remove first occurrence of atom; rows hold each atom at most once */
static void xb_row_remove(xb_row_t *r, int atom)
{
    for (int i = 0; i < r->n; i++) {
        if (r->atoms[i] == atom) {
            r->atoms[i] = r->atoms[--r->n];
            return;
        }
    }
    assert(!"xb_row_remove: atom not present");
}

static int xb_row_has_atom(const xb_row_t *r, int atom)
{
    for (int i = 0; i < r->n; i++)
        if (r->atoms[i] == atom)
            return 1;
    return 0;
}

static void xb_free_greedy(xb_row_t *rows, long nbits_out, uint8_t *has,
                           int *counts, int *ta, int *tb)
{
    if (rows) {
        for (long ob = 0; ob < nbits_out; ob++)
            free(rows[ob].atoms);
        free(rows);
    }
    free(has);
    free(counts);
    free(ta);
    free(tb);
}

/* Sort output-bit ids by (bit position, byte): the emitter fans a value
 * out to all users sharing a bit position with one shift, so this order
 * makes each bit-position group consecutive. */
static int xb_cmp_bj(const void *pa, const void *pb)
{
    int a = *(const int *)pa, b = *(const int *)pb;
    int ba = a & 7, bb = b & 7;
    if (ba != bb)
        return ba - bb;
    return (a >> 3) - (b >> 3);
}

xorbit_t *xorbit_build(const plan_t *pl)
{
    if (!pl)
        return NULL;
    if (pl->kind != P_MULACC && pl->kind != P_DELTA &&
        pl->kind != P_HORNER)
        return NULL;
    int n_out = pl->n_out;
    int ni_sched =
        (pl->kind == P_DELTA) ? 2 * pl->nchanged : pl->n_in;
    if (n_out < 1 || ni_sched < 1)
        return NULL;
    if (ni_sched > XB_MAX_NIN)
        return NULL;
    if ((pl->kind == P_MULACC || pl->kind == P_DELTA) && !pl->mat)
        return NULL;
    if (pl->kind == P_DELTA && pl->nchanged < 1)
        return NULL;
    if (pl->kind == P_HORNER && !pl->horner_c)
        return NULL;
    long nbits_in = 8L * (long)ni_sched;
    long nbits_out = 8L * (long)n_out;
    if (nbits_in * nbits_out > XB_MAX_DIMS)
        return NULL;

    /* 1. incidence sets: for each output bit (j,b), the input bits
     * (i,k) with bitmatrix(C[j][i])[b][k] = 1. Column k of the
     * bitmatrix is gf_mul(C, 1<<k); bit b of that column feeds
     * output bit b. */
    xb_row_t *rows = XB_CALLOC((size_t)nbits_out, sizeof(xb_row_t));
    uint8_t *has =
        XB_CALLOC((size_t)nbits_out * (size_t)nbits_in, 1);
    int *counts =
        XB_CALLOC((size_t)nbits_in * (size_t)nbits_in, sizeof(int));
    int *ta = XB_MALLOC(XB_MAX_TEMPS * sizeof(int));
    int *tb = XB_MALLOC(XB_MAX_TEMPS * sizeof(int));
    if (!rows || !has || !counts || !ta || !tb) {
        xb_free_greedy(rows, nbits_out, has, counts, ta, tb);
        return NULL;
    }
    int oom = 0;
    int ntemps = 0;
    xorbit_t *s = NULL;
    for (int j = 0; j < n_out && !oom; j++) {
        for (int b = 0; b < 8 && !oom; b++) {
            long ob = 8L * (long)j + (long)b;
            for (int i = 0; i < ni_sched && !oom; i++) {
                uint8_t c = xb_coeff(pl, j, i);
                if (!c)
                    continue;
                /* columns of the bitmatrix of x -> c*x */
                uint8_t col[8];
                for (int k = 0; k < 8; k++)
                    col[k] = gf_mul(c, (uint8_t)(1u << k));
                for (int k = 0; k < 8; k++) {
                    if (!((col[k] >> b) & 1u))
                        continue;
                    int q = 8 * i + k;
                    if (xb_row_add(&rows[ob], q) != 0) {
                        oom = 1;
                        break;
                    }
                    has[ob * nbits_in + q] = 1;
                }
            }
            for (int x = 0; x < rows[ob].n; x++)
                for (int y = x + 1; y < rows[ob].n; y++) {
                    int qa = rows[ob].atoms[x], qb = rows[ob].atoms[y];
                    int p = qa < qb ? qa : qb, q = qa < qb ? qb : qa;
                    counts[(size_t)p * (size_t)nbits_in + (size_t)q]++;
                }
        }
    }
    if (oom)
        goto done;

    /* 2. greedy flat pair extraction over input-bit pairs (never
     * nested: temps are pairs of input bits only, like xorsched). */
    for (;;) {
        long ba = -1, bb = -1;
        int best = 0;
        for (long a = 0; a < nbits_in; a++)
            for (long b = a + 1; b < nbits_in; b++) {
                int cc = counts[a * nbits_in + b];
                if (cc > best) {
                    best = cc;
                    ba = a;
                    bb = b;
                }
            }
        if (best < 2 || ntemps >= XB_MAX_TEMPS)
            break;
        ta[ntemps] = (int)ba;
        tb[ntemps] = (int)bb;
        int tatom = (int)nbits_in + ntemps;
        ntemps++;
        for (long ob = 0; ob < nbits_out; ob++) {
            uint8_t *h = has + ob * nbits_in;
            if (!h[ba] || !h[bb])
                continue;
            /* the row loses pairs (ba,bb), (ba,x), (bb,x) */
            counts[ba * nbits_in + bb]--;
            for (long x = 0; x < nbits_in; x++) {
                if (!h[x] || x == ba || x == bb)
                    continue;
                long p = x < ba ? x : ba, q = x < ba ? ba : x;
                counts[p * nbits_in + q]--;
                p = x < bb ? x : bb, q = x < bb ? bb : x;
                counts[p * nbits_in + q]--;
            }
            h[ba] = h[bb] = 0;
            xb_row_remove(&rows[ob], (int)ba);
            xb_row_remove(&rows[ob], (int)bb);
            if (xb_row_add(&rows[ob], tatom) != 0) {
                oom = 1;
                break;
            }
        }
        if (oom)
            break;
    }
    if (oom)
        goto done;

    if (ntemps > 0 && !oom) {
        s = XB_CALLOC(1, sizeof(*s));
        if (!s) {
            oom = 1;
            goto done;
        }
        s->n_in_sched = ni_sched;
        s->n_out = n_out;
        s->n_in = pl->n_in;
        s->ntemps = ntemps;
        s->kind = pl->kind;
        s->temps = XB_CALLOC((size_t)ntemps, sizeof(xorbit_temp_t));
        s->bit_natoms = XB_CALLOC((size_t)nbits_out, sizeof(int));
        s->bit_atoms = XB_CALLOC((size_t)nbits_out, sizeof(int *));
        s->inbit_nusers = XB_CALLOC((size_t)nbits_in, sizeof(int));
        s->inbit_users = XB_CALLOC((size_t)nbits_in, sizeof(int *));
        if (!s->temps || !s->bit_natoms || !s->bit_atoms ||
            !s->inbit_nusers || !s->inbit_users) {
            oom = 1;
            goto done;
        }
        /* temps */
        for (int m = 0; m < ntemps; m++) {
            s->temps[m].a = ta[m];
            s->temps[m].b = tb[m];
            int nu = 0;
            for (long ob = 0; ob < nbits_out; ob++)
                if (xb_row_has_atom(&rows[ob], (int)nbits_in + m))
                    nu++;
            s->temps[m].nusers = nu;
            s->temps[m].users = XB_MALLOC((size_t)nu * sizeof(int));
            if (!s->temps[m].users && nu > 0) {
                oom = 1;
                goto done;
            }
            nu = 0;
            for (long ob = 0; ob < nbits_out; ob++)
                if (xb_row_has_atom(&rows[ob], (int)nbits_in + m))
                    s->temps[m].users[nu++] = (int)ob;
            qsort(s->temps[m].users, (size_t)nu, sizeof(int), xb_cmp_bj);
        }
        /* output bits: temps in formation order, then input bits
         * ascending */
        for (long ob = 0; ob < nbits_out; ob++) {
            int nt = 0, nq = 0;
            for (int i = 0; i < rows[ob].n; i++) {
                if (rows[ob].atoms[i] >= nbits_in)
                    nt++;
                else
                    nq++;
            }
            s->bit_natoms[ob] = nt + nq;
            s->bit_atoms[ob] = XB_MALLOC((size_t)(nt + nq) * sizeof(int));
            if (!s->bit_atoms[ob] && nt + nq > 0) {
                oom = 1;
                goto done;
            }
            int w = 0;
            for (int m = 0; m < ntemps; m++)
                if (xb_row_has_atom(&rows[ob], (int)nbits_in + m))
                    s->bit_atoms[ob][w++] = m; /* temp m -> atom m */
            for (long q = 0; q < nbits_in; q++)
                if (has[ob * nbits_in + q])
                    s->bit_atoms[ob][w++] = ntemps + (int)q;
            assert(w == nt + nq);
        }
        /* per-input-bit direct users */
        for (long q = 0; q < nbits_in; q++) {
            int nu = 0;
            for (long ob = 0; ob < nbits_out; ob++)
                if (has[ob * nbits_in + q])
                    nu++;
            s->inbit_nusers[q] = nu;
            s->inbit_users[q] = XB_MALLOC((size_t)nu * sizeof(int));
            if (!s->inbit_users[q] && nu > 0) {
                oom = 1;
                goto done;
            }
            nu = 0;
            for (long ob = 0; ob < nbits_out; ob++)
                if (has[ob * nbits_in + q])
                    s->inbit_users[q][nu++] = (int)ob;
            qsort(s->inbit_users[q], (size_t)nu, sizeof(int), xb_cmp_bj);
        }
        /* dead output bytes (no atoms in any of their 8 bits) */
        s->ndead_bytes = 0;
        for (int j = 0; j < n_out; j++) {
            int live = 0;
            for (int b = 0; b < 8; b++)
                if (s->bit_natoms[8 * j + b] > 0) {
                    live = 1;
                    break;
                }
            if (!live)
                s->ndead_bytes++;
        }
    }

done:
    xb_free_greedy(rows, nbits_out, has, counts, ta, tb);
    if (oom) {
        if (s) {
            if (s->temps) {
                for (int m = 0; m < ntemps; m++)
                    free(s->temps[m].users);
                free(s->temps);
            }
            if (s->bit_atoms) {
                for (long ob = 0; ob < nbits_out; ob++)
                    free(s->bit_atoms[ob]);
                free(s->bit_atoms);
            }
            free(s->bit_natoms);
            if (s->inbit_users) {
                for (long q = 0; q < nbits_in; q++)
                    free(s->inbit_users[q]);
                free(s->inbit_users);
            }
            free(s->inbit_nusers);
            free(s);
        }
        return NULL;
    }
    if (ntemps == 0 || !s) {
        xorbit_free(s);
        return NULL;
    }
    return s;
}

void xorbit_free(xorbit_t *s)
{
    if (!s)
        return;
    for (int m = 0; m < s->ntemps; m++)
        free(s->temps[m].users);
    free(s->temps);
    for (int ob = 0; ob < 8 * s->n_out; ob++)
        free(s->bit_atoms[ob]);
    free(s->bit_natoms);
    free(s->bit_atoms);
    for (long q = 0; q < 8L * (long)s->n_in_sched; q++)
        free(s->inbit_users[q]);
    free(s->inbit_nusers);
    free(s->inbit_users);
    free(s);
}

/* ---- static cost model for the bit-program twin ---- */

int xorbit_need_regs(isa_t isa, const plan_t *pl, int U)
{
    /* n_out*U accumulators + inreg + r1 + r2 (+ mask1, except RVV where
     * vand.vi takes an immediate). The nibble tables are dead on this
     * path.
     * Emitter redesign (track A3): + maxlive for temp-stack storage.
     * For flat temps (current scheduler), maxlive == 0. */
    int extra = (isa == ISA_RVV) ? 3 : 4;
    int maxlive = 0;
    if (pl && pl->bit_sched) {
        int *def = NULL, *lastuse = NULL;
        if (xorbit_temp_liveness(pl->bit_sched, &def, &lastuse) == 0) {
            maxlive = xorbit_maxlive(pl->bit_sched, def, lastuse);
        }
        free(def);
        free(lastuse);
    }
    return pl->n_out * U + extra + maxlive;
}

/* ---- static cost model for the bit-program twin ----
 * (plan_candidates gate + emitter lockstep tests).
 * Mirrors the x86_64 emitter's counting convention exactly:
 * body + stores per group; prologue const setup uncounted; prefetch
 * excluded (the gate passes prefetch_dist=0). Only x86 ISAs (the twin
 * is x86-only); -1 otherwise. */

/* nhoist mirror of ptr_hoist_count (x86_64): P_DELTA never hoists. */
static int xb_nhoist(const plan_t *pl)
{
    if (pl->kind == P_DELTA)
        return 0;
    int n_in = pl->n_in;
    return (n_in < 2) ? 0 : (n_in < 5 ? n_in : 5);
}

/* Fan-out uops for one (b, jlist) group, mirroring xorbit_fanout.
 * touched[j] is updated exactly like the emitter's. */
static long xb_fanout_uops(const int *users, int nusers, isa_t isa,
                           int is_delta, int *touched)
{
    long uops = 0;
    /* c_sllw cost: SSE needs mov+shift (vd != s1); AVX/NEON/SVE/RVV have
     * 3-operand shifts (1 uop). */
    long sh = (isa == ISA_SSE) ? 2 : 1;
    int w = 0;
    while (w < nusers) {
        int b = users[w] & 7;
        int w1 = w;
        while (w1 < nusers && (users[w1] & 7) == b)
            w1++;
        int nj = w1 - w;
        int j0 = users[w] >> 3;
        if (nj == 1 && !is_delta && !touched[j0]) {
            /* single first-touch user: shift/move straight into acc */
            uops += (b == 0) ? 1 : sh;
            touched[j0] = 1;
        } else if (b == 0) {
            for (int x = w; x < w1; x++) {
                int j = users[x] >> 3;
                uops += 1; /* c_vmov / c_vpxor */
                if (!is_delta && !touched[j])
                    touched[j] = 1;
            }
        } else {
            uops += sh; /* c_sllw(vtmp, vsrc): vd != s1 */
            for (int x = w; x < w1; x++) {
                int j = users[x] >> 3;
                uops += 1; /* c_vmov / c_vpxor */
                if (!is_delta && !touched[j])
                    touched[j] = 1;
            }
        }
        w = w1;
    }
    return uops;
}

/* Scheduled body uops for U vectors. Direct simulation of
 * gen_body_xorbit's emission order and 1-entry input-byte cache. */
static long xb_body_sched(const xorbit_t *s, isa_t isa, int U, int nhoist)
{
    int is_sse = (isa == ISA_SSE);
    int is_x86 = (isa == ISA_SSE || isa == ISA_AVX2 || isa == ISA_AVX512);
    int is_aarch64 = (isa == ISA_NEON || isa == ISA_SVE);
    int n_out = s->n_out;
    int ni_sched = s->n_in_sched;
    int is_delta = (s->kind == P_DELTA);
    long uops = 0;
    if (is_aarch64)
        uops += 1; /* prologue mask setup (DUP/MOVI #1), counted for lockstep */
    if (!is_delta)
        uops += (long)s->ndead_bytes * U; /* prologue c_vzero */
    else
        uops += (long)n_out * U; /* seed accs: c_ld (mov uncounted) */
    int touched[RJ_MAX_DISKS];
    for (int u = 0; u < U; u++) {
        for (int j = 0; j < n_out; j++)
            touched[j] = 0;
        int cur_in = -1;
        /* phase 1: temps */
        for (int m = 0; m < s->ntemps; m++) {
            int ia = s->temps[m].a >> 3, ib = s->temps[m].b >> 3;
            if (cur_in != ia) {
                /* x86 only: AArch64/RVV inptr (c_inptr/c_ldptr) is 0 uops. */
                if (!is_delta && ia >= nhoist && is_x86)
                    uops += 1; /* body_inptr mov */
                uops += 1;     /* c_ld */
                cur_in = ia;
            }
            /* bit extraction: shift + and. SSE: 2+1=3 (mov+shift); others: 1+1=2. */
            uops += is_sse ? 3 : 2;
            if (cur_in != ib) {
                if (!is_delta && ib >= nhoist && is_x86)
                    uops += 1;
                uops += 1;
                cur_in = ib;
            }
            uops += is_sse ? 3 : 2; /* bit extraction: shift + and */
            uops += 1; /* c_vpxor */
            uops += xb_fanout_uops(s->temps[m].users, s->temps[m].nusers,
                                   isa, is_delta, touched);
        }
        /* phase 2: surviving input bits */
        for (long q = 0; q < 8L * (long)ni_sched; q++) {
            if (!s->inbit_nusers[q])
                continue;
            int i = (int)(q >> 3);
            if (cur_in != i) {
                if (!is_delta && i >= nhoist && is_x86)
                    uops += 1;
                uops += 1;
                cur_in = i;
            }
            uops += is_sse ? 3 : 2; /* bit extraction: shift + and */
            uops += xb_fanout_uops(s->inbit_users[q],
                                   s->inbit_nusers[q], isa, is_delta,
                                   touched);
        }
    }
    return uops;
}

/* first contributing column per row (-1 if none), from the matrix */
static void xb_first_col(const plan_t *pl, int ncols, int *first_col)
{
    int n_out = pl->n_out;
    for (int j = 0; j < n_out; j++) {
        first_col[j] = -1;
        for (int i = 0; i < ncols; i++) {
            if (pl->mat[(size_t)j * (size_t)ncols + (size_t)i]) {
                first_col[j] = i;
                break;
            }
        }
    }
}

/* AArch64 naive nibble body uops for U vectors (MUL_NIBBLE, no
 * ternlog/permi2b/remat/pipe/pred). Mirrors src/aarch64/jit.c exactly:
 * mulacc_column for P_MULACC/P_DELTA, the inline i-loop for P_HORNER.
 * Excludes prologue const-setup (mask DUP/MOVI + HORNER table loads).
 * NOTE: The AArch64 emitter DOES count prologue (gen_const_setup adds
 * uops), while this model does not. This makes the gate (u_sched < u_naive)
 * require winning by more than the prologue cost, not a strict 1-uop win.
 * The gate remains SOUND (no false admits) but may miss narrow wins.
 * TODO: Count prologue in model for tight gate. See adversarial review.
 *
 * is_sve selects NEON vs SVE counting (c_ldpool is 2 uops on SVE, 1 on
 * NEON; the first-column nontrivial accumulate is 1 uop on SVE via
 * unpredicated EOR, 2 on NEON via mov+eor). SVE2 EOR3 is assumed
 * available: this makes the model a LOWER BOUND on the true naive cost
 * (on non-SVE2 hardware the true naive is costlier), so the
 * enumeration gate (u_sched < u_naive) stays sound -- it may miss wins
 * on non-SVE2, but never admits a losing twin. */
static long xb_body_naive_aarch64(const plan_t *pl, int U, int is_sve)
{
    int n_in = pl->n_in, n_out = pl->n_out;
    int use_eor3 = is_sve; /* lower bound; see above */
    int ldpool = is_sve ? 2 : 1;
    long uops = 0;
    int first_col[RJ_MAX_DISKS];

    if (pl->kind == P_MULACC) {
        xb_first_col(pl, n_in, first_col);
        /* No dead-row prologue on AArch64 (first_col logic overwrites
         * accumulators on first touch). */
        for (int i = 0; i < n_in; i++) {
            int any = 0;
            for (int j = 0; j < n_out; j++)
                if (pl->mat[(size_t)j * (size_t)n_in + (size_t)i]) {
                    any = 1;
                    break;
                }
            if (!any)
                continue;
            /* c_inptr: 0 uops. */
            for (int j = 0; j < n_out; j++) {
                uint8_t c = pl->mat[(size_t)j * (size_t)n_in + (size_t)i];
                if (!c)
                    continue;
                int first = (i == first_col[j]);
                /* Hoisted table load per (j,i), nontrivial only. */
                if (c != 1)
                    uops += 2L * ldpool;
                for (int u = 0; u < U; u++) {
                    uops += 1; /* c_ld t */
                    if (c == 1) {
                        uops += 1; /* copy (first) / xor */
                    } else {
                        uops += 4; /* and3, lsr4, 2 tbl */
                        if (first)
                            uops += is_sve ? 1 : 2;
                        else
                            uops += use_eor3 ? 1 : 2;
                    }
                }
            }
        }
    } else if (pl->kind == P_DELTA) {
        int nc = pl->nchanged;
        /* seed accs from oldpar: c_inptr (0) + c_ld per (j,u). */
        uops += (long)n_out * U;
        for (int i = 0; i < nc; i++) {
            int any = 0;
            for (int j = 0; j < n_out; j++)
                if (pl->mat[(size_t)j * (size_t)nc + (size_t)i]) {
                    any = 1;
                    break;
                }
            if (!any)
                continue;
            /* c_inptr + raw X3 ldr/add: 0 uops. first_col=NULL: never
             * first, always accumulate. */
            for (int j = 0; j < n_out; j++) {
                uint8_t c = pl->mat[(size_t)j * (size_t)nc + (size_t)i];
                if (!c)
                    continue;
                if (c != 1)
                    uops += 2L * ldpool;
                for (int u = 0; u < U; u++) {
                    uops += 3; /* delta formation: 2 ld + xor */
                    if (c == 1) {
                        uops += 1;
                    } else {
                        uops += 4;
                        uops += use_eor3 ? 1 : 2;
                    }
                }
            }
        }
    } else { /* P_HORNER */
        for (int i = 0; i < n_in; i++) {
            /* mulacc_col_info: any nonzero in this column. (Real HORNER
             * plans carry the Vandermonde matrix; a NULL mat is treated
             * as all-active, matching the emitter's crash-free prefix.) */
            int any = 1;
            if (pl->mat) {
                any = 0;
                for (int j = 0; j < n_out; j++)
                    if (pl->mat[(size_t)j * (size_t)n_in + (size_t)i]) {
                        any = 1;
                        break;
                    }
            }
            if (!any)
                continue;
            /* c_inptr (u==0): 0 uops. */
            for (int u = 0; u < U; u++) {
                if (i == 0) {
                    uops += 1; /* c_ld t */
                    uops += n_out; /* c_copy per row */
                } else {
                    uops += 1; /* c_ld r1 */
                    for (int j = 0; j < n_out; j++) {
                        uint8_t c = pl->horner_c[j];
                        if (c == 1 || c == 0) {
                            uops += 1;
                        } else {
                            uops += 4;
                            uops += use_eor3 ? 1 : 2;
                        }
                    }
                }
            }
        }
    }
    return uops;
}

/* RVV naive nibble body uops for U vectors (MUL_NIBBLE, no
 * ternlog/permi2b/remat/pipe, no Zvbc). Mirrors src/riscv/jit.c exactly:
 * mulacc_rows_c for P_MULACC/P_DELTA, the inline i-loop for P_HORNER.
 * Excludes prologue const-setup and vsetvli (uncounted by convention)
 * and stores (added by xorbit_version_uops' caller). Prefetch is
 * excluded (the gate always passes prefetch_dist=0). Uop counts are
 * LMUL-independent (only mem scales with LMUL). */
static long xb_body_naive_rvv(const plan_t *pl, int U)
{
    int n_in = pl->n_in, n_out = pl->n_out;
    long uops = 0;
    int first_col[RJ_MAX_DISKS];

    if (pl->kind == P_MULACC || pl->kind == P_DELTA) {
        int ncols = (pl->kind == P_DELTA) ? pl->nchanged : n_in;
        if (pl->kind == P_MULACC) {
            xb_first_col(pl, n_in, first_col);
            for (int j = 0; j < n_out; j++)
                if (first_col[j] < 0)
                    uops += U; /* dead-row prologue vzero */
        }
        /* Table hoisting: single distinct nontrivial coefficient in
         * the whole matrix -> tables live from the prologue
         * (uncounted); per-column c_ldtab calls are skipped. */
        uint8_t seen_all[256] = { 0 };
        int ndistinct = 0;
        for (int k = 0; k < n_out * ncols; k++) {
            uint8_t c = pl->mat[k];
            if (c < 2 || seen_all[c])
                continue;
            seen_all[c] = 1;
            ndistinct++;
        }
        int hoist = (ndistinct == 1);
        if (pl->kind == P_DELTA) {
            /* seed accs from oldpar: c_vle per (j,u). */
            uops += (long)n_out * U;
        }
        for (int i = 0; i < ncols; i++) {
            int any = 0;
            for (int j = 0; j < n_out; j++)
                if (pl->mat[(size_t)j * (size_t)ncols + (size_t)i]) {
                    any = 1;
                    break;
                }
            if (!any)
                continue;
            if (pl->kind == P_MULACC) {
                /* input loads: c_vle per u. */
                uops += U;
            } else {
                /* delta formation: 2 vle + vxor per u. */
                uops += 3L * U;
            }
            /* c == 1 rows: 1 uop per (j,u) (vmv if first, else vxor). */
            for (int j = 0; j < n_out; j++)
                if (pl->mat[(size_t)j * (size_t)ncols + (size_t)i] == 1)
                    uops += U;
            /* distinct nontrivial coefficients. */
            uint8_t seen[256] = { 0 };
            for (int j = 0; j < n_out; j++) {
                uint8_t c = pl->mat[(size_t)j * (size_t)ncols + (size_t)i];
                if (c < 2 || seen[c])
                    continue;
                seen[c] = 1;
                if (!hoist)
                    uops += 4; /* c_ldtab (win=0) */
                for (int jj = 0; jj < n_out; jj++) {
                    if (pl->mat[(size_t)jj * (size_t)ncols + (size_t)i] != c)
                        continue;
                    /* vand, vsrl4, 2 vrg, vxor, vmv/vxor: 6 per u. */
                    uops += 6L * U;
                }
            }
        }
    } else { /* P_HORNER */
        if (n_in == 0) {
            /* Degenerate: vzero per (j,u). */
            uops += (long)n_out * U;
        }
        for (int i = 0; i < n_in; i++) {
            /* input loads: c_vle per u. */
            uops += U;
            if (i == 0) {
                /* acc = t: vmv per (j,u). */
                uops += (long)n_out * U;
                continue;
            }
            for (int j = 0; j < n_out; j++) {
                uint8_t c = pl->horner_c[j];
                if (c == 1 || c == 0) {
                    uops += U; /* vxor/vmv */
                } else {
                    /* vand, vsrl4, 2 vrg, 2 vxor: 6 per u. Tables live
                     * in hreg from the prologue (uncounted). */
                    uops += 6L * U;
                }
            }
        }
    }
    return uops;
}

/* Naive nibble body uops for U vectors (MUL_NIBBLE, no ternlog/permi2b/
 * remat). Mirrors the x86_64 naive emitter's main path exactly,
 * including its quirks (DELTA pointer movs uncounted; AVX512-DELTA
 * NIBHI pinning). */
static long xb_body_naive(const plan_t *pl, isa_t isa, int U, int nhoist)
{
    /* AArch64 has its own naive emitter shape (hoisted tables,
     * EOR3, zero-cost inptr); the x86 mirror below would miscount it. */
    if (isa == ISA_NEON || isa == ISA_SVE)
        return xb_body_naive_aarch64(pl, U, isa == ISA_SVE);
    /* RVV has its own naive emitter shape (per-coefficient table
     * loads, hoisting, vzero dead rows); the x86 mirror below would
     * miscount it. */
    if (isa == ISA_RVV)
        return xb_body_naive_rvv(pl, U);
    int n_in = pl->n_in, n_out = pl->n_out;
    int is_sse = (isa == ISA_SSE);
    int is_avx512 = (isa == ISA_AVX512);
    long uops = 0;
    int first_col[RJ_MAX_DISKS];

    if (pl->kind == P_MULACC) {
        xb_first_col(pl, n_in, first_col);
        for (int j = 0; j < n_out; j++)
            if (first_col[j] < 0)
                uops += U; /* dead-row prologue vzero */
        for (int i = 0; i < n_in; i++) {
            int any = 0, neednib = 0;
            for (int j = 0; j < n_out; j++) {
                uint8_t c = pl->mat[(size_t)j * (size_t)n_in + (size_t)i];
                if (c) {
                    any = 1;
                    if (c != 1)
                        neednib = 1;
                }
            }
            if (!any)
                continue;
            if (i >= nhoist)
                uops += 1; /* body_inptr mov */
            for (int u = 0; u < U; u++) {
                uops += 1; /* c_ld t */
                if (neednib)
                    uops += 3; /* and, srlw, and */
                for (int j = 0; j < n_out; j++) {
                    uint8_t c =
                        pl->mat[(size_t)j * (size_t)n_in + (size_t)i];
                    if (!c)
                        continue;
                    if (c == 1) {
                        uops += 1;
                    } else {
                        uops += 4; /* 2 ldpool + 2 shuf */
                        uops += (i == first_col[j]) ? 1 : 2;
                    }
                }
            }
        }
    } else if (pl->kind == P_DELTA) {
        int nc = pl->nchanged;
        uops += (long)n_out * U; /* seed accs: c_ld (mov uncounted) */
        int nib_c = -1;          /* AVX512 NIBHI pinning (mulacc_nib_c) */
        for (int i = 0; i < nc; i++) {
            int any = 0, neednib = 0;
            for (int j = 0; j < n_out; j++) {
                uint8_t c = pl->mat[(size_t)j * (size_t)nc + (size_t)i];
                if (c) {
                    any = 1;
                    if (c != 1)
                        neednib = 1;
                }
            }
            if (!any)
                continue;
            /* two raw pointer movs (uncounted, like the emitter) */
            for (int u = 0; u < U; u++) {
                uops += is_sse ? 3 : 2; /* delta formation */
                if (neednib)
                    uops += 3;
                for (int j = 0; j < n_out; j++) {
                    uint8_t c =
                        pl->mat[(size_t)j * (size_t)nc + (size_t)i];
                    if (!c)
                        continue;
                    if (c == 1) {
                        uops += 1;
                    } else {
                        uops += 1 + 1; /* NIBLO ldpool + shuf */
                        if (is_avx512) {
                            if (nib_c != c) {
                                uops += 1; /* NIBHI ldpool */
                                nib_c = c;
                            }
                        } else {
                            uops += 1; /* NIBHI ldpool */
                        }
                        uops += 1 + 2; /* shuf + 2 xor */
                    }
                }
            }
        }
    } else { /* P_HORNER */
        for (int i = 0; i < n_in; i++) {
            if (i >= nhoist)
                uops += 1; /* body_inptr mov */
            for (int u = 0; u < U; u++) {
                if (i == 0) {
                    uops += 1; /* c_ld into acc[0] */
                    uops += n_out - 1; /* moves to the other accs */
                    continue;
                }
                uops += 1; /* c_ld t */
                for (int j = 0; j < n_out; j++) {
                    uint8_t c = pl->horner_c[j];
                    if (c == 1 || c == 0) {
                        uops += 1;
                    } else {
                        /* nibble split of acc (3) + 2 shuf from hreg
                         * (non-remat) + 2 xor */
                        uops += 3 + 2 + 2;
                    }
                }
            }
        }
    }
    return uops;
}

long xorbit_version_uops(const xorbit_t *s, const plan_t *pl,
                         isa_t isa, int U, int sched, int nt)
{
    if (isa != ISA_SSE && isa != ISA_AVX2 && isa != ISA_AVX512 &&
        isa != ISA_NEON && isa != ISA_SVE && isa != ISA_RVV)
        return -1;
    if (!pl || U < 1)
        return -1;
    if (sched && !s)
        return -1;
    /* nt is a wash on x86 (one uop per vector store either way);
     * accepted so tests can check +nt twins against the model. */
    (void)nt;
    int nhoist = xb_nhoist(pl);
    long stU = (long)pl->n_out * U; /* stores */
    long body = sched ? xb_body_sched(s, isa, U, nhoist)
                      : xb_body_naive(pl, isa, U, nhoist);
    /* RVV: the F6 prologue vsetvli is counted (src/riscv/jit.c). The
     * remainder loop is steady-state-excluded (its vsetvli is not
     * counted). */
    long vsetvli = (isa == ISA_RVV) ? 1 : 0;
    return body + stU + vsetvli;
}

/* ---- Temp-stack liveness for nested temps (emitter redesign, track A3) ----
 *
 * For each temp m in an xorbit_t schedule, compute:
 *   def[m]: formation index (== m, temps are in formation order)
 *   lastuse[m]: last temp index that uses m as an operand, or -1 if m is
 *               never used by another temp (only in fanout/output bits).
 *
 * IMPORTANT: In the current flat-only scheduler, temps[m].a/b are
 * input-bit ids (0 <= q < 8*n_in_sched), NEVER temp ids. The data
 * structure does not currently encode temp operands. Therefore,
 * lastuse[m] == -1 for all m (no temp is used by another temp).
 *
 * For FUTURE nested-temp support, the temps[m].a/b encoding would need
 * to distinguish temp ids from input-bit ids (e.g., via the atom-id
 * encoding: temp m -> m, input q -> ntemps+q). This function is written
 * to handle that future encoding: if a/b < ntemps, it's a temp id.
 *
 * The caller owns the returned arrays (free with free()).
 * Returns 0 on success, -1 on allocation failure. */
int xorbit_temp_liveness(const xorbit_t *s, int **def_out, int **lastuse_out)
{
    if (!s || s->ntemps <= 0)
        return -1;
    int n = s->ntemps;
    int *def = (int *)calloc((size_t)n, sizeof(int));
    int *lastuse = (int *)calloc((size_t)n, sizeof(int));
    if (!def || !lastuse) {
        free(def);
        free(lastuse);
        return -1;
    }
    for (int m = 0; m < n; m++) {
        def[m] = m;
        lastuse[m] = -1;
    }
    /* Find last use of each temp as an operand of a later temp.
     * In the current flat-only scheduler, temps[m].a/b are input-bit
     * ids, which are >= n (ntemps) when interpreted as atom ids, OR
     * they are small values that could be confused with temp ids.
     * To avoid false positives, we ONLY treat a/b as temp ids if the
     * scheduler explicitly marks them as such. Currently, it never does,
     * so this loop is a no-op. Future nested-temp scheduler must use
     * a distinct encoding (e.g., negative values or a flag) for temp
     * operands. */
    /* NOTE: Disabled until the data structure supports temp operands.
     * The current encoding cannot distinguish temp ids from input-bit
     * ids in temps[m].a/b. */
#if 0
    for (int m = 0; m < n; m++) {
        int a = s->temps[m].a, b = s->temps[m].b;
        if (a >= 0 && a < n)
            lastuse[a] = m;
        if (b >= 0 && b < n)
            lastuse[b] = m;
    }
#endif
    *def_out = def;
    *lastuse_out = lastuse;
    return 0;
}

/* Compute max simultaneously-live temps (for register pressure).
 * A temp m is live at point p if def[m] <= p <= lastuse[m].
 * For flat temps, maxlive == 0 (each temp dies immediately).
 * Returns the maxlive count. */
int xorbit_maxlive(const xorbit_t *s, const int *def, const int *lastuse)
{
    if (!s || s->ntemps <= 0)
        return 0;
    int n = s->ntemps;
    int maxlive = 0;
    for (int p = 0; p < n; p++) {
        int live = 0;
        for (int m = 0; m < n; m++) {
            if (def[m] <= p && p <= lastuse[m])
                live++;
        }
        if (live > maxlive)
            maxlive = live;
    }
    return maxlive;
}
