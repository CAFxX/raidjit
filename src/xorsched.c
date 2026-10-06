/* XOR-schedule / cross-row CSE (track A3). See xorsched.h for the design. */
#include "xorsched.h"

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

    /* Store uops: NEON pairs adjacent lanes with STNP on the nt path
     * (one uop per pair, trailing lane falls back to STR); every other
     * ISA emits one uop per vector on both paths (RVV nt is an uncounted
     * NTL.ALL hint). */
    long stU = (nt && isa == ISA_NEON) ? (long)pl->n_out * ((U + 1) / 2)
                                       : (long)pl->n_out * U;

    if (isa == ISA_RVV) {
        /* Full version: gen_body is emitted 3x (nu=U, U, 1); stores +
         * 3 vsetvli. Notail twin: the counted loop is the whole version
         * (dispatcher guarantees len % group == 0), so only the single
         * nu=U body is emitted, plus the one prologue vsetvli
         * (src/riscv/jit.c counts exactly one cg->uops += 1 outside
         * gen_body on the notail path). */
        long bU = sched ? xs_body_sched(s, isa, U, nhoist)
                        : xs_body_naive(pl, isa, U, nhoist, first_in);
        if (notail)
            return (bU + stU) + 1;
        long b1 = sched ? xs_body_sched(s, isa, 1, nhoist)
                        : xs_body_naive(pl, isa, 1, nhoist, first_in);
        long st1 = (long)pl->n_out;
        return 2 * (bU + stU) + (b1 + st1) + 3;
    }
    /* x86/AArch64: the notail twin omits only the scalar tail, which
     * emits no counted uops, so the notail prediction is identical. */
    long bU = sched ? xs_body_sched(s, isa, U, nhoist)
                    : xs_body_naive(pl, isa, U, nhoist, first_in);
    return bU + stU; /* stores */
}

int xorsched_need_regs(isa_t isa, const plan_t *pl, int U)
{
    int base = plan_need_regs(isa, MUL_NIBBLE, pl, U, 0);
    if (pl->kind != P_XOR)
        return base;
    /* AVX folds the temp formation into a memory XOR (one live temp);
     * SSE/AArch64/RVV need a second temp register. */
    if (isa == ISA_AVX2 || isa == ISA_AVX512)
        return base;
    return base + 1;
}
