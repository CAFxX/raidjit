/* Topology constructors and the internal (ndata/npar/matrix) view. */
#include "../include/raidjit.h"
#include "plan.h"
#include "gf.h"
#include "xorcode.h"
#include "matrix_search.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

struct rj_topology {
    int ndata, npar;
    uint8_t *mat; /* npar*ndata row-major, NULL when npar == 0 */
    char name[32];
    int xor_rows; /* rows per data disk for XOR array-code topologies
                   * (RDP/EVENODD/Liberation, expanded row model);
                   * 0 for everything else */
};

static rj_topology *topo_new(int ndata, int npar, const char *name)
{
    if (ndata < 1 || npar < 0 || ndata + npar > RJ_MAX_DISKS)
        return NULL;
    rj_topology *t = calloc(1, sizeof(*t));
    if (!t)
        return NULL;
    t->ndata = ndata;
    t->npar = npar;
    snprintf(t->name, sizeof(t->name), "%s", name);
    if (npar) {
        t->mat = malloc((size_t)npar * (size_t)ndata);
        if (!t->mat) {
            free(t);
            return NULL;
        }
    }
    return t;
}

rj_topology *rj_topo_raid0(int ndata)
{
    char nm[32];
    snprintf(nm, sizeof(nm), "raid0(%d)", ndata);
    return topo_new(ndata, 0, nm);
}

rj_topology *rj_topo_raid1(void)
{
    rj_topology *t = topo_new(1, 1, "raid1");
    if (t)
        t->mat[0] = 1; /* mirror = data * 1 */
    return t;
}

rj_topology *rj_topo_raid5(int ndata)
{
    if (ndata < 2)
        return NULL;
    char nm[32];
    snprintf(nm, sizeof(nm), "raid5(%d)", ndata);
    rj_topology *t = topo_new(ndata, 1, nm);
    if (t)
        memset(t->mat, 1, (size_t)ndata);
    return t;
}

rj_topology *rj_topo_raid6(int ndata)
{
    /* Benchmark-selected default per docs/xor_code_default.md: the
     * fastest MEASURED code for this ndata (RDP/EVENODD/Liberation lower
     * to expanded row-model topologies; RS fallback when no XOR code
     * applies or wins). rj_topo_raid6_rs() is the explicit RS opt-in.
     * NOTE: when an XOR code wins, the topology uses the expanded row
     * model -- in[d*R+r] is row r of data disk d -- not one pointer per
     * data disk. See src/xorcode.h and docs/xor_codes.md. */
    if (ndata < 3 || ndata > RS_MDS_MAX_NDATA)
        return NULL;
    rj_topology *t = NULL;
    switch (xorcode_default_for_ndata(ndata)) {
    case XORCODE_RDP:
        t = rj_topo_rdp(ndata + 1);
        break;
    case XORCODE_EVENODD:
        t = rj_topo_evenodd(ndata);
        break;
    case XORCODE_LIBERATION:
        t = rj_topo_liberation(ndata);
        break;
    case XORCODE_RS:
        break;
    }
    /* Safety net: a default-table entry naming an XOR code whose
     * constructor rejects the corresponding ndata (RDP needs ndata+1
     * prime, EVENODD needs ndata prime, Liberation needs the expanded
     * size within RJ_MAX_DISKS) is a table bug; fall back to RS rather
     * than returning NULL, so rj_topo_raid6() stays total on its
     * documented domain 3..RS_MDS_MAX_NDATA. Every table entry is
     * checked against its constructor by tests/test_xorcode.c
     * (test_default_table), so a bad entry fails loudly at test time
     * instead of silently taking this path. */
    if (!t)
        t = rj_topo_raid6_rs(ndata);
    return t;
}

rj_topology *rj_topo_raid6_rs(int ndata)
{
    if (ndata < 3 || ndata > RS_MDS_MAX_NDATA)
        return NULL;
    char nm[32];
    snprintf(nm, sizeof(nm), "raid6(%d)", ndata);
    rj_topology *t = topo_new(ndata, 2, nm);
    /* RS default via the plan-time search (docs/matrix_search.md). Since
     * A2 2.0 the search minimizes bitmatrix weight and emits strictly
     * lighter non-Vandermonde matrices, so the bytes differ from the
     * historical default by design (deliberate parity-format change; no
     * backward compatibility). Routes through matrix_search_default(),
     * never rs_default_matrix() directly (A6 proof gate). */
    if (t && matrix_search_default(ndata, 2, t->mat) != 0) {
        rj_topo_free(t);
        return NULL;
    }
    return t;
}

/* XOR array codes (RDP/EVENODD/Liberation). Expanded row model: the
 * topology's ndata = k*R row-disks, npar = 2*R parity rows; see
 * src/xorcode.h for the pointer layout contract. */

rj_topology *rj_topo_rdp(int p)
{
    if (!xorcode_rdp_valid(p))
        return NULL;
    int R = p - 1;
    /* Size math in size_t: (p-1)^2 overflows int for p > 46341 (signed
     * overflow is UB). The RJ_MAX_DISKS cap — the same engineering
     * envelope topo_new enforces, checked here before any narrowing —
     * doubles as the overflow guard: every p that would overflow also
     * exceeds the cap, so the casts below are exact. */
    size_t nd = (size_t)R * (size_t)R, np = 2u * (size_t)R;
    if (nd + np > RJ_MAX_DISKS)
        return NULL;
    char nm[32];
    snprintf(nm, sizeof(nm), "rdp(%d)", p);
    rj_topology *t = topo_new((int)nd, (int)np, nm);
    if (!t)
        return NULL;
    if (xorcode_rdp_matrix(p, t->mat) != 0) {
        rj_topo_free(t);
        return NULL;
    }
    t->xor_rows = R;
    return t;
}

rj_topology *rj_topo_evenodd(int p)
{
    if (!xorcode_evenodd_valid(p))
        return NULL;
    int R = p - 1;
    size_t nd = (size_t)p * (size_t)R, np = 2u * (size_t)R;
    if (nd + np > RJ_MAX_DISKS)
        return NULL;
    char nm[32];
    snprintf(nm, sizeof(nm), "evenodd(%d)", p);
    rj_topology *t = topo_new((int)nd, (int)np, nm);
    if (!t)
        return NULL;
    if (xorcode_evenodd_matrix(p, t->mat) != 0) {
        rj_topo_free(t);
        return NULL;
    }
    t->xor_rows = R;
    return t;
}

rj_topology *rj_topo_liberation(int k)
{
    if (k < 2)
        return NULL;
    /* Early bound before the prime search and the multiplication: any
     * k > RJ_MAX_DISKS fails the expanded-size cap below regardless
     * (w >= k, so k*w >= k^2), and skipping the search keeps absurd
     * inputs (e.g. INT_MAX) from burning a sqrt(n) trial-division
     * search on a parameter that can never construct. */
    if (k > RJ_MAX_DISKS)
        return NULL;
    int w = xorcode_next_prime(k < 3 ? 3 : k);
    if (!xorcode_liberation_valid(k, w))
        return NULL;
    size_t nd = (size_t)k * (size_t)w, np = 2u * (size_t)w;
    if (nd + np > RJ_MAX_DISKS)
        return NULL;
    char nm[32];
    snprintf(nm, sizeof(nm), "liberation(%d)", k);
    rj_topology *t = topo_new((int)nd, (int)np, nm);
    if (!t)
        return NULL;
    if (xorcode_liberation_matrix(k, w, t->mat) != 0) {
        rj_topo_free(t);
        return NULL;
    }
    t->xor_rows = w;
    return t;
}

int rj_xorcode_rows(const rj_topology *t)
{
    return t ? t->xor_rows : 0;
}

rj_topology *rj_topo_rs(int ndata, int npar)
{
    if (npar > RS_MDS_MAX_NPAR)
        return NULL;
    if (npar >= 2 && ndata > RS_MDS_MAX_NDATA)
        return NULL;
    char nm[32];
    snprintf(nm, sizeof(nm), "rs(%d,%d)", ndata, npar);
    rj_topology *t = topo_new(ndata, npar, nm);
    /* The default matrix comes from the plan-time search 2.0
     * (docs/matrix_search.md): the bitmatrix-weight objective emits
     * non-Vandermonde matrices, so the bytes differ from the historical
     * Vandermonde default by design (deliberate parity-format change).
     * Consequence: plan_select_strategy picks P_MULACC, not P_HORNER,
     * for these topologies. */
    if (t && npar > 0 && matrix_search_default(ndata, npar, t->mat) != 0) {
        rj_topo_free(t);
        return NULL;
    }
    return t;
}

rj_topology *rj_topo_custom(int ndata, int npar, const uint8_t *matrix)
{
    if (!matrix)
        return NULL;
    char nm[32];
    snprintf(nm, sizeof(nm), "custom(%d,%d)", ndata, npar);
    rj_topology *t = topo_new(ndata, npar, nm);
    if (t)
        memcpy(t->mat, matrix, (size_t)ndata * (size_t)npar);
    return t;
}

void rj_topo_free(rj_topology *t)
{
    if (t) {
        free(t->mat);
        free(t);
    }
}

int rj_topo_ndata(const rj_topology *t) { return t ? t->ndata : 0; }
int rj_topo_npar(const rj_topology *t) { return t ? t->npar : 0; }

/* ---------------- internal view ---------------- */

int topo_int_build(topo_int_t *ti, const rj_topology *t, char *err,
                   size_t errlen)
{
    if (!t) {
        if (err && errlen)
            snprintf(err, errlen, "null topology");
        return -1;
    }
    ti->ndata = t->ndata;
    ti->npar = t->npar;
    ti->mat = NULL;
    if (t->npar) {
        ti->mat = malloc((size_t)t->npar * (size_t)t->ndata);
        if (!ti->mat) {
            if (err && errlen)
                snprintf(err, errlen, "out of memory");
            return -1;
        }
        memcpy(ti->mat, t->mat, (size_t)t->npar * (size_t)t->ndata);
    }
    return 0;
}

void topo_int_free(topo_int_t *ti)
{
    free(ti->mat);
    ti->mat = NULL;
}

const char *topo_internal_name(const rj_topology *t)
{
    return t ? t->name : "?";
}

/* ---------------- op constructors ---------------- */

static int cmp_int(const void *a, const void *b)
{
    return *(const int *)a - *(const int *)b;
}

rj_op rj_op_encode(void)
{
    rj_op op;
    memset(&op, 0, sizeof(op));
    op.is_recover = 0;
    return op;
}

rj_op rj_op_recover(const int *failed, int nfailed)
{
    rj_op op;
    memset(&op, 0, sizeof(op));
    op.is_recover = 1;
    if (nfailed < 0)
        nfailed = 0;
    if (nfailed > RJ_MAX_DISKS)
        nfailed = RJ_MAX_DISKS;
    op.nfailed = nfailed;
    if (nfailed > 0 && failed) {
        op.failed = malloc(sizeof(int) * (size_t)nfailed);
        if (op.failed) {
            memcpy(op.failed, failed, sizeof(int) * (size_t)nfailed);
            qsort(op.failed, (size_t)nfailed, sizeof(int), cmp_int);
        } else {
            op.oom = 1; /* distinct from empty list; plan_build reports OOM */
        }
    }
    return op;
}

rj_op rj_op_parity_update(const int *changed, int nchanged)
{
    rj_op op;
    memset(&op, 0, sizeof(op));
    op.is_parity_update = 1;
    if (nchanged < 0)
        nchanged = 0;
    if (nchanged > RJ_MAX_DISKS)
        nchanged = RJ_MAX_DISKS;
    op.nchanged = nchanged;
    if (nchanged > 0) {
        if (!changed) {
            op.nchanged = 0;
            return op;
        }
        op.changed = malloc(sizeof(int) * (size_t)nchanged);
        if (op.changed) {
            memcpy(op.changed, changed, sizeof(int) * (size_t)nchanged);
            qsort(op.changed, (size_t)nchanged, sizeof(int), cmp_int);
        } else {
            op.oom = 1; /* distinct from empty list; plan_build reports OOM */
        }
    }
    return op;
}

void rj_op_free(rj_op *op)
{
    if (op) {
        free(op->failed);
        op->failed = NULL;
        op->nfailed = 0;
        free(op->changed);
        op->changed = NULL;
        op->nchanged = 0;
    }
}
