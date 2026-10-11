/* Study: does any non-power-of-2 unroll factor (U=3, U=5, U=6, U=7)
 * ever win the compile-time benchmark? Compiles each (topology, mode)
 * with benchmark=1 and reports the picked variant's U.
 *
 * Usage: ./study_nonpow2
 * Output: one line per config: <topo> <mode> <picked desc>
 * Summary at end: count of configs where a non-pow2 U was picked.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "raidjit.h"
#include "../src/gf.h"

static int n_nonpow2 = 0, n_total = 0;

static void run(const char *topo_name, rj_topology *t, const char *mode_name,
                int *failed, int nfailed)
{
    rj_options o;
    memset(&o, 0, sizeof o);
    o.benchmark = 1;
    /* Full default benchmark settings for faithful selection */

    rj_jit *j;
    char err[256] = {0};
    if (nfailed == 0) {
        rj_op op = rj_op_encode();
        j = rj_compile(t, &op, &o, err, sizeof err);
        rj_op_free(&op);
    } else {
        rj_op op = rj_op_recover(failed, nfailed);
        j = rj_compile(t, &op, &o, err, sizeof err);
        rj_op_free(&op);
    }
    if (!j) {
        printf("%-28s %-12s COMPILE-FAILED: %s\n", topo_name, mode_name,
               err);
        return;
    }
    rj_info info;
    rj_get_info(j, &info);
    n_total++;
    int u = 0;
    const char *p = strstr(info.desc, "U=");
    if (p)
        u = atoi(p + 2);
    int nonpow2 = (u == 3 || u == 5 || u == 6);
    if (nonpow2)
        n_nonpow2++;
    printf("%-28s %-12s %s%s\n", topo_name, mode_name, info.desc,
           nonpow2 ? "   <-- NON-POW2" : "");
    rj_free(j);
}

int main(void)
{
    /* (ndata, npar, is_raid6_vandermonde) */
    struct { int ndata, npar, vdm; const char *name; } topos[] = {
        {1, 1, 0, "raid1(1+1)"},
        {3, 1, 0, "raid5(3+1)"},
        {4, 1, 0, "raid5(4+1)"},
        {8, 1, 0, "raid5(8+1)"},
        {4, 2, 1, "raid6(4+2)"},
        {6, 2, 1, "raid6(6+2)"},
        {8, 2, 1, "raid6(8+2)"},
        {6, 3, 0, "rs(6,3)"},
        {10, 4, 0, "rs(10,4)-cauchy"},
    };
    int f1[] = {0};
    int f2[] = {0, 1};

    for (unsigned ti = 0; ti < sizeof(topos) / sizeof(topos[0]); ti++) {
        rj_topology *t;
        if (!strcmp(topos[ti].name, "rs(10,4)-cauchy")) {
            /* Custom Cauchy matrix: C[j][i] = inv(x_i ^ y_j), MDS by
             * construction. rj_topo_rs(10,4) is NULL (default matrix
             * guaranteed only for npar <= 3). */
            int nd = 10, np = 4;
            uint8_t *mat = malloc((size_t)np * nd);
            for (int j = 0; j < np && mat; j++)
                for (int i = 0; i < nd; i++)
                    mat[j * nd + i] =
                        gf_inv((uint8_t)((i + 1) ^ (nd + 1 + j)));
            t = mat ? rj_topo_custom(nd, np, mat) : NULL;
            free(mat);
        } else if (topos[ti].vdm)
            t = rj_topo_raid6(topos[ti].ndata);
        else if (topos[ti].npar == 1 && topos[ti].ndata > 1)
            t = rj_topo_raid5(topos[ti].ndata);
        else if (topos[ti].ndata == 1)
            t = rj_topo_raid1();
        else
            t = rj_topo_rs(topos[ti].ndata, topos[ti].npar);
        if (!t) {
            printf("%-28s TOPO-FAILED\n", topos[ti].name);
            continue;
        }
        run(topos[ti].name, t, "encode", NULL, 0);
        run(topos[ti].name, t, "decode", f1, 1);
        if (topos[ti].npar >= 2)
            run(topos[ti].name, t, "recon-data", f2, 2);
        /* recon-parity: rebuild all parities */
        int *fp = malloc((size_t)topos[ti].npar * sizeof(int));
        for (int i = 0; i < topos[ti].npar; i++)
            fp[i] = topos[ti].ndata + i;
        run(topos[ti].name, t, "recon-parity", fp, topos[ti].npar);
        free(fp);
        rj_topo_free(t);
    }
    printf("\n%d/%d configs picked non-pow2 U (3, 5 or 6)\n", n_nonpow2, n_total);
    return 0;
}
