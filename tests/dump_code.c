/* Dump generated machine code for inspection. */
#include "../include/raidjit.h"
#include "../src/plan.h"
#include "../src/jit.h"
#include "../src/cpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    printf("cpu: %s\n", cpu.brand);

    rj_topology *t = rj_topo_raid6(8);
    topo_int_t ti;
    char err[256];
    topo_int_build(&ti, t, err, sizeof(err));
    rj_op op = rj_op_encode();
    plan_t pl;
    plan_build(&ti, &op, "raid6(8)", &pl, err, sizeof(err));
    printf("plan kind=%d n_in=%d n_out=%d name=%s\n", pl.kind, pl.n_in,
           pl.n_out, pl.name);

    cand_t cands[32];
    int nc = plan_candidates(&pl, &cpu, cands, 32, 0, 0);
    printf("%d candidates\n", nc);
    int which = argc > 1 ? atoi(argv[1]) : 0;
    if (which >= nc)
        which = 0;

    rj_options o;
    rj_options_default(&o);
    version_t v;
    if (jit_build_version(&pl, &cands[which], &cpu, &o, &v, err, sizeof(err))) {
        printf("build failed: %s\n", err);
        return 1;
    }
    printf("candidate: %s\n", v.desc);
    printf("code_size=%zu const_size=%zu\n", v.code_size, v.const_size);

    FILE *f = fopen("/tmp/raidjit_code.bin", "wb");
    fwrite(v.mem, 1, v.code_size + v.const_size, f);
    fclose(f);
    printf("wrote /tmp/raidjit_code.bin\n");
    return 0;
}
