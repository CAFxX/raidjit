/* Dump JITed kernels for specific topologies and ISAs.
 * Usage: dump_kernel <topo> <isa> <output.bin>
 *   topo: raid5-4 (4+1), raid6-5 (5+2)
 *   isa: avx512-gfni, rvv, sve2
 * Dumps the raw machine code bytes to output.bin and prints the
 * candidate description and code size to stdout.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/plan.h"
#include "../src/jit.h"
#include "../include/raidjit.h"

static void make_cpu(rj_cpu_info *cpu, const char *isa)
{
    memset(cpu, 0, sizeof(*cpu));
    if (strcmp(isa, "avx512-gfni") == 0) {
        cpu->sse2 = cpu->ssse3 = cpu->avx = cpu->avx2 = 1;
        cpu->avx512f = cpu->avx512bw = cpu->avx512vl = cpu->gfni = 1;
        snprintf(cpu->brand, sizeof(cpu->brand), "synthetic-avx512-gfni");
    } else if (strcmp(isa, "rvv") == 0) {
        cpu->rvv = 1;
        cpu->rvv_vlen = 16; /* 128-bit VLEN */
        snprintf(cpu->brand, sizeof(cpu->brand), "synthetic-rvv");
    } else if (strcmp(isa, "sve2") == 0) {
        cpu->neon = cpu->sve = cpu->sve2 = 1;
        cpu->sve_vl = 32; /* 256-bit SVE */
        snprintf(cpu->brand, sizeof(cpu->brand), "synthetic-sve2");
    } else {
        fprintf(stderr, "unknown isa: %s\n", isa);
        exit(1);
    }
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "usage: %s <raid5-4|raid6-5> <avx512-gfni|rvv|sve2> <output.bin>\n", argv[0]);
        return 1;
    }

    const char *topo_name = argv[1];
    const char *isa_arg = argv[2];
    const char *out_path = argv[3];

    rj_topology *topo = NULL;
    rj_op op;
    if (strcmp(topo_name, "raid5-4") == 0) {
        topo = rj_topo_raid5(4);
        op = rj_op_encode();
    } else if (strcmp(topo_name, "raid6-5") == 0) {
        topo = rj_topo_raid6(5);
        op = rj_op_encode();
    } else {
        fprintf(stderr, "unknown topo: %s\n", topo_name);
        return 1;
    }
    if (!topo) {
        fprintf(stderr, "topo creation failed\n");
        return 1;
    }

    rj_cpu_info cpu;
    make_cpu(&cpu, isa_arg);

    /* Build plan using internal API */
    char err[256];
    /* We need topo_int_t; use the public compile path but with benchmark=0
     * and capture via internal hooks. Simpler: use plan_build directly.
     * For this we need to convert rj_topology to topo_int_t. */
    
    /* Actually, easiest: use rj_compile with benchmark=0, then extract
     * the version via rj_get_info or similar. But we need the raw code.
     * Let's use the internal plan API directly. */
    
    /* Get the internal topology representation */
    topo_int_t ti;
    if (topo_int_build(&ti, topo, err, sizeof(err)) != 0) {
        fprintf(stderr, "topo_int_build failed: %s\n", err);
        return 1;
    }

    plan_t pl;
    memset(&pl, 0, sizeof(pl));
    if (plan_build(&ti, &op, topo_name, &pl, err, sizeof(err)) != 0) {
        fprintf(stderr, "plan_build failed: %s\n", err);
        return 1;
    }

    cand_t cands[64];
    int ncand = plan_candidates(&pl, &cpu, cands, 64, 0, 0);
    if (ncand <= 0) {
        fprintf(stderr, "no candidates\n");
        return 1;
    }

    /* Find the best candidate matching the requested ISA.
     * For avx512-gfni: prefer GFNI multiply method.
     * For rvv/sve2: there's only one multiply method. */
    cand_t *best = NULL;
    for (int i = 0; i < ncand; i++) {
        if (strcmp(isa_arg, "avx512-gfni") == 0) {
            if (cands[i].isa == ISA_AVX512 && cands[i].mul == MUL_GFNI) {
                best = &cands[i];
                break;
            }
        } else if (strcmp(isa_arg, "rvv") == 0) {
            if (cands[i].isa == ISA_RVV) {
                best = &cands[i];
                break;
            }
        } else if (strcmp(isa_arg, "sve2") == 0) {
            if (cands[i].isa == ISA_SVE) {
                best = &cands[i];
                break;
            }
        }
    }
    if (!best) {
        /* Fall back to first candidate */
        best = &cands[0];
        fprintf(stderr, "warning: no exact ISA match, using %s\n", isa_name(cands[0].isa));
    }

    rj_options opts;
    rj_options_default(&opts);
    opts.benchmark = 0;

    version_t v;
    memset(&v, 0, sizeof(v));
    if (jit_build_version(&pl, best, &cpu, &opts, &v, err, sizeof(err)) != 0) {
        fprintf(stderr, "jit_build_version failed: %s\n", err);
        return 1;
    }

    printf("topo: %s\n", topo_name);
    printf("isa: %s\n", isa_arg);
    printf("desc: %s\n", v.desc);
    printf("code_size: %zu\n", v.code_size);
    printf("const_size: %zu\n", v.const_size);
    printf("total: %zu\n", v.code_size + v.const_size);

    /* Dump code + const pool to file */
    FILE *f = fopen(out_path, "wb");
    if (!f) {
        perror("fopen");
        return 1;
    }
    /* The mem layout is: code followed by const pool.
     * We need to find where the code ends and const begins.
     * For simplicity, dump the entire mem region up to code_size + const_size.
     * Actually, version_t doesn't expose the split directly.
     * Let's dump code_size bytes from mem start. */
    fwrite(v.mem, 1, v.code_size, f);
    fclose(f);
    printf("wrote %zu bytes to %s\n", v.code_size, out_path);

    jit_free_version(&v);
    plan_free(&pl);
    topo_int_free(&ti);
    rj_op_free(&op);
    rj_topo_free(topo);
    return 0;
}
