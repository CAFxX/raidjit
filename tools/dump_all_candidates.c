/* Dump every buildable candidate's machine code for offline review.
 * Usage: dump_all_candidates <isa> <topo> <op> <outdir>
 *   isa:  x86_64 | aarch64 | riscv64   (synthetic CPU, benchmark=0)
 *   topo: raid5 | raid6 | rs64 | custom64
 *   op:   encode | recover
 * For each candidate from plan_candidates: build it, write
 * <outdir>/<idx>_<sanitized-desc>.bin (raw code bytes) and print a
 * manifest line: idx, desc, code_size, U, lmul, build_rc.
 * Caller disassembles the .bin files with the matching objdump.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>

#include "../include/raidjit.h"
#include "../src/plan.h"
#include "../src/jit.h"
#include "../src/cpu.h"

/* Stub for rj_cpu_detect (cpu.c is arch-guarded; we use synthetic CPUs). */
void rj_cpu_detect(rj_cpu_info *cpu)
{
    memset(cpu, 0, sizeof(*cpu));
}

static void make_cpu(rj_cpu_info *cpu, const char *isa)
{
    memset(cpu, 0, sizeof(*cpu));
    if (strcmp(isa, "x86_64") == 0) {
        cpu->sse2 = cpu->ssse3 = cpu->avx = cpu->avx2 = 1;
        cpu->avx512f = cpu->avx512bw = cpu->avx512vl = 1;
        cpu->gfni = cpu->vbmi = 1;
        snprintf(cpu->brand, sizeof(cpu->brand), "synthetic-x86_64");
    } else if (strcmp(isa, "aarch64") == 0) {
        cpu->neon = cpu->sve = cpu->sve2 = 1;
        cpu->sve_vl = 32;
        snprintf(cpu->brand, sizeof(cpu->brand), "synthetic-aarch64");
    } else if (strcmp(isa, "riscv64") == 0) {
        cpu->rvv = 1;
        cpu->rvv_vlen = 16;
        cpu->rvv_zvbc = 1;
        cpu->rvv_zicbop = 1;
        snprintf(cpu->brand, sizeof(cpu->brand), "synthetic-riscv64");
    } else {
        fprintf(stderr, "unknown isa %s\n", isa);
        exit(1);
    }
}

static void sanitize(char *dst, size_t n, const char *src)
{
    size_t j = 0;
    for (size_t i = 0; src[i] && j + 1 < n; i++) {
        char c = src[i];
        dst[j++] = (isalnum((unsigned char)c) || c == '-') ? c : '_';
    }
    dst[j] = '\0';
}

int main(int argc, char **argv)
{
    if (argc != 5 && argc != 7) {
        fprintf(stderr, "usage: %s <x86_64|aarch64|riscv64> <raid5|raid6|rs64|custom64> <encode|recover> <outdir> [allow_nt prefetch_dist]\n",
                argv[0]);
        return 1;
    }
    const char *isa = argv[1], *topo = argv[2], *opname = argv[3];
    const char *outdir = argv[4];
    int allow_nt = (argc == 7) ? atoi(argv[5]) : 0;
    size_t prefetch_dist = (argc == 7) ? (size_t)atol(argv[6]) : 0;

    /* Exclusive staging lock (same as gen_kernels): two invocations
     * sharing an outdir corrupt each other's dumps. */
    char lock_path[1024];
    snprintf(lock_path, sizeof(lock_path), "%s/.lock", outdir);
    int lock_fd = open(lock_path, O_RDWR | O_CREAT, 0644);
    if (lock_fd < 0) {
        fprintf(stderr, "cannot open lock file %s\n", lock_path);
        return 1;
    }
    if (flock(lock_fd, LOCK_EX | LOCK_NB) != 0) {
        fprintf(stderr, "%s is locked by another run; use a unique outdir\n",
                outdir);
        return 1;
    }

    rj_cpu_info cpu;
    make_cpu(&cpu, isa);

    rj_topology *rt = NULL;
    if (strcmp(topo, "raid5") == 0)
        rt = rj_topo_raid5(4);
    else if (strcmp(topo, "raid6") == 0)
        rt = rj_topo_raid6(5);
    else if (strcmp(topo, "rs64") == 0)
        rt = rj_topo_rs(6, 3);
    else if (strcmp(topo, "custom64") == 0) {
        static const uint8_t m[24] = {
            1,1,1,1,1,1,
            1,1,1,0,0,1,
            1,0,0,1,1,0,
            0,1,0,0,1,1,
        };
        rt = rj_topo_custom(6, 4, m);
    } else if (strcmp(topo, "mulacc") == 0) {
        /* Non-geometric matrix to trigger P_MULACC */
        static const uint8_t m[12] = {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,
            10, 11, 12,
        };
        rt = rj_topo_custom(3, 4, m);
    } else {
        fprintf(stderr, "unknown topo %s\n", topo);
        return 1;
    }
    if (!rt) {
        fprintf(stderr, "topo alloc failed\n");
        return 1;
    }

    rj_op op;
    if (strcmp(opname, "recover") == 0) {
        static const int failed[] = {0};
        op = rj_op_recover(failed, 1);
    } else if (strcmp(opname, "pdelta") == 0) {
        static const int changed[] = {0, 1};
        op = rj_op_parity_update(changed, 2);
    } else {
        op = rj_op_encode();
    }

    topo_int_t ti;
    plan_t pl;
    char err[256];
    if (topo_int_build(&ti, rt, err, sizeof(err)) != 0) {
        fprintf(stderr, "topo_int_build failed: %s\n", err);
        return 1;
    }
    if (plan_build(&ti, &op, "dump", &pl, err, sizeof(err)) != 0) {
        fprintf(stderr, "plan_build failed: %s\n", err);
        return 1;
    }

    cand_t cands[512];
    int ncand = plan_candidates(&pl, &cpu, cands, 512, allow_nt,
                                prefetch_dist);
    printf("# candidates enumerated: %d\n", ncand);

    rj_options o;
    rj_options_default(&o);
    o.benchmark = 0;

    int built = 0;
    for (int i = 0; i < ncand; i++) {
        /* Build the base candidate */
        version_t v;
        memset(&v, 0, sizeof(v));
        char e2[256];
        int rc = jit_build_version(&pl, &cands[i], &cpu, &o, &v, e2, sizeof(e2));
        if (rc != 0) {
            printf("# skip %d: build failed: %s\n", i, e2);
            continue;
        }
        char sd[128];
        sanitize(sd, sizeof(sd), v.desc);
        char path[512];
        snprintf(path, sizeof(path), "%s/%02d_%s.bin", outdir, i, sd);
        FILE *f = fopen(path, "wb");
        if (!f) {
            fprintf(stderr, "cannot write %s\n", path);
            jit_free_version(&v);
            continue;
        }
        fwrite(v.fn, 1, v.code_size, f);
        fclose(f);
        printf("%02d desc=\"%s\" code_size=%zu U=%d lmul=%d file=%s\n",
               i, v.desc, v.code_size, cands[i].U, cands[i].lmul, path);
        jit_free_version(&v);
        built++;

        /* Build twin variants (notail, predtail, maskedtail, permi2b) */
        cand_t twins[4];
        int ntwin = 0;
        /* notail: always applicable */
        twins[ntwin] = cands[i];
        twins[ntwin].notail = 1;
        ntwin++;
        /* predtail: SVE only */
        if (cands[i].isa == ISA_SVE) {
            twins[ntwin] = cands[i];
            twins[ntwin].predtail = 1;
            ntwin++;
        }
        /* maskedtail: x86 AVX512 only, P_XOR without xorsched/ternlog
         * (matches gen_tail_masked's assertion; api.c sets this, but we
         * can test) */
        if (cands[i].isa == ISA_AVX512 && pl.kind == P_XOR &&
            !cands[i].xorsched && !cands[i].ternlog) {
            twins[ntwin] = cands[i];
            twins[ntwin].maskedtail = 1;
            ntwin++;
        }
        /* permi2b: x86 AVX512 with VBMI, P_MULACC/P_DELTA nibble only
         * (matches jit_build_version's assertion) */
        if (cands[i].isa == ISA_AVX512 && cpu.vbmi &&
            cands[i].mul == MUL_NIBBLE &&
            (pl.kind == P_MULACC || pl.kind == P_DELTA)) {
            twins[ntwin] = cands[i];
            twins[ntwin].permi2b = 1;
            ntwin++;
        }
        for (int t = 0; t < ntwin; t++) {
            version_t vt;
            memset(&vt, 0, sizeof(vt));
            char et[256];
            int rct = jit_build_version(&pl, &twins[t], &cpu, &o, &vt,
                                        et, sizeof(et));
            if (rct != 0) {
                printf("# skip %d twin %d: %s\n", i, t, et);
                continue;
            }
            char sdt[128];
            sanitize(sdt, sizeof(sdt), vt.desc);
            char patht[512];
            snprintf(patht, sizeof(patht), "%s/%02d_t%d_%s.bin",
                     outdir, i, t, sdt);
            FILE *ft = fopen(patht, "wb");
            if (ft) {
                fwrite(vt.fn, 1, vt.code_size, ft);
                fclose(ft);
                printf("%02d_t%d desc=\"%s\" code_size=%zu\n",
                       i, t, vt.desc, vt.code_size);
                built++;
            }
            jit_free_version(&vt);
        }
    }
    printf("# built: %d\n", built);
    plan_free(&pl);
    topo_int_free(&ti);
    return 0;
}
