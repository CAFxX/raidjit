/* tools/sweep_dump.c — dump generated machine code for a codegen review
 * sweep. REVIEW TOOL ONLY: emits bytes, never executes them.
 *
 * Usage:
 *   sweep_dump <cpu-spec> <outdir> [--nt] [--prefetch=N]
 *
 * cpu-spec (must match the ARCH this binary was built for):
 *   x86_64 : sse2 | ssse3 | avx | avx2 | avx512 | avx512-gfni
 *   aarch64: neon | sve-128 | sve-256 | sve2-128 | sve2-256
 *   riscv64: rvv-128 | rvv-256 | rvv-512
 *
 * For each (topology, op) cell in the built-in matrix and each candidate
 * from plan_candidates(), builds the version with jit_build_version()
 * (benchmarks disabled) and writes:
 *   <outdir>/<cell>__<cand>.code.bin   raw code bytes [0, code_size)
 * and appends one JSON object per line to <outdir>/manifest.jsonl.
 *
 * --nt: also enumerate non-temporal store twins (kept: all candidates,
 *   flags recorded in manifest).
 * --prefetch=N: also enumerate prefetch twins N bytes ahead.
 *
 * File name encoding:
 *   <topo>__<op>__<isa>[-gfni|-nib][-tlog][-remat][-nt][-pfN]-U<U>.code.bin
 *
 * Private build dirs only; never touches build-x86_64/-aarch64/-riscv64.
 */
#include "../include/raidjit.h"
#include "../src/plan.h"
#include "../src/jit.h"
#include "../src/cpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>
#include <errno.h>

static void die(const char *msg)
{
    fprintf(stderr, "sweep_dump: FATAL: %s\n", msg);
    exit(1);
}

/* ---------------- synthetic CPU ---------------- */

static int fill_cpu(const char *spec, rj_cpu_info *cpu)
{
    memset(cpu, 0, sizeof(*cpu));
    snprintf(cpu->brand, sizeof(cpu->brand), "synthetic-%s", spec);
#if defined(__x86_64__)
    cpu->sse2 = cpu->ssse3 = 1; /* baseline for all x86 specs here */
    if (!strcmp(spec, "sse2")) { cpu->ssse3 = 0; }
    else if (!strcmp(spec, "ssse3")) { }
    else if (!strcmp(spec, "avx")) { cpu->avx = 1; }
    else if (!strcmp(spec, "avx2")) { cpu->avx = cpu->avx2 = 1; }
    else if (!strcmp(spec, "avx512")) {
        cpu->avx = cpu->avx2 = 1;
        cpu->avx512f = cpu->avx512bw = cpu->avx512vl = 1;
    } else if (!strcmp(spec, "avx512-gfni")) {
        cpu->avx = cpu->avx2 = 1;
        cpu->avx512f = cpu->avx512bw = cpu->avx512vl = 1;
        cpu->gfni = 1;
    } else return -1;
    return 0;
#elif defined(__aarch64__)
    if (!strcmp(spec, "neon")) { cpu->neon = 1; }
    else if (!strcmp(spec, "sve-128")) { cpu->neon = cpu->sve = 1; cpu->sve_vl = 16; }
    else if (!strcmp(spec, "sve-256")) { cpu->neon = cpu->sve = 1; cpu->sve_vl = 32; }
    else if (!strcmp(spec, "sve2-128")) { cpu->neon = cpu->sve = cpu->sve2 = 1; cpu->sve_vl = 16; }
    else if (!strcmp(spec, "sve2-256")) { cpu->neon = cpu->sve = cpu->sve2 = 1; cpu->sve_vl = 32; }
    else return -1;
    return 0;
#elif defined(__riscv)
    if (!strcmp(spec, "rvv-128")) { cpu->rvv = 1; cpu->rvv_vlen = 16; }
    else if (!strcmp(spec, "rvv-256")) { cpu->rvv = 1; cpu->rvv_vlen = 32; }
    else if (!strcmp(spec, "rvv-512")) { cpu->rvv = 1; cpu->rvv_vlen = 64; }
    else return -1;
    return 0;
#else
    (void)spec;
    return -1;
#endif
}

/* ---------------- matrix ---------------- */

typedef struct {
    const char *name; /* short file-safe name */
    rj_topology *(*mk)(void);
} topo_cell_t;

static rj_topology *mk_raid5_4(void) { return rj_topo_raid5(4); }
static rj_topology *mk_raid5_8(void) { return rj_topo_raid5(8); }
static rj_topology *mk_raid6_5(void) { return rj_topo_raid6(5); }
static rj_topology *mk_raid6_8(void) { return rj_topo_raid6(8); }
static rj_topology *mk_rs12_3(void) { return rj_topo_rs(12, 3); }
/* 2+2 all-ones custom matrix -> P_XOR with 2 rows (multi-row XOR path). */
static rj_topology *mk_xor22(void)
{
    static const uint8_t m[4] = { 1, 1, 1, 1 };
    return rj_topo_custom(2, 2, m);
}
/* 4+2 non-Horner custom matrix -> P_MULACC with 2 rows. */
static rj_topology *mk_mulacc42(void)
{
    static const uint8_t m[8] = { 1, 1, 1, 1, 1, 2, 4, 8 };
    return rj_topo_custom(4, 2, m);
}
/* 6+3 non-Horner custom matrix -> P_MULACC with 3 rows. */
static rj_topology *mk_mulacc63(void)
{
    static const uint8_t m[18] = {
        1, 1, 1, 1, 1, 1,
        1, 2, 3, 4, 5, 6,
        2, 1, 4, 3, 6, 5,
    };
    return rj_topo_custom(6, 3, m);
}

static const topo_cell_t TOPOS[] = {
    { "raid5-4", mk_raid5_4 }, { "raid5-8", mk_raid5_8 },
    { "raid6-5", mk_raid6_5 }, { "raid6-8", mk_raid6_8 },
    { "rs12-3", mk_rs12_3 },   { "xor22", mk_xor22 },
    { "mulacc42", mk_mulacc42 }, { "mulacc63", mk_mulacc63 },
};

typedef struct {
    const char *name;
    rj_op (*mk)(int ndata, int npar);
} op_cell_t;

static rj_op mk_enc(int ndata, int npar) { (void)ndata; (void)npar; return rj_op_encode(); }
static rj_op mk_rec1(int ndata, int npar)
{
    (void)ndata; (void)npar;
    int f[] = { 0 };
    return rj_op_recover(f, 1);
}
static rj_op mk_rec2(int ndata, int npar)
{
    (void)ndata; (void)npar;
    int f[] = { 0, 1 };
    return rj_op_recover(f, 2);
}
static rj_op mk_pu1(int ndata, int npar)
{
    (void)ndata; (void)npar;
    int c[] = { 0 };
    return rj_op_parity_update(c, 1);
}
/* parity update touching every data disk (largest legal nchanged). */
static rj_op mk_pufull(int ndata, int npar)
{
    (void)npar;
    int c[64];
    for (int i = 0; i < ndata && i < 64; i++)
        c[i] = i;
    return rj_op_parity_update(c, ndata < 64 ? ndata : 64);
}

static const op_cell_t OPS[] = {
    { "enc", mk_enc }, { "rec1", mk_rec1 }, { "rec2", mk_rec2 },
    { "pu1", mk_pu1 }, { "pufull", mk_pufull },
};

/* ---------------- main ---------------- */

static const char *kind_name(plan_kind_t k)
{
    switch (k) {
    case P_XOR: return "xor";
    case P_MULACC: return "mulacc";
    case P_HORNER: return "horner";
    case P_DELTA: return "delta";
    default: return "?";
    }
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr,
                "usage: %s <cpu-spec> <outdir> [--nt] [--prefetch=N]\n",
                argv[0]);
        return 1;
    }
    const char *spec = argv[1];
    const char *outdir = argv[2];
    int allow_nt = 0;
    size_t prefetch = 0;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--nt"))
            allow_nt = 1;
        else if (!strncmp(argv[i], "--prefetch=", 11))
            prefetch = (size_t)strtoul(argv[i] + 11, NULL, 0);
        else
            die("unknown flag");
    }
    if (mkdir(outdir, 0755) && errno != EEXIST)
        die("mkdir outdir failed");

    rj_cpu_info cpu;
    if (fill_cpu(spec, &cpu))
        die("cpu-spec not valid for this build arch");

    char mfpath[1024];
    snprintf(mfpath, sizeof(mfpath), "%s/manifest.jsonl", outdir);
    FILE *mf = fopen(mfpath, "w");
    if (!mf)
        die("cannot open manifest");

    rj_options o;
    rj_options_default(&o);
    o.benchmark = 0;

    long total = 0, failed = 0;
    for (size_t ti = 0; ti < sizeof(TOPOS) / sizeof(TOPOS[0]); ti++) {
        rj_topology *t = TOPOS[ti].mk();
        if (!t) {
            fprintf(stderr, "topo %s: constructor returned NULL (skip)\n",
                    TOPOS[ti].name);
            continue;
        }
        int ndata = rj_topo_ndata(t), npar = rj_topo_npar(t);
        for (size_t oi = 0; oi < sizeof(OPS) / sizeof(OPS[0]); oi++) {
            /* rec2 needs npar >= 2 worth of redundancy; skip nonsense. */
            if (!strcmp(OPS[oi].name, "rec2") && npar < 2)
                continue;
            rj_op op = OPS[oi].mk(ndata, npar);
            if (op.oom) {
                fprintf(stderr, "op %s on %s: oom (skip)\n", OPS[oi].name,
                        TOPOS[ti].name);
                continue;
            }
            topo_int_t tii;
            char err[512];
            if (topo_int_build(&tii, t, err, sizeof(err))) {
                fprintf(stderr, "topo_int_build %s: %s (skip)\n",
                        TOPOS[ti].name, err);
                rj_op_free(&op);
                continue;
            }
            char opname[96];
            snprintf(opname, sizeof(opname), "%s %s", TOPOS[ti].name,
                     OPS[oi].name);
            plan_t pl;
            if (plan_build(&tii, &op, opname, &pl, err, sizeof(err))) {
                fprintf(stderr, "plan_build %s: %s (skip)\n", opname, err);
                topo_int_free(&tii);
                rj_op_free(&op);
                continue;
            }
            cand_t cands[128];
            int nc = plan_candidates(&pl, &cpu, cands, 128, allow_nt,
                                     prefetch);
            for (int ci = 0; ci < nc; ci++) {
                version_t v;
                if (jit_build_version(&pl, &cands[ci], &cpu, &o, &v, err,
                                      sizeof(err))) {
                    fprintf(stderr, "build failed %s cand %d: %s\n",
                            opname, ci, err);
                    failed++;
                    continue;
                }
                char cname[160];
                snprintf(cname, sizeof(cname), "%s__%s__%s%s%s%s%s%s-U%d",
                         TOPOS[ti].name, OPS[oi].name,
                         isa_name(cands[ci].isa),
                         cands[ci].mul == MUL_GFNI ? "-gfni" : "-nib",
                         cands[ci].ternlog ? "-tlog" : "",
                         cands[ci].remat ? "-remat" : "",
                         cands[ci].nt ? "-nt" : "",
                         cands[ci].prefetch_dist ? "-pf" : "",
                         cands[ci].U);
                char path[1100];
                snprintf(path, sizeof(path), "%s/%s.code.bin", outdir,
                         cname);
                FILE *f = fopen(path, "wb");
                if (!f)
                    die("cannot open output file");
                if (v.code_size &&
                    fwrite(v.mem, 1, v.code_size, f) != v.code_size)
                    die("short write");
                fclose(f);
                /* JSON-escape desc minimally (no quotes expected). */
                fprintf(mf,
                        "{\"name\":\"%s\",\"topo\":\"%s\",\"op\":\"%s\","
                        "\"cpu\":\"%s\",\"isa\":\"%s\",\"mul\":\"%s\","
                        "\"U\":%d,\"nt\":%d,\"prefetch\":%zu,\"remat\":%d,"
                        "\"ternlog\":%d,\"kind\":\"%s\",\"code_size\":%zu,"
                        "\"const_size\":%zu,\"desc\":\"%s\","
                        "\"uops_per_byte\":%.4f,\"mem_per_byte\":%.4f}\n",
                        cname, TOPOS[ti].name, OPS[oi].name, spec,
                        isa_name(cands[ci].isa),
                        cands[ci].mul == MUL_GFNI ? "gfni" : "nibble",
                        cands[ci].U, cands[ci].nt,
                        cands[ci].prefetch_dist, cands[ci].remat,
                        cands[ci].ternlog, kind_name(pl.kind), v.code_size,
                        v.const_size, v.desc, v.uops_per_byte,
                        v.mem_per_byte);
                jit_free_version(&v);
                total++;
            }
            plan_free(&pl);
            topo_int_free(&tii);
            rj_op_free(&op);
        }
        rj_topo_free(t);
    }
    fclose(mf);
    printf("sweep_dump: %ld versions dumped, %ld build failures -> %s\n",
           total, failed, outdir);
    return failed ? 1 : 0;
}
