/* REDUX crc_fold kernel smoke test.
 *
 * 1. Gate closed (SKYLAKEX override -- deterministic on ANY host): the
 *    fold gate opens only on the SPR uarch bucket (src/fused.c), so a
 *    host in that bucket (e.g. CI's Xeon 8573C, Emerald Rapids) must not
 *    be able to fail this section. plan_candidates with want_crc=1
 *    returns zero candidates; jit_build_version with hand-set crc_fold=1
 *    fails cleanly.
 * 2. Gate open (SPR override): plan_candidates yields crc_fold=1
 *    candidates; jit_build_version succeeds; calling the kernel once
 *    through the 4-arg entry does not crash.
 *
 * Full property tests (fold state vs scalar model, multi-length) come
 * later; this proves the kernel builds, runs, and respects the gate.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/plan.h"
#include "../src/jit.h"
#include "../src/fused.h"
#include "../src/cpu.h"
#include "../include/raidjit.h"

static int fails = 0;
#define CHECK(cond, msg)                          \
    do {                                          \
        if (!(cond)) {                            \
            printf("FAIL %s\n", msg);             \
            fails++;                              \
        }                                         \
    } while (0)

int main(void)
{
#if !defined(__x86_64__)
    printf("SKIP: x86-64 only\n");
    return 0;
#else
    setvbuf(stdout, NULL, _IONBF, 0);
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    if (!cpu.avx512f || !cpu.pclmulqdq) {
        printf("SKIP: host lacks avx512f/pclmulqdq\n");
        return 0;
    }
    rj_uarch_t host_uarch = rj_uarch_current();

    /* ---- 1. gate closed ---- */
    {
        plan_t pl;
        rj_topology *topo = rj_topo_raid5(4);
        topo_int_t ti;
        char err[256];
        rj_op op = rj_op_encode();
        cand_t *cands = malloc(512 * sizeof(cand_t));
        int n, nfold, i;

        /* Force the gate closed deterministically: the gate opens only
         * for the SPR uarch bucket, so on a host in that bucket (e.g. an
         * Emerald Rapids CI machine) the unfixed section failed its
         * nfold==0 check. SKYLAKEX keeps the gate closed on ANY host,
         * exactly mirroring how section 2 forces it open. */
        rj_uarch_override(RJ_UARCH_INTEL_SKYLAKEX);
        memset(&pl, 0, sizeof(pl));
        topo_int_build(&ti, topo, err, sizeof(err));
        plan_build(&ti, &op, "raid5-4", &pl, err, sizeof(err));
        pl.want_crc = 1;
        n = plan_candidates(&pl, &cpu, cands, 512, 0, 0);
        nfold = 0;
        for (i = 0; i < n; i++)
            if (cands[i].crc_fold)
                nfold++;
        CHECK(nfold == 0, "fold candidate with gate closed");
        printf("gate closed: %d candidates, %d fold\n", n, nfold);
        if (nfold == 0 && n > 0) {
            /* Backend gate: hand-set crc_fold=1; must fail cleanly. */
            cand_t *plain = NULL;
            for (i = 0; i < n; i++)
                if (cands[i].isa == ISA_AVX512) {
                    plain = &cands[i];
                    break;
                }
            if (plain) {
                cand_t fc = *plain;
                version_t v;
                rj_options opts;
                char berr[256] = { 0 };
                fc.crc_fold = 1;
                memset(&v, 0, sizeof(v));
                memset(&opts, 0, sizeof(opts));
                CHECK(jit_build_version(&pl, &fc, &cpu, &opts, &v, berr,
                                        sizeof(berr)) != 0,
                      "backend built crc_fold=1, gate closed");
                printf("backend reject: %s\n", berr);
            }
        }
        plan_free(&pl);
        topo_int_free(&ti);
        rj_topo_free(topo);
        free(cands);
    }

    /* ---- 2. gate open (SPR): build + single call ---- */
    {
        plan_t pl;
        rj_topology *topo = rj_topo_raid5(4);
        topo_int_t ti;
        char err[256];
        rj_op op = rj_op_encode();
        cand_t *cands = malloc(512 * sizeof(cand_t));
        int n, i;
        cand_t *fold = NULL;

        rj_uarch_override(RJ_UARCH_INTEL_SAPPHIRERAPIDS);
        memset(&pl, 0, sizeof(pl));
        topo_int_build(&ti, topo, err, sizeof(err));
        plan_build(&ti, &op, "raid5-4", &pl, err, sizeof(err));
        pl.want_crc = 1;
        n = plan_candidates(&pl, &cpu, cands, 512, 0, 0);
        for (i = 0; i < n; i++)
            if (cands[i].crc_fold && cands[i].isa == ISA_AVX512) {
                fold = &cands[i];
                break;
            }
        CHECK(fold != NULL, "no fold candidate (SPR)");
        printf("gate open: %d candidates, fold %s\n", n,
               fold ? "found" : "MISSING");
        if (fold) {
            version_t v;
            rj_options opts;
            char berr[256] = { 0 };
            memset(&v, 0, sizeof(v));
            memset(&opts, 0, sizeof(opts));
            CHECK(jit_build_version(&pl, fold, &cpu, &opts, &v, berr,
                                    sizeof(berr)) == 0,
                  "build failed (SPR)");
            if (v.fn) {
                /* Single call through the 4-arg entry. Buffers must be
                 * 16-byte aligned (the kernel uses VMOVDQA). */
                int n_in = pl.n_in, n_out = pl.n_out;
                size_t len = 1000;
                uint8_t *ib[4], *ob;
                uint8_t fs[8 * 16] __attribute__((aligned(64)));
                const void *in[4];
                void *out[1];
                int s, i;
                printf("n_in=%d n_out=%d\n", n_in, n_out);
                for (i = 0; i < n_in; i++) {
                    if (posix_memalign((void **)&ib[i], 64, len))
                        return 1;
                    memset(ib[i], i + 1, len);
                    in[i] = ib[i];
                }
                if (posix_memalign((void **)&ob, 64, len))
                    return 1;
                memset(ob, 0xA5, len);
                out[0] = ob;
                memset(fs, 0, sizeof(fs));
                for (s = 0; s < n_in + n_out; s++) {
                    uint32_t init = 0xFFFFFFFFu;
                    memcpy(fs + (size_t)s * 16, &init, sizeof(init));
                }
                printf("calling kernel (len=%zu)...\n", len);
                ((rj_fn_crc_fold)(void *)v.fn)(in, out, len, fs);
                printf("kernel returned without crashing\n");
                for (i = 0; i < n_in; i++)
                    free(ib[i]);
                free(ob);
            } else {
                printf("build error: %s\n", berr);
            }
        }
        plan_free(&pl);
        topo_int_free(&ti);
        rj_topo_free(topo);
        free(cands);
        rj_uarch_override(host_uarch);
    }

    if (!fails)
        printf("crc_fold smoke: all OK\n");
    return fails != 0;
#endif
}
