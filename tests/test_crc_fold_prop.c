/* REDUX crc_fold property test (spec §8.2 kernel differential).
 *
 * For random topologies (RAID-5, RAID-6, RS(k,m)), random lengths (all
 * residues mod 16, 1..65536 plus a few large), random data:
 *   - compile fused [ENCODE, CRC->0] (benchmark=0)
 *   - run through the driver; compare each stream's CRC vs the
 *     rj_crc32_update-chained reference AND (for small lengths) an
 *     independent bitwise oracle over the exact buffers
 *   - compare parity bytes vs a separately-compiled plain rj_compile
 *     encode (the fold must not perturb encode output)
 *
 * argv[1]: "open"   -> force the uarch gate open via rj_uarch_override(SPR);
 *                     assert the fold path is actually used
 *                     (rj_jit_handle_crc_fold == 1 on a direct
 *                     rj_compile_want_crc of the same op)
 *          "closed" -> host gate (closed on non-SPR CPUs): scalar fallback;
 *                     assert rj_compile_want_crc fails closed and CRCs are
 *                     still correct.
 *
 * argv[2]: optional PRNG seed (default 0xC0FFEE).
 *
 * The override has no reset, so open/closed are separate processes.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "../include/raidjit.h"
#include "../include/raidjit_fused.h"
#include "../src/cpu.h"   /* rj_uarch_override (test hook) */
#include "../src/api.h"   /* rj_compile_want_crc, rj_jit_handle_crc_fold */

static int fails = 0;
static long ncases = 0, nfold_used = 0, nstreams_checked = 0;
#define CHECK(cond, ...)                                                  \
    do {                                                                  \
        if (!(cond)) {                                                    \
            printf("FAIL %s:%d: ", __func__, __LINE__);                    \
            printf(__VA_ARGS__);                                          \
            printf("\n");                                                 \
            fails++;                                                      \
        }                                                                 \
    } while (0)

static uint64_t rng_state;
static uint64_t xrng(void)
{
    uint64_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return rng_state = x;
}

static void *abuf(size_t n)
{
    void *p = NULL;
    if (posix_memalign(&p, 64, n ? n : 64)) {
        fprintf(stderr, "oom\n");
        exit(1);
    }
    return p;
}

static void randfill(void *p, size_t n)
{
    uint8_t *b = p;
    for (size_t i = 0; i < n; i++)
        b[i] = (uint8_t)xrng();
}

/* Independent bitwise CRC-32/ISO-HDLC (shares no code with src/crc*.c). */
static uint32_t crc32_bitwise(const uint8_t *p, size_t n)
{
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; i++) {
        crc ^= p[i];
        for (int k = 0; k < 8; k++)
            crc = (crc & 1) ? (crc >> 1) ^ 0xEDB88320u : crc >> 1;
    }
    return crc ^ 0xFFFFFFFFu;
}

/* rj_crc32_update-chained reference over one buffer. */
static uint32_t crc32_ref(const uint8_t *p, size_t n)
{
    uint32_t c;
    rj_crc32_init(&c);
    c = rj_crc32_update(c, p, n);
    return rj_crc32_final(c);
}

static rj_topology *mk_topo(int cls, int a, int b)
{
    switch (cls) {
    case 0: return rj_topo_raid5(a);      /* a = ndata 2..8 */
    case 1: return rj_topo_raid6_rs(a);   /* a = ndata 4..8 */
    default: return rj_topo_rs(a, b);     /* a = ndata 4..8, b = npar 1..3 */
    }
}

static const char *cls_name(int cls, int a, int b)
{
    static char buf[32];
    if (cls == 0)
        snprintf(buf, sizeof(buf), "raid5(%d)", a);
    else if (cls == 1)
        snprintf(buf, sizeof(buf), "raid6_rs(%d)", a);
    else
        snprintf(buf, sizeof(buf), "rs(%d,%d)", a, b);
    return buf;
}

int main(int argc, char **argv)
{
#if !defined(__x86_64__)
    printf("SKIP: x86-64 only\n");
    return 0;
#else
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc < 2 || (strcmp(argv[1], "open") && strcmp(argv[1], "closed"))) {
        fprintf(stderr, "usage: %s open|closed [seed]\n", argv[0]);
        return 2;
    }
    int gate_open = !strcmp(argv[1], "open");
    rng_state = argc > 2 ? strtoull(argv[2], NULL, 0) : 0xC0FFEEULL;

    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    if (!cpu.avx512f || !cpu.pclmulqdq) {
        printf("SKIP: host lacks avx512f/pclmulqdq\n");
        return 0;
    }
    if (gate_open)
        rj_uarch_override(RJ_UARCH_INTEL_SAPPHIRERAPIDS);

    rj_fuse_options fopts;
    rj_fuse_options_default(&fopts);
    fopts.benchmark = 0;

    rj_options co;
    rj_options_default(&co);
    co.benchmark = 0;

    /* 120 random cases; length classes: all residues mod 16 (explicit),
     * small/medium/large. */
    const int NCASE = 120;
    for (int ci = 0; ci < NCASE; ci++) {
        int cls = (int)(xrng() % 3);
        int nd, np;
        if (cls == 0) {
            nd = 2 + (int)(xrng() % 7);
            np = 1;
        } else if (cls == 1) {
            nd = 4 + (int)(xrng() % 5);
            np = 2;
        } else {
            nd = 4 + (int)(xrng() % 5);
            np = 1 + (int)(xrng() % 3);
        }
        rj_topology *t = mk_topo(cls, nd, np);
        if (!t) {
            printf("FAIL case %d: topo build failed (%s)\n", ci,
                   cls_name(cls, nd, np));
            fails++;
            continue;
        }
        size_t len;
        int lcls = ci % 4;
        if (lcls == 0) {
            /* Explicit residue sweep: base + residue, all 16 classes. */
            size_t base = (size_t)(xrng() % 4096);
            base -= base % 16;
            len = base + (size_t)(ci % 16);
            if (len == 0)
                len = 16; /* keep len>=1 in the residue-0 class */
        } else if (lcls == 1) {
            len = 1 + (size_t)(xrng() % 4096);
        } else if (lcls == 2) {
            len = 4097 + (size_t)(xrng() % 61440);
        } else {
            len = 65537 + (size_t)(xrng() % (1u << 20));
        }

        char err[256];
        rj_fop_t fops[2];
        memset(fops, 0, sizeof(fops));
        fops[0].kind = RJ_FOP_ENCODE;
        fops[1].kind = RJ_FOP_CRC;
        fops[1].target = 0;
        rj_fused *f = rj_fuse_compile(t, fops, 2, &fopts, err, sizeof(err));
        if (!f) {
            printf("FAIL case %d (%s len=%zu): fuse failed: %s\n", ci,
                   cls_name(cls, nd, np), len, err);
            fails++;
            rj_topo_free(t);
            continue;
        }
        rj_fn_fused ffn = rj_get_fn_fused(f);
        if (!ffn) {
            printf("FAIL case %d: fused fn NULL\n", ci);
            fails++;
            rj_free_fused(f);
            rj_topo_free(t);
            continue;
        }

        /* Exercise guarantee: the direct want_crc compile of the same op
         * must fold iff the gate is open. */
        rj_op op = rj_op_encode();
        rj_jit *wu =
            rj_compile_want_crc(t, &op, &co, err, sizeof(err));
        int folded = wu && rj_jit_handle_crc_fold(wu);
        if (gate_open) {
            if (!folded) {
                printf("FAIL case %d (%s len=%zu): gate open but no fold "
                       "handle\n", ci, cls_name(cls, nd, np), len);
                fails++;
            } else {
                nfold_used++;
            }
        } else {
            if (wu != NULL || folded) {
                printf("FAIL case %d: gate closed but want_crc compiled "
                       "fold=%d\n", ci, folded);
                fails++;
            }
        }
        if (wu)
            rj_free(wu);

        /* Reference plain encode (fold must not perturb output bytes). */
        rj_jit *pe = rj_compile(t, &op, &co, err, sizeof(err));
        if (!pe) {
            printf("FAIL case %d: plain encode compile failed: %s\n", ci,
                   err);
            fails++;
            rj_free_fused(f);
            rj_topo_free(t);
            continue;
        }
        rj_fn pfn = rj_get_fn(pe);

        const void **in = malloc(sizeof(*in) * (size_t)nd);
        void **oF = malloc(sizeof(*oF) * (size_t)np);
        void **oP = malloc(sizeof(*oP) * (size_t)np);
        for (int i = 0; i < nd; i++) {
            uint8_t *b = abuf(len);
            randfill(b, len);
            in[i] = b;
        }
        for (int p = 0; p < np; p++) {
            oF[p] = abuf(len);
            oP[p] = abuf(len);
        }
        uint32_t crc[512];
        memset(crc, 0xA5, sizeof(crc));
        ffn(in, oF, len, crc);
        pfn(in, oP, len);

        /* (a) parity bytes identical (fold must not perturb encode). */
        for (int p = 0; p < np; p++)
            CHECK(memcmp(oF[p], oP[p], len) == 0,
                  "case %d (%s len=%zu): parity %d differs vs plain encode",
                  ci, cls_name(cls, nd, np), len, p);
        /* (b) CRCs vs rj_crc32_update-chained reference on exact buffers. */
        for (int i = 0; i < nd; i++) {
            uint32_t want = crc32_ref(in[i], len);
            CHECK(crc[i] == want,
                  "case %d (%s len=%zu): crc_in[%d] got %08x want %08x",
                  ci, cls_name(cls, nd, np), len, i, crc[i], want);
            nstreams_checked++;
        }
        for (int p = 0; p < np; p++) {
            uint32_t want = crc32_ref(oF[p], len);
            CHECK(crc[nd + p] == want,
                  "case %d (%s len=%zu): crc_out[%d] got %08x want %08x",
                  ci, cls_name(cls, nd, np), len, p, crc[nd + p], want);
            nstreams_checked++;
        }
        /* (c) bitwise-oracle spot check on small lengths. */
        if (len <= 4096) {
            for (int i = 0; i < nd; i++)
                CHECK(crc[i] == crc32_bitwise(in[i], len),
                      "case %d (%s len=%zu): oracle crc_in[%d]",
                      ci, cls_name(cls, nd, np), len, i);
            for (int p = 0; p < np; p++)
                CHECK(crc[nd + p] == crc32_bitwise(oF[p], len),
                      "case %d (%s len=%zu): oracle crc_out[%d]",
                      ci, cls_name(cls, nd, np), len, p);
        }

        for (int i = 0; i < nd; i++)
            free((void *)in[i]);
        for (int p = 0; p < np; p++) {
            free(oF[p]);
            free(oP[p]);
        }
        free(in);
        free(oF);
        free(oP);
        rj_free(pe);
        rj_free_fused(f);
        rj_topo_free(t);
        ncases++;
        if ((ci + 1) % 30 == 0)
            printf("... %d/%d cases done\n", ci + 1, NCASE);
    }

    printf("%s: %ld cases, %ld fold handles, %ld stream CRCs checked, "
           "fails=%d\n",
           gate_open ? "gate-open" : "gate-closed", ncases, nfold_used,
           nstreams_checked, fails);
    if (gate_open && nfold_used == 0) {
        printf("FAIL: gate open but fold path never exercised\n");
        fails++;
    }
    if (fails == 0)
        printf("PASS\n");
    return fails != 0;
#endif
}
