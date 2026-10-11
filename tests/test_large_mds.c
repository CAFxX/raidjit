/* Large MDS topology test: Cauchy matrix above 64 disks.
 *
 * Uses a Cauchy matrix C[j][i] = inv(x_i ^ y_j) over GF(2^8), which is
 * provably MDS for distinct x_i, y_j with x_i != y_j. This tests the
 * JIT on topologies wider than 64 disks, exercising the register-pressure
 * parity ceilings and the planner's candidate selection at scale.
 *
 * Topology: 100 data + 4 parity = 104 disks (> 64).
 */
#include "../include/raidjit.h"
#include "../src/gf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static int fails;
#define CHECK(c, ...) do {                                             \
    if (!(c)) {                                                        \
        printf("FAIL "); printf(__VA_ARGS__); printf("\n"); fails++;   \
    }                                                                  \
} while (0)

static uint64_t xrng_state = 0x123456789ABCDEF0ULL;
static uint64_t xrng(void)
{
    uint64_t x = xrng_state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return xrng_state = x;
}

static uint8_t *alloc64(size_t n)
{
    uint8_t *raw = NULL;
    if (posix_memalign((void **)&raw, 64, n + 128))
        abort();
    memset(raw, 0xA5, 64);
    memset(raw + 64 + n, 0x5A, 64);
    return raw + 64;
}

/* uses gf_inv from src/gf.h */
int main(void)
{
    const int nd = 100, np = 4;
    /* Cauchy: x_i = i+1, y_j = nd+1+j; all distinct, non-zero, x_i != y_j */
    uint8_t *mat = malloc((size_t)np * nd);
    for (int j = 0; j < np; j++) {
        for (int i = 0; i < nd; i++) {
            uint8_t x = (uint8_t)(i + 1);
            uint8_t y = (uint8_t)(nd + 1 + j);
            mat[j * nd + i] = gf_inv(x ^ y);
        }
    }

    rj_topology *t = rj_topo_custom(nd, np, mat);
    CHECK(t, "rj_topo_custom(100,4) failed");
    if (!t) return 1;
    free(mat);

    char err[512];
    rj_op op = rj_op_encode();
    rj_options o; rj_options_default(&o);
    o.benchmark = 0;
    rj_jit *j = rj_compile(t, &op, &o, err, sizeof(err));
    CHECK(j, "compile failed: %s", err);
    if (!j) { rj_op_free(&op); rj_topo_free(t); return 1; }

    rj_info info; rj_get_info(j, &info);
    printf("large MDS: %s\n", info.desc);

    /* encode correctness at a few lengths */
    size_t lens[] = { 0, 1, 64, 1000, 65536 };
    for (size_t li = 0; li < sizeof(lens)/sizeof(lens[0]); li++) {
        size_t len = lens[li];
        uint8_t **d = malloc(sizeof(*d) * nd);
        uint8_t **p = malloc(sizeof(*p) * np);
        uint8_t **ref = malloc(sizeof(*ref) * np);
        for (int i = 0; i < nd; i++) {
            d[i] = alloc64(len);
            for (size_t k = 0; k < len; k++)
                d[i][k] = (uint8_t)(xrng() >> 33);
        }
        for (int k = 0; k < np; k++) {
            p[k] = alloc64(len);
            ref[k] = alloc64(len);
            memset(ref[k], 0, len);
        }
        /* reference: scalar Cauchy encode */
        for (size_t k = 0; k < len; k++) {
            for (int jj = 0; jj < np; jj++) {
                uint8_t acc = 0;
                for (int i = 0; i < nd; i++) {
                    uint8_t x = (uint8_t)(i + 1);
                    uint8_t y = (uint8_t)(nd + 1 + jj);
                    acc ^= gf_mul(gf_inv(x ^ y), d[i][k]);
                }
                ref[jj][k] = acc;
            }
        }

        const void *in[100];
        void *out[4];
        for (int i = 0; i < nd; i++) in[i] = d[i];
        for (int k = 0; k < np; k++) out[k] = p[k];
        rj_get_fn(j)(in, out, len);

        for (int k = 0; k < np; k++)
            CHECK(memcmp(p[k], ref[k], len) == 0,
                  "len=%zu parity %d mismatch", len, k);

        for (int i = 0; i < nd; i++) free(d[i] - 64);
        for (int k = 0; k < np; k++) { free(p[k] - 64); free(ref[k] - 64); }
        free(d); free(p); free(ref);
        if (fails) break;
    }

    /* recover: fail 4 data disks, verify reconstruction */
    {
        int failed[] = { 0, 25, 50, 75 };
        rj_op rop = rj_op_recover(failed, 4);
        rj_jit *rj = rj_compile(t, &rop, &o, err, sizeof(err));
        CHECK(rj, "recover compile failed: %s", err);
        if (rj) {
            size_t len = 4096;
            uint8_t **d = malloc(sizeof(*d) * nd);
            uint8_t **p = malloc(sizeof(*p) * np);
            for (int i = 0; i < nd; i++) {
                d[i] = alloc64(len);
                for (size_t k = 0; k < len; k++)
                    d[i][k] = (uint8_t)(xrng() >> 33);
            }
            for (int k = 0; k < np; k++) p[k] = alloc64(len);
            /* encode to get parities */
            const void *ein[100]; void *eout[4];
            for (int i = 0; i < nd; i++) ein[i] = d[i];
            for (int k = 0; k < np; k++) eout[k] = p[k];
            rj_get_fn(j)(ein, eout, len);
            /* save failed disks */
            uint8_t *saved[4];
            for (int f = 0; f < 4; f++) {
                saved[f] = alloc64(len);
                memcpy(saved[f], d[failed[f]], len);
                memset(d[failed[f]], 0, len); /* simulate loss */
            }
            /* recover: inputs are survivors + parities */
            const void *rin[104]; void *rout[4];
            int ri = 0;
            for (int i = 0; i < nd; i++) {
                int is_failed = 0;
                for (int f = 0; f < 4; f++)
                    if (i == failed[f]) is_failed = 1;
                if (!is_failed) rin[ri++] = d[i];
            }
            for (int k = 0; k < np; k++) rin[ri++] = p[k];
            for (int f = 0; f < 4; f++) rout[f] = d[failed[f]];
            rj_get_fn(rj)(rin, rout, len);
            for (int f = 0; f < 4; f++)
                CHECK(memcmp(d[failed[f]], saved[f], len) == 0,
                      "recover disk %d mismatch", failed[f]);
            for (int i = 0; i < nd; i++) free(d[i] - 64);
            for (int k = 0; k < np; k++) free(p[k] - 64);
            for (int f = 0; f < 4; f++) free(saved[f] - 64);
            free(d); free(p);
            rj_free(rj);
        }
        rj_op_free(&rop);
    }

    rj_free(j); rj_op_free(&op); rj_topo_free(t);
    if (!fails) printf("large MDS (100+4 Cauchy): all OK\n");
    return fails ? 1 : 0;
}
