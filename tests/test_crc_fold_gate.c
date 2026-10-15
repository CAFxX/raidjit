/* REDUX gate unit test (spec §3.6, §8.4).
 *
 * fused_crc_interleave_ok must return 1 ONLY for (SPR uarch +
 * pclmulqdq + avx512f); every other combination fails closed.
 * Covers: flag-clearing simulation, non-SPR uarch with flags set,
 * NULL cpu. The SPR-override checks run in a subprocess because
 * rj_uarch_override has no reset.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/fused.h"
#include "../src/cpu.h"

static int fails = 0;
#define CHECK(cond, ...)                                                  \
    do {                                                                  \
        if (!(cond)) {                                                    \
            printf("FAIL %s:%d: ", __func__, __LINE__);                    \
            printf(__VA_ARGS__);                                          \
            printf("\n");                                                 \
            fails++;                                                      \
        }                                                                 \
    } while (0)

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    int spr_mode = argc > 1 && !strcmp(argv[1], "spr");

    if (!spr_mode) {
        /* Gate closed, deterministically: force a non-SPR uarch. The gate
         * opens for the SPR uarch bucket, so on a host in that bucket
         * (e.g. an Emerald Rapids CI machine) the unfixed check failed.
         * SKYLAKEX keeps the gate closed on ANY host. */
        rj_uarch_override(RJ_UARCH_INTEL_SKYLAKEX);
        rj_cpu_info c;
        rj_cpu_detect(&c);
        c.pclmulqdq = 1;
        c.avx512f = 1;
        CHECK(fused_crc_interleave_ok(&c) == 0,
              "non-SPR uarch with flags set must be closed");
        CHECK(fused_crc_interleave_ok(NULL) == 0,
              "NULL cpu must be closed");
        /* Now the SPR-override half in a subprocess. */
        char cmd[512];
        snprintf(cmd, sizeof(cmd), "%s spr", argv[0]);
        int rc = system(cmd);
        CHECK(rc == 0, "spr subprocess failed");
        printf("host-gate checks done\n");
    } else {
        rj_uarch_override(RJ_UARCH_INTEL_SAPPHIRERAPIDS);
        rj_cpu_info c;
        memset(&c, 0, sizeof(c));
        c.pclmulqdq = 1;
        c.avx512f = 1;
        CHECK(fused_crc_interleave_ok(&c) == 1,
              "SPR + pclmulqdq + avx512f must be open");
        c.pclmulqdq = 0;
        CHECK(fused_crc_interleave_ok(&c) == 0,
              "cleared pclmulqdq must close the gate");
        c.pclmulqdq = 1;
        c.avx512f = 0;
        CHECK(fused_crc_interleave_ok(&c) == 0,
              "cleared avx512f must close the gate");
        memset(&c, 0, sizeof(c));
        CHECK(fused_crc_interleave_ok(&c) == 0,
              "no flags must close the gate");
        printf("spr-gate checks done\n");
    }

    if (fails == 0)
        printf("PASS\n");
    return fails != 0;
}
