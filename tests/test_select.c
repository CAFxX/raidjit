/* Unit tests for select_by_time(): the benchmark selection rules.
 *
 * The rule: measure median seconds/call per candidate; every candidate
 * within `window` of the best time is a noise tie, where window =
 * max(1.5%, measured noise); ties break on uops/byte, then mem/byte, then
 * code size. A candidate outside the window must NEVER win on tie-breaks
 * alone.
 */
#include "../src/bench.h"

#include <stdio.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, ...)                                                   \
    do {                                                                   \
        if (!(cond)) {                                                     \
            printf("FAIL %s:%d: ", __func__, __LINE__);                    \
            printf(__VA_ARGS__);                                           \
            printf("\n");                                                  \
            failures++;                                                    \
        }                                                                  \
    } while (0)

/* build nv versions with given uops/mem/code */
static void mkvs(version_t *vs, int nv, const double *uops, const double *mem,
                 const size_t *code)
{
    for (int i = 0; i < nv; i++) {
        memset(&vs[i], 0, sizeof(vs[i]));
        vs[i].uops_per_byte = uops[i];
        vs[i].mem_per_byte = mem[i];
        vs[i].code_size = code[i];
        snprintf(vs[i].desc, sizeof(vs[i].desc), "v%d", i);
    }
}

int main(void)
{
    version_t vs[4];
    double secs[4];

    /* 1. clear winner: fastest time wins even with worse uops */
    {
        double u[] = { 5.0, 1.0 };
        double m[] = { 9.0, 9.0 };
        size_t c[] = { 100, 100 };
        mkvs(vs, 2, u, m, c);
        secs[0] = 1.0;
        secs[1] = 1.5; /* 50% slower: outside any tie */
        CHECK(select_by_time(vs, 2, secs, 0.015) == 0, "fastest must win");
    }

    /* 2. tie within 1.5%: lower uops wins despite slightly slower time */
    {
        double u[] = { 5.0, 1.0 };
        double m[] = { 9.0, 9.0 };
        size_t c[] = { 100, 100 };
        mkvs(vs, 2, u, m, c);
        secs[0] = 1.0;
        secs[1] = 1.01; /* 1% slower: tied */
        CHECK(select_by_time(vs, 2, secs, 0.015) == 1, "tie must prefer lower uops");
    }

    /* 3. regression: slower candidate outside 1.5% must not win on uops.
     * (old code seeded winner=0 and let an out-of-window candidate win) */
    {
        double u[] = { 1.0, 5.0 };
        double m[] = { 9.0, 9.0 };
        size_t c[] = { 100, 100 };
        mkvs(vs, 2, u, m, c);
        secs[0] = 1.03; /* 3% slower than best, better uops */
        secs[1] = 1.0;  /* best time */
        CHECK(select_by_time(vs, 2, secs, 0.015) == 1,
              "out-of-window candidate must not win on tie-break");
    }

    /* 4. uops tie -> lower mem/byte wins */
    {
        double u[] = { 2.0, 2.0 };
        double m[] = { 12.0, 8.0 };
        size_t c[] = { 100, 100 };
        mkvs(vs, 2, u, m, c);
        secs[0] = 1.0;
        secs[1] = 1.005;
        CHECK(select_by_time(vs, 2, secs, 0.015) == 1, "tie must prefer lower mem");
    }

    /* 5. uops+mem tie -> smaller code wins */
    {
        double u[] = { 2.0, 2.0 };
        double m[] = { 8.0, 8.0 };
        size_t c[] = { 5000, 1000 };
        mkvs(vs, 2, u, m, c);
        secs[0] = 1.0;
        secs[1] = 1.005;
        CHECK(select_by_time(vs, 2, secs, 0.015) == 1,
              "tie must prefer smaller code");
    }

    /* 6. exact 1.5% boundary is eligible; just past it is not */
    {
        double u[] = { 9.0, 1.0, 1.0 };
        double m[] = { 9.0, 9.0, 9.0 };
        size_t c[] = { 100, 100, 100 };
        mkvs(vs, 3, u, m, c);
        secs[0] = 1.0;         /* best time, worst uops */
        secs[1] = 1.015;       /* exactly at boundary: eligible */
        secs[2] = 1.0150001;   /* just past: not eligible */
        int w = select_by_time(vs, 3, secs, 0.015);
        CHECK(w == 1, "boundary candidate must win tie-break, got %d", w);
    }

    /* 7. single candidate */
    {
        double u[] = { 3.0 };
        double m[] = { 9.0 };
        size_t c[] = { 100 };
        mkvs(vs, 1, u, m, c);
        secs[0] = 2.5;
        CHECK(select_by_time(vs, 1, secs, 0.015) == 0, "single candidate");
    }

    /* 8. best time not at index 0; full ordering among three tied */
    {
        double u[] = { 4.0, 3.0, 2.0 };
        double m[] = { 9.0, 9.0, 9.0 };
        size_t c[] = { 100, 100, 100 };
        mkvs(vs, 3, u, m, c);
        secs[0] = 1.01;
        secs[1] = 1.0; /* best */
        secs[2] = 1.012;
        CHECK(select_by_time(vs, 3, secs, 0.015) == 2, "best uops among ties");
    }

    /* 9. equal everything: first best-time index wins (deterministic) */
    {
        double u[] = { 2.0, 2.0 };
        double m[] = { 8.0, 8.0 };
        size_t c[] = { 100, 100 };
        mkvs(vs, 2, u, m, c);
        secs[0] = 1.0;
        secs[1] = 1.0;
        CHECK(select_by_time(vs, 2, secs, 0.015) == 0,
              "deterministic on full tie");
    }

    /* 10. adaptive window: with window=10%, a 5%-slower but more efficient
     * candidate ties and wins the tie-break; with 1.5% it loses outright */
    {
        double u[] = { 5.0, 1.0 };
        double m[] = { 9.0, 9.0 };
        size_t c[] = { 100, 100 };
        mkvs(vs, 2, u, m, c);
        secs[0] = 1.0;
        secs[1] = 1.05; /* 5% slower */
        CHECK(select_by_time(vs, 2, secs, 0.10) == 1,
              "wide window must admit efficient candidate");
        CHECK(select_by_time(vs, 2, secs, 0.015) == 0,
              "narrow window must keep fastest");
    }

    /* 11. outside even a wide window: must never win on tie-break */
    {
        double u[] = { 5.0, 1.0 };
        double m[] = { 9.0, 9.0 };
        size_t c[] = { 100, 100 };
        mkvs(vs, 2, u, m, c);
        secs[0] = 1.0;
        secs[1] = 1.25; /* 25% slower */
        CHECK(select_by_time(vs, 2, secs, 0.10) == 0,
              "out-of-window candidate must not win");
    }

    /* 12. zero window: pure argmin, no tie-break */
    {
        double u[] = { 5.0, 1.0 };
        double m[] = { 9.0, 9.0 };
        size_t c[] = { 100, 100 };
        mkvs(vs, 2, u, m, c);
        secs[0] = 1.0;
        secs[1] = 1.000001;
        CHECK(select_by_time(vs, 2, secs, 0.0) == 0,
              "zero window must pick argmin");
    }

    /* 13. noise_window: pooled median-of-medians with the 1.5% floor.
     * csn rows are per-candidate, columns per size. */
    {
        /* 2 candidates x 3 sizes; noises: {0.10,0.20,0.30} and
         * {0.40,0.50,0.60} -> per-candidate medians 0.20, 0.50 ->
         * pooled: sorted {0.20,0.50}, ts[2/2]=ts[1]=0.50 */
        double csn[] = { 0.10, 0.20, 0.30, 0.40, 0.50, 0.60 };
        double pc[2];
        double w = noise_window(csn, 2, 3, pc);
        CHECK(w > 0.499 && w < 0.501, "pooled window, got %f", w);
        CHECK(pc[0] > 0.199 && pc[0] < 0.201, "per-cand[0], got %f", pc[0]);
        CHECK(pc[1] > 0.499 && pc[1] < 0.501, "per-cand[1], got %f", pc[1]);
    }

    /* 14. noise_window: floor applies when pooled noise < 1.5% */
    {
        double csn[] = { 0.001, 0.002, 0.003 };
        double w = noise_window(csn, 1, 3, NULL);
        CHECK(w == 0.015, "floor must apply, got %f", w);
    }

    /* 15. noise_window: single size (nsizes=1 dedup case) */
    {
        double csn[] = { 0.25, 0.10 };
        double w = noise_window(csn, 2, 1, NULL);
        /* per-candidate medians 0.25, 0.10 -> pooled ts[1]=0.25 */
        CHECK(w > 0.249 && w < 0.251, "nsizes=1, got %f", w);
    }

    /* 16. NaN time must never win, even with the best uops */
    {
        double u[] = { 5.0, 1.0 };
        double m[] = { 9.0, 9.0 };
        size_t c[] = { 100, 100 };
        mkvs(vs, 2, u, m, c);
        secs[0] = 1.0;
        secs[1] = 0.0 / 0.0; /* NaN */
        CHECK(select_by_time(vs, 2, secs, 0.10) == 0,
              "NaN candidate must not win");
    }

    if (failures) {
        printf("SELECT TESTS FAILED (%d)\n", failures);
        return 1;
    }
    printf("select: all OK\n");
    return 0;
}
