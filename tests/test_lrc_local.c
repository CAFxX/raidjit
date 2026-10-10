/* test_lrc_local.c — unit tests for LRC locality-aware recovery.
 *
 * Tests survivor_lrc_detect() and survivor_lrc_local_cover() edge cases:
 *   1. Non-LRC topology (RS: no pure-XOR rows) → detect returns 0
 *   2. RAID6 (single pure-XOR P row) → detect returns 1, ngroups=1
 *   3. LRC Azure → detect returns 1, correct groups
 *   4. Single data failure in local group → local cover found
 *   5. Single local parity failure → recompute (no survival check)
 *   6. Single global parity failure → generic path (cover returns 0)
 *   7. Two failures in distinct groups → local cover
 *   8. Two failures in same group → generic fallback (cover returns 0)
 *   9. Data failure + its local row failed → generic fallback
 *  10. Custom LRC-like matrix → detected by structure, not name
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/raidjit.h"
#include "../src/survivor.h"

static int nchecks = 0;
static int nfail = 0;

#define CHECK(cond, fmt, ...) do { \
    nchecks++; \
    if (!(cond)) { \
        nfail++; \
        printf("FAIL %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
    } \
} while (0)

static int mk_ti(rj_topology *t, topo_int_t *ti)
{
    char err[512];
    int rc = topo_int_build(ti, t, err, sizeof(err));
    if (rc != 0)
        printf("mk_ti failed: %s\n", err);
    return rc;
}

/* 1. Custom matrix with NO pure-XOR rows → detect returns 0.
 * All rows have GF coefficients (no binary rows). */
static void test_no_xor_rows_not_lrc(void)
{
    /* 4 data, 2 parity, all coefficients non-binary */
    static const uint8_t mat[2*4] = {
        2,3,4,5,
        6,7,8,9,
    };
    rj_topology *t = rj_topo_custom(4, 2, mat);
    CHECK(t != NULL, "rj_topo_custom returned NULL");
    if (!t) return;
    topo_int_t ti;
    CHECK(mk_ti(t, &ti) == 0, "mk_ti failed");
    int group_of[RJ_MAX_DISKS], local_row[RJ_MAX_DISKS], ngroups = -1;
    int r = survivor_lrc_detect(&ti, group_of, local_row, &ngroups);
    CHECK(r == 0, "non-XOR matrix detected as LRC (r=%d)", r);
    topo_int_free(&ti);
    rj_topo_free(t);
}

/* 2. RAID6: P row is pure-XOR covering all data → 1 group. */
static void test_raid6_single_group(void)
{
    rj_topology *t = rj_topo_raid6(6);
    CHECK(t != NULL, "rj_topo_raid6(6) returned NULL");
    if (!t) return;
    topo_int_t ti;
    CHECK(mk_ti(t, &ti) == 0, "mk_ti failed");
    int group_of[RJ_MAX_DISKS], local_row[RJ_MAX_DISKS], ngroups = -1;
    int r = survivor_lrc_detect(&ti, group_of, local_row, &ngroups);
    CHECK(r == 1, "RAID6 not detected as LRC-shaped (r=%d)", r);
    if (r == 1) {
        CHECK(ngroups == 1, "RAID6 ngroups=%d, expected 1", ngroups);
        /* All data disks in group 0 */
        for (int i = 0; i < ti.ndata; i++)
            CHECK(group_of[i] == 0, "RAID6 disk %d group=%d, expected 0",
                  i, group_of[i]);
        /* Single data failure → local cover is the P row */
        int F[1] = { 2 };
        int cover[RJ_MAX_DISKS];
        int cr = survivor_lrc_local_cover(&ti, F, 1, group_of, local_row,
                                          ngroups, cover);
        CHECK(cr == 1, "RAID6 single data failure: no local cover");
        if (cr == 1)
            CHECK(cover[0] == local_row[0],
                  "RAID6 cover row=%d, expected local %d",
                  cover[0], local_row[0]);
    }
    topo_int_free(&ti);
    rj_topo_free(t);
}

/* 3. LRC Azure (12,2,2): 2 local groups of 6 (locality 6). */
static void test_azure_detect(void)
{
    rj_topology *t = rj_topo_lrc_azure();
    CHECK(t != NULL, "rj_topo_lrc_azure() returned NULL");
    if (!t) return;
    topo_int_t ti;
    CHECK(mk_ti(t, &ti) == 0, "mk_ti failed");
    int group_of[RJ_MAX_DISKS], local_row[RJ_MAX_DISKS], ngroups = -1;
    int r = survivor_lrc_detect(&ti, group_of, local_row, &ngroups);
    CHECK(r == 1, "Azure not detected as LRC (r=%d)", r);
    if (r == 1) {
        CHECK(ngroups == 2, "Azure ngroups=%d, expected 2", ngroups);
        CHECK(ti.ndata == 12, "Azure ndata=%d, expected 12", ti.ndata);
        /* Disks 0-5 in group 0; 6-11 in group 1. */
        for (int i = 0; i < 6; i++)
            CHECK(group_of[i] == 0, "Azure disk %d group=%d, expected 0",
                  i, group_of[i]);
        for (int i = 6; i < 12; i++)
            CHECK(group_of[i] == 1, "Azure disk %d group=%d, expected 1",
                  i, group_of[i]);
        /* Local rows are 2 and 3 (after 2 global rows). */
        CHECK(local_row[0] == 2, "local_row[0]=%d, expected 2",
              local_row[0]);
        CHECK(local_row[1] == 3, "local_row[1]=%d, expected 3",
              local_row[1]);
    }
    topo_int_free(&ti);
    rj_topo_free(t);
}

/* 4. Single data failure in Azure group 0 → local cover = group's row. */
static void test_azure_single_data(void)
{
    rj_topology *t = rj_topo_lrc_azure();
    if (!t) return;
    topo_int_t ti;
    if (mk_ti(t, &ti) != 0) { rj_topo_free(t); return; }
    int group_of[RJ_MAX_DISKS], local_row[RJ_MAX_DISKS], ngroups;
    CHECK(survivor_lrc_detect(&ti, group_of, local_row, &ngroups) == 1,
          "detect failed");
    /* Fail disk 0 (group 0) */
    int F[1] = { 0 };
    int cover[RJ_MAX_DISKS];
    int r = survivor_lrc_local_cover(&ti, F, 1, group_of, local_row,
                                     ngroups, cover);
    CHECK(r == 1, "single data failure: no local cover");
    if (r == 1) {
        CHECK(cover[0] == local_row[0],
              "cover=%d, expected local_row[0]=%d", cover[0], local_row[0]);
    }
    topo_int_free(&ti);
    rj_topo_free(t);
}

/* 5. Single local parity failure → recompute from group (no survival
 * check on the failed row itself). */
static void test_azure_local_parity_fail(void)
{
    rj_topology *t = rj_topo_lrc_azure();
    if (!t) return;
    topo_int_t ti;
    if (mk_ti(t, &ti) != 0) { rj_topo_free(t); return; }
    int group_of[RJ_MAX_DISKS], local_row[RJ_MAX_DISKS], ngroups;
    CHECK(survivor_lrc_detect(&ti, group_of, local_row, &ngroups) == 1,
          "detect failed");
    /* Fail local parity of group 0: disk index = ndata + local_row[0] */
    int F[1] = { ti.ndata + local_row[0] };
    int cover[RJ_MAX_DISKS];
    int r = survivor_lrc_local_cover(&ti, F, 1, group_of, local_row,
                                     ngroups, cover);
    CHECK(r == 1, "local parity failure: no local cover (r=%d)", r);
    if (r == 1)
        CHECK(cover[0] == local_row[0],
              "cover=%d, expected %d", cover[0], local_row[0]);
    topo_int_free(&ti);
    rj_topo_free(t);
}

/* 6. Single global parity failure → generic path (cover returns 0). */
static void test_azure_global_parity_fail(void)
{
    rj_topology *t = rj_topo_lrc_azure();
    if (!t) return;
    topo_int_t ti;
    if (mk_ti(t, &ti) != 0) { rj_topo_free(t); return; }
    int group_of[RJ_MAX_DISKS], local_row[RJ_MAX_DISKS], ngroups;
    CHECK(survivor_lrc_detect(&ti, group_of, local_row, &ngroups) == 1,
          "detect failed");
    /* Find a global parity row (not in local_row[]) */
    int global_pj = -1;
    for (int j = 0; j < ti.npar && global_pj < 0; j++) {
        int is_local = 0;
        for (int g = 0; g < ngroups; g++)
            if (local_row[g] == j) is_local = 1;
        if (!is_local) global_pj = j;
    }
    CHECK(global_pj >= 0, "no global parity row found");
    if (global_pj >= 0) {
        int F[1] = { ti.ndata + global_pj };
        int cover[RJ_MAX_DISKS];
        int r = survivor_lrc_local_cover(&ti, F, 1, group_of, local_row,
                                         ngroups, cover);
        CHECK(r == 0, "global parity failure: unexpected local cover");
    }
    topo_int_free(&ti);
    rj_topo_free(t);
}

/* 7. Two failures in distinct groups → local cover with 2 rows. */
static void test_azure_two_distinct_groups(void)
{
    rj_topology *t = rj_topo_lrc_azure();
    if (!t) return;
    topo_int_t ti;
    if (mk_ti(t, &ti) != 0) { rj_topo_free(t); return; }
    int group_of[RJ_MAX_DISKS], local_row[RJ_MAX_DISKS], ngroups;
    CHECK(survivor_lrc_detect(&ti, group_of, local_row, &ngroups) == 1,
          "detect failed");
    /* Fail disk 0 (group 0) and disk 6 (group 1) */
    int F[2] = { 0, 6 };
    int cover[RJ_MAX_DISKS];
    int r = survivor_lrc_local_cover(&ti, F, 2, group_of, local_row,
                                     ngroups, cover);
    CHECK(r == 1, "two distinct-group failures: no local cover");
    if (r == 1) {
        CHECK(cover[0] == local_row[0] || cover[1] == local_row[0],
              "group 0 row not in cover");
        CHECK(cover[0] == local_row[1] || cover[1] == local_row[1],
              "group 1 row not in cover");
        CHECK(cover[0] < cover[1], "cover not sorted");
    }
    topo_int_free(&ti);
    rj_topo_free(t);
}

/* 8. Two failures in same group → generic fallback (cover returns 0). */
static void test_azure_two_same_group(void)
{
    rj_topology *t = rj_topo_lrc_azure();
    if (!t) return;
    topo_int_t ti;
    if (mk_ti(t, &ti) != 0) { rj_topo_free(t); return; }
    int group_of[RJ_MAX_DISKS], local_row[RJ_MAX_DISKS], ngroups;
    CHECK(survivor_lrc_detect(&ti, group_of, local_row, &ngroups) == 1,
          "detect failed");
    /* Fail disks 0 and 1 (both group 0) */
    int F[2] = { 0, 1 };
    int cover[RJ_MAX_DISKS];
    int r = survivor_lrc_local_cover(&ti, F, 2, group_of, local_row,
                                     ngroups, cover);
    CHECK(r == 0, "same-group failures: unexpected local cover");
    topo_int_free(&ti);
    rj_topo_free(t);
}

/* 9. Data failure + its local row also failed → generic fallback. */
static void test_azure_data_plus_local_row(void)
{
    rj_topology *t = rj_topo_lrc_azure();
    if (!t) return;
    topo_int_t ti;
    if (mk_ti(t, &ti) != 0) { rj_topo_free(t); return; }
    int group_of[RJ_MAX_DISKS], local_row[RJ_MAX_DISKS], ngroups;
    CHECK(survivor_lrc_detect(&ti, group_of, local_row, &ngroups) == 1,
          "detect failed");
    /* Fail disk 0 (group 0) and local parity of group 0 */
    int F[2] = { 0, ti.ndata + local_row[0] };
    /* Sort F (required: validated sorted/deduped) */
    if (F[0] > F[1]) { int tmp = F[0]; F[0] = F[1]; F[1] = tmp; }
    int cover[RJ_MAX_DISKS];
    int r = survivor_lrc_local_cover(&ti, F, 2, group_of, local_row,
                                     ngroups, cover);
    CHECK(r == 0, "data+local-row failure: unexpected local cover");
    topo_int_free(&ti);
    rj_topo_free(t);
}

/* 10. Custom LRC-like matrix → detected by structure.
 * Build a 6-data, 3-parity matrix: 2 local XOR rows (disks 0-2, 3-5)
 * + 1 global row (all ones, but with a GF coef to make it non-XOR...
 * actually all-ones IS pure XOR, so use coefficient 2 to break it). */
static void test_custom_lrc_shape(void)
{
    /* 6 data, 3 parity:
     * row 0: 1 1 1 0 0 0  (local group 0)
     * row 1: 0 0 0 1 1 1  (local group 1)
     * row 2: 1 2 3 4 5 6  (global, non-XOR) */
    static const uint8_t mat[3*6] = {
        1,1,1,0,0,0,
        0,0,0,1,1,1,
        1,2,3,4,5,6,
    };
    rj_topology *t = rj_topo_custom(6, 3, mat);
    CHECK(t != NULL, "rj_topo_custom returned NULL");
    if (!t) return;
    topo_int_t ti;
    CHECK(mk_ti(t, &ti) == 0, "mk_ti failed");
    int group_of[RJ_MAX_DISKS], local_row[RJ_MAX_DISKS], ngroups = -1;
    int r = survivor_lrc_detect(&ti, group_of, local_row, &ngroups);
    CHECK(r == 1, "custom LRC shape not detected (r=%d)", r);
    if (r == 1) {
        CHECK(ngroups == 2, "ngroups=%d, expected 2", ngroups);
        CHECK(group_of[0] == 0 && group_of[1] == 0 && group_of[2] == 0,
              "group 0 wrong");
        CHECK(group_of[3] == 1 && group_of[4] == 1 && group_of[5] == 1,
              "group 1 wrong");
        CHECK(local_row[0] == 0, "local_row[0]=%d, expected 0",
              local_row[0]);
        CHECK(local_row[1] == 1, "local_row[1]=%d, expected 1",
              local_row[1]);
    }
    topo_int_free(&ti);
    rj_topo_free(t);
}

/* 11. Overlapping pure-XOR rows → NOT LRC (disjointness violated). */
static void test_overlapping_rows(void)
{
    /* 4 data, 2 parity:
     * row 0: 1 1 0 0
     * row 1: 0 1 1 0  (overlaps row 0 on disk 1) */
    static const uint8_t mat[2*4] = {
        1,1,0,0,
        0,1,1,0,
    };
    rj_topology *t = rj_topo_custom(4, 2, mat);
    CHECK(t != NULL, "rj_topo_custom returned NULL");
    if (!t) return;
    topo_int_t ti;
    CHECK(mk_ti(t, &ti) == 0, "mk_ti failed");
    int group_of[RJ_MAX_DISKS], local_row[RJ_MAX_DISKS], ngroups = -1;
    int r = survivor_lrc_detect(&ti, group_of, local_row, &ngroups);
    CHECK(r == 0, "overlapping rows detected as LRC (r=%d)", r);
    topo_int_free(&ti);
    rj_topo_free(t);
}

/* 12. Incomplete partition (disk in no local row) → NOT LRC. */
static void test_incomplete_partition(void)
{
    /* 4 data, 1 parity:
     * row 0: 1 1 0 0  (disks 2,3 uncovered) */
    static const uint8_t mat[1*4] = { 1,1,0,0 };
    rj_topology *t = rj_topo_custom(4, 1, mat);
    CHECK(t != NULL, "rj_topo_custom returned NULL");
    if (!t) return;
    topo_int_t ti;
    CHECK(mk_ti(t, &ti) == 0, "mk_ti failed");
    int group_of[RJ_MAX_DISKS], local_row[RJ_MAX_DISKS], ngroups = -1;
    int r = survivor_lrc_detect(&ti, group_of, local_row, &ngroups);
    CHECK(r == 0, "incomplete partition detected as LRC (r=%d)", r);
    topo_int_free(&ti);
    rj_topo_free(t);
}

int main(void)
{
    test_no_xor_rows_not_lrc();
    test_raid6_single_group();
    test_azure_detect();
    test_azure_single_data();
    test_azure_local_parity_fail();
    test_azure_global_parity_fail();
    test_azure_two_distinct_groups();
    test_azure_two_same_group();
    test_azure_data_plus_local_row();
    test_custom_lrc_shape();
    test_overlapping_rows();
    test_incomplete_partition();
    printf("lrc_local: %d checks, %d failures\n", nchecks, nfail);
    return nfail ? 1 : 0;
}
