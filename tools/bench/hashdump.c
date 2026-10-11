/* Hash oracle for the benchmark facility (CRITICAL-1 fix).
 *
 * Reconstructs a kernel from (topology, op, plan_id) and outputs the
 * SHA-256 of the generated machine code. This materializes
 * (config -> code_hash) for the CURRENT checkout, enabling the staleness
 * model to compare old measurements against current code.
 *
 * Usage:
 *   hashdump <topology> <op> <failed_csv> <plan_id>
 *
 *   topology: "rs(6,3)", "raid5(4)", "raid6(6)", "raid0(4)"
 *   op: "encode", "recover", "parity_update"
 *   failed_csv: comma-separated failed disk indices (for recover), or "-"
 *   plan_id: the full plan ID string from a result file
 *
 * Output (stdout): hex SHA-256 of the code bytes, or "ERROR: <msg>" on failure.
 * Exit 0 on success, 1 on clean rejection, 2 on internal error.
 *
 * This is NOT benchmarking: it only compiles, never times.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <openssl/sha.h>

#define _POSIX_C_SOURCE 200809L

#include "include/raidjit.h"
#include "src/plan.h"
#include "src/planid.h"
#include "src/jit.h"
#include "src/cpu.h"

static rj_topology *parse_topology(const char *s, char *err, size_t errlen)
{
    int a = 0, b = 0;
    if (sscanf(s, "rs(%d,%d)", &a, &b) == 2 && a > 0 && b >= 0) {
        rj_topology *t = rj_topo_rs(a, b);
        if (!t) { snprintf(err, errlen, "rj_topo_rs(%d,%d) failed", a, b); }
        return t;
    }
    if (sscanf(s, "raid5(%d)", &a) == 1 && a > 0) {
        return rj_topo_raid5(a);
    }
    if (sscanf(s, "raid6(%d)", &a) == 1 && a > 0) {
        return rj_topo_raid6(a);
    }
    if (sscanf(s, "raid0(%d)", &a) == 1 && a > 0) {
        return rj_topo_raid0(a);
    }
    snprintf(err, errlen, "unsupported topology string: %s", s);
    return NULL;
}

int main(int argc, char **argv)
{
    if (argc != 5) {
        fprintf(stderr, "usage: %s <topology> <op> <failed_csv|-> <plan_id>\n", argv[0]);
        return 2;
    }
    const char *topo_s = argv[1];
    const char *op_s = argv[2];
    const char *failed_s = argv[3];
    const char *plan_id = argv[4];
    char err[512];

    /* 1. Topology */
    rj_topology *topo = parse_topology(topo_s, err, sizeof(err));
    if (!topo) { printf("ERROR: %s\n", err); return 1; }

    /* 2. Op */
    rj_op op;
    if (strcmp(op_s, "encode") == 0) {
        op = rj_op_encode();
    } else if (strcmp(op_s, "recover") == 0) {
        int failed[64]; int nf = 0;
        if (strcmp(failed_s, "-") != 0) {
            char *copy = strdup(failed_s);
            char *tok = strtok(copy, ",");
            while (tok && nf < 64) {
                failed[nf++] = atoi(tok);
                tok = strtok(NULL, ",");
            }
            free(copy);
        }
        op = rj_op_recover(failed, nf);
    } else {
        printf("ERROR: unsupported op: %s (parity_update not yet supported)\n", op_s);
        rj_topo_free(topo);
        return 1;
    }

    /* 3. Internal topo + name */
    topo_int_t ti;
    if (topo_int_build(&ti, topo, err, sizeof(err)) != 0) {
        printf("ERROR: topo_int_build: %s\n", err);
        rj_topo_free(topo);
        return 1;
    }
    const char *tname = topo_internal_name(topo);

    /* 4. Parse plan ID -> candidate spec */
    planid_spec_t spec;
    memset(&spec, 0, sizeof(spec));
    if (planid_parse(plan_id, &ti, tname, &op, &spec, err, sizeof(err)) != 0) {
        printf("ERROR: planid_parse: %s\n", err);
        topo_int_free(&ti);
        rj_topo_free(topo);
        return 1;
    }

    /* 5. Build the plan for this (topo, op) */
    plan_t pl;
    memset(&pl, 0, sizeof(pl));
    if (plan_build(&ti, &op, tname, &pl, err, sizeof(err)) != 0) {
        printf("ERROR: plan_build: %s\n", err);
        topo_int_free(&ti);
        rj_topo_free(topo);
        return 1;
    }
    /* Verify the built plan's kind matches the plan_id's expectation.
     * (plan_build picks the kind; a mismatch means our reconstruction
     * diverged from what the benchmark measured.) */
    if (pl.kind != spec.kind) {
        printf("ERROR: kind mismatch: plan_build gave %d, plan_id expects %d\n",
               (int)pl.kind, (int)spec.kind);
        plan_free(&pl);
        topo_int_free(&ti);
        rj_topo_free(topo);
        return 1;
    }

    /* 6. Validate the candidate (converts backend asserts to clean errors) */
    if (planid_cand_valid(&pl, &spec.cand, err, sizeof(err)) != 0) {
        printf("ERROR: planid_cand_valid: %s\n", err);
        plan_free(&pl);
        topo_int_free(&ti);
        rj_topo_free(topo);
        return 1;
    }

    /* 7. Build the version */
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    rj_options opts;
    rj_options_default(&opts);
    opts.benchmark = 0;
    version_t v;
    memset(&v, 0, sizeof(v));
    if (jit_build_version(&pl, &spec.cand, &cpu, &opts, &v, err, sizeof(err)) != 0) {
        printf("ERROR: jit_build_version: %s\n", err);
        plan_free(&pl);
        topo_int_free(&ti);
        rj_topo_free(topo);
        return 1;
    }

    /* 8. Hash the code bytes */
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256(v.mem, v.code_size, hash);
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++)
        printf("%02x", hash[i]);
    printf("\n");

    jit_free_version(&v);
    plan_free(&pl);
    topo_int_free(&ti);
    rj_topo_free(topo);
    return 0;
}
