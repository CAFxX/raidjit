/* test_planid: stable plan IDs + variant enumeration (#29).
 *
 * Covers:
 *  Part 1: ID format basics -- 9 axes, "rj<V>:" prefix, determinism
 *          across recompiles (benchmark=0 path is deterministic).
 *  Part 2: pinned round-trip -- rj_compile_pinned(ID) succeeds,
 *          rj_plan_id(pinned) == ID, and the pinned function agrees
 *          with the original at many lengths (incl. tails).
 *  Part 3: loud failures -- malformed IDs, version mismatch, topology
 *          mismatch, op mismatch, unbuildable candidates: NULL with a
 *          non-empty err, never a crash, never a silent fallback.
 *  Part 4: mutation -- single-character corruptions of a valid ID must
 *          either fail loudly or name a different valid spec (in which
 *          case the round-trip reports the mutated ID, honestly).
 *  Part 5: rj_list_variants -- total covers the pre-prune space
 *          exactly (== plan_candidates count), every ID parses and
 *          round-trips, IDs unique, survivor/pruned tagging coherent
 *          (pruned => non-empty reason; +ternlog-pruned => a surviving
 *          +ternlog twin entry exists; benchmark=0 => no static
 *          pruning), count-only call agrees, short-nout fills nout and
 *          still returns the total.
 *
 * Fail-capable: every assertion pins a specific observable. The
 * negative controls (garbage IDs, mutated versions) must fail; a test
 * that cannot fail is not a test.
 */
#include "../include/raidjit.h"
#include "../src/planid.h"
#include "../src/plan.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int nfail = 0;

#define CHECK(cond, ...)                                                      \
    do {                                                                      \
        if (!(cond)) {                                                        \
            printf("FAIL %s:%d: ", __FILE__, __LINE__);                        \
            printf(__VA_ARGS__);                                              \
            printf("\n");                                                     \
            nfail++;                                                          \
        }                                                                     \
    } while (0)

static void *xalloc(size_t n, size_t align)
{
    void *p = NULL;
    if (posix_memalign(&p, align, n) != 0) {
        printf("FAIL: out of memory\n");
        exit(1);
    }
    return p;
}

/* Fill buffers with deterministic pseudorandom data. */
static void fill_bufs(uint8_t **b, int n, size_t len, unsigned long long seed)
{
    unsigned long long s = seed;
    for (int i = 0; i < n; i++)
        for (size_t k = 0; k < len; k++) {
            s = s * 6364136223846793005ULL + 1442695040888963407ULL;
            b[i][k] = (uint8_t)(s >> 33);
        }
}

/* Compare fn outputs of two units over in[] at several lengths. */
static void check_fn_agree(rj_fn a, rj_fn b, int nin, int nout)
{
    static const size_t lens[] = {1, 7, 63, 64, 65, 100, 1000,
                                  4096, 65537, 100000};
    uint8_t **in = xalloc(sizeof(uint8_t *) * (size_t)nin, 64);
    uint8_t **oa = xalloc(sizeof(uint8_t *) * (size_t)nout, 64);
    uint8_t **ob = xalloc(sizeof(uint8_t *) * (size_t)nout, 64);
    const void **vin = xalloc(sizeof(void *) * (size_t)nin, 64);
    void **voa = xalloc(sizeof(void *) * (size_t)nout, 64);
    void **vob = xalloc(sizeof(void *) * (size_t)nout, 64);
    for (int i = 0; i < nin; i++) {
        in[i] = xalloc(100000, 64);
        vin[i] = in[i];
    }
    for (int j = 0; j < nout; j++) {
        oa[j] = xalloc(100000, 64);
        ob[j] = xalloc(100000, 64);
        voa[j] = oa[j];
        vob[j] = ob[j];
    }
    for (unsigned li = 0; li < sizeof(lens) / sizeof(lens[0]); li++) {
        size_t len = lens[li];
        fill_bufs(in, nin, len, 0x1234 + len);
        for (int j = 0; j < nout; j++) {
            memset(oa[j], 0xA5, len);
            memset(ob[j], 0xA5, len);
        }
        a(vin, voa, len);
        b(vin, vob, len);
        for (int j = 0; j < nout; j++)
            CHECK(memcmp(oa[j], ob[j], len) == 0,
                  "pinned/original disagree at len=%zu out=%d", len, j);
    }
    for (int i = 0; i < nin; i++)
        free(in[i]);
    for (int j = 0; j < nout; j++) {
        free(oa[j]);
        free(ob[j]);
    }
    free(in);
    free(oa);
    free(ob);
    free(vin);
    free(voa);
    free(vob);
}

/* Compile deterministically (no benchmarking) and return the unit. */
static rj_jit *compile_det(const rj_topology *t, const rj_op *op)
{
    rj_options o;
    rj_options_default(&o);
    o.benchmark = 0;
    char err[256];
    rj_jit *j = rj_compile(t, op, &o, err, sizeof(err));
    CHECK(j != NULL, "deterministic compile failed: %s", err);
    return j;
}

/* Count the ':' axes in an ID (must be 9 fields). */
static int count_axes(const char *id)
{
    int n = 1;
    for (const char *s = id; *s; s++)
        if (*s == ':')
            n++;
    return n;
}

int main(void)
{
    /* ============ Part 1: format basics + determinism ============ */
    {
        rj_topology *t = rj_topo_raid6(8);
        CHECK(t != NULL, "topo");
        rj_op op = rj_op_encode();
        rj_jit *j1 = compile_det(t, &op);
        rj_jit *j2 = compile_det(t, &op);
        const char *id1 = rj_plan_id(j1);
        const char *id2 = rj_plan_id(j2);
        CHECK(id1 != NULL && id2 != NULL, "rj_plan_id null");
        CHECK(strncmp(id1, "rj2:", 4) == 0, "version prefix: %s", id1);
        CHECK(count_axes(id1) == 9, "9 axes: %s", id1);
        CHECK(strcmp(id1, id2) == 0,
              "ID not stable across recompiles:\n%s\n%s", id1, id2);
        /* topo axis carries the topology name */
        CHECK(strstr(id1, "raid6(8)#") != NULL, "topo axis: %s", id1);
        /* op axis */
        CHECK(strstr(id1, ":enc:") != NULL, "op axis: %s", id1);
        printf("raid6(8) enc ID: %s\n", id1);
        rj_free(j1);
        rj_free(j2);

        /* pipe twin token: format emits it, parser accepts it (I8 axis).
         * On x86-64 pipe never wins (gate closed), so synthesize via
         * splice: the token must parse; rj_compile_pinned must reject
         * it loudly for the x86 ISA (pipe is NEON/SVE/RVV-only),
         * proving the axis is wired end-to-end. */
        {
            const char *tw = strstr(id1, ":u8:-:");
            CHECK(tw != NULL, "twins axis loc");
            char pid[4096];
            size_t pre = (size_t)(tw - id1) + 4; /* through ":u8:" */
            CHECK(pre + 6 < sizeof(pid), "pid size");
            memcpy(pid, id1, pre);
            memcpy(pid + pre, "pipe", 4);
            /* tw+5 is ":<tail>"; append it to get ":u8:pipe:<tail>" */
            CHECK(snprintf(pid + pre + 4, sizeof(pid) - pre - 4, "%s",
                           tw + 5) < (int)(sizeof(pid) - pre - 4),
                  "pid tail");
            /* syntactic parse must accept the token */
            topo_int_t ti;
            char eb[256];
            CHECK(topo_int_build(&ti, t, eb, sizeof(eb)) == 0,
                  "topo_int_build");
            planid_spec_t spec;
            int pr = planid_parse(pid, &ti, topo_internal_name(t), &op,
                                  &spec, eb, sizeof(eb));
            CHECK(pr == 0 && spec.cand.pipe == 1,
                  "pipe token parse: pr=%d pipe=%d eb='%s'", pr,
                  spec.cand.pipe, eb);
            topo_int_free(&ti);
            /* pinned compile must reject loudly: pipe is ARM/RVV-only */
            char err[512];
            err[0] = '\0';
            rj_jit *pp = rj_compile_pinned(t, &op, NULL, pid, err,
                                           sizeof(err));
            CHECK(pp == NULL, "pipe ID accepted on x86-64");
            CHECK(strstr(err, "pipe") != NULL,
                  "pipe ISA rejection: '%s' pid='%s'", err, pid);
            if (pp)
                rj_free(pp);
        }

        /* parity update op axis */
        int ch[] = {2, 0};
        rj_op pu = rj_op_parity_update(ch, 2);
        rj_jit *jp = compile_det(t, &pu);
        const char *idp = rj_plan_id(jp);
        CHECK(idp && strstr(idp, ":pupd0+2:") != NULL, "pupd axis: %s",
              idp ? idp : "(null)");
        printf("raid6(8) pupd ID: %s\n", idp);
        rj_free(jp);
        rj_op_free(&pu);

        /* recover op axis carries failed set + rows */
        int fl[] = {3};
        rj_op rc = rj_op_recover(fl, 1);
        rj_jit *jr = compile_det(t, &rc);
        const char *idr = rj_plan_id(jr);
        CHECK(idr && strstr(idr, ":rec3.rows") != NULL, "rec axis: %s",
              idr ? idr : "(null)");
        printf("raid6(8) rec ID: %s\n", idr);
        rj_free(jr);
        rj_op_free(&rc);
        rj_op_free(&op);
        rj_topo_free(t);

        /* rj_plan_id(NULL) */
        CHECK(rj_plan_id(NULL) == NULL, "rj_plan_id(NULL) != NULL");
    }

    /* ============ Part 2: pinned round-trip ============ */
    {
        /* raid6(8) encode (HORNER), raid5(8) encode (XOR), rs(6,3)
         * encode, custom MULACC encode. */
        rj_topology *t1 = rj_topo_raid6(8);
        rj_topology *t2 = rj_topo_raid5(8);
        rj_topology *t3 = rj_topo_rs(6, 3);
        uint8_t m[4];
        for (int i = 0; i < 4; i++)
            m[i] = (uint8_t)(i * 7 + 1);
        rj_topology *t4 = rj_topo_custom(2, 2, m);
        rj_topology *tops[] = {t1, t2, t3, t4};
        const char *labels[] = {"raid6(8)", "raid5(8)", "rs(6,3)",
                                "custom(2,2)"};
        for (int ti = 0; ti < 4; ti++) {
            rj_op op = rj_op_encode();
            rj_jit *j = compile_det(tops[ti], &op);
            if (!j) {
                rj_op_free(&op);
                continue;
            }
            char id[4096];
            snprintf(id, sizeof(id), "%s", rj_plan_id(j));
            char err[256];
            /* pin twice: deterministic */
            rj_jit *p1 = rj_compile_pinned(tops[ti], &op, NULL, id, err,
                                          sizeof(err));
            CHECK(p1 != NULL, "%s pinned compile failed: %s", labels[ti],
                  err);
            if (p1) {
                CHECK(strcmp(rj_plan_id(p1), id) == 0,
                      "%s pinned ID round-trip:\n%s\n%s", labels[ti], id,
                      rj_plan_id(p1));
                rj_jit *p2 =
                    rj_compile_pinned(tops[ti], &op, NULL, id, err,
                                      sizeof(err));
                CHECK(p2 != NULL, "%s second pin failed: %s", labels[ti],
                      err);
                if (p2) {
                    CHECK(strcmp(rj_plan_id(p2), id) == 0,
                          "%s pin not deterministic", labels[ti]);
                    rj_free(p2);
                }
                /* functional agreement with the original unit */
                int nd = rj_topo_ndata(tops[ti]);
                int np = rj_topo_npar(tops[ti]);
                check_fn_agree(rj_get_fn(j), rj_get_fn(p1), nd, np);
                rj_free(p1);
            }
            rj_free(j);
            rj_op_free(&op);
        }
        /* recover + parity update round-trips */
        {
            int fl[] = {1};
            rj_op op = rj_op_recover(fl, 1);
            rj_jit *j = compile_det(t1, &op);
            char id[4096];
            snprintf(id, sizeof(id), "%s", rj_plan_id(j));
            char err[256];
            rj_jit *p =
                rj_compile_pinned(t1, &op, NULL, id, err, sizeof(err));
            CHECK(p != NULL, "recover pinned failed: %s", err);
            if (p) {
                CHECK(strcmp(rj_plan_id(p), id) == 0,
                      "recover pinned round-trip:\n%s\n%s", id,
                      rj_plan_id(p));
                /* recover: nin = survivors, nout = nfailed */
                check_fn_agree(rj_get_fn(j), rj_get_fn(p),
                               rj_topo_ndata(t1) + rj_topo_npar(t1) - 1, 1);
                rj_free(p);
            }
            rj_free(j);
            rj_op_free(&op);
        }
        {
            int ch[] = {5};
            rj_op op = rj_op_parity_update(ch, 1);
            rj_jit *j = compile_det(t1, &op);
            char id[4096];
            snprintf(id, sizeof(id), "%s", rj_plan_id(j));
            char err[256];
            rj_jit *p =
                rj_compile_pinned(t1, &op, NULL, id, err, sizeof(err));
            CHECK(p != NULL, "pupd pinned failed: %s", err);
            if (p) {
                CHECK(strcmp(rj_plan_id(p), id) == 0,
                      "pupd pinned round-trip");
                rj_free(p);
            }
            rj_free(j);
            rj_op_free(&op);
        }
        for (int ti = 0; ti < 4; ti++)
            rj_topo_free(tops[ti]);
    }

    /* ============ Part 3: loud failures ============ */
    {
        rj_topology *t = rj_topo_raid6(8);
        rj_op op = rj_op_encode();
        rj_jit *j = compile_det(t, &op);
        char id[4096];
        snprintf(id, sizeof(id), "%s", rj_plan_id(j));
        rj_free(j);

        char err[512];
        /* null args */
        CHECK(rj_compile_pinned(NULL, &op, NULL, id, err, sizeof(err)) ==
                  NULL,
              "null topo accepted");
        CHECK(rj_compile_pinned(t, NULL, NULL, id, err, sizeof(err)) ==
                  NULL,
              "null op accepted");
        CHECK(rj_compile_pinned(t, &op, NULL, NULL, err, sizeof(err)) ==
                  NULL,
              "null ID accepted");

        /* garbage IDs: each must fail with non-empty err, no crash */
        const char *bad[] = {
            "",
            "rj",
            "rj1",
            "rj1:",
            "rj1:a:b:c:d:e:f:g",       /* 8 axes */
            "rj1:a:b:c:d:e:f:g:h:i",    /* 10 axes */
            "rj0:rs(6,3)#00:enc:avx512:mulacc:gfni:u8:-:scalar", /* old */
            "rj2:rs(6,3)#00:enc:avx512:mulacc:gfni:u8:-:scalar", /* bad hash */
            "rj99:rs(6,3)#00:enc:avx512:mulacc:gfni:u8:-:scalar",
            "rx1:rs(6,3)#00:enc:avx512:mulacc:gfni:u8:-:scalar",
            "rj1",
        };
        for (unsigned i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
            err[0] = '\0';
            rj_jit *p =
                rj_compile_pinned(t, &op, NULL, bad[i], err, sizeof(err));
            CHECK(p == NULL, "garbage ID accepted: '%s'", bad[i]);
            CHECK(err[0] != '\0', "empty err for garbage ID '%s'", bad[i]);
            if (p)
                rj_free(p);
        }
        /* version mismatch must say so explicitly */
        {
            char vid[4096];
            snprintf(vid, sizeof(vid), "rj9%s", id + 3);
            err[0] = '\0';
            rj_jit *p =
                rj_compile_pinned(t, &op, NULL, vid, err, sizeof(err));
            CHECK(p == NULL, "version-mismatched ID accepted");
            CHECK(strstr(err, "version mismatch") != NULL,
                  "version mismatch not reported: '%s'", err);
            if (p)
                rj_free(p);
        }
        /* bad axes: splice into a valid ID by replacing one axis */
        {
            /* helper: rebuild the ID with axis k replaced */
            char axes[9][512];
            {
                const char *s = id;
                for (int k = 0; k < 9; k++) {
                    const char *e = strchr(s, ':');
                    size_t n = e ? (size_t)(e - s) : strlen(s);
                    CHECK(n < sizeof(axes[k]), "axis too long");
                    memcpy(axes[k], s, n);
                    axes[k][n] = '\0';
                    s = e ? e + 1 : s + n;
                }
            }
            struct {
                int axis;
                const char *val;
            } subs[] = {
                {3, "frobnicator"}, /* isa */
                {4, "wobble"},      /* kind */
                {5, "magic"},       /* mul */
                {6, "u0"},          /* unroll */
                {6, "u9"},
                {6, "u99999999999999999999999"},
                {7, "nt+bogus"},    /* twin */
                {7, "nt++ternlog"}, /* empty token */
                {7, "nt+nt"},       /* duplicate */
                {7, "pf0"},
                {7, "pf18446744073709551615"}, /* == (size_t)-1 */
                {7, "lmul3"},
                {7, "lmul16"},
                {8, "sideways"}, /* tail */
                {8, "notail+notail"},
                {8, "predtail+maskedtail"},
            };
            for (unsigned i = 0; i < sizeof(subs) / sizeof(subs[0]); i++) {
                char nid[4608];
                const char *ax[9];
                for (int k = 0; k < 9; k++)
                    ax[k] = axes[k];
                ax[subs[i].axis] = subs[i].val;
                snprintf(nid, sizeof(nid), "%s:%s:%s:%s:%s:%s:%s:%s:%s",
                         ax[0], ax[1], ax[2], ax[3], ax[4], ax[5], ax[6],
                         ax[7], ax[8]);
                err[0] = '\0';
                rj_jit *p = rj_compile_pinned(t, &op, NULL, nid, err,
                                              sizeof(err));
                CHECK(p == NULL, "bad axis accepted: '%s'", nid);
                CHECK(err[0] != '\0', "empty err for bad axis '%s'", nid);
                if (p)
                    rj_free(p);
            }
        }
        /* topology mismatch: different dims, same name shape */
        {
            rj_topology *t4 = rj_topo_raid6(4);
            err[0] = '\0';
            rj_jit *p =
                rj_compile_pinned(t4, &op, NULL, id, err, sizeof(err));
            CHECK(p == NULL, "topology-mismatched ID accepted");
            CHECK(err[0] != '\0', "empty err for topo mismatch");
            if (p)
                rj_free(p);
            rj_topo_free(t4);
        }
        /* topology mismatch: same name+dims, different matrix */
        {
            uint8_t m1[16], m2[16];
            for (int i = 0; i < 16; i++) {
                m1[i] = (uint8_t)(i + 1);
                m2[i] = (uint8_t)(i + 2);
            }
            rj_topology *c1 = rj_topo_custom(8, 2, m1);
            rj_topology *c2 = rj_topo_custom(8, 2, m2);
            rj_op o2 = rj_op_encode();
            rj_jit *jc = compile_det(c1, &o2);
            char cid[4096];
            snprintf(cid, sizeof(cid), "%s", rj_plan_id(jc));
            rj_free(jc);
            err[0] = '\0';
            rj_jit *p =
                rj_compile_pinned(c2, &o2, NULL, cid, err, sizeof(err));
            CHECK(p == NULL, "same-name different-matrix ID accepted");
            CHECK(strstr(err, "fingerprint") != NULL,
                  "matrix mismatch not reported: '%s'", err);
            if (p)
                rj_free(p);
            rj_op_free(&o2);
            rj_topo_free(c1);
            rj_topo_free(c2);
        }
        /* op mismatch: encode ID + recover op, and reverse */
        {
            int fl[] = {0};
            rj_op rop = rj_op_recover(fl, 1);
            err[0] = '\0';
            rj_jit *p =
                rj_compile_pinned(t, &rop, NULL, id, err, sizeof(err));
            CHECK(p == NULL, "op-mismatched ID accepted (enc/recover)");
            CHECK(err[0] != '\0', "empty err for op mismatch");
            if (p)
                rj_free(p);
            rj_op_free(&rop);
        }
        {
            int fl[] = {2};
            rj_op rop = rj_op_recover(fl, 1);
            rj_jit *jr = compile_det(t, &rop);
            char rid[4096];
            snprintf(rid, sizeof(rid), "%s", rj_plan_id(jr));
            rj_free(jr);
            /* same op kind, different failed set */
            int fl2[] = {5};
            rj_op rop2 = rj_op_recover(fl2, 1);
            err[0] = '\0';
            rj_jit *p =
                rj_compile_pinned(t, &rop2, NULL, rid, err, sizeof(err));
            CHECK(p == NULL, "failed-set-mismatched ID accepted");
            CHECK(err[0] != '\0', "empty err for failed-set mismatch");
            if (p)
                rj_free(p);
            rj_op_free(&rop2);
            /* recover ID + encode op */
            err[0] = '\0';
            p = rj_compile_pinned(t, &op, NULL, rid, err, sizeof(err));
            CHECK(p == NULL, "op-mismatched ID accepted (recover/enc)");
            if (p)
                rj_free(p);
            rj_op_free(&rop);
        }
        rj_op_free(&op);
        rj_topo_free(t);
    }

    /* ============ Part 4: mutation ============ */
    {
        rj_topology *t = rj_topo_raid6(8);
        rj_op op = rj_op_encode();
        rj_jit *j = compile_det(t, &op);
        char id[4096];
        snprintf(id, sizeof(id), "%s", rj_plan_id(j));
        rj_free(j);
        size_t idlen = strlen(id);
        const char corrupt[] = {';', 'X', '9', '+', ' ', '\0'};
        char err[512];
        int nmut = 0;
        for (size_t pos = 0; pos < idlen; pos++) {
            for (unsigned ci = 0; corrupt[ci]; ci++) {
                char mid[4096];
                memcpy(mid, id, idlen + 1);
                if (mid[pos] == corrupt[ci])
                    continue;
                mid[pos] = corrupt[ci];
                nmut++;
                /* classify at the parse level (cheap, internal) */
                topo_int_t ti;
                char eb[256];
                CHECK(topo_int_build(&ti, t, eb, sizeof(eb)) == 0,
                      "topo_int_build");
                planid_spec_t spec;
                int pr = planid_parse(mid, &ti, topo_internal_name(t),
                                      &op, &spec, eb, sizeof(eb));
                topo_int_free(&ti);
                err[0] = '\0';
                rj_jit *p = rj_compile_pinned(t, &op, NULL, mid, err,
                                              sizeof(err));
                if (pr != 0) {
                    /* unparseable: pinned must fail loudly */
                    CHECK(p == NULL,
                          "unparseable mutated ID accepted: '%s'", mid);
                    CHECK(err[0] != '\0',
                          "empty err for mutated ID '%s'", mid);
                    if (p)
                        rj_free(p);
                } else {
                    /* parses: pinned either names a real (different)
                     * spec -- then the round-trip must report the
                     * mutated ID honestly -- or fails loudly (e.g.
                     * topo hash mismatch, unbuildable candidate). */
                    if (p != NULL) {
                        CHECK(strcmp(rj_plan_id(p), mid) == 0,
                              "mutated ID round-trip dishonest:\n%s\n%s",
                              mid, rj_plan_id(p));
                        rj_free(p);
                    } else {
                        CHECK(err[0] != '\0',
                              "empty err for mutated ID '%s'", mid);
                    }
                }
            }
        }
        /* truncations */
        for (size_t cut = 10; cut < idlen; cut += 10) {
            char mid[4096];
            memcpy(mid, id, cut);
            mid[cut] = '\0';
            err[0] = '\0';
            rj_jit *p =
                rj_compile_pinned(t, &op, NULL, mid, err, sizeof(err));
            CHECK(p == NULL, "truncated ID accepted: '%s'", mid);
            CHECK(err[0] != '\0', "empty err for truncated ID");
            if (p)
                rj_free(p);
        }
        rj_op_free(&op);
        rj_topo_free(t);
    }

    /* ============ Part 5: rj_list_variants ============ */
    {
        rj_topology *t = rj_topo_raid6(8);
        rj_op op = rj_op_encode();
        rj_options o;
        rj_options_default(&o);
        o.benchmark = 0;

        /* count-only agrees with fill */
        int total0 = rj_list_variants(t, op, &o, NULL, 0);
        CHECK(total0 > 0 && total0 < 1000000,
              "list_variants count-only failed: %d", total0);
        if (total0 <= 0 || total0 >= 1000000)
            total0 = 1; /* keep the alloc below well-defined; the CHECK
                         * above already failed the test */

        /* == plan_candidates count (the exact pre-prune space) */
        {
            topo_int_t ti;
            char eb[256];
            CHECK(topo_int_build(&ti, t, eb, sizeof(eb)) == 0, "ti");
            plan_t pl;
            CHECK(plan_build(&ti, &op, topo_internal_name(t), &pl, eb,
                             sizeof(eb)) == 0,
                  "plan_build: %s", eb);
            rj_cpu_info cpu;
            rj_cpu_detect(&cpu);
            cand_t cands[512];
            int nc = plan_candidates(&pl, &cpu, cands, 512, 0, 0);
            CHECK(nc == total0,
                  "list_variants total %d != plan_candidates %d", total0,
                  nc);
            plan_free(&pl);
            topo_int_free(&ti);
        }

        rj_plan_variant *vars =
            calloc((size_t)total0, sizeof(*vars));
        CHECK(vars != NULL, "oom");
        int total1 = rj_list_variants(t, op, &o, vars, total0);
        CHECK(total1 == total0, "fill total %d != count %d", total1,
              total0);
        /* every ID parses, round-trips, and is unique */
        for (int i = 0; i < total0; i++) {
            CHECK(count_axes(vars[i].plan_id) == 9, "entry %d axes: %s",
                  i, vars[i].plan_id);
            topo_int_t ti;
            char eb[256];
            CHECK(topo_int_build(&ti, t, eb, sizeof(eb)) == 0, "ti");
            planid_spec_t spec;
            int pr = planid_parse(vars[i].plan_id, &ti,
                                  topo_internal_name(t), &op, &spec, eb,
                                  sizeof(eb));
            topo_int_free(&ti);
            CHECK(pr == 0, "entry %d ID does not parse: %s (%s)", i,
                  vars[i].plan_id, eb);
            if (pr == 0) {
                char rt[4096];
                planid_tail_t no_tail = {0, 0, 0};
                /* re-format from the parsed spec: must round-trip.
                 * (needs the plan for the kind axis: rebuild it) */
                topo_int_t ti2;
                CHECK(topo_int_build(&ti2, t, eb, sizeof(eb)) == 0,
                      "ti2");
                plan_t pl;
                CHECK(plan_build(&ti2, &op, topo_internal_name(t), &pl,
                                 eb, sizeof(eb)) == 0,
                      "plan_build");
                planid_opdesc_t opd;
                CHECK(planid_opdesc_build(&opd, &ti2, &op, NULL, 0, eb,
                                          sizeof(eb)) == 0,
                      "opdesc");
                int fr = planid_format(rt, sizeof(rt), &ti2,
                                       topo_internal_name(t), &opd, &pl,
                                       &spec.cand, &no_tail);
                plan_free(&pl);
                topo_int_free(&ti2);
                CHECK(fr == 0, "re-format failed");
                CHECK(strcmp(rt, vars[i].plan_id) == 0,
                      "ID round-trip mismatch:\n%s\n%s",
                      vars[i].plan_id, rt);
            }
            for (int k = 0; k < i; k++)
                CHECK(strcmp(vars[i].plan_id, vars[k].plan_id) != 0,
                      "duplicate ID at entries %d,%d", k, i);
            /* benchmark=0: no static pruning; only build failures */
            if (!vars[i].survivor)
                CHECK(strncmp(vars[i].reason, "build failed", 12) == 0,
                      "entry %d pruned without build failure (bench=0): "
                      "%s",
                      i, vars[i].reason);
            else
                CHECK(vars[i].reason[0] == '\0',
                      "survivor has reason: %s", vars[i].reason);
        }
        free(vars);

        /* short nout: fills nout, still returns total */
        {
            int small = total0 > 3 ? 3 : total0;
            rj_plan_variant *vs2 = calloc((size_t)small, sizeof(*vs2));
            int tr = rj_list_variants(t, op, &o, vs2, small);
            CHECK(tr == total0, "short-nout total %d != %d", tr, total0);
            free(vs2);
        }

        /* benchmark=1 (default opts): static pruning may fire; every
         * pruned entry has a reason, and every +ternlog-pruned entry
         * has a surviving +ternlog twin entry. */
        {
            rj_options od;
            rj_options_default(&od); /* benchmark=1 */
            int tb = rj_list_variants(t, op, &od, NULL, 0);
            CHECK(tb > 0 && tb < 1000000, "bench=1 list failed: %d", tb);
            if (tb <= 0 || tb >= 1000000)
                tb = 1;
            rj_plan_variant *vb = calloc((size_t)tb, sizeof(*vb));
            int tb2 = rj_list_variants(t, op, &od, vb, tb);
            CHECK(tb2 == tb, "bench=1 fill total");
            int npruned = 0;
            for (int i = 0; i < tb; i++) {
                if (!vb[i].survivor) {
                    npruned++;
                    CHECK(vb[i].reason[0] != '\0',
                          "pruned entry %d has empty reason", i);
                    if (strstr(vb[i].reason, "ternlog") != NULL) {
                        /* find the surviving twin: same ID with
                         * +ternlog in the twins axis */
                        char want[4096];
                        snprintf(want, sizeof(want), "%s",
                                 vb[i].plan_id);
                        /* insert "+ternlog" after the 7th ':' */
                        char *axes[9];
                        {
                            char *s = want;
                            for (int k = 0; k < 9; k++) {
                                axes[k] = s;
                                char *e = strchr(s, ':');
                                if (e) {
                                    *e = '\0';
                                    s = e + 1;
                                }
                            }
                        }
                        char twinid[4096];
                        if (strcmp(axes[7], "-") == 0)
                            snprintf(twinid, sizeof(twinid),
                                     "%s:%s:%s:%s:%s:%s:%s:ternlog:%s",
                                     axes[0], axes[1], axes[2], axes[3],
                                     axes[4], axes[5], axes[6], axes[8]);
                        else
                            snprintf(twinid, sizeof(twinid),
                                     "%s:%s:%s:%s:%s:%s:%s:%s+ternlog:%s",
                                     axes[0], axes[1], axes[2], axes[3],
                                     axes[4], axes[5], axes[6], axes[7],
                                     axes[8]);
                        int found = 0;
                        for (int k = 0; k < tb; k++)
                            if (strcmp(vb[k].plan_id, twinid) == 0) {
                                found = 1;
                                CHECK(vb[k].survivor,
                                      "dominating twin not a survivor");
                                break;
                            }
                        CHECK(found,
                              "no surviving +ternlog twin for pruned "
                              "entry %d",
                              i);
                    }
                }
            }
            printf("bench=1: %d variants, %d pruned\n", tb, npruned);
            free(vb);
        }

        /* recover: survivor path enumerates subsets */
        {
            int fl[] = {0};
            rj_op rop = rj_op_recover(fl, 1);
            int tr = rj_list_variants(t, rop, &o, NULL, 0);
            CHECK(tr > 0 && tr < 1000000, "recover list failed: %d", tr);
            if (tr <= 0 || tr >= 1000000)
                tr = 1;
            rj_plan_variant *vr = calloc((size_t)tr, sizeof(*vr));
            int tr2 = rj_list_variants(t, rop, &o, vr, tr);
            CHECK(tr2 == tr, "recover fill total");
            for (int i = 0; i < tr; i++) {
                CHECK(strstr(vr[i].plan_id, ".rows") != NULL,
                      "recover entry %d lacks rows axis: %s", i,
                      vr[i].plan_id);
                topo_int_t ti;
                char eb[256];
                CHECK(topo_int_build(&ti, t, eb, sizeof(eb)) == 0,
                      "ti");
                planid_spec_t spec;
                int pr = planid_parse(vr[i].plan_id, &ti,
                                      topo_internal_name(t), &rop, &spec,
                                      eb, sizeof(eb));
                topo_int_free(&ti);
                CHECK(pr == 0, "recover entry %d unparseable: %s (%s)",
                      i, vr[i].plan_id, eb);
            }
            free(vr);
            rj_op_free(&rop);
        }

        /* invalid input */
        CHECK(rj_list_variants(NULL, op, &o, NULL, 0) == -1,
              "null topo accepted");

        rj_op_free(&op);
        rj_topo_free(t);
    }

    /* ============ Part 6: benchmark=1 end-to-end (tail twins) ============ */
    {
        /* One full rj_compile with benchmarking: the ID may name stage-2
         * tail twins. Pinned must reproduce them exactly (round-trip),
         * and agree functionally at tail-heavy lengths. */
        rj_topology *t = rj_topo_raid6(8);
        rj_op op = rj_op_encode();
        rj_options o;
        rj_options_default(&o); /* benchmark=1 */
        char err[256];
        rj_jit *j = rj_compile(t, &op, &o, err, sizeof(err));
        CHECK(j != NULL, "benchmarked compile failed: %s", err);
        if (j) {
            char id[4096];
            snprintf(id, sizeof(id), "%s", rj_plan_id(j));
            printf("benchmarked ID: %s\n", id);
            rj_jit *p =
                rj_compile_pinned(t, &op, NULL, id, err, sizeof(err));
            CHECK(p != NULL, "pinned (bench=1 ID) failed: %s", err);
            if (p) {
                CHECK(strcmp(rj_plan_id(p), id) == 0,
                      "bench=1 pinned round-trip:\n%s\n%s", id,
                      rj_plan_id(p));
                check_fn_agree(rj_get_fn(j), rj_get_fn(p),
                               rj_topo_ndata(t), rj_topo_npar(t));
                rj_free(p);
            }
            rj_free(j);
        }
        rj_op_free(&op);
        rj_topo_free(t);
    }

    if (nfail == 0)
        printf("test_planid: all OK\n");
    else
        printf("test_planid: %d FAILURES\n", nfail);
    return nfail != 0;
}
