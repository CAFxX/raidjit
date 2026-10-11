/* Length-specialized runtime dispatch (docs/length_dispatch.md §9).
 *
 * 1. White-box selection (no JIT): the admission predicate, variant
 *    sorting, dispatcher routing to recording mock variants (pow2 and
 *    non-pow2 groups), single-variant dispatchers, create-arg validation,
 *    and slot-exhaustion behavior.
 * 2. Integration (JIT): for several topology x op configs, build the
 *    general version and its notail twin directly via jit_build_version
 *    (known group), and check both against the scalar reference over a
 *    length matrix hitting group boundaries. Includes the NT x notail
 *    combination.
 * 3. End-to-end (rj_compile): dense length matrix vs the reference for
 *    encode / recover / parity-update x {benchmark on/off, NT on/off};
 *    bit-agreement between dispatched and general-only compiles.
 * 4. Informational: per-call dispatcher overhead (ns/call, printed only).
 *
 * Mutation / red-green evidence (run manually, documented here):
 *  (a) dispatch.c: force dispatch_call to always take variants[0] ->
 *      white-box routing tests fail (len=257 routed to the notail mock).
 *  (b) x86_64/jit.c: in the notail branch, bind the loop exit to tail_lab
 *      instead of done_lab (i.e. emit the general shape) -> the twin's
 *      desc still says " notail" but validation of the *omitted* tail
 *      path is unaffected; instead flip cg->notail handling to skip
 *      gen_body's last store -> integration matrix fails vs ref.
 *      (Simpler red-green: build the twin, then flip one byte of its
 *      emitted body via mprotect RW -> matrix fails.)
 *  (c) rj_variant_admits: the TAIL_NONE group==0 and non-pow2 cases are
 *      covered by direct asserts below (red: return 1 unconditionally).
 *  (d) rj_variant_admits TAIL_MASKED case forced to `return 1` (admits
 *      everything): the "masked rejects len 0/640/1280" asserts fail
 *      (verified 2026-10-03, closeout).
 */
#include "../include/raidjit.h"
#include "../src/plan.h"
#include "../src/jit.h"
#include "../src/bench.h"
#include "../src/cpu.h"
#include "../src/dispatch.h"
#include "../src/ref.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

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

static uint64_t rng_state = 0x51ed270b23a7c1d3ULL;
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

/* ---------------- 1. white-box selection ---------------- */

static int mock_calls[4];
static void mock_fn(int idx, const void *in[], void *out[], size_t len)
{
    (void)in;
    (void)out;
    (void)len;
    mock_calls[idx]++;
}
static void mock0(const void *in[], void *out[], size_t len)
{
    mock_fn(0, in, out, len);
}
static void mock1(const void *in[], void *out[], size_t len)
{
    mock_fn(1, in, out, len);
}
static void mock2(const void *in[], void *out[], size_t len)
{
    mock_fn(2, in, out, len);
}
static void mock3(const void *in[], void *out[], size_t len)
{
    mock_fn(3, in, out, len);
}

static void test_admits(void)
{
    rj_variant v;
    memset(&v, 0, sizeof(v));

    v.tail = TAIL_SCALAR;
    CHECK(rj_variant_admits(&v, 0) == 1, "scalar admits len 0");
    CHECK(rj_variant_admits(&v, 1) == 1, "scalar admits len 1");
    CHECK(rj_variant_admits(&v, 1234567) == 1, "scalar admits odd len");

    /* pow2 group: & fast path */
    v.tail = TAIL_NONE;
    v.group = 256;
    rj_variant_derive(&v);
    CHECK(v.group_pow2 == 1 && v.group_mask == 255, "derive pow2");
    CHECK(rj_variant_admits(&v, 0) == 1, "notail/256 admits 0");
    CHECK(rj_variant_admits(&v, 256) == 1, "notail/256 admits 256");
    CHECK(rj_variant_admits(&v, 512) == 1, "notail/256 admits 512");
    CHECK(rj_variant_admits(&v, 1) == 0, "notail/256 rejects 1");
    CHECK(rj_variant_admits(&v, 255) == 0, "notail/256 rejects 255");
    CHECK(rj_variant_admits(&v, 257) == 0, "notail/256 rejects 257");

    /* non-pow2 group: % path */
    v.group = 96;
    rj_variant_derive(&v);
    CHECK(v.group_pow2 == 0, "derive non-pow2");
    CHECK(rj_variant_admits(&v, 0) == 1, "notail/96 admits 0");
    CHECK(rj_variant_admits(&v, 96) == 1, "notail/96 admits 96");
    CHECK(rj_variant_admits(&v, 192) == 1, "notail/96 admits 192");
    CHECK(rj_variant_admits(&v, 95) == 0, "notail/96 rejects 95");
    CHECK(rj_variant_admits(&v, 97) == 0, "notail/96 rejects 97");
    CHECK(rj_variant_admits(&v, 100) == 0, "notail/96 rejects 100");

    /* degenerate group: admit nothing */
    v.group = 0;
    rj_variant_derive(&v);
    CHECK(rj_variant_admits(&v, 0) == 0, "notail/0 admits nothing");
    CHECK(rj_variant_admits(&v, 256) == 0, "notail/0 admits nothing (256)");

    /* TAIL_MASKED (tracks B1sve/B1x86): admits ONLY tail lengths
     * (len % group != 0). The masked twin owns tails; when one is
     * installed the scalar general variant defers to it
     * (test_length_aware_routing). */
    v.tail = TAIL_MASKED;
    v.group = 640;
    rj_variant_derive(&v);
    CHECK(rj_variant_admits(&v, 0) == 0, "masked rejects len 0");
    CHECK(rj_variant_admits(&v, 1) == 1, "masked admits len 1");
    CHECK(rj_variant_admits(&v, 639) == 1, "masked admits len 639");
    CHECK(rj_variant_admits(&v, 640) == 0, "masked rejects len 640");
    CHECK(rj_variant_admits(&v, 1280) == 0, "masked rejects len 1280");
    CHECK(rj_variant_admits(&v, 1234567) == 1, "masked admits odd len");
    /* degenerate group: admits nothing (cannot test len % 0) */
    v.group = 0;
    rj_variant_derive(&v);
    CHECK(rj_variant_admits(&v, 0) == 0, "masked/0 admits nothing");
    CHECK(rj_variant_admits(&v, 1) == 0, "masked/0 admits nothing (1)");

    /* TAIL_SCALAR_MULT (derived at create time when a masked twin is
     * installed): scalar code, but tail lengths defer to the masked
     * twin, so only multiples admit. */
    v.tail = TAIL_SCALAR_MULT;
    v.group = 640;
    rj_variant_derive(&v);
    CHECK(rj_variant_admits(&v, 0) == 1, "scalar-mult admits len 0");
    CHECK(rj_variant_admits(&v, 640) == 1, "scalar-mult admits 640");
    CHECK(rj_variant_admits(&v, 1280) == 1, "scalar-mult admits 1280");
    CHECK(rj_variant_admits(&v, 1) == 0, "scalar-mult rejects len 1");
    CHECK(rj_variant_admits(&v, 639) == 0, "scalar-mult rejects 639");
    CHECK(rj_variant_admits(&v, 641) == 0, "scalar-mult rejects 641");
}

static void test_sort(void)
{
    rj_variant vs[3];
    memset(vs, 0, sizeof(vs));
    vs[0].gbps = 10;
    vs[1].gbps = 50;
    vs[2].gbps = 30;
    rj_variant_sort(vs, 3);
    CHECK(vs[0].gbps == 50 && vs[1].gbps == 30 && vs[2].gbps == 10,
          "sort fastest-first");
    rj_variant_sort(vs, 1); /* single: no-op, must not crash */
    CHECK(vs[0].gbps == 50, "sort single keeps element");
}

/* Build a 2-variant dispatcher from recording mocks; variants passed
 * unsorted to prove create() enforces fastest-first itself. Returns the
 * dispatcher (caller destroys). */
static rj_dispatcher *mk2(size_t group, double gbps_notail,
                           double gbps_general)
{
    rj_variant vs[2];
    memset(vs, 0, sizeof(vs));
    vs[0].fn = mock1; /* general, deliberately first + slower */
    vs[0].tail = TAIL_SCALAR;
    vs[0].gbps = gbps_general;
    vs[1].fn = mock0; /* notail, faster */
    vs[1].tail = TAIL_NONE;
    vs[1].group = group;
    vs[1].gbps = gbps_notail;
    rj_variant_derive(&vs[0]);
    rj_variant_derive(&vs[1]);
    return rj_dispatcher_create(vs, 2);
}

static void test_routing(void)
{
    /* A dispatcher with a TAIL_MASKED tail-server and a TAIL_NONE
     * multiple-server (no TAIL_SCALAR): the set covers every len, so
     * create must accept it. */
    {
        rj_variant vs[2];
        memset(vs, 0, sizeof(vs));
        vs[0].fn = mock1;
        vs[0].tail = TAIL_MASKED;
        vs[0].group = 640;
        vs[0].gbps = 60.0;
        vs[1].fn = mock0;
        vs[1].tail = TAIL_NONE;
        vs[1].group = 640;
        vs[1].gbps = 100.0;
        rj_variant_derive(&vs[0]);
        rj_variant_derive(&vs[1]);
        rj_dispatcher *d = rj_dispatcher_create(vs, 2);
        CHECK(d != NULL, "create accepts {masked, notail} cover-every-len");
        if (d) {
            /* fastest-first: notail (100) before masked (60) */
            CHECK(rj_dispatcher_pick(d, 640) == 0, "masked pick(640)->notail");
            CHECK(rj_dispatcher_pick(d, 1) == 1, "masked pick(1)->masked");
            CHECK(rj_dispatcher_pick(d, 639) == 1, "masked pick(639)->masked");
            rj_dispatcher_destroy(d);
        }
    }
    /* pow2 group */
    rj_dispatcher *d = mk2(256, 100.0, 50.0);
    CHECK(d != NULL, "create 2-variant dispatcher");
    CHECK(rj_dispatcher_nvar(d) == 2, "nvar==2");
    /* fastest-first after internal sort: index 0 == notail mock0 */
    CHECK(rj_dispatcher_pick(d, 0) == 0, "pick(0) -> notail");
    CHECK(rj_dispatcher_pick(d, 256) == 0, "pick(256) -> notail");
    CHECK(rj_dispatcher_pick(d, 512) == 0, "pick(512) -> notail");
    CHECK(rj_dispatcher_pick(d, 1) == 1, "pick(1) -> general");
    CHECK(rj_dispatcher_pick(d, 255) == 1, "pick(255) -> general");
    CHECK(rj_dispatcher_pick(d, 257) == 1, "pick(257) -> general");
    CHECK(rj_dispatcher_pick(d, 100000) == 1, "pick(100000) -> general");

    memset(mock_calls, 0, sizeof(mock_calls));
    rj_fn f = rj_dispatcher_fn(d);
    CHECK(f != NULL, "dispatcher fn non-null");
    f(NULL, NULL, 256);
    CHECK(mock_calls[0] == 1 && mock_calls[1] == 0,
          "call len=256 -> notail mock");
    f(NULL, NULL, 257);
    CHECK(mock_calls[0] == 1 && mock_calls[1] == 1,
          "call len=257 -> general mock");
    f(NULL, NULL, 0);
    CHECK(mock_calls[0] == 2 && mock_calls[1] == 1,
          "call len=0 -> notail mock");
    rj_dispatcher_destroy(d);

    /* non-pow2 group exercises the len % group path */
    d = mk2(96, 100.0, 50.0);
    CHECK(d != NULL, "create non-pow2 dispatcher");
    CHECK(rj_dispatcher_pick(d, 192) == 0, "pick(192) -> notail/96");
    CHECK(rj_dispatcher_pick(d, 100) == 1, "pick(100) -> general/96");
    memset(mock_calls, 0, sizeof(mock_calls));
    rj_dispatcher_fn(d)(NULL, NULL, 192);
    CHECK(mock_calls[0] == 1 && mock_calls[1] == 0,
          "call len=192 -> notail mock");
    rj_dispatcher_fn(d)(NULL, NULL, 100);
    CHECK(mock_calls[0] == 1 && mock_calls[1] == 1,
          "call len=100 -> general mock");
    rj_dispatcher_destroy(d);

    /* single general variant: everything routes to it */
    rj_variant g;
    memset(&g, 0, sizeof(g));
    g.fn = mock2;
    g.tail = TAIL_SCALAR;
    g.gbps = 10.0;
    rj_variant_derive(&g);
    d = rj_dispatcher_create(&g, 1);
    CHECK(d != NULL, "create single-variant dispatcher");
    CHECK(rj_dispatcher_pick(d, 0) == 0, "single: pick(0)");
    CHECK(rj_dispatcher_pick(d, 12345) == 0, "single: pick(12345)");
    memset(mock_calls, 0, sizeof(mock_calls));
    rj_dispatcher_fn(d)(NULL, NULL, 77);
    CHECK(mock_calls[2] == 1, "single: call routes to the variant");
    rj_dispatcher_destroy(d);
}

/* Length-aware tail-twin routing (docs/length_dispatch.md §12; track
 * B1x86, generalized for B1sve): when a TAIL_MASKED twin is installed,
 * it owns tail lengths and the scalar general variant defers to it --
 * deterministically, regardless of measured GB/s. Red-green: give the
 * general variant the HIGHER gbps; without the deferral the old
 * fastest-first-among-admitters routing would send every length to
 * general (TAIL_MASKED admitted everything), so the masked-pick asserts
 * fail on the old code. */
static void test_length_aware_routing(void)
{
    /* {general(scalar, faster), masked(slower)}: tails must still route
     * to the masked twin deterministically. */
    rj_variant vs[2];
    memset(vs, 0, sizeof(vs));
    vs[0].fn = mock1; /* general, faster */
    vs[0].tail = TAIL_SCALAR;
    vs[0].group = 512;
    vs[0].gbps = 100.0;
    vs[1].fn = mock0; /* masked twin, slower but owns tails */
    vs[1].tail = TAIL_MASKED;
    vs[1].group = 512;
    vs[1].gbps = 50.0;
    rj_variant_derive(&vs[0]);
    rj_variant_derive(&vs[1]);
    rj_dispatcher *d = rj_dispatcher_create(vs, 2);
    CHECK(d != NULL, "create {general, masked}");
    if (d) {
        /* fastest-first: general (100) is index 0, masked (50) index 1 */
        CHECK(rj_dispatcher_pick(d, 512) == 0, "defer pick(512)->general");
        CHECK(rj_dispatcher_pick(d, 1024) == 0, "defer pick(1024)->general");
        CHECK(rj_dispatcher_pick(d, 0) == 0, "defer pick(0)->general");
        CHECK(rj_dispatcher_pick(d, 1) == 1, "defer pick(1)->masked");
        CHECK(rj_dispatcher_pick(d, 511) == 1, "defer pick(511)->masked");
        CHECK(rj_dispatcher_pick(d, 513) == 1, "defer pick(513)->masked");
        CHECK(rj_dispatcher_pick(d, 100000) == 1, "defer pick(100000)->masked");
        /* and through the real hot path, not just the white-box hook */
        memset(mock_calls, 0, sizeof(mock_calls));
        rj_fn f = rj_dispatcher_fn(d);
        f(NULL, NULL, 7);    /* tail -> masked (mock0) */
        f(NULL, NULL, 512);  /* multiple -> general (mock1) */
        f(NULL, NULL, 1000); /* tail -> masked */
        CHECK(mock_calls[0] == 2 && mock_calls[1] == 1,
              "defer calls route deterministically (masked=%d general=%d)",
              mock_calls[0], mock_calls[1]);
        rj_dispatcher_destroy(d);
    }

    /* {general, notail, masked}: multiples go fastest-first among
     * {general, notail}; tails go to masked. */
    rj_variant vs3[3];
    memset(vs3, 0, sizeof(vs3));
    vs3[0].fn = mock1; /* general */
    vs3[0].tail = TAIL_SCALAR;
    vs3[0].group = 512;
    vs3[0].gbps = 80.0;
    vs3[1].fn = mock2; /* notail, fastest at multiples */
    vs3[1].tail = TAIL_NONE;
    vs3[1].group = 512;
    vs3[1].gbps = 90.0;
    vs3[2].fn = mock0; /* masked twin */
    vs3[2].tail = TAIL_MASKED;
    vs3[2].group = 512;
    vs3[2].gbps = 70.0;
    for (int i = 0; i < 3; i++)
        rj_variant_derive(&vs3[i]);
    d = rj_dispatcher_create(vs3, 3);
    CHECK(d != NULL, "create {general, notail, masked}");
    if (d) {
        /* fastest-first: notail(90)=0, general(80)=1, masked(70)=2 */
        CHECK(rj_dispatcher_pick(d, 512) == 0, "3v pick(512)->notail");
        CHECK(rj_dispatcher_pick(d, 0) == 0, "3v pick(0)->notail");
        CHECK(rj_dispatcher_pick(d, 1) == 2, "3v pick(1)->masked");
        CHECK(rj_dispatcher_pick(d, 511) == 2, "3v pick(511)->masked");
        CHECK(rj_dispatcher_pick(d, 513) == 2, "3v pick(513)->masked");
        memset(mock_calls, 0, sizeof(mock_calls));
        rj_fn f = rj_dispatcher_fn(d);
        f(NULL, NULL, 512);
        f(NULL, NULL, 3);
        CHECK(mock_calls[2] == 1 && mock_calls[0] == 1,
              "3v calls: notail=%d masked=%d", mock_calls[2],
              mock_calls[0]);
        rj_dispatcher_destroy(d);
    }

    /* Legacy: no masked twin installed -> TAIL_SCALAR admits everything. */
    {
        rj_variant g;
        memset(&g, 0, sizeof(g));
        g.fn = mock3;
        g.tail = TAIL_SCALAR;
        g.group = 512;
        g.gbps = 10.0;
        rj_variant_derive(&g);
        rj_dispatcher *d1 = rj_dispatcher_create(&g, 1);
        CHECK(d1 != NULL, "create lone general");
        if (d1) {
            CHECK(rj_dispatcher_pick(d1, 1) == 0, "legacy pick(1)->general");
            CHECK(rj_dispatcher_pick(d1, 512) == 0,
                  "legacy pick(512)->general");
            rj_dispatcher_destroy(d1);
        }
    }
}

static void test_create_args(void)
{
    rj_variant g, n;
    memset(&g, 0, sizeof(g));
    memset(&n, 0, sizeof(n));
    g.fn = mock2;
    g.tail = TAIL_SCALAR;
    n.fn = mock3;
    n.tail = TAIL_NONE;
    n.group = 64;
    rj_variant_derive(&g);
    rj_variant_derive(&n);
    rj_variant vs[2] = { g, n };

    CHECK(rj_dispatcher_create(NULL, 1) == NULL, "NULL vars rejected");
    CHECK(rj_dispatcher_create(vs, 0) == NULL, "nvar=0 rejected");
    CHECK(rj_dispatcher_create(vs, RJ_DISPATCH_MAXV + 1) == NULL,
          "nvar>MAXV rejected");
    CHECK(rj_dispatcher_create(&n, 1) == NULL,
          "lone notail rejected (covers nothing alone)");
    /* lone masked twin: covers tails only, multiples uncovered */
    rj_variant m;
    memset(&m, 0, sizeof(m));
    m.fn = mock0;
    m.tail = TAIL_MASKED;
    m.group = 64;
    rj_variant_derive(&m);
    CHECK(rj_dispatcher_create(&m, 1) == NULL,
          "lone masked rejected (multiples uncovered)");
    /* TAIL_SCALAR_MULT is derived-only: callers must not set it */
    rj_variant sm = g;
    sm.tail = TAIL_SCALAR_MULT;
    CHECK(rj_dispatcher_create(&sm, 1) == NULL,
          "caller-set TAIL_SCALAR_MULT rejected");
    rj_variant bad = g;
    bad.fn = NULL;
    CHECK(rj_dispatcher_create(&bad, 1) == NULL, "NULL fn rejected");
    CHECK(rj_dispatcher_nvar(NULL) == 0, "nvar(NULL)==0");
    CHECK(rj_dispatcher_pick(NULL, 10) == -1, "pick(NULL)==-1");
    CHECK(rj_dispatcher_fn(NULL) == NULL, "fn(NULL)==NULL");
    rj_dispatcher_destroy(NULL); /* must not crash */
}

static void test_slot_exhaustion(void)
{
    rj_variant g;
    memset(&g, 0, sizeof(g));
    g.fn = mock2;
    g.tail = TAIL_SCALAR;
    rj_variant_derive(&g);
    rj_dispatcher *ds[RJ_DISPATCH_SLOTS];
    int i;
    for (i = 0; i < RJ_DISPATCH_SLOTS; i++) {
        ds[i] = rj_dispatcher_create(&g, 1);
        CHECK(ds[i] != NULL, "slot %d allocated", i);
        if (!ds[i])
            break;
    }
    CHECK(rj_dispatcher_create(&g, 1) == NULL,
          "slot exhaustion fails cleanly");
    for (int k = 0; k < i; k++)
        rj_dispatcher_destroy(ds[k]);
    /* slots are reusable after destroy */
    rj_dispatcher *d = rj_dispatcher_create(&g, 1);
    CHECK(d != NULL, "slot reusable after destroy");
    rj_dispatcher_destroy(d);
}

/* ---------------- 2. integration: direct twin builds ---------------- */

typedef struct {
    const char *name;
    rj_topology *(*mk_topo)(void);
    rj_op (*mk_op)(void);
    int allow_nt;   /* request NT twins from the planner */
    int want_nt;    /* pick an NT candidate for the twin */
    int want_prefetch;    /* pick a prefetch candidate for the twin */
    size_t prefetch_dist; /* requested from the planner when want_prefetch */
} dcfg_t;

static rj_topology *t_raid6_6(void) { return rj_topo_raid6(6); }
static rj_topology *t_raid5_4(void) { return rj_topo_raid5(4); }
static rj_topology *t_raid1(void) { return rj_topo_raid1(); }
static rj_topology *t_rs83(void) { return rj_topo_rs(8, 3); }
static rj_op o_enc(void) { return rj_op_encode(); }
static rj_op o_rec0(void)
{
    int f[1] = { 0 };
    return rj_op_recover(f, 1);
}
static rj_op o_pup1(void)
{
    int c[1] = { 1 };
    return rj_op_parity_update(c, 1);
}

/* Build general + notail twin for up to 3 buildable candidates of a
 * config (distinct U variants, not just the first); check both against
 * the scalar reference over a group-boundary length matrix. Includes the
 * NT x notail and prefetch x notail combinations. */
static void test_direct_twin(const dcfg_t *c)
{
    char err[256];
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);

    rj_topology *t = c->mk_topo();
    topo_int_t ti;
    CHECK(topo_int_build(&ti, t, err, sizeof(err)) == 0, "%s topo: %s",
          c->name, err);
    if (failures)
        goto out_t;
    rj_op op = c->mk_op();
    plan_t pl;
    CHECK(plan_build(&ti, &op, c->name, &pl, err, sizeof(err)) == 0,
          "%s plan: %s", c->name, err);
    if (failures) {
        rj_op_free(&op);
        goto out_ti;
    }
    if (pl.noop) {
        printf("note: %s is a noop plan, skipped\n", c->name);
        rj_op_free(&op);
        plan_free(&pl);
        goto out_ti;
    }

    cand_t cands[64];
    int ncand = plan_candidates(&pl, &cpu, cands, 64, c->allow_nt,
                                c->prefetch_dist);
    CHECK(ncand > 0, "%s: no candidates", c->name);

    if (c->want_prefetch) {
        int any_pf = 0;
        for (int i = 0; i < ncand; i++)
            if (cands[i].prefetch_dist) {
                any_pf = 1;
                break;
            }
        if (!any_pf) {
            /* RVV without Zicbop: the planner emits no prefetch twins
             * (the backend would ignore prefetch_dist), so there is no
             * prefetch candidate to twin. Skip, don't fail. */
            printf("note: %s: no prefetch candidates on this CPU, "
                   "skipped\n",
                   c->name);
            rj_op_free(&op);
            plan_free(&pl);
            goto out_ti;
        }
    }

    rj_options o;
    rj_options_default(&o);
    o.benchmark = 0;

    /* up to 3 buildable (general, twin) pairs: the matrix used to test
     * only the first buildable candidate, so a U-specific twin bug in a
     * later candidate would slip through. */
    version_t gens[3], twins[3];
    int npair = 0;
    for (int i = 0; i < ncand && npair < 3; i++) {
        if (c->want_nt && !cands[i].nt)
            continue;
        if (c->want_prefetch && !cands[i].prefetch_dist)
            continue;
        char e2[256];
        version_t gen, twin;
        memset(&gen, 0, sizeof(gen));
        memset(&twin, 0, sizeof(twin));
        if (jit_build_version(&pl, &cands[i], &cpu, &o, &gen, e2,
                              sizeof(e2)) != 0)
            continue;
        cand_t tc = cands[i];
        tc.notail = 1;
        if (jit_build_version(&pl, &tc, &cpu, &o, &twin, e2,
                              sizeof(e2)) != 0) {
            jit_free_version(&gen);
            continue;
        }
        gens[npair] = gen;
        twins[npair] = twin;
        npair++;
    }
    const char *want = c->want_nt ? " with NT"
                         : c->want_prefetch ? " with prefetch" : "";
    CHECK(npair > 0, "%s: no buildable candidate%s", c->name, want);
    if (npair == 0) {
        rj_op_free(&op);
        plan_free(&pl);
        goto out_ti;
    }

    int n_in = pl.n_in, n_out = pl.n_out;
    /* (65536/G)*G <= 65536 for every G, so a single bound covers all
     * pairs however their groups differ. (A per-pair max under-sizes
     * the buffers: a smaller group can round 65536 down LESS than the
     * largest group does, e.g. G=128 -> 65536 vs gmax=160 -> 65440 --
     * that heap overflow crashed the SVE QEMU run.) */
    size_t maxlen = 65536 + 64;

    uint8_t **ib = malloc(sizeof(*ib) * (size_t)n_in);
    uint8_t **ob = malloc(sizeof(*ob) * (size_t)n_out);
    uint8_t **rb = malloc(sizeof(*rb) * (size_t)n_out);
    const void **in = malloc(sizeof(*in) * (size_t)n_in);
    void **out = malloc(sizeof(*out) * (size_t)n_out);
    for (int i = 0; i < n_in; i++) {
        ib[i] = abuf(maxlen);
        in[i] = ib[i];
    }
    for (int j = 0; j < n_out; j++) {
        ob[j] = abuf(maxlen);
        rb[j] = abuf(maxlen);
        out[j] = ob[j];
    }

    for (int p = 0; p < npair; p++) {
        version_t *gen = &gens[p], *twin = &twins[p];
        CHECK(twin->notail == 1, "%s pair %d: twin flagged notail", c->name,
              p);
        size_t G = twin->group_bytes;
        CHECK(G > 0 && G <= 4096, "%s pair %d: sane group_bytes=%zu",
              c->name, p, G);
        CHECK(strstr(twin->desc, "notail") != NULL,
              "%s pair %d: twin desc tagged: %s", c->name, p, twin->desc);
        CHECK(gen->group_bytes == G, "%s pair %d: general/twin group agree",
              c->name, p);
        if (c->want_prefetch)
            CHECK(gen->cand.prefetch_dist == c->prefetch_dist,
                  "%s pair %d: twin carries prefetch_dist", c->name, p);

        size_t pBIG = (65536 / G) * G;
        size_t BIGNM = pBIG + 7;
        while (BIGNM % G == 0)
            BIGNM++;
        size_t lens[] = { 0,         1,     G - 1, G, G + 1,
                          2 * G - 1, 2 * G, 2 * G + 1, pBIG, BIGNM };
        int nmult[] = { 1, 0, 0, 1, 0, 0, 1, 0, 1, 0 }; /* twin-admissible */

        for (size_t li = 0; li < sizeof(lens) / sizeof(lens[0]); li++) {
            size_t len = lens[li];
            for (int i = 0; i < n_in; i++)
                randfill(ib[i], len);
            /* general: every length */
            for (int j = 0; j < n_out; j++)
                memset(ob[j], 0xA5, len);
            gen->fn(in, out, len);
            ref_run(&pl, (const uint8_t **)ib, rb, len);
            for (int j = 0; j < n_out; j++)
                CHECK(memcmp(ob[j], rb[j], len) == 0,
                      "%s pair %d general mismatch at len=%zu out=%d",
                      c->name, p, len, j);
            /* twin: only group multiples (its contract) */
            if (nmult[li]) {
                for (int j = 0; j < n_out; j++)
                    memset(ob[j], 0xA5, len);
                twin->fn(in, out, len);
                ref_run(&pl, (const uint8_t **)ib, rb, len);
                for (int j = 0; j < n_out; j++)
                    CHECK(memcmp(ob[j], rb[j], len) == 0,
                          "%s pair %d twin mismatch at len=%zu out=%d",
                          c->name, p, len, j);
            }
        }
    }

    for (int i = 0; i < n_in; i++)
        free(ib[i]);
    for (int j = 0; j < n_out; j++) {
        free(ob[j]);
        free(rb[j]);
    }
    free(ib);
    free(ob);
    free(rb);
    free(in);
    free(out);
    for (int p = 0; p < npair; p++) {
        jit_free_version(&gens[p]);
        jit_free_version(&twins[p]);
    }
    rj_op_free(&op);
    plan_free(&pl);
out_ti:
    topo_int_free(&ti);
out_t:
    rj_topo_free(t);
}

/* ---------------- 2b. twin-source mapping regression ---------------- */

/* The stage-2 twin must be built from the WINNER's candidate. vs[] skips
 * failed candidate builds, so a bench_select winner index must never be
 * used on cands[] (it once was: `cands[winner]`, pairing the dispatcher
 * with the wrong candidate's twin whenever any build failed). Builds only
 * fail on OOM, so this deterministically replicates api.c's select+twin
 * flow with a deliberately skipped index and pins the pairing invariant:
 * twin.desc == winner.desc + " notail". It also builds the twin the buggy
 * expression would have produced and asserts the two differ, proving the
 * test discriminates. */
static void test_twin_source_mapping(void)
{
    char err[256];
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    rj_topology *t = rj_topo_raid6(6);
    topo_int_t ti;
    CHECK(topo_int_build(&ti, t, err, sizeof(err)) == 0, "topo: %s", err);
    rj_op op = rj_op_encode();
    plan_t pl;
    CHECK(plan_build(&ti, &op, "map", &pl, err, sizeof(err)) == 0,
          "plan: %s", err);

    cand_t cands[64];
    int ncand = plan_candidates(&pl, &cpu, cands, 64, 0, 0);
    CHECK(ncand > 2, "need >2 candidates, got %d", ncand);

    rj_options o;
    rj_options_default(&o);
    o.benchmark = 1;
    o.bench_bytes = 1 << 16;
    o.bench_iters = 2;

    /* stage 1, mirroring api.c: skip index 0 (simulated failed build) */
    version_t vs[64];
    int nv = 0;
    for (int i = 1; i < ncand && nv < 64; i++) {
        char e2[256];
        if (jit_build_version(&pl, &cands[i], &cpu, &o, &vs[nv], e2,
                              sizeof(e2)) == 0)
            nv++;
    }
    CHECK(nv > 0, "no versions built");
    int winner = bench_select(vs, nv, &pl, &o, 0, NULL, err, sizeof(err));
    CHECK(winner >= 0, "bench_select: %s", err);

    /* stage 2 with the FIXED expression: twin of vs[winner].cand */
    cand_t twinc = vs[winner].cand;
    twinc.notail = 1;
    version_t twin;
    memset(&twin, 0, sizeof(twin));
    CHECK(jit_build_version(&pl, &twinc, &cpu, &o, &twin, err,
                            sizeof(err)) == 0,
          "twin build: %s", err);
    size_t wlen = strlen(vs[winner].desc);
    CHECK(strncmp(twin.desc, vs[winner].desc, wlen) == 0 &&
              strstr(twin.desc, "notail") != NULL,
          "twin pairs with winner: '%s' vs '%s'", twin.desc,
          vs[winner].desc);

    /* the buggy expression cands[winner]: with index 0 skipped, cands[w]
     * is a *different* candidate for every w -- the test must be able to
     * tell them apart. */
    cand_t badc = cands[winner];
    badc.notail = 1;
    version_t badtwin;
    memset(&badtwin, 0, sizeof(badtwin));
    if (jit_build_version(&pl, &badc, &cpu, &o, &badtwin, err,
                          sizeof(err)) == 0) {
        CHECK(strcmp(badtwin.desc, twin.desc) != 0,
              "buggy twin source is distinguishable (both '%s')",
              twin.desc);
        jit_free_version(&badtwin);
    }

    jit_free_version(&twin);

    /* Predtail twin source mapping (track B1sve): the predtail twin is
     * built from the winner's candidate, and its desc is the winner's
     * desc + " predtail". SVE winners only (the whilelt tail lives in
     * the AArch64 SVE backend). */
    if (vs[winner].cand.isa == ISA_SVE) {
        cand_t ptwinc = vs[winner].cand;
        ptwinc.predtail = 1;
        version_t ptwin;
        memset(&ptwin, 0, sizeof(ptwin));
        CHECK(jit_build_version(&pl, &ptwinc, &cpu, &o, &ptwin, err,
                                sizeof(err)) == 0,
              "predtail twin build: %s", err);
        if (ptwin.fn) {
            size_t pwlen = strlen(vs[winner].desc);
            CHECK(strncmp(ptwin.desc, vs[winner].desc, pwlen) == 0 &&
                      strstr(ptwin.desc, "predtail") != NULL,
                  "predtail twin pairs with winner: '%s' vs '%s'",
                  ptwin.desc, vs[winner].desc);
            CHECK(ptwin.predtail == 1 && ptwin.notail == 0,
                  "predtail twin flags: predtail=%d notail=%d",
                  ptwin.predtail, ptwin.notail);
            jit_free_version(&ptwin);
        }
    }
    for (int i = 0; i < nv; i++)
        jit_free_version(&vs[i]);
    plan_free(&pl);
    rj_op_free(&op);
    topo_int_free(&ti);
    rj_topo_free(t);
}

/* M1 regression: validate_versions must size its scratch buffers for
 * 2*group_bytes, not for the fixed vmax. Simulates a wide-vector backend
 * (e.g. RVV with a large vlenb) by doctoring a real notail twin's
 * group_bytes up to the smallest multiple of its true group that is
 * >= 4096, so that 2*gb > vmax. Every validation length stays a true
 * group multiple, so the twin's output matches the reference -- the ONLY
 * thing exercised is the buffer sizing. Against the old fixed-vmax
 * sizing this trips ASan (heap-buffer-overflow in the validation fill
 * loop); against the fixed code it passes. */
static void test_large_group_validation(void)
{
    char err[256];
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    rj_topology *t = rj_topo_raid6(6);
    topo_int_t ti;
    CHECK(topo_int_build(&ti, t, err, sizeof(err)) == 0, "topo: %s", err);
    rj_op op = rj_op_encode();
    plan_t pl;
    CHECK(plan_build(&ti, &op, "largegroup", &pl, err, sizeof(err)) == 0,
          "plan: %s", err);

    cand_t cands[64];
    int ncand = plan_candidates(&pl, &cpu, cands, 64, 0, 0);
    CHECK(ncand > 0, "no candidates");

    rj_options o;
    rj_options_default(&o);
    o.benchmark = 0;

    version_t twin;
    memset(&twin, 0, sizeof(twin));
    int got = 0;
    for (int i = 0; i < ncand && !got; i++) {
        cand_t tc = cands[i];
        tc.notail = 1;
        char e2[256];
        if (jit_build_version(&pl, &tc, &cpu, &o, &twin, e2,
                              sizeof(e2)) == 0)
            got = 1;
    }
    CHECK(got, "no buildable notail twin");
    size_t real_g = twin.group_bytes;
    CHECK(real_g > 0, "real group_bytes set");
    /* smallest multiple of the true group >= 4096: 2*sim_gb > vmax, and
     * every validation length is a true group multiple */
    size_t sim_gb = ((4096 + real_g - 1) / real_g) * real_g;
    CHECK(sim_gb >= 4096 && sim_gb % real_g == 0, "sim_gb=%zu", sim_gb);
    twin.group_bytes = sim_gb;

    rj_options bo;
    rj_options_default(&bo);
    bo.bench_bytes = 1 << 16;
    bo.bench_iters = 2;
    int rc = bench_select(&twin, 1, &pl, &bo, 0, NULL, err, sizeof(err));
    CHECK(rc == 0, "large-group (gb=%zu) validation passes: %s", sim_gb,
          err);

    jit_free_version(&twin);
    plan_free(&pl);
    rj_op_free(&op);
    topo_int_free(&ti);
    rj_topo_free(t);
}

/* R1-m13: bench_select must reject a notail twin with group_bytes == 0
 * instead of silently recording garbage timings at the nominal sizes. */
static void test_group_bytes_zero_rejected(void)
{
    char err[256];
    plan_t pl;
    memset(&pl, 0, sizeof(pl)); /* noop plan: validation trivially clean */
    rj_options o;
    rj_options_default(&o);
    o.bench_bytes = 4096;
    o.bench_iters = 2;

    version_t v;
    memset(&v, 0, sizeof(v));
    v.fn = mock0;
    v.notail = 1;
    v.group_bytes = 0;
    snprintf(v.desc, sizeof(v.desc), "zero-group notail");
    int rc = bench_select(&v, 1, &pl, &o, 0, NULL, err, sizeof(err));
    CHECK(rc == -1, "notail group_bytes==0 rejected (rc=%d)", rc);
    CHECK(strstr(err, "group_bytes") != NULL,
          "rejection names group_bytes: '%s'", err);

    /* positive control: a sane group passes the gate (noop plan
     * validates trivially; the mock fn ignores its buffers) */
    v.group_bytes = 64;
    rc = bench_select(&v, 1, &pl, &o, 0, NULL, err, sizeof(err));
    CHECK(rc == 0, "notail group_bytes=64 accepted (rc=%d err='%s')", rc,
          err);
}

/* Plausible full-group sizes across backends: U in 1..8 x VL in
 * {16,32,64} (x86/AArch64-NEON), SVE VL bytes, RVV vlenb. The matrix hits
 * g-1/g/g+1 for every one of them, so both dispatch variants run no
 * matter which group the winner has. */
static const size_t egroups[] = { 16, 32, 48, 64, 80, 96, 112, 128,
                                  160, 192, 224, 256, 320, 384, 448, 512 };

typedef struct {
    const char *name;
    rj_topology *(*mk_topo)(void);
    rj_op (*mk_op)(void);
    int benchmark;
    int nt;
} ecfg_t;

/* Compile via rj_compile and check a dense length matrix vs ref_run.
 * Returns 1 if the dispatcher was installed, 0 otherwise. */
static int test_e2e_config(const ecfg_t *c)
{
    char err[256];
    rj_topology *t = c->mk_topo();
    rj_op op = c->mk_op();

    rj_options o;
    rj_options_default(&o);
    o.benchmark = c->benchmark;
    o.bench_bytes = 1 << 16; /* keep the shootout fast */
    o.bench_iters = 3;
    if (c->nt)
        o.nt_threshold = 1;

    rj_jit *j = rj_compile(t, &op, &o, err, sizeof(err));
    CHECK(j != NULL, "%s compile: %s", c->name, err);
    if (!j) {
        rj_op_free(&op);
        rj_topo_free(t);
        return 0;
    }
    rj_fn fn = rj_get_fn(j);
    CHECK(fn != NULL, "%s: fn non-null", c->name);
    rj_info info;
    rj_get_info(j, &info);
    int dispatched = strstr(info.desc, "notail") != NULL;
    printf("note: %-28s -> %-44s %s\n", c->name, info.desc,
           dispatched ? "[dispatcher]" : "[general-only]");

    /* rebuild the plan for the reference */
    topo_int_t ti;
    plan_t pl;
    CHECK(topo_int_build(&ti, t, err, sizeof(err)) == 0, "%s topo: %s",
          c->name, err);
    CHECK(plan_build(&ti, &op, c->name, &pl, err, sizeof(err)) == 0,
          "%s plan: %s", c->name, err);
    int n_in = pl.n_in, n_out = pl.n_out;

    size_t maxlen = 65537 + 64;
    uint8_t **ib = malloc(sizeof(*ib) * (size_t)n_in);
    uint8_t **ob = malloc(sizeof(*ob) * (size_t)n_out);
    uint8_t **rb = malloc(sizeof(*rb) * (size_t)n_out);
    const void **in = malloc(sizeof(*in) * (size_t)n_in);
    void **out = malloc(sizeof(*out) * (size_t)n_out);
    for (int i = 0; i < n_in; i++) {
        ib[i] = abuf(maxlen);
        in[i] = ib[i];
    }
    for (int j = 0; j < n_out; j++) {
        ob[j] = abuf(maxlen);
        rb[j] = abuf(maxlen);
        out[j] = ob[j];
    }

    /* dense matrix: 0, 1, every plausible group boundary, one large odd */
    size_t lens[64];
    int nl = 0;
    lens[nl++] = 0;
    lens[nl++] = 1;
    for (size_t gi = 0; gi < sizeof(egroups) / sizeof(egroups[0]); gi++) {
        size_t g = egroups[gi];
        lens[nl++] = g - 1;
        lens[nl++] = g;
        lens[nl++] = g + 1;
    }
    lens[nl++] = 65537;
    for (int li = 0; li < nl; li++) {
        size_t len = lens[li];
        for (int i = 0; i < n_in; i++)
            randfill(ib[i], len);
        for (int j = 0; j < n_out; j++)
            memset(ob[j], 0xA5, len);
        fn(in, out, len);
        ref_run(&pl, (const uint8_t **)ib, rb, len);
        for (int j = 0; j < n_out; j++)
            CHECK(memcmp(ob[j], rb[j], len) == 0,
                  "%s mismatch at len=%zu out=%d", c->name, len, j);
    }

    for (int i = 0; i < n_in; i++)
        free(ib[i]);
    for (int j = 0; j < n_out; j++) {
        free(ob[j]);
        free(rb[j]);
    }
    free(ib);
    free(ob);
    free(rb);
    free(in);
    free(out);
    plan_free(&pl);
    topo_int_free(&ti);
    rj_op_free(&op);
    rj_topo_free(t);

    int rc = dispatched;
    rj_free(j);
    return rc;
}

/* ---------------- 4. overhead (informational) ---------------- */

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Time direct general fn vs dispatcher-routed call at a group multiple. */
static void test_overhead(void)
{
    char err[256];
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    rj_topology *t = rj_topo_raid6(6);
    topo_int_t ti;
    if (topo_int_build(&ti, t, err, sizeof(err)))
        goto out;
    rj_op op = rj_op_encode();
    plan_t pl;
    if (plan_build(&ti, &op, "oh", &pl, err, sizeof(err))) {
        rj_op_free(&op);
        goto out_ti;
    }
    cand_t cands[64];
    int ncand = plan_candidates(&pl, &cpu, cands, 64, 0, 0);
    rj_options o;
    rj_options_default(&o);
    o.benchmark = 0;
    version_t gen, twin;
    memset(&gen, 0, sizeof(gen));
    memset(&twin, 0, sizeof(twin));
    int got = 0;
    for (int i = 0; i < ncand && !got; i++) {
        char e2[256];
        if (jit_build_version(&pl, &cands[i], &cpu, &o, &gen, e2,
                              sizeof(e2)))
            continue;
        cand_t tc = cands[i];
        tc.notail = 1;
        if (jit_build_version(&pl, &tc, &cpu, &o, &twin, e2, sizeof(e2))) {
            jit_free_version(&gen);
            continue;
        }
        got = 1;
    }
    if (!got) {
        printf("note: overhead test: no buildable candidate\n");
        rj_op_free(&op);
        plan_free(&pl);
        goto out_ti;
    }
    size_t G = twin.group_bytes;
    rj_variant vs[2];
    memset(vs, 0, sizeof(vs));
    vs[0].fn = twin.fn;
    vs[0].tail = TAIL_NONE;
    vs[0].group = G;
    vs[0].gbps = 2.0;
    vs[1].fn = gen.fn;
    vs[1].tail = TAIL_SCALAR;
    vs[1].gbps = 1.0;
    rj_variant_derive(&vs[0]);
    rj_variant_derive(&vs[1]);
    rj_dispatcher *d = rj_dispatcher_create(vs, 2);
    if (!d) {
        printf("note: overhead test: no dispatcher slot\n");
    } else {
        int n_in = pl.n_in, n_out = pl.n_out;
        const void **in = malloc(sizeof(*in) * (size_t)n_in);
        void **out = malloc(sizeof(*out) * (size_t)n_out);
        for (int i = 0; i < n_in; i++)
            in[i] = abuf(G);
        for (int j = 0; j < n_out; j++)
            out[j] = abuf(G);
        rj_fn disp = rj_dispatcher_fn(d);
        const int N = 200000;
        double t0 = now_sec();
        for (int k = 0; k < N; k++)
            gen.fn(in, out, G);
        double t1 = now_sec();
        for (int k = 0; k < N; k++)
            disp(in, out, G);
        double t2 = now_sec();
        printf("note: per-call ns: direct=%.1f dispatcher=%.1f overhead=%.1f\n",
               (t1 - t0) / N * 1e9, (t2 - t1) / N * 1e9,
               ((t2 - t1) - (t1 - t0)) / N * 1e9);
        for (int i = 0; i < n_in; i++)
            free((void *)in[i]);
        for (int j = 0; j < n_out; j++)
            free(out[j]);
        free(in);
        free(out);
        rj_dispatcher_destroy(d);
    }
    jit_free_version(&gen);
    jit_free_version(&twin);
    rj_op_free(&op);
    plan_free(&pl);
out_ti:
    topo_int_free(&ti);
out:
    rj_topo_free(t);
}

int main(void)
{
    /* 1. white-box */
    test_admits();
    test_sort();
    test_routing();
    test_length_aware_routing();
    test_create_args();
    test_slot_exhaustion();
    test_group_bytes_zero_rejected();

    /* 2. integration: direct twin builds */
    static const dcfg_t dcfgs[] = {
        { "raid6(6) enc", t_raid6_6, o_enc, 0, 0, 0, 0 },
        { "raid5(4) enc", t_raid5_4, o_enc, 0, 0, 0, 0 },
        { "raid1 enc", t_raid1, o_enc, 0, 0, 0, 0 },
        { "rs(8,3) enc", t_rs83, o_enc, 0, 0, 0, 0 },
        { "raid6(6) rec d0", t_raid6_6, o_rec0, 0, 0, 0, 0 },
        { "raid6(6) pup d1", t_raid6_6, o_pup1, 0, 0, 0, 0 },
        { "raid6(6) enc nt", t_raid6_6, o_enc, 1, 1, 0, 0 },
        { "raid6(6) enc prefetch", t_raid6_6, o_enc, 0, 0, 1, 512 },
    };
    for (size_t i = 0; i < sizeof(dcfgs) / sizeof(dcfgs[0]); i++)
        test_direct_twin(&dcfgs[i]);

    /* 2b. twin-source mapping regression (cands[]/vs[] index confusion) */
    test_twin_source_mapping();

    /* 2c. M1: large-group validation buffer sizing */
    test_large_group_validation();

    /* 3. end-to-end */
    static const ecfg_t ecfgs[] = {
        { "e2e raid6(6) enc bench", t_raid6_6, o_enc, 1, 0 },
        { "e2e raid5(4) rec d0 bench", t_raid5_4, o_rec0, 1, 0 },
        { "e2e raid6(6) pup d1 bench", t_raid6_6, o_pup1, 1, 0 },
        { "e2e raid6(6) enc nt bench", t_raid6_6, o_enc, 1, 1 },
        { "e2e raid6(6) enc nobench", t_raid6_6, o_enc, 0, 0 },
    };
    int any_dispatch = 0;
    for (size_t i = 0; i < sizeof(ecfgs) / sizeof(ecfgs[0]); i++)
        any_dispatch |= test_e2e_config(&ecfgs[i]);
    printf("note: dispatcher installed in %s\n",
           any_dispatch ? ">=1 config" : "no config (twin lost)");
    /* The dispatch-vs-general agreement checks above are vacuous when
     * the twin loses everywhere: pin that at least one benchmark config
     * installed the dispatcher, so a silent all-general run cannot pass
     * as dispatch coverage. */
    CHECK(any_dispatch,
          "no benchmark config installed the dispatcher: agreement "
          "coverage is vacuous on this machine");

    /* agreement: benchmark=1 vs benchmark=0 bit-identical */
    {
        char err[256];
        rj_options o;
        rj_options_default(&o);
        o.bench_bytes = 1 << 16;
        o.bench_iters = 3;
        rj_topology *t1 = rj_topo_raid6(6), *t2 = rj_topo_raid6(6);
        rj_op op1 = rj_op_encode(), op2 = rj_op_encode();
        o.benchmark = 1;
        rj_jit *j1 = rj_compile(t1, &op1, &o, err, sizeof(err));
        o.benchmark = 0;
        rj_jit *j2 = rj_compile(t2, &op2, &o, err, sizeof(err));
        CHECK(j1 && j2, "agreement compiles: %s", err);
        if (j1 && j2) {
            rj_fn f1 = rj_get_fn(j1), f2 = rj_get_fn(j2);
            size_t lens[] = { 0, 1, 255, 256, 257, 4096, 65537 };
            uint8_t *ib[6], *o1[2], *o2[2];
            const void *in[6];
            void *out1[2], *out2[2];
            for (int i = 0; i < 6; i++) {
                ib[i] = abuf(65537 + 64);
                randfill(ib[i], 65537 + 64);
                in[i] = ib[i];
            }
            for (int k = 0; k < 2; k++) {
                o1[k] = abuf(65537 + 64);
                o2[k] = abuf(65537 + 64);
                out1[k] = o1[k];
                out2[k] = o2[k];
            }
            for (size_t li = 0; li < sizeof(lens) / sizeof(lens[0]);
                 li++) {
                size_t len = lens[li];
                f1(in, out1, len);
                f2(in, out2, len);
                for (int k = 0; k < 2; k++)
                    CHECK(memcmp(o1[k], o2[k], len) == 0,
                          "dispatch/general agree at len=%zu", len);
            }
            for (int i = 0; i < 6; i++)
                free(ib[i]);
            for (int k = 0; k < 2; k++) {
                free(o1[k]);
                free(o2[k]);
            }
        }
        rj_free(j1);
        rj_free(j2);
        rj_op_free(&op1);
        rj_op_free(&op2);
        rj_topo_free(t1);
        rj_topo_free(t2);
    }

    /* 4. overhead */
    test_overhead();

    if (failures == 0)
        printf("PASS test_dispatch\n");
    return failures != 0;
}
