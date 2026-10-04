/* RVV Zvbc32e carryless multiply (B3) and Zicond absence (B6) tests.
 *
 * B3: the RVV backend emits a second multiply-method twin, MUL_ZVBC,
 * using the `vclmul.vv` / `vclmulh.vv` carryless multiplies for
 * GF(2^8) multiplies (SEW=8 lane formulation, 0x1B reduction; see
 * ~/workspace/raidjit-tracks/Brvv/DESIGN.md). Both instructions are
 * Zvbc-family: the ratified Vector Crypto spec (§36.2/36.3) defines
 * them for SEW=64 only; the SEW=8 forms the backend emits require the
 * separate Zvbc32e extension (fast-track vector-crypto-extra §33),
 * which is independent of Zvbc -- Zvbc does NOT imply Zvbc32e, and
 * Zvbc32e is legal standalone (ELEN >= 32). On Zvbc-without-Zvbc32e
 * silicon the SEW=8 encodings are reserved (-> illegal-insn). The
 * twin is therefore gated on rvv_zvbc32e alone; the backend
 * re-checks it before emitting (defense in depth). Until a
 * kernel/QEMU reports the zvbc32e token the twin stays off (fail
 * closed). The historical rvv_zvbc/rvv_zvkg gates were dropped:
 * vclmulh.vv was misattributed to Zvkg (which actually defines
 * vghsh/vgmul, not vclmulh), and neither legacy gate buys any safety
 * for the SEW=8 encodings actually emitted.
 *
 * B6: Zicond was evaluated and REJECTED (no profitable branch-kill;
 * see DESIGN.md). This file pins that verdict: representative RVV
 * versions must contain zero czero.eqz/nez words.
 *
 * Checks:
 *  A. emitter shape: em_vclmul_vv/em_vclmulh_vv byte-exact vs
 *     GAS-verified words (incl. v0/v31 edge regs).
 *  B. lane model: C model of the 5-clmul + xor reduction sequence vs
 *     gf_mul, exhaustive over all 65536 input pairs + edge patterns.
 *  C. planner: MUL_ZVBC twins enumerated iff rvv_zvbc32e (all 8
 *     (zvbc, zvkg, zvbc32e) gate combos); MUL_NIBBLE always present.
 *     The (1,1,0) combo must NOT enumerate (SEW=8 vclmul on
 *     Zvbc-without-Zvbc32e is a reserved encoding); (0,0,1) MUST
 *     (Zvbc32e standalone is legal).
 *  D. backend gates: force-building MUL_ZVBC with zvbc32e=0 fails
 *     cleanly (no code, error message) regardless of the legacy
 *     zvbc/zvkg gates; zvbc32e=1 (even with zvbc=zvkg=0) builds, the
 *     desc carries "+zbc", and the real emitted bytes are checked:
 *     (i) the first vclmul.vv of a multiply group has the expected
 *     register wiring, (ii) the 9-word backend sequence is executed
 *     by the interpreter and must compute gf_mul(c,a)^t per lane.
 *     D2 extends the interpreter check to one MULACC and one DELTA
 *     ZVBC twin (the historically buggy paths); D3 verifies the
 *     prologue's addi/vmv.v.x broadcast pairs (0x1B -> kvec,
 *     coefficients -> cvec slots), which D/D2 otherwise trust.
 *  E. emitted-bytes interpreter: straight-line vector GF-multiplies
 *     built with the real emitter (the exact sequences from the
 *     backend's algorithm) are executed by a minimal interpreter over
 *     the actual emitted bytes and checked vs gf_mul; the ZVBC and
 *     nibble paths must agree (differential vs the non-extension
 *     path). QEMU 8.2.2 cannot execute vclmul.vv/vclmulh.vv
 *     (illegal-insn), so execution is via the interpreter, not the
 *     CPU.
 *  F. Zicond absence: representative RVV versions (Horner/MULACC/
 *     P_XOR plans, nibble AND zvbc twins, every U the planner's RVV
 *     branch enumerates) contain zero czero words. NOTE: the planner's
 *     RVV branch only sets .isa/.mul/.U/.remat -- nt/prefetch/notail
 *     are api.c-level options, never enumerated by plan_candidates,
 *     so they are not covered here; P_DELTA plans are not in the F
 *     set either (they are covered for the table-load scan in G). A
 *     positive control pins the czero matcher to real GAS-verified
 *     czero words so the scan can never pass vacuously.
 *  G. Zvbc MULACC/DELTA: no constant-pool table loads (regression:
 *     the path once emitted nibble-table loads into the unassigned
 *     tbl regs, clobbering accumulator v0).
 *
 * RVV-only: the planner enumerates RVV candidates and the JIT backend
 * only in a riscv64 build. On other hosts the binary builds and prints
 * SKIP.
 *
 * Run under QEMU at several VLENs, e.g.
 *   QEMU_LD_PREFIX=/usr/riscv64-linux-gnu qemu-riscv64 -cpu max \
 *       ./build-riscv64/test_riscv_zvbc
 *   ... -cpu max,vlen=256 ... ; ... -cpu max,vlen=512 ...
 */
#include "../include/raidjit.h"
#include "../src/plan.h"
#include "../src/jit.h"
#include "../src/cpu.h"
#include "../src/gf.h"
#include "../src/ref.h"
#include "../src/riscv/emit.h"

#include <stdio.h>
#include <stdlib.h>
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

#ifdef __riscv

static uint32_t word_at(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint64_t rng_state = 0x243f6a8885a308d3ULL;
static uint64_t xrng(void)
{
    uint64_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return rng_state = x;
}

/* ---------------- A. emitter shape ---------------- */

static void test_emitter_shape(void)
{
    /* GAS-verified reference words
     * (riscv64-linux-gnu-as -march=rv64gcv_zvbc_zvkg, verified
     * 2026-10-03; the extra combos were cross-checked against GAS,
     * not hand-computed): */
    struct {
        int vd, vs1, vs2;
        uint32_t want;
        int hi; /* 0 = vclmul.vv, 1 = vclmulh.vv */
    } tv[] = {
        { 8, 9, 10, 0x32952457u, 0 },
        { 0, 0, 0, 0x32002057u, 0 },
        { 31, 1, 2, 0x32112fd7u, 0 },
        { 11, 9, 10, 0x369525d7u, 1 },
        { 0, 31, 15, 0x37f7a057u, 1 },
        /* extra combos exercising different bit patterns (AGENTS.md:
         * emitter tests must use even/odd, low/high regs) */
        { 1, 2, 3, 0x3221a0d7u, 0 },
        { 16, 17, 18, 0x33192857u, 0 },
        { 5, 6, 7, 0x3663a2d7u, 1 },
        { 30, 0, 31, 0x360faf57u, 1 },
    };
    for (size_t i = 0; i < sizeof(tv) / sizeof(tv[0]); i++) {
        uint8_t buf[8];
        em_t e;
        em_init(&e, buf, sizeof(buf));
        if (tv[i].hi)
            em_vclmulh_vv(&e, tv[i].vd, tv[i].vs1, tv[i].vs2);
        else
            em_vclmul_vv(&e, tv[i].vd, tv[i].vs1, tv[i].vs2);
        uint32_t got = word_at(buf);
        CHECK(got == tv[i].want,
              "%s v%d,v%d,v%d: got 0x%08x want 0x%08x",
              tv[i].hi ? "vclmulh.vv" : "vclmul.vv", tv[i].vd, tv[i].vs1,
              tv[i].vs2, got, tv[i].want);
    }
}

/* ---------------- B. lane model vs gf_mul ---------------- */

/* Carryless multiply of two bytes -> 16-bit product. */
static unsigned clmul8(unsigned a, unsigned b)
{
    unsigned p = 0;
    for (unsigned i = 0; i < 8; i++)
        if (b & (1u << i))
            p ^= a << i;
    return p & 0xffffu;
}

/* The exact SEW=8 lane sequence the backend emits per multiply-add
 * (5 carryless + 3 xor); K = 0x1B reduction constant. */
static unsigned zvbc_lane(unsigned a, unsigned b)
{
    const unsigned K = 0x1b;
    unsigned lo = clmul8(a, b) & 0xffu;
    unsigned hi = (clmul8(a, b) >> 8) & 0xffu;
    unsigned q = clmul8(hi, K) & 0xffu;
    unsigned qh = (clmul8(hi, K) >> 8) & 0xffu;
    unsigned n = clmul8(qh, K) & 0xffu;
    return (lo ^ q ^ n) & 0xffu;
}

static void test_lane_model(void)
{
    /* Exhaustive: every input pair must match gf_mul. */
    for (unsigned a = 0; a < 256; a++)
        for (unsigned b = 0; b < 256; b++) {
            unsigned got = zvbc_lane(a, b);
            unsigned want = gf_mul((uint8_t)a, (uint8_t)b);
            if (got != want) {
                CHECK(0, "lane model mismatch: a=0x%02x b=0x%02x "
                         "got=0x%02x want=0x%02x",
                      a, b, got, want);
                return;
            }
        }
    /* Edge patterns, stated explicitly (zero/one/all-ones): */
    CHECK(zvbc_lane(0x00, 0x00) == 0x00, "0*0");
    CHECK(zvbc_lane(0x00, 0xff) == 0x00, "0*ff");
    CHECK(zvbc_lane(0xff, 0x00) == 0x00, "ff*0");
    CHECK(zvbc_lane(0x01, 0x01) == 0x01, "1*1");
    CHECK(zvbc_lane(0x01, 0xab) == 0xab, "1*x");
    CHECK(zvbc_lane(0xff, 0xff) == gf_mul(0xff, 0xff), "ff*ff");
    CHECK(zvbc_lane(0x80, 0x80) == gf_mul(0x80, 0x80),
          "0x80*0x80 (reduction stress)");
    CHECK(zvbc_lane(0x1b, 0xff) == gf_mul(0x1b, 0xff), "K*x");
}

/* ---------------- C. planner twin enumeration ---------------- */

static void test_planner_twins(void)
{
    char err[256];
    rj_topology *t = rj_topo_rs(6, 3);
    CHECK(t != NULL, "rj_topo_rs(6,3)");
    if (!t)
        return;
    topo_int_t ti;
    CHECK(topo_int_build(&ti, t, err, sizeof(err)) == 0, "topo: %s", err);
    rj_op op = rj_op_encode();
    plan_t pl;
    CHECK(plan_build(&ti, &op, "rs63", &pl, err, sizeof(err)) == 0,
          "plan: %s", err);
    if (failures) {
        rj_topo_free(t);
        return;
    }

    /* All 8 (zvbc, zvkg, zvbc32e) combos: ZVBC twins iff zvbc32e.
     * The SEW=8 vclmul.vv/vclmulh.vv encodings are Zvbc32e, independent
     * of Zvbc (Zvbc defines them for SEW=64 only); Zvkg defines
     * vghsh/vgmul, not vclmulh, so its gate is meaningless here.
     * (1,1,0) must NOT enumerate: on Zvbc-without-Zvbc32e silicon the
     * SEW=8 encoding is reserved (-> illegal-insn). (0,0,1) MUST:
     * Zvbc32e standalone (ELEN=32, no Zvbc) is legal. */
    for (int zb = 0; zb < 2; zb++)
        for (int zk = 0; zk < 2; zk++)
            for (int z3 = 0; z3 < 2; z3++) {
                rj_cpu_info cpu;
                memset(&cpu, 0, sizeof(cpu));
                cpu.rvv = 1;
                cpu.rvv_zvbc = zb;
                cpu.rvv_zvkg = zk;
                cpu.rvv_zvbc32e = z3;
                cand_t cands[256];
                int nc = plan_candidates(&pl, &cpu, cands, 256, 1, 128);
                CHECK(nc > 0, "zb=%d zk=%d z3e=%d: no candidates", zb,
                      zk, z3);
                int n_zvbc = 0, n_nibble = 0;
                for (int i = 0; i < nc; i++) {
                    if (cands[i].isa != ISA_RVV)
                        continue;
                    if (cands[i].mul == MUL_ZVBC)
                        n_zvbc++;
                    else if (cands[i].mul == MUL_NIBBLE)
                        n_nibble++;
                    else
                        CHECK(0, "zb=%d zk=%d z3e=%d: unexpected mul=%d",
                              zb, zk, z3, (int)cands[i].mul);
                    /* zvbc+remat must never be enumerated */
                    CHECK(!(cands[i].mul == MUL_ZVBC && cands[i].remat),
                          "zb=%d zk=%d z3e=%d: zvbc+remat enumerated",
                          zb, zk, z3);
                }
                CHECK(n_nibble > 0,
                      "zb=%d zk=%d z3e=%d: nibble fallback missing", zb,
                      zk, z3);
                if (z3)
                    CHECK(n_zvbc > 0,
                          "zb=%d zk=%d z3e=1: no ZVBC twins enumerated",
                          zb, zk);
                else
                    CHECK(n_zvbc == 0,
                          "zb=%d zk=%d z3e=0: ZVBC twins enumerated "
                          "without zvbc32e", zb, zk);
            }

    plan_free(&pl);
    rj_op_free(&op);
    topo_int_free(&ti);
    rj_topo_free(t);
}

/* ---------------- interpreter helpers (D, E) ---------------- */

/* Minimal straight-line RVV interpreter: SEW=8, LMUL=1, no masking.
 * Supports exactly the instructions in the multiply sequences under
 * test. Decodes the actual emitted words independently of the
 * emitter (field extraction written from the spec here); the shape
 * test (A) pins the emitter to GAS, so A+interpreter together tie the
 * bytes to both the encoding and the semantics. */
typedef struct {
    uint8_t v[32][64]; /* vlenb up to 64 */
    unsigned long x[32];
    size_t vl; /* in bytes */
} mini_t;

static unsigned mini_clmul(unsigned a, unsigned b)
{
    unsigned p = 0;
    for (unsigned i = 0; i < 8; i++)
        if (b & (1u << i))
            p ^= a << i;
    return p & 0xffffu;
}

/* Execute one OP-V word. Returns 0 on success, -1 on unknown word. */
static int mini_step(mini_t *m, uint32_t w)
{
    if ((w & 0x7fu) != 0x57u)
        return -1;
    unsigned vd = (w >> 7) & 31u;
    unsigned f3 = (w >> 12) & 7u;
    unsigned vs1 = (w >> 15) & 31u; /* bits[19:15] */
    unsigned vs2 = (w >> 20) & 31u; /* bits[24:20] */
    unsigned f6 = (w >> 26) & 63u;
    size_t vl = m->vl;
    if (f3 == 2 && (f6 == 0x0cu || f6 == 0x0du)) {
        /* vclmul.vv / vclmulh.vv: the two multiplicands are the
         * first source (bits[24:20]) and second source
         * (bits[19:15]); commutative so order is irrelevant. */
        for (size_t i = 0; i < vl; i++) {
            unsigned p = mini_clmul(m->v[vs2][i], m->v[vs1][i]);
            m->v[vd][i] = (uint8_t)((f6 == 0x0cu ? p : p >> 8) & 0xffu);
        }
        return 0;
    }
    if (f3 == 0 && f6 == 0x0bu) { /* vxor.vv (OPIVV per GAS) */
        for (size_t i = 0; i < vl; i++)
            m->v[vd][i] = m->v[vs2][i] ^ m->v[vs1][i];
        return 0;
    }
    if (f3 == 0 && f6 == 0x09u) { /* vand.vv */
        for (size_t i = 0; i < vl; i++)
            m->v[vd][i] = m->v[vs2][i] & m->v[vs1][i];
        return 0;
    }
    if (f3 == 3 && f6 == 0x28u) { /* vsrl.vi: uimm in bits[19:15] */
        for (size_t i = 0; i < vl; i++)
            m->v[vd][i] = (uint8_t)(m->v[vs2][i] >> vs1);
        return 0;
    }
    if (f3 == 0 && f6 == 0x0cu) { /* vrgather.vv: vtab=bits[24:20],
                                  * vidx=bits[19:15]. The index is
                                  * compared against VLMAX (64 here),
                                  * not vl: the 16-byte tables are fully
                                  * populated even when vl < 16. */
        uint8_t tab[64], idx[64];
        memcpy(tab, m->v[vs2], 64);
        memcpy(idx, m->v[vs1], 64);
        for (size_t i = 0; i < vl; i++)
            m->v[vd][i] = idx[i] < 64 ? tab[idx[i]] : 0;
        return 0;
    }
    if (f3 == 0 && f6 == 0x17u) { /* vmv.v.v */
        for (size_t i = 0; i < vl; i++)
            m->v[vd][i] = m->v[vs1][i];
        return 0;
    }
    if (f3 == 4 && f6 == 0x17u) { /* vmv.v.x: rs1 in bits[19:15] */
        uint8_t b = (uint8_t)m->x[vs1];
        for (size_t i = 0; i < vl; i++)
            m->v[vd][i] = b;
        return 0;
    }
    return -1;
}

/* Run a word buffer through the interpreter. */
static int mini_run(mini_t *m, const uint8_t *code, size_t nwords)
{
    for (size_t i = 0; i < nwords; i++)
        if (mini_step(m, word_at(code + 4 * i)) != 0)
            return -1;
    return 0;
}

/* ---------------- D. backend gates ---------------- */

/* Build one RVV candidate directly, bypassing the planner's twin
 * policy (mirrors tests/test_riscv_remat.c). */
static int build_cand(const plan_t *pl, const rj_cpu_info *cpu,
                      mul_t mul, int U, int remat, version_t *v, char *err,
                      size_t errlen)
{
    rj_options o;
    rj_options_default(&o);
    o.benchmark = 0;
    cand_t cand;
    memset(&cand, 0, sizeof(cand));
    cand.isa = ISA_RVV;
    cand.mul = mul;
    cand.U = U;
    cand.remat = remat;
    memset(v, 0, sizeof(*v));
    return jit_build_version(pl, &cand, cpu, &o, v, err, errlen);
}

static int is_vclmul(uint32_t w)
{
    /* OPMVV (funct3=010), funct6 in {001100, 001101}, opcode 1010111.
     * Mask keeps funct3/opcode; funct6 is checked separately (two
     * values); vm and the register fields are don't-cares. */
    if ((w & 0x0000707fu) != ((0x2u << 12) | 0x57u))
        return 0;
    unsigned f6 = (w >> 26) & 0x3fu;
    return f6 == 0x0cu || f6 == 0x0du;
}

/* D2/D3 helpers, defined after test_backend_gates. */
static void check_broadcasts(const uint8_t *mem, size_t code_size,
                             const plan_t *pl, int U, int kvec, int cvec,
                             const char *tag);

static void test_backend_gates(void)
{
    char err[256];
    rj_topology *t = rj_topo_rs(6, 3);
    CHECK(t != NULL, "rj_topo_rs(6,3)");
    if (!t)
        return;
    topo_int_t ti;
    CHECK(topo_int_build(&ti, t, err, sizeof(err)) == 0, "topo: %s", err);
    rj_op op = rj_op_encode();
    plan_t pl;
    CHECK(plan_build(&ti, &op, "rs63", &pl, err, sizeof(err)) == 0,
          "plan: %s", err);
    if (failures) {
        rj_topo_free(t);
        return;
    }
    CHECK(pl.kind == P_HORNER, "expected P_HORNER, kind=%d",
          (int)pl.kind);

    /* Missing zvbc32e: clean failure, no code emitted -- whatever the
     * legacy zvbc/zvkg gates say. SEW=8 vclmul.vv/vclmulh.vv are
     * Zvbc32e; Zvbc-without-Zvbc32e silicon would take illegal-insn
     * on the emitted words. */
    const int gates[4][3] = {
        { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 }, { 1, 1, 0 }
    };
    for (int g = 0; g < 4; g++) {
        rj_cpu_info cpu;
        memset(&cpu, 0, sizeof(cpu));
        cpu.rvv = 1;
        cpu.rvv_zvbc = gates[g][0];
        cpu.rvv_zvkg = gates[g][1];
        cpu.rvv_zvbc32e = gates[g][2];
        version_t v;
        int rc = build_cand(&pl, &cpu, MUL_ZVBC, 2, 0, &v, err,
                            sizeof(err));
        CHECK(rc != 0, "zb=%d zk=%d z3e=%d: zvbc build should fail",
              gates[g][0], gates[g][1], gates[g][2]);
        CHECK(err[0] != '\0', "zb=%d zk=%d z3e=%d: no error message",
              gates[g][0], gates[g][1], gates[g][2]);
        if (rc == 0)
            jit_free_version(&v);
    }

    /* zvbc+remat: planner never enumerates it; the backend refuses. */
    {
        rj_cpu_info cpu;
        memset(&cpu, 0, sizeof(cpu));
        cpu.rvv = 1;
        cpu.rvv_zvbc = 1;
        cpu.rvv_zvkg = 1;
        cpu.rvv_zvbc32e = 1;
        version_t v;
        int rc = build_cand(&pl, &cpu, MUL_ZVBC, 2, 1, &v, err,
                            sizeof(err));
        CHECK(rc != 0, "zvbc+remat build should fail");
        if (rc == 0)
            jit_free_version(&v);
    }

    /* zvbc32e alone (legacy gates clear): builds, desc carries "+zbc",
     * code has vclmul words. This pins that the legacy zvbc/zvkg gates
     * are not required -- the SEW=8 encodings are Zvbc32e. */
    {
        rj_cpu_info cpu;
        memset(&cpu, 0, sizeof(cpu));
        cpu.rvv = 1;
        cpu.rvv_zvbc32e = 1;
        version_t v;
        int rc = build_cand(&pl, &cpu, MUL_ZVBC, 2, 0, &v, err,
                            sizeof(err));
        CHECK(rc == 0, "zvbc build failed: %s", err);
        if (rc != 0) {
            plan_free(&pl);
            rj_op_free(&op);
            topo_int_free(&ti);
            rj_topo_free(t);
            return;
        }
        CHECK(strstr(v.desc, "+zbc") != NULL,
              "desc missing +zbc: %s", v.desc);
        CHECK(strstr(v.desc, "nibble") == NULL,
              "desc still says nibble: %s", v.desc);
        int ncl = 0;
        for (size_t i = 0; i + 4 <= v.code_size; i += 4)
            if (is_vclmul(word_at((const uint8_t *)v.mem + i)))
                ncl++;
        CHECK(ncl > 0, "no vclmul words in zvbc build");

        /* Wiring + semantic checks on the REAL backend output.
         * Register assignment for HORNER zvbc, U=2, n_out=3:
         * vd0=6, r1=8, r2=9, r3=10, kvec=11, cvec per nontrivial
         * row from 12. */
        int U = 2, n_out = pl.n_out;
        int vd0 = n_out * U, r1 = vd0 + U;
        int kvec = r1 + 3;
        int r = r1 + 4; /* first cvec slot */
        int seq_at = -1, seq_j = -1, seq_u = -1, seq_cv = -1;
        for (int j = 0; j < n_out && seq_at < 0; j++) {
            uint8_t c = pl.horner_c[j];
            if (c == 0 || c == 1)
                continue;
            int cv = r++;
            for (int u = 0; u < U && seq_at < 0; u++) {
                int acc = j * U + u;
                for (size_t i = 0; i + 4 * 9 <= v.code_size; i += 4) {
                    uint32_t w = word_at((const uint8_t *)v.mem + i);
                    if ((w & 0x0000707fu) != ((0x2u << 12) | 0x57u))
                        continue;
                    if (((w >> 26) & 0x3fu) != 0x0cu)
                        continue;
                    unsigned wd = (w >> 7) & 31u;
                    unsigned w1 = (w >> 20) & 31u;
                    unsigned w2 = (w >> 15) & 31u;
                    /* first source (bits[24:20]) = acc, second
                     * (bits[19:15]) = cvec */
                    if ((int)wd == r1 && (int)w1 == acc &&
                        (int)w2 == cv) {
                        seq_at = (int)i;
                        seq_j = j;
                        seq_u = u;
                        seq_cv = cv;
                        break;
                    }
                }
            }
        }
        CHECK(seq_at >= 0, "no zvbc sequence with expected wiring");
        if (seq_at >= 0) {
            /* Run the backend's actual 9 emitted words through the
             * interpreter: acc = gf_mul(c, a) ^ t per lane. */
            uint8_t c = pl.horner_c[seq_j];
            int acc = seq_j * U + seq_u;
            int vd = vd0 + seq_u;
            int cv = seq_cv;
            mini_t m;
            memset(&m, 0, sizeof(m));
            m.vl = 16;
            uint8_t a[16], tt[16];
            for (int i = 0; i < 16; i++) {
                a[i] = (uint8_t)xrng();
                tt[i] = (uint8_t)(xrng() >> 32);
            }
            memcpy(m.v[acc], a, 16);
            memset(m.v[cv], c, 16);
            memset(m.v[kvec], 0x1b, 16);
            memcpy(m.v[vd], tt, 16);
            CHECK(mini_run(&m, (const uint8_t *)v.mem + seq_at, 9) == 0,
                  "backend sequence interp failed");
            for (int i = 0; i < 16 && !failures; i++) {
                uint8_t want = gf_mul(c, a[i]) ^ tt[i];
                CHECK(m.v[acc][i] == want,
                      "backend seq lane %d: got 0x%02x want 0x%02x",
                      i, m.v[acc][i], want);
            }
        }

        /* D3 on this HORNER build: verify the prologue's broadcast
         * pairs -- 0x1B -> kvec, row coefficients -> cvec slots. */
        check_broadcasts((const uint8_t *)v.mem, v.code_size, &pl, U,
                         kvec, -1, "horner");

        /* Nibble twin on the same plan: no vclmul words, "+nibble". */
        rc = build_cand(&pl, &cpu, MUL_NIBBLE, 2, 0, &v, err,
                        sizeof(err));
        CHECK(rc == 0, "nibble build failed: %s", err);
        if (rc == 0) {
            int ncl2 = 0;
            for (size_t i = 0; i + 4 <= v.code_size; i += 4)
                if (is_vclmul(word_at((const uint8_t *)v.mem + i)))
                    ncl2++;
            CHECK(ncl2 == 0, "nibble build emitted %d vclmul words",
                  ncl2);
            jit_free_version(&v);
        }
    }

    plan_free(&pl);
    rj_op_free(&op);
    topo_int_free(&ti);
    rj_topo_free(t);
}

/* ---------------- D2. MULACC/DELTA ZVBC semantic validation ---------------- */

/* addi decoder: OP-IMM (0001011... no: 0010011), funct3=000. */
static int dec_addi(uint32_t w, unsigned *rd, unsigned *rs1, int *imm)
{
    if ((w & 0x707fu) != 0x13u)
        return 0;
    *rd = (w >> 7) & 31u;
    *rs1 = (w >> 15) & 31u;
    int v = (int)(w >> 20);
    *imm = (v & 0x800) ? v - 0x1000 : v;
    return 1;
}

/* vmv.v.x decoder: funct6=010111, funct3=100, opcode 1010111. */
static int dec_vmv_v_x(uint32_t w, unsigned *vd, unsigned *rs1)
{
    if ((w & 0xfc00707fu) != ((0x17u << 26) | (0x4u << 12) | 0x57u))
        return 0;
    *vd = (w >> 7) & 31u;
    *rs1 = (w >> 15) & 31u;
    return 1;
}

/* vmv.v.v decoder: funct6=010111, funct3=000, opcode 1010111. */
static int dec_vmv_v_v(uint32_t w, unsigned *vd, unsigned *vs1)
{
    if ((w & 0xfc00707fu) != ((0x17u << 26) | (0x0u << 12) | 0x57u))
        return 0;
    *vd = (w >> 7) & 31u;
    *vs1 = (w >> 15) & 31u;
    return 1;
}

/* vxor.vv decoder: funct6=001011, funct3=000, opcode 1010111. */
static int dec_vxor_vv(uint32_t w, unsigned *vd, unsigned *vsA,
                       unsigned *vsB)
{
    if ((w & 0xfc00707fu) != ((0x0bu << 26) | (0x0u << 12) | 0x57u))
        return 0;
    *vd = (w >> 7) & 31u;
    *vsB = (w >> 15) & 31u; /* bits[19:15]: second source */
    *vsA = (w >> 20) & 31u; /* bits[24:20]: first source */
    return 1;
}

/* Find the first ZVBC multiply sequence in the backend's emitted code:
 * a vclmul.vv word with vd==r1, first source in [vd0, vd0+U), second
 * source == cvec. Returns the word index, or -1; vd_out gets the
 * multiplicand register. */
static int find_mul_seq(const uint8_t *mem, size_t code_size, int r1,
                        int vd0, int U, int cvec, int *vd_out,
                        const char *tag)
{
    size_t nwords = code_size / 4;
    for (size_t i = 0; i < nwords; i++) {
        uint32_t w = word_at(mem + 4 * i);
        if ((w & 0x0000707fu) != ((0x2u << 12) | 0x57u))
            continue;
        if (((w >> 26) & 0x3fu) != 0x0cu)
            continue;
        unsigned wd = (w >> 7) & 31u;
        unsigned wA = (w >> 20) & 31u; /* bits[24:20]: first source */
        unsigned wB = (w >> 15) & 31u; /* bits[19:15]: second source */
        if ((int)wd == r1 && (int)wA >= vd0 && (int)wA < vd0 + U &&
            (int)wB == cvec) {
            *vd_out = (int)wA;
            return (int)i;
        }
    }
    CHECK(0, "%s: no zvbc multiply sequence found", tag);
    return -1;
}

/* Decode the coefficient for the multiply sequence at word seq_at: the
 * nearest preceding vmv.v.x(cvec, rs1) must be immediately preceded by
 * addi(rd==rs1, x0, imm); the backend always broadcasts through T3. */
static int decode_seq_coeff(const uint8_t *mem, int seq_at, int cvec,
                            const char *tag)
{
    for (int i = seq_at - 1; i >= 1; i--) {
        unsigned vvd, rs1;
        if (!dec_vmv_v_x(word_at(mem + 4 * i), &vvd, &rs1) ||
            (int)vvd != cvec)
            continue;
        unsigned rd, rs;
        int imm;
        if (!dec_addi(word_at(mem + 4 * (i - 1)), &rd, &rs, &imm) ||
            rd != rs1 || rs != 0) {
            CHECK(0, "%s: vmv.v.x(cvec) not preceded by addi", tag);
            return -1;
        }
        CHECK(rd == 28, "%s: broadcast scalar reg %u, want T3(28)",
              tag, rd);
        CHECK(imm >= 2 && imm < 256,
              "%s: broadcast coefficient %d out of range", tag, imm);
        return imm;
    }
    CHECK(0, "%s: no coefficient broadcast before multiply", tag);
    return -1;
}

/* Run one MULACC/DELTA ZVBC multiply sequence from the backend's real
 * emitted code through the interpreter and validate vs gf_mul. The
 * 8-word sequence is c_gfmul_acc_zvbc: 5 clmul + 2 xor + (vmv.v.v for
 * first-touch | vxor.vv for accumulate). nacc_lim bounds the
 * accumulator register (n_out*U); kvec carries the 0x1B broadcast. */
static void check_one_mul_seq(const uint8_t *mem, size_t code_size,
                              int r1, int vd0, int U, int cvec, int kvec,
                              int nacc_lim, const char *tag)
{
    int vd = -1;
    int seq_at = find_mul_seq(mem, code_size, r1, vd0, U, cvec, &vd,
                              tag);
    if (seq_at < 0 || failures)
        return;
    int c = decode_seq_coeff(mem, seq_at, cvec, tag);
    if (c < 0 || failures)
        return;

    /* The 8th word decides move vs xor-accumulate and names acc. */
    uint32_t w8 = word_at(mem + 4 * (seq_at + 7));
    int acc = -1, first = -1;
    {
        unsigned d, s;
        if (dec_vmv_v_v(w8, &d, &s) && (int)s == r1) {
            acc = (int)d;
            first = 1;
        }
    }
    if (acc < 0) {
        unsigned d, sA, sB;
        if (dec_vxor_vv(w8, &d, &sA, &sB) &&
            ((int)sA == (int)d || (int)sB == (int)d) &&
            ((int)sA == r1 || (int)sB == r1)) {
            acc = (int)d;
            first = 0;
        }
    }
    CHECK(acc >= 0 && acc < nacc_lim,
          "%s: 8th word not a recognized acc update (0x%08x)", tag,
          w8);
    if (acc < 0 || failures)
        return;

    mini_t m;
    memset(&m, 0, sizeof(m));
    m.vl = 16;
    uint8_t a[16], old[16];
    for (int i = 0; i < 16; i++) {
        a[i] = (uint8_t)xrng();
        old[i] = (uint8_t)(xrng() >> 32);
    }
    memcpy(m.v[vd], a, 16);
    memset(m.v[cvec], (uint8_t)c, 16);
    memset(m.v[kvec], 0x1b, 16);
    memcpy(m.v[acc], old, 16);
    CHECK(mini_run(&m, mem + 4 * seq_at, 8) == 0,
          "%s: interpreter failed on backend sequence", tag);
    for (int i = 0; i < 16 && !failures; i++) {
        uint8_t want = first ? gf_mul((uint8_t)c, a[i])
                             : (uint8_t)(old[i] ^
                                         gf_mul((uint8_t)c, a[i]));
        CHECK(m.v[acc][i] == want,
              "%s lane %d: got 0x%02x want 0x%02x (c=0x%02x "
              "a=0x%02x old=0x%02x first=%d)",
              tag, i, m.v[acc][i], want, c, a[i], old[i], first);
    }
}

/* ---------------- D3. prologue broadcast verification ---------------- */

/* Collect every adjacent addi(rd, x0, imm); vmv.v.x(vd, rd) pair in the
 * emitted code. The backend emits broadcasts only as c_li(T3, val) +
 * vmv.v.x (gen_const_setup prologue, per-column c_broadcast in the
 * body); nothing else in a ZVBC build has this shape, so every decoded
 * pair is a genuine broadcast. Verifies the scalar and the target
 * register -- the D/D2 interpreter checks prime the broadcast regs
 * themselves and would miss a wrong scalar or a wrong target. */
static int collect_broadcasts(const uint8_t *mem, size_t code_size,
                              int *vd_out, int *imm_out, int cap,
                              const char *tag)
{
    int n = 0;
    size_t nwords = code_size / 4;
    for (size_t i = 1; i < nwords && n < cap; i++) {
        unsigned vvd, rs1;
        if (!dec_vmv_v_x(word_at(mem + 4 * i), &vvd, &rs1))
            continue;
        unsigned rd, rs;
        int imm;
        if (!dec_addi(word_at(mem + 4 * (i - 1)), &rd, &rs, &imm) ||
            rd != rs1 || rs != 0)
            continue; /* vmv.v.x fed by something else; not ours */
        CHECK(rd == 28, "%s: broadcast scalar reg %u, want T3(28)",
              tag, rd);
        vd_out[n] = (int)vvd;
        imm_out[n] = imm;
        n++;
    }
    return n;
}

/* Verify the broadcast pairs of one built ZVBC version.
 * kind: P_HORNER (per-row prologue broadcasts into hreg slots),
 * P_MULACC/P_DELTA (per-column body broadcasts into the shared cvec).
 * kvec must be bound to 0x1B and to nothing else. */
static void check_broadcasts(const uint8_t *mem, size_t code_size,
                             const plan_t *pl, int U, int kvec, int cvec,
                             const char *tag)
{
    int vd[256], imm[256];
    int n = collect_broadcasts(mem, code_size, vd, imm, 256, tag);
    CHECK(n > 0, "%s: no broadcast pairs decoded", tag);
    if (failures)
        return;

    int n_kvec_1b = 0;
    for (int i = 0; i < n; i++) {
        if (vd[i] == kvec) {
            CHECK(imm[i] == 0x1b,
                  "%s: kvec bound to 0x%x, want 0x1b", tag, imm[i]);
            n_kvec_1b += (imm[i] == 0x1b);
        }
    }
    CHECK(n_kvec_1b > 0, "%s: no 0x1b -> kvec(v%d) broadcast", tag,
          kvec);

    if (pl->kind == P_HORNER) {
        /* Per-row coefficient broadcasts into the hreg[2j] cvec slots
         * (nontrivial rows only); recompute the slot assignment the
         * same way the backend does (r1/r2/r3/kvec then slots). */
        int n_out = pl->n_out;
        int r = kvec + 1;
        int nrow = 0;
        for (int j = 0; j < n_out; j++) {
            uint8_t c = pl->horner_c[j];
            if (c == 0 || c == 1)
                continue;
            int slot = r++;
            int nslot = 0;
            for (int i = 0; i < n; i++) {
                if (vd[i] != slot)
                    continue;
                CHECK(imm[i] == c,
                      "%s: row %d cvec(v%d) bound to 0x%x, want 0x%02x",
                      tag, j, slot, imm[i], c);
                nslot += (imm[i] == c);
            }
            CHECK(nslot > 0,
                  "%s: row %d (c=0x%02x): no broadcast to cvec(v%d)",
                  tag, j, c, slot);
            nrow++;
        }
        CHECK(nrow > 0, "%s: no row broadcasts checked", tag);
        (void)U;
    } else {
        /* MULACC/DELTA: every cvec broadcast must carry a coefficient
         * that actually occurs (nontrivially) in the plan matrix. */
        int ncols = pl->kind == P_DELTA ? pl->nchanged : pl->n_in;
        int ncvec = 0;
        for (int i = 0; i < n; i++) {
            if (vd[i] != cvec)
                continue;
            int ok = 0;
            for (int e = 0; e < pl->n_out * ncols; e++)
                if (pl->mat[e] == imm[i] && imm[i] >= 2)
                    ok = 1;
            CHECK(ok, "%s: cvec(v%d) bound to 0x%x, not a plan "
                      "coefficient", tag, cvec, imm[i]);
            ncvec++;
        }
        CHECK(ncvec > 0, "%s: no cvec broadcasts decoded", tag);
    }
}

/* D2 + D3 driver: build one MULACC and one DELTA ZVBC twin (first
 * enumerated U), run a real multiply sequence through the interpreter
 * vs gf_mul (D2), and verify the broadcast pairs (D3). These are the
 * historically buggy paths (P0 lived in the MULACC/DELTA ZVBC body). */
static void test_zvbc_mulacc_delta(void)
{
    char err[256];
    /* Non-Vandermonde 2x4 matrix (row-major): forces P_MULACC with
     * nontrivial coefficients (row 1: 2,3,4). */
    static const uint8_t cm[8] = { 1, 1, 1, 1, 1, 2, 3, 4 };
    rj_topology *tm = rj_topo_custom(4, 2, cm);
    /* RS(4,2) parity update of channel 0: P_DELTA with a nontrivial
     * delta coefficient (M[1][0] = 8). */
    rj_topology *td = rj_topo_rs(4, 2);

    rj_cpu_info cpu;
    memset(&cpu, 0, sizeof(cpu));
    cpu.rvv = 1;
    cpu.rvv_zvbc = 1;
    cpu.rvv_zvkg = 1;
    cpu.rvv_zvbc32e = 1;

    /* MULACC case, then DELTA case. */
    for (int ci = 1; ci <= 2 && !failures; ci++) {
        rj_topology *t;
        rj_op op;
        const char *tag;
        plan_kind_t want_kind;
        if (ci == 1) {
            t = tm;
            op = rj_op_encode();
            tag = "mulacc";
            want_kind = P_MULACC;
        } else {
            t = td;
            int ch[1] = { 0 };
            op = rj_op_parity_update(ch, 1);
            tag = "delta";
            want_kind = P_DELTA;
        }
        CHECK(t != NULL, "%s: topo", tag);
        if (!t)
            continue;
        topo_int_t ti;
        CHECK(topo_int_build(&ti, t, err, sizeof(err)) == 0,
              "%s topo: %s", tag, err);
        plan_t pl;
        CHECK(plan_build(&ti, &op, "rs", &pl, err, sizeof(err)) == 0,
              "%s plan: %s", tag, err);
        if (failures) {
            rj_op_free(&op);
            topo_int_free(&ti);
            continue;
        }
        CHECK(pl.kind == want_kind, "%s: kind=%d, want %d", tag,
              (int)pl.kind, (int)want_kind);
        if (pl.kind != want_kind) {
            plan_free(&pl);
            rj_op_free(&op);
            topo_int_free(&ti);
            continue;
        }
        cand_t cands[256];
        int nc = plan_candidates(&pl, &cpu, cands, 256, 1, 128);
        int U = -1;
        for (int i = 0; i < nc; i++)
            if (cands[i].isa == ISA_RVV && cands[i].mul == MUL_ZVBC &&
                !cands[i].remat) {
                U = cands[i].U;
                break;
            }
        CHECK(U > 0, "%s: no ZVBC twin enumerated", tag);
        if (U <= 0) {
            plan_free(&pl);
            rj_op_free(&op);
            topo_int_free(&ti);
            continue;
        }
        version_t v;
        int rc = build_cand(&pl, &cpu, MUL_ZVBC, U, 0, &v, err,
                            sizeof(err));
        CHECK(rc == 0, "%s zvbc build U=%d failed: %s", tag, U, err);
        if (rc == 0) {
            /* Register assignment mirrors src/riscv/jit.c
             * (P_MULACC/P_DELTA ZVBC, nontrivial coefficients). */
            int n_out = pl.n_out;
            int vd0 = n_out * U;
            int r = vd0 + U;
            if (pl.kind == P_DELTA)
                r += 2; /* tlo, thi delta-formation temps */
            int r1 = r, cvec = r + 3, kvec = r + 4;
            check_one_mul_seq((const uint8_t *)v.mem, v.code_size,
                              r1, vd0, U, cvec, kvec, n_out * U, tag);
            check_broadcasts((const uint8_t *)v.mem, v.code_size, &pl,
                             U, kvec, cvec, tag);
            jit_free_version(&v);
        }
        plan_free(&pl);
        rj_op_free(&op);
        topo_int_free(&ti);
    }
    rj_topo_free(tm);
    rj_topo_free(td);
}

/* ---------------- E. interpreter on emitter-built sequences ---------------- */

/* The exact 8-uop sequence the backend emits per ZVBC multiply
 * (c_gfmul_acc_zvbc with first=1), built with the real emitter:
 *   inputs:  v4 = multiplicand lanes, v5 = coefficient broadcast,
 *            v6 = 0x1B broadcast
 *   temps:   v1, v2, v3
 *   output:  v7 = gf lanes */
static size_t emit_zvbc_seq(uint8_t *buf)
{
    em_t e;
    em_init(&e, buf, 256);
    em_vclmul_vv(&e, 1, 4, 5);   /* lo */
    em_vclmulh_vv(&e, 2, 4, 5);  /* hi */
    em_vclmul_vv(&e, 3, 2, 6);   /* q */
    em_vclmulh_vv(&e, 2, 2, 6);  /* qh */
    em_vclmul_vv(&e, 2, 2, 6);   /* n */
    em_vxor_vv(&e, 1, 1, 3);     /* lo^q */
    em_vxor_vv(&e, 1, 1, 2);     /* res */
    em_vmv_v_v(&e, 7, 1);
    return 8;
}

/* The nibble-table multiply sequence (same shape as the backend's
 * non-remat path) built with the real emitter:
 *   inputs:  v4 = multiplicand lanes, v8/v9 = lo/hi tables,
 *            v10 = 0x0f mask broadcast
 *   temps:   v1, v2, v3
 *   output:  v7 */
static size_t emit_nibble_seq(uint8_t *buf)
{
    em_t e;
    em_init(&e, buf, 256);
    em_vand_vv(&e, 1, 4, 10);    /* tlo = x & mask */
    em_vsrl_vi(&e, 2, 4, 4);     /* thi = x >> 4 */
    em_vrgather_vv(&e, 3, 8, 1); /* r1 = tablo[tlo] */
    em_vrgather_vv(&e, 1, 9, 2); /* r1 = tabhi[thi] */
    em_vxor_vv(&e, 1, 3, 1);     /* res = lo ^ hi */
    em_vmv_v_v(&e, 7, 1);
    return 6;
}

static void test_interpreter(void)
{
    /* Nibble log tables for a fixed coefficient, built from gf_mul
     * (same construction as the backend's pool: lo/hi nibble tables). */
    const unsigned C = 0x53;
    uint8_t tablo[16], tabhi[16];
    for (unsigned x = 0; x < 16; x++) {
        tablo[x] = gf_mul(C, (uint8_t)x);
        tabhi[x] = gf_mul(C, (uint8_t)(x << 4));
    }

    uint8_t zcode[256], ncode[256];
    size_t zn = emit_zvbc_seq(zcode);
    size_t nn = emit_nibble_seq(ncode);

    const size_t vlens[] = { 8, 16, 32, 64 };
    for (size_t vi = 0; vi < 4; vi++) {
        size_t vl = vlens[vi];
        for (int trial = 0; trial < 50; trial++) {
            uint8_t a[64];
            for (size_t i = 0; i < vl; i++) {
                /* edge-heavy pattern mix: zeros, all-ones, random */
                unsigned r = (unsigned)xrng();
                a[i] = (r & 3u) == 0 ? 0x00 :
                       (r & 3u) == 1 ? 0xff : (uint8_t)(r >> 16);
            }
            /* ZVBC path */
            mini_t mz;
            memset(&mz, 0, sizeof(mz));
            mz.vl = vl;
            memcpy(mz.v[4], a, vl);
            memset(mz.v[5], C, vl);    /* coefficient broadcast */
            memset(mz.v[6], 0x1b, vl);  /* K broadcast */
            CHECK(mini_run(&mz, zcode, zn) == 0, "zvbc interp failed");
            /* Nibble path */
            mini_t mn;
            memset(&mn, 0, sizeof(mn));
            mn.vl = vl;
            memcpy(mn.v[4], a, vl);
            memcpy(mn.v[8], tablo, 16);
            memcpy(mn.v[9], tabhi, 16);
            memset(mn.v[10], 0x0f, vl); /* mask broadcast */
            CHECK(mini_run(&mn, ncode, nn) == 0, "nibble interp failed");
            if (failures)
                return;
            /* Both vs gf_mul, and vs each other (differential). */
            for (size_t i = 0; i < vl; i++) {
                uint8_t want = gf_mul(C, a[i]);
                CHECK(mz.v[7][i] == want,
                      "vl=%zu i=%zu: zvbc=0x%02x want=0x%02x (a=0x%02x)",
                      vl, i, mz.v[7][i], want, a[i]);
                CHECK(mn.v[7][i] == want,
                      "vl=%zu i=%zu: nibble=0x%02x want=0x%02x", vl, i,
                      mn.v[7][i], want);
                CHECK(mz.v[7][i] == mn.v[7][i],
                      "vl=%zu i=%zu: zvbc/nibble disagree", vl, i);
            }
        }
    }
}

/* ---------------- G. zvbc MULACC/DELTA: no table loads ---------------- */

/* vle8.v, unit stride: funct6=000000, funct3=000, opcode=0000111.
 * Detects a constant-pool table load: c_pooladdr computes the table
 * address into T3 (x28), the pool scratch, and c_ldtab/c_ldtab1 are
 * its only users. The Zvbc path has no tables and must emit no such
 * word. (The P_XOR xorsched path reuses T3 as a stream pointer, but
 * P_XOR has no ZVBC twin, so this scan is scoped to ZVBC twins.) */
static int is_pool_load(uint32_t w)
{
    if ((w & 0xfc00707fu) != 0x00000007u)
        return 0;
    return ((w >> 15) & 31u) == 28; /* T3: pool-address scratch */
}

static void test_zvbc_no_tableloads(void)
{
    char err[256];
    /* Three multiply plan kinds: RS(6,3) Horner, custom non-Vandermonde
     * MULACC, RS(4,2) parity-update P_DELTA (nontrivial delta
     * coefficients). RS(3,1) is P_XOR and has no ZVBC twin by design. */
    struct {
        rj_topology *t;
        int is_pu;
    } cases[3];
    cases[0].t = rj_topo_rs(6, 3);
    cases[0].is_pu = 0;
    /* Non-Vandermonde 2x4 matrix (row-major): forces P_MULACC with
     * nontrivial coefficients. */
    static const uint8_t cm[8] = { 1, 1, 1, 1, 1, 2, 3, 4 };
    cases[1].t = rj_topo_custom(4, 2, cm);
    cases[1].is_pu = 0;
    cases[2].t = rj_topo_rs(4, 2);
    cases[2].is_pu = 1;
    for (int ti2 = 0; ti2 < 3; ti2++) {
        rj_topology *t = cases[ti2].t;
        CHECK(t != NULL, "topo %d", ti2);
        if (!t)
            continue;
        topo_int_t ti;
        CHECK(topo_int_build(&ti, t, err, sizeof(err)) == 0, "topo: %s",
              err);
        rj_op op;
        if (cases[ti2].is_pu) {
            int ch[1] = { 0 };
            op = rj_op_parity_update(ch, 1);
        } else {
            op = rj_op_encode();
        }
        plan_t pl;
        CHECK(plan_build(&ti, &op, "rs", &pl, err, sizeof(err)) == 0,
              "plan: %s", err);
        if (failures) {
            rj_op_free(&op);
            rj_topo_free(t);
            continue;
        }
        /* Pin the plan kinds so the MULACC/DELTA paths stay covered. */
        if (ti2 == 0)
            CHECK(pl.kind == P_HORNER, "case 0: kind=%d, want HORNER",
                  (int)pl.kind);
        else if (ti2 == 1)
            CHECK(pl.kind == P_MULACC, "case 1: kind=%d, want MULACC",
                  (int)pl.kind);
        else
            CHECK(pl.kind == P_DELTA, "case 2: kind=%d, want DELTA",
                  (int)pl.kind);
        rj_cpu_info cpu;
        memset(&cpu, 0, sizeof(cpu));
        cpu.rvv = 1;
        cpu.rvv_zvbc = 1;
        cpu.rvv_zvkg = 1;
        cpu.rvv_zvbc32e = 1;
        cand_t cands[256];
        int nc = plan_candidates(&pl, &cpu, cands, 256, 1, 128);
        int n_zvbc = 0;
        for (int i = 0; i < nc && !failures; i++) {
            if (cands[i].isa != ISA_RVV ||
                cands[i].mul != MUL_ZVBC)
                continue;
            n_zvbc++;
            version_t v;
            int rc = build_cand(&pl, &cpu, MUL_ZVBC, cands[i].U,
                                0, &v, err, sizeof(err));
            CHECK(rc == 0, "zvbc build U=%d failed: %s", cands[i].U,
                  err);
            if (rc != 0)
                continue;
            int npool = 0;
            for (size_t k = 0; k + 4 <= v.code_size; k += 4)
                if (is_pool_load(
                        word_at((const uint8_t *)v.mem + k)))
                    npool++;
            /* Regression: the MULACC/DELTA Zvbc path once emitted
             * nibble-table loads into the unassigned tbl_lo/tbl_hi
             * (v0), clobbering accumulator v0. The Zvbc path needs
             * no constant-pool loads at all. */
            CHECK(npool == 0, "topo %d U=%d (%s): %d pool table loads",
                  ti2, cands[i].U, v.desc, npool);
            jit_free_version(&v);
        }
        CHECK(n_zvbc > 0, "topo %d: no ZVBC twins built", ti2);
        plan_free(&pl);
        rj_op_free(&op);
        topo_int_free(&ti);
        rj_topo_free(t);
    }
}

/* ---------------- F. Zicond absence (B6) ---------------- */

/* czero.eqz/czero.nez: OP (0110011), funct7=0000111, funct3 in
 * {101, 111}. The mask keeps funct7+opcode only (funct3 is checked
 * separately); GAS-verified:
 *   czero.eqz a0,a1,a2 -> 0x0ec5d533
 *   czero.nez t6,t5,t4 -> 0x0fdf7fb3 */
static int is_czero(uint32_t w)
{
    if ((w & 0xfe00007fu) != 0x0e000033u)
        return 0;
    unsigned f3 = (w >> 12) & 7u;
    return f3 == 5 || f3 == 7;
}

/* Positive control: the matcher must fire on real czero words, so the
 * absence scan below can never pass vacuously. */
static void test_czero_matcher(void)
{
    CHECK(is_czero(0x0ec5d533u) != 0, "czero.eqz word not matched");
    CHECK(is_czero(0x0fdf7fb3u) != 0, "czero.nez word not matched");
    /* Near misses must not match: different funct7 (add), different
     * funct3 (sltu), different opcode. */
    CHECK(is_czero(0x00c5d533u) == 0, "add word false-matched");
    /* funct7+opcode of czero but funct3=011 (not 101/111): must not
     * match. */
    CHECK(is_czero(0x0ec53533u) == 0, "f3=3 word false-matched");
    CHECK(is_czero(0x0ec5d537u) == 0, "wrong opcode false-matched");
}

static void test_zicond_absent(void)
{
    char err[256];
    /* Positive control first: the matcher must fire on real czero
     * words, so the absence scan can never pass vacuously. */
    test_czero_matcher();
    if (failures)
        return;
    /* Representative topologies: RS(6,3) Horner, RS(4,2) Horner,
     * RS(3,1) XOR (RAID-5-like). Build every RVV candidate the planner
     * offers and scan for czero words. */
    rj_topology *ts[3];
    ts[0] = rj_topo_rs(6, 3);
    ts[1] = rj_topo_rs(4, 2);
    ts[2] = rj_topo_rs(3, 1);
    for (int ti2 = 0; ti2 < 3; ti2++) {
        rj_topology *t = ts[ti2];
        CHECK(t != NULL, "topo %d", ti2);
        if (!t)
            continue;
        topo_int_t ti;
        CHECK(topo_int_build(&ti, t, err, sizeof(err)) == 0, "topo: %s",
              err);
        rj_op op = rj_op_encode();
        plan_t pl;
        CHECK(plan_build(&ti, &op, "rs", &pl, err, sizeof(err)) == 0,
              "plan: %s", err);
        if (failures) {
            rj_topo_free(t);
            continue;
        }
        /* Two gate settings: nibble twins only, then with the ZVBC
         * twins added (the planner enumerates MUL_ZVBC iff zvbc32e is
         * set). No czero may appear in any twin. */
        for (int zb = 0; zb < 2 && !failures; zb++) {
            rj_cpu_info cpu;
            memset(&cpu, 0, sizeof(cpu));
            cpu.rvv = 1;
            /* zicond advertised: even so, no czero may appear. */
            cpu.rvv_zicond = 1;
            cpu.rvv_zvbc = zb;
            cpu.rvv_zvkg = zb;
            cpu.rvv_zvbc32e = zb;
            cand_t cands[256];
            int nc = plan_candidates(&pl, &cpu, cands, 256, 1, 128);
            CHECK(nc > 0, "no candidates");
            for (int i = 0; i < nc && !failures; i++) {
                if (cands[i].isa != ISA_RVV)
                    continue;
                version_t v;
                memset(&v, 0, sizeof(v));
                rj_options o;
                rj_options_default(&o);
                o.benchmark = 0;
                int rc = jit_build_version(&pl, &cands[i], &cpu, &o, &v,
                                           err, sizeof(err));
                if (rc != 0)
                    continue; /* pressure-failed twins: nothing emitted */
                int ncz = 0;
                for (size_t k = 0; k + 4 <= v.code_size; k += 4)
                    if (is_czero(word_at((const uint8_t *)v.mem + k)))
                        ncz++;
                CHECK(ncz == 0, "zb=%d cand %d (%s): %d czero words "
                                "emitted", zb, i, v.desc, ncz);
                jit_free_version(&v);
            }
        }
        plan_free(&pl);
        rj_op_free(&op);
        topo_int_free(&ti);
        rj_topo_free(t);
    }
}

int main(void)
{
    test_emitter_shape();
    test_lane_model();
    test_planner_twins();
    test_backend_gates();
    test_zvbc_mulacc_delta();
    test_interpreter();
    test_zicond_absent();
    test_zvbc_no_tableloads();
    if (failures == 0)
        printf("test_riscv_zvbc: all green\n");
    return failures != 0;
}

#else /* !__riscv */

int main(void)
{
    printf("test_riscv_zvbc: SKIP (not riscv64)\n");
    return failures != 0;
}

#endif
