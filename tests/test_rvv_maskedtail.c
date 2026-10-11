/* RVV maskedtail twin tests.
 *
 * The maskedtail twin replaces the strip-mining remainder loop with an
 * unrolled full-VL masked tail (v0 mask, one chunk per U; see
 * src/riscv/jit.c gen_tail_masked).
 *
 *  1. Shape: for every RVV candidate the planner enumerates, the
 *     maskedtail twin builds; its desc carries " maskedtail"; the code
 *     contains exactly U vid.v and U vmsltu.vx words plus masked (vm=0)
 *     data loads/stores -- none of which appear in the general twin;
 *     v0 is written only by vmsltu.vx; every other vector register
 *     operand is lmul-aligned and nonzero (the v0 reservation).
 *  2. Differential: general vs maskedtail vs the ref_run oracle on ALL
 *     remainders 1..U*VLMAX-1 plus boundary lengths, across
 *     LMUL x U x plan kinds x topologies, with output canaries past len
 *     (no over-write) and input preservation (no stray stores).
 *  3. Guard page: tail buffers abut a PROT_NONE page; masked-off lanes
 *     reaching into it must not fault (masked loads don't over-read,
 *     masked stores don't over-write).
 *
 * RVV-only (needs the riscv64 build; execution needs RVV hardware or
 * QEMU). On other hosts the binary builds and prints SKIP.
 * Run under QEMU at several VLENs, e.g.
 *   QEMU_LD_PREFIX=/usr/riscv64-linux-gnu qemu-riscv64 -cpu max \
 *       ./build-riscv64/test_rvv_maskedtail
 *   ... -cpu max,vlen=256 ... ; ... -cpu max,vlen=512 ...
 *
 * NO PERF CLAIM IS MADE: this test is correctness-only. Whether the
 * maskedtail twin beats the strip-mining tail is decided at compile
 * time by the tail duel (bench_tail_duel) on real RVV hardware.
 */
#include "../include/raidjit.h"
#include "../src/plan.h"
#include "../src/jit.h"
#include "../src/cpu.h"
#include "../src/gf.h"
#include "../src/ref.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/mman.h>

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

static uint64_t xrng_state = 0x243f6a8885a308d3ULL;
static uint64_t xrng(void)
{
    uint64_t x = xrng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return xrng_state = x;
}

static size_t read_vlenb(void)
{
    size_t v;
    __asm__ volatile("csrr %0, vlenb" : "=r"(v));
    return v;
}

static void *abuf(size_t n)
{
    void *p = NULL;
    if (posix_memalign(&p, 64, n ? n : 64)) {
        printf("oom\n");
        exit(1);
    }
    return p;
}

static void randfill(uint8_t *p, size_t n)
{
    for (size_t i = 0; i < n; i++)
        p[i] = (uint8_t)xrng();
}

/* ---------------- 1. shape ---------------- */

/* vid.v: OPMVV (f3=010), funct6=010100, vs1=10001, vm=1. Mask keeps the
 * opcode fields, clears vd. GAS-verified: vid.v v9 -> 0x5208a4d7. */
#define VID_MASK 0xfffff07fu
#define VID_BASE 0x5208a057u
/* vmsltu.vx: OPIVX (f3=100), funct6=011010, vm=1. Mask keeps the opcode
 * fields, clears vs2/rs1 (vd is always v0). GAS-verified:
 * vmsltu.vx v0,v9,a3 -> 0x6a96c057. */
#define VMSLTU_MASK 0xfe00707fu
#define VMSLTU_BASE 0x6a004057u

static void shape_one(const plan_t *pl, const cand_t *c,
                      const rj_cpu_info *cpu, const char *tag)
{
    char err[256];
    rj_options o;
    rj_options_default(&o);
    o.benchmark = 0;
    version_t vg, vm;
    memset(&vg, 0, sizeof(vg));
    memset(&vm, 0, sizeof(vm));
    cand_t cm = *c;
    cm.maskedtail = 1;
    CHECK(jit_build_version(pl, c, cpu, &o, &vg, err, sizeof(err)) == 0,
          "%s: general build failed: %s", tag, err);
    int mt_built =
        jit_build_version(pl, &cm, cpu, &o, &vm, err, sizeof(err)) == 0;
    if (vg.fn == NULL)
        return;
    /* The twin must build whenever the v0 reservation fits; when it
     * doesn't fit the backend must fail cleanly (never wrong code). */
    int lmul = c->lmul ? c->lmul : 1;
    size_t vlmax = (size_t)lmul * read_vlenb();
    if (vlmax > 255) {
        /* Mask math (vmsltu.vx at SEW=8) is exact only for VLMAX <= 255:
         * the backend must reject the twin on wider configs, never
         * build wrong code. */
        CHECK(!mt_built, "%s: maskedtail built with VLMAX=%zu > 255",
              tag, vlmax);
        jit_free_version(&vg);
        return;
    }
    int need_mt = plan_need_regs(ISA_RVV, c->mul, pl, c->U, c->remat,
                                 c->pipe, 1, 0) * lmul;
    if (need_mt <= 32) {
        CHECK(mt_built, "%s: maskedtail twin should fit (%d regs) but "
              "build failed: %s", tag, need_mt, err);
    }
    if (!mt_built) {
        jit_free_version(&vg);
        return;
    }

    CHECK(vm.maskedtail == 1 && vm.notail == 0, "%s: twin flags wrong", tag);
    CHECK(vg.maskedtail == 0, "%s: general twin has maskedtail set", tag);
    CHECK(strstr(vm.desc, " maskedtail") != NULL,
          "%s: desc lacks ' maskedtail': %s", tag, vm.desc);
    CHECK(strstr(vg.desc, " maskedtail") == NULL,
          "%s: general desc has spurious ' maskedtail': %s", tag, vg.desc);
    CHECK(vm.group_bytes == vg.group_bytes, "%s: group_bytes differ", tag);

    const uint32_t *wg = (const uint32_t *)vg.mem;
    const uint32_t *wm = (const uint32_t *)vm.mem;
    size_t ng = vg.code_size / 4, nm = vm.code_size / 4;
    int nvid_g = 0, nvid_m = 0, ncmp_g = 0, ncmp_m = 0;
    int nmasked_g = 0, nmasked_m = 0;
    for (size_t i = 0; i < ng; i++) {
        if ((wg[i] & VID_MASK) == VID_BASE)
            nvid_g++;
        if ((wg[i] & VMSLTU_MASK) == VMSLTU_BASE)
            ncmp_g++;
        unsigned op = wg[i] & 0x7fu;
        if ((op == 0x07u || op == 0x27u) && ((wg[i] >> 25) & 1u) == 0)
            nmasked_g++;
    }
    for (size_t i = 0; i < nm; i++) {
        if ((wm[i] & VID_MASK) == VID_BASE)
            nvid_m++;
        if ((wm[i] & VMSLTU_MASK) == VMSLTU_BASE)
            ncmp_m++;
        unsigned op = wm[i] & 0x7fu;
        if ((op == 0x07u || op == 0x27u) && ((wm[i] >> 25) & 1u) == 0)
            nmasked_m++;
    }
    CHECK(nvid_g == 0 && ncmp_g == 0,
          "%s: general twin has vid.v/vmsltu.vx (%d/%d)", tag, nvid_g,
          ncmp_g);
    CHECK(nmasked_g == 0, "%s: general twin has %d masked mem ops", tag,
          nmasked_g);
    /* Exactly one vid.v + one vmsltu.vx per unrolled chunk. */
    CHECK(nvid_m == c->U, "%s: maskedtail has %d vid.v (want U=%d)", tag,
          nvid_m, c->U);
    CHECK(ncmp_m == c->U, "%s: maskedtail has %d vmsltu.vx (want U=%d)", tag,
          ncmp_m, c->U);
    CHECK(nmasked_m > 0, "%s: maskedtail has no masked mem ops", tag);

    /* v0 reservation: the only vector instruction writing v0 is
     * vmsltu.vx; every other vector register operand is nonzero and
     * lmul-aligned. Vector/scalar field classification per funct3:
     * OPIVV/OPFVV (0-1): vd/vs1/vs2 vector; OPMVV (2): vd/vs1/vs2 vector
     * except vid.v (vs1 field is the 10001 selector, vs2 field is the
     * reserved-0 encoding, neither a register); OPIVI (3): vd vector,
     * vs1 field is imm/selector (not a reg), vs2 vector;
     * OPIVX/OPMVX (4/6): vd/vs2 vector, vs1 field is a GPR. */
    int nvec = 0;
    for (size_t i = 0; i < nm; i++) {
        uint32_t w = wm[i];
        unsigned op = w & 0x7fu;
        if (op == 0x07u) { /* vle8.v: vd */
            int vd = (int)((w >> 7) & 31u);
            nvec++;
            CHECK(vd != 0, "%s: vle8.v targets v0 at +%zu", tag, i);
            CHECK(vd % lmul == 0, "%s: vle8.v vd=v%d not %d-aligned",
                  tag, vd, lmul);
        } else if (op == 0x27u) { /* vse8.v: vs3 */
            int vs3 = (int)((w >> 7) & 31u);
            nvec++;
            CHECK(vs3 != 0, "%s: vse8.v sources v0 at +%zu", tag, i);
            CHECK(vs3 % lmul == 0, "%s: vse8.v vs3=v%d not %d-aligned",
                  tag, vs3, lmul);
        } else if (op == 0x57u) {
            unsigned f3 = (w >> 12) & 7u;
            if (f3 == 0x7u)
                continue; /* vsetvli/vsetivli: no vector operands */
            int vd = (int)((w >> 7) & 31u);
            int vs1 = (int)((w >> 15) & 31u);
            int vs2 = (int)((w >> 20) & 31u);
            unsigned f6 = (w >> 26) & 0x3fu;
            int is_vid = ((w & VID_MASK) == VID_BASE);
            /* vmv.v.v/v.x/v.i (funct6=010111) are single-source moves:
             * the vs2 field is architecturally 0, not a v0 read. */
            int is_vmv = (f6 == 0x17u);
            nvec++;
            if (vd == 0) {
                CHECK((w & VMSLTU_MASK) == VMSLTU_BASE,
                      "%s: v0 written by non-vmsltu.vx 0x%08x at +%zu",
                      tag, w, i);
            } else {
                CHECK(vd % lmul == 0, "%s: vd=v%d at +%zu not %d-aligned",
                      tag, vd, i, lmul);
            }
            /* vs1 is a vector reg only for f3=0,1,2, except vid.v (vs1
             * field is the 10001 selector, not a register). */
            if (!is_vid && f3 <= 0x2u) {
                CHECK(vs1 != 0, "%s: vector vs1=v0 at +%zu", tag, i);
                CHECK(vs1 % lmul == 0,
                      "%s: vs1=v%d at +%zu not %d-aligned", tag, vs1, i,
                      lmul);
            }
            /* vs2 is a vector reg for f3=0,1,2,4,6 and OPIVI,
             * except vmv (vs2 field unused) and vid.v (vs2 field is
             * the reserved-0 encoding, not a register read). */
            if (!is_vmv && !is_vid &&
                (f3 <= 0x2u || f3 == 0x4u || f3 == 0x6u || f3 == 0x3u)) {
                CHECK(vs2 != 0, "%s: vector vs2=v0 at +%zu", tag, i);
                CHECK(vs2 % lmul == 0,
                      "%s: vs2=v%d at +%zu not %d-aligned", tag, vs2, i,
                      lmul);
            }
        }
    }
    CHECK(nvec > 0, "%s: no vector instructions decoded", tag);

    jit_free_version(&vg);
    jit_free_version(&vm);
}

/* ---------------- 2. differential execution ---------------- */

/* Build general + maskedtail versions once, then run the length list.
 * lens/len_nl: explicit lengths (NULL => full remainder sweep
 * 0..U*VLMAX-1 plus the multi-group boundary lengths). */
/* One differential length: general vs maskedtail vs ref_run, with
 * output canaries past len and input preservation. Buffers are caller-
 * owned; in[] is restored from in_snap[] around each run. */
static void exec_check_len(const plan_t *pl, const version_t *vg,
                           const version_t *vm, uint8_t **in,
                           uint8_t **in_snap, uint8_t **out_g,
                           uint8_t **out_m, uint8_t **exp, size_t alen,
                           size_t L, const char *tag)
{
    int n_in_total = (pl->kind == P_DELTA) ? 2 * pl->nchanged + pl->n_out
                                           : pl->n_in;
    int np = pl->n_out;
    const void *vin[96];
    void *vog[32], *vom[32];
    for (int i = 0; i < n_in_total; i++)
        vin[i] = in[i];
    for (int j = 0; j < np; j++) {
        vog[j] = out_g[j];
        vom[j] = out_m[j];
    }
    for (int j = 0; j < np; j++) {
        memset(out_g[j], 0xA5, alen);
        memset(out_m[j], 0xA5, alen);
        memset(exp[j], 0xA5, alen);
    }
    for (int i = 0; i < n_in_total; i++)
        memcpy(in[i], in_snap[i], alen);
    vg->fn(vin, vog, L);
    for (int i = 0; i < n_in_total; i++)
        memcpy(in[i], in_snap[i], alen);
    vm->fn(vin, vom, L);
    ref_run(pl, (const uint8_t **)in_snap, exp, L);
    for (int j = 0; j < np; j++) {
        if (L && memcmp(out_g[j], exp[j], L) != 0)
            CHECK(0, "%s len=%zu: general != ref (parity %d)", tag, L,
                  j);
        if (L && memcmp(out_m[j], exp[j], L) != 0)
            CHECK(0, "%s len=%zu: maskedtail != ref (parity %d)", tag, L,
                  j);
        if (L && memcmp(out_g[j], out_m[j], L) != 0)
            CHECK(0, "%s len=%zu: general != maskedtail (parity %d)",
                  tag, L, j);
        /* canaries past len: no over-write */
        for (size_t k = L; k < alen; k++) {
            if (out_m[j][k] != 0xA5) {
                CHECK(0, "%s len=%zu: maskedtail overwrote past len "
                      "(parity %d, off %zu)", tag, L, j, k);
                break;
            }
        }
    }
    /* input preservation */
    for (int i = 0; i < n_in_total; i++)
        if (memcmp(in[i], in_snap[i], alen) != 0)
            CHECK(0, "%s len=%zu: input %d mutated", tag, L, i);
}

/* Build general + maskedtail versions once, then run every length:
 * the full remainder sweep 0..U*VLMAX-1 followed by multi-group
 * boundary lengths (these exercise main-loop + tail interaction). */
static void exec_sweep(const plan_t *pl, const cand_t *c,
                       const rj_cpu_info *cpu, const char *tag)
{
    char err[256];
    rj_options o;
    rj_options_default(&o);
    o.benchmark = 0;
    version_t vg, vm;
    memset(&vg, 0, sizeof(vg));
    memset(&vm, 0, sizeof(vm));
    cand_t cm = *c;
    cm.maskedtail = 1;
    if (jit_build_version(pl, c, cpu, &o, &vg, err, sizeof(err)) != 0) {
        CHECK(0, "%s: general build failed: %s", tag, err);
        return;
    }
    if (jit_build_version(pl, &cm, cpu, &o, &vm, err, sizeof(err)) != 0) {
        jit_free_version(&vg);
        return; /* pressure: twin doesn't fit; shape test covers it */
    }

    int n_in_total = (pl->kind == P_DELTA) ? 2 * pl->nchanged + pl->n_out
                                           : pl->n_in;
    int np = pl->n_out;
    size_t g = vg.group_bytes;
    int lmul = c->lmul ? c->lmul : 1;
    size_t vlmax = (size_t)lmul * read_vlenb();
    size_t sweep_max = (size_t)c->U * vlmax; /* exclusive */
    size_t alen = 2 * g + 128;
    if (alen < 256)
        alen = 256;
    uint8_t *in[96], *in_snap[96], *out_g[32], *out_m[32], *exp[32];
    for (int i = 0; i < n_in_total; i++) {
        in[i] = abuf(alen);
        in_snap[i] = abuf(alen);
        randfill(in[i], alen);
        memcpy(in_snap[i], in[i], alen);
    }
    for (int j = 0; j < np; j++) {
        out_g[j] = abuf(alen);
        out_m[j] = abuf(alen);
        exp[j] = abuf(alen);
    }

    for (size_t L = 0; L < sweep_max && !failures; L++)
        exec_check_len(pl, &vg, &vm, in, in_snap, out_g, out_m, exp,
                       alen, L, tag);
    size_t b[] = { g - 1, g, g + 1, 2 * g - 1, 2 * g };
    for (size_t k = 0; k < sizeof(b) / sizeof(b[0]) && !failures; k++)
        exec_check_len(pl, &vg, &vm, in, in_snap, out_g, out_m, exp,
                       alen, b[k], tag);

    for (int i = 0; i < n_in_total; i++) {
        free(in[i]);
        free(in_snap[i]);
    }
    for (int j = 0; j < np; j++) {
        free(out_g[j]);
        free(out_m[j]);
        free(exp[j]);
    }
    jit_free_version(&vg);
    jit_free_version(&vm);
}

/* ---------------- 3. guard page ---------------- */

/* Buffers abut a PROT_NONE page; masked-off lanes reaching into it must
 * not fault. Exercises the exact over-read/over-write the mask
 * suppresses: the last tail chunk's full-VLMAX access straddles the
 * guard page whenever len % VLMAX != 0. */
static void guard_one(const plan_t *pl, const cand_t *c,
                      const rj_cpu_info *cpu, const char *tag)
{
    char err[256];
    rj_options o;
    rj_options_default(&o);
    o.benchmark = 0;
    version_t vm;
    memset(&vm, 0, sizeof(vm));
    cand_t cm = *c;
    cm.maskedtail = 1;
    if (jit_build_version(pl, &cm, cpu, &o, &vm, err, sizeof(err)) != 0)
        return;

    int nd = pl->n_in, np = pl->n_out, nc = pl->nchanged;
    int n_in_total = (pl->kind == P_DELTA) ? 2 * nc + np : nd;
    size_t g = vm.group_bytes;
    int lmul = c->lmul ? c->lmul : 1;
    size_t vlmax = (size_t)lmul * read_vlenb();
    long ps = sysconf(_SC_PAGESIZE);

    size_t test_lens[] = { 1, vlmax - 1, vlmax, vlmax + 1,
                           (size_t)c->U * vlmax - 1, g - 1 };
    for (size_t ti = 0; ti < sizeof(test_lens) / sizeof(test_lens[0]);
         ti++) {
        size_t len = test_lens[ti];
        /* data region then one guard page; buffers end ps-vlmax before
         * the guard so a full-VLMAX masked access at the tail reaches
         * exactly to (not past) the guard boundary on active lanes and
         * into it on masked-off lanes. */
        int nbufs0 = n_in_total + np;
        size_t slot0 = len + vlmax;
        size_t region =
            ((size_t)nbufs0 * slot0 + (size_t)ps - 1) & ~((size_t)ps - 1);
        uint8_t *base =
            mmap(NULL, region + (size_t)ps, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (base == MAP_FAILED) {
            CHECK(0, "%s: mmap failed", tag);
            break;
        }
        if (mprotect(base + region, (size_t)ps, PROT_NONE) != 0) {
            CHECK(0, "%s: mprotect failed", tag);
            munmap(base, region + (size_t)ps);
            break;
        }
        /* Pack each buffer in its own slot of (len + vlmax) bytes; the
         * last slot ends exactly at the guard page. A full-VLMAX masked
         * access at [len, len+vlmax) from any buffer start then reaches
         * into the guard page on masked-off lanes. */
        int nbufs = n_in_total + np;
        size_t slot = len + vlmax;
        uint8_t *in[96], *out[32];
        const void *vin[96];
        void *vout[32];
        for (int i = 0; i < n_in_total; i++) {
            in[i] = base + region - (size_t)(nbufs - i) * slot;
            randfill(in[i], len);
            vin[i] = in[i];
        }
        for (int j = 0; j < np; j++) {
            out[j] = base + region - (size_t)(np - j) * slot;
            memset(out[j], 0xA5, len);
            vout[j] = out[j];
        }
        /* A fault here means the mask failed to suppress an inactive
         * lane: the test dies loudly (no CHECK can run after SIGSEGV,
         * which is the point -- silence would be a missed bug). */
        vm.fn(vin, vout, len);
        /* output bytes [0,len) were written; verify against ref */
        uint8_t *exp[32];
        for (int j = 0; j < np; j++)
            exp[j] = abuf(len ? len : 1);
        ref_run(pl, (const uint8_t **)vin, exp, len);
        for (int j = 0; j < np; j++) {
            if (memcmp(out[j], exp[j], len) != 0)
                CHECK(0, "%s guard len=%zu: mismatch (parity %d)", tag,
                      len, j);
            free(exp[j]);
        }
        munmap(base, region + (size_t)ps);
        if (failures)
            break;
    }
    jit_free_version(&vm);
}

/* ---------------- driver ---------------- */

static void test_topo(rj_topology *t, rj_op op, const char *name,
                      const rj_cpu_info *cpu)
{
    char err[256];
    topo_int_t ti;
    if (topo_int_build(&ti, t, err, sizeof(err))) {
        printf("FAIL %s topo: %s\n", name, err);
        failures++;
        return;
    }
    plan_t pl;
    if (plan_build(&ti, &op, name, &pl, err, sizeof(err))) {
        printf("FAIL %s plan: %s\n", name, err);
        failures++;
        topo_int_free(&ti);
        return;
    }
    cand_t cands[128];
    int nc = plan_candidates(&pl, cpu, cands, 128, 0, 0);
    CHECK(nc > 0, "%s: no candidates", name);
    int tested = 0;
    for (int ci = 0; ci < nc && !failures; ci++) {
        if (cands[ci].isa != ISA_RVV)
            continue;
        char tag[128];
        snprintf(tag, sizeof(tag), "%s U=%d lmul=%d mul=%d", name,
                 cands[ci].U, cands[ci].lmul, cands[ci].mul);
        shape_one(&pl, &cands[ci], cpu, tag);
        tested++;
        /* Full remainder sweep 0..U*VLMAX-1 plus multi-group boundary
         * lengths; versions are built once per candidate. */
        if (!failures)
            exec_sweep(&pl, &cands[ci], cpu, tag);
        if (!failures)
            guard_one(&pl, &cands[ci], cpu, tag);
    }
    CHECK(tested > 0, "%s: no RVV candidate tested", name);
    plan_free(&pl);
    topo_int_free(&ti);
}

int main(void)
{
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    if (!cpu.rvv) {
        printf("SKIP test_rvv_maskedtail (no RVV)\n");
        return 0;
    }
    /* RAID-5: P_XOR. */
    {
        rj_topology *t = rj_topo_raid5(4);
        rj_op op = rj_op_encode();
        if (t) {
            test_topo(t, op, "raid5", &cpu);
            rj_topo_free(t);
        }
        rj_op_free(&op);
    }
    /* RAID-6: P_MULACC (nibble + Zvbc when present). */
    {
        rj_topology *t = rj_topo_raid6(4);
        rj_op op = rj_op_encode();
        if (t) {
            test_topo(t, op, "raid6", &cpu);
            rj_topo_free(t);
        }
        rj_op_free(&op);
    }
    /* RS(6,3): P_HORNER. */
    {
        rj_topology *t = rj_topo_rs(6, 3);
        rj_op op = rj_op_encode();
        if (t) {
            test_topo(t, op, "rs63", &cpu);
            rj_topo_free(t);
        }
        rj_op_free(&op);
    }
    /* RAID-6 parity update: P_DELTA. */
    {
        int ch[] = { 0, 2 };
        rj_topology *t = rj_topo_raid6(4);
        rj_op op = rj_op_parity_update(ch, 2);
        if (t) {
            test_topo(t, op, "raid6-delta", &cpu);
            rj_topo_free(t);
        }
        rj_op_free(&op);
    }
    if (failures == 0)
        printf("PASS test_rvv_maskedtail\n");
    else
        printf("FAIL test_rvv_maskedtail: %d failures\n", failures);
    return failures != 0;
}

#else /* !__riscv */

int main(void)
{
    (void)failures;
    printf("SKIP test_rvv_maskedtail (not a riscv64 build)\n");
    return 0;
}

#endif
