/* tests/test_sve_predtail.c
 *
 * B1sve (track item 14): fully predicated SVE tails via whilelt.
 *
 * The predtail twin replaces the scalar byte tail with a whilelt-driven
 * chunk loop: `whilelt p1.b, xzr, x13` builds the predicate for the
 * remaining count, and the vector body runs with p1 governing loads,
 * stores and arithmetic. This test pins the twin's shape and proves it
 * exact:
 *
 *  1. Codegen shape: exactly one WHILELT word decoding to
 *     `whilelt p1.b, xzr, x13`; tail LD1B/ST1B use pg=1 while the general
 *     twin uses pg=0 everywhere; no scalar-tail STRB words remain in the
 *     predtail twin. The whilelo main loop is untouched.
 *  2. Exhaustive correctness: for P_XOR / P_MULACC / P_HORNER / P_DELTA
 *     plans, every length 1..2*group is checked against the scalar
 *     ref_run() oracle AND differentially general-vs-predtail, with
 *     output canaries past len (no over-write) and input preservation
 *     (no stray stores). HORNER twins additionally prove x19 is
 *     preserved across the call (the predtail prologue skips the x19
 *     spill) via an asm wrapper that plants a canary in x19.
 *  3. Guard pages: buffers abut a PROT_NONE page (modulo the 64-byte
 *     align-down gap; see guard_test); tail lengths 1..VL-1 (whole
 *     stripe predicated), VL-1/VL/VL+1 and group-1 run with no fault,
 *     proving predication suppresses every inactive lane.
 *
 * Run under QEMU with SVE, e.g.
 *   QEMU_LD_PREFIX=/usr/aarch64-linux-gnu qemu-aarch64 \
 *       -cpu max,sve-default-vector-length=128 ./build-aarch64/test_sve_predtail
 * Without SVE it prints SKIP and passes.
 * Note: qemu-user 8.2.2 interprets sve-default-vector-length=N as N
 * BYTES and clamps to the 2048-bit architectural max, so flag=512
 * yields 256B: the Makefile's 128/256/512 legs exercise two distinct
 * VLs (128B, 256B), not three.
 *
 * NO PERF CLAIM IS MADE FOR SVE: this test is correctness-only. Whether
 * the predtail twin beats the scalar tail is decided at compile time by
 * the tail duel (bench_tail_duel) on real SVE hardware.
 */
#include "../include/raidjit.h"
#include "../src/plan.h"
#include "../src/jit.h"
#include "../src/cpu.h"
#include "../src/ref.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/mman.h>

static int fails;
#define CHECK(c, ...) do {                                             \
    if (!(c)) {                                                        \
        printf("FAIL "); printf(__VA_ARGS__); printf("\n"); fails++;   \
    }                                                                  \
} while (0)

static uint64_t xrng_state = 0x243F6A8885A308D3ULL;
static uint64_t xrng(void)
{
    uint64_t x = xrng_state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return xrng_state = x;
}

/* True emulated VL in bytes. qemu-user 8.2.2 interprets
 * sve-default-vector-length=N as N BYTES while PR_SVE_GET_VL/rdvl
 * under-report 8x; the functional cntb probe is the ground truth the
 * JIT itself uses (src/aarch64/jit.c, tests/test_boundaries.c). */
static long sve_true_vl_bytes(const rj_cpu_info *cpu)
{
#ifdef __aarch64__
    unsigned long probe = 0;
    __asm__ volatile(".arch_extension sve\n\tcntb %0" : "=r"(probe));
    if (probe >= 16)
        return (long)probe;
#endif
    return cpu->sve_vl > 0 ? (long)cpu->sve_vl : 16;
}

/* ---------------------------------------------------------------- */
/* 1. Codegen shape                                                   */
/* ---------------------------------------------------------------- */

/* WHILELT pd.b, xRn, xRm: 0x25201400 | rm<<16 | rn<<5 | pd.
 * Mask keeps the opcode bits and the size/opc fields, clears the three
 * register fields (distinguishes whilelo 0x25201C00 via bits 11:10). */
#define WHILELT_MASK 0xFFE0FC00u
#define WHILELT_BASE 0x25201400u
/* WHILELO: same mask, base 0x25201C00 (the main loop's p0 predicate;
 * must survive in both twins). */
#define WHILELO_MASK 0xFFE0FC00u
#define WHILELO_BASE 0x25201C00u
/* LD1B/ST1B (immediate offset): mask keeps the opcode (bits 31:13),
 * clears pg (12:10), rn (9:5), zt (4:0). */
#define LD1B_MASK 0xFFFFE000u
#define LD1B_BASE 0xA400A000u
#define ST1B_MASK 0xFFFFE000u
#define ST1B_BASE 0xE400E000u
/* STNT1B (non-temporal): same mask, base 0xE410E000. */
#define STNT1B_MASK 0xFFFFE000u
#define STNT1B_BASE 0xE410E000u
/* STRB (register offset): the scalar tail's store. */
#define STRB_MASK 0xFFE0FC00u
#define STRB_BASE 0x38206800u

static void shape_one(const plan_t *pl, const cand_t *c,
                      const rj_cpu_info *cpu, const rj_options *o,
                      const char *tag)
{
    char err[256];
    version_t vg, vp;
    memset(&vg, 0, sizeof(vg));
    memset(&vp, 0, sizeof(vp));
    cand_t cp = *c;
    cp.predtail = 1;
    CHECK(jit_build_version(pl, c, cpu, o, &vg, err, sizeof(err)) == 0,
          "%s: general build failed: %s", tag, err);
    CHECK(jit_build_version(pl, &cp, cpu, o, &vp, err, sizeof(err)) == 0,
          "%s: predtail build failed: %s", tag, err);
    if (vg.fn == NULL || vp.fn == NULL)
        return;

    CHECK(vp.predtail == 1 && vp.notail == 0, "%s: predtail flags wrong", tag);
    CHECK(vg.predtail == 0, "%s: general twin has predtail set", tag);
    CHECK(strstr(vp.desc, " predtail") != NULL, "%s: desc lacks tag: %s",
          tag, vp.desc);

    const uint32_t *wg = (const uint32_t *)vg.mem;
    const uint32_t *wp = (const uint32_t *)vp.mem;
    size_t ng = vg.code_size / 4, np = vp.code_size / 4;

    /* Exactly one WHILELT in the predtail twin, none in general; it must
     * decode to whilelt p1.b, xzr, x13 (the tail predicate). */
    int nlt_g = 0, nlt_p = 0;
    uint32_t lt_word = 0;
    for (size_t i = 0; i < ng; i++)
        if ((wg[i] & WHILELT_MASK) == WHILELT_BASE)
            nlt_g++;
    for (size_t i = 0; i < np; i++)
        if ((wp[i] & WHILELT_MASK) == WHILELT_BASE) {
            nlt_p++;
            lt_word = wp[i];
        }
    CHECK(nlt_g == 0, "%s: general twin has %d WHILELT", tag, nlt_g);
    CHECK(nlt_p == 1, "%s: predtail twin has %d WHILELT (want 1)", tag,
          nlt_p);
    if (nlt_p == 1) {
        int pd = (int)(lt_word & 15u);
        int rn = (int)((lt_word >> 5) & 31u);
        int rm = (int)((lt_word >> 16) & 31u);
        CHECK(pd == 1 && rn == 31 && rm == 13,
              "%s: WHILELT decodes to p%d.b,x%d,x%d (want p1.b,xzr,x13)",
              tag, pd, rn, rm);
    }

    /* Tail loads/stores use pg=1 in the predtail twin; the general twin
     * never uses pg=1 (its scalar tail is STRB-based). NT twins use
     * STNT1B instead of ST1B. */
    int pg1_ld_g = 0, pg1_ld_p = 0, pg1_st_g = 0, pg1_st_p = 0;
    int pg1_nt_g = 0, pg1_nt_p = 0;
    int nlo_g = 0, nlo_p = 0;
    for (size_t i = 0; i < ng; i++) {
        if ((wg[i] & WHILELO_MASK) == WHILELO_BASE)
            nlo_g++;
        if ((wg[i] & LD1B_MASK) == LD1B_BASE && ((wg[i] >> 10) & 7u) == 1)
            pg1_ld_g++;
        if ((wg[i] & ST1B_MASK) == ST1B_BASE && ((wg[i] >> 10) & 7u) == 1)
            pg1_st_g++;
        if ((wg[i] & STNT1B_MASK) == STNT1B_BASE &&
            ((wg[i] >> 10) & 7u) == 1)
            pg1_nt_g++;
    }
    for (size_t i = 0; i < np; i++) {
        if ((wp[i] & WHILELO_MASK) == WHILELO_BASE)
            nlo_p++;
        if ((wp[i] & LD1B_MASK) == LD1B_BASE && ((wp[i] >> 10) & 7u) == 1)
            pg1_ld_p++;
        if ((wp[i] & ST1B_MASK) == ST1B_BASE && ((wp[i] >> 10) & 7u) == 1)
            pg1_st_p++;
        if ((wp[i] & STNT1B_MASK) == STNT1B_BASE &&
            ((wp[i] >> 10) & 7u) == 1)
            pg1_nt_p++;
    }
    /* The whilelo main loop is untouched by the predtail twin: exactly
     * one WHILELO (p0 = all-true) in both twins. */
    CHECK(nlo_g == 1, "%s: general twin has %d WHILELO (want 1)", tag,
          nlo_g);
    CHECK(nlo_p == 1, "%s: predtail twin has %d WHILELO (want 1)", tag,
          nlo_p);
    CHECK(pg1_ld_g == 0 && pg1_st_g == 0 && pg1_nt_g == 0,
          "%s: general twin uses pg=1 (ld=%d st=%d nt=%d)", tag, pg1_ld_g,
          pg1_st_g, pg1_nt_g);
    CHECK(pg1_ld_p > 0, "%s: predtail twin has no pg=1 LD1B", tag);
    if (c->nt) {
        CHECK(pg1_nt_p > 0 && pg1_st_p == 0,
              "%s: nt predtail twin pg=1 stores wrong (nt=%d st=%d)", tag,
              pg1_nt_p, pg1_st_p);
    } else {
        CHECK(pg1_st_p > 0 && pg1_nt_p == 0,
              "%s: predtail twin pg=1 stores wrong (st=%d nt=%d)", tag,
              pg1_st_p, pg1_nt_p);
    }

    /* No scalar-tail STRB words in the predtail twin; the general twin
     * keeps one per output row. */
    int strb_g = 0, strb_p = 0;
    for (size_t i = 0; i < ng; i++)
        if ((wg[i] & STRB_MASK) == STRB_BASE)
            strb_g++;
    for (size_t i = 0; i < np; i++)
        if ((wp[i] & STRB_MASK) == STRB_BASE)
            strb_p++;
    CHECK(strb_p == 0, "%s: predtail twin has %d scalar-tail STRB", tag,
          strb_p);
    CHECK(strb_g == pl->n_out, "%s: general twin has %d STRB (n_out=%d)",
          tag, strb_g, pl->n_out);

    jit_free_version(&vg);
    jit_free_version(&vp);
    printf("shape %s: OK (whilelt=1 pg1 ld/st, no scalar tail)\n", tag);
}

/* ---------------------------------------------------------------- */
/* 2. Exhaustive correctness + differential + canaries                */
/* ---------------------------------------------------------------- */

/* Call fn with a canary planted in x19 (callee-saved). The predtail
 * twin skips the x19 spill for P_HORNER (the tail does not need it);
 * if any emitted code clobbered x19 this returns nonzero. Returns 1
 * when x19 survived, 0 when it was clobbered. */
static int check_x19_preserved(rj_fn fn, const void * const *in, void **out,
                               size_t len)
{
#ifdef __aarch64__
    uint64_t after;
    __asm__ volatile(
        "mov x0, %1\n\t"
        "mov x1, %2\n\t"
        "mov x2, %3\n\t"
        "movz x19, #0xbeef\n\t"
        "movk x19, #0xdead, lsl #16\n\t"
        "movk x19, #0xcafe, lsl #32\n\t"
        "movk x19, #0xf00d, lsl #48\n\t"
        "blr %4\n\t"
        "mov %0, x19\n\t"
        : "=r"(after)
        : "r"(in), "r"(out), "r"(len), "r"(fn)
        : "x0", "x1", "x2", "x19", "x30", "memory");
    return after == 0xf00dcafedeadbeefULL;
#else
    (void)fn; (void)in; (void)out; (void)len;
    return 1;
#endif
}

/* Run one (general, predtail) twin pair at a single length: scalar-oracle
 * check, general-vs-predtail differential, output canaries past len, and
 * input preservation. Buffers are preallocated with canary fill. */
static void check_len(const plan_t *pl, const version_t *vg,
                      const version_t *vp, uint8_t **inb, uint8_t **outg,
                      uint8_t **outp, uint8_t **want, uint8_t **in_save,
                      size_t len, size_t bufsz, const char *tag)
{
    int n_in = pl->n_in, n_out = pl->n_out;
    for (int i = 0; i < n_in; i++) {
        for (size_t b = 0; b < bufsz; b++)
            inb[i][b] = (uint8_t)(xrng() >> 33);
        memcpy(in_save[i], inb[i], bufsz);
    }
    for (int j = 0; j < n_out; j++) {
        memset(outg[j], 0xA5, bufsz);
        memset(outp[j], 0xA5, bufsz);
    }
    ref_run(pl, (const uint8_t **)inb, want, len);
    vg->fn((const void **)inb, (void **)outg, len);
    vp->fn((const void **)inb, (void **)outp, len);
    for (int j = 0; j < n_out; j++) {
        if (memcmp(outg[j], want[j], len) != 0)
            CHECK(0, "%s len=%zu: general vs scalar ref mismatch (par %d)",
                  tag, len, j);
        if (memcmp(outp[j], want[j], len) != 0)
            CHECK(0, "%s len=%zu: predtail vs scalar ref mismatch (par %d)",
                  tag, len, j);
        if (memcmp(outg[j], outp[j], len) != 0)
            CHECK(0, "%s len=%zu: general vs predtail differ (par %d)",
                  tag, len, j);
        for (size_t b = len; b < bufsz; b++) {
            if (outg[j][b] != 0xA5) {
                CHECK(0, "%s len=%zu: general over-wrote past len (par %d off %zu)",
                      tag, len, j, b);
                break;
            }
            if (outp[j][b] != 0xA5) {
                CHECK(0, "%s len=%zu: predtail over-wrote past len (par %d off %zu)",
                      tag, len, j, b);
                break;
            }
        }
    }
    for (int i = 0; i < n_in; i++)
        if (memcmp(inb[i], in_save[i], bufsz) != 0)
            CHECK(0, "%s len=%zu: input %d modified by JIT", tag, len, i);
}

/* Drive one plan x candidate-shape: exhaustive 1..2*group for U=1 and
 * U=max, boundary sampling for the middle U values, plus the xorsched
 * combination when the planner enumerated one. */
static int drive(const plan_t *pl, cand_t *cands, int nc,
                  const rj_cpu_info *cpu, const rj_options *o,
                  const char *pname)
{
    char err[256], tag[160];
    int umax = 0;
    for (int i = 0; i < nc; i++)
        if (cands[i].isa == ISA_SVE && cands[i].U > umax)
            umax = cands[i].U;
    CHECK(umax > 0, "%s: no plain SVE candidates enumerated", pname);
    if (umax <= 0)
        return 0;

    int tested_shapes = 0;
    int tested_xorsched = 0;
    for (int ci = 0; ci < nc; ci++) {
        cand_t *c = &cands[ci];
        if (c->isa != ISA_SVE)
            continue;
        int exhaustive = (c->U == 1 || c->U == umax);
        version_t vg, vp;
        memset(&vg, 0, sizeof(vg));
        memset(&vp, 0, sizeof(vp));
        cand_t cp = *c;
        cp.predtail = 1;
        snprintf(tag, sizeof(tag), "%s U=%d%s%s%s", pname, c->U,
                 c->xorsched ? " xorsched" : "",
                 c->nt ? " nt" : "",
                 c->prefetch_dist ? " pf" : "");
        if (jit_build_version(pl, c, cpu, o, &vg, err, sizeof(err))) {
            printf("FAIL %s: general build: %s\n", tag, err);
            fails++;
            continue;
        }
        if (jit_build_version(pl, &cp, cpu, o, &vp, err, sizeof(err))) {
            printf("FAIL %s: predtail build: %s\n", tag, err);
            fails++;
            jit_free_version(&vg);
            continue;
        }
        size_t group = vg.group_bytes;
        CHECK(group == (size_t)c->U * (size_t)sve_true_vl_bytes(cpu),
              "%s: group_bytes=%zu != U*VL=%d*%ld", tag, group, c->U,
              sve_true_vl_bytes(cpu));

        size_t bufsz = 2 * group + 64;
        int n_in = pl->n_in, n_out = pl->n_out;
        uint8_t **inb = malloc(sizeof(*inb) * (size_t)n_in);
        uint8_t **in_save = malloc(sizeof(*in_save) * (size_t)n_in);
        uint8_t **outg = malloc(sizeof(*outg) * (size_t)n_out);
        uint8_t **outp = malloc(sizeof(*outp) * (size_t)n_out);
        uint8_t **want = malloc(sizeof(*want) * (size_t)n_out);
        for (int i = 0; i < n_in; i++) {
            inb[i] = malloc(bufsz);
            in_save[i] = malloc(bufsz);
        }
        for (int j = 0; j < n_out; j++) {
            outg[j] = malloc(bufsz);
            outp[j] = malloc(bufsz);
            want[j] = malloc(bufsz);
        }

        if (exhaustive) {
            /* Every length 1..2*group: covers single-chunk tails
             * (len <= VL), multi-chunk tails (rem up to group-1 across
             * 1..2 main-loop iterations), and len < group with no main
             * loop at all. */
            for (size_t len = 1; len <= 2 * group; len++)
                check_len(pl, &vg, &vp, inb, outg, outp, want, in_save,
                          len, bufsz, tag);
            printf("exhaustive %s: 1..%zu OK\n", tag, 2 * group);
        } else {
            /* Boundary sample for middle U values: the tail code path is
             * identical to U=max, only the group differs. */
            long vl = sve_true_vl_bytes(cpu);
            size_t lens[8];
            int nl = 0;
            size_t cand_lens[] = { 1, (size_t)vl - 1, (size_t)vl,
                                   (size_t)vl + 1, group - 1, group,
                                   group + 1, 2 * group };
            for (int k = 0; k < 8; k++)
                if (cand_lens[k] >= 1 && cand_lens[k] <= 2 * group)
                    lens[nl++] = cand_lens[k];
            for (int k = 0; k < nl; k++)
                check_len(pl, &vg, &vp, inb, outg, outp, want, in_save,
                          lens[k], bufsz, tag);
            printf("sampled %s: %d boundary lens OK\n", tag, nl);
        }

        /* x19 preservation: the predtail prologue skips the x19 spill for
         * P_HORNER (only it uses x19); prove the tail never touches it. */
        if (pl->kind == P_HORNER) {
            for (int i = 0; i < n_in; i++)
                for (size_t b = 0; b < bufsz; b++)
                    inb[i][b] = (uint8_t)(xrng() >> 33);
            for (int j = 0; j < n_out; j++)
                memset(outp[j], 0xA5, bufsz);
            CHECK(check_x19_preserved(vp.fn, (const void * const *)inb,
                                      (void **)outp, group + 1),
                  "%s: predtail clobbered x19", tag);
            CHECK(check_x19_preserved(vg.fn, (const void * const *)inb,
                                      (void **)outg, group + 1),
                  "%s: general clobbered x19", tag);
        }

        for (int i = 0; i < n_in; i++) {
            free(inb[i]);
            free(in_save[i]);
        }
        for (int j = 0; j < n_out; j++) {
            free(outg[j]);
            free(outp[j]);
            free(want[j]);
        }
        free(inb); free(in_save); free(outg); free(outp); free(want);
        jit_free_version(&vg);
        jit_free_version(&vp);
        tested_shapes++;
        if (c->xorsched)
            tested_xorsched++;
    }
    CHECK(tested_shapes > 0, "%s: no candidate shapes tested", pname);
    printf("%s: %d candidate shapes tested\n", pname, tested_shapes);
    return tested_xorsched;
}

/* ---------------------------------------------------------------- */
/* 3. Guard pages: predication must suppress every inactive lane      */
/* ---------------------------------------------------------------- */

/* Run the predtail twin with every buffer ending exactly at a PROT_NONE
 * page. Any load/store past len -- i.e. any inactive lane the whilelt
 * predicate failed to suppress -- faults loudly (no signal handler:
 * a crash IS the failure signal). */
static void guard_test(const plan_t *pl, const version_t *vp,
                       const rj_cpu_info *cpu, const char *tag)
{
    long vl = sve_true_vl_bytes(cpu);
    size_t group = vp->group_bytes;
    long pagesz = sysconf(_SC_PAGESIZE);
    CHECK(pagesz >= 4096, "%s: sysconf(_SC_PAGESIZE) failed", tag);
    if (pagesz < 4096)
        return;

    size_t rems[8];
    int nr = 0;
    size_t cand_rems[] = { 1, (size_t)vl - 1, (size_t)vl, (size_t)vl + 1,
                           group - 1 };
    for (int k = 0; k < 5; k++)
        if (cand_rems[k] >= 1 && cand_rems[k] < group)
            rems[nr++] = cand_rems[k];

    int n_in = pl->n_in, n_out = pl->n_out;
    int nbufs = n_in + n_out;
    for (int r = 0; r < nr; r++) {
        size_t rem = rems[r];
        /* Two modes: tail after one full main-loop iteration
         * (len = group + rem), and the whole stripe predicated
         * (len = rem < VL, no main loop at all). */
        for (int mode = 0; mode < 2; mode++) {
            size_t len = (mode == 0) ? group + rem : rem;
            if (mode == 1 && rem >= (size_t)vl)
                continue;
            /* One 3-page mapping per buffer; the last page is PROT_NONE.
             * The 64-byte align-down (API contract) leaves a 0-63 byte
             * mapped gap between buffer end and the guard: a total
             * predication failure over-accesses up to VL-1 >= 127 bytes
             * (VL >= 128B in all legs), so it still faults. Sub-64B
             * over-writes are caught by the exhaustive test's 0xA5
             * canaries; sub-64B over-reads into mapped memory cannot
             * fault and their values are discarded. Adequate in
             * combination; the guard leg alone would be vacuous for
             * over-access narrower than the gap (e.g. a future VL=16B
             * leg). */
            uint8_t **maps = malloc(sizeof(*maps) * (size_t)nbufs);
            uint8_t **inb = malloc(sizeof(*inb) * (size_t)n_in);
            uint8_t **outb = malloc(sizeof(*outb) * (size_t)n_out);
            uint8_t **want = malloc(sizeof(*want) * (size_t)n_out);
            uint8_t **in_save = malloc(sizeof(*in_save) * (size_t)n_in);
            for (int b = 0; b < nbufs; b++) {
                maps[b] = mmap(NULL, (size_t)(3 * pagesz),
                               PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                CHECK(maps[b] != MAP_FAILED, "%s: mmap failed", tag);
                if (maps[b] == MAP_FAILED) {
                    for (int k = 0; k < b; k++)
                        munmap(maps[k], (size_t)(3 * pagesz));
                    free(maps); free(inb); free(outb); free(want);
                    free(in_save);
                    return;
                }
                uint8_t *guard = maps[b] + (size_t)(2 * pagesz);
                if (mprotect(guard, (size_t)pagesz, PROT_NONE)) {
                    printf("FAIL %s: mprotect failed\n", tag);
                    fails++;
                }
                /* Align down to 64 (API contract); note the mask must be
                 * uintptr_t-wide: ~63u is 32-bit and would zero the
                 * pointer's upper half. */
                uintptr_t start =
                    ((uintptr_t)(guard - len)) & ~(uintptr_t)63u;
                uint8_t *buf = (uint8_t *)start;
                for (size_t i = 0; i < len; i++)
                    buf[i] = (uint8_t)(xrng() >> 33);
                if (b < n_in) {
                    inb[b] = buf;
                    in_save[b] = malloc(len ? len : 1);
                    memcpy(in_save[b], buf, len);
                } else {
                    outb[b - n_in] = buf;
                    want[b - n_in] = malloc(len ? len : 1);
                }
            }
            printf("guard %s rem=%zu mode=%s len=%zu: calling...\n", tag,
                   rem, mode == 0 ? "tail" : "stripe", len);
            fflush(stdout);
            ref_run(pl, (const uint8_t **)inb, want, len);
            vp->fn((const void **)inb, (void **)outb, len);
            for (int j = 0; j < n_out; j++)
                if (memcmp(outb[j], want[j], len) != 0)
                    CHECK(0, "%s rem=%zu mode=%d: output mismatch (par %d)",
                          tag, rem, mode, j);
            for (int i = 0; i < n_in; i++)
                if (memcmp(inb[i], in_save[i], len) != 0)
                    CHECK(0, "%s rem=%zu mode=%d: input %d modified", tag,
                          rem, mode, i);
            for (int b = 0; b < nbufs; b++)
                munmap(maps[b], (size_t)(3 * pagesz));
            for (int i = 0; i < n_in; i++)
                free(in_save[i]);
            for (int j = 0; j < n_out; j++)
                free(want[j]);
            free(maps); free(inb); free(outb); free(want); free(in_save);
            printf("guard %s rem=%zu mode=%s: no fault, outputs exact\n",
                   tag, rem, mode == 0 ? "tail" : "stripe");
        }
    }
}

/* Build the U=max plain SVE candidate's predtail twin and run the guard
 * test on it. */
static void guard_drive(const plan_t *pl, cand_t *cands, int nc,
                        const rj_cpu_info *cpu, const rj_options *o,
                        const char *pname)
{
    char err[256], tag[160];
    cand_t *best = NULL;
    for (int i = 0; i < nc; i++)
        if (cands[i].isa == ISA_SVE && !cands[i].nt &&
            !cands[i].prefetch_dist && !cands[i].xorsched &&
            (!best || cands[i].U > best->U))
            best = &cands[i];
    CHECK(best != NULL, "%s: no plain SVE candidate for guard test", pname);
    if (!best)
        return;
    cand_t cp = *best;
    cp.predtail = 1;
    version_t vp;
    memset(&vp, 0, sizeof(vp));
    snprintf(tag, sizeof(tag), "%s U=%d predtail", pname, best->U);
    if (jit_build_version(pl, &cp, cpu, o, &vp, err, sizeof(err))) {
        printf("FAIL %s: predtail build: %s\n", tag, err);
        fails++;
        return;
    }
    guard_test(pl, &vp, cpu, tag);
    jit_free_version(&vp);
}

/* ---------------------------------------------------------------- */
/* main                                                               */
/* ---------------------------------------------------------------- */

static int kinds_seen[4];

static int test_plan(rj_topology *t, rj_op *op, const char *pname,
                      plan_kind_t want_kind, const rj_cpu_info *cpu,
                      const rj_options *o)
{
    char err[256];
    topo_int_t ti;
    if (topo_int_build(&ti, t, err, sizeof(err))) {
        printf("FAIL %s: topo_int_build: %s\n", pname, err);
        fails++;
        return 0;
    }
    plan_t pl;
    if (plan_build(&ti, op, pname, &pl, err, sizeof(err))) {
        printf("FAIL %s: plan_build: %s\n", pname, err);
        fails++;
        topo_int_free(&ti);
        return 0;
    }
    CHECK(pl.kind == want_kind, "%s: plan kind %d, want %d", pname, pl.kind,
          want_kind);
    if (pl.kind >= P_XOR && pl.kind <= P_DELTA)
        kinds_seen[pl.kind]++;

    cand_t cands[64];
    int nc = plan_candidates(&pl, cpu, cands, 64,
                             o->nt_threshold > 0 ? 1 : 0,
                             o->prefetch_distance);
    CHECK(nc > 0, "%s: no candidates enumerated", pname);

    /* Shape: first U=1 and first U=max plain SVE candidates. */
    cand_t *first1 = NULL, *firstmax = NULL, *firstnt = NULL;
    int umax = 0;
    for (int i = 0; i < nc; i++)
        if (cands[i].isa == ISA_SVE && !cands[i].nt &&
            !cands[i].prefetch_dist && !cands[i].xorsched &&
            cands[i].U > umax)
            umax = cands[i].U;
    for (int i = 0; i < nc; i++)
        if (cands[i].isa == ISA_SVE && !cands[i].nt &&
            !cands[i].prefetch_dist && !cands[i].xorsched) {
            if (cands[i].U == 1 && !first1)
                first1 = &cands[i];
            if (cands[i].U == umax && !firstmax)
                firstmax = &cands[i];
        }
    /* P2-1: also shape-test an nt candidate (predicated STNT1B). */
    for (int i = 0; i < nc; i++)
        if (cands[i].isa == ISA_SVE && cands[i].nt &&
            !cands[i].prefetch_dist && !cands[i].xorsched) {
            firstnt = &cands[i];
            break;
        }
    if (first1) {
        char tag[160];
        snprintf(tag, sizeof(tag), "%s U=1", pname);
        shape_one(&pl, first1, cpu, o, tag);
    }
    if (firstmax && firstmax != first1) {
        char tag[160];
        snprintf(tag, sizeof(tag), "%s U=%d", pname, umax);
        shape_one(&pl, firstmax, cpu, o, tag);
    }
    if (firstnt) {
        char tag[160];
        snprintf(tag, sizeof(tag), "%s U=%d nt", pname, firstnt->U);
        shape_one(&pl, firstnt, cpu, o, tag);
    }

    int nxs = drive(&pl, cands, nc, cpu, o, pname);
    guard_drive(&pl, cands, nc, cpu, o, pname);
    plan_free(&pl);
    topo_int_free(&ti);
    return nxs;
}

int main(void)
{
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    long vlprobe = sve_true_vl_bytes(&cpu);
    if (!cpu.sve) {
        printf("test_sve_predtail: SKIP (no SVE)\n");
        return 0;
    }
    printf("test_sve_predtail: SVE detected, true vl=%ld bytes\n",
           vlprobe);
    fflush(stdout);
    rj_options o;
    rj_options_default(&o);
    /* P2-1/P2-3: opt-in dimensions that the predtail twin inherits from
     * the winner. nt_threshold>0 enumerates NT-store twins (predicated
     * STNT1B in the tail); prefetch_distance>0 enumerates prefetch
     * twins (prefetch suppressed in pred mode, but the twin as a whole
     * must still be correct). Both are covered by drive() below. */
    o.nt_threshold = 1;
    o.prefetch_distance = 64;

    rj_topology *t;
    rj_op op;
    int nxs = 0; /* xorsched+predtail combinations exercised */

    /* P_XOR: raid5(6) encode -- single parity row, all coefficients 1. */
    t = rj_topo_raid5(6);
    op = rj_op_encode();
    nxs += test_plan(t, &op, "raid5(6) encode", P_XOR, &cpu, &o);
    rj_op_free(&op);
    rj_topo_free(t);

    /* P_XOR with 2 parity rows sharing a subexpression: exercises
     * xorsched+predtail. Not a real erasure code; the test checks the
     * JIT implements the plan's matrix exactly. */
    {
        static const uint8_t mat[2 * 4] = {
            1, 1, 1, 0,
            1, 1, 0, 1,
        };
        t = rj_topo_custom(4, 2, mat);
        op = rj_op_encode();
        nxs += test_plan(t, &op, "custom-xor(4,2) encode", P_XOR, &cpu, &o);
        rj_op_free(&op);
        rj_topo_free(t);
    }

    /* P_HORNER: rs(6,2) encode -- Vandermonde in Horner order. */
    t = rj_topo_rs(6, 2);
    op = rj_op_encode();
    nxs += test_plan(t, &op, "rs(6,2) encode", P_HORNER, &cpu, &o);
    rj_op_free(&op);
    rj_topo_free(t);

    /* P_MULACC: non-geometric row -> not Horner-Vandermonde. */
    {
        static const uint8_t mat[2 * 4] = {
            1, 1, 1, 1, /* P: xor */
            1, 2, 3, 5, /* Q: not geometric */
        };
        t = rj_topo_custom(4, 2, mat);
        op = rj_op_encode();
        nxs += test_plan(t, &op, "custom(4,2) encode", P_MULACC, &cpu, &o);
        rj_op_free(&op);
        rj_topo_free(t);
    }

    /* P_MULACC with a leading nontrivial first coefficient (P1-1).
     * Row {0,2,3,5} has first nonzero coefficient 2, exercising
     * mulacc_column's first-column nontrivial path (the bare
     * em_sve_eor_unpred) under the predtail twin -- the one branch
     * DESIGN.md §3.2 calls out as load-bearing: a merging predicated
     * EOR there would fold the dead accumulator (stale previous-chunk
     * values in the tail) into active lanes. No other test row has a
     * nontrivial first coefficient, so this path had zero coverage. */
    {
        static const uint8_t mat[2 * 4] = {
            1, 1, 1, 1, /* P: xor */
            0, 2, 3, 5, /* Q: first nonzero coeff is 2 */
        };
        t = rj_topo_custom(4, 2, mat);
        op = rj_op_encode();
        nxs += test_plan(t, &op, "custom(4,2) mulacc-first-nontrivial",
                         P_MULACC, &cpu, &o);
        rj_op_free(&op);
        rj_topo_free(t);
    }

    /* P_DELTA: parity update. */
    {
        int ch[1] = { 0 };
        t = rj_topo_raid5(6);
        op = rj_op_parity_update(ch, 1);
        nxs += test_plan(t, &op, "raid5(6) delta", P_DELTA, &cpu, &o);
        rj_op_free(&op);
        rj_topo_free(t);
    }

    /* P_DELTA with nontrivial coefficients (P2-2): rs(6,2) parity update
     * on changed disk 1 exercises the DELTA+neednib path (c_and3/c_lsr4
     * nibble split + mulacc_column with table loads) under the predtail
     * twin. The raid5 delta above has all-ones coefficients, so its
     * neednib is 0 and the nibble path never runs. */
    {
        int ch[1] = { 1 };
        t = rj_topo_rs(6, 2);
        op = rj_op_parity_update(ch, 1);
        nxs += test_plan(t, &op, "rs(6,2) delta", P_DELTA, &cpu, &o);
        rj_op_free(&op);
        rj_topo_free(t);
    }

    /* Non-vacuous: all four plan kinds were actually exercised. */
    CHECK(kinds_seen[P_XOR] > 0, "P_XOR plan never exercised");
    CHECK(kinds_seen[P_MULACC] > 0, "P_MULACC plan never exercised");
    CHECK(kinds_seen[P_HORNER] > 0, "P_HORNER plan never exercised");
    CHECK(kinds_seen[P_DELTA] > 0, "P_DELTA plan never exercised");
    /* The xorsched+predtail combination must be exercised (api.c builds
     * the predtail twin from the winner, which may be xorsched). */
    CHECK(nxs > 0, "xorsched+predtail combination never exercised");

    if (fails == 0)
        printf("test_sve_predtail: all OK\n");
    else
        printf("test_sve_predtail: %d FAILURES\n", fails);
    return fails != 0;
}
