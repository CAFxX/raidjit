/* test_codegen_shape: structural verification of F1/F2/F5 codegen.
 *
 * These tests verify the *shape* of the generated code (instruction counts
 * and patterns), not just correctness:
 *   F1: P_HORNER i==0 does one input load per (i,u) + vmov per row
 *       (not one load per (j,u)).
 *   F2: GFNI MULACC loads each distinct table once (not once per use).
 *   F5: vzeroupper emitted on AVX/AVX2/AVX-512, absent on SSE.
 *
 * The tests walk the emitted machine code with a length decoder (adapted
 * from test_scan_align.c) and count specific instruction patterns. They
 * are fail-capable: reverting the optimization changes the counts and
 * fails the test (verified by red-green during development).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "../src/jit.h"
#include "../src/plan.h"
#include "../include/raidjit.h"

static int failures = 0;
#define CHECK(c, ...) do {                                             \
    if (!(c)) {                                                        \
        printf("FAIL "); printf(__VA_ARGS__); printf("\n"); failures++; \
    }                                                                  \
} while (0)

/* length of ModRM+SIB+disp starting at p[0]=ModRM; 0 = truncated */
static size_t mrm_len(const uint8_t *p, size_t avail)
{
    if (avail < 1)
        return 0;
    unsigned mod = p[0] >> 6, rm = p[0] & 7;
    size_t n = 1;
    if (mod == 3)
        return 1;
    if (rm == 4) { /* SIB follows */
        if (avail < 2)
            return 0;
        n = 2;
        if (mod == 0 && (p[1] & 7) == 5) { /* disp32 */
            if (avail < 6)
                return 0;
            return 6;
        }
    } else if (mod == 0 && rm == 5) { /* disp32 (RIP-relative) */
        if (avail < 5)
            return 0;
        return 5;
    }
    if (mod == 1) {
        if (avail < n + 1)
            return 0;
        return n + 1;
    }
    if (mod == 2) {
        if (avail < n + 4)
            return 0;
        return n + 4;
    }
    return n;
}

typedef struct {
    size_t len;   /* 0 = undecodable/truncated */
    int pp;       /* legacy mandatory prefix / VEX.vvvv pp: 0,1=66,2=F3,3=F2 */
    int opcode;   /* first opcode byte */
    int map;      /* VEX/EVEX map: 1=0F, 2=0F38, 3=0F3A; 0 if not VEX/EVEX */
    int has_mrm;
    int mod;      /* valid iff has_mrm */
    int rm;       /* valid iff has_mrm */
} insn_t;

/* decode one instruction; covers the opcode universe src/x86_64/emit.c produces
 * (adapted from test_scan_align.c; 0F 18 prefetch added per peer review).
 * Fail-closed: unknown patterns return len 0. */
static insn_t decode(const uint8_t *p, size_t avail)
{
    insn_t d;
    memset(&d, 0, sizeof d);
    size_t i = 0;
    int rex_w = 0;

    /* VEX / EVEX prefixes (no legacy prefix may precede them here) */
    if (avail > 0 && (p[0] == 0xC4 || p[0] == 0xC5 || p[0] == 0x62)) {
        int is_c4 = p[0] == 0xC4, is_evex = p[0] == 0x62;
        size_t ph = is_evex ? 4 : (is_c4 ? 3 : 2); /* prefix bytes */
        if (avail < ph + 2)
            return d;
        unsigned p0 = p[1];
        /* VEX: pp = P1[1:0] (C4) or P0[1:0] (C5); EVEX: pp = P1[1:0].
         * The map lives in the prefix: VEX C4 P0[4:0] (1..3 valid),
         * EVEX P0[1:0] (1=0F, 2=0F38, 3=0F3A). Exactly one opcode byte
         * follows the prefix in all cases; imm8 (if any) comes after
         * the ModRM. */
        unsigned p1 = is_evex ? p[2] : p[is_c4 ? 2 : 1];
        int map = is_evex ? (int)(p0 & 3)
                          : (is_c4 ? (int)(p0 & 0x1F) : 1);
        if (map < 1 || map > 3)
            return d;
        d.map = map;
        d.pp = (int)(p1 & 3);
        size_t o = ph;
        d.opcode = p[o];
        o += 1;
        if (d.opcode == 0x77) {
            /* VZEROUPPER / VZEROALL: no ModRM byte follows the opcode. */
            d.len = o;
            return d;
        }
        size_t ml = mrm_len(p + o, avail - o);
        if (!ml)
            return d;
        d.has_mrm = 1;
        d.mod = p[o] >> 6;
        d.rm = p[o] & 7;
        o += ml;
        /* imm8 group/affine/ternlog forms */
        if ((map == 1 && d.opcode == 0x71) ||
            (map == 3 && (d.opcode == 0xCE || d.opcode == 0x25))) {
            if (avail < o + 1)
                return d;
            o += 1;
        }
        d.len = o;
        return d;
    }

    /* legacy prefixes */
    for (;;) {
        if (i >= avail)
            return d;
        unsigned b = p[i];
        if (b == 0x66) {
            d.pp = 1;
            i++;
        } else if (b == 0xF3) {
            d.pp = 2;
            i++;
        } else if (b == 0xF2) {
            d.pp = 3;
            i++;
        } else if (b >= 0x40 && b <= 0x4F) {
            rex_w = (b >> 3) & 1;
            i++;
        } else
            break;
    }
    if (i >= avail)
        return d;
    unsigned b = p[i];

    if (b == 0x0F) { /* two/three-byte opcodes */
        if (avail - i < 2)
            return d;
        unsigned op2 = p[i + 1];
        size_t o = i + 2;
        if (op2 >= 0x80 && op2 <= 0x8F) { /* jcc rel32 */
            if (avail - i < 6)
                return d;
            d.len = i + 6;
            return d;
        }
        switch (op2) {
        case 0x1E: /* endbr64: F3 0F 1E FA */
            if (avail - i < 3 || p[i + 2] != 0xFA || d.pp != 2)
                return d;
            d.len = i + 3;
            return d;
        case 0x01: /* F3 0F 01 EA/E9: setssbsy/clrssbsy (ModRM, mod=3) */
            if (avail - i < 3 || d.pp != 2 ||
                (p[i + 2] != 0xEA && p[i + 2] != 0xE9))
                return d;
            d.len = i + 3;
            return d;
        case 0x0B: /* ud2 */
            d.len = i + 2;
            return d;
        case 0x38: /* 0F38 escape */
            if (avail - i < 3 || p[i + 2] != 0x00)
                return d;
            d.opcode = 0x00;
            o = i + 3;
            break;
        case 0x6F:
        case 0x7F:
        case 0xE7: /* movntdq (66 0F E7): aligned NT store */
        case 0xEF:
        case 0xDB:
            d.opcode = (int)op2;
            break;
        case 0xAE: /* group 5: /7 = sfence (F8), no vector memory op */
            d.opcode = 0xAE;
            break;
        case 0x71: /* group: ModRM + imm8 */
            d.opcode = 0x71;
            {
                size_t ml = mrm_len(p + o, avail - o);
                if (!ml || avail - o - ml < 1)
                    return d;
                d.has_mrm = 1;
                d.mod = p[o] >> 6;
        d.rm = p[o] & 7;
                d.len = o + ml + 1;
                return d;
            }
        case 0x18: /* prefetch (ModRM, no immediate) */
            d.opcode = (int)op2;
            break;
        case 0x1F: /* multi-byte nop /0 */
        case 0xB6: /* movzx */
            d.opcode = (int)op2;
            break;
        default:
            return d;
        }
        size_t ml = mrm_len(p + o, avail - o);
        if (!ml)
            return d;
        d.has_mrm = 1;
        d.mod = p[o] >> 6;
        d.rm = p[o] & 7;
        d.len = o + ml;
        return d;
    }

    /* single-byte opcodes */
    if (b >= 0x50 && b <= 0x57) { /* push r */
        d.len = i + 1;
        return d;
    }
    if (b >= 0x58 && b <= 0x5F) { /* pop r */
        d.len = i + 1;
        return d;
    }
    switch (b) {
    case 0x90: /* nop (also 66 90) */
    case 0xC3: /* ret */
    case 0xCC: /* int3: 1-byte trap in the post-ret trap sled */
        d.len = i + 1;
        return d;
    case 0xE9: /* jmp rel32 */
        if (avail - i < 5)
            return d;
        d.len = i + 5;
        return d;
    case 0x8B:
    case 0x89:
    case 0x8D:
    case 0x03:
    case 0x33:
    case 0x3B:
    case 0x85:
    case 0x32:
    case 0x8A:
    case 0x88:
    case 0xFF: {
        size_t ml = mrm_len(p + i + 1, avail - i - 1);
        if (!ml)
            return d;
        d.len = i + 1 + ml;
        return d;
    }
    case 0xC7: { /* /0 + imm32 */
        size_t ml = mrm_len(p + i + 1, avail - i - 1);
        if (!ml || avail - i - 1 - ml < 4)
            return d;
        d.len = i + 1 + ml + 4;
        return d;
    }
    case 0x83:   /* /r + imm8 */
    case 0xC1: { /* /r + imm8 */
        size_t ml = mrm_len(p + i + 1, avail - i - 1);
        if (!ml || avail - i - 1 - ml < 1)
            return d;
        d.len = i + 1 + ml + 1;
        return d;
    }
    case 0x81: { /* /r + imm32 */
        size_t ml = mrm_len(p + i + 1, avail - i - 1);
        if (!ml || avail - i - 1 - ml < 4)
            return d;
        d.len = i + 1 + ml + 4;
        return d;
    }
    default:
        break;
    }
    if (b >= 0xB8 && b <= 0xBF) { /* mov r, imm */
        size_t im = rex_w ? 8 : 4;
        if (avail - i < 1 + im)
            return d;
        d.len = i + 1 + im;
        return d;
    }
    return d; /* unknown opcode: fail closed */
}

/* Walk code, counting 0x6F (vmovdqa/movdqa) by operand type. */
typedef struct {
    int n_load;      /* 0x6F, mod != 3, base=R10 (input loads) */
    int n_tbl_load;  /* 0x6F, mod != 3, base=R15 (nibble table loads) */
    int n_pool_load; /* 0x6F, RIP-relative (constant pool loads) */
    int n_move;      /* 0x6F, mod == 3 (reg-reg moves) */
    int n_vzeroupper;/* C5 F8 77 byte sequence */
    int n_ternlog;   /* EVEX.0F3A 0x25 (VPTERNLOGD) */
} counts_t;

/* Base register for a VEX/EVEX-encoded memory operand. Returns 0-15, or -1
 * if RIP-relative / not a vector memory operand. */
static int vex_mem_base(const uint8_t *p, const insn_t *d)
{
    if (!d->has_mrm || d->mod == 3)
        return -1;
    int is_evex = p[0] == 0x62;
    int is_c4 = p[0] == 0xC4;
    int is_c5 = p[0] == 0xC5;
    if (!is_evex && !is_c4 && !is_c5)
        return -1;
    size_t ph;
    int B;
    if (is_evex) {
        ph = 4;
        B = ((p[1] >> 5) & 1) ^ 1; /* B' inverted in EVEX P0 */
    } else if (is_c4) {
        ph = 3;
        B = ((p[1] >> 5) & 1) ^ 1; /* B' inverted */
    } else {
        ph = 2;
        B = 0; /* 2-byte VEX has no B bit */
    }
    size_t o = ph + 1; /* ModRM */
    unsigned mrm = p[o];
    unsigned mod = mrm >> 6, rm = mrm & 7;
    if (mod == 0 && rm == 5)
        return -1; /* RIP-relative */
    if (rm == 4) {
        unsigned sib = p[o + 1];
        return (int)((sib & 7) | (unsigned)(B << 3));
    }
    return (int)(rm | (unsigned)(B << 3));
}

static int scan_counts(const version_t *v, counts_t *c)
{
    memset(c, 0, sizeof *c);
    const uint8_t *p = (const uint8_t *)v->mem;
    size_t n = v->code_size;
    size_t i = 0;

    /* vzeroupper: C5 F8 77 (see em_vzeroupper) */
    for (size_t k = 0; k + 3 <= n; k++) {
        if (p[k] == 0xC5 && p[k+1] == 0xF8 && p[k+2] == 0x77)
            c->n_vzeroupper++;
    }

    while (i < n) {
        insn_t d = decode(p + i, n - i);
        if (!d.len) {
            printf("FAIL: undecodable at offset %zu in %s\n", i, v->desc);
            return -1;
        }
        if (d.map == 3 && d.opcode == 0x25)
            c->n_ternlog++; /* VPTERNLOGD (EVEX.66.0F3A.W0 25 /r ib) */
        if (d.opcode == 0x6F && d.has_mrm) {
            if (d.mod == 3)
                c->n_move++;
            else if (d.mod == 0 && d.rm == 5)
                c->n_pool_load++;
            else {
                int base = vex_mem_base(p + i, &d);
                if (base == 10)
                    c->n_load++;
                else if (base == 15)
                    c->n_tbl_load++;
                else
                    c->n_load++; /* unknown base: count as load (fail-safe) */
            }
        }
        i += d.len;
    }
    return 0;
}

/* Build the first candidate matching (isa, U, mul, ternlog) for a plan.
 * Returns 1 on success. */
static int build_cand(const plan_t *pl, const rj_cpu_info *cpu, isa_t isa,
                      int U, mul_t mul, int ternlog, version_t *v, char *err,
                      size_t errlen)
{
    cand_t cands[128];
    int nc = plan_candidates(pl, cpu, cands, 128, 1, 0);
    rj_options opts;
    rj_options_default(&opts);
    opts.benchmark = 0;
    for (int i = 0; i < nc; i++) {
        if (cands[i].isa != isa || cands[i].U != U)
            continue;
        if ((pl->kind == P_MULACC || pl->kind == P_HORNER ||
             pl->kind == P_DELTA) && cands[i].mul != mul)
            continue;
        if (!!cands[i].ternlog != !!ternlog)
            continue;
        if (jit_build_version(pl, &cands[i], cpu, &opts, v, err, errlen) == 0)
            return 1;
        printf("FAIL: build %d: %s\n", i, err);
        failures++;
        return 0;
    }
    printf("FAIL: no candidate isa=%d U=%d mul=%d ternlog=%d\n", isa, U, mul,
           ternlog);
    failures++;
    return 0;
}

/* F1: P_HORNER i==0 loads once per (i,u) into t, then vmovs per row.
 * raid6(6): n_in=6, n_out=2. With U=1:
 *   input loads: n_in*U = 6 (one per (i,u); without F1 it would be 7).
 *   reg-reg moves: at least n_out*U = 2 (the F1 vmovs; without F1, 0
 *   from i==0, though other paths may add some).
 * The load count is the precise F1 invariant. */
static void test_f1(void)
{
    rj_cpu_info cpu;
    memset(&cpu, 0, sizeof cpu);
    cpu.avx = cpu.avx2 = 1;

    rj_topology *topo = rj_topo_raid6(6);
    rj_op op = rj_op_encode();
    topo_int_t ti;
    char err[256];
    if (topo_int_build(&ti, topo, err, sizeof err)) {
        printf("FAIL f1: topo: %s\n", err);
        failures++;
        return;
    }
    plan_t pl;
    if (plan_build(&ti, &op, "enc", &pl, err, sizeof err)) {
        printf("FAIL f1: plan: %s\n", err);
        failures++;
        topo_int_free(&ti);
        return;
    }

    version_t v;
    if (!build_cand(&pl, &cpu, ISA_AVX2, 1, MUL_NIBBLE, 0, &v, err, sizeof err)) {
        plan_free(&pl);
        topo_int_free(&ti);
        return;
    }
    counts_t c;
    if (scan_counts(&v, &c) == 0) {
        /* n_in=6 inputs, U=1: F1 gives 6 R10-based input loads (one per
         * (i,u)). Without F1 the i==0 path would do n_out=2 loads instead
         * of 1, totaling 7. Table loads (R15) are excluded from the count. */
        CHECK(c.n_load == 6, "f1 [%s]: n_input_loads=%d, want 6", v.desc, c.n_load);
        /* F1 adds (n_out-1)*U=1 reg-reg moves on i==0: the input is loaded
         * directly into acc[0] and moved to the remaining rows (not
         * load-to-scratch + n_out moves). */
        CHECK(c.n_move >= 1, "f1 [%s]: n_moves=%d, want >=1", v.desc, c.n_move);
        printf("f1 [%s]: loads=%d tbl_loads=%d moves=%d pool=%d OK\n",
               v.desc, c.n_load, c.n_tbl_load, c.n_move, c.n_pool_load);
    }
    jit_free_version(&v);
    plan_free(&pl);
    topo_int_free(&ti);
    rj_topo_free(topo);
    rj_op_free(&op);
}

/* F2: GFNI MULACC loads each distinct table once. Custom 2x1 plan with
 * mat=[[5],[5]] (same coefficient in both rows), U=1:
 *   pool loads: 1 (the table for c=5, loaded once and reused).
 *   Without F2 it would be 2 (one per row). */
static void test_f2(void)
{
    rj_cpu_info cpu;
    memset(&cpu, 0, sizeof cpu);
    cpu.avx = cpu.avx2 = 1;
    cpu.avx512f = cpu.avx512bw = cpu.avx512vl = cpu.gfni = 1;

    uint8_t m[2] = {5, 5};
    rj_topology *topo = rj_topo_custom(2, 1, m);
    rj_op op = rj_op_encode();
    topo_int_t ti;
    char err[256];
    if (topo_int_build(&ti, topo, err, sizeof err)) {
        printf("FAIL f2: topo: %s\n", err);
        failures++;
        return;
    }
    plan_t pl;
    if (plan_build(&ti, &op, "enc", &pl, err, sizeof err)) {
        printf("FAIL f2: plan: %s\n", err);
        failures++;
        topo_int_free(&ti);
        return;
    }
    if (pl.kind != P_MULACC) {
        printf("FAIL f2: expected P_MULACC, got kind %d\n", pl.kind);
        failures++;
        plan_free(&pl);
        topo_int_free(&ti);
        return;
    }

    version_t v;
    if (!build_cand(&pl, &cpu, ISA_AVX512, 1, MUL_GFNI, 0, &v, err, sizeof err)) {
        plan_free(&pl);
        topo_int_free(&ti);
        return;
    }
    counts_t c;
    if (scan_counts(&v, &c) == 0) {
        /* One distinct table (c=5); F2 loads it once from the pool
         * (R15-based). Without F2 the two rows would each load it
         * (2 loads). Nibble tables are absent on the GFNI path. */
        CHECK(c.n_tbl_load == 1, "f2 [%s]: n_pool_loads=%d, want 1",
              v.desc, c.n_tbl_load);
        printf("f2 [%s]: pool_loads=%d OK\n", v.desc, c.n_tbl_load);
    }
    jit_free_version(&v);
    plan_free(&pl);
    topo_int_free(&ti);
    rj_topo_free(topo);
    rj_op_free(&op);
}

/* F5: vzeroupper (C5 F8 77) present on AVX/AVX2/AVX-512, absent on SSE. */
static void test_f5(void)
{
    rj_topology *topo = rj_topo_raid6(6);
    rj_op op = rj_op_encode();
    topo_int_t ti;
    char err[256];
    if (topo_int_build(&ti, topo, err, sizeof err)) {
        printf("FAIL f5: topo: %s\n", err);
        failures++;
        return;
    }
    plan_t pl;
    if (plan_build(&ti, &op, "enc", &pl, err, sizeof err)) {
        printf("FAIL f5: plan: %s\n", err);
        failures++;
        topo_int_free(&ti);
        return;
    }

    struct { isa_t isa; int has_avx; const char *nm; } cfgs[] = {
        { ISA_SSE, 0, "SSE" },
        { ISA_AVX2, 1, "AVX2" },
        { ISA_AVX512, 1, "AVX-512" },
    };
    for (unsigned k = 0; k < sizeof cfgs / sizeof cfgs[0]; k++) {
        rj_cpu_info cpu;
        memset(&cpu, 0, sizeof cpu);
        if (cfgs[k].isa == ISA_SSE) {
            cpu.sse2 = cpu.ssse3 = 1;
        } else if (cfgs[k].isa == ISA_AVX2) {
            cpu.avx = cpu.avx2 = 1;
        } else {
            cpu.avx = cpu.avx2 = 1;
            cpu.avx512f = cpu.avx512bw = cpu.avx512vl = 1;
        }
        version_t v;
        if (!build_cand(&pl, &cpu, cfgs[k].isa, 1, MUL_NIBBLE, 0, &v, err,
                      sizeof err))
            continue;
        counts_t c;
        if (scan_counts(&v, &c) == 0) {
            if (cfgs[k].has_avx) {
                CHECK(c.n_vzeroupper >= 1,
                      "f5 [%s]: vzeroupper missing on %s", v.desc, cfgs[k].nm);
            } else {
                CHECK(c.n_vzeroupper == 0,
                      "f5 [%s]: vzeroupper present on %s (want 0)",
                      v.desc, cfgs[k].nm);
            }
            printf("f5 [%s]: vzeroupper=%d (%s) OK\n",
                   v.desc, c.n_vzeroupper, cfgs[k].nm);
        }
        jit_free_version(&v);
    }
    plan_free(&pl);
    topo_int_free(&ti);
    rj_topo_free(topo);
    rj_op_free(&op);
}

/* VPTERNLOGD twin: custom 4x2 P_XOR plan, mat=[[1,1,1,1],[1,1,0,0]],
 * AVX-512, U=1. The ternlog twin must emit exactly 3 VPTERNLOGD (pair
 * (0,1): both rows use both columns -> 2; pair (2,3): only row 0 uses
 * both -> 1) and its desc must carry the +ternlog tag; the plain twin
 * must emit none. Also: the planner must never produce a ternlog
 * candidate for a non-AVX-512 ISA. */
static void test_tl(void)
{
    rj_cpu_info cpu;
    memset(&cpu, 0, sizeof cpu);
    cpu.avx = cpu.avx2 = 1;
    cpu.avx512f = cpu.avx512bw = cpu.avx512vl = 1;

    uint8_t m[8] = {1, 1, 1, 1, 1, 1, 0, 0};
    rj_topology *topo = rj_topo_custom(4, 2, m);
    rj_op op = rj_op_encode();
    topo_int_t ti;
    char err[256];
    if (topo_int_build(&ti, topo, err, sizeof err)) {
        printf("FAIL tl: topo: %s\n", err);
        failures++;
        return;
    }
    plan_t pl;
    if (plan_build(&ti, &op, "enc", &pl, err, sizeof err)) {
        printf("FAIL tl: plan: %s\n", err);
        failures++;
        topo_int_free(&ti);
        return;
    }
    if (pl.kind != P_XOR) {
        printf("FAIL tl: expected P_XOR, got kind %d\n", pl.kind);
        failures++;
        plan_free(&pl);
        topo_int_free(&ti);
        return;
    }

    /* Planner gating: no ternlog candidate may have a non-AVX-512 ISA. */
    {
        cand_t cands[128];
        int nc = plan_candidates(&pl, &cpu, cands, 128, 1, 0);
        int ntl = 0;
        for (int i = 0; i < nc; i++) {
            if (cands[i].ternlog) {
                ntl++;
                CHECK(cands[i].isa == ISA_AVX512,
                      "tl: ternlog candidate %d has isa %d (want AVX-512)",
                      i, cands[i].isa);
            }
        }
        CHECK(ntl > 0, "tl: no ternlog candidate enumerated (want >0)");
    }

    version_t v;
    if (build_cand(&pl, &cpu, ISA_AVX512, 1, MUL_NIBBLE, 1, &v, err,
                   sizeof err)) {
        counts_t c;
        if (scan_counts(&v, &c) == 0) {
            CHECK(c.n_ternlog == 3,
                  "tl [%s]: n_vpternlogd=%d, want 3", v.desc, c.n_ternlog);
            CHECK(strstr(v.desc, "+ternlog") != NULL,
                  "tl [%s]: desc lacks +ternlog tag", v.desc);
            printf("tl [%s]: vpternlogd=%d OK\n", v.desc, c.n_ternlog);
        }
        jit_free_version(&v);
    }
    if (build_cand(&pl, &cpu, ISA_AVX512, 1, MUL_NIBBLE, 0, &v, err,
                   sizeof err)) {
        counts_t c;
        if (scan_counts(&v, &c) == 0) {
            CHECK(c.n_ternlog == 0,
                  "tl [%s]: plain twin has n_vpternlogd=%d, want 0",
                  v.desc, c.n_ternlog);
            CHECK(strstr(v.desc, "+ternlog") == NULL,
                  "tl [%s]: plain desc wrongly has +ternlog tag", v.desc);
            printf("tl [%s]: plain vpternlogd=0 OK\n", v.desc);
        }
        jit_free_version(&v);
    }
    plan_free(&pl);
    topo_int_free(&ti);
    rj_topo_free(topo);
    rj_op_free(&op);

    /* Gating: with n_out == 1 the twin is statically worse (vzero
     * overhead, no amortization), so no ternlog candidate may be
     * enumerated. */
    {
        uint8_t m1[4] = {1, 1, 1, 1};
        rj_topology *topo1 = rj_topo_custom(4, 1, m1);
        topo_int_t ti1;
        if (topo_int_build(&ti1, topo1, err, sizeof err)) {
            printf("FAIL tl: topo1: %s\n", err);
            failures++;
            rj_topo_free(topo1);
            return;
        }
        plan_t pl1;
        if (plan_build(&ti1, &op, "enc", &pl1, err, sizeof err)) {
            printf("FAIL tl: plan1: %s\n", err);
            failures++;
            topo_int_free(&ti1);
            rj_topo_free(topo1);
            return;
        }
        cand_t cands[128];
        int nc = plan_candidates(&pl1, &cpu, cands, 128, 1, 0);
        for (int i = 0; i < nc; i++)
            CHECK(!cands[i].ternlog,
                  "tl: n_out=1 candidate %d has ternlog=1 (want 0)", i);
        printf("tl: n_out=1 gating OK (%d candidates, none ternlog)\n", nc);
        plan_free(&pl1);
        topo_int_free(&ti1);
        rj_topo_free(topo1);
    }
}

/* Pointer-hoisting shape: count scalar `mov r64,[rX+disp]` pointer-array
 * loads by (base, dst).
 *   n_hoist:      base=R8, dst in {RSI,RDI,R12,R13,R14} (prologue hoists)
 *   n_inptr_loop: base=R8, dst in {R10,RDX} (per-iteration input loads;
 *                 the scalar tail uses RAX, so it is excluded)
 *   n_outptr:     base=R9, dst=R10 (output loads, body + tail)
 * raid6(6) encode HORNER (n_in=6, n_out=2), U=1:
 *   hoisted:   (5, 1, 4) -- 5 prologue hoists; input 5 falls back to a
 *              per-iteration R10 load; 2 body + 2 tail output loads.
 *   unhoisted: (0, 6, 4).
 * P_DELTA is excluded from hoisting by design: n_hoist must be 0 there. */
typedef struct { int n_hoist, n_inptr_loop, n_outptr; } ptr_counts_t;

static void scan_ptr_loads(const version_t *v, ptr_counts_t *c)
{
    memset(c, 0, sizeof *c);
    const uint8_t *p = (const uint8_t *)v->mem;
    size_t n = v->code_size, i = 0;
    while (i < n) {
        insn_t d = decode(p + i, n - i);
        if (!d.len) {
            printf("FAIL: undecodable at offset %zu in %s\n", i, v->desc);
            failures++;
            return;
        }
        /* REX.W + 0x8B: mov r64, r/m64. */
        unsigned rex = 0;
        size_t o = 0;
        if (p[i] >= 0x40 && p[i] <= 0x4F) {
            rex = p[i];
            o = 1;
        }
        if ((rex & 0x08) && i + o + 2 <= n && p[i + o] == 0x8B) {
            unsigned mrm = p[i + o + 1];
            unsigned mod = mrm >> 6, rm = mrm & 7;
            if (mod != 3 && !(mod == 0 && rm == 5)) {
                int reg = (int)(((mrm >> 3) & 7) | (((rex >> 2) & 1) << 3));
                int base;
                if (rm == 4) {
                    if (i + o + 3 > n) {
                        i += d.len;
                        continue;
                    }
                    unsigned sib = p[i + o + 2];
                    base = (int)((sib & 7) | ((rex & 1) << 3));
                } else {
                    base = (int)(rm | ((rex & 1) << 3));
                }
                if (base == 8) {
                    if (reg == 6 || reg == 7 || reg == 12 || reg == 13 ||
                        reg == 14)
                        c->n_hoist++;
                    else if (reg == 10 || reg == 2)
                        c->n_inptr_loop++;
                } else if (base == 9 && reg == 10) {
                    c->n_outptr++;
                }
            }
        }
        i += d.len;
    }
}

static void test_ptr_hoist(void)
{
    rj_cpu_info cpu;
    memset(&cpu, 0, sizeof cpu);
    cpu.avx = cpu.avx2 = 1;

    rj_topology *topo = rj_topo_raid6(6);
    rj_op op = rj_op_encode();
    topo_int_t ti;
    char err[256];
    if (topo_int_build(&ti, topo, err, sizeof err)) {
        printf("FAIL hoist: topo: %s\n", err);
        failures++;
        return;
    }
    plan_t pl;
    if (plan_build(&ti, &op, "enc", &pl, err, sizeof err)) {
        printf("FAIL hoist: plan: %s\n", err);
        failures++;
        topo_int_free(&ti);
        return;
    }

    version_t v;
    if (!build_cand(&pl, &cpu, ISA_AVX2, 1, MUL_NIBBLE, 0, &v, err, sizeof err)) {
        plan_free(&pl);
        topo_int_free(&ti);
        return;
    }
    ptr_counts_t c;
    scan_ptr_loads(&v, &c);
    /* n_in=6 -> 5 hoisted; input 5 keeps its per-iteration R10 load;
     * 2 body + 2 tail output loads via R10. */
    CHECK(c.n_hoist == 5, "hoist [%s]: n_hoist=%d, want 5", v.desc, c.n_hoist);
    CHECK(c.n_inptr_loop == 1, "hoist [%s]: n_inptr_loop=%d, want 1",
          v.desc, c.n_inptr_loop);
    CHECK(c.n_outptr == 4, "hoist [%s]: n_outptr=%d, want 4",
          v.desc, c.n_outptr);
    printf("hoist [%s]: hoist=%d inptr_loop=%d outptr=%d OK\n",
           v.desc, c.n_hoist, c.n_inptr_loop, c.n_outptr);
    jit_free_version(&v);
    plan_free(&pl);
    topo_int_free(&ti);
    rj_topo_free(topo);
    rj_op_free(&op);

    /* P_DELTA is excluded from hoisting by design. */
    {
        int ch[1] = { 0 };
        rj_topology *topo2 = rj_topo_raid6(4);
        rj_op op2 = rj_op_parity_update(ch, 1);
        topo_int_t ti2;
        if (topo_int_build(&ti2, topo2, err, sizeof err)) {
            printf("FAIL hoist-delta: topo: %s\n", err);
            failures++;
            return;
        }
        plan_t pl2;
        if (plan_build(&ti2, &op2, "pu1", &pl2, err, sizeof err)) {
            printf("FAIL hoist-delta: plan: %s\n", err);
            failures++;
            topo_int_free(&ti2);
            return;
        }
        version_t v2;
        if (!build_cand(&pl2, &cpu, ISA_AVX2, 1, MUL_NIBBLE, 0, &v2, err,
                        sizeof err)) {
            plan_free(&pl2);
            topo_int_free(&ti2);
            return;
        }
        ptr_counts_t c2;
        scan_ptr_loads(&v2, &c2);
        CHECK(c2.n_hoist == 0, "hoist-delta [%s]: n_hoist=%d, want 0",
              v2.desc, c2.n_hoist);
        CHECK(c2.n_inptr_loop > 0, "hoist-delta [%s]: n_inptr_loop=%d, want >0",
              v2.desc, c2.n_inptr_loop);
        printf("hoist-delta [%s]: hoist=%d inptr_loop=%d outptr=%d OK\n",
               v2.desc, c2.n_hoist, c2.n_inptr_loop, c2.n_outptr);
        jit_free_version(&v2);
        plan_free(&pl2);
        topo_int_free(&ti2);
        rj_topo_free(topo2);
        rj_op_free(&op2);
    }
}

/* Coefficient-aware allocation: P_DELTA with an all-{0,1} delta matrix
 * (RAID-5 parity update) must not allocate the dead nibble temps.
 *   - plan_need_regs drops by exactly the dead regs: AVX2/AVX-512 nibble
 *     and GFNI lose tlo/thi/r1/r2/tbl/mask (6 regs); SSE loses 5, keeping
 *     r1 as the delta temp dt (the two-operand old^new needs it).
 *   - the built version has an EMPTY constant pool: the mask/table setup
 *     is absent (was already gated; this pins it).
 *   - the backend assert inside jit_build_version (vmax+1 ==
 *     plan_need_regs) proves the emitter's numbering agrees with the
 *     formula, so a successful build is itself allocation evidence.
 *   - the SSE build is EXECUTED against the scalar reference, proving
 *     the kept delta temp dt (= r1) is really wired up.
 *   - nontrivial control (RAID-6 delta): setup present, old counts.
 * Fail-capable: reverting the backend allocation (unconditional t..t+6)
 * trips the vmax assert (abort); reverting the formula breaks the CHECKs
 * (verified by red-green during development). */
static void test_delta_trivial(void)
{
    int ch[1] = { 0 };
    rj_topology *topo = rj_topo_raid5(4);
    rj_op op = rj_op_parity_update(ch, 1);
    topo_int_t ti;
    char err[256];
    if (topo_int_build(&ti, topo, err, sizeof err)) {
        printf("FAIL dta: topo: %s\n", err);
        failures++;
        return;
    }
    plan_t pl;
    if (plan_build(&ti, &op, "r5pu", &pl, err, sizeof err)) {
        printf("FAIL dta: plan: %s\n", err);
        failures++;
        topo_int_free(&ti);
        return;
    }
    if (pl.kind != P_DELTA) {
        printf("FAIL dta: expected P_DELTA, got kind %d\n", pl.kind);
        failures++;
        plan_free(&pl);
        topo_int_free(&ti);
        return;
    }

    /* AVX2 nibble, U=1: need = 1*1+1 = 2 (was 8); empty pool. */
    {
        rj_cpu_info cpu;
        memset(&cpu, 0, sizeof cpu);
        cpu.avx = cpu.avx2 = 1;
        version_t v;
        if (build_cand(&pl, &cpu, ISA_AVX2, 1, MUL_NIBBLE, 0, &v, err,
                       sizeof err)) {
            CHECK(plan_need_regs(ISA_AVX2, MUL_NIBBLE, &pl, 1, 0, 0, 0) == 2,
                  "dta [%s]: need=%d, want 2",
                  v.desc, plan_need_regs(ISA_AVX2, MUL_NIBBLE, &pl, 1, 0, 0, 0));
            CHECK(v.const_size == 0,
                  "dta [%s]: const_size=%zu, want 0 (mask/table setup dead)",
                  v.desc, v.const_size);
            printf("dta [%s]: need=2 const_size=0 OK\n", v.desc);
            jit_free_version(&v);
        }
    }
    /* AVX-512 GFNI, U=1: t2/tbl dead too; need = 2 (was 4). */
    {
        rj_cpu_info cpu;
        memset(&cpu, 0, sizeof cpu);
        cpu.avx = cpu.avx2 = 1;
        cpu.avx512f = cpu.avx512bw = cpu.avx512vl = cpu.gfni = 1;
        version_t v;
        if (build_cand(&pl, &cpu, ISA_AVX512, 1, MUL_GFNI, 0, &v, err,
                       sizeof err)) {
            CHECK(plan_need_regs(ISA_AVX512, MUL_GFNI, &pl, 1, 0, 0, 0) == 2,
                  "dta [%s]: need=%d, want 2",
                  v.desc, plan_need_regs(ISA_AVX512, MUL_GFNI, &pl, 1, 0, 0, 0));
            CHECK(v.const_size == 0,
                  "dta [%s]: const_size=%zu, want 0", v.desc, v.const_size);
            printf("dta [%s]: need=2 const_size=0 OK\n", v.desc);
            jit_free_version(&v);
        }
    }
    /* SSE nibble, U=1: r1 survives as the delta temp; need = 1*1+2 = 3
     * (was 8). Execute the build: newpar = oldpar ^ old ^ new. */
    {
        rj_cpu_info cpu;
        memset(&cpu, 0, sizeof cpu);
        cpu.sse2 = cpu.ssse3 = 1;
        version_t v;
        if (build_cand(&pl, &cpu, ISA_SSE, 1, MUL_NIBBLE, 0, &v, err,
                       sizeof err)) {
            CHECK(plan_need_regs(ISA_SSE, MUL_NIBBLE, &pl, 1, 0, 0, 0) == 3,
                  "dta [%s]: need=%d, want 3",
                  v.desc, plan_need_regs(ISA_SSE, MUL_NIBBLE, &pl, 1, 0, 0, 0));
            CHECK(v.const_size == 0,
                  "dta [%s]: const_size=%zu, want 0", v.desc, v.const_size);
            /* functional check of the kept delta temp */
            size_t len = 100; /* not a multiple of 16: covers the tail */
            size_t alloc = (len + 63) & ~(size_t)63;
            uint8_t *oldb = aligned_alloc(64, alloc);
            uint8_t *newb = aligned_alloc(64, alloc);
            uint8_t *oldp = aligned_alloc(64, alloc);
            uint8_t *newp = aligned_alloc(64, alloc);
            if (oldb && newb && oldp && newp) {
                for (size_t k = 0; k < len; k++) {
                    oldb[k] = (uint8_t)(k * 37 + 11);
                    newb[k] = (uint8_t)(k * 91 + 5);
                    oldp[k] = (uint8_t)(k * 13 + 7);
                }
                const void *in[3] = { oldb, newb, oldp };
                void *out[1] = { newp };
                v.fn(in, out, len);
                int ok = 1;
                for (size_t k = 0; k < len; k++)
                    if (newp[k] != (uint8_t)(oldp[k] ^ oldb[k] ^ newb[k]))
                        ok = 0;
                CHECK(ok, "dta [%s]: SSE delta-temp execution mismatch",
                      v.desc);
                printf("dta [%s]: need=3 const_size=0 exec OK\n", v.desc);
            } else {
                printf("FAIL dta: alloc\n");
                failures++;
            }
            free(oldb); free(newb); free(oldp); free(newp);
            jit_free_version(&v);
        }
    }
    plan_free(&pl);
    topo_int_free(&ti);
    rj_topo_free(topo);
    rj_op_free(&op);

    /* Nontrivial control: RAID-6 parity update (Q row outside {0,1}).
     * n_out=2, U=1: need = 2+7 = 9, pool present. */
    {
        int ch6[1] = { 0 };
        rj_topology *topo6 = rj_topo_raid6(4);
        rj_op op6 = rj_op_parity_update(ch6, 1);
        topo_int_t ti6;
        if (topo_int_build(&ti6, topo6, err, sizeof err)) {
            printf("FAIL dta: topo6: %s\n", err);
            failures++;
            return;
        }
        plan_t pl6;
        if (plan_build(&ti6, &op6, "r6pu", &pl6, err, sizeof err)) {
            printf("FAIL dta: plan6: %s\n", err);
            failures++;
            topo_int_free(&ti6);
            return;
        }
        if (pl6.kind == P_DELTA) {
            rj_cpu_info cpu;
            memset(&cpu, 0, sizeof cpu);
            cpu.avx = cpu.avx2 = 1;
            version_t v;
            if (build_cand(&pl6, &cpu, ISA_AVX2, 1, MUL_NIBBLE, 0, &v, err,
                           sizeof err)) {
                CHECK(plan_need_regs(ISA_AVX2, MUL_NIBBLE, &pl6, 1, 0, 0, 0) == 9,
                      "dta-ctrl [%s]: need=%d, want 9", v.desc,
                      plan_need_regs(ISA_AVX2, MUL_NIBBLE, &pl6, 1, 0, 0, 0));
                CHECK(v.const_size > 0,
                      "dta-ctrl [%s]: const_size=%zu, want >0",
                      v.desc, v.const_size);
                printf("dta-ctrl [%s]: need=9 const_size=%zu OK\n",
                       v.desc, v.const_size);
                jit_free_version(&v);
            }
        } else {
            printf("FAIL dta: expected P_DELTA for r6pu, got kind %d\n",
                   pl6.kind);
            failures++;
        }
        plan_free(&pl6);
        topo_int_free(&ti6);
        rj_topo_free(topo6);
        rj_op_free(&op6);
    }
}

/* Hand-built all-trivial P_MULACC on RVV: the planner never generates
 * this (all-{0,1} classifies P_XOR), but the formula must agree with the
 * backend for hand-built plans. Regression test: plan_need_regs used to
 * return (n+1)*U+1 while the RVV backend allocates (n+1)*U, tripping its
 * vmax assert in debug builds. (The backend side is exercised by the
 * RVV QEMU tests; the native x86 test binary doesn't link the RVV
 * backend, so this pins the formula here.) */
static void test_mulacc_trivial_rvv(void)
{
    uint8_t mat[2] = { 1, 1 };
    plan_t pl;
    memset(&pl, 0, sizeof pl);
    pl.kind = P_MULACC;
    pl.n_in = 2;
    pl.n_out = 1;
    pl.mat = mat;
    snprintf(pl.name, sizeof pl.name, "hand-mulacc-trivial");

    /* (n_out+1)*U + 0 = 2: only the U data regs survive. The RVV backend
     * stages inputs into vd0..vd0+U-1, already counted in (n_out+1)*U. */
    int need = plan_need_regs(ISA_RVV, MUL_NIBBLE, &pl, 1, 0, 0, 0);
    CHECK(need == 2, "mtr: need=%d, want 2", need);
    if (need == 2)
        printf("mtr [rvv trivial mulacc]: need=2 OK\n");
}

int main(void)
{
    test_f1();
    test_f2();
    test_f5();
    test_tl();
    test_ptr_hoist();
    test_delta_trivial();
    test_mulacc_trivial_rvv();
    if (failures == 0)
        printf("test_codegen_shape: all OK\n");
    else
        printf("test_codegen_shape: %d FAILURES\n", failures);
    return failures != 0;
}
