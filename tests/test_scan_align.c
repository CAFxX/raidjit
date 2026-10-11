/* test_scan_align: prove the alignment contract is fully exploited.
 *
 * For every legal (ISA x strategy x unroll) candidate of several plans, the
 * test builds the JIT version (codegen only -- no execution, so AVX-512
 * codegen is covered even on CPUs without it) and walks every instruction
 * with a small length decoder covering exactly emit.c's opcode universe:
 *   - any vector load/store with a memory operand (opcode 6F/7F in any
 *     encoding) must use the aligned form (mandatory 66 prefix / pp=66);
 *     anything else is a latent #GP once callers honor the contract.
 *   - at least one aligned vector-memory move must be found per version
 *     (positive check, so the scan cannot pass vacuously).
 *
 * The decoder is fail-closed: an unrecognized byte pattern or a total
 * length != code_size fails the test instead of silently skipping bytes.
 * It was cross-validated against objdump on every generated version during
 * development.
 */
#include <stdio.h>
#include <string.h>
#include "../src/jit.h"
#include "../src/plan.h"

static int failures = 0;

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
    int has_mrm;
    int mod;      /* valid iff has_mrm */
} insn_t;

/* decode one instruction; covers exactly what src/x86_64/emit.c can produce */
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
        d.pp = (int)(p1 & 3);
        size_t o = ph;
        d.opcode = p[o];
        o += 1;
        size_t ml = mrm_len(p + o, avail - o);
        if (!ml)
            return d;
        d.has_mrm = 1;
        d.mod = p[o] >> 6;
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
                d.len = o + ml + 1;
                return d;
            }
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

static void scan_version(const plan_t *pl, const cand_t *cand,
                         const rj_cpu_info *cpu, const rj_options *opts)
{
    version_t v;
    char err[256];
    memset(&v, 0, sizeof v);
    if (jit_build_version(pl, cand, cpu, opts, &v, err, sizeof err) != 0) {
        printf("FAIL: build %s isa=%d U=%d: %s\n", pl->name, cand->isa,
               cand->U, err);
        failures++;
        return;
    }
    const uint8_t *code = (const uint8_t *)v.mem;
    size_t off = 0;
    int n_aligned = 0;
    while (off < v.code_size) {
        insn_t d = decode(code + off, v.code_size - off);
        if (!d.len) {
            printf("FAIL: undecodable bytes at +0x%zx in %s isa=%d U=%d "
                   "mul=%d nt=%d\n",
                   off, pl->name, cand->isa, cand->U, cand->mul, cand->nt);
            failures++;
            break;
        }
        /* In emit.c's universe, opcode 6F/7F/E7 with a memory operand is
         * always a vector load/store (E7 = MOVNTDQ/VMOVNTDQ, an aligned
         * NT store); it must use the aligned (66) form. */
        if (d.has_mrm && d.mod != 3 &&
            (d.opcode == 0x6F || d.opcode == 0x7F || d.opcode == 0xE7)) {
            if (d.pp != 1) {
                printf("FAIL: unaligned vector move at +0x%zx in %s "
                       "isa=%d U=%d mul=%d nt=%d (pp=%d op=%02x)\n",
                       off, pl->name, cand->isa, cand->U, cand->mul,
                       cand->nt, d.pp, d.opcode);
                failures++;
            } else
                n_aligned++;
        }
        off += d.len;
    }
    if (off != v.code_size) {
        printf("FAIL: decoded %zu of %zu bytes in %s isa=%d U=%d\n", off,
               v.code_size, pl->name, cand->isa, cand->U);
        failures++;
    }
    if (!pl->noop && n_aligned == 0 && !failures) {
        printf("FAIL: no aligned vector move found in %s isa=%d U=%d "
               "(vacuous scan?)\n",
               pl->name, cand->isa, cand->U);
        failures++;
    }
    jit_free_version(&v);
}

static void scan_plan(rj_topology *topo, rj_op *op)
{
    topo_int_t ti;
    char err[256];
    if (topo_int_build(&ti, topo, err, sizeof err) != 0) {
        printf("FAIL: topo build: %s\n", err);
        failures++;
        return;
    }
    plan_t pl;
    if (plan_build(&ti, op, "enc", &pl, err, sizeof err) != 0) {
        printf("FAIL: plan build: %s\n", err);
        failures++;
        topo_int_free(&ti);
        return;
    }
    /* synthetic max CPU: enumerate every legal candidate on every ISA */
    rj_cpu_info cpu;
    memset(&cpu, 0, sizeof cpu);
    cpu.sse2 = cpu.ssse3 = cpu.avx = cpu.avx2 = 1;
    cpu.avx512f = cpu.avx512bw = cpu.avx512vl = cpu.gfni = 1;
    cand_t cands[128];
    int nc = plan_candidates(&pl, &cpu, cands, 128, 1, 0);
    if (nc <= 0) {
        printf("FAIL: no candidates for %s\n", pl.name);
        failures++;
    }
    rj_options opts;
    rj_options_default(&opts);
    opts.benchmark = 0;
    for (int i = 0; i < nc; i++)
        scan_version(&pl, &cands[i], &cpu, &opts);
    plan_free(&pl);
    topo_int_free(&ti);
}

int main(void)
{
    int f0;
    rj_op op;

    /* encode plans */
    {
        rj_topology *t = rj_topo_raid6(8);
        op = rj_op_encode();
        scan_plan(t, &op);
        rj_op_free(&op);
        rj_topo_free(t);
    }
    {
        rj_topology *t = rj_topo_rs(6, 3);
        op = rj_op_encode();
        scan_plan(t, &op);
        rj_op_free(&op);
        rj_topo_free(t);
    }
    {
        rj_topology *t = rj_topo_raid5(8);
        op = rj_op_encode();
        scan_plan(t, &op);
        rj_op_free(&op);
        rj_topo_free(t);
    }
    {
        rj_topology *t = rj_topo_raid1();
        op = rj_op_encode();
        scan_plan(t, &op);
        rj_op_free(&op);
        rj_topo_free(t);
    }
    /* recover plans: single data, double data+parity */
    {
        rj_topology *t = rj_topo_raid6(8);
        f0 = 3;
        op = rj_op_recover(&f0, 1);
        scan_plan(t, &op);
        rj_op_free(&op);
        rj_topo_free(t);
    }
    {
        rj_topology *t = rj_topo_rs(6, 3);
        int fl[2] = { 1, 7 }; /* data + parity */
        op = rj_op_recover(fl, 2);
        scan_plan(t, &op);
        rj_op_free(&op);
        rj_topo_free(t);
    }
    /* parity-update (partial stripe) plans */
    {
        rj_topology *t = rj_topo_raid6(8);
        int ch[2] = { 1, 5 };
        op = rj_op_parity_update(ch, 2);
        scan_plan(t, &op);
        rj_op_free(&op);
        rj_topo_free(t);
    }
    {
        rj_topology *t = rj_topo_rs(4, 3);
        int ch[3] = { 0, 2, 3 };
        op = rj_op_parity_update(ch, 3);
        scan_plan(t, &op);
        rj_op_free(&op);
        rj_topo_free(t);
    }

    if (failures) {
        printf("scan_align: %d FAILURES\n", failures);
        return 1;
    }
    printf("scan_align: all OK\n");
    return 0;
}
