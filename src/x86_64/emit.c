#include "emit.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/* ---------------- buffer / labels / fixups ---------------- */

void em_init(em_t *e, uint8_t *buf, size_t cap)
{
    memset(e, 0, sizeof(*e));
    e->buf = buf;
    e->cap = cap;
}

static void em_u8(em_t *e, unsigned v)
{
    assert(e->len < e->cap);
    e->buf[e->len++] = (uint8_t)v;
}

void em_byte(em_t *e, uint8_t v) { em_u8(e, v); }

static void em_i32(em_t *e, int v)
{
    em_u8(e, (unsigned)v);
    em_u8(e, (unsigned)v >> 8);
    em_u8(e, (unsigned)v >> 16);
    em_u8(e, (unsigned)v >> 24);
}

static void em_i64(em_t *e, uint64_t v)
{
    for (int i = 0; i < 8; i++)
        em_u8(e, (unsigned)(v >> (8 * i)));
}

int em_label(em_t *e)
{
    if (e->nlab == e->clab) {
        e->clab = e->clab ? e->clab * 2 : 32;
        e->lpos = realloc(e->lpos, (size_t)e->clab * sizeof(size_t));
        assert(e->lpos);
        for (int i = e->nlab; i < e->clab; i++)
            e->lpos[i] = (size_t)-1;
    }
    return e->nlab++;
}

void em_bind(em_t *e, int lab)
{
    assert(lab >= 0 && lab < e->nlab);
    e->lpos[lab] = e->len;
}

static void em_fixup(em_t *e, size_t at, int lab, int add)
{
    if (e->nfix == e->cfix) {
        e->cfix = e->cfix ? e->cfix * 2 : 64;
        e->fix = realloc(e->fix, (size_t)e->cfix * sizeof(*e->fix));
        assert(e->fix);
    }
    e->fix[e->nfix].at = at;
    e->fix[e->nfix].lab = lab;
    e->fix[e->nfix].add = add;
    e->nfix++;
}

void em_resolve(em_t *e)
{
    for (int i = 0; i < e->nfix; i++) {
        size_t at = e->fix[i].at;
        int lab = e->fix[i].lab;
        assert(lab >= 0 && lab < e->nlab && e->lpos[lab] != (size_t)-1);
        int64_t target = (int64_t)(e->lpos[lab] + e->fix[i].add);
        int64_t disp = target - (int64_t)(at + 4);
        assert(disp >= INT32_MIN && disp <= INT32_MAX);
        e->buf[at + 0] = (uint8_t)disp;
        e->buf[at + 1] = (uint8_t)(disp >> 8);
        e->buf[at + 2] = (uint8_t)(disp >> 16);
        e->buf[at + 3] = (uint8_t)(disp >> 24);
    }
}

void em_free(em_t *e)
{
    free(e->lpos);
    free(e->fix);
    e->lpos = NULL;
    e->fix = NULL;
    e->nlab = e->clab = e->nfix = e->cfix = 0;
}

/* Multi-byte NOPs.
 *
 * 1-9: Intel-recommended patterns (Intel SDM / Optimization Manual Table 3-3).
 * 10-15: Agner Fog "Optimizing subroutines in assembly", sec. 10.7, and the
 * AMD Software Optimization Guide (Fam 16h/17h) 1-15 byte table: the 8-byte
 * base 0F 1F 84 00 00 00 00 00 ("nopl 0(%rax,%rax,1)") with redundant 66h
 * operand-size prefixes. The dummy memory operand is never accessed, 0F 1F
 * carries no immediate (no length-changing-prefix stall), and modern
 * Intel/AMD decoders targeted here do not penalize the extra 66h prefixes.
 *
 * Compatibility note: AMD Family 15h and earlier penalize >3 redundant 66h
 * prefixes (use 4-byte + 8..11-byte NOP pairs there instead), and Agner caps
 * single NOPs at ~10 bytes for older Intel parts. The JIT targets modern
 * x86-64 only, so the single-instruction 10-15 byte forms are used. Sandy
 * Bridge decodes 0F 1F on decoder 0 only -- fine for alignment padding.
 */
static const uint8_t nops[][15] = {
    {0x90},
    {0x66, 0x90},
    {0x0F, 0x1F, 0x00},
    {0x0F, 0x1F, 0x40, 0x00},
    {0x0F, 0x1F, 0x44, 0x00, 0x00},
    {0x66, 0x0F, 0x1F, 0x44, 0x00, 0x00},
    {0x0F, 0x1F, 0x80, 0x00, 0x00, 0x00, 0x00},
    {0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00},
    {0x66, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00},
    {0x66, 0x66, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00},
    {0x66, 0x66, 0x66, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00},
    {0x66, 0x66, 0x66, 0x66, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00},
    {0x66, 0x66, 0x66, 0x66, 0x66, 0x0F, 0x1F, 0x84,
     0x00, 0x00, 0x00, 0x00, 0x00},
    {0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x0F, 0x1F, 0x84,
     0x00, 0x00, 0x00, 0x00, 0x00},
    {0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x0F, 0x1F, 0x84,
     0x00, 0x00, 0x00, 0x00, 0x00},
};

void em_align(em_t *e, unsigned a)
{
    size_t need = ((e->len + a - 1) & ~(size_t)(a - 1)) - e->len;
    while (need) {
        unsigned n = need > 15 ? 15 : (unsigned)need;
        for (unsigned i = 0; i < n; i++)
            em_u8(e, nops[n - 1][i]);
        need -= n;
    }
}

/* ---------------- prefixes ---------------- */

static void emit_rex(em_t *e, int W, int R, int X, int B)
{
    int r = 0x40 | (W << 3) | (R << 2) | (X << 1) | B;
    if (r != 0x40)
        em_u8(e, (unsigned)r);
}

/* VEX prefix. map: 1=0F, 2=0F38, 3=0F3A. pp: 0,1=66,2=F3,3=F2. L: 0=128,1=256. */
static void emit_vex(em_t *e, int map, int W, int R, int X, int B,
                     int vvvv, int L, int pp)
{
    assert(vvvv >= 0 && vvvv < 16);
    if (map == 1 && W == 0 && X == 0 && B == 0) {
        em_u8(e, 0xC5);
        em_u8(e, (unsigned)(((R ^ 1) << 7) | (((vvvv ^ 0xF) & 0xF) << 3) |
                              (L << 2) | pp));
    } else {
        em_u8(e, 0xC4);
        em_u8(e, (unsigned)(((R ^ 1) << 7) | ((X ^ 1) << 6) | ((B ^ 1) << 5) |
                              map));
        em_u8(e, (unsigned)((W << 7) | (((vvvv ^ 0xF) & 0xF) << 3) | (L << 2) |
                              pp));
    }
}

/* EVEX prefix. vvvv: 5-bit src1 reg. LL: 0=128, 1=256, 2=512.
 * aaa: opmask register (0 = unmasked). z: 1 = zeroing-masking. */
static void emit_evex(em_t *e, int map, int W, int R, int X, int B, int R2,
                      int vvvv, int LL, int pp, int aaa, int z)
{
    assert(vvvv >= 0 && vvvv < 32);
    assert(aaa >= 0 && aaa < 8);
    em_u8(e, 0x62);
    em_u8(e, (unsigned)(((R ^ 1) << 7) | ((X ^ 1) << 6) | ((B ^ 1) << 5) |
                          ((R2 ^ 1) << 4) | map));
    em_u8(e, (unsigned)((W << 7) | (((vvvv ^ 0xF) & 0xF) << 3) | (1 << 2) |
                          pp));
    em_u8(e, (unsigned)((z << 7) | (LL == 2 ? 0x40 : LL == 1 ? 0x20 : 0) |
                          ((((vvvv >> 4) ^ 1) & 1) << 3) | (aaa & 7)));
}

/* Emit ModRM [+ SIB + disp] for reg field rf and mem operand.
 * b: base reg, -1 = none (disp32 absolute-ish via SIB), -2 = RIP (fixup).
 * x: index reg or -1. Caller handles RIP via the _rip helpers instead. */
static void emit_mrm(em_t *e, int rf, int b, int x, int sc, int disp,
                     int dscale)
{
    assert(b != -2);
    /* SIB index field 100b means "no index" only when REX.X=0. With
     * REX.X=1 it encodes r12 (low 3 bits 100b, high bit from REX.X), so
     * r12 is a legal index and every caller sets X via rxb() above.
     * r4 (RSP) can never be an index: it has no high bit, so it is
     * indistinguishable from "no index". ids >= 16 are not registers. */
    assert(x < 0 || (x < 16 && x != 4));
    int need_sib = (x >= 0) || ((b & 7) == 4);
    int mod;
    int disp8 = disp;
    if (b < 0) {
        mod = 0;
        need_sib = 1;
    } else if (disp == 0 && (b & 7) != 5) {
        mod = 0;
    } else if (dscale > 1) {
        /* EVEX: disp8 is always scaled by the vector length. Use it only
         * for exact multiples; anything else needs disp32 (a raw disp8
         * would decode as disp8*dscale). */
        if (disp % dscale == 0) {
            int sd = disp / dscale;
            if (sd >= -128 && sd <= 127) {
                mod = 1;
                disp8 = sd;
            } else {
                mod = 2;
            }
        } else {
            mod = 2;
        }
    } else if (disp >= -128 && disp <= 127) {
        mod = 1;
    } else {
        mod = 2;
    }
    em_u8(e, (unsigned)((mod << 6) | ((rf & 7) << 3) |
                          (need_sib ? 4 : (b & 7))));
    if (need_sib) {
        int xi = (x < 0) ? 4 : (x & 7);
        int bb = (b < 0) ? 5 : (b & 7);
        em_u8(e, (unsigned)((sc << 6) | (xi << 3) | bb));
    }
    if (mod == 0 && b < 0)
        em_i32(e, disp);
    else if (mod == 1)
        em_u8(e, (unsigned)disp8);
    else if (mod == 2)
        em_i32(e, disp);
}

static void rxb(int rf, int b, int x, int *R, int *X, int *B)
{
    *R = (rf >> 3) & 1;
    *X = (x >= 0) ? ((x >> 3) & 1) : 0;
    *B = (b >= 0) ? ((b >> 3) & 1) : 0;
}

/* ---------------- misc ---------------- */

void em_endbr64(em_t *e)
{
    em_u8(e, 0xF3);
    em_u8(e, 0x0F);
    em_u8(e, 0x1E);
    em_u8(e, 0xFA);
}

void em_setssbsy(em_t *e)
{
    em_u8(e, 0xF3);
    em_u8(e, 0x0F);
    em_u8(e, 0x01);
    em_u8(e, 0xEA);
}

void em_clrssbsy(em_t *e)
{
    em_u8(e, 0xF3);
    em_u8(e, 0x0F);
    em_u8(e, 0x01);
    em_u8(e, 0xE9);
}

void em_ud2(em_t *e)
{
    em_u8(e, 0x0F);
    em_u8(e, 0x0B);
}

void em_int3(em_t *e) { em_u8(e, 0xCC); }

/* Pad to alignment a with trap instructions: UD2 pairs, plus a single
 * INT3 when the pad length is odd (x86 has no 1-byte UD2; INT3 is the
 * standard 1-byte trap and raises SIGTRAP). Used only for dead space
 * after ret: if execution ever falls through past ret it must fault
 * immediately instead of sliding through NOPs into the const pool.
 * Executable alignment (loops) still uses em_align's real NOPs. */
void em_trap_align(em_t *e, unsigned a)
{
    assert(a && (a & (a - 1)) == 0);
    size_t need = ((e->len + a - 1) & ~(size_t)(a - 1)) - e->len;
    while (need >= 2) {
        em_ud2(e);
        need -= 2;
    }
    if (need)
        em_int3(e);
}

void em_ret(em_t *e) { em_u8(e, 0xC3); }

void em_push_r(em_t *e, int r)
{
    int R, X, B;
    rxb(0, r, -1, &R, &X, &B);
    emit_rex(e, 0, 0, 0, B);
    em_u8(e, 0x50 + (r & 7));
}

void em_pop_r(em_t *e, int r)
{
    int R, X, B;
    rxb(0, r, -1, &R, &X, &B);
    emit_rex(e, 0, 0, 0, B);
    em_u8(e, 0x58 + (r & 7));
}

void em_jcc(em_t *e, int cc, int lab)
{
    em_u8(e, 0x0F);
    em_u8(e, 0x80 + (cc & 0xF));
    em_fixup(e, e->len, lab, 0);
    em_i32(e, 0);
}

void em_jmp(em_t *e, int lab)
{
    em_u8(e, 0xE9);
    em_fixup(e, e->len, lab, 0);
    em_i32(e, 0);
}

/* ---------------- integer ---------------- */

void em_mov_rr(em_t *e, int d, int s)
{
    int R, X, B;
    rxb(d, s, -1, &R, &X, &B);
    emit_rex(e, 1, R, X, B);
    em_u8(e, 0x8B);
    em_u8(e, (unsigned)(0xC0 | ((d & 7) << 3) | (s & 7)));
}

void em_mov_ri(em_t *e, int d, uint64_t v)
{
    if ((int64_t)v == (int32_t)v) {
        int R, X, B;
        rxb(0, d, -1, &R, &X, &B);
        emit_rex(e, 1, 0, 0, B);
        em_u8(e, 0xC7);
        em_u8(e, (unsigned)(0xC0 | (d & 7)));
        em_i32(e, (int)v);
    } else {
        int R, X, B;
        rxb(0, d, -1, &R, &X, &B);
        emit_rex(e, 1, 0, 0, B);
        em_u8(e, 0xB8 + (d & 7));
        em_i64(e, v);
    }
}

void em_mov_rm(em_t *e, int d, int b, int x, int sc, int disp)
{
    int R, X, B;
    rxb(d, b, x, &R, &X, &B);
    emit_rex(e, 1, R, X, B);
    em_u8(e, 0x8B);
    emit_mrm(e, d, b, x, sc, disp, 1);
}

void em_mov_mr(em_t *e, int b, int x, int sc, int disp, int s)
{
    int R, X, B;
    rxb(s, b, x, &R, &X, &B);
    emit_rex(e, 1, R, X, B);
    em_u8(e, 0x89);
    emit_mrm(e, s, b, x, sc, disp, 1);
}

void em_movzx_rm8(em_t *e, int d, int b, int x, int sc, int disp)
{
    int R, X, B;
    rxb(d, b, x, &R, &X, &B);
    emit_rex(e, 0, R, X, B);
    em_u8(e, 0x0F);
    em_u8(e, 0xB6);
    emit_mrm(e, d, b, x, sc, disp, 1);
}

/* Byte-register aliasing guard: without a REX prefix, ModRM fields 100b-111b
 * decode as AH/CH/DH/BH, not SPL/BPL/SIL/DIL. A byte operand in r4-r7 with
 * no REX bit set would therefore silently assemble to the wrong register.
 * All byte helpers below run with W=0, so REX is emitted iff R/X/B is set. */
static void check_byte_rex(int d, int s, int R, int X, int B)
{
    int rex = R || X || B;
    assert(rex || (d & 8) || (d & 12) != 4);
    assert(rex || (s & 8) || (s & 12) != 4);
}

void em_mov_r_m8(em_t *e, int d, int b, int x, int sc, int disp)
{
    int R, X, B;
    rxb(d, b, x, &R, &X, &B);
    check_byte_rex(d, 0, R, X, B);
    emit_rex(e, 0, R, X, B);
    em_u8(e, 0x8A);
    emit_mrm(e, d, b, x, sc, disp, 1);
}

void em_mov_m8_r(em_t *e, int b, int x, int sc, int disp, int s)
{
    int R, X, B;
    rxb(s, b, x, &R, &X, &B);
    check_byte_rex(s, 0, R, X, B);
    emit_rex(e, 0, R, X, B);
    em_u8(e, 0x88);
    emit_mrm(e, s, b, x, sc, disp, 1);
}

void em_lea(em_t *e, int d, int b, int x, int sc, int disp)
{
    int R, X, B;
    rxb(d, b, x, &R, &X, &B);
    emit_rex(e, 1, R, X, B);
    em_u8(e, 0x8D);
    emit_mrm(e, d, b, x, sc, disp, 1);
}

void em_lea_rip(em_t *e, int d, int lab, int add)
{
    int R, X, B;
    rxb(d, -1, -1, &R, &X, &B);
    emit_rex(e, 1, R, X, B);
    em_u8(e, 0x8D);
    em_u8(e, (unsigned)(0x05 | ((d & 7) << 3))); /* mod=00 rm=101 */
    em_fixup(e, e->len, lab, add);
    em_i32(e, 0);
}

static void alu_ri(em_t *e, int op, int d, int v, int is64)
{
    int R, X, B;
    rxb(op, d, -1, &R, &X, &B);
    emit_rex(e, is64, 0, 0, B);
    if (v >= -128 && v <= 127) {
        em_u8(e, 0x83);
        em_u8(e, (unsigned)(0xC0 | (op << 3) | (d & 7)));
        em_u8(e, (unsigned)v);
    } else {
        em_u8(e, 0x81);
        em_u8(e, (unsigned)(0xC0 | (op << 3) | (d & 7)));
        em_i32(e, v);
    }
}

void em_add_ri(em_t *e, int d, int v) { alu_ri(e, 0, d, v, 1); }
void em_sub_ri(em_t *e, int d, int v) { alu_ri(e, 5, d, v, 1); }
void em_cmp_ri(em_t *e, int d, int v) { alu_ri(e, 7, d, v, 1); }
void em_add_r32_ri(em_t *e, int d, int v) { alu_ri(e, 0, d, v, 0); }
void em_cmp_r32_ri(em_t *e, int d, int v) { alu_ri(e, 7, d, v, 0); }

void em_add_rr(em_t *e, int d, int s)
{
    int R, X, B;
    rxb(d, s, -1, &R, &X, &B);
    emit_rex(e, 1, R, X, B);
    em_u8(e, 0x03); /* ADD r64, r/m64: d += s */
    em_u8(e, (unsigned)(0xC0 | ((d & 7) << 3) | (s & 7)));
}

void em_sub_rr(em_t *e, int d, int s)
{
    int R, X, B;
    rxb(d, s, -1, &R, &X, &B);
    emit_rex(e, 1, R, X, B);
    em_u8(e, 0x2B); /* SUB r64, r/m64: d -= s */
    em_u8(e, (unsigned)(0xC0 | ((d & 7) << 3) | (s & 7)));
}

void em_xor_rr(em_t *e, int d, int s)
{
    int R, X, B;
    rxb(d, s, -1, &R, &X, &B);
    emit_rex(e, 1, R, X, B);
    em_u8(e, 0x33); /* XOR r64, r/m64: d ^= s */
    em_u8(e, (unsigned)(0xC0 | ((d & 7) << 3) | (s & 7)));
}

void em_xor_r32(em_t *e, int d, int s)
{
    int R, X, B;
    rxb(d, s, -1, &R, &X, &B);
    emit_rex(e, 0, R, X, B);
    em_u8(e, 0x33); /* XOR r32, r/m32: d ^= s */
    em_u8(e, (unsigned)(0xC0 | ((d & 7) << 3) | (s & 7)));
}

void em_xor_r8(em_t *e, int d, int s)
{
    int R, X, B;
    rxb(d, s, -1, &R, &X, &B);
    check_byte_rex(d, s, R, X, B);
    emit_rex(e, 0, R, X, B);
    em_u8(e, 0x32); /* XOR r8, r/m8: d ^= s */
    em_u8(e, (unsigned)(0xC0 | ((d & 7) << 3) | (s & 7)));
}

void em_shr_ri(em_t *e, int d, unsigned n)
{
    int R, X, B;
    rxb(5, d, -1, &R, &X, &B);
    emit_rex(e, 1, 0, 0, B);
    em_u8(e, 0xC1);
    em_u8(e, (unsigned)(0xE8 | (d & 7))); /* /5 */
    em_u8(e, n & 0xFF);
}

void em_test_rr(em_t *e, int d, int s)
{
    int R, X, B;
    rxb(d, s, -1, &R, &X, &B);
    emit_rex(e, 1, R, X, B);
    em_u8(e, 0x85);
    em_u8(e, (unsigned)(0xC0 | ((d & 7) << 3) | (s & 7)));
}

void em_dec_r(em_t *e, int d)
{
    int R, X, B;
    rxb(1, d, -1, &R, &X, &B);
    emit_rex(e, 1, 0, 0, B);
    em_u8(e, 0xFF);
    em_u8(e, (unsigned)(0xC8 | (d & 7))); /* /1 */
}

void em_inc_r(em_t *e, int d)
{
    int R, X, B;
    rxb(0, d, -1, &R, &X, &B);
    emit_rex(e, 1, 0, 0, B);
    em_u8(e, 0xFF);
    em_u8(e, (unsigned)(0xC0 | (d & 7))); /* /0 */
}

/* ---------------- vector ---------------- */

/* 3-operand vector op with reg/reg or reg/mem rm operand.
 * rm_is_mem: emit_mrm with (b,x,sc,disp), else modrm reg=rm_reg.
 * aaa/z: EVEX opmask and zeroing (VL64 only; must be 0 for VEX paths). */
static void vec_op_mrm(em_t *e, int map, int W, int vd, int s1, int opcode,
                       int vl, int pp, int rm_is_mem, int b, int x, int sc,
                       int disp, int rm_reg, int aaa, int z)
{
    int R, X, B;
    if (rm_is_mem)
        rxb(vd, b, x, &R, &X, &B);
    else {
        rxb(vd, rm_reg, -1, &R, &X, &B);
        /* EVEX: for register rm operands, bit4 comes from EVEX.X (P0[6]),
         * not from the SIB index. rxb leaves X=0 when x=-1. */
        if (vl == VL64)
            X = (rm_reg >> 4) & 1;
    }
    if (vl == VL64) {
        int R2 = (vd >> 4) & 1;
        emit_evex(e, map, W, R, X, B, R2, s1, 2, pp, aaa, z);
    } else {
        assert(aaa == 0 && z == 0); /* masking is an AVX-512-only feature */
        emit_vex(e, map, W, R, X, B, s1, vl == VL32, pp);
    }
    em_u8(e, (unsigned)opcode);
    if (rm_is_mem)
        emit_mrm(e, vd, b, x, sc, disp, vl == VL64 ? 64 : 1);
    else
        em_u8(e, (unsigned)(0xC0 | ((vd & 7) << 3) | (rm_reg & 7)));
}

void em_vmovdqa_ld(em_t *e, int vd, int b, int x, int sc, int disp, int vl)
{
    /* Aligned loads: caller contract guarantees 64-byte aligned buffers and a
     * 64-byte aligned constant pool, so the aligned forms are always safe.
     * VEX: 66 0F 6F ; EVEX: VMOVDQA32 = 66 0F 6F (AVX512F only).
     * vvvv reserved (=0). */
    if (vl == VL64)
        vec_op_mrm(e, 1, 0, vd, 0, 0x6F, vl, 1, 1, b, x, sc, disp, 0, 0, 0);
    else
        vec_op_mrm(e, 1, 0, vd, 0, 0x6F, vl, 1, 1, b, x, sc, disp, 0, 0, 0);
}

void em_vmovdqa_st(em_t *e, int b, int x, int sc, int disp, int vs, int vl)
{
    /* Aligned stores: see em_vmovdqa_ld. EVEX: VMOVDQA32 = 66 0F 7F. */
    if (vl == VL64)
        vec_op_mrm(e, 1, 0, vs, 0, 0x7F, vl, 1, 1, b, x, sc, disp, 0, 0, 0);
    else
        vec_op_mrm(e, 1, 0, vs, 0, 0x7F, vl, 1, 1, b, x, sc, disp, 0, 0, 0);
}

void em_vmovdqa_rr(em_t *e, int vd, int vs, int vl)
{
    /* Register move: VEX 66 0F 6F /r ; EVEX VMOVDQA32 66 0F 6F /r.
     * vvvv is reserved (0). Used to fold "xor with known zero" into a
     * single move at JIT time. */
    vec_op_mrm(e, 1, 0, vd, 0, 0x6F, vl, 1, 0, 0, -1, 0, 0, vs, 0, 0);
}

void em_vpxor(em_t *e, int vd, int s1, int s2, int vl)
{
    /* VEX: 66 0F EF ; EVEX: VPXORD (W=0) */
    vec_op_mrm(e, 1, 0, vd, s1, 0xEF, vl, 1, 0, 0, -1, 0, 0, s2, 0, 0);
}

void em_vpxor_rr(em_t *e, int vd, int vs, int vl)
{
    em_vpxor(e, vd, vd, vs, vl);
}

void em_vpxor_mem(em_t *e, int vd, int vs1, int b, int x, int sc, int disp,
                  int vl)
{
    vec_op_mrm(e, 1, 0, vd, vs1, 0xEF, vl, 1, 1, b, x, sc, disp, 0, 0, 0);
}

void em_vpshufb(em_t *e, int vd, int s1, int s2, int vl)
{
    /* map 0F38, opcode 00 */
    vec_op_mrm(e, 2, 0, vd, s1, 0x00, vl, 1, 0, 0, -1, 0, 0, s2, 0, 0);
}

void em_vpshufb_masked(em_t *e, int vd, int s1, int s2, int vl, int k, int z)
{
    /* VPSHUFB with EVEX masking: vd{k}{z} = shufb(s1, s2). k in 1..7
     * (k0 = unmasked). z=1 zeroes masked-off lanes. AVX-512 only. */
    assert(vl == VL64 && k >= 1 && k < 8);
    vec_op_mrm(e, 2, 0, vd, s1, 0x00, vl, 1, 0, 0, -1, 0, 0, s2, k, z);
}

void em_vmovdqu8_ld_masked(em_t *e, int vd, int b, int x, int sc, int disp,
                          int vl, int k, int z)
{
    /* Masked load: vd{k}{z} = [mem], byte-granular mask. k in 1..7, z=1
     * zeroes masked-off lanes. AVX-512 only. Masked-off lanes never fault
     * (verified empirically on AMD EPYC 9D25, track Bx86). W=0 selects
     * the VMOVDQU8 form (byte mask granularity; the VMOVDQA64 form masks
     * at 64-bit granularity which is wrong for byte tails). The address
     * is 64-byte aligned in the tail chunk loop, so the unaligned form
     * has no penalty. */
    assert(vl == VL64 && k >= 1 && k < 8);
    vec_op_mrm(e, 1, 0, vd, 0, 0x6F, vl, 3, 1, b, x, sc, disp, 0, k, z);
}

void em_vmovdqu8_st_masked(em_t *e, int b, int x, int sc, int disp, int vs,
                          int vl, int k)
{
    /* Masked store: [mem]{k} = vs (merging; masked-off lanes keep their
     * old value, so no zeroing past the tail), byte-granular mask.
     * k in 1..7. AVX-512 only. W=0 selects the VMOVDQU8 form (see above).
     */
    assert(vl == VL64 && k >= 1 && k < 8);
    vec_op_mrm(e, 1, 0, vs, 0, 0x7F, vl, 3, 1, b, x, sc, disp, 0, k, 0);
}

void em_kmovq_km(em_t *e, int k, int b, int x, int sc, int disp)
{
    /* KMOVQ k, m64: VEX.0F.W1 90 /r. Loads a 64-bit tail mask from the
     * constant pool into opmask register k (1..7). vvvv reserved.
     * VEX has no disp8 scaling (unlike EVEX), so dscale=1. */
    assert(k >= 1 && k < 8);
    int R, X, B;
    rxb(k, b, x, &R, &X, &B);
    emit_vex(e, 1, 1, R, X, B, 0, 0, 0);
    em_u8(e, 0x90);
    emit_mrm(e, k, b, x, sc, disp, 1);
}

void em_vpor_mem(em_t *e, int vd, int s1, int b, int x, int sc, int disp,
                 int vl)
{
    /* VPORD with memory source: vd = s1 | [mem]. EVEX.66.0F EB /r (dword
     * granularity -- the encoding is VPORD, not VPORB; GAS rejects
     * `vporb` with zmm operands). Used by the vpermi2b nibble twin
     * (track B2) to fold the +16 index offset into a single fused uop
     * with no extra register. */
    vec_op_mrm(e, 1, 0, vd, s1, 0xEB, vl, 1, 1, b, x, sc, disp, 0, 0, 0);
}

void em_vpermi2b(em_t *e, int vd_idx, int vtbl1, int vtbl2, int vl)
{
    /* VPERMI2B: full permute of bytes from two tables, overwriting the
     * index. EVEX.66.0F38.W0 75 /r (AVX512_VBMI).
     *   vd_idx = (idxbit ? vtbl2 : vtbl1)[vd_idx & 63]   (per byte)
     * On entry vd_idx holds the byte indices (bit 6 selects the table,
     * bits [5:0] the byte); it is overwritten with the result. vtbl1 is
     * EVEX.vvvv, vtbl2 is ModRM.r/m; both tables stay intact, so the same
     * register can serve as both tables. Encoding verified against GAS
     * 2.42 (`vpermi2b %zmm1,%zmm2,%zmm3` = 62 f2 6d 48 75 d9) and by
     * execution against a scalar reference (track B2). */
    assert(vl == VL64);
    vec_op_mrm(e, 2, 0, vd_idx, vtbl1, 0x75, vl, 1, 0, 0, -1, 0, 0, vtbl2,
               0, 0);
}

void em_vpmultishiftqb(em_t *e, int vd, int vctrl, int vdata, int vl)
{
    /* VPMULTISHIFTQB: per-qword variable byte-field extract (track I2).
     * EVEX.66.0F38.W1 83 /r (AVX512_VBMI). Note W=1 (unlike the W0
     * neighbors in this file). Semantics per 64-bit lane:
     *   vd.qword[i].byte[j] = (vdata.qword[i] >> (vctrl.qword[i].byte[j]
     *                           & 0x3F)) & 0xFF
     * vctrl is EVEX.vvvv (shift counts; register-only -- no memory
     * form), vdata is ModRM r/m. Encoding verified against GAS 2.42
     * (`vpmultishiftqb %zmm2,%zmm1,%zmm0` = 62 f2 f5 48 83 c2, i.e.
     * vd=0/vctrl=1/vdata=2; `%zmm17,%zmm18,%zmm31` = 62 22 ed 40 83 f9)
     * and by execution against a scalar model on a VBMI host. */
    assert(vl == VL64);
    vec_op_mrm(e, 2, 1, vd, vctrl, 0x83, vl, 1, 0, 0, -1, 0, 0, vdata,
               0, 0);
}

void em_vpand(em_t *e, int vd, int s1, int s2, int vl)
{
    /* VEX: 66 0F DB ; EVEX: VPANDD (W=0) */
    vec_op_mrm(e, 1, 0, vd, s1, 0xDB, vl, 1, 0, 0, -1, 0, 0, s2, 0, 0);
}

void em_vpsrlw(em_t *e, int vd, int s1, unsigned imm, int vl)
{
    /* 66 0F 71 /2 ib: dst in vvvv, src in modrm.rm, /2 = word shift.
     * Group opcode: ModRM.reg holds the /2 extension, so the src
     * register's bit4 comes from EVEX.X (not SIB) and EVEX.R' must be 0. */
    if (vl == VL64)
        emit_evex(e, 1, 0, 0, (s1 >> 4) & 1, (s1 >> 3) & 1, 0, vd, 2, 1, 0, 0);
    else
        emit_vex(e, 1, 0, 0, 0, (s1 >> 3) & 1, vd, vl == VL32, 1);
    em_u8(e, 0x71);
    em_u8(e, (unsigned)(0xD0 | (s1 & 7))); /* mod=11, /2, rm=src */
    em_u8(e, imm & 0xFF);
}

void em_vpsllw(em_t *e, int vd, int s1, unsigned imm, int vl)
{
    /* 66 0F 71 /6 ib: same shape as VPSRLW, /6 = word shift left.
     * Used by the bit-program (xorbit) emitter to place a
     * bit-0-normalized bit value at its output bit position. */
    if (vl == VL64)
        emit_evex(e, 1, 0, 0, (s1 >> 4) & 1, (s1 >> 3) & 1, 0, vd, 2, 1, 0, 0);
    else
        emit_vex(e, 1, 0, 0, 0, (s1 >> 3) & 1, vd, vl == VL32, 1);
    em_u8(e, 0x71);
    em_u8(e, (unsigned)(0xF0 | (s1 & 7))); /* mod=11, /6, rm=src */
    em_u8(e, imm & 0xFF);
}

void em_vgf2p8affineqb(em_t *e, int vd, int s1, int s2, unsigned imm, int vl)
{
    /* EVEX.66.0F3A.W1 CE /r ib */
    assert(vl == VL64);
    int R, X, B;
    rxb(vd, s2, -1, &R, &X, &B);
    /* EVEX.X carries rm bit4 for register operands */
    X = (s2 >> 4) & 1;
    emit_evex(e, 3, 1, R, X, B, (vd >> 4) & 1, s1, 2, 1, 0, 0);
    em_u8(e, 0xCE);
    em_u8(e, (unsigned)(0xC0 | ((vd & 7) << 3) | (s2 & 7)));
    em_u8(e, imm & 0xFF);
}

void em_vpternlog(em_t *e, int vd, int s1, int s2, unsigned imm, int vl)
{
    /* EVEX.66.0F3A.W0 25 /r ib : vd = lut[vd, s1, s2] (dword elements,
     * bitwise op). imm=0x96 gives vd ^= s1 ^ s2. AVX-512 only. */
    assert(vl == VL64);
    int R, X, B;
    rxb(vd, s2, -1, &R, &X, &B);
    /* EVEX.X carries rm bit4 for register operands */
    X = (s2 >> 4) & 1;
    emit_evex(e, 3, 0, R, X, B, (vd >> 4) & 1, s1, 2, 1, 0, 0);
    em_u8(e, 0x25);
    em_u8(e, (unsigned)(0xC0 | ((vd & 7) << 3) | (s2 & 7)));
    em_u8(e, imm & 0xFF);
}

void em_vpternlog_mem(em_t *e, int vd, int s1, int b, int x, int sc, int disp,
                      unsigned imm, int vl)
{
    /* EVEX.66.0F3A.W0 25 /r ib : vd = lut[vd, s1, mem512] (dword elements,
     * bitwise op). imm=0x96 gives vd ^= s1 ^ mem. AVX-512 only. Unlike
     * the register form, EVEX.X comes from the SIB index (rxb), not from
     * an rm bit4 override. */
    assert(vl == VL64);
    int R, X, B;
    rxb(vd, b, x, &R, &X, &B);
    emit_evex(e, 3, 0, R, X, B, (vd >> 4) & 1, s1, 2, 1, 0, 0);
    em_u8(e, 0x25);
    emit_mrm(e, vd, b, x, sc, disp, 64);
    em_u8(e, imm & 0xFF);
}

void em_vzero(em_t *e, int vd, int vl)
{
    em_vpxor(e, vd, vd, vd, vl);
}

/* VZEROUPPER: VEX.128.0F.WIG 77 — zeroes bits [VLMAX-1:128] of all vector
 * registers. Emitted in the epilogue (gated on !is_sse) so a caller that
 * executes legacy SSE after rj_fn does not pay the AVX-SSE transition
 * penalty. 3 bytes, renamer-handled. */
void em_vzeroupper(em_t *e)
{
    /* vvvv=0 (not 15): emit_vex inverts vvvv (stores ~vvvv), so vvvv=0
     * encodes as 1111 in the VEX byte, matching the canonical C5 F8 77
     * (GCC/Clang output). vvvv is don't-care for VZEROUPPER, but the
     * canonical encoding avoids any decoder quirks. */
    emit_vex(e, 1, 0, 0, 0, 0, 0, 0, 0);
    em_u8(e, 0x77);
}

/* ---------------- legacy SSE (no VEX) ---------------- */

static void sse_op_mrm(em_t *e, unsigned prefix, unsigned op2, int vd,
                       int is_mem, int b, int x, int sc, int disp, int vs)
{
    int R, X, B;
    if (is_mem)
        rxb(vd, b, x, &R, &X, &B);
    else
        rxb(vd, vs, -1, &R, &X, &B);
    if (prefix)
        em_u8(e, prefix); /* mandatory prefix precedes REX */
    emit_rex(e, 0, R, X, B);
    em_u8(e, 0x0F);
    em_u8(e, op2);
    if (is_mem)
        emit_mrm(e, vd, b, x, sc, disp, 1);
    else
        em_u8(e, (unsigned)(0xC0 | ((vd & 7) << 3) | (vs & 7)));
}

void em_movdqa_ld_sse(em_t *e, int vd, int b, int x, int sc, int disp)
{
    sse_op_mrm(e, 0x66, 0x6F, vd, 1, b, x, sc, disp, 0);
}

void em_movdqa_st_sse(em_t *e, int b, int x, int sc, int disp, int vs)
{
    sse_op_mrm(e, 0x66, 0x7F, vs, 1, b, x, sc, disp, 0);
}

/* Non-temporal (streaming) stores. Same alignment requirements as the
 * aligned forms (16/32/64 by vector width); weakly ordered -- the caller
 * must emit SFENCE before the data is consumed. */
void em_movntdq_st_sse(em_t *e, int b, int x, int sc, int disp, int vs)
{
    /* SSE: MOVNTDQ = 66 0F E7 /r (store xmm -> m128) */
    sse_op_mrm(e, 0x66, 0xE7, vs, 1, b, x, sc, disp, 0);
}

void em_vmovntdq_st(em_t *e, int b, int x, int sc, int disp, int vs, int vl)
{
    /* VEX: VMOVNTDQ = VEX.128/256.66.0F E7 /r ;
     * EVEX: VMOVNTDQ = EVEX.512.66.0F E7 /r. WIG. */
    vec_op_mrm(e, 1, 0, vs, 0, 0xE7, vl, 1, 1, b, x, sc, disp, 0, 0, 0);
}

void em_sfence(em_t *e)
{
    /* SFENCE = 0F AE /8 (F8) */
    em_u8(e, 0x0F);
    em_u8(e, 0xAE);
    em_u8(e, 0xF8);
}

void em_prefetch(em_t *e, int hint, int b, int x, int sc, int disp)
{
    /* PREFETCHh = 0F 18 /r: /0=NTA, /1=T0, /2=T1, /3=T2. Legacy encoding
     * with a REX prefix (W=0) whenever the base/index needs it: without
     * REX.B/REX.X, r8-r15 would mis-encode as r0-r7 (the JIT always
     * passes r8+ bases and r11 as the index). dscale=1: byte-granular. */
    int R, X, B;
    assert(hint >= 0 && hint <= 3);
    rxb(hint & 3, b, x, &R, &X, &B);
    emit_rex(e, 0, R, X, B);
    em_u8(e, 0x0F);
    em_u8(e, 0x18);
    emit_mrm(e, hint & 3, b, x, sc, disp, 1);
}

void em_movdqa_rr_sse(em_t *e, int vd, int vs)
{
    /* register-register: alignment-irrelevant; use the aligned form so no
     * unaligned vector-memory opcode appears anywhere in emitted code. */
    sse_op_mrm(e, 0x66, 0x6F, vd, 0, 0, -1, 0, 0, vs);
}

void em_pxor_sse(em_t *e, int vd, int vs)
{
    sse_op_mrm(e, 0x66, 0xEF, vd, 0, 0, -1, 0, 0, vs);
}
void em_pxor_mem_sse(em_t *e, int vd, int b, int x, int sc, int disp)
{
    /* PXOR xmm1, xmm2/m128: 66 0F EF /r. The memory form is valid
     * legacy SSE -- one uop vs movdqa+pxor's two. */
    sse_op_mrm(e, 0x66, 0xEF, vd, 1, b, x, sc, disp, 0);
}

void em_pshufb_sse(em_t *e, int vd, int vs)
{
    int R, X, B;
    rxb(vd, vs, -1, &R, &X, &B);
    em_u8(e, 0x66);
    emit_rex(e, 0, R, X, B);
    em_u8(e, 0x0F);
    em_u8(e, 0x38);
    em_u8(e, 0x00);
    em_u8(e, (unsigned)(0xC0 | ((vd & 7) << 3) | (vs & 7)));
}

void em_pand_sse(em_t *e, int vd, int vs)
{
    sse_op_mrm(e, 0x66, 0xDB, vd, 0, 0, -1, 0, 0, vs);
}

void em_psrlw_sse(em_t *e, int vd, unsigned imm)
{
    int R, X, B;
    /* 0F 71 is a group opcode: /2 sits in ModRM.reg, the destination
     * register is in ModRM.r/m, so vd's high bit goes to REX.B
     * (rxb's 2nd operand), not REX.R. */
    rxb(0, vd, -1, &R, &X, &B);
    em_u8(e, 0x66);
    emit_rex(e, 0, R, X, B);
    em_u8(e, 0x0F);
    em_u8(e, 0x71);
    em_u8(e, (unsigned)(0xD0 | (vd & 7))); /* /2 */
    em_u8(e, imm & 0xFF);
}

void em_psllw_sse(em_t *e, int vd, unsigned imm)
{
    int R, X, B;
    /* 0F 71 /6: word shift left by immediate (SSE form of VPSLLW). */
    rxb(0, vd, -1, &R, &X, &B);
    em_u8(e, 0x66);
    emit_rex(e, 0, R, X, B);
    em_u8(e, 0x0F);
    em_u8(e, 0x71);
    em_u8(e, (unsigned)(0xF0 | (vd & 7))); /* /6 */
    em_u8(e, imm & 0xFF);
}

void em_cmp_rr(em_t *e, int d, int s)
{
    int R, X, B;
    rxb(d, s, -1, &R, &X, &B);
    emit_rex(e, 1, R, X, B);
    em_u8(e, 0x3B); /* CMP r64, r/m64: flags for d - s */
    em_u8(e, (unsigned)(0xC0 | ((d & 7) << 3) | (s & 7)));
}

void em_mov_r8r8(em_t *e, int d, int s)
{
    int R, X, B;
    rxb(d, s, -1, &R, &X, &B);
    check_byte_rex(d, s, R, X, B);
    emit_rex(e, 0, R, X, B);
    em_u8(e, 0x8A);
    em_u8(e, (unsigned)(0xC0 | ((d & 7) << 3) | (s & 7)));
}

/* VPCLMULQDQ (carryless multiply, quadword granular).
 * EVEX.66.0F3A.W1 44 /r ib : VPCLMULQDQ zmm1, zmm2, zmm3, imm8.
 * One 64x64->128 carryless multiply per 128-bit lane:
 *   imm=0x00: lane = CLMUL(src1[63:0], src2[63:0])
 *   imm=0x11: lane = CLMUL(src1[127:64], src2[127:64])
 *   imm=0x01: lane = CLMUL(src1[63:0], src2[127:64])
 *   imm=0x10: lane = CLMUL(src1[127:64], src2[63:0])
 * Requires AVX512F + VPCLMULQDQ. (Phase 2 REDUX does not use this for
 * the CRC fold — the per-stream 128-bit fold uses em_pclmulqdq instead.
 * Kept for other consumers.) */
void em_vpclmulqdq(em_t *e, int vd, int vs1, int vs2, unsigned imm)
{
    /* dest (vd) in ModRM.reg, src2 (vs2) in r/m, src1 (vs1) in vvvv.
     * EVEX.R/R' extend reg (vd); EVEX.X/B extend r/m (vs2). */
    int R = (vd >> 3) & 1;
    int X = (vs2 >> 4) & 1;
    int B = (vs2 >> 3) & 1;
    int R2 = (vd >> 4) & 1;
    /* WIG: W is ignored; use 0 to match GAS. */
    emit_evex(e, 3, 0, R, X, B, R2, vs1, 2, 1, 0, 0);
    em_u8(e, 0x44);
    em_u8(e, (unsigned)(0xC0 | ((vd & 7) << 3) | (vs2 & 7)));
    em_u8(e, imm & 0xFF);
}

/* VEXTRACTI32X4: extract 128 bits from a zmm to an xmm.
 * EVEX.66.0F3A.W0 39 /r ib : VEXTRACTI32X4 xmm1, zmm2, imm8.
 * imm8 selects the 128-bit lane (0..3). Dest xmm is zero-extended.
 * Requires AVX512F. Used by the crc_fold twin to split a 64B vector
 * into 16B chunks for the sequential single-stream CRC fold. */
void em_vextracti32x4(em_t *e, int vd, int vs, unsigned imm)
{
    /* VEXTRACT is store-like: src (zmm) in ModRM.reg, dest (xmm) in r/m.
     * EVEX.R/R' extend reg (vs); EVEX.X/B extend r/m (vd).
     * vvvv unused: 1111b (R=1, V'=1). */
    int R = (vs >> 3) & 1;
    int X = (vd >> 4) & 1;
    int B = (vd >> 3) & 1;
    int R2 = (vs >> 4) & 1;
    /* vvvv=0 encodes 1111b (inverted storage); LL=2 for the 512-bit source. */
    emit_evex(e, 3, 0, R, X, B, R2, 0, 2, 1, 0, 0);
    em_u8(e, 0x39);
    em_u8(e, (unsigned)(0xC0 | ((vs & 7) << 3) | (vd & 7)));
    em_u8(e, imm & 0xFF);
}

/* PCLMULQDQ (carryless multiply, quadword granular), 128-bit VEX form.
 * VEX.128.66.0F3A.44 /r ib : PCLMULQDQ xmm1, xmm2, xmm3/m128, imm8.
 * xmm1(dst) = CLMUL(xmm2, xmm3) with the 64-bit halves selected by imm8:
 *   imm=0x00: dst = CLMUL(src1[63:0], src2[63:0])   (lo*lo)
 *   imm=0x11: dst = CLMUL(src1[127:64], src2[127:64]) (hi*hi)
 *   imm=0x01: dst = CLMUL(src1[127:64], src2[63:0])  (hi*lo)
 *   imm=0x10: dst = CLMUL(src1[63:0], src2[127:64])  (lo*hi)
 * Non-destructive 3-operand: vd=xmm1 (ModRM.reg), vs1=xmm2 (VEX.vvvv),
 * vs2=xmm3 (ModRM r/m). WIG (W=0 to match GAS). VEX addresses only
 * xmm0-15: callers must keep all three operands in the low 16 vector
 * registers (the crc_fold backend reserves its temps/consts there).
 * Requires PCLMULQDQ. Used by the REDUX crc_fold (per-stream 128-bit
 * fold); the 512-bit em_vpclmulqdq stays for other consumers.
 * (GAS assembles this form under the vpclmul{lq|hq}{lq|hq}dq mnemonics;
 * the legacy 66 0F 3A 44 form is a different, destructive-2-operand
 * encoding and is NOT what this emitter produces.) */
void em_pclmulqdq(em_t *e, int vd, int vs1, int vs2, unsigned imm)
{
    int R = (vd >> 3) & 1;
    int B = (vs2 >> 3) & 1;
    /* 3-byte VEX (map 3 = 0F3A); W=0, L=0 (128-bit), pp=1 (66). */
    emit_vex(e, 3, 0, R, 0, B, vs1, 0, 1);
    em_u8(e, 0x44);
    em_u8(e, (unsigned)(0xC0 | ((vd & 7) << 3) | (vs2 & 7)));
    em_u8(e, imm & 0xFF);
}

/* VPXORD, 128-bit EVEX form: vd = s1 ^ s2 (32-bit lanes).
 * EVEX.66.0F.W0 EF /r : bits [511:128] of vd's zmm are zeroed.
 * Needed for the crc_fold state XORs: the VEX VPXOR cannot address
 * xmm16-31, but fold states may live there. Requires AVX512F. */
void em_vpxor_evex128(em_t *e, int vd, int s1, int s2)
{
    int R = (vd >> 3) & 1;
    int X = (s2 >> 4) & 1;
    int B = (s2 >> 3) & 1;
    int R2 = (vd >> 4) & 1;
    emit_evex(e, 1, 0, R, X, B, R2, s1, 0, 1, 0, 0);
    em_u8(e, 0xEF);
    em_u8(e, (unsigned)(0xC0 | ((vd & 7) << 3) | (s2 & 7)));
}

/* VMOVDQA32, 128-bit EVEX forms (load/store).
 * EVEX.66.0F.W0 6F /r (load) and 7F /r (store): 16-byte aligned.
 * Needed for crc_fold state load/store: the VEX VMOVDQA cannot address
 * xmm16-31. Requires AVX512F. */
void em_vmovdqa_ld_evex128(em_t *e, int vd, int b, int x, int sc, int disp)
{
    int R, X, B;
    rxb(vd, b, x, &R, &X, &B);
    int R2 = (vd >> 4) & 1;
    emit_evex(e, 1, 0, R, X, B, R2, 0, 0, 1, 0, 0);
    em_u8(e, 0x6F);
    /* EVEX disp8 is scaled by the 16-byte memory operand size. */
    emit_mrm(e, vd, b, x, sc, disp, 16);
}

void em_vmovdqa_st_evex128(em_t *e, int b, int x, int sc, int disp, int vs)
{
    int R, X, B;
    rxb(vs, b, x, &R, &X, &B);
    int R2 = (vs >> 4) & 1;
    emit_evex(e, 1, 0, R, X, B, R2, 0, 0, 1, 0, 0);
    em_u8(e, 0x7F);
    /* EVEX disp8 is scaled by the 16-byte memory operand size. */
    emit_mrm(e, vs, b, x, sc, disp, 16);
}
