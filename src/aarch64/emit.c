/* raidjit AArch64 machine-code emitter. All encodings verified against
 * aarch64-linux-gnu-as; see tests/test_aarch64_emit.c. */
#include "emit.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

#define FX_B26   1   /* B:   imm26 = (target-pc)/4            */
#define FX_B19   2   /* B.cc/CBZ: imm19 = (target-pc)/4       */
#define FX_ADR21 3   /* ADR: immlo/immhi, offset = target-pc */

void em_init(em_t *e, uint8_t *buf, size_t cap)
{
    memset(e, 0, sizeof(*e));
    e->buf = buf;
    e->cap = cap;
    e->lb.n = e->lb.nfx = 0;
    e->lb.cap = e->lb.fxcap = 0;
    e->lb.pos = NULL;
    e->lb.fx = NULL;
}

void em_free(em_t *e)
{
    free(e->lb.pos);
    free(e->lb.fx);
    e->lb.pos = NULL;
    e->lb.fx = NULL;
}

void em_byte(em_t *e, uint8_t v)
{
    if (e->len >= e->cap) { e->err = 1; return; }
    e->buf[e->len++] = v;
}

void em_word(em_t *e, uint32_t w)
{
    if (e->len + 4 > e->cap) { e->err = 1; return; }
    e->buf[e->len++] = (uint8_t)(w & 0xff);
    e->buf[e->len++] = (uint8_t)((w >> 8) & 0xff);
    e->buf[e->len++] = (uint8_t)((w >> 16) & 0xff);
    e->buf[e->len++] = (uint8_t)((w >> 24) & 0xff);
}

void em_align(em_t *e, unsigned a)
{
    size_t want = ((e->len + ((size_t)1 << a) - 1) >> a) << a;
    while (e->len < want)
        em_word(e, 0xD503201Fu); /* hint #0 = nop */
}

/* Pad to 2^a alignment with UDF #0 (permanently undefined). Dead space
 * after ret only: a fall-through past ret must fault immediately, never
 * slide through NOPs into the const pool. The 64-byte pool alignment is
 * always a whole number of 4-byte words, so no odd-byte filler is needed. */
void em_trap_align(em_t *e, unsigned a)
{
    assert((e->len & 3) == 0); /* all emits are 32-bit; no sub-word tail */
    size_t want = ((e->len + ((size_t)1 << a) - 1) >> a) << a;
    while (e->len < want)
        em_udf(e);
}

static void lb_grow(em_t *e)
{
    if (e->lb.n >= e->lb.cap) {
        int nc = e->lb.cap ? e->lb.cap * 2 : 16;
        int *np = realloc(e->lb.pos, (size_t)nc * sizeof(*np));
        if (!np) { e->err = 1; return; }
        for (int i = e->lb.cap; i < nc; i++) np[i] = -1;
        e->lb.pos = np;
        e->lb.cap = nc;
    }
}

static void fx_grow(em_t *e)
{
    if (e->lb.nfx >= e->lb.fxcap) {
        int nc = e->lb.fxcap ? e->lb.fxcap * 2 : 16;
        fixup_t *nf = realloc(e->lb.fx, (size_t)nc * sizeof(*nf));
        if (!nf) { e->err = 1; return; }
        e->lb.fx = nf;
        e->lb.fxcap = nc;
    }
}

int em_label(em_t *e)
{
    lb_grow(e);
    if (e->err) return -1;
    return e->lb.n++;
}

void em_bind(em_t *e, int lab)
{
    if (lab < 0 || lab >= e->lb.n) { e->err = 1; return; }
    e->lb.pos[lab] = (int)e->len;
}

static uint32_t rd_le(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static void wr_le(uint8_t *p, uint32_t w)
{
    p[0] = (uint8_t)(w & 0xff);
    p[1] = (uint8_t)((w >> 8) & 0xff);
    p[2] = (uint8_t)((w >> 16) & 0xff);
    p[3] = (uint8_t)((w >> 24) & 0xff);
}

void em_resolve(em_t *e)
{
    for (int i = 0; i < e->lb.nfx; i++) {
        int lab = e->lb.fx[i].lab;
        size_t at = e->lb.fx[i].at;
        if (lab < 0 || lab >= e->lb.n || e->lb.pos[lab] < 0) { e->err = 1; return; }
        long off = (long)e->lb.pos[lab] - (long)at; /* bytes, target - pc */
        uint32_t w = rd_le(e->buf + at);
        switch (e->lb.fx[i].kind) {
        case FX_B26:
            if ((off & 3) || off < -(128L << 20) || off >= (128L << 20)) { e->err = 1; return; }
            w |= ((uint32_t)(off >> 2) & 0x03ffffffu);
            break;
        case FX_B19:
            if ((off & 3) || off < -(1L << 20) || off >= (1L << 20)) { e->err = 1; return; }
            w |= (((uint32_t)(off >> 2) & 0x7ffffu) << 5);
            break;
        case FX_ADR21:
            if ((off & 3) || off < -(1L << 20) || off >= (1L << 20)) { e->err = 1; return; }
            w |= (((uint32_t)off & 3u) << 29) | ((((uint32_t)(off >> 2)) & 0x7ffffu) << 5);
            break;
        default:
            e->err = 1;
            return;
        }
        wr_le(e->buf + at, w);
    }
}

/* ---- control flow ---- */

void em_ret(em_t *e) { em_word(e, 0xD65F03C0u); }                 /* ret x30 */
void em_udf(em_t *e) { em_word(e, 0x00000000u); }                 /* udf #0 */

static void em_fx_at(em_t *e, int lab, int kind, size_t at)
{
    fx_grow(e);
    if (e->err) return;
    e->lb.fx[e->lb.nfx].lab = lab;
    e->lb.fx[e->lb.nfx].at = at;
    e->lb.fx[e->lb.nfx].kind = kind;
    e->lb.nfx++;
}

void em_b(em_t *e, int lab)
{
    size_t at = e->len;
    em_word(e, 0x14000000u);
    em_fx_at(e, lab, FX_B26, at);
}
void em_bcc(em_t *e, int cc, int lab)
{
    size_t at = e->len;
    em_word(e, 0x54000000u | ((uint32_t)cc & 15u));
    em_fx_at(e, lab, FX_B19, at);
}
void em_cbz(em_t *e, int rt, int lab)
{
    size_t at = e->len;
    em_word(e, 0xB4000000u | ((uint32_t)rt & 31u));
    em_fx_at(e, lab, FX_B19, at);
}
void em_cbz_w(em_t *e, int rt, int lab)
{
    size_t at = e->len;
    em_word(e, 0x34000000u | ((uint32_t)rt & 31u));
    em_fx_at(e, lab, FX_B19, at);
}
void em_cbnz(em_t *e, int rt, int lab)
{
    size_t at = e->len;
    em_word(e, 0xB5000000u | ((uint32_t)rt & 31u));
    em_fx_at(e, lab, FX_B19, at);
}
void em_adr(em_t *e, int rd, int lab)
{
    size_t at = e->len;
    em_word(e, 0x10000000u | ((uint32_t)rd & 31u));
    em_fx_at(e, lab, FX_ADR21, at);
}

/* ---- 64-bit integer ---- */

void em_mov_rr(em_t *e, int rd, int rs)  /* orr xD,xzr,xS */
{
    em_word(e, 0xAA000000u | (((uint32_t)rs & 31u) << 16) | (31u << 5) |
                     ((uint32_t)rd & 31u));
}
void em_movz(em_t *e, int rd, unsigned imm16)
{
    em_word(e, 0xD2800000u | (((uint32_t)imm16 & 0xffffu) << 5) | ((uint32_t)rd & 31u));
}
void em_add_ri(em_t *e, int rd, int rn, unsigned imm)
{
    em_word(e, 0x91000000u | (((uint32_t)imm & 0xfffu) << 10) |
                     (((uint32_t)rn & 31u) << 5) | ((uint32_t)rd & 31u));
}
void em_add_lsl12(em_t *e, int rd, int rn, unsigned imm)
{
    em_word(e, 0x91400000u | (((uint32_t)imm & 0xfffu) << 10) |
                     (((uint32_t)rn & 31u) << 5) | ((uint32_t)rd & 31u));
}
void em_add_rr(em_t *e, int rd, int rn, int rm)
{
    em_word(e, 0x8B000000u | (((uint32_t)rm & 31u) << 16) |
                     (((uint32_t)rn & 31u) << 5) | ((uint32_t)rd & 31u));
}
void em_sub_rr(em_t *e, int rd, int rn, int rm)
{
    em_word(e, 0xCB000000u | (((uint32_t)rm & 31u) << 16) |
                     (((uint32_t)rn & 31u) << 5) | ((uint32_t)rd & 31u));
}
void em_subs_rr(em_t *e, int rd, int rn, int rm) /* subs xRd,xRn,xRm */
{
    /* 0xEB0C01AD verified by GAS for subs x13,x13,x12. */
    em_word(e, 0xEB000000u | (((uint32_t)rm & 31u) << 16) |
                     (((uint32_t)rn & 31u) << 5) | ((uint32_t)rd & 31u));
}
void em_subs_ri(em_t *e, int rd, int rn, unsigned imm)
{
    em_word(e, 0xF1000000u | (((uint32_t)imm & 0xfffu) << 10) |
                     (((uint32_t)rn & 31u) << 5) | ((uint32_t)rd & 31u));
}
void em_cmp_rr(em_t *e, int rn, int rm)  /* subs xzr,xN,xM */
{
    em_word(e, 0xEB000000u | (((uint32_t)rm & 31u) << 16) |
                     (((uint32_t)rn & 31u) << 5) | 31u);
}
void em_cmp_ri(em_t *e, int rn, unsigned imm)
{
    em_word(e, 0xF1000000u | (((uint32_t)imm & 0xfffu) << 10) |
                     (((uint32_t)rn & 31u) << 5) | 31u);
}
void em_lsl_ri(em_t *e, int rd, int rn, unsigned sh) /* ubfm alias */
{
    unsigned immr = (64u - sh) & 63u, imms = 63u - sh;
    em_word(e, 0xD3400000u | (immr << 16) | (imms << 10) |
                     (((uint32_t)rn & 31u) << 5) | ((uint32_t)rd & 31u));
}
void em_lsr_ri(em_t *e, int rd, int rn, unsigned sh) /* ubfm alias */
{
    em_word(e, 0xD3400000u | ((sh & 63u) << 16) | (63u << 10) |
                     (((uint32_t)rn & 31u) << 5) | ((uint32_t)rd & 31u));
}
void em_mul(em_t *e, int rd, int rn, int rm)
{
    em_word(e, 0x9B000000u | (((uint32_t)rm & 31u) << 16) |
                     (((uint32_t)rn & 31u) << 5) | (31u << 10) |
                     ((uint32_t)rd & 31u));
}
void em_udiv(em_t *e, int rd, int rn, int rm)
{
    em_word(e, 0x9AC00800u | (((uint32_t)rm & 31u) << 16) |
                     (((uint32_t)rn & 31u) << 5) | ((uint32_t)rd & 31u));
}
void em_ldr64(em_t *e, int rt, int rn, unsigned off)
{
    em_word(e, 0xF9400000u | ((((uint32_t)off >> 3) & 0xfffu) << 10) |
                     (((uint32_t)rn & 31u) << 5) | ((uint32_t)rt & 31u));
}
/* str xRt, [xRn, #off]!  (off: signed, -256..255); ldr xRt, [xRn], #off */
void em_str64_pre(em_t *e, int rt, int rn, int off)
{
    em_word(e, 0xF8000000u | ((((uint32_t)off) & 0x1ffu) << 12) | (3u << 10) |
                     (((uint32_t)rn & 31u) << 5) | ((uint32_t)rt & 31u));
}
void em_ldr64_post(em_t *e, int rt, int rn, int off)
{
    em_word(e, 0xF8000000u | (1u << 22) | ((((uint32_t)off) & 0x1ffu) << 12) |
                     (1u << 10) | (((uint32_t)rn & 31u) << 5) |
                     ((uint32_t)rt & 31u));
}
void em_ldrb(em_t *e, int rt, int rn, int rm) /* wRt = [xRn + xRm] */
{
    em_word(e, 0x38606800u | (((uint32_t)rm & 31u) << 16) |
                     (((uint32_t)rn & 31u) << 5) | ((uint32_t)rt & 31u));
}
void em_ldrb_ri(em_t *e, int rt, int rn, unsigned off)
{
    em_word(e, 0x39400000u | (((uint32_t)off & 0xfffu) << 10) |
                     (((uint32_t)rn & 31u) << 5) | ((uint32_t)rt & 31u));
}
void em_strb(em_t *e, int rt, int rn, int rm) /* [xRn + xRm] = wRt */
{
    em_word(e, 0x38206800u | (((uint32_t)rm & 31u) << 16) |
                     (((uint32_t)rn & 31u) << 5) | ((uint32_t)rt & 31u));
}
void em_strb_ri(em_t *e, int rt, int rn, unsigned off)
{
    em_word(e, 0x39000000u | (((uint32_t)off & 0xfffu) << 10) |
                     (((uint32_t)rn & 31u) << 5) | ((uint32_t)rt & 31u));
}

/* ---- 32-bit integer (scalar tail) ---- */

void em_mov_wr(em_t *e, int rd, int rs)  /* orr wD,wzr,wS */
{
    em_word(e, 0x2A000000u | (((uint32_t)rs & 31u) << 16) | (31u << 5) |
                     ((uint32_t)rd & 31u));
}
void em_mov_wi(em_t *e, int rd, unsigned imm16)
{
    em_word(e, 0x52800000u | (((uint32_t)imm16 & 0xffffu) << 5) | ((uint32_t)rd & 31u));
}
void em_eor_w(em_t *e, int rd, int rn, int rm)
{
    em_word(e, 0x4A000000u | (((uint32_t)rm & 31u) << 16) |
                     (((uint32_t)rn & 31u) << 5) | ((uint32_t)rd & 31u));
}
void em_add_wr_i(em_t *e, int rd, int rn, unsigned imm)
{
    em_word(e, 0x11000000u | (((uint32_t)imm & 0xfffu) << 10) |
                     (((uint32_t)rn & 31u) << 5) | ((uint32_t)rd & 31u));
}
void em_sub_wr_i(em_t *e, int rd, int rn, unsigned imm)
{
    em_word(e, 0x51000000u | (((uint32_t)imm & 0xfffu) << 10) |
                     (((uint32_t)rn & 31u) << 5) | ((uint32_t)rd & 31u));
}
void em_cmp_wr_i(em_t *e, int rn, unsigned imm)
{
    em_word(e, 0x71000000u | (((uint32_t)imm & 0xfffu) << 10) |
                     (((uint32_t)rn & 31u) << 5) | 31u);
}

/* ---- prefetch ---- */

void em_prfm(em_t *e, unsigned prfop, int rn, unsigned off)
{
    /* verified: imm12 field = off/8 at [21:10]; off must be 8-aligned
     * and <= 32760. The backend validates this (jit.c, "not encodable")
     * so the mask below can never silently truncate a bad input. */
    em_word(e, 0xF9800000u | ((((uint32_t)off >> 3) & 0xfffu) << 10) |
                     (((uint32_t)rn & 31u) << 5) | ((uint32_t)prfop & 31u));
}

/* ---- NEON ---- */

void em_neon_ldr(em_t *e, int qt, int rn, unsigned off)
{
    em_word(e, 0x3DC00000u | ((((uint32_t)off >> 4) & 0xfffu) << 10) |
                     (((uint32_t)rn & 31u) << 5) | ((uint32_t)qt & 31u));
}
void em_neon_str(em_t *e, int qt, int rn, unsigned off)
{
    em_word(e, 0x3D800000u | ((((uint32_t)off >> 4) & 0xfffu) << 10) |
                     (((uint32_t)rn & 31u) << 5) | ((uint32_t)qt & 31u));
}
void em_neon_stnp(em_t *e, int qt1, int qt2, int rn, int off)
{
    em_word(e, 0xAC000000u | ((((uint32_t)(off >> 4)) & 0x7fu) << 15) |
                     (((uint32_t)qt2 & 31u) << 10) |
                     (((uint32_t)rn & 31u) << 5) | ((uint32_t)qt1 & 31u));
}
void em_neon_eor(em_t *e, int vd, int vn, int vm)
{
    em_word(e, 0x6E201C00u | (((uint32_t)vm & 31u) << 16) |
                     (((uint32_t)vn & 31u) << 5) | ((uint32_t)vd & 31u));
}
void em_neon_and(em_t *e, int vd, int vn, int vm)
{
    em_word(e, 0x4E201C00u | (((uint32_t)vm & 31u) << 16) |
                     (((uint32_t)vn & 31u) << 5) | ((uint32_t)vd & 31u));
}
void em_neon_orr(em_t *e, int vd, int vn, int vm)
{
    em_word(e, 0x4EA01C00u | (((uint32_t)vm & 31u) << 16) |
                     (((uint32_t)vn & 31u) << 5) | ((uint32_t)vd & 31u));
}
void em_neon_ushr(em_t *e, int vd, int vn, unsigned sh) /* .16b */
{
    /* USHR immh:immb encodes (2*esize - shift); esize=8 for .16b */
    em_word(e, 0x6F000400u | ((((16u - sh) & 63u) << 16) |
                     (((uint32_t)vn & 31u) << 5) | ((uint32_t)vd & 31u)));
}
void em_neon_shl(em_t *e, int vd, int vn, unsigned sh) /* .16b */
{
    /* SHL immh:immb = 8+sh for .16b (like SVE byte shifts); U=0 (bit 29)
     * vs USHR's U=1. GAS-verified 2026-10-10 (base 0x4F005400):
     *   shl v16.16b,v17.16b,#1 -> 0x4f095630 (imm=0x09)
     *   shl v16.16b,v17.16b,#7 -> 0x4f0f5630 (imm=0x0f) */
    assert(sh >= 1 && sh <= 7);
    em_word(e, 0x4F005400u | ((((8u + sh) & 63u) << 16) |
                     (((uint32_t)vn & 31u) << 5) | ((uint32_t)vd & 31u)));
}
void em_neon_tbl(em_t *e, int vd, int vn, int vm) /* vd = tbl({vn}, vm) */
{
    em_word(e, 0x4E000000u | (((uint32_t)vm & 31u) << 16) |
                     (((uint32_t)vn & 31u) << 5) | ((uint32_t)vd & 31u));
}
void em_neon_movi(em_t *e, int vd, unsigned imm8)
{
    em_word(e, 0x4F00E400u | ((((uint32_t)imm8 >> 5) & 3u) << 16) |
                     (((uint32_t)imm8 & 31u) << 5) | ((uint32_t)vd & 31u));
}
void em_neon_stp_pre(em_t *e, int qt1, int qt2, int rn, int off)
{
    em_word(e, 0xAD800400u | ((((uint32_t)(off >> 4)) & 0x7fu) << 15) |
                     (((uint32_t)qt2 & 31u) << 10) |
                     (((uint32_t)rn & 31u) << 5) | ((uint32_t)qt1 & 31u));
}
void em_neon_ldp_post(em_t *e, int qt1, int qt2, int rn, int off)
{
    em_word(e, 0xACC00400u | ((((uint32_t)(off >> 4)) & 0x7fu) << 15) |
                     (((uint32_t)qt2 & 31u) << 10) |
                     (((uint32_t)rn & 31u) << 5) | ((uint32_t)qt1 & 31u));
}

/* ---- SVE ---- */

void em_sve_whilelo(em_t *e, int pd, int rn, int rm) /* whilelo pd.b, xRn, xRm */
{
    em_word(e, 0x25201C00u | (((uint32_t)rm & 31u) << 16) |
                     (((uint32_t)rn & 31u) << 5) | ((uint32_t)pd & 15u));
}
/* WHILELT pd.b, xRn, xRm: element i active iff (Rn + i) < Rm (unsigned).
 * Base 0x25201400 derived from GAS (armv8-a+sve) and cross-checked on 5
 * operand combos (p1.b,xzr,x13 -> 0x252d17e1; p0.b,x5,x6 -> 0x252614a0;
 * p7.b,x0,x1 -> 0x25211407; p15.b,x30,x29 -> 0x253d17cf;
 * p3.b,x16,x17 -> 0x25311603). The naive guess 0x25209C00 was wrong --
 * always GAS-check a new emitter base. */
void em_sve_whilelt(em_t *e, int pd, int rn, int rm) /* whilelt pd.b, xRn, xRm */
{
    em_word(e, 0x25201400u | (((uint32_t)rm & 31u) << 16) |
                     (((uint32_t)rn & 31u) << 5) | ((uint32_t)pd & 15u));
}
void em_sve_ptrue(em_t *e, int pd) /* ptrue pd.b, all */
{
    em_word(e, 0x2518E3E0u | ((uint32_t)pd & 15u));
}
void em_sve_cntb(em_t *e, int xd) /* cntb xD, all, mul #1 */
{
    /* Base has the Xd field (bits 4:0) cleared; the old base 0x0420E3EC
     * had Xd=12 baked in and ORed xd (worked by accident for X12). */
    em_word(e, 0x0420E3E0u | ((uint32_t)xd & 31u));
}
void em_sve_ld1b(em_t *e, int zt, int pg, int rn)
{
    em_word(e, 0xA400A000u | (((uint32_t)pg & 7u) << 10) |
                     (((uint32_t)rn & 31u) << 5) | ((uint32_t)zt & 31u));
}
/* LD1B (scalar + immediate, MUL VL): ld1b zt.b, pg/z, [xRn, #off, MUL VL].
 * The offset is a signed 4-bit immediate in units of the runtime vector
 * length (bits 19:16; GAS rejects anything outside -8..7, verified).
 * This is the lane-addressing workhorse: with the base pointer held at
 * the lane-0 address, lane u is [xRn, #u, MUL VL] with no pointer bump.
 * GAS-verified (4 combos incl. p1 and non-zero offsets); the #0 form
 * assembles to the identical word as the plain [xRn] form above. */
void em_sve_ld1b_off(em_t *e, int zt, int pg, int rn, int off)
{
    assert(off >= -8 && off <= 7);
    em_word(e, 0xA400A000u | ((((uint32_t)off & 0xfu)) << 16) |
                     (((uint32_t)pg & 7u) << 10) |
                     (((uint32_t)rn & 31u) << 5) | ((uint32_t)zt & 31u));
}
void em_sve_ld1rqb(em_t *e, int zt, int pg, int rn)
{
    em_word(e, 0xA4002000u | (((uint32_t)pg & 7u) << 10) |
                     (((uint32_t)rn & 31u) << 5) | ((uint32_t)zt & 31u));
}
void em_sve_st1b(em_t *e, int zt, int pg, int rn)
{
    em_word(e, 0xE400E000u | (((uint32_t)pg & 7u) << 10) |
                     (((uint32_t)rn & 31u) << 5) | ((uint32_t)zt & 31u));
}
/* ST1B (scalar + immediate, MUL VL): st1b zt.b, pg, [xRn, #off, MUL VL].
 * Same imm4 encoding/range note as em_sve_ld1b_off. GAS-verified. */
void em_sve_st1b_off(em_t *e, int zt, int pg, int rn, int off)
{
    assert(off >= -8 && off <= 7);
    em_word(e, 0xE400E000u | ((((uint32_t)off & 0xfu)) << 16) |
                     (((uint32_t)pg & 7u) << 10) |
                     (((uint32_t)rn & 31u) << 5) | ((uint32_t)zt & 31u));
}
void em_sve_stnt1b(em_t *e, int zt, int pg, int rn)
{
    em_word(e, 0xE410E000u | (((uint32_t)pg & 7u) << 10) |
                     (((uint32_t)rn & 31u) << 5) | ((uint32_t)zt & 31u));
}
/* STNT1B (scalar + immediate, MUL VL): stnt1b zt.b, pg, [xRn, #off, MUL VL].
 * Same imm4 encoding/range note as em_sve_ld1b_off. GAS-verified. */
void em_sve_stnt1b_off(em_t *e, int zt, int pg, int rn, int off)
{
    assert(off >= -8 && off <= 7);
    em_word(e, 0xE410E000u | ((((uint32_t)off & 0xfu)) << 16) |
                     (((uint32_t)pg & 7u) << 10) |
                     (((uint32_t)rn & 31u) << 5) | ((uint32_t)zt & 31u));
}
void em_sve_eor(em_t *e, int zdn, int pg, int zm) /* destructive */
{
    em_word(e, 0x04190000u | (((uint32_t)pg & 7u) << 10) |
                     (((uint32_t)zm & 31u) << 5) | ((uint32_t)zdn & 31u));
}
/* EOR3 <Zdn>.D, <Zdn>.D, <Zm>.D, <Za>.D : Zdn ^= Zm ^ Za (unpredicated,
 * destructive). Base SVE2 (verified: assembles with -march=armv8.2-a+sve2,
 * rejected with +sve; e.g. eor3 z16.d,z16.d,z17.d,z18.d -> 0x04313a50).
 * Element size is fixed to .D by the architecture; irrelevant for XOR. */
void em_sve_eor3(em_t *e, int zdn, int zm, int za)
{
    em_word(e, 0x04203800u | (((uint32_t)zm & 31u) << 16) |
                     (((uint32_t)za & 31u) << 5) | ((uint32_t)zdn & 31u));
}
void em_sve_and(em_t *e, int zdn, int pg, int zm)
{
    em_word(e, 0x041A0000u | (((uint32_t)pg & 7u) << 10) |
                     (((uint32_t)zm & 31u) << 5) | ((uint32_t)zdn & 31u));
}
/* AND (unpredicated): and zd.d, zn.d, zm.d. Base SVE, single uop, no
 * predicate register involved. The assembler only accepts .d for the
 * unpredicated logicals (element size is fixed by the encoding); the
 * operation is bitwise so .d is semantically identical to .b.
 * GAS-verified: and z16.d,z17.d,z18.d -> 0x04323230. */
void em_sve_and_unpred(em_t *e, int zd, int zn, int zm)
{
    em_word(e, 0x04203000u | (((uint32_t)zm & 31u) << 16) |
                     (((uint32_t)zn & 31u) << 5) | ((uint32_t)zd & 31u));
}
/* EOR (unpredicated): eor zd.d, zn.d, zm.d. Base SVE, single uop.
 * Same .d-only note as AND above.
 * GAS-verified: eor z16.d,z17.d,z18.d -> 0x04b23230. */
void em_sve_eor_unpred(em_t *e, int zd, int zn, int zm)
{
    em_word(e, 0x04a03000u | (((uint32_t)zm & 31u) << 16) |
                     (((uint32_t)zn & 31u) << 5) | ((uint32_t)zd & 31u));
}
void em_sve_orr(em_t *e, int zdn, int pg, int zm) /* destructive: zDn|=zM */
{
    em_word(e, 0x04180000u | (((uint32_t)pg & 7u) << 10) |
                     (((uint32_t)zm & 31u) << 5) | ((uint32_t)zdn & 31u));
}
/* zD.d = zN.d (unpredicated ORR, assembles as MOV): a true vector copy */
void em_sve_mov(em_t *e, int zd, int zn)
{
    em_word(e, 0x04603000u | (((uint32_t)zn & 31u) << 16) |
                     (((uint32_t)zn & 31u) << 5) | ((uint32_t)zd & 31u));
}
void em_sve_lsr(em_t *e, int zdn, int pg, unsigned sh) /* .b, destructive */
{
    em_word(e, 0x04018100u | ((((8u - sh) & 7u) << 5)) |
                     (((uint32_t)pg & 7u) << 10) | ((uint32_t)zdn & 31u));
}
/* LSR (unpredicated, immediate): lsr zd.b, zn.b, #sh. Base SVE, single
 * uop, no predicate register involved.
 * GAS-verified: lsr z16.b,z17.b,#1..#7 -> 0x042f9630..0x04299630
 * (bits 19:16 = 16-sh). Base 0x04209400 is the GAS word with zd/zn/imm
 * fields cleared; do NOT derive it by clearing only imm (that leaves
 * zd/zn bits set and corrupts other operands -- see test regression). */
/* LSL (unpredicated, immediate): lsl zd.b, zn.b, #sh. Base SVE, single
 * uop, no predicate register involved.
 * ENCODING: base 0x04209C00, bits 19:16 = 8+sh (see em_sve_lsl_unpred).
 * GAS-verified 2026-10-10. */
void em_sve_lsr_unpred(em_t *e, int zd, int zn, unsigned sh) /* sh in 1..7 */
{
    assert(sh >= 1 && sh <= 7);
    em_word(e, 0x04209400u | ((((16u - sh) & 15u) << 16) |
                     (((uint32_t)zn & 31u) << 5) | ((uint32_t)zd & 31u)));
}
void em_sve_lsl_unpred(em_t *e, int zd, int zn, unsigned sh) /* sh in 1..7 */
{
    assert(sh >= 1 && sh <= 7);
    /* GAS-verified 2026-10-10: base 0x04209C00, bits 19:16 = 8+sh.
     *   lsl z16.b,z17.b,#1 -> 0x04299e30
     *   lsl z16.b,z17.b,#7 -> 0x042f9e30
     * The 8-bias in the immediate comes from the tszh:tszl:imm3 format
     * for byte elements (ARM ARM DDI 0487, LSL_Z_ZI). A naive "LSR base
     * + bit 13" derivation gets both the base and the immediate wrong. */
    em_word(e, 0x04209C00u | ((((8u + sh) & 15u) << 16) |
                     (((uint32_t)zn & 31u) << 5) | ((uint32_t)zd & 31u)));
}
void em_sve_tbl(em_t *e, int zd, int zn, int zm) /* zd = tbl({zn}, zm) */
{
    em_word(e, 0x05203000u | (((uint32_t)zm & 31u) << 16) |
                     (((uint32_t)zn & 31u) << 5) | ((uint32_t)zd & 31u));
}
void em_sve_dup(em_t *e, int zd, int imm8) /* dup zd.b, #imm8 (signed) */
{
    /* Base 0x2538C000 has the Zd field (bits 4:0) cleared; the old base
     * 0x2538C010 had Zd=16 baked in and ORed zd, which worked by accident
     * for z16-z31 (16|x == x for x >= 16) but misencodes z0-z7
     * (e.g. zd=0 gave z16). GAS-verified: dup z0.b,#15 -> 0x2538C1E0. */
    em_word(e, 0x2538C000u | ((((uint32_t)imm8 & 0xffu) << 5)) |
                     ((uint32_t)zd & 31u));
}
