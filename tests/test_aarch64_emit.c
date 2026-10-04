/* raidjit AArch64 emitter unit test.
 * Runs on ANY host (no execution): emits each supported instruction with
 * several register/operand combos and checks the bytes against the words
 * produced by aarch64-linux-gnu-as. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "../src/aarch64/emit.h"

static int fails = 0, checks = 0;

#define CHECK_WORD(e, expect) do { \
    checks++; \
    if ((e)->len != 4) { printf("FAIL %s:%d: len=%zu\n", __func__, __LINE__, (e)->len); fails++; } \
    else { uint32_t got = (uint32_t)(e)->buf[0] | ((uint32_t)(e)->buf[1] << 8) | \
                          ((uint32_t)(e)->buf[2] << 16) | ((uint32_t)(e)->buf[3] << 24); \
        if (got != (uint32_t)(expect)) { \
            printf("FAIL %s:%d: got %08x want %08x\n", __func__, __LINE__, got, (uint32_t)(expect)); \
            fails++; \
        } \
    } \
    (e)->len = 0; \
} while (0)

static void t_control(void)
{
    em_t e; uint8_t b[64]; em_init(&e, b, sizeof b);
    em_ret(&e);   CHECK_WORD(&e, 0xD65F03C0u);
    em_udf(&e);   CHECK_WORD(&e, 0x00000000u);
    em_free(&e);
}

static void t_int64(void)
{
    em_t e; uint8_t b[64]; em_init(&e, b, sizeof b);
    em_mov_rr(&e, X0, X1);       CHECK_WORD(&e, 0xAA0103E0u);
    em_mov_rr(&e, X11, XZR);     CHECK_WORD(&e, 0xAA1F03EBu);
    em_movz(&e, X14, 0x1234);   CHECK_WORD(&e, 0xD282468Eu);
    em_movz(&e, X0, 0);         CHECK_WORD(&e, 0xD2800000u);
    em_add_ri(&e, X11, X11, 0x80);  CHECK_WORD(&e, 0x9102016Bu);
    em_add_ri(&e, X10, X10, 4095);  CHECK_WORD(&e, 0x913FFD4Au);
    em_add_lsl12(&e, X11, X11, 1);  CHECK_WORD(&e, 0x9140056Bu);
    em_add_rr(&e, X10, X10, X11);   CHECK_WORD(&e, 0x8B0B014Au);
    em_add_rr(&e, X5, X6, X7);      CHECK_WORD(&e, 0x8B0700C5u);
    em_sub_rr(&e, X5, X6, X7);      CHECK_WORD(&e, 0xCB0700C5u);
    em_subs_ri(&e, X12, X12, 1);    CHECK_WORD(&e, 0xF100058Cu);
    em_cmp_rr(&e, X12, X2);         CHECK_WORD(&e, 0xEB02019Fu);
    em_cmp_ri(&e, X12, 0xFF);       CHECK_WORD(&e, 0xF103FD9Fu);
    em_lsl_ri(&e, X13, X12, 3);     CHECK_WORD(&e, 0xD37DF18Du);
    em_lsl_ri(&e, X0, X1, 63);      CHECK_WORD(&e, 0xD3410020u);
    em_lsr_ri(&e, X13, X12, 3);     CHECK_WORD(&e, 0xD343FD8Du);
    em_mul(&e, X15, X14, X13);      CHECK_WORD(&e, 0x9B0D7DCFu);
    em_udiv(&e, X14, X2, X13);      CHECK_WORD(&e, 0x9ACD084Eu);
    em_ldr64(&e, X10, X8, 8);       CHECK_WORD(&e, 0xF940050Au);
    em_ldr64(&e, X0, X1, 0);        CHECK_WORD(&e, 0xF9400020u);
    em_str64_pre(&e, 19, XSP, -16); CHECK_WORD(&e, 0xF81F0FF3u);
    em_ldr64_post(&e, 19, XSP, 16); CHECK_WORD(&e, 0xF84107F3u);
    em_ldrb(&e, 13, 17, 12);        CHECK_WORD(&e, 0x386C6A2Du);
    em_ldrb_ri(&e, 14, 17, 255);    CHECK_WORD(&e, 0x3943FE2Eu);
    em_strb(&e, 12, 10, 12);        CHECK_WORD(&e, 0x382C694Cu);
    em_strb_ri(&e, 12, 10, 7);      CHECK_WORD(&e, 0x39001D4Cu);
    em_free(&e);
}

static void t_int32(void)
{
    em_t e; uint8_t b[64]; em_init(&e, b, sizeof b);
    em_mov_wr(&e, 12, 13);      CHECK_WORD(&e, 0x2A0D03ECu);
    em_mov_wi(&e, 12, 0);       CHECK_WORD(&e, 0x5280000Cu);
    em_eor_w(&e, 12, 12, 13);   CHECK_WORD(&e, 0x4A0D018Cu);
    em_add_wr_i(&e, 14, 14, 200); CHECK_WORD(&e, 0x110321CEu);
    em_sub_wr_i(&e, 14, 14, 255); CHECK_WORD(&e, 0x5103FDCEu);
    em_cmp_wr_i(&e, 14, 255);    CHECK_WORD(&e, 0x7103FDDFu);
    em_free(&e);
}

static void t_prfm(void)
{
    em_t e; uint8_t b[64]; em_init(&e, b, sizeof b);
    em_prfm(&e, PRF_PLDL1KEEP, 10, 0);    CHECK_WORD(&e, 0xF9800140u);
    em_prfm(&e, PRF_PLDL1STRM, 10, 0);    CHECK_WORD(&e, 0xF9800141u);
    em_prfm(&e, PRF_PLDL1STRM, 10, 512);  CHECK_WORD(&e, 0xF9810141u);
    em_prfm(&e, PRF_PLDL1STRM, 10, 4096); CHECK_WORD(&e, 0xF9880141u);
    em_prfm(&e, PRF_PLDL1STRM, 10, 8192);  CHECK_WORD(&e, 0xF9900141u);
    em_free(&e);
}

static void t_neon(void)
{
    em_t e; uint8_t b[64]; em_init(&e, b, sizeof b);
    em_neon_ldr(&e, 16, 8, 0);     CHECK_WORD(&e, 0x3DC00110u);
    em_neon_ldr(&e, 17, 10, 16);   CHECK_WORD(&e, 0x3DC00551u);
    em_neon_str(&e, 18, 10, 16);   CHECK_WORD(&e, 0x3D800552u);
    em_neon_stnp(&e, 0, 1, 10, 0); CHECK_WORD(&e, 0xAC000540u);
    em_neon_stnp(&e, 3, 4, 10, 0); CHECK_WORD(&e, 0xAC001143u);
    em_neon_stnp(&e, 3, 4, 10, -16); CHECK_WORD(&e, 0xAC3F9143u);
    em_neon_eor(&e, 0, 0, 1);      CHECK_WORD(&e, 0x6E211C00u);
    em_neon_and(&e, 4, 5, 6);      CHECK_WORD(&e, 0x4E261CA4u);
    em_neon_orr(&e, 13, 14, 14);   CHECK_WORD(&e, 0x4EAE1DCDu);
    em_neon_ushr(&e, 8, 9, 4);     CHECK_WORD(&e, 0x6F0C0528u);
    em_neon_tbl(&e, 9, 10, 11);    CHECK_WORD(&e, 0x4E0B0149u);
    em_neon_movi(&e, 12, 15);      CHECK_WORD(&e, 0x4F00E5ECu);
    em_neon_stp_pre(&e, 8, 9, 31, -32);  CHECK_WORD(&e, 0xADBF27E8u);
    em_neon_stp_pre(&e, 10, 11, 31, -32); CHECK_WORD(&e, 0xADBF2FEAu);
    em_neon_ldp_post(&e, 8, 9, 31, 32);  CHECK_WORD(&e, 0xACC127E8u);
    em_neon_ldp_post(&e, 14, 15, 31, 32); CHECK_WORD(&e, 0xACC13FEEu);
    em_free(&e);
}

static void t_sve(void)
{
    em_t e; uint8_t b[64]; em_init(&e, b, sizeof b);
    em_sve_whilelo(&e, 0, 31, 12); CHECK_WORD(&e, 0x252C1FE0u);
    /* WHILELT pd.b, xRn, xRm = 0x25201400 | rm<<16 | rn<<5 | pd.
     * Cross-checked against aarch64-linux-gnu-as on low/high,
     * even/odd regs and xzr (track B1sve; base from a GAS word with
     * every operand field cleared). */
    em_sve_whilelt(&e, 1, 31, 13); CHECK_WORD(&e, 0x252D17E1u);
    em_sve_whilelt(&e, 0, 5, 6);   CHECK_WORD(&e, 0x252614A0u);
    em_sve_whilelt(&e, 7, 0, 1);   CHECK_WORD(&e, 0x25211407u);
    em_sve_whilelt(&e, 15, 30, 29); CHECK_WORD(&e, 0x253D17CFu);
    em_sve_whilelt(&e, 3, 16, 17); CHECK_WORD(&e, 0x25311603u);
    em_sve_ptrue(&e, 1);           CHECK_WORD(&e, 0x2518E3E1u);
    em_sve_cntb(&e, 12);           CHECK_WORD(&e, 0x0420E3ECu);
    em_sve_ld1b(&e, 16, 0, 10);    CHECK_WORD(&e, 0xA400A150u);
    em_sve_ld1rqb(&e, 16, 0, 10);  CHECK_WORD(&e, 0xA4002150u);
    em_sve_st1b(&e, 16, 0, 10);    CHECK_WORD(&e, 0xE400E150u);
    em_sve_stnt1b(&e, 18, 0, 10);  CHECK_WORD(&e, 0xE410E152u);
    em_sve_stnt1b(&e, 19, 3, 11);  CHECK_WORD(&e, 0xE410ED73u);
    em_sve_eor(&e, 25, 0, 26);     CHECK_WORD(&e, 0x04190359u);
    /* EOR3 words cross-checked against aarch64-linux-gnu-as:
     *   eor3 z16.d,z16.d,z17.d,z18.d -> 0x04313a50
     *   eor3 z0.d,z0.d,z1.d,z2.d     -> 0x04213840 */
    em_sve_eor3(&e, 16, 17, 18);   CHECK_WORD(&e, 0x04313a50u);
    em_sve_eor3(&e, 0, 1, 2);      CHECK_WORD(&e, 0x04213840u);
    em_sve_eor3(&e, 31, 30, 29);   CHECK_WORD(&e, 0x043E3BBFu);
    em_sve_and(&e, 28, 0, 29);     CHECK_WORD(&e, 0x041A03BCu);
    em_sve_orr(&e, 28, 0, 29);     CHECK_WORD(&e, 0x041803BCu);
    em_sve_mov(&e, 17, 20);        CHECK_WORD(&e, 0x04743291u);
    em_sve_lsr(&e, 22, 0, 4);      CHECK_WORD(&e, 0x04018196u);
    em_sve_lsr(&e, 23, 0, 1);      CHECK_WORD(&e, 0x040181F7u);
    /* Unpredicated logicals/shift (N1/N2): GAS cross-checked --
     *   and z16.d,z17.d,z18.d -> 0x04323230
     *   eor z16.d,z17.d,z18.d -> 0x04b23230
     *   lsr z16.b,z17.b,#1..#7 -> 0x042f9630..0x04299630 */
    em_sve_and_unpred(&e, 16, 17, 18); CHECK_WORD(&e, 0x04323230u);
    em_sve_and_unpred(&e, 0, 1, 2);    CHECK_WORD(&e, 0x04223020u);
    em_sve_eor_unpred(&e, 16, 17, 18); CHECK_WORD(&e, 0x04b23230u);
    em_sve_eor_unpred(&e, 31, 30, 29); CHECK_WORD(&e, 0x04bd33dfu);
    em_sve_lsr_unpred(&e, 16, 17, 1);  CHECK_WORD(&e, 0x042f9630u);
    em_sve_lsr_unpred(&e, 16, 17, 4);  CHECK_WORD(&e, 0x042c9630u);
    em_sve_lsr_unpred(&e, 16, 17, 7);  CHECK_WORD(&e, 0x04299630u);
    /* Regression: the LSR base was once derived by clearing only the
     * imm field from a GAS word, leaving zd/zn bits set (0x04209630).
     * It produced the right word for (16,17) by accident but corrupted
     * zn for other operands (e.g. (22,18,4) -> 0x042c9676, zn=19).
     * These vectors pin the clean base 0x04209400. */
    em_sve_lsr_unpred(&e, 22, 18, 4);  CHECK_WORD(&e, 0x042c9656u);
    em_sve_lsr_unpred(&e, 0, 1, 7);    CHECK_WORD(&e, 0x04299420u);
    em_sve_lsr_unpred(&e, 31, 30, 1);  CHECK_WORD(&e, 0x042f97dfu);
    em_sve_tbl(&e, 25, 26, 27);    CHECK_WORD(&e, 0x053B3359u);
    em_sve_dup(&e, 30, 15);        CHECK_WORD(&e, 0x2538C1FEu);
    em_sve_dup(&e, 21, -128);      CHECK_WORD(&e, 0x2538D015u);
    em_free(&e);
}

/* branch fixups: forward/backward B, B.cc, CBZ, ADR */
static void t_branch(void)
{
    em_t e; uint8_t b[128]; em_init(&e, b, sizeof b);
    int l0 = em_label(&e), l1 = em_label(&e), l2 = em_label(&e);
    em_b(&e, l1);          /* +8 */
    em_word(&e, 0xD503201Fu);
    em_bind(&e, l1);
    em_cbz(&e, X12, l0);   /* backward -8 */
    em_bind(&e, l0);
    em_adr(&e, X16, l2);
    em_bind(&e, l2);
    em_bcc(&e, CC_LO, l2); /* +4 */
    em_bind(&e, l2);
    em_cbnz(&e, X14, l1);  /* backward */
    em_resolve(&e);
    /* verify by decoding offsets */
    checks++;
    uint32_t w;
    memcpy(&w, b + 0, 4);  /* B +8 -> imm26 = 2 */
    if ((w & 0xFC000000u) != 0x14000000u || (w & 0x3FFFFFFu) != 2) {
        printf("FAIL B fwd: %08x\n", w); fails++;
    }
    checks++;
    memcpy(&w, b + 8, 4);  /* CBZ x12 -> l0: off = +4, imm19 = 1 */
    if ((w & 0xFF000000u) != 0xB4000000u || ((w >> 5) & 0x7FFFFu) != 1u) {
        printf("FAIL CBZ fwd: %08x\n", w); fails++;
    }
    checks++;
    memcpy(&w, b + 12, 4); /* ADR x16 -> l2: off = 8 */
    if ((w & 0x9F000000u) != 0x10000000u || (w & 0x1Fu) != 16 ||
        (((w >> 29) & 3u) | (((w >> 5) & 0x7FFFFu) << 2)) != 8u) {
        printf("FAIL ADR fwd: %08x\n", w); fails++;
    }
    checks++;
    memcpy(&w, b + 16, 4); /* B.lo -> l2 (rebound to +4): imm19 = 1 */
    if (w != (0x54000000u | (1u << 5) | CC_LO)) { printf("FAIL B.cc fwd: %08x\n", w); fails++; }
    checks++;
    memcpy(&w, b + 20, 4); /* CBNZ x14 -> l1 (off = -12): imm19 = -3 */
    if ((w & 0xFF000000u) != 0xB5000000u || ((w >> 5) & 0x7FFFFu) != 0x7FFFDu) {
        printf("FAIL CBNZ back: %08x\n", w); fails++;
    }
    if (e.err) { printf("FAIL: emitter err flag\n"); fails++; }
    em_free(&e);
}

int main(void)
{
    t_control();
    t_int64();
    t_int32();
    t_prfm();
    t_neon();
    t_sve();
    t_branch();
    printf("%s: %d checks, %d failures\n", fails ? "FAIL" : "PASS", checks, fails);
    return fails != 0;
}
