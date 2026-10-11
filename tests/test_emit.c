/* Differential test: emitter output vs GNU as reference encodings.
 *
 * Reference bytes were produced with GNU as and cross-checked by hand for
 * the integer two-register forms, where as picks the r/m,dst opcode
 * direction (0x01/0x31/0x39/0x30) and the emitter picks the reg,dst
 * direction (0x03/0x33/0x3b/0x32) -- same semantics.
 */
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include "../src/x86_64/emit.h"

static int fails;

#define CHECK(name, ...)                                                     \
    do {                                                                     \
        static const uint8_t want[] = { __VA_ARGS__ };                        \
        uint8_t buf[64];                                                     \
        em_t e;                                                              \
        em_init(&e, buf, sizeof buf);                                        \
        name;                                                                \
        if (e.len != sizeof want || memcmp(buf, want, sizeof want)) {        \
            printf("FAIL ");                                                 \
            printf(#name);                                                   \
            printf("\n  got :");                                             \
            for (size_t i = 0; i < e.len; i++)                                \
                printf(" %02x", buf[i]);                                     \
            printf("\n  want:");                                             \
            for (size_t i = 0; i < sizeof want; i++)                          \
                printf(" %02x", want[i]);                                    \
            printf("\n");                                                    \
            fails++;                                                         \
        }                                                                    \
        em_free(&e);                                                         \
    } while (0)

int main(void)
{
    /* ---- EVEX register forms, high regs in every position ---- */
    /* vpxord %zmm16, %zmm8, %zmm8  (vd=8, vvvv=8, rm=16) */
    CHECK(em_vpxor(&e, 8, 8, 16, VL64),
          0x62, 0x31, 0x3d, 0x48, 0xef, 0xc0);
    /* vpxord %zmm8, %zmm16, %zmm16 (vd=16, vvvv=16, rm=8) */
    CHECK(em_vpxor(&e, 16, 16, 8, VL64),
          0x62, 0xc1, 0x7d, 0x40, 0xef, 0xc0);
    /* vpxord %zmm29, %zmm30, %zmm31 */
    CHECK(em_vpxor(&e, 31, 30, 29, VL64),
          0x62, 0x01, 0x0d, 0x40, 0xef, 0xfd);
    /* vpxord %zmm16, %zmm16, %zmm16 */
    CHECK(em_vpxor(&e, 16, 16, 16, VL64),
          0x62, 0xa1, 0x7d, 0x40, 0xef, 0xc0);
    /* vpandd %zmm19, %zmm18, %zmm17 */
    CHECK(em_vpand(&e, 17, 18, 19, VL64),
          0x62, 0xa1, 0x6d, 0x40, 0xdb, 0xcb);
    /* vpshufb %zmm22, %zmm21, %zmm20 */
    CHECK(em_vpshufb(&e, 20, 21, 22, VL64),
          0x62, 0xa2, 0x55, 0x40, 0x00, 0xe6);
    /* vgf2p8affineqb $1, %zmm24, %zmm25, %zmm23 */
    CHECK(em_vgf2p8affineqb(&e, 23, 25, 24, 1, VL64),
          0x62, 0x83, 0xb5, 0x40, 0xce, 0xf8, 0x01);
    /* vgf2p8affineqb $0, %zmm8, %zmm18, %zmm17 */
    CHECK(em_vgf2p8affineqb(&e, 17, 18, 8, 0, VL64),
          0x62, 0xc3, 0xed, 0x40, 0xce, 0xc8, 0x00);
    /* vpternlogd $0x96, %zmm1, %zmm2, %zmm3 */
    CHECK(em_vpternlog(&e, 3, 2, 1, 0x96, VL64),
          0x62, 0xf3, 0x6d, 0x48, 0x25, 0xd9, 0x96);
    /* vpternlogd $0x96, %zmm17, %zmm18, %zmm31 (high regs) */
    CHECK(em_vpternlog(&e, 31, 18, 17, 0x96, VL64),
          0x62, 0x23, 0x6d, 0x40, 0x25, 0xf9, 0x96);
    /* vpternlogd $0x96, (%r10,%r11,1), %zmm1, %zmm0 (mem form) */
    CHECK(em_vpternlog_mem(&e, 0, 1, 10, 11, 0, 0, 0x96, VL64),
          0x62, 0x93, 0x75, 0x48, 0x25, 0x04, 0x1a, 0x96);
    /* vpternlogd $0x96, 0x140(%r10,%r11,1), %zmm17, %zmm31 (high regs,
     * 320/64=5 fits disp8) */
    CHECK(em_vpternlog_mem(&e, 31, 17, 10, 11, 0, 0x140, 0x96, VL64),
          0x62, 0x03, 0x75, 0x40, 0x25, 0x7c, 0x1a, 0x05, 0x96);
    /* vpternlogd $0xe8, 100(%r8,%r9,4), %zmm26, %zmm17 (100 not divisible
     * by 64 -> disp32) */
    CHECK(em_vpternlog_mem(&e, 17, 26, 8, 9, 2, 100, 0xe8, VL64),
          0x62, 0x83, 0x2d, 0x40, 0x25, 0x8c, 0x88, 0x64, 0x00, 0x00, 0x00,
          0xe8);
    /* vpternlogd $0x96, (%rdx,%r11,1), %zmm16, %zmm0: B=0 (base rdx),
     * the form the P_XOR twin actually emits for in[b] */
    CHECK(em_vpternlog_mem(&e, 0, 16, 2, 11, 0, 0, 0x96, VL64),
          0x62, 0xb3, 0x7d, 0x40, 0x25, 0x04, 0x1a, 0x96);
    /* vpternlogd $0x96, 0x40(%rdx,%r11,1), %zmm16, %zmm8 (B=0, disp8) */
    CHECK(em_vpternlog_mem(&e, 8, 16, 2, 11, 0, 64, 0x96, VL64),
          0x62, 0x33, 0x7d, 0x40, 0x25, 0x44, 0x1a, 0x01, 0x96);
    /* vpermi2b %zmm1, %zmm2, %zmm3 (track B2; GAS 2.42 reference bytes:
     * 62 f2 6d 48 75 d9) */
    CHECK(em_vpermi2b(&e, 3, 2, 1, VL64),
          0x62, 0xf2, 0x6d, 0x48, 0x75, 0xd9);
    /* vpermi2b %zmm17, %zmm18, %zmm31 (high regs: vvvv bit4 -> V'=0,
     * rm bit4 -> X, dest bit4 -> R') */
    CHECK(em_vpermi2b(&e, 31, 18, 17, VL64),
          0x62, 0x22, 0x6d, 0x40, 0x75, 0xf9);
    /* vpord %zmm1, %zmm2, (%r10,%r11,1) (track B2: the memory-source
     * VPORB equivalent; GAS rejects vporb with zmm operands) */
    CHECK(em_vpor_mem(&e, 2, 1, 10, 11, 0, 0, VL64),
          0x62, 0x91, 0x75, 0x48, 0xeb, 0x14, 0x1a);
    /* vpmultishiftqb %zmm2, %zmm1, %zmm0 (track I2; GAS 2.42 reference
     * bytes: 62 f2 f5 48 83 c2; note W=1, unlike the W0 neighbors) */
    CHECK(em_vpmultishiftqb(&e, 0, 1, 2, VL64),
          0x62, 0xf2, 0xf5, 0x48, 0x83, 0xc2);
    /* vpmultishiftqb %zmm17, %zmm18, %zmm31 (high regs: vvvv bit4 -> V'=0,
     * rm bit4 -> X, dest bit4 -> R'; GAS 2.42: 62 22 ed 40 83 f9) */
    CHECK(em_vpmultishiftqb(&e, 31, 18, 17, VL64),
          0x62, 0x22, 0xed, 0x40, 0x83, 0xf9);
    /* vpsrlw $4, %zmm11 -> %zmm10 (high regs) */
    CHECK(em_vpsrlw(&e, 10, 11, 4, VL64),
          0x62, 0xd1, 0x2d, 0x48, 0x71, 0xd3, 0x04);
    /* vpsrlw $4, %zmm16 -> %zmm18: src bit4 rides EVEX.X (P0[6]), and
     * EVEX.R' must be 0 because ModRM.reg is the /2 opcode extension */
    CHECK(em_vpsrlw(&e, 18, 16, 4, VL64),
          0x62, 0xb1, 0x6d, 0x40, 0x71, 0xd0, 0x04);

    /* ---- EVEX memory forms: compressed disp8 vs disp32 ---- */
    /* vmovdqa32 0x40(%r10), %zmm17: 64/64=1 fits disp8 */
    CHECK(em_vmovdqa_ld(&e, 17, 10, -1, 0, 64, VL64),
          0x62, 0xc1, 0x7d, 0x48, 0x6f, 0x4a, 0x01);
    /* vmovdqa32 0xc0(%r10), %zmm19: 192/64=3 fits disp8 */
    CHECK(em_vmovdqa_ld(&e, 19, 10, -1, 0, 192, VL64),
          0x62, 0xc1, 0x7d, 0x48, 0x6f, 0x5a, 0x03);
    /* vmovdqa32 0x64(%r10), %zmm20: 100 not divisible by 64 -> disp32 */
    CHECK(em_vmovdqa_ld(&e, 20, 10, -1, 0, 100, VL64),
          0x62, 0xc1, 0x7d, 0x48, 0x6f, 0xa2, 0x64, 0x00, 0x00, 0x00);
    /* vmovdqa32 (%r10,%r11,1), %zmm16 */
    CHECK(em_vmovdqa_ld(&e, 16, 10, 11, 0, 0, VL64),
          0x62, 0x81, 0x7d, 0x48, 0x6f, 0x04, 0x1a);
    /* vmovdqa32 %zmm18, 0x80(%r10): 128/64=2 fits disp8 */
    CHECK(em_vmovdqa_st(&e, 10, -1, 0, 128, 18, VL64),
          0x62, 0xc1, 0x7d, 0x48, 0x7f, 0x52, 0x02);
    /* vmovdqa32 %zmm31, (%r8,%r9,1) */
    CHECK(em_vmovdqa_st(&e, 8, 9, 0, 0, 31, VL64),
          0x62, 0x01, 0x7d, 0x48, 0x7f, 0x3c, 0x08);
    /* vpxord (%r10,%r11,1), %zmm5, %zmm16 */
    CHECK(em_vpxor_mem(&e, 16, 5, 10, 11, 0, 0, VL64),
          0x62, 0x81, 0x55, 0x48, 0xef, 0x04, 0x1a);

    /* ---- VEX (AVX2/SSE levels), high regs ---- */
    /* vpxor %ymm14, %ymm13, %ymm12 */
    CHECK(em_vpxor(&e, 12, 13, 14, VL32),
          0xc4, 0x41, 0x15, 0xef, 0xe6);
    /* vpshufb %ymm9, %ymm11, %ymm10 */
    CHECK(em_vpshufb(&e, 10, 11, 9, VL32),
          0xc4, 0x42, 0x25, 0x00, 0xd1);
    /* vpxor %xmm10, %xmm9, %xmm8 */
    CHECK(em_vpxor(&e, 8, 9, 10, VL16),
          0xc4, 0x41, 0x31, 0xef, 0xc2);
    /* vpxor (%r10,%r11,1), %ymm5, %ymm14 */
    CHECK(em_vpxor_mem(&e, 14, 5, 10, 11, 0, 0, VL32),
          0xc4, 0x01, 0x55, 0xef, 0x34, 0x1a);
    /* vmovdqa 64(%r10), %ymm12 */
    CHECK(em_vmovdqa_ld(&e, 12, 10, -1, 0, 64, VL32),
          0xc4, 0x41, 0x7d, 0x6f, 0x62, 0x40);
    /* vmovdqa %xmm9, 32(%r10) */
    CHECK(em_vmovdqa_st(&e, 10, -1, 0, 32, 9, VL16),
          0xc4, 0x41, 0x79, 0x7f, 0x4a, 0x20);

    /* ---- legacy SSE: 66 prefix must precede REX ---- */
    /* movdqa 16(%r10), %xmm8 */
    CHECK(em_movdqa_ld_sse(&e, 8, 10, -1, 0, 16),
          0x66, 0x45, 0x0f, 0x6f, 0x42, 0x10);
    /* movdqa %xmm9, (%r10) */
    CHECK(em_movdqa_st_sse(&e, 10, -1, 0, 0, 9),
          0x66, 0x45, 0x0f, 0x7f, 0x0a);
    /* pxor %xmm9, %xmm8 */
    CHECK(em_pxor_sse(&e, 8, 9),
          0x66, 0x45, 0x0f, 0xef, 0xc1);
    /* pshufb %xmm11, %xmm10 */
    CHECK(em_pshufb_sse(&e, 10, 11),
          0x66, 0x45, 0x0f, 0x38, 0x00, 0xd3);
    /* psrlw $4, %xmm2 (low reg, no REX) */
    CHECK(em_psrlw_sse(&e, 2, 4),
          0x66, 0x0f, 0x71, 0xd2, 0x04);
    /* psrlw $4, %xmm10: 0F 71 is a group opcode (/2 in ModRM.reg), the
     * destination lives in ModRM.r/m, so the high bit needs REX.B (0x41),
     * not REX.R */
    CHECK(em_psrlw_sse(&e, 10, 4),
          0x66, 0x41, 0x0f, 0x71, 0xd2, 0x04);

    /* ---- integer two-register forms (direction verified by hand) ---- */
    /* add %r13, %r12  (r12 += r13); as emits 4d 01 ec, same semantics */
    CHECK(em_add_rr(&e, 12, 13),
          0x4d, 0x03, 0xe5);
    /* xor %r15, %r14  (r14 ^= r15); as emits 4d 31 fe */
    CHECK(em_xor_rr(&e, 14, 15),
          0x4d, 0x33, 0xf7);
    /* cmp %r9, %r8; as emits 4d 39 c8 */
    CHECK(em_cmp_rr(&e, 8, 9),
          0x4d, 0x3b, 0xc1);
    /* xor %r9b, %r8b; as emits 45 30 c8 */
    CHECK(em_xor_r8(&e, 8, 9),
          0x45, 0x32, 0xc1);
    /* movzbl (%r10), %r12d */
    CHECK(em_movzx_rm8(&e, 12, 10, -1, 0, 0),
          0x45, 0x0f, 0xb6, 0x22);

    /* ---- r12 as SIB index (REX.X=1): reference bytes from GNU as ----
     * SIB index field 100b means "no index" only when REX.X=0; with
     * REX.X=1 it encodes r12. r4 (RSP) remains illegal as an index.
     * Each vector below was assembled with `as` and the emitter output
     * compared byte-for-byte against the objdump disassembly. */
    /* mov (%rbx,%r12,1),%rax: REX carries X only (0x4a), no other bits */
    CHECK(em_mov_rm(&e, 0, 3, 12, 0, 0),
          0x4a, 0x8b, 0x04, 0x23);
    /* mov 0x10(%rsi,%r12,2),%r9: REX.WRX, disp8 */
    CHECK(em_mov_rm(&e, 9, 6, 12, 1, 16),
          0x4e, 0x8b, 0x4c, 0x66, 0x10);
    /* mov (%r15,%r12,4),%r13: REX.WRXB, no disp */
    CHECK(em_mov_rm(&e, 13, 15, 12, 2, 0),
          0x4f, 0x8b, 0x2c, 0xa7);
    /* mov 0x1234(%r8,%r12,8),%rax: REX.WXB, disp32 */
    CHECK(em_mov_rm(&e, 0, 8, 12, 3, 0x1234),
          0x4b, 0x8b, 0x84, 0xe0, 0x34, 0x12, 0x00, 0x00);
    /* mov -8(%rbp,%r12,1),%rcx: REX.WX, disp8, rbp base */
    CHECK(em_mov_rm(&e, 1, 5, 12, 0, -8),
          0x4a, 0x8b, 0x4c, 0x25, 0xf8);
    /* mov %r10,(%r11,%r12,1): store form, REX.WRXB */
    CHECK(em_mov_mr(&e, 11, 12, 0, 0, 10),
          0x4f, 0x89, 0x14, 0x23);
    /* lea (%rdi,%r12,4),%rax: REX.WX */
    CHECK(em_lea(&e, 0, 7, 12, 2, 0),
          0x4a, 0x8d, 0x04, 0xa7);
    /* r12 as base (not index): encodings unchanged by the index fix */
    /* mov (%r12),%rax */
    CHECK(em_mov_rm(&e, 0, 12, -1, 0, 0),
          0x49, 0x8b, 0x04, 0x24);
    /* mov 0x20(%r12),%rdx */
    CHECK(em_mov_rm(&e, 2, 12, -1, 0, 32),
          0x49, 0x8b, 0x54, 0x24, 0x20);
    /* vmovdqa (%r10,%r12,1),%ymm14: VEX X bit set (byte2 0x01) */
    CHECK(em_vmovdqa_ld(&e, 14, 10, 12, 0, 0, VL32),
          0xc4, 0x01, 0x7d, 0x6f, 0x34, 0x22);
    /* vmovdqa32 (%r10,%r12,1),%zmm16: EVEX X bit set (P0 0x81) */
    CHECK(em_vmovdqa_ld(&e, 16, 10, 12, 0, 0, VL64),
          0x62, 0x81, 0x7d, 0x48, 0x6f, 0x04, 0x22);

    /* ---- non-temporal stores + sfence (reference: GNU as, hand-checked) */
    /* movntdq %xmm8, (%r10) */
    CHECK(em_movntdq_st_sse(&e, 10, -1, 0, 0, 8),
          0x66, 0x45, 0x0f, 0xe7, 0x02);
    /* vmovntdq %xmm9, (%r10) */
    CHECK(em_vmovntdq_st(&e, 10, -1, 0, 0, 9, VL16),
          0xc4, 0x41, 0x79, 0xe7, 0x0a);
    /* vmovntdq %ymm10, 0x40(%r10) */
    CHECK(em_vmovntdq_st(&e, 10, -1, 0, 64, 10, VL32),
          0xc4, 0x41, 0x7d, 0xe7, 0x52, 0x40);
    /* vmovntdq %zmm17, (%r10) */
    CHECK(em_vmovntdq_st(&e, 10, -1, 0, 0, 17, VL64),
          0x62, 0xc1, 0x7d, 0x48, 0xe7, 0x0a);
    /* vmovntdq %zmm31, 0x1c0(%r15): 0x1c0/64=7 fits compressed disp8 */
    CHECK(em_vmovntdq_st(&e, 15, -1, 0, 0x1c0, 31, VL64),
          0x62, 0x41, 0x7d, 0x48, 0xe7, 0x7f, 0x07);
    /* sfence */
    CHECK(em_sfence(&e),
          0x0f, 0xae, 0xf8);

    /* ---- multi-byte NOPs, byte-exact (Agner Fog sec. 10.7 / AMD 16h/17h) ----
     * em_align pads with the longest single NOP available (now 15 bytes).
     * Each test emits (16-n) 0xcc filler bytes then aligns to 16, so the
     * padding is exactly one n-byte NOP. */
    /* 10-byte: 66 66 0F 1F 84 00 00 00 00 00 */
    CHECK({ for (int i = 0; i < 6; i++) em_byte(&e, 0xcc); em_align(&e, 16); },
          0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc,
          0x66, 0x66, 0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00);
    /* 11-byte */
    CHECK({ for (int i = 0; i < 5; i++) em_byte(&e, 0xcc); em_align(&e, 16); },
          0xcc, 0xcc, 0xcc, 0xcc, 0xcc,
          0x66, 0x66, 0x66, 0x0f, 0x1f, 0x84,
          0x00, 0x00, 0x00, 0x00, 0x00);
    /* 12-byte */
    CHECK({ for (int i = 0; i < 4; i++) em_byte(&e, 0xcc); em_align(&e, 16); },
          0xcc, 0xcc, 0xcc, 0xcc,
          0x66, 0x66, 0x66, 0x66, 0x0f, 0x1f, 0x84,
          0x00, 0x00, 0x00, 0x00, 0x00);
    /* 13-byte */
    CHECK({ for (int i = 0; i < 3; i++) em_byte(&e, 0xcc); em_align(&e, 16); },
          0xcc, 0xcc, 0xcc,
          0x66, 0x66, 0x66, 0x66, 0x66, 0x0f, 0x1f, 0x84,
          0x00, 0x00, 0x00, 0x00, 0x00);
    /* 14-byte */
    CHECK({ for (int i = 0; i < 2; i++) em_byte(&e, 0xcc); em_align(&e, 16); },
          0xcc, 0xcc,
          0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x0f, 0x1f, 0x84,
          0x00, 0x00, 0x00, 0x00, 0x00);
    /* 15-byte */
    CHECK({ em_byte(&e, 0xcc); em_align(&e, 16); },
          0xcc,
          0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x0f, 0x1f, 0x84,
          0x00, 0x00, 0x00, 0x00, 0x00);
    /* composite 16 = 15 + 1 */
    CHECK({ for (int i = 0; i < 16; i++) em_byte(&e, 0xcc); em_align(&e, 32); },
          0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc,
          0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc,
          0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x0f, 0x1f, 0x84,
          0x00, 0x00, 0x00, 0x00, 0x00,
          0x90);
    /* composite 20 = 15 + 5 */
    CHECK({ for (int i = 0; i < 12; i++) em_byte(&e, 0xcc); em_align(&e, 32); },
          0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc,
          0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc,
          0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x0f, 0x1f, 0x84,
          0x00, 0x00, 0x00, 0x00, 0x00,
          0x0f, 0x1f, 0x44, 0x00, 0x00);
    /* composite 31 = 15 + 15 + 1 */
    CHECK({ em_byte(&e, 0xcc); em_align(&e, 32); },
          0xcc,
          0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x0f, 0x1f, 0x84,
          0x00, 0x00, 0x00, 0x00, 0x00,
          0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x0f, 0x1f, 0x84,
          0x00, 0x00, 0x00, 0x00, 0x00,
          0x90);

    /* ---- masked AVX-512 tail emitters (track B1x86) ----
     * Byte-exact vs GAS 2.42 (captured 2026-10-03; the .s source is
     * archived at ~/workspace/raidjit-tracks/Bx86/gasref.s). These are
     * the load-bearing encodings of the masked tail: a wrong k/z bit or
     * a VMOVDQA64-form opcode silently mis-masks the tail, and the
     * disp8*N scaling (N=64) must match GAS exactly. Red-green: flipping
     * any byte of a 'want' sequence fails the test. */
    /* kmovq %k1, 0x1234(%r15,%rdx,8): VEX.0F.W1 90 /r, dscale=1 */
    CHECK(em_kmovq_km(&e, 1, R15, RDX, 3, 0x1234),
          0xc4, 0xc1, 0xf8, 0x90, 0x8c, 0xd7, 0x34, 0x12, 0x00, 0x00);
    /* kmovq %k5, 0x10(%rax): no index */
    CHECK(em_kmovq_km(&e, 5, RAX, -1, 0, 0x10),
          0xc4, 0xe1, 0xf8, 0x90, 0x68, 0x10);
    /* vmovdqu8 (%r8,%r11,1), %zmm5{%k1}{z}: F2 0F 6F, z=1 */
    CHECK(em_vmovdqu8_ld_masked(&e, 5, R8, R11, 0, 0, VL64, 1, 1),
          0x62, 0x91, 0x7f, 0xc9, 0x6f, 0x2c, 0x18);
    /* vmovdqu8 0x40(%r15,%rdx,8), %zmm30{%k7}: high reg, disp8*N=64*1,
     * merging (z=0) */
    CHECK(em_vmovdqu8_ld_masked(&e, 30, R15, RDX, 3, 64, VL64, 7, 0),
          0x62, 0x41, 0x7f, 0x4f, 0x6f, 0x74, 0xd7, 0x01);
    /* vmovdqu8 %zmm4, (%rax,%r11,1){%k1}: merging store (no {z}) */
    CHECK(em_vmovdqu8_st_masked(&e, RAX, R11, 0, 0, 4, VL64, 1),
          0x62, 0xb1, 0x7f, 0x49, 0x7f, 0x24, 0x18);
    /* vmovdqu8 %zmm31, 128(%r9){%k3}: disp8*N=64*2, high regs */
    CHECK(em_vmovdqu8_st_masked(&e, R9, -1, 0, 128, 31, VL64, 3),
          0x62, 0x41, 0x7f, 0x4b, 0x7f, 0x79, 0x02);
    /* vpshufb %zmm2, %zmm1, %zmm3{%k2}{z}: reg-reg masked form */
    CHECK(em_vpshufb_masked(&e, 3, 1, 2, VL64, 2, 1),
          0x62, 0xf2, 0x75, 0xca, 0x00, 0xda);
    /* vpshufb %zmm25, %zmm20, %zmm29{%k4}: high regs, merging */
    CHECK(em_vpshufb_masked(&e, 29, 20, 25, VL64, 4, 0),
          0x62, 0x02, 0x5d, 0x44, 0x00, 0xe9);

    /* ---- rsp-as-index must stay rejected ----
     * RSP has no high bit, so SIB index field 100b with REX.X=0 is
     * indistinguishable from "no index"; the emitter must refuse it.
     * The assert aborts the process, so check it in a child. */
    {
        pid_t p = fork();
        if (p < 0) {
            printf("FAIL rsp-as-index: fork failed\n");
            fails++;
        } else if (p == 0) {
            uint8_t buf[64];
            em_t e;
            em_init(&e, buf, sizeof buf);
            em_mov_rm(&e, 0, 3, 4, 0, 0); /* x=RSP: must not encode */
            em_free(&e);
            _exit(0); /* reached: the guard is gone */
        } else {
            int st;
            if (waitpid(p, &st, 0) < 0 ||
                !(WIFSIGNALED(st) && WTERMSIG(st) == SIGABRT)) {
                printf("FAIL rsp-as-index not rejected (child exit=%d)\n",
                       WIFEXITED(st) ? WEXITSTATUS(st) : -1);
                fails++;
            }
        }
    }

    if (!fails)
        printf("emitter differential: all OK\n");

    /* ---- REDUX crc_fold emitters (GAS cross-checked 2026-10-10) ---- */
    /* em_pclmulqdq: VEX.128.66.0F3A.44 /r ib.
     * GAS reference: `vpclmullqlqdq %xmm2, %xmm1, %xmm0` (vd=0, vs1=1,
     * vs2=2, imm=0x00) -> c4 e3 71 44 c2 00. AT&T order is (vs2, vs1,
     * vd); 406/406 randomized cases byte-exact, 8000/8000 vectors
     * functionally verified against a C 64x64->128 CLMUL model. */
    CHECK(em_pclmulqdq(&e, 0, 1, 2, 0x00),
          0xc4, 0xe3, 0x71, 0x44, 0xc2, 0x00);
    /* `vpclmulhqhqdq %xmm15, %xmm15, %xmm15` (vd=vs1=vs2=15, imm=0x11) */
    CHECK(em_pclmulqdq(&e, 15, 15, 15, 0x11),
          0xc4, 0x43, 0x01, 0x44, 0xff, 0x11);
    /* `vpclmulhqlqdq %xmm9, %xmm10, %xmm11` (vd=11, vs1=10, vs2=9,
     * imm=0x01) */
    CHECK(em_pclmulqdq(&e, 11, 10, 9, 0x01),
          0xc4, 0x43, 0x29, 0x44, 0xd9, 0x01);
    /* `vpclmullqhqdq %xmm14, %xmm3, %xmm7` (vd=7, vs1=3, vs2=14,
     * imm=0x10): high regs in vd and the r/m position */
    CHECK(em_pclmulqdq(&e, 7, 3, 14, 0x10),
          0xc4, 0xc3, 0x61, 0x44, 0xfe, 0x10);
    /* em_vpxor_evex128: `vpxord %xmm16, %xmm8, %xmm8` (vd=8, s1=8,
     * s2=16) */
    CHECK(em_vpxor_evex128(&e, 8, 8, 16),
          0x62, 0x31, 0x3d, 0x08, 0xef, 0xc0);
    /* `vpxord %xmm31, %xmm30, %xmm29` */
    CHECK(em_vpxor_evex128(&e, 29, 30, 31),
          0x62, 0x01, 0x0d, 0x00, 0xef, 0xef);
    /* em_vmovdqa_ld_evex128: `vmovdqa32 64(%r10,%r11,1), %xmm8` */
    CHECK(em_vmovdqa_ld_evex128(&e, 8, 10, 11, 0, 64),
          0x62, 0x11, 0x7d, 0x08, 0x6f, 0x44, 0x1a, 0x04);
    /* em_vmovdqa_st_evex128: `vmovdqa32 %xmm24, 128(%r10)` (EVEX disp8
     * scaled by 16) */
    CHECK(em_vmovdqa_st_evex128(&e, 10, -1, 0, 128, 24),
          0x62, 0x41, 0x7d, 0x08, 0x7f, 0x42, 0x08);
    /* `vmovdqa32 %xmm5, (%r10)` (disp 0) */
    CHECK(em_vmovdqa_st_evex128(&e, 10, -1, 0, 0, 5),
          0x62, 0xd1, 0x7d, 0x08, 0x7f, 0x2a);

    if (!fails)
        printf("crc_fold emitters: all OK\n");
    return fails != 0;
}
