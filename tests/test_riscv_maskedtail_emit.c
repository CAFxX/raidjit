/* RVV maskedtail emitter shape test (portable: pure byte checks, runs on
 * any host -- src/riscv/emit.c is plain C).
 *
 * Pins the exact encodings of the maskedtail twin's new emitters:
 *   em_vle8_v_m / em_vse8_v_m (masked unit-stride byte load/store, vm=0)
 *   em_vid_v                   (vid.v: OPMVV, funct6=010100, vs1=10001)
 *   em_vmsltu_vx               (vmsltu.vx: OPIVX, funct6=011010, vd=v0)
 *   em_blt / em_bge            (B-type, funct3=100/101; the tail's blez)
 *   em_srai / em_and           (the tail's min(A3, VLMAX) clamp)
 *
 * Checks:
 *  A. Literal-word pins: GAS-verified words (even/odd, low/high regs
 *     per AGENTS.md). A single wrong field bit fails.
 *  B. Field decode: for randomized operands, every field of the emitted
 *     word (funct6, vm, vs2, vs1/rs1, funct3, vd, opcode) is decoded
 *     back and checked -- an independent re-derivation of the encoding
 *     that catches systematic field-placement slips the literal pins
 *     could share with the emitter.
 *  C. Masked-vs-unmasked: em_vle8_v_m/em_vse8_v_m differ from em_vle8/
 *     em_vse8 in exactly bit 25 (the vm bit) and nothing else.
 *  D. vmsltu.vx vd!=v0 trips the emitter assert (death check via a
 *     forked child, so the parent keeps running).
 *
 * GAS-verified against riscv64-linux-gnu-as (-march=rv64gcv); every
 * word in section A was assembled independently and compared
 * word-for-word with the emitter output:
 *   vle8.v v0,(x0),v0.t  -> 0x00000007   vse8.v v0,(x0),v0.t  -> 0x00000027
 *   vle8.v v31,(x31),v0.t-> 0x000f8f87   vse8.v v31,(x31),v0.t-> 0x000f8fa7
 *   vle8.v v1,(x2),v0.t  -> 0x00010087   vse8.v v2,(x1),v0.t  -> 0x00008127
 *   vle8.v v16,(x17),v0.t-> 0x00088807   vse8.v v17,(x16),v0.t-> 0x000808a7
 *   vid.v v0  -> 0x5208a057    vid.v v5  -> 0x5208a2d7
 *   vid.v v31 -> 0x5208afd7    vid.v v16 -> 0x5208a857
 *   vmsltu.vx v0,v0,x0   -> 0x6a004057
 *   vmsltu.vx v0,v31,x31 -> 0x6bffc057
 *   vmsltu.vx v0,v1,x2   -> 0x6a114057
 *   vmsltu.vx v0,v16,x17 -> 0x6b08c057
 *   srai t3,t2,63 -> 0x43f3de13    and t2,t2,t3 -> 0x01c3f3b3
 *   srai x0,x1,0  -> 0x4000d013    and x0,x0,x1  -> 0x00107033
 *   srai x31,x30,31 -> 0x41ff5f93  and x31,x30,x29 -> 0x01df7fb3
 * Note: vmsltu.vx is funct6=011010 (unsigned <). 011011 is vmslt.vx
 * (signed <, wrong for tmp >= 128 at SEW=8) and 011100 is vmsleu.vx
 * (unsigned <=, one extra lane); the GAS cross-check caught the 011100
 * slip before it shipped.
 */
#include "../src/riscv/emit.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

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

static uint32_t word_at(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint64_t xrng_state = 0x243f6a8885a308d3ULL;
static uint64_t xrng(void)
{
    uint64_t x = xrng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return xrng_state = x;
}

/* ---------------- A. literal-word pins (all GAS-verified) ------------ */

static void test_literal_pins(void)
{
    /* em_vle8_v_m: (rs1<<15)|(vd<<7)|0x07, bit 25 = 0 */
    struct { int vd, rs1; uint32_t want; } ldt[] = {
        { 0, 0, 0x00000007u },
        { 31, 31, 0x000f8f87u },
        { 1, 2, 0x00010087u },
        { 16, 17, 0x00088807u }, /* even/odd, low/high mix */
    };
    for (size_t i = 0; i < sizeof(ldt) / sizeof(ldt[0]); i++) {
        uint8_t buf[8];
        em_t e;
        em_init(&e, buf, sizeof(buf));
        em_vle8_v_m(&e, ldt[i].vd, ldt[i].rs1);
        uint32_t got = word_at(buf);
        CHECK(got == ldt[i].want, "vle8.v.m v%d,(x%d): got 0x%08x want 0x%08x",
              ldt[i].vd, ldt[i].rs1, got, ldt[i].want);
        em_free(&e);
    }
    /* em_vse8_v_m: same with the store opcode 0x27 */
    struct { int vs3, rs1; uint32_t want; } stt[] = {
        { 0, 0, 0x00000027u },
        { 31, 31, 0x000f8fa7u },
        { 2, 1, 0x00008127u },
        { 17, 16, 0x000808a7u },
    };
    for (size_t i = 0; i < sizeof(stt) / sizeof(stt[0]); i++) {
        uint8_t buf[8];
        em_t e;
        em_init(&e, buf, sizeof(buf));
        em_vse8_v_m(&e, stt[i].vs3, stt[i].rs1);
        uint32_t got = word_at(buf);
        CHECK(got == stt[i].want, "vse8.v.m v%d,(x%d): got 0x%08x want 0x%08x",
              stt[i].vs3, stt[i].rs1, got, stt[i].want);
        em_free(&e);
    }
    /* em_vid_v: (010100<<26)|(1<<25)|(10001<<15)|(010<<12)|(vd<<7)|0x57 */
    struct { int vd; uint32_t want; } vidt[] = {
        { 0, 0x5208a057u },
        { 31, 0x5208afd7u },
        { 5, 0x5208a2d7u },
        { 16, 0x5208a857u },
    };
    for (size_t i = 0; i < sizeof(vidt) / sizeof(vidt[0]); i++) {
        uint8_t buf[8];
        em_t e;
        em_init(&e, buf, sizeof(buf));
        em_vid_v(&e, vidt[i].vd);
        uint32_t got = word_at(buf);
        CHECK(got == vidt[i].want, "vid.v v%d: got 0x%08x want 0x%08x",
              vidt[i].vd, got, vidt[i].want);
        em_free(&e);
    }
    /* em_vmsltu_vx:
     * (011010<<26)|(1<<25)|(vs2<<20)|(rs1<<15)|(100<<12)|0x57 (vd=v0) */
    struct { int vs2, rs1; uint32_t want; } cmpt[] = {
        { 0, 0, 0x6a004057u },
        { 31, 31, 0x6bffc057u },
        { 1, 2, 0x6a114057u },
        { 16, 17, 0x6b08c057u },
    };
    for (size_t i = 0; i < sizeof(cmpt) / sizeof(cmpt[0]); i++) {
        uint8_t buf[8];
        em_t e;
        em_init(&e, buf, sizeof(buf));
        em_vmsltu_vx(&e, 0, cmpt[i].vs2, cmpt[i].rs1);
        uint32_t got = word_at(buf);
        CHECK(got == cmpt[i].want,
              "vmsltu.vx v0,v%d,x%d: got 0x%08x want 0x%08x", cmpt[i].vs2,
              cmpt[i].rs1, got, cmpt[i].want);
        em_free(&e);
    }
    /* em_srai: (0100000<<25)|(sh<<20)|(rs1<<15)|(101<<12)|(rd<<7)|0x13 */
    struct { int rd, rs1; unsigned sh; uint32_t want; } srait[] = {
        { 28, 7, 63, 0x43f3de13u },
        { 0, 1, 0, 0x4000d013u },
        { 31, 30, 31, 0x41ff5f93u },
    };
    for (size_t i = 0; i < sizeof(srait) / sizeof(srait[0]); i++) {
        uint8_t buf[8];
        em_t e;
        em_init(&e, buf, sizeof(buf));
        em_srai(&e, srait[i].rd, srait[i].rs1, srait[i].sh);
        uint32_t got = word_at(buf);
        CHECK(got == srait[i].want, "srai x%d,x%d,%u: got 0x%08x want 0x%08x",
              srait[i].rd, srait[i].rs1, srait[i].sh, got, srait[i].want);
        em_free(&e);
    }
    /* em_and: (rs2<<20)|(rs1<<15)|(111<<12)|(rd<<7)|0x33 */
    struct { int rd, rs1, rs2; uint32_t want; } andt[] = {
        { 7, 7, 28, 0x01c3f3b3u },
        { 0, 0, 1, 0x00107033u },
        { 31, 30, 29, 0x01df7fb3u },
    };
    for (size_t i = 0; i < sizeof(andt) / sizeof(andt[0]); i++) {
        uint8_t buf[8];
        em_t e;
        em_init(&e, buf, sizeof(buf));
        em_and(&e, andt[i].rd, andt[i].rs1, andt[i].rs2);
        uint32_t got = word_at(buf);
        CHECK(got == andt[i].want, "and x%d,x%d,x%d: got 0x%08x want 0x%08x",
              andt[i].rd, andt[i].rs1, andt[i].rs2, got, andt[i].want);
        em_free(&e);
    }
    /* em_blt / em_bge: B-type skeleton, funct3 100/101, opcode 0x63.
     * Bind the label before emitting for a true zero offset. */
    {
        uint8_t buf[16];
        em_t e;
        em_init(&e, buf, sizeof(buf));
        int lab = em_label(&e);
        em_bind(&e, lab);
        em_blt(&e, 0, 0, lab);
        em_resolve(&e);
        CHECK(word_at(buf) == 0x00004063u, "blt x0,x0: got 0x%08x",
              word_at(buf));
        em_free(&e);
    }
    {
        uint8_t buf[16];
        em_t e;
        em_init(&e, buf, sizeof(buf));
        int lab = em_label(&e);
        em_bind(&e, lab);
        em_bge(&e, 5, 6, lab);
        em_resolve(&e);
        /* (6<<20)|(5<<15)|(101<<12)|0x63 */
        CHECK(word_at(buf) == 0x0062d063u, "bge x5,x6: got 0x%08x",
              word_at(buf));
        em_free(&e);
    }
}

/* ---------------- B. field decode (independent re-derivation) -------- */

static void test_field_decode(void)
{
    for (int iter = 0; iter < 200; iter++) {
        int vd = (int)(xrng() & 31), vs = (int)(xrng() & 31);
        int rs = (int)(xrng() & 31);
        uint8_t buf[8];
        em_t e;
        uint32_t w;

        em_init(&e, buf, sizeof(buf));
        em_vle8_v_m(&e, vd, rs);
        w = word_at(buf);
        CHECK(((w >> 29) & 7u) == 0, "vle8.m nf!=0");
        CHECK(((w >> 25) & 1u) == 0, "vle8.m vm!=0");
        CHECK(((w >> 15) & 31u) == (unsigned)rs, "vle8.m rs1");
        CHECK(((w >> 12) & 7u) == 0, "vle8.m width!=0");
        CHECK(((w >> 7) & 31u) == (unsigned)vd, "vle8.m vd");
        CHECK((w & 0x7fu) == 0x07u, "vle8.m opcode");
        em_free(&e);

        em_init(&e, buf, sizeof(buf));
        em_vse8_v_m(&e, vs, rs);
        w = word_at(buf);
        CHECK(((w >> 25) & 1u) == 0, "vse8.m vm!=0");
        CHECK(((w >> 15) & 31u) == (unsigned)rs, "vse8.m rs1");
        CHECK(((w >> 7) & 31u) == (unsigned)vs, "vse8.m vs3");
        CHECK((w & 0x7fu) == 0x27u, "vse8.m opcode");
        em_free(&e);

        em_init(&e, buf, sizeof(buf));
        em_vid_v(&e, vd);
        w = word_at(buf);
        CHECK(((w >> 26) & 0x3fu) == 0x14u, "vid.v funct6!=010100");
        CHECK(((w >> 25) & 1u) == 1, "vid.v vm!=1");
        CHECK(((w >> 20) & 31u) == 0, "vid.v vs2 field != 0");
        CHECK(((w >> 15) & 31u) == 0x11u, "vid.v vs1!=10001");
        CHECK(((w >> 12) & 7u) == 0x2u, "vid.v funct3!=OPMVV");
        CHECK(((w >> 7) & 31u) == (unsigned)vd, "vid.v vd");
        CHECK((w & 0x7fu) == 0x57u, "vid.v opcode");
        em_free(&e);

        em_init(&e, buf, sizeof(buf));
        em_vmsltu_vx(&e, 0, vs, rs);
        w = word_at(buf);
        CHECK(((w >> 26) & 0x3fu) == 0x1au, "vmsltu.vx funct6!=011010");
        CHECK(((w >> 25) & 1u) == 1, "vmsltu.vx vm!=1");
        CHECK(((w >> 20) & 31u) == (unsigned)vs, "vmsltu.vx vs2");
        CHECK(((w >> 15) & 31u) == (unsigned)rs, "vmsltu.vx rs1");
        CHECK(((w >> 12) & 7u) == 0x4u, "vmsltu.vx funct3!=OPIVX");
        CHECK(((w >> 7) & 31u) == 0, "vmsltu.vx vd!=v0");
        CHECK((w & 0x7fu) == 0x57u, "vmsltu.vx opcode");
        em_free(&e);

        /* srai/and: randomized field decode of the min-clamp emitters. */
        {
            unsigned sh = (unsigned)(xrng() & 63);
            em_init(&e, buf, sizeof(buf));
            em_srai(&e, vd, rs, sh);
            w = word_at(buf);
            CHECK(((w >> 26) & 0x3fu) == 0x10u, "srai imm[11:6]!=010000");
            CHECK(((w >> 20) & 63u) == sh, "srai shamt");
            CHECK(((w >> 15) & 31u) == (unsigned)rs, "srai rs1");
            CHECK(((w >> 12) & 7u) == 0x5u, "srai funct3!=101");
            CHECK(((w >> 7) & 31u) == (unsigned)vd, "srai rd");
            CHECK((w & 0x7fu) == 0x13u, "srai opcode");
            em_free(&e);

            em_init(&e, buf, sizeof(buf));
            em_and(&e, vd, rs, vs);
            w = word_at(buf);
            CHECK(((w >> 25) & 0x7fu) == 0x00u, "and funct7!=0000000");
            CHECK(((w >> 20) & 31u) == (unsigned)vs, "and rs2");
            CHECK(((w >> 15) & 31u) == (unsigned)rs, "and rs1");
            CHECK(((w >> 12) & 7u) == 0x7u, "and funct3!=111");
            CHECK(((w >> 7) & 31u) == (unsigned)vd, "and rd");
            CHECK((w & 0x7fu) == 0x33u, "and opcode");
            em_free(&e);
        }

        em_init(&e, buf, sizeof(buf));
        int lab = em_label(&e);
        em_blt(&e, vd, rs, lab);
        em_bind(&e, lab);
        em_resolve(&e);
        w = word_at(buf);
        CHECK(((w >> 12) & 7u) == 0x4u, "blt funct3!=100");
        CHECK((w & 0x7fu) == 0x63u, "blt opcode");
        CHECK(((w >> 15) & 31u) == (unsigned)vd, "blt rs1");
        CHECK(((w >> 20) & 31u) == (unsigned)rs, "blt rs2");
        em_free(&e);

        em_init(&e, buf, sizeof(buf));
        lab = em_label(&e);
        em_bge(&e, vd, rs, lab);
        em_bind(&e, lab);
        em_resolve(&e);
        w = word_at(buf);
        CHECK(((w >> 12) & 7u) == 0x5u, "bge funct3!=101");
        CHECK((w & 0x7fu) == 0x63u, "bge opcode");
        em_free(&e);
    }
}

/* ---------------- C. masked == unmasked ^ bit25 -------------------- */

static void test_masked_diff(void)
{
    for (int iter = 0; iter < 200; iter++) {
        int vd = (int)(xrng() & 31), rs = (int)(xrng() & 31);
        uint8_t b1[8], b2[8];
        em_t e1, e2;
        em_init(&e1, b1, sizeof(b1));
        em_init(&e2, b2, sizeof(b2));
        em_vle8(&e1, vd, rs);
        em_vle8_v_m(&e2, vd, rs);
        CHECK((word_at(b1) ^ word_at(b2)) == (1u << 25),
              "vle8.v.m differs from vle8.v outside bit 25");
        em_free(&e1);
        em_free(&e2);
        em_init(&e1, b1, sizeof(b1));
        em_init(&e2, b2, sizeof(b2));
        em_vse8(&e1, vd, rs);
        em_vse8_v_m(&e2, vd, rs);
        CHECK((word_at(b1) ^ word_at(b2)) == (1u << 25),
              "vse8.v.m differs from vse8.v outside bit 25");
        em_free(&e1);
        em_free(&e2);
    }
}

/* ---------------- D. vd!=v0 death check ---------------------------- */

static void test_vmsltu_vd_assert(void)
{
    /* The vd==v0 assert must fire: run the violating call in a child
     * so the parent survives to report. A missing assert (child exits
     * 0) fails the test. */
    pid_t pid = fork();
    if (pid < 0) {
        printf("FAIL fork failed\n");
        failures++;
        return;
    }
    if (pid == 0) {
        uint8_t buf[8];
        em_t e;
        em_init(&e, buf, sizeof(buf));
        em_vmsltu_vx(&e, 5, 1, 2); /* vd != v0: must abort */
        _exit(0); /* reached: assert missing */
    }
    int st = 0;
    if (waitpid(pid, &st, 0) < 0) {
        printf("FAIL waitpid failed\n");
        failures++;
        return;
    }
    CHECK(!(WIFEXITED(st) && WEXITSTATUS(st) == 0),
          "em_vmsltu_vx accepted vd!=v0 (assert missing)");
}

int main(void)
{
    test_literal_pins();
    test_field_decode();
    test_masked_diff();
    test_vmsltu_vd_assert();
    if (failures == 0)
        printf("PASS test_riscv_maskedtail_emit\n");
    return failures != 0;
}
