#include "cpu.h"
#include <stdio.h>
#include <string.h>

#if defined(__x86_64__)

static void cpuid(unsigned leaf, unsigned sub,
                  unsigned *a, unsigned *b, unsigned *c, unsigned *d)
{
    __asm__ volatile("cpuid"
                     : "=a"(*a), "=b"(*b), "=c"(*c), "=d"(*d)
                     : "a"(leaf), "c"(sub));
}

static unsigned long long xgetbv(unsigned c)
{
    unsigned lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(c));
    return ((unsigned long long)hi << 32) | lo;
}

void rj_cpu_detect(rj_cpu_info *out)
{
    memset(out, 0, sizeof(*out));
    unsigned a, b, c, d;

    cpuid(0, 0, &a, &b, &c, &d);
    unsigned maxleaf = a;
    char vendor[13];
    memcpy(vendor + 0, &b, 4);
    memcpy(vendor + 4, &d, 4);
    memcpy(vendor + 8, &c, 4);
    vendor[12] = 0;

    /* Processor brand string (e.g. "AMD EPYC 9D25 64-Core Processor"), not
     * the vendor ID. Fall back to the vendor ID when unavailable. */
    cpuid(0x80000000, 0, &a, &b, &c, &d);
    if (a >= 0x80000004) {
        char bs[49];
        for (int i = 0; i < 3; i++) {
            cpuid(0x80000002u + (unsigned)i, 0, &a, &b, &c, &d);
            memcpy(bs + i * 16 + 0, &a, 4);
            memcpy(bs + i * 16 + 4, &b, 4);
            memcpy(bs + i * 16 + 8, &c, 4);
            memcpy(bs + i * 16 + 12, &d, 4);
        }
        bs[48] = 0;
        /* trim leading/trailing whitespace */
        char *s = bs;
        while (*s == ' ')
            s++;
        size_t L = strlen(s);
        while (L && s[L - 1] == ' ')
            s[--L] = 0;
        snprintf(out->brand, sizeof(out->brand), "%s", *s ? s : vendor);
    } else {
        snprintf(out->brand, sizeof(out->brand), "%s", vendor);
    }

    cpuid(1, 0, &a, &b, &c, &d);
    out->sse2  = (d >> 26) & 1;
    out->ssse3 = (c >> 9) & 1;
    out->pclmulqdq = (c >> 1) & 1;  /* CLMUL carryless multiply; CPUID.1:ECX[1].
                                    * No XCR0 state needed (integer/SSE state). */
    out->sse42 = (c >> 20) & 1;    /* SSE4.2 (crc32 insn); CPUID.1:ECX[20].
                                    * NOTE: the x86 crc32 instruction implements
                                    * CRC-32C (Castagnoli, 0x1EDC6F41), NOT the
                                    * IEEE CRC-32 (0x04C11DB7) this library
                                    * computes, so it is not usable for
                                    * rj_crc32_fast (see src/x86_64/crc_fast.c). */
    int osxsave = (c >> 27) & 1;
    int avx_hw  = (c >> 28) & 1;

    unsigned long long xcr0 = 0;
    if (osxsave)
        xcr0 = xgetbv(0);
    int sse_os  = (xcr0 & 0x6) == 0x6;   /* XCR0[2:1]: SSE + YMM state saved
                                         * (bit 2 also required: sse_os only
                                         * feeds the AVX/AVX-512 gates below) */
    int avx_os  = sse_os && (xcr0 & 0x4); /* XCR0[2]: YMM state */
    int avx512_os = avx_os && ((xcr0 & 0xE0) == 0xE0); /* XCR0[7:5]:
                                         * opmask, ZMM_Hi256, Hi16_ZMM */

    out->avx = avx_hw && avx_os;

    if (maxleaf >= 7) {
        cpuid(7, 0, &a, &b, &c, &d);
        out->avx2     = ((b >> 5) & 1) && avx_os;
        out->avx512f  = ((b >> 16) & 1) && avx512_os;
        out->avx512bw = ((b >> 30) & 1) && avx512_os;
        out->avx512vl = ((b >> 31) & 1) && avx512_os;
        /* VBMI (vpermi2b): leaf 7 ECX bit 1. Needs AVX512F HW + ZMM OS. */
        out->vbmi     = ((c >> 1) & 1) && out->avx512f && avx512_os;
        /* BITAlg (vpshufbitqmb): leaf 7 ECX bit 12. A separate flag from
         * VBMI: Cannon Lake has VBMI without BITAlg, so BITAlg
         * instructions must be gated on this bit, never on vbmi alone.
         * Same AVX512F HW + ZMM OS requirement as the other 512-bit
         * extensions. Track I1 evaluated VPSHUFBITQMB as a GF-multiply
         * twin and rejected it (it is a bit-gather, not a bit-matrix
         * multiply -- no XOR/parity -- per the SDM); the bit stays as
         * the correct gate for any future BITAlg use. */
        out->bitalg   = ((c >> 12) & 1) && out->avx512f && avx512_os;
        /* GFNI: this JIT only emits EVEX (512-bit) VGF2P8AFFINEQB, which
         * needs AVX512F HW and ZMM OS state in addition to the GFNI bit. */
        out->gfni     = ((c >> 8) & 1) && out->avx512f && avx512_os;
        out->cet_ss   = ((c >> 7) & 1); /* CET_SS: HW support; OS enablement
                                         * is separate (see rj_options) */
        /* VPCLMULQDQ: vector carryless multiply (4x64-bit); CPUID.7:ECX[10].
         * No XCR0 state needed for the bit itself (spec R3). Note the 256-bit
         * VPCLMULQDQ path in src/x86_64/crc_fast.c uses YMM registers; see
         * the dispatch notes there. */
        out->vpclmulqdq = ((c >> 10) & 1);
    }
}

#else

void rj_cpu_detect(rj_cpu_info *out)
{
    memset(out, 0, sizeof(*out));
    snprintf(out->brand, sizeof(out->brand), "non-x86_64");
}

#endif
