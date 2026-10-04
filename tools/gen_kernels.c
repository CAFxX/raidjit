/* tools/gen_kernels.c — regenerate the prebuilt kernels shipped in
 * ~/workspace/your_files/raidjit/kernels/.
 *
 * Usage:
 *   gen_kernels <target> <outdir> [--redgreen]
 *     target: avx512-gfni | sve2 | rvv
 *
 * For each of raid5(4) and raid6(5) encode it:
 *   1. hand-fills an rj_cpu_info for the synthetic target ISA (never
 *      rj_cpu_detect: the target ISA differs from the host for the
 *      cross-compiled runs),
 *   2. builds every candidate, picks the winner with heuristic_select
 *      (benchmark=0),
 *   3. correctness-verifies the winner in-process against the scalar
 *      reference (src/ref.c) at several lengths x several seeds,
 *   4. asserts an ISA marker is present in the emitted code bytes
 *      (fails the build if absent),
 *   5. dumps <name>.code.bin / <name>.const.bin (exact bytes from the
 *      built version: code at mem[0,code_size), const pool at
 *      mem[code_size,code_size+const_size) — the pool references are
 *      PC-relative in every backend, so the pair is one relocatable
 *      image) and a <name>.json manifest fragment,
 *   6. re-mmaps the dumped files as a fresh executable image and
 *      re-runs the verification against THOSE bytes, proving the
 *      shipped bytes are the tested bytes.
 *
 * --redgreen (run with outdir in /tmp, never shipped): after building,
 *   flips one const-pool byte and shows the in-process check FAILS, and
 *   corrupts the expected output and shows the comparator FAILS —
 *   proving the verification is not vacuous. Exits 0 only if both
 *   negative checks fail as expected.
 *
 * Selection justification (recorded in manifest.json): cross-ISA
 * empirical benchmarking under QEMU is cycle-meaningless, so no timing
 * is used. heuristic_select ranks all built candidates: highest ISA
 * wins; among equal ISA, min roofline max(uops/3.5, mem/64); ties keep
 * the earliest candidate (benchmark=0). This is heuristic selection,
 * not first-match.
 */
#include "../include/raidjit.h"
#include "../src/plan.h"
#include "../src/jit.h"
#include "../src/cpu.h"
#include "../src/ref.h"
#include "../src/bench.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdint.h>
#include <sys/mman.h>
#include <unistd.h>

/* ---------------- utilities ---------------- */

static void die(const char *fmt, ...)
{
    va_list ap;
    fprintf(stderr, "gen_kernels: FATAL: ");
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fprintf(stderr, "\n");
    exit(1);
}

static uint64_t xrng_state = 0x9E3779B97F4A7C15ULL;
static uint64_t xrng(void)
{
    uint64_t x = xrng_state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    xrng_state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

/* JSON-escape s into out (outsz incl. NUL). */
static void json_escape(char *out, size_t outsz, const char *s)
{
    size_t j = 0;
    for (size_t i = 0; s[i] && j + 2 < outsz; i++) {
        if (s[i] == '"' || s[i] == '\\') {
            out[j++] = '\\';
            out[j++] = s[i];
        } else if ((unsigned char)s[i] < 0x20) {
            if (j + 7 >= outsz)
                break;
            j += (size_t)snprintf(out + j, outsz - j, "\\u%04x", s[i]);
        } else {
            out[j++] = s[i];
        }
    }
    out[j] = '\0';
}

/* ---------------- synthetic CPUs ---------------- */

static void fill_cpu(const char *target, rj_cpu_info *cpu)
{
    memset(cpu, 0, sizeof(*cpu));
    if (!strcmp(target, "avx512-gfni")) {
        cpu->sse2 = cpu->ssse3 = 1;
        cpu->avx = cpu->avx2 = 1;
        cpu->avx512f = cpu->avx512bw = cpu->avx512vl = 1;
        cpu->gfni = 1;
        snprintf(cpu->brand, sizeof(cpu->brand), "synthetic-avx512-gfni");
    } else if (!strcmp(target, "sve2")) {
        cpu->neon = 1;
        cpu->sve = 1;
        cpu->sve2 = 1;
        /* Bytes. Matches qemu-aarch64 -cpu max,sve-default-vector-length=128
         * (128 bits = 16 bytes), which is how this tool is executed. SVE
         * codegen is VL-agnostic; this only feeds U selection. */
        cpu->sve_vl = 16;
        snprintf(cpu->brand, sizeof(cpu->brand), "synthetic-sve2");
    } else if (!strcmp(target, "rvv")) {
        cpu->rvv = 1;
        /* Bytes. Matches qemu-riscv64 -cpu max default vlen=128 bits. */
        cpu->rvv_vlen = 16;
        snprintf(cpu->brand, sizeof(cpu->brand), "synthetic-rvv");
    } else {
        die("unknown target '%s' (want avx512-gfni|sve2|rvv)", target);
    }
}

/* ---------------- verification ---------------- */

typedef struct {
    const plan_t *pl;
    rj_fn fn;
} kernel_t;

/* Run one (len, seed) case. Returns 0 if fn output matches ref_run. */
static int verify_case(const kernel_t *k, size_t len, uint64_t seed)
{
    int n_in = k->pl->n_in, n_out = k->pl->n_out;
    const void *in[256];
    void *out[256];
    uint8_t *inb[256], *outb[256], *wantb[256];
    size_t alloc = len ? len : 64;

    xrng_state = seed;
    for (int i = 0; i < n_in; i++) {
        if (posix_memalign((void **)&inb[i], 64, alloc))
            die("oom");
        for (size_t j = 0; j < len; j++)
            inb[i][j] = (uint8_t)xrng();
        in[i] = inb[i];
    }
    for (int j = 0; j < n_out; j++) {
        if (posix_memalign((void **)&outb[j], 64, alloc))
            die("oom");
        if (posix_memalign((void **)&wantb[j], 64, alloc))
            die("oom");
        /* Sentinel: kernel must overwrite exactly len bytes; ref must too. */
        memset(outb[j], 0xA5, alloc);
        memset(wantb[j], 0xA5, alloc);
        out[j] = outb[j];
    }

    ref_run(k->pl, (const uint8_t **)in, wantb, len);
    k->fn(in, out, len);

    int ok = 1;
    /* Compare the full allocation, not just len: at len=0 the whole
     * buffer is sentinel, so this also catches a kernel that writes
     * when it must write nothing. */
    for (int j = 0; j < n_out && ok; j++)
        if (memcmp(outb[j], wantb[j], alloc))
            ok = 0;

    for (int i = 0; i < n_in; i++)
        free(inb[i]);
    for (int j = 0; j < n_out; j++) {
        free(outb[j]);
        free(wantb[j]);
    }
    return ok ? 0 : -1;
}

static const size_t kLens[] = {0, 1, 7, 1003, 65537, 1u << 20};
static const uint64_t kSeeds[] = {0x12345678ULL, 0xDEADBEEFULL, 0x0BADF00DULL};

/* Full matrix: every length x every seed. Returns 0 on success. */
static int verify_kernel(const kernel_t *k, long *runs_out)
{
    long runs = 0;
    for (size_t li = 0; li < sizeof(kLens) / sizeof(kLens[0]); li++) {
        for (size_t si = 0; si < sizeof(kSeeds) / sizeof(kSeeds[0]); si++) {
            if (verify_case(k, kLens[li], kSeeds[si] ^ (uint64_t)kLens[li]))
                return -1;
            runs++;
        }
    }
    if (runs_out)
        *runs_out = runs;
    return 0;
}

/* ---------------- ISA markers ----------------
 * Fails the build if the expected instruction class is absent from the
 * emitted code bytes. (w = 32-bit little-endian words over code.) */

static int count_eor3(const uint8_t *code, size_t code_size)
{
    /* EOR3 Zdn.D,Zdn.D,Zm.D,Za.D = 0x04203800 | Zm<<16 | Za<<5 | Zdn. */
    int n = 0;
    for (size_t i = 0; i + 4 <= code_size; i += 4) {
        uint32_t w;
        memcpy(&w, code + i, 4);
        if ((w & 0xFFE0FC00u) == 0x04203800u)
            n++;
    }
    return n;
}

static int count_sve_major(const uint8_t *code, size_t code_size)
{
    /* Counts SVE data-processing encodings (bits[31:24] in 0x04..0x07);
     * catches eor/eor3/cntb. Other SVE encodings exist (predicate ops at
     * 0x25, loads/stores at 0xA4..0xA7) but are not counted: 16+ hits on
     * the data-processing range already proves the SVE codegen path, and
     * NEON (0x0E/0x2E/0x4E/0x6E) cannot false-positive here. */
    int n = 0;
    for (size_t i = 0; i + 4 <= code_size; i += 4) {
        uint32_t w;
        memcpy(&w, code + i, 4);
        unsigned top = w >> 24;
        if (top >= 0x04 && top <= 0x07)
            n++;
    }
    return n;
}

static int count_evex(const uint8_t *code, size_t code_size,
                      int map /* -1 any */, int opcode /* -1 any */)
{
    /* EVEX: 0x62 P0 P1 P2 opcode ...; per the Intel encoding the map is
     * P0[1:0] (1=0F, 2=0F38, 3=0F3A) and this emitter hardwires P0[3:2]=0
     * (src/x86_64/emit.c), so masking P0 with 0x07 is equivalent here —
     * verified byte-identical against GCC/GAS for vpxord (62 91 7d 48 ef)
     * and vgf2p8affineqb (62 a3 bd 48 ce). */
    int n = 0;
    for (size_t i = 0; i + 5 <= code_size; i++) {
        if (code[i] != 0x62)
            continue;
        int m = code[i + 1] & 0x07;
        if (map >= 0 && m != map)
            continue;
        if (opcode >= 0 && code[i + 4] != (uint8_t)opcode)
            continue;
        n++;
    }
    return n;
}

static int count_rvv(const uint8_t *code, size_t code_size)
{
    /* RVV vector major opcode 1010111 = 0x57. */
    int n = 0;
    for (size_t i = 0; i + 4 <= code_size; i += 4) {
        uint32_t w;
        memcpy(&w, code + i, 4);
        if ((w & 0x7Fu) == 0x57u)
            n++;
    }
    return n;
}

static int count_kmovq(const uint8_t *code, size_t code_size)
{
    /* KMOVQ k,m64 (tail-mask load, track B1x86): VEX.0F.W1 90 /r, i.e.
     * 3-byte VEX 0xC4, two payload bytes, opcode 0x90. See
     * em_kmovq_km (src/x86_64/emit.c). */
    int n = 0;
    for (size_t i = 0; i + 3 < code_size; i++) {
        if (code[i] == 0xC4 && code[i + 3] == 0x90)
            n++;
    }
    return n;
}

static int count_whilelt(const uint8_t *code, size_t code_size)
{
    /* WHILELT pd.b,xRn,xRm (predicated-tail predicate, track B1sve):
     * 0x25201400 | rm<<16 | rn<<5 | pd. Mask out the register fields;
     * cf. em_sve_whilelt (src/aarch64/emit.c) and the CHECK_WORD pins in
     * tests/test_aarch64_emit.c. */
    int n = 0;
    for (size_t i = 0; i + 4 <= code_size; i += 4) {
        uint32_t w;
        memcpy(&w, code + i, 4);
        if ((w & 0xFFE0FC10u) == 0x25201400u)
            n++;
    }
    return n;
}

/* Returns marker description (static). Dies if the marker is absent. */
static const char *check_marker(const char *target, const char *kname,
                                const uint8_t *code, size_t code_size,
                                const char *desc, int *count_out)
{
    int n = 0;
    const char *m = NULL;
    if (strstr(kname, "_maskedtail")) {
        /* Item-14 tail twin: the marker is the tail-mask load. The body
         * is the winner's (already ISA-marked); the twin exists for the
         * tail, so the tail instruction class is the assertion. */
        n = count_kmovq(code, code_size);
        m = "KMOVQ (tail-mask load into k1) present";
        if (!strstr(desc, "maskedtail"))
            die("%s: expected maskedtail twin, desc='%s'", kname, desc);
    } else if (strstr(kname, "_predtail")) {
        n = count_whilelt(code, code_size);
        m = "WHILELT (predicated-tail predicate) present";
        if (!strstr(desc, "predtail"))
            die("%s: expected predtail twin, desc='%s'", kname, desc);
    } else if (!strcmp(target, "sve2")) {
        if (strstr(kname, "raid6")) {
            /* P_HORNER has GF multiplies: SVE2 EOR3 must be present.
             * This is the proof the "sve2" label is accurate. */
            n = count_eor3(code, code_size);
            m = "EOR3 (SVE2 three-way xor) present";
            if (!strstr(desc, "+eor3"))
                die("%s: code has EOR3 but desc lacks +eor3 tag", kname);
        } else {
            /* P_XOR is pure XOR: no GF multiply exists for EOR3 to fold,
             * so EOR3 can never appear here (see docs/sve2_audit.md). The
             * marker is the SVE codegen path itself (desc "sve ...", SVE
             * major-opcode instructions in the bytes). */
            n = count_sve_major(code, code_size);
            m = "SVE codegen path (P_XOR: no GF multiply, EOR3 n/a by design)";
            /* Desc now carries the pl->name prefix on all backends
             * (aarch64 desc fix, Phase C); match " sve " anywhere. */
            if (!strstr(desc, " sve "))
                die("%s: expected SVE codegen, desc='%s'", kname, desc);
        }
    } else if (!strcmp(target, "avx512-gfni")) {
        if (strstr(kname, "raid6")) {
            /* VGF2P8AFFINEQB: EVEX.66.0F3A.W1 CE (map=3 in this emitter's
             * P0[2:0] convention; byte-identical to GCC/GAS). */
            n = count_evex(code, code_size, 3, 0xCE);
            m = "VGF2P8AFFINEQB (EVEX.66.0F3A.W1 CE) present";
            if (!strstr(desc, "gfni"))
                die("%s: code has VGF2P8AFFINEQB but desc lacks gfni",
                    kname);
        } else {
            /* P_XOR on AVX-512: VPXORD = EVEX.66.0F.W0 EF (map=1). */
            n = count_evex(code, code_size, 1, 0xEF);
            m = "VPXORD (EVEX.66.0F.W0 EF) present";
            if (!strstr(desc, "avx512"))
                die("%s: expected avx512 codegen, desc='%s'", kname, desc);
        }
    } else { /* rvv */
        n = count_rvv(code, code_size);
        m = "RVV vector opcodes (major 0x57) present";
        if (!strstr(desc, "rvv"))
            die("%s: expected rvv codegen, desc='%s'", kname, desc);
    }
    if (n <= 0)
        die("%s: ISA marker absent: %s", kname, m);
    if (count_out)
        *count_out = n;
    return m;
}

/* Forward: strategy_name is defined in the build section below, but
 * emit_kernel (manifest fragment) needs it. */
static const char *strategy_name(plan_kind_t k);

/* Emit one kernel: verify the live image against the scalar reference,
 * check the ISA (or tail-twin) marker on the exact emitted bytes, dump the
 * bytes, re-verify the shipped bytes from a fresh executable mapping, and
 * write the manifest fragment. Used for the winning version and for the
 * item-14 tail twins. */
static void emit_kernel(const plan_t *pl, version_t *v, const char *kname,
                        const char *target, const char *outdir,
                        const char *sname, int ndata, int redgreen)
{
const uint8_t *code = (const uint8_t *)v->mem;
const uint8_t *cnst = code + v->code_size;

/* 1. verify the live image against the scalar reference */
kernel_t k = {pl, v->fn};
long runs = 0;
if (verify_kernel(&k, &runs))
    die("%s: correctness verification FAILED", kname);

/* 2. ISA marker on the exact emitted bytes */
int mcount = 0;
const char *marker =
    check_marker(target, kname, code, v->code_size, v->desc, &mcount);

if (redgreen) {
    /* --- red-green proof (never shipped) ---
     * (a) flip one const-pool byte -> the check must FAIL. */
    int rg_ok = 1;
    if (v->const_size > 0) {
        long ps = sysconf(_SC_PAGESIZE);
        size_t msz = (v->code_size + v->const_size +
                      (size_t)ps - 1) &
                     ~((size_t)ps - 1);
        uint8_t *mem = (uint8_t *)v->mem;
        /* Keep PROT_EXEC: we execute the kernel below to observe
         * the corruption. */
        if (mprotect(mem, msz, PROT_READ | PROT_WRITE | PROT_EXEC))
            die("%s: redgreen mprotect failed", kname);
        /* Flip one const-pool byte. For the maskedtail twin the pool
         * is the tail-mask table whose entry 0 is never read by
         * construction (chunk length rdx in 1..64), so flip the first
         * byte of the last entry (mask[64], exercised by every
         * 64-byte tail chunk) instead of pool[0]. */
        size_t flip_off = v->code_size;
        if (strstr(kname, "_maskedtail"))
            flip_off = v->code_size + v->const_size - 8;
        mem[flip_off] ^= 0xFF;
        if (verify_case(&k, 1003, 0xC0FFEEULL) == 0) {
            printf("REDGREEN-FAIL %s: const corruption NOT detected\n",
                   kname);
            rg_ok = 0;
        }
        mem[flip_off] ^= 0xFF; /* restore */
        if (mprotect(mem, msz, PROT_READ | PROT_EXEC))
            die("%s: redgreen mprotect restore failed", kname);
        /* sanity: restored image verifies again */
        if (verify_case(&k, 1003, 0xC0FFEEULL) != 0) {
            printf("REDGREEN-FAIL %s: restore did not recover\n",
                   kname);
            rg_ok = 0;
        }
    } else {
        printf("redgreen %s: no const pool, skipping flip test\n",
               kname);
    }
    /* (b) corrupt the expected output -> comparator must FAIL.
     * Proves the memcmp is not vacuous (also covers const_size==0
     * kernels). */
    {
        int n_in = pl->n_in, n_out = pl->n_out;
        const void *in[256];
        void *out[256];
        uint8_t *inb[256], *outb[256], *wantb[256];
        size_t len = 1003;
        xrng_state = 0xB16B00B5ULL;
        for (int i = 0; i < n_in; i++) {
            if (posix_memalign((void **)&inb[i], 64, len))
                die("oom");
            for (size_t j = 0; j < len; j++)
                inb[i][j] = (uint8_t)xrng();
            in[i] = inb[i];
        }
        for (int j = 0; j < n_out; j++) {
            if (posix_memalign((void **)&outb[j], 64, len))
                die("oom");
            if (posix_memalign((void **)&wantb[j], 64, len))
                die("oom");
            out[j] = outb[j];
        }
        ref_run(pl, (const uint8_t **)in, wantb, len);
        v->fn(in, out, len);
        wantb[0][0] ^= 0xFF; /* corrupt expectation */
        int cmp = 0;
        for (int j = 0; j < n_out; j++)
            if (memcmp(outb[j], wantb[j], len))
                cmp = 1;
        if (!cmp) {
            printf("REDGREEN-FAIL %s: comparator vacuous\n", kname);
            rg_ok = 0;
        }
        for (int i = 0; i < n_in; i++)
            free(inb[i]);
        for (int j = 0; j < n_out; j++) {
            free(outb[j]);
            free(wantb[j]);
        }
    }
    if (!rg_ok)
        die("%s: red-green proof failed", kname);
    printf("redgreen %s: OK (corruption detected)\n", kname);
}

/* 3. dump exact bytes */
char code_path[512], const_path[512], frag_path[512];
snprintf(code_path, sizeof(code_path), "%s/%s.code.bin", outdir,
         kname);
snprintf(const_path, sizeof(const_path), "%s/%s.const.bin", outdir,
         kname);
snprintf(frag_path, sizeof(frag_path), "%s/%s.json", outdir, kname);
FILE *f = fopen(code_path, "wb");
if (!f)
    die("cannot write %s", code_path);
if (v->code_size && fwrite(code, 1, v->code_size, f) != v->code_size)
    die("short write %s", code_path);
fclose(f);
f = fopen(const_path, "wb");
if (!f)
    die("cannot write %s", const_path);
if (v->const_size &&
    fwrite(cnst, 1, v->const_size, f) != v->const_size)
    die("short write %s", const_path);
fclose(f);

/* 4. re-mmap the SHIPPED bytes as a fresh image, re-verify those */
{
    size_t total = v->code_size + v->const_size;
    long ps = sysconf(_SC_PAGESIZE);
    size_t msz = (total + (size_t)ps - 1) & ~((size_t)ps - 1);
    uint8_t *img =
        mmap(NULL, msz ? msz : (size_t)ps, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (img == MAP_FAILED)
        die("%s: reload mmap failed", kname);
    /* re-read from disk: the bytes under test are the files */
    FILE *fc = fopen(code_path, "rb");
    FILE *fp = fopen(const_path, "rb");
    if (!fc || !fp)
        die("%s: cannot re-read dumped bins", kname);
    if (v->code_size &&
        fread(img, 1, v->code_size, fc) != v->code_size)
        die("%s: short re-read code", kname);
    if (v->const_size &&
        fread(img + v->code_size, 1, v->const_size, fp) !=
            v->const_size)
        die("%s: short re-read const", kname);
    fclose(fc);
    fclose(fp);
    __builtin___clear_cache((char *)img, (char *)img + total);
    if (mprotect(img, msz ? msz : (size_t)ps, PROT_READ | PROT_EXEC))
        die("%s: reload mprotect failed", kname);
    kernel_t kr = {pl, (rj_fn)img};
    long rruns = 0;
    if (verify_kernel(&kr, &rruns))
        die("%s: SHIPPED-BYTES verification FAILED", kname);
    munmap(img, msz ? msz : (size_t)ps);
    printf("%s: shipped-bytes re-verified (%ld runs)\n", kname,
           rruns);
}

/* 5. manifest fragment (sha256 added by tools/manifest.py) */
{
    char edesc[256], emarker[256];
    json_escape(edesc, sizeof(edesc), v->desc);
    json_escape(emarker, sizeof(emarker), marker);
    int nominal_vl =
        !strcmp(target, "avx512-gfni")
            ? 64
            : 16; /* sve2/rvv nominal VL in bytes for selection */
    f = fopen(frag_path, "w");
    if (!f)
        die("cannot write %s", frag_path);
    fprintf(f,
            "{\n"
            "  \"name\": \"%s\",\n"
            "  \"target\": \"%s\",\n"
            "  \"topology\": \"%s(%d)\",\n"
            "  \"op\": \"encode\",\n"
            "  \"desc\": \"%s\",\n"
            "  \"strategy\": \"%s\",\n"
            "  \"unroll\": %d,\n"
            "  \"nominal_vl_bytes\": %d,\n"
            "  \"code_size\": %zu,\n"
            "  \"const_size\": %zu,\n"
            "  \"uops_per_byte\": %.6f,\n"
            "  \"mem_per_byte\": %.6f,\n"
            "  \"code_file\": \"%s.code.bin\",\n"
            "  \"const_file\": \"%s.const.bin\",\n"
            "  \"asm_file\": \"%s.asm\",\n"
            "  \"selection\": \"heuristic\",\n"
            "  \"marker\": \"%s\",\n"
            "  \"marker_count\": %d\n"
            "}\n",
            kname, target, sname, ndata, edesc,
            strategy_name(pl->kind), v->cand.U, nominal_vl,
            v->code_size, v->const_size, v->uops_per_byte,
            v->mem_per_byte, kname, kname, kname, emarker, mcount);
    fclose(f);
}

printf("%s: desc=\"%s\" code=%zu const=%zu verified=%ld runs "
       "marker=\"%s\" x%d\n",
       kname, v->desc, v->code_size, v->const_size, runs, marker,
       mcount);
}

/* ---------------- build ---------------- */

static const char *strategy_name(plan_kind_t k)
{
    return k == P_XOR ? "xor" : k == P_HORNER ? "horner" : k == P_DELTA
                                                            ? "delta"
                                                            : "mulacc";
}

typedef struct {
    const char *sname; /* "raid5-4" */
    rj_topology *(*mk)(int);
    int ndata;
} spec_t;

int main(int argc, char **argv)
{
    if (argc < 3 || argc > 4)
        die("usage: %s <avx512-gfni|sve2|rvv> <outdir> [--redgreen]",
            argv[0]);
    const char *target = argv[1];
    const char *outdir = argv[2];
    int redgreen = argc == 4 && !strcmp(argv[3], "--redgreen");
    if (argc == 4 && !redgreen)
        die("unknown flag '%s'", argv[3]);

    rj_cpu_info cpu;
    fill_cpu(target, &cpu);

    rj_options o;
    rj_options_default(&o);
    o.benchmark = 0; /* heuristic selection; see file header */
    o.nt_threshold = 0;
    o.prefetch_distance = 0;

    spec_t specs[] = {
        {"raid5-4", rj_topo_raid5, 4},
        {"raid6-5", rj_topo_raid6, 5},
    };

    for (size_t s = 0; s < sizeof(specs) / sizeof(specs[0]); s++) {
        char kname[96];
        snprintf(kname, sizeof(kname), "%s_%s", specs[s].sname, target);

        rj_topology *t = specs[s].mk(specs[s].ndata);
        if (!t)
            die("%s: topology constructor failed", kname);
        rj_op op = rj_op_encode();

        topo_int_t ti;
        char err[256];
        if (topo_int_build(&ti, t, err, sizeof(err)))
            die("%s: topo_int_build: %s", kname, err);
        plan_t pl;
        if (plan_build(&ti, &op, specs[s].sname, &pl, err, sizeof(err)))
            die("%s: plan_build: %s", kname, err);

        cand_t cands[512];
        int ncand =
            plan_candidates(&pl, &cpu, cands, 512, 0, 0);
        if (ncand <= 0)
            die("%s: no candidates for synthetic CPU", kname);

        version_t *vs = calloc((size_t)ncand, sizeof(*vs));
        if (!vs)
            die("oom");
        int nv = 0;
        for (int i = 0; i < ncand; i++) {
            char e2[256];
            if (jit_build_version(&pl, &cands[i], &cpu, &o, &vs[nv], e2,
                                  sizeof(e2)) == 0)
                nv++;
        }
        if (nv == 0)
            die("%s: no version built", kname);
        int w = heuristic_select(vs, nv, &pl);
        if (w < 0)
            die("%s: heuristic_select failed", kname);
        version_t *v = &vs[w];

        emit_kernel(&pl, v, kname, target, outdir,
                    specs[s].sname, specs[s].ndata, redgreen);

        /* Item-14 tail twins: built from the winning candidate exactly as
         * api.c does (plan_candidates never sets these flags). The twin
         * is a complete version whose scalar tail is replaced by the
         * masked (AVX-512) or predicated (SVE) vector tail. Eligibility
         * mirrors api.c; a twin build failure is fatal here -- silently
         * dropping it would leave the deliverable without the item-14
         * representative again. */
        {
            cand_t twc = vs[w].cand;
            const char *suffix = NULL;
            if (twc.isa == ISA_SVE) {
                twc.predtail = 1;
                suffix = "predtail";
            } else if (twc.isa == ISA_AVX512 && pl.kind == P_XOR &&
                       !twc.ternlog && !twc.xorsched && !twc.nt &&
                       cpu.avx512f && cpu.avx512bw) {
                twc.maskedtail = 1;
                suffix = "maskedtail";
            }
            if (suffix) {
                char twname[128], e3[256];
                snprintf(twname, sizeof(twname), "%s_%s", kname, suffix);
                version_t twin;
                memset(&twin, 0, sizeof(twin));
                if (jit_build_version(&pl, &twc, &cpu, &o, &twin, e3,
                                      sizeof(e3)))
                    die("%s: tail twin build failed: %s", twname, e3);
                emit_kernel(&pl, &twin, twname, target, outdir,
                            specs[s].sname, specs[s].ndata, redgreen);
                jit_free_version(&twin);
            }
        }

        for (int i = 0; i < nv; i++)
            jit_free_version(&vs[i]);
        free(vs);
        plan_free(&pl);
        topo_int_free(&ti);
        rj_op_free(&op);
        rj_topo_free(t);
    }
    return 0;
}
