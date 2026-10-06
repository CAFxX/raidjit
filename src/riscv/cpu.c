/* RISC-V CPU detection for raidjit.
 *
 * Vector-extension presence is probed in two ways:
 *   1. the riscv_hwprobe syscall (NR 258) with
 *      RISCV_HWPROBE_KEY_IMA_EXT_0, checking the V bit;
 *   2. if the syscall is unavailable, by parsing the "isa" line of
 *      /proc/cpuinfo for the 'v' extension letter or any "zve*"
 *      sub-extension (the JIT only needs the integer vector subset
 *      present in Zve32x and up).
 *
 * Multi-letter extensions (Zicbop, Zvbc, Zvbc32e, Zvkg, Zicond) have
 * no riscv_hwprobe key, so they are detected by parsing the "isa" line
 * of /proc/cpuinfo for their tokens (has_ext_cpuinfo). QEMU user-mode
 * synthesizes the guest's /proc/cpuinfo "isa" line from its -cpu
 * properties, so the tokens there reflect the -cpu flags, not real
 * hardware; tests use synthetic rj_cpu_info values for those paths.
 *
 * VLEN cannot be queried portably from C here (the VLENB CSR needs
 * vector instructions, and hwprobe exposes no VLEN key on the kernels
 * we target), so rvv_vlen falls back to 16 (128-bit, the RVV 1.0
 * minimum and the qemu default). This is documented rather than
 * precise: the code generator is fully VL-agnostic (vsetvli
 * strip-mining), so the exact VLEN never affects correctness.
 */
#include "cpu.h"
#include <string.h>
#include <stdio.h>

#if defined(__riscv)

/* riscv_hwprobe: issued via a raw ecall so this file needs no libc
 * syscall() wrapper or feature-test macros. */
#define NR_RISCV_HWPROBE 258L
#define RISCV_HWPROBE_KEY_IMA_EXT_0 4L
/* per Linux arch/riscv/include/uapi/asm/hwprobe.h:
 * RISCV_HWPROBE_IMA_V (1 << 2) */
#define RISCV_HWPROBE_IMA_V (1ULL << 2)

struct rv_hwprobe {
    long long key;
    unsigned long long value;
};

static long raw_hwprobe(struct rv_hwprobe *pairs, long count)
{
    register long a0 __asm__("a0") = (long)pairs;
    register long a1 __asm__("a1") = count;
    register long a2 __asm__("a2") = 0;
    register long a3 __asm__("a3") = 0;
    register long a4 __asm__("a4") = 0;
    register long a7 __asm__("a7") = NR_RISCV_HWPROBE;
    __asm__ volatile("ecall"
                     : "+r"(a0)
                     : "r"(a1), "r"(a2), "r"(a3), "r"(a4), "r"(a7)
                     : "memory");
    return a0;
}

/* 1 if the V extension is present via hwprobe, 0 if present-and-absent,
 * -1 if the probe itself is unavailable. */
static int has_v_hwprobe(void)
{
    struct rv_hwprobe p;
    p.key = RISCV_HWPROBE_KEY_IMA_EXT_0;
    p.value = 0;
    long r = raw_hwprobe(&p, 1);
    if (r != 0)
        return -1;
    return (p.value & RISCV_HWPROBE_IMA_V) ? 1 : 0;
}

/* Fallback: scan /proc/cpuinfo "isa:" lines for the 'v' extension
 * letter (single-letter run before the first '_') or a "zve"
 * sub-extension. Returns 1/0. */
static int has_v_cpuinfo(void)
{
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (!f)
        return 0;
    char line[1024];
    int found = 0;
    while (fgets(line, sizeof(line), f)) {
        /* Match the "isa" field exactly: a bare prefix match would also
         * accept a hypothetical "isafoo:" line. */
        if (strncmp(line, "isa", 3) != 0 ||
            (line[3] != ':' && line[3] != '\t' && line[3] != ' '))
            continue;
        char *c = strchr(line, ':');
        if (!c)
            continue;
        c++;
        while (*c == ' ' || *c == '\t')
            c++;
        /* single-letter extensions: "rv64" prefix, then letters up to
         * '_' or end */
        if ((c[0] == 'r' && c[1] == 'v')) {
            c += 2;
            while (*c >= '0' && *c <= '9')
                c++;
            while ((*c >= 'a' && *c <= 'z') || *c == '_') {
                if (*c == 'v') {
                    found = 1;
                    break;
                }
                if (*c == '_')
                    break;
                c++;
            }
        }
        if (!found && strstr(line, "zve"))
            found = 1;
        if (found)
            break;
    }
    fclose(f);
    return found;
}

static void cpuinfo_brand(char *brand, size_t n)
{
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (!f)
        return;
    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        const char *key = NULL;
        if (!strncmp(line, "model name", 10))
            key = "model name";
        else if (!strncmp(line, "uarch", 5))
            key = "uarch";
        else
            continue;
        (void)key;
        char *c = strchr(line, ':');
        if (!c)
            continue;
        c++;
        while (*c == ' ' || *c == '\t')
            c++;
        size_t L = strlen(c);
        while (L && (c[L - 1] == '\n' || c[L - 1] == ' ' || c[L - 1] == '\t'))
            c[--L] = 0;
        if (L) {
            snprintf(brand, n, "%s", c);
            break;
        }
    }
    fclose(f);
}

/* Multi-letter extension probe: no riscv_hwprobe key exposes Zicbop /
 * Zvbc / Zvbc32e / Zvkg / Zicond (hwprobe keys cover the single-letter
 * IMA extensions plus a few specific keys; none for these), so parse
 * the "isa" line of /proc/cpuinfo for the extension token (the kernel
 * prints the full extension list there on real hardware; QEMU
 * user-mode synthesizes a RISC-V cpuinfo from its -cpu properties).
 * NOTE: QEMU 8.2.2's -cpu max advertises Zvbc/Zvkg in this string but
 * cannot execute vclmul.vv/vclmulh.vv (illegal instruction; verified
 * SIGILL, exit 132); it does not know the Zvbc32e extension at all (no
 * zvbc32e token, and -cpu ...,zvbc32e=false is rejected as unknown),
 * so rvv_zvbc32e reads 0 under QEMU 8.2.2 regardless. The QEMU test
 * recipes pass zvbc=false,zvkg=false (see Makefile QEMU_RVV_CPU).
 * Returns 1/0. */
static int has_ext_cpuinfo(const char *ext)
{
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (!f)
        return 0;
    char line[1024];
    int found = 0;
    while (fgets(line, sizeof(line), f)) {
        /* Match the "isa" field exactly, same as has_v_cpuinfo. */
        if (strncmp(line, "isa", 3) != 0 ||
            (line[3] != ':' && line[3] != '\t' && line[3] != ' '))
            continue;
        char *c = strchr(line, ':');
        if (!c)
            continue;
        c++;
        /* tokens: "rv64imafdcv", then "_"-separated extensions */
        for (char *tok = strtok(c, "_ \t\n"); tok;
             tok = strtok(NULL, "_ \t\n")) {
            if (strcmp(tok, ext) == 0) {
                found = 1;
                break;
            }
        }
        if (found)
            break;
    }
    fclose(f);
    return found;
}

void rj_cpu_detect(rj_cpu_info *out)
{
    memset(out, 0, sizeof(*out));

    int v = has_v_hwprobe();
    if (v < 0)
        v = has_v_cpuinfo();
    out->rvv = v ? 1 : 0;
    /* VLEN fallback, documented in the file header: the JIT is
     * VL-agnostic, so this never affects codegen. */
    out->rvv_vlen = v ? 16 : 0;
    out->rvv_zicbop = has_ext_cpuinfo("zicbop") ? 1 : 0;
    /* vclmul.vv and vclmulh.vv are both Zvbc (Vector Crypto §36.2/36.3;
     * Zvkg defines vghsh/vgmul, not vclmulh -- the old "Zvkg provides
     * the high half" attribution was wrong). Zicond (czero.eqz/nez) is
     * detected for future use; no backend path consumes it yet (see
     * Brvv DESIGN.md item B6). */
    out->rvv_zvbc = has_ext_cpuinfo("zvbc") ? 1 : 0;
    out->rvv_zvkg = has_ext_cpuinfo("zvkg") ? 1 : 0;
    /* Zvbc32e (vclmul.vv/vclmulh.vv for SEW=32/16/8) is independent of
     * Zvbc (SEW=64 only): probe its own token, token-exact. This is
     * THE gate for the SEW=8 MUL_ZVBC twin (see the comment above):
     * Zvbc-without-Zvbc32e must not enumerate it (reserved encoding),
     * Zvbc32e-standalone must. rvv_zvbc/rvv_zvkg stay informational. */
    out->rvv_zvbc32e = has_ext_cpuinfo("zvbc32e") ? 1 : 0;
    /* Zvbb (vector bitmanip: vandn.vv, vbrev8.v, ...; ratified 1.0).
     * Token-exact probe, like Zvbc32e. Detected for future use; no
     * backend path consumes it yet (I5: no profitable twin found). */
    out->rvv_zvbb = has_ext_cpuinfo("zvbb") ? 1 : 0;
    out->rvv_zicond = has_ext_cpuinfo("zicond") ? 1 : 0;

    cpuinfo_brand(out->brand, sizeof(out->brand));
    if (!out->brand[0])
        snprintf(out->brand, sizeof(out->brand), "riscv64");
}

#else

void rj_cpu_detect(rj_cpu_info *out)
{
    memset(out, 0, sizeof(*out));
    snprintf(out->brand, sizeof(out->brand), "non-riscv");
}

#endif
