/* raidjit AArch64 CPU detection: HWCAP/HWCAP2 via getauxval, SVE vector
 * length via prctl(PR_SVE_GET_VL), brand from /proc/cpuinfo. */
#define _GNU_SOURCE
#include "cpu.h"
#include <stdio.h>
#include <string.h>
#include <sys/auxv.h>

/* HWCAP bits without <asm/hwcap.h> (same missing-sysroot reason as
 * above). Stable UAPI from asm/hwcap.h. */
#ifndef HWCAP_ASIMD
#define HWCAP_ASIMD (1UL << 1)
#endif
#ifndef HWCAP_SVE
#define HWCAP_SVE (1UL << 22)
#endif
#ifndef HWCAP2_SVE2
#define HWCAP2_SVE2 (1UL << 1)
#endif

/* PR_SVE_GET_VL without <sys/prctl.h>: that header's kernel-header chain
 * (linux/prctl.h -> asm/types.h) is not always present in cross sysroots.
 * The constants below are stable UAPI (linux/prctl.h, since 4.15). */
#ifndef PR_SVE_GET_VL
#define PR_SVE_GET_VL 51
#endif
#ifndef PR_SVE_VL_LEN_MASK
#define PR_SVE_VL_LEN_MASK 0xffff
#endif
int prctl(int __option, ...);

#ifndef PR_SVE_GET_VL
#define PR_SVE_GET_VL 51
#endif
#ifndef PR_SVE_VL_LEN_MASK
#define PR_SVE_VL_LEN_MASK 0xffffu
#endif

static void read_brand(char *out, size_t cap)
{
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (!f) return;
    char line[256];
    /* Prefer "model name"; fall back to "Model" / "Hardware" (ARM kernels). */
    char model[256] = {0}, hw[256] = {0};
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "model name", 10) == 0) {
            char *c = strchr(line, ':');
            if (c) {
                strncpy(model, c + 1, sizeof(model) - 1);
                break;
            }
        } else if (strncmp(line, "Model", 5) == 0 && !hw[0]) {
            char *c = strchr(line, ':');
            if (c) strncpy(hw, c + 1, sizeof(hw) - 1);
        } else if (strncmp(line, "Hardware", 8) == 0 && !hw[0]) {
            char *c = strchr(line, ':');
            if (c) strncpy(hw, c + 1, sizeof(hw) - 1);
        }
    }
    fclose(f);
    const char *src = model[0] ? model : hw;
    /* trim leading/trailing whitespace */
    while (*src == ' ' || *src == '\t') src++;
    size_t n = strlen(src);
    while (n > 0 && (src[n - 1] == ' ' || src[n - 1] == '\t' || src[n - 1] == '\n'))
        n--;
    if (n >= cap) n = cap - 1;
    memcpy(out, src, n);
    out[n] = '\0';
}

void rj_cpu_detect(rj_cpu_info *out)
{
    memset(out, 0, sizeof(*out));
    unsigned long hwcap = getauxval(AT_HWCAP);
    unsigned long hwcap2 = getauxval(AT_HWCAP2);
    if (hwcap & HWCAP_ASIMD)
        out->neon = 1;
    if (hwcap & HWCAP_SVE) {
        long vl = prctl(PR_SVE_GET_VL, 0, 0, 0, 0);
        if (vl > 0) {
            int sve_vl = (int)(((unsigned long)vl & PR_SVE_VL_LEN_MASK) / 8);
            /* PR_SVE_GET_VL is unreliable under qemu-user (reports 8
             * bytes while the true VL is 64). A VL below 16 bytes is
             * not useful; fall back to NEON in that case. */
            if (sve_vl >= 16) {
                out->sve = 1;
                out->sve_vl = sve_vl;
                if (hwcap2 & HWCAP2_SVE2)
                    out->sve2 = 1;
            }
        }
    }
    read_brand(out->brand, sizeof(out->brand));
}
