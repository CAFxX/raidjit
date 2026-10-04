/* CPU/CET audit: verifies CPUID max-leaf handling, XCR0 OS-state gating per
 * encoding width, and the behavior of the optional CET/trap instructions in
 * isolated child processes.
 *
 * Portable assertions:
 *  - endbr64 never faults (decodes as NOP on CPUs without CET).
 *  - ud2 always raises SIGILL.
 *  - setssbsy MUST raise SIGILL when rj_cpu_detect() reports cet_ss=0
 *    (hardware absent => #UD is architectural). When cet_ss=1 the outcome
 *    depends on OS enablement (CR4.CET / IA32_U_CET), so it is reported only.
 *  - XCR0 cross-check: avx/avx2 set => XCR0[2]; avx512* set => XCR0[7:5].
 */
#define _GNU_SOURCE
#include <setjmp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "raidjit.h"

static int failures;

#define CHECK(cond, ...) do {                                             \
    if (!(cond)) {                                                        \
        printf("FAIL %s:%d: ", __FILE__, __LINE__);                       \
        printf(__VA_ARGS__);                                              \
        printf("\n");                                                     \
        failures++;                                                       \
    }                                                                     \
} while (0)

/* ---- child probe: executes one instruction sequence, reports SIGILL ---- */

static volatile sig_atomic_t saw_sigill;
static sigjmp_buf jmp;

static void ill_handler(int sig, siginfo_t *si, void *uc)
{
    (void)sig; (void)si; (void)uc;
    saw_sigill = 1;
    siglongjmp(jmp, 1);
}

/* which: 0=endbr64, 1=setssbsy+clrssbsy, 2=ud2. _exit(10+faulted). */
static void child_probe(int which)
{
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = ill_handler;
    sa.sa_flags = SA_SIGINFO | SA_NODEFER;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGILL, &sa, NULL);

    saw_sigill = 0;
    if (sigsetjmp(jmp, 1) == 0) {
        switch (which) {
        case 0: /* endbr64: F3 0F 1E FA */
            __asm__ volatile(".byte 0xF3,0x0F,0x1E,0xFA" ::: "memory");
            break;
        case 1: /* setssbsy / clrssbsy */
            __asm__ volatile(".byte 0xF3,0x0F,0x01,0xEA" ::: "memory");
            __asm__ volatile(".byte 0xF3,0x0F,0x01,0xE9" ::: "memory");
            break;
        case 2: /* ud2 */
            __asm__ volatile(".byte 0x0F,0x0B" ::: "memory");
            break;
        }
    }
    _exit(10 + (saw_sigill ? 1 : 0));
}

/* returns 1 if the probed instruction faulted, 0 if it executed, -1 on error */
static int probe(int which)
{
    pid_t p = fork();
    if (p < 0)
        return -1;
    if (p == 0)
        child_probe(which);
    int st = 0;
    if (waitpid(p, &st, 0) < 0)
        return -1;
    if (!WIFEXITED(st))
        return -1; /* child died by signal: unexpected */
    int code = WEXITSTATUS(st);
    if (code < 10 || code > 11)
        return -1;
    return code - 10;
}

static unsigned long long xgetbv0(void)
{
    unsigned lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return ((unsigned long long)hi << 32) | lo;
}

static int osxsave_enabled(void)
{
    unsigned a, b, c, d;
    __asm__ volatile("cpuid"
                     : "=a"(a), "=b"(b), "=c"(c), "=d"(d)
                     : "a"(1), "c"(0));
    return (c >> 27) & 1; /* CPUID.1:ECX.OSXSAVE: OS set CR4.OSXSAVE */
}

int main(void)
{
#if !defined(__x86_64__)
    printf("cet: skipped (non-x86_64)\n");
    return 0;
#else
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    printf("cet: cpu brand: %s\n", cpu.brand);
    CHECK(cpu.brand[0] != 0, "empty brand string");
    CHECK(cpu.sse2 == 1, "sse2 not detected on x86-64");

    /* XCR0 gating cross-check against live XCR0. XGETBV faults (#UD) when
     * the OS has not enabled XSAVE, so only probe when CPUID reports
     * OSXSAVE -- same gate rj_cpu_detect() uses. */
    if (osxsave_enabled()) {
        unsigned long long xcr0 = xgetbv0();
        if (cpu.avx || cpu.avx2)
            CHECK((xcr0 & 0x4) != 0,
                  "avx/avx2 reported but XCR0[2] (YMM) clear: xcr0=0x%llx",
                  xcr0);
        if (cpu.avx512f || cpu.avx512bw || cpu.avx512vl)
            CHECK((xcr0 & 0xE0) == 0xE0,
                  "avx512 reported but XCR0[7:5] not all set: xcr0=0x%llx",
                  xcr0);
    } else {
        printf("cet: OSXSAVE clear, skipping XCR0 cross-check\n");
    }
    if (cpu.gfni)
        CHECK(cpu.avx512f == 1, "gfni without avx512f");

    /* endbr64 must execute everywhere (NOP where CET is absent). */
    int r = probe(0);
    CHECK(r == 0, "endbr64 faulted (r=%d); must be NOP on non-CET CPUs", r);

    /* ud2 must always trap. */
    r = probe(2);
    CHECK(r == 1, "ud2 did not raise SIGILL (r=%d)", r);

    /* setssbsy: architectural #UD when CET_SS hardware is absent. */
    r = probe(1);
    if (cpu.cet_ss == 0) {
        CHECK(r == 1, "setssbsy executed without CET_SS hw (r=%d); want #UD",
              r);
    } else {
        printf("cet: cet_ss=1, setssbsy %s (OS enablement decides)\n",
               r == 0 ? "executed: OS enabled shadow stack"
                      : r == 1 ? "faulted: OS did not enable shadow stack"
                               : "probe error");
    }

    if (!failures)
        printf("cet: all OK\n");
    return failures ? 1 : 0;
#endif
}
