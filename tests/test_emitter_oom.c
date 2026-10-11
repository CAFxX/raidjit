/* Emitter OOM error-propagation test.
 *
 * The JIT emitter must handle out-of-memory cleanly (sticky error flag +
 * clean rejection) instead of aborting via assert(). This test forces
 * OOM conditions and verifies the error flag is set (no abort).
 *
 * Fork-isolated per case (same pattern as test_pinned_negative.c):
 * an abort (signal) fails the test; clean flag-setting passes.
 *
 * Non-tautological: without the OOM fix, the buffer-overflow case
 * aborts on assert(e->len < e->cap) instead of setting e.err.
 *
 * Audit note (2026-10-11): all allocation sites in the emitter paths
 * were audited by reading the code (not just grepping for assert).
 * Every site was at least assert-guarded; no unguarded sites were found.
 * This test covers the converted assert sites.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#if defined(__x86_64__)
#include "../src/x86_64/emit.h"
#elif defined(__aarch64__)
#include "../src/aarch64/emit.h"
#elif defined(__riscv)
#include "../src/riscv/emit.h"
#else
#error "unsupported arch for emitter OOM test"
#endif

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

enum { OC_OK = 0, OC_ABORTED = 2 };

/* Run fn in a forked child. Returns OC_OK if the child exits 0,
 * OC_ABORTED if it dies by signal (assert abort). */
static int fork_isolated(void (*fn)(void))
{
    pid_t pid = fork();
    if (pid == 0) {
        fn();
        _exit(0);
    }
    int status;
    waitpid(pid, &status, 0);
    if (WIFSIGNALED(status))
        return OC_ABORTED;
    return WIFEXITED(status) && WEXITSTATUS(status) == 0 ? OC_OK : OC_ABORTED;
}

/* Case 1: buffer overflow sets the sticky err flag instead of aborting.
 * Without the fix: assert(e->len < e->cap) fires on the 17th byte. */
static void case_buffer_overflow(void)
{
    em_t e;
    uint8_t buf[16];
    em_init(&e, buf, sizeof(buf));
    for (int i = 0; i < 100; i++)
        em_byte(&e, 0x90);
    /* Must have set the flag, not aborted. */
    if (!e.err)
        _exit(1);
    /* Length must not have exceeded capacity (no overflow write). */
    if (e.len > e.cap)
        _exit(1);
    em_free(&e);
}

/* Case 2: once err is set, em_label returns -1 and em_bind/em_resolve
 * are no-ops (no crash on the -1 label). */
static void case_err_propagation(void)
{
    em_t e;
    uint8_t buf[16];
    em_init(&e, buf, sizeof(buf));
    /* Force the error flag via overflow. */
    for (int i = 0; i < 100; i++)
        em_byte(&e, 0x90);
    if (!e.err)
        _exit(1);
    /* em_label must return -1, not crash. */
    int lab = em_label(&e);
    if (lab != -1)
        _exit(1);
    /* em_bind/em_resolve with the bad label must be safe no-ops. */
    em_bind(&e, lab);
    em_resolve(&e);
    em_free(&e);
}

/* Case 3: normal operation is unaffected (regression check).
 * Small emits within capacity must not set err. */
static void case_normal_ok(void)
{
    em_t e;
    uint8_t buf[1024];
    em_init(&e, buf, sizeof(buf));
    int lab = em_label(&e);
    if (lab < 0)
        _exit(1);
    for (int i = 0; i < 100; i++)
        em_byte(&e, 0x90);
    em_bind(&e, lab);
    em_resolve(&e);
    if (e.err)
        _exit(1);
    if (e.len != 100)
        _exit(1);
    em_free(&e);
}

int main(void)
{
    int rc;

    rc = fork_isolated(case_buffer_overflow);
    CHECK(rc == OC_OK, "buffer overflow: expected clean err flag, got %s",
          rc == OC_ABORTED ? "ABORT" : "child failure");
    if (rc == OC_OK)
        printf("ok: buffer overflow sets err flag (no abort)\n");

    rc = fork_isolated(case_err_propagation);
    CHECK(rc == OC_OK, "err propagation: expected safe no-ops, got %s",
          rc == OC_ABORTED ? "ABORT" : "child failure");
    if (rc == OC_OK)
        printf("ok: err flag propagates safely through label/bind/resolve\n");

    rc = fork_isolated(case_normal_ok);
    CHECK(rc == OC_OK, "normal operation: unexpected err flag or abort");
    if (rc == OC_OK)
        printf("ok: normal emission unaffected\n");

    if (failures == 0)
        printf("PASS: all emitter OOM tests passed\n");
    return failures != 0;
}
