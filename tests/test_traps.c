/* test_traps: trap sled after ret on every backend.
 *
 * Verifies that each emitted version ends its code region with
 *   ret; <trap>; <trap-fill padding to the const-pool alignment>
 * and that the dead space between ret and the pool holds only trap
 * instructions -- never NOPs:
 *   x86-64 : ret (C3), UD2 (0F 0B) pairs, optional trailing INT3 (CC)
 *   AArch64: ret (D65F03C0), UDF #0 (00000000) words
 *   RISC-V : ret (jalr x0, ra, 0 = 00008067), UNIMP (00000000) words
 * A fall-through past ret is a codegen bug and must fault immediately
 * instead of sliding through NOPs into the const pool.
 *
 * Method: build one P_XOR version for the host ISA via jit_build_version,
 * smoke-check it against the scalar reference, then scan
 * v.mem[0..v.code_size): locate the LAST ret+trap pattern (the real ret;
 * any spurious pattern inside the body would sit before it) and verify
 * every byte/word from the trap through code_size is a valid trap-fill
 * unit, and that code_size honors the pool alignment.
 *
 * Only encode versions are built: the trap sled is emitted in
 * gen_function's single shared epilogue, identical for every op and
 * every twin, so per-op coverage adds nothing.
 *
 * Red-green (2026-10-04, isolated tree copy): (M1) em_trap_align mutated
 * to emit NOPs instead of traps: the test FAILs ("non-trap byte 0x90 in
 * dead region"). (M2) the explicit post-ret trap call deleted from
 * gen_function: the test FAILs on the ndata=15 versions whose ret lands
 * exactly on the 64-byte boundary ("ret+trap pattern not found") --
 * proving the explicit trap is load-bearing in the aligned edge case,
 * not just the padding. Both restored to PASS afterwards.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "../src/jit.h"
#include "../src/plan.h"
#include "../src/ref.h"
#include "../include/raidjit.h"

static int failures = 0;
static int checked = 0;

static void *abuf(size_t n)
{
    void *p = NULL;
    if (posix_memalign(&p, 64, n ? n : 64)) {
        printf("oom\n");
        exit(1);
    }
    return p;
}

/* Verify the dead region [trap_off, code_size) is all trap-fill.
 * Returns 1 on success. Each backend supplies its own alphabet. */
#if defined(__x86_64__)
static int dead_is_traps(const uint8_t *code, size_t trap_off, size_t code_size)
{
    /* trap_off points at the UD2 right after ret */
    size_t p = trap_off;
    if (p + 2 > code_size || code[p] != 0x0F || code[p + 1] != 0x0B) {
        printf("FAIL x86_64: no UD2 immediately after ret (off %zu)\n", p);
        return 0;
    }
    p += 2;
    while (p + 2 <= code_size && code[p] == 0x0F && code[p + 1] == 0x0B)
        p += 2;
    if (p < code_size) {
        if (code[p] != 0xCC || p + 1 != code_size) {
            printf("FAIL x86_64: non-trap byte 0x%02x at off %zu "
                   "(code_size %zu)\n", code[p], p, code_size);
            return 0;
        }
        p += 1;
    }
    if (p != code_size) {
        printf("FAIL x86_64: trap fill ends at %zu, want %zu\n", p, code_size);
        return 0;
    }
    return 1;
}

/* Last occurrence of C3 0F 0B (ret; ud2) below code_size. The dead zone
 * itself contains no C3, so the last match is the real ret. */
static long find_ret_trap(const uint8_t *code, size_t code_size)
{
    long found = -1;
    for (size_t i = 0; i + 3 <= code_size; i++)
        if (code[i] == 0xC3 && code[i + 1] == 0x0F && code[i + 2] == 0x0B)
            found = (long)i;
    return found;
}
#elif defined(__aarch64__)
static int dead_is_traps(const uint8_t *code, size_t trap_off, size_t code_size)
{
    if (trap_off + 4 > code_size) {
        printf("FAIL aarch64: trap off %zu beyond code_size %zu\n",
               trap_off, code_size);
        return 0;
    }
    if (code_size % 4 != 0) {
        printf("FAIL aarch64: code_size %zu not word-aligned\n", code_size);
        return 0;
    }
    for (size_t p = trap_off; p < code_size; p += 4) {
        uint32_t w = (uint32_t)code[p] | ((uint32_t)code[p + 1] << 8) |
                     ((uint32_t)code[p + 2] << 16) |
                     ((uint32_t)code[p + 3] << 24);
        if (w != 0x00000000u) { /* UDF #0 */
            printf("FAIL aarch64: non-UDF word %08x at off %zu\n", w, p);
            return 0;
        }
    }
    return 1;
}

static long find_ret_trap(const uint8_t *code, size_t code_size)
{
    long found = -1;
    for (size_t i = 0; i + 8 <= code_size; i += 4) {
        uint32_t w0 = (uint32_t)code[i] | ((uint32_t)code[i + 1] << 8) |
                      ((uint32_t)code[i + 2] << 16) |
                      ((uint32_t)code[i + 3] << 24);
        uint32_t w1 = (uint32_t)code[i + 4] | ((uint32_t)code[i + 5] << 8) |
                      ((uint32_t)code[i + 6] << 16) |
                      ((uint32_t)code[i + 7] << 24);
        if (w0 == 0xD65F03C0u && w1 == 0x00000000u) /* ret; udf #0 */
            found = (long)i;
    }
    return found;
}
#elif defined(__riscv)
static int dead_is_traps(const uint8_t *code, size_t trap_off, size_t code_size)
{
    if (code_size % 4 != 0) {
        printf("FAIL riscv: code_size %zu not word-aligned\n", code_size);
        return 0;
    }
    for (size_t p = trap_off; p < code_size; p += 4) {
        uint32_t w = (uint32_t)code[p] | ((uint32_t)code[p + 1] << 8) |
                     ((uint32_t)code[p + 2] << 16) |
                     ((uint32_t)code[p + 3] << 24);
        if (w != 0x00000000u) { /* UNIMP */
            printf("FAIL riscv: non-UNIMP word %08x at off %zu\n", w, p);
            return 0;
        }
    }
    return 1;
}

static long find_ret_trap(const uint8_t *code, size_t code_size)
{
    long found = -1;
    for (size_t i = 0; i + 8 <= code_size; i += 4) {
        uint32_t w0 = (uint32_t)code[i] | ((uint32_t)code[i + 1] << 8) |
                      ((uint32_t)code[i + 2] << 16) |
                      ((uint32_t)code[i + 3] << 24);
        uint32_t w1 = (uint32_t)code[i + 4] | ((uint32_t)code[i + 5] << 8) |
                      ((uint32_t)code[i + 6] << 16) |
                      ((uint32_t)code[i + 7] << 24);
        /* jalr x0, ra, 0 = 0x00008067; unimp = 0x00000000 */
        if (w0 == 0x00008067u && w1 == 0x00000000u)
            found = (long)i;
    }
    return found;
}
#else
static int dead_is_traps(const uint8_t *code, size_t trap_off, size_t code_size)
{
    (void)code; (void)trap_off; (void)code_size;
    printf("SKIP: unknown host ISA\n");
    return 1;
}
static long find_ret_trap(const uint8_t *code, size_t code_size)
{
    (void)code; (void)code_size;
    return -2; /* signal skip */
}
#endif

static void test_version(version_t *v, plan_t *pl, const char *tag)
{
    const uint8_t *code = (const uint8_t *)v->mem;
    size_t code_size = v->code_size;

    /* pool alignment: 64 on x86-64/AArch64, 16 on RISC-V */
#if defined(__x86_64__) || defined(__aarch64__)
    size_t align = 64;
#else
    size_t align = 16;
#endif
    if (code_size % align != 0) {
        printf("FAIL %s: code_size %zu not %zu-aligned\n", tag, code_size,
               align);
        failures++;
        return;
    }

    long r = find_ret_trap(code, code_size);
    if (r == -2)
        return; /* unknown ISA: skipped above */
    if (r < 0) {
        printf("FAIL %s: ret+trap pattern not found in %zu code bytes\n",
               tag, code_size);
        failures++;
        return;
    }
#if defined(__x86_64__)
    size_t trap_off = (size_t)r + 1; /* UD2 right after the C3 */
#else
    size_t trap_off = (size_t)r + 4; /* trap word right after ret */
#endif
    if (!dead_is_traps(code, trap_off, code_size)) {
        failures++;
        return;
    }
    checked++;
    (void)pl;
}

int main(void)
{
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    char err[256];

    /* ndata=15 is known to produce ret landing exactly on the 64-byte
     * boundary (ret_off % 64 == 63) for several candidates -- the edge
     * case where em_trap_align emits nothing and the explicit post-ret
     * trap is load-bearing. ndata=4 gives the common padded case. */
    int ndatas[] = { 4, 15 };
    for (int t = 0; t < 2; t++) {
        rj_topology *top = rj_topo_raid5(ndatas[t]);
        rj_op op = rj_op_encode();
        topo_int_t ti;
        if (topo_int_build(&ti, top, err, sizeof(err))) {
            printf("FAIL topo: %s\n", err);
            return 1;
        }
        plan_t pl;
        if (plan_build(&ti, &op, "traps", &pl, err, sizeof(err))) {
            printf("FAIL plan: %s\n", err);
            return 1;
        }

        cand_t cands[64];
        int nc = plan_candidates(&pl, &cpu, cands, 64, 0, 0);
        if (nc <= 0) {
            printf("FAIL: no candidates\n");
            return 1;
        }
        for (int ci = 0; ci < nc; ci++) {
            cand_t c = cands[ci];
            rj_options o;
            rj_options_default(&o);
            o.benchmark = 0; /* no timing needed; we inspect the bytes */
            version_t v;
            if (jit_build_version(&pl, &c, &cpu, &o, &v, err, sizeof(err))) {
                printf("FAIL build nd=%d cand=%d: %s\n", ndatas[t], ci, err);
                failures++;
                continue;
            }

            /* Smoke on the first version per topology: it must still
             * execute correctly (traps are never on the executed path). */
            if (ci == 0) {
                int n_in = pl.n_in;
                size_t group = v.group_bytes;
                size_t len = group > 0 ? group : 64;
                uint8_t **in = malloc(sizeof(*in) * (size_t)n_in);
                uint8_t *out[1], *ref[1];
                for (int i = 0; i < n_in; i++) {
                    in[i] = abuf(len);
                    for (size_t k = 0; k < len; k++)
                        in[i][k] = (uint8_t)(i * 31 + k * 7);
                }
                out[0] = abuf(len);
                ref[0] = abuf(len);
                memset(out[0], 0xA5, len);
                ref_run(&pl, (const uint8_t **)in, ref, len);
                v.fn((const void **)in, (void **)out, len);
                if (memcmp(out[0], ref[0], len) != 0) {
                    printf("FAIL: smoke mismatch after trap change\n");
                    failures++;
                }
                for (int i = 0; i < n_in; i++)
                    free(in[i]);
                free(in);
                free(out[0]);
                free(ref[0]);
            }

            char tag[64];
            snprintf(tag, sizeof(tag), "raid5-%d/cand%d", ndatas[t], ci);
            test_version(&v, &pl, tag);
        }
        plan_free(&pl);
        topo_int_free(&ti);
        rj_topo_free(top);
    }

    if (failures == 0 && checked > 0)
        printf("PASS test_traps (%d version(s) checked)\n", checked);
    if (checked == 0) {
        /* Unreachable via the Makefile (unknown ARCH is a hard error),
         * but a zero-check PASS must never be shaped like success. */
        printf("FAIL test_traps: no versions checked\n");
        return 1;
    }
    return failures != 0;
}
