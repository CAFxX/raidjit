/* test_maskedtail: correctness of the masked AVX-512 tail (track B1x86).
 *
 * Builds maskedtail P_XOR versions (all AVX-512 U values) for a RAID-5
 * (n_out=1) and a custom dual-XOR-parity (n_out=2) topology, and checks
 * each against the scalar reference (ref_run) at every length
 * 0..2*group, plus verifies bytes past the tail are untouched (merging
 * stores). No RAID-6 case: the masked tail is eligible only for P_XOR
 * winners, and no RAID-6 encode path in this codebase is P_XOR
 * (rj_topo_raid6() dispatches to XOR array codes or Reed-Solomon, whose
 * masked tails are not implemented); the dual-XOR-parity topology is
 * the multi-parity P_XOR coverage.
 *
 * Guard-page probe (F5): masked 64B loads/stores with lanes masked off
 * across a PROT_NONE page must not fault (SDM claim, verified
 * empirically on this silicon). Buffers are abutted to the guard page so
 * the masked-off lanes reach into it; any over-read/over-write faults
 * loudly.
 *
 * Red-green (verified 2026-10-03 in an isolated tree copy, closeout):
 * (M1) mask table forced to ~0ULL for all i (mask-all-lanes): the
 * masked stores write full 64B chunks past the tail -> the test
 * crashes with SIGSEGV on the heap overwrite (loud catch).
 * (M2) mask table one bit short ((1<<(i-1))-1): the last tail byte is
 * never loaded/stored -> "mismatch" FAILs vs the scalar reference.
 * Both restored to PASS afterwards.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>
#include <unistd.h>
#include "../src/jit.h"
#include "../src/plan.h"
#include "../src/ref.h"
#include "../include/raidjit.h"

static int failures = 0;

static void *abuf(size_t n)
{
    void *p = NULL;
    if (posix_memalign(&p, 64, n ? n : 64)) {
        printf("oom\n");
        exit(1);
    }
    return p;
}

static void test_one_topo(rj_topology *t, rj_op op, const char *name)
{
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    topo_int_t ti;
    char err[256];
    if (topo_int_build(&ti, t, err, sizeof(err))) {
        printf("FAIL %s topo: %s\n", name, err);
        failures++;
        return;
    }
    plan_t pl;
    if (plan_build(&ti, &op, name, &pl, err, sizeof(err))) {
        printf("FAIL %s plan: %s\n", name, err);
        failures++;
        topo_int_free(&ti);
        return;
    }
    if (pl.kind != P_XOR) {
        printf("FAIL %s plan kind=%d, want P_XOR\n", name, pl.kind);
        failures++;
        plan_free(&pl);
        topo_int_free(&ti);
        return;
    }

    cand_t cands[64];
    int nc = plan_candidates(&pl, &cpu, cands, 64, 0, 0);
    int tested = 0;
    for (int ci = 0; ci < nc; ci++) {
        if (cands[ci].isa != ISA_AVX512)
            continue;
        if (cands[ci].xorsched || cands[ci].ternlog)
            continue; /* masked tail supports the plain P_XOR body only
                       * (gen_tail_masked asserts this); api.c likewise
                       * excludes these from maskedtail eligibility */
        cand_t mc = cands[ci];
        mc.maskedtail = 1;
        rj_options o;
        rj_options_default(&o);
        version_t v;
        if (jit_build_version(&pl, &mc, &cpu, &o, &v, err, sizeof(err))) {
            printf("FAIL %s: build U=%d: %s\n", name, mc.U, err);
            failures++;
            continue;
        }
        size_t group = v.group_bytes;
        int n_in = pl.n_in, n_out = pl.n_out;
        size_t maxlen = 2 * group + 7;
        uint8_t **in = malloc(sizeof(*in) * (size_t)n_in);
        uint8_t **out = malloc(sizeof(*out) * (size_t)n_out);
        uint8_t **ref = malloc(sizeof(*ref) * (size_t)n_out);
        for (int i = 0; i < n_in; i++) {
            in[i] = abuf(maxlen);
            for (size_t k = 0; k < maxlen; k++)
                in[i][k] = (uint8_t)(i * 37 + k * 11 + (k * k & 0xff));
        }
        for (int j = 0; j < n_out; j++) {
            out[j] = abuf(maxlen);
            ref[j] = abuf(maxlen);
        }

        int bad = 0;
        for (size_t len = 0; len <= 2 * group + 7; len++) {
            for (int j = 0; j < n_out; j++) {
                memset(out[j], 0xA5, maxlen);
                memset(ref[j], 0xA5, maxlen);
            }
            ref_run(&pl, (const uint8_t **)in, ref, len);
            v.fn((const void **)in, (void **)out, len);
            for (int j = 0; j < n_out; j++) {
                if (memcmp(out[j], ref[j], len) != 0) {
                    if (bad < 3)
                        printf("FAIL %s U=%d len=%zu out=%d mismatch\n",
                               name, mc.U, len, j);
                    bad++;
                }
                for (size_t k = len; k < maxlen && k < len + 128; k++) {
                    if (out[j][k] != 0xA5) {
                        if (bad < 3)
                            printf("FAIL %s U=%d len=%zu out=%d overwrite "
                                   "k=%zu\n",
                                   name, mc.U, len, j, k);
                        bad++;
                        break;
                    }
                }
            }
            if (bad >= 10)
                break;
        }
        if (bad) {
            printf("FAIL %s U=%d: %d mismatches/overwrites\n", name, mc.U,
                   bad);
            failures++;
        } else {
            printf("ok %s U=%d group=%zu n_out=%d\n", name, mc.U, group,
                   n_out);
        }
        tested++;

        for (int i = 0; i < n_in; i++)
            free(in[i]);
        for (int j = 0; j < n_out; j++) {
            free(out[j]);
            free(ref[j]);
        }
        free(in);
        free(out);
        free(ref);
        jit_free_version(&v);
    }
    if (!tested) {
        printf("FAIL %s: no AVX-512 candidates tested\n", name);
        failures++;
    }
    plan_free(&pl);
    topo_int_free(&ti);
}

/* Guard-page probe: the masked tail's 64B loads/stores must not fault on
 * masked-off lanes, even when those lanes lie in a PROT_NONE page.
 * Buffers end exactly at the guard page; for len in 1..63 the masked-off
 * tail of the 64B chunk reaches into it. A faulting implementation
 * crashes here (loud); a correct one produces bit-exact output. */
static void test_guard_page(void)
{
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    long ps = sysconf(_SC_PAGESIZE);
    if (ps < 4096)
        ps = 4096;

    rj_topology *t = rj_topo_raid5(4);
    rj_op op = rj_op_encode();
    topo_int_t ti;
    char err[256];
    if (topo_int_build(&ti, t, err, sizeof(err))) {
        printf("FAIL guard topo: %s\n", err);
        failures++;
        rj_topo_free(t);
        return;
    }
    plan_t pl;
    if (plan_build(&ti, &op, "guard", &pl, err, sizeof(err))) {
        printf("FAIL guard plan: %s\n", err);
        failures++;
        topo_int_free(&ti);
        rj_topo_free(t);
        return;
    }

    uint8_t *base = mmap(NULL, (size_t)2 * ps, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED) {
        printf("FAIL guard: mmap failed\n");
        failures++;
        plan_free(&pl);
        topo_int_free(&ti);
        rj_topo_free(t);
        return;
    }
    /* Second page is the guard: any access faults. */
    if (mprotect(base + ps, (size_t)ps, PROT_NONE)) {
        printf("FAIL guard: mprotect failed\n");
        failures++;
        munmap(base, (size_t)2 * ps);
        plan_free(&pl);
        topo_int_free(&ti);
        rj_topo_free(t);
        return;
    }

    cand_t cands[64];
    int nc = plan_candidates(&pl, &cpu, cands, 64, 0, 0);
    int tested = 0;
    for (int ci = 0; ci < nc; ci++) {
        if (cands[ci].isa != ISA_AVX512)
            continue;
        if (cands[ci].xorsched || cands[ci].ternlog)
            continue; /* plain P_XOR body only (see test_one_topo) */
        cand_t mc = cands[ci];
        mc.maskedtail = 1;
        rj_options o;
        rj_options_default(&o);
        version_t v;
        if (jit_build_version(&pl, &mc, &cpu, &o, &v, err, sizeof(err))) {
            printf("FAIL guard: build U=%d: %s\n", mc.U, err);
            failures++;
            continue;
        }
        int n_in = pl.n_in, n_out = pl.n_out;
        /* one input/output/ref triple, reused across lens */
        uint8_t *refb[4];
        uint8_t *ref = abuf((size_t)ps);
        refb[0] = ref;
        const void *in[8];
        void *out[4];
        uint8_t *inb[8], *outb[4];
        for (int i = 0; i < n_in; i++) {
            inb[i] = abuf((size_t)ps);
            in[i] = inb[i];
        }
        for (int j = 0; j < n_out; j++) {
            outb[j] = abuf((size_t)ps);
            out[j] = outb[j];
        }

        size_t lens[] = { 1, 31, 63 };
        for (size_t li = 0; li < sizeof(lens) / sizeof(lens[0]); li++) {
            size_t len = lens[li];
            /* Abut buffers to the guard page: the 64B masked chunk at
             * [base+ps-len, base+ps-len+64) has its masked-off tail
             * [base+ps, base+ps-len+64) inside PROT_NONE for len<64. */
            for (int i = 0; i < n_in; i++) {
                uint8_t *p = base + ps - len;
                for (size_t k = 0; k < len; k++)
                    p[k] = (uint8_t)(i * 131 + k * 17 + (k * k & 0xff));
                in[i] = p;
            }
            for (int j = 0; j < n_out; j++) {
                out[j] = base + ps - len;
                memset(ref, 0xA5, len);
            }
            /* reference from the shadow copies */
            for (int i = 0; i < n_in; i++)
                memcpy(inb[i], (const void *)in[i], len);
            ref_run(&pl, (const uint8_t **)inb, refb, len);
            /* the probe: masked-off lanes reach into PROT_NONE; a
             * faulting implementation dies here */
            v.fn(in, out, len);
            for (int j = 0; j < n_out; j++) {
                if (memcmp(out[j], ref, len) != 0) {
                    printf("FAIL guard U=%d len=%zu out=%d mismatch\n",
                           mc.U, len, j);
                    failures++;
                }
            }
            /* restore pointer arrays for the next iteration */
            for (int i = 0; i < n_in; i++)
                in[i] = inb[i];
            for (int j = 0; j < n_out; j++)
                out[j] = outb[j];
        }
        printf("ok guard U=%d (masked-off lanes across PROT_NONE)\n",
               mc.U);
        tested++;
        for (int i = 0; i < n_in; i++)
            free(inb[i]);
        for (int j = 0; j < n_out; j++)
            free(outb[j]);
        free(ref);
        jit_free_version(&v);
    }
    if (!tested) {
        printf("FAIL guard: no AVX-512 candidates tested\n");
        failures++;
    }
    munmap(base, (size_t)2 * ps);
    plan_free(&pl);
    topo_int_free(&ti);
    rj_topo_free(t);
}

int main(void)
{
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    if (!cpu.avx512f || !cpu.avx512bw) {
        printf("SKIP: need AVX-512F+BW\n");
        return 0;
    }

    rj_topology *t5 = rj_topo_raid5(4);
    rj_op op5 = rj_op_encode();
    test_one_topo(t5, op5, "raid5(4)");
    rj_op_free(&op5);
    rj_topo_free(t5);

    /* dual-XOR-parity custom topology: P_XOR with n_out=2 */
    uint8_t mat[8] = { 1, 1, 1, 1, 1, 1, 1, 1 };
    rj_topology *tc = rj_topo_custom(4, 2, mat);
    rj_op opc = rj_op_encode();
    test_one_topo(tc, opc, "custom4+2");
    rj_op_free(&opc);
    rj_topo_free(tc);

    test_guard_page();

    printf(failures ? "FAIL test_maskedtail\n" : "PASS test_maskedtail\n");
    return failures ? 1 : 0;
}
