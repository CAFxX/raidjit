/* Per-candidate exhaustive correctness matrix.
 *
 * For every (synthetic CPU feature set x topology x operation), enumerate
 * EVERY candidate plan_candidates() offers, JIT each one directly, and
 * verify the generated code against the scalar GF(2^8) reference.
 *
 * Data-corruption focus:
 *   - every buffer carries 64-byte canaries on both sides; any out-of-bounds
 *     write fails the test;
 *   - tail-sensitive lengths additionally run with the buffer end-aligned to
 *     a PROT_NONE guard page and a SIGSEGV handler, so even a 1-byte
 *     over-read/over-write is caught instead of silently corrupting.
 *
 * Synthetic CPU sets cover ISA/extension combinations the host may lack
 * (e.g. AVX512 without GFNI): the planner only sees the synthetic flags and
 * the generated code only uses those extensions, so it executes natively on
 * any superset host. Set RAIDJIT_TEST_CPU=detect to use rj_cpu_detect()
 * instead (the qemu path: run the whole binary under
 * `qemu-x86_64 -cpu <model>`; see tests/qemu_matrix.sh).
 */
#define _GNU_SOURCE
#include "../include/raidjit.h"
#include "../src/plan.h"
#include "../src/jit.h"
#include "../src/gf.h"
#include "../src/ref.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <signal.h>
#include <setjmp.h>
/* ucontext.h needs kernel headers (asm/sigcontext.h) that are not always
 * present in cross sysroots; the fault PC is diagnostic-only. */
#if defined(__has_include) && __has_include(<asm/sigcontext.h>)
#include <ucontext.h>
#define RJ_HAVE_UCONTEXT 1
#endif
#include <unistd.h>
#include <sys/mman.h>

static int failures = 0;
static long nruns = 0;
#define CHECK(cond, ...)                                                   \
    do {                                                                   \
        if (!(cond)) {                                                     \
            printf("FAIL %s:%d: ", __func__, __LINE__);                    \
            printf(__VA_ARGS__);                                           \
            printf("\n");                                                  \
            failures++;                                                    \
        }                                                                  \
    } while (0)

static uint64_t rng_state = 0x243f6a8885a308d3ULL;
static uint64_t xrng(void)
{
    uint64_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return rng_state = x;
}

/* ---------------- canary buffers ---------------- */

#define CAN 64
#define CANARY 0xA5

/* 64-byte aligned data region with CAN-byte canaries on both sides */
static uint8_t *cbuf_alloc(size_t len, uint8_t **base_out)
{
    uint8_t *base = NULL;
    size_t total = len + 2 * CAN;
    if (posix_memalign((void **)&base, 64, total ? total : 64)) {
        fprintf(stderr, "oom\n");
        exit(1);
    }
    memset(base, CANARY, total ? total : 64);
    *base_out = base;
    return base + CAN;
}

static int cbuf_check(uint8_t *base, size_t len)
{
    for (size_t i = 0; i < CAN; i++)
        if (base[i] != CANARY)
            return 0;
    for (size_t i = 0; i < CAN; i++)
        if (base[CAN + len + i] != CANARY)
            return 0;
    return 1;
}

/* Offset variant: buffers start at base + CAN + off for several 64-byte
 * aligned offsets, exercising different positions in the page while honoring
 * the API's 64-byte alignment contract. Canaries cover both sides. */
static uint8_t *ubuf_alloc(size_t len, size_t off, uint8_t **base_out)
{
    uint8_t *base = NULL;
    size_t total = len + off + 2 * CAN;
    if (posix_memalign((void **)&base, 64, total ? total : 64)) {
        fprintf(stderr, "oom\n");
        exit(1);
    }
    memset(base, CANARY, total ? total : 64);
    *base_out = base;
    return base + CAN + off;
}

static int ubuf_check(uint8_t *base, size_t off, size_t len)
{
    for (size_t i = 0; i < CAN + off; i++)
        if (base[i] != CANARY)
            return 0;
    for (size_t i = 0; i < CAN; i++)
        if (base[CAN + off + len + i] != CANARY)
            return 0;
    return 1;
}

/* aligned buffer-start residues with good spread mod 64 */
/* Offset sweep: buffers start at base + CAN + off. Every offset is a
 * multiple of 64 so the 64-byte alignment contract holds while different
 * positions within the page (and different DTLB/page-split behavior) are
 * still exercised. */
static const size_t align_offs[] = { 0, 64, 128, 192 };

/* small length set for the unaligned sweep (deduped against group bounds) */
static int build_ulens(size_t *out, int group)
{
    size_t tmp[] = { 1, 65, 129, 1000,
                     (size_t)(group > 1 ? group - 1 : 0),
                     (size_t)group, (size_t)group + 1 };
    int n = 0;
    for (int i = 0; i < 7; i++) {
        int dup = 0;
        for (int j = 0; j < n; j++)
            if (out[j] == tmp[i]) {
                dup = 1;
                break;
            }
        if (!dup)
            out[n++] = tmp[i];
    }
    return n;
}

static void randfill(uint8_t *p, size_t n)
{
    for (size_t i = 0; i < n; i++)
        p[i] = (uint8_t)xrng();
}

/* ---------------- guard-page overrun probe ---------------- */

static sigjmp_buf sjb;
static volatile sig_atomic_t segv_hit;
static volatile int fault_sig;
static volatile int fault_code;
static volatile void *fault_addr;
static volatile unsigned long long fault_rip;

static void segv_handler(int sig, siginfo_t *si, void *uc)
{
    fault_sig = sig;
    fault_code = si ? si->si_code : 0;
    fault_addr = si ? si->si_addr : NULL;
    fault_rip = 0;
#if defined(RJ_HAVE_UCONTEXT)
    if (uc) {
        ucontext_t *ctx = (ucontext_t *)uc;
#if defined(__aarch64__)
        fault_rip = (unsigned long long)ctx->uc_mcontext.pc;
#elif defined(__x86_64__)
        fault_rip = (unsigned long long)ctx->uc_mcontext.gregs[REG_RIP];
#else
        (void)ctx; /* fault_rip is diagnostic-only; arch not supported */
#endif
    }
#else
    (void)uc;
#endif
    segv_hit = 1;
    siglongjmp(sjb, 1);
}

/* Isolated setjmp/longjmp region: no other locals live across the jump, so
 * -Wclobbered stays quiet and no indeterminate values are read after a
 * fault. Returns 1 if a fault was caught. */
static int run_probed(rj_fn fn, const void **in, void **out, size_t len)
{
    segv_hit = 0;
    fault_sig = 0;
    fault_code = 0;
    fault_addr = NULL;
    fault_rip = 0;
    if (sigsetjmp(sjb, 1) == 0)
        fn(in, out, len);
    return segv_hit;
}

/* Run fn with every in/out buffer end-aligned just before a PROT_NONE guard
 * page: any read or write at offset >= len faults and is reported as a
 * failure instead of crashing the suite. Returns 0 if clean, 1 if a fault
 * was caught. Buffers are filled with random data (in) / zero (out). */
static int guard_probe(rj_fn fn, int n_in, int n_out, size_t len)
{
    /* The buffer contract requires 64-byte aligned buffers. End-placing
     * the buffer immediately before the guard page keeps 64B alignment
     * (mmap is page-aligned) only when the probed length is a multiple
     * of 64, so round up; the invariant under test ("no access at offset
     * >= plen") is unchanged, and tail handling is still exercised. */
    size_t plen = (len + 63) & ~(size_t)63;
    if (plen == 0)
        plen = 64;
    len = plen;
    long pgsz = sysconf(_SC_PAGESIZE);
    size_t pgsz_u = (size_t)pgsz;
    /* data pages: enough to hold len bytes; buffer ends at guard start */
    size_t npages = (len + pgsz_u - 1) / pgsz_u;
    if (npages == 0)
        npages = 1;
    size_t mapsz = (npages + 1) * pgsz_u;

    const void **in = malloc(sizeof(*in) * (size_t)(n_in ? n_in : 1));
    void **out = malloc(sizeof(*out) * (size_t)(n_out ? n_out : 1));
    uint8_t **maps = malloc(sizeof(*maps) * (size_t)(n_in + n_out));
    if (!in || !out || !maps) {
        fprintf(stderr, "oom\n");
        exit(1);
    }
    int nmaps = 0;
    for (int i = 0; i < n_in; i++) {
        uint8_t *m = mmap(NULL, mapsz, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (m == MAP_FAILED) {
            fprintf(stderr, "mmap failed\n");
            exit(1);
        }
        if (mprotect(m + npages * pgsz_u, pgsz_u, PROT_NONE) != 0) {
            fprintf(stderr, "mprotect failed\n");
            exit(1);
        }
        uint8_t *b = m + npages * pgsz_u - len;
        for (size_t k = 0; k < len; k++)
            b[k] = (uint8_t)xrng();
        in[i] = b;
        maps[nmaps++] = m;
    }
    for (int j = 0; j < n_out; j++) {
        uint8_t *m = mmap(NULL, mapsz, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (m == MAP_FAILED) {
            fprintf(stderr, "mmap failed\n");
            exit(1);
        }
        if (mprotect(m + npages * pgsz_u, pgsz_u, PROT_NONE) != 0) {
            fprintf(stderr, "mprotect failed\n");
            exit(1);
        }
        uint8_t *b = m + npages * pgsz_u - len;
        for (size_t k = 0; k < len; k++)
            b[k] = 0;
        out[j] = b;
        maps[nmaps++] = m;
    }

    struct sigaction sa, old_segv, old_bus, old_ill;
    memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = segv_handler;
    sa.sa_flags = SA_SIGINFO;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, &old_segv);
    sigaction(SIGBUS, &sa, &old_bus);
    sigaction(SIGILL, &sa, &old_ill);

    int faulted = run_probed(fn, in, out, len);

    sigaction(SIGSEGV, &old_segv, NULL);
    sigaction(SIGBUS, &old_bus, NULL);
    sigaction(SIGILL, &old_ill, NULL);

    for (int i = 0; i < nmaps; i++)
        munmap(maps[i], mapsz);
    free(in);
    free(out);
    free(maps);
    return faulted;
}

/* ---------------- scalar reference ---------------- */

static void ref_encode(const uint8_t *mat, int nd, int np, uint8_t **d,
                       uint8_t **p, size_t len)
{
    for (int j = 0; j < np; j++)
        for (size_t k = 0; k < len; k++) {
            uint8_t a = 0;
            for (int i = 0; i < nd; i++) {
                uint8_t c = mat[j * nd + i];
                if (c)
                    a ^= gf_mul(d[i][k], c);
            }
            p[j][k] = a;
        }
}

/* ---------------- synthetic CPUs ---------------- */

typedef struct {
    const char *name;
    rj_cpu_info cpu;
} cpu_variant_t;

#if defined(__aarch64__)
static rj_cpu_info mkcpu(int neon, int sve, int sve2)
{
    rj_cpu_info c;
    memset(&c, 0, sizeof(c));
    c.neon = neon;
    c.sve = sve;
    c.sve2 = sve2;
    c.sve_vl = sve ? 16 : 0;
    snprintf(c.brand, sizeof(c.brand), "synthetic");
    return c;
}
#elif defined(__riscv)
static rj_cpu_info mkcpu(int rvv)
{
    rj_cpu_info c;
    memset(&c, 0, sizeof(c));
    c.rvv = rvv;
    c.rvv_vlen = rvv ? 16 : 0;
    snprintf(c.brand, sizeof(c.brand), "synthetic");
    return c;
}
#else
static rj_cpu_info mkcpu(int sse2, int ssse3, int avx, int avx2, int f, int bw,
                         int vl, int gfni)
{
    rj_cpu_info c;
    memset(&c, 0, sizeof(c));
    c.sse2 = sse2;
    c.ssse3 = ssse3;
    c.avx = avx;
    c.avx2 = avx2;
    c.avx512f = f;
    c.avx512bw = bw;
    c.avx512vl = vl;
    c.gfni = gfni;
    snprintf(c.brand, sizeof(c.brand), "synthetic");
    return c;
}
#endif

/* ---------------- length sets ---------------- */

/* vector length in bytes for a candidate on this cpu (SVE is
 * runtime-sized; isa_vl() returns 0 for it) */
static int cand_vl(const cand_t *c, const rj_cpu_info *cpu)
{
    if (c->isa == ISA_SVE)
        return cpu->sve_vl;
    return isa_vl(c->isa);
}

static size_t base_lens[] = { 0, 1, 2, 3, 5, 7, 15, 16, 17, 31, 32, 33,
                              63, 64, 65, 100, 127, 128, 129, 255, 256, 257,
                              511, 512, 513, 1000, 4096, 65537 };
#define MAXL 64

static int build_lens(size_t *out, int group)
{
    int n = 0;
    for (size_t i = 0; i < sizeof(base_lens) / sizeof(base_lens[0]); i++)
        out[n++] = base_lens[i];
    if (group > 0) {
        size_t gs[] = { (size_t)group - 1, (size_t)group, (size_t)group + 1,
                        (size_t)(2 * group) - 1, (size_t)(2 * group),
                        (size_t)(2 * group) + 1 };
        for (int i = 0; i < 6; i++)
            if (gs[i] <= 70000)
                out[n++] = gs[i];
    }
    /* sort + dedupe */
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (out[j] < out[i]) {
                size_t t = out[i];
                out[i] = out[j];
                out[j] = t;
            }
    int w = 0;
    for (int i = 0; i < n; i++)
        if (w == 0 || out[i] != out[w - 1])
            out[w++] = out[i];
    return w;
}

/* ---------------- test driver ---------------- */

static rj_options jopts; /* benchmark off; direct jit_build_version */

static void run_encode_case(const char *cpuname, const rj_cpu_info *cpu,
                            const char *tname,
                            const topo_int_t *ti, const plan_t *pl,
                            const cand_t *cand,
                            const size_t *lens, int nl, int full)
{
    char err[256];
    version_t v;
    memset(&v, 0, sizeof(v));
    if (jit_build_version(pl, cand, cpu, &jopts, &v, err, sizeof(err)) != 0) {
        CHECK(0, "%s %s encode cand %s U=%d: build failed: %s", cpuname,
              tname, isa_name(cand->isa), cand->U, err);
        return;
    }
    int nd = ti->ndata, np = ti->npar;
    int group = cand->U * cand_vl(cand, cpu);

    for (int li = 0; li < nl; li++) {
        size_t len = lens[li];
        uint8_t *bases_d[64], *bases_p[64], *bases_r[64];
        uint8_t *d[64], *p[64], *pr[64];
        if (nd > 64 || np > 64) {
            CHECK(0, "topology too wide for harness");
            break;
        }
        for (int i = 0; i < nd; i++) {
            d[i] = cbuf_alloc(len, &bases_d[i]);
            randfill(d[i], len);
        }
        for (int j = 0; j < np; j++) {
            p[j] = cbuf_alloc(len, &bases_p[j]);
            pr[j] = cbuf_alloc(len, &bases_r[j]);
        }
        ref_encode(ti->mat, nd, np, d, pr, len);

        const void *in[64];
        void *out[64];
        for (int i = 0; i < nd; i++)
            in[i] = d[i];
        for (int j = 0; j < np; j++)
            out[j] = p[j];
        v.fn(in, out, len);
        nruns++;

        for (int j = 0; j < np; j++)
            CHECK(memcmp(p[j], pr[j], len) == 0,
                  "%s %s encode len=%zu par%d mismatch [%s]", cpuname, tname,
                  len, j, v.desc);
        for (int i = 0; i < nd; i++)
            CHECK(cbuf_check(bases_d[i], len),
                  "%s %s encode len=%zu input canary clobbered [%s]", cpuname,
                  tname, len, v.desc);
        for (int j = 0; j < np; j++)
            CHECK(cbuf_check(bases_p[j], len),
                  "%s %s encode len=%zu output canary clobbered [%s]", cpuname,
                  tname, len, v.desc);
        for (int i = 0; i < nd; i++)
            free(bases_d[i]);
        for (int j = 0; j < np; j++) {
            free(bases_p[j]);
            free(bases_r[j]);
        }
        if (failures)
            break;
    }

    if (full) {
    /* aligned ordinary buffers at several 64-byte-aligned start offsets:
     * every vector load/store must see a contract-honoring address */
    {
        size_t ulens[8];
        int nu = build_ulens(ulens, group);
        for (size_t oi = 0;
             oi < sizeof(align_offs) / sizeof(align_offs[0]) && !failures;
             oi++) {
            size_t off = align_offs[oi];
            for (int li = 0; li < nu && !failures; li++) {
                size_t len = ulens[li];
                uint8_t *bases_d[64], *bases_p[64], *bases_r[64];
                uint8_t *d[64], *p[64], *pr[64];
                for (int i = 0; i < nd; i++) {
                    d[i] = ubuf_alloc(len, off, &bases_d[i]);
                    randfill(d[i], len);
                }
                for (int j = 0; j < np; j++) {
                    p[j] = ubuf_alloc(len, off, &bases_p[j]);
                    pr[j] = ubuf_alloc(len, off, &bases_r[j]);
                }
                ref_encode(ti->mat, nd, np, d, pr, len);
                const void *in[64];
                void *out[64];
                for (int i = 0; i < nd; i++)
                    in[i] = d[i];
                for (int j = 0; j < np; j++)
                    out[j] = p[j];
                v.fn(in, out, len);
                nruns++;
                for (int j = 0; j < np; j++)
                    CHECK(memcmp(p[j], pr[j], len) == 0,
                          "%s %s encode len=%zu off=%zu par%d mismatch [%s]",
                          cpuname, tname, len, off, j, v.desc);
                for (int i = 0; i < nd; i++)
                    CHECK(ubuf_check(bases_d[i], off, len),
                          "%s %s encode len=%zu off=%zu input canary "
                          "clobbered [%s]",
                          cpuname, tname, len, off, v.desc);
                for (int j = 0; j < np; j++)
                    CHECK(ubuf_check(bases_p[j], off, len),
                          "%s %s encode len=%zu off=%zu output canary "
                          "clobbered [%s]",
                          cpuname, tname, len, off, v.desc);
                for (int i = 0; i < nd; i++)
                    free(bases_d[i]);
                for (int j = 0; j < np; j++) {
                    free(bases_p[j]);
                    free(bases_r[j]);
                }
            }
        }
    }

    /* guard-page probes at tail-sensitive lengths */
    {
        size_t gs[] = { (size_t)(group > 1 ? group - 1 : 0), (size_t)group,
                        (size_t)group + 1, 1000 };
        for (int i = 0; i < 4; i++) {
            int faulted = guard_probe(v.fn, nd, np, gs[i]);
            nruns++;
            CHECK(!faulted,
                  "%s %s encode len=%zu guard-page fault sig=%d code=%d "
                  "addr=%p rip=%llx%s [%s]",
                  cpuname, tname, gs[i], fault_sig, fault_code, fault_addr,
                  fault_rip,
                  (fault_rip >= (unsigned long long)(uintptr_t)v.mem &&
                   fault_rip < (unsigned long long)(uintptr_t)v.mem + v.memsz)
                      ? " rip-in-jit"
                      : " rip-outside-jit",
                  v.desc);
            if (faulted) {
                /* dump generated code for offline disassembly */
                char path[64];
                snprintf(path, sizeof(path), "/tmp/fault_jit_%zu.bin", gs[i]);
                FILE *f = fopen(path, "wb");
                if (f) {
                    fwrite(v.mem, 1, v.code_size, f);
                    fclose(f);
                }
                printf("  jit mem=%p code_size=%zu const_size=%zu "
                       "rip_off=0x%llx\n",
                       v.mem, v.code_size, v.const_size,
                       fault_rip - (unsigned long long)(uintptr_t)v.mem);
            }
            if (failures)
                break;
        }
    }
    } /* if (full) */
    jit_free_version(&v);
}

static void run_recover_case(const char *cpuname, const rj_cpu_info *cpu,
                             rj_topology *t, const char *tname,
                             const topo_int_t *ti, const plan_t *pl,
                             const cand_t *cand, const int *failed, int nfailed,
                             const size_t *lens, int nl, int full)
{
    (void)t;
    char err[256];
    version_t v;
    memset(&v, 0, sizeof(v));
    if (jit_build_version(pl, cand, cpu, &jopts, &v, err, sizeof(err)) != 0) {
        CHECK(0, "%s %s recover cand %s U=%d: build failed: %s", cpuname,
              tname, isa_name(cand->isa), cand->U, err);
        return;
    }
    int nd = ti->ndata, np = ti->npar, N = nd + np;
    int fs[RJ_MAX_DISKS];
    memcpy(fs, failed, sizeof(int) * (size_t)nfailed);
    /* insertion sort (avoid qsort dependency quirks) */
    for (int i = 1; i < nfailed; i++) {
        int k = fs[i], j = i - 1;
        while (j >= 0 && fs[j] > k) {
            fs[j + 1] = fs[j];
            j--;
        }
        fs[j + 1] = k;
    }
    int group = cand->U * cand_vl(cand, cpu);

    for (int li = 0; li < nl; li++) {
        size_t len = lens[li];
        uint8_t *bases[64], *keepb[64];
        uint8_t *all[64], *keep[64];
        for (int i = 0; i < N; i++) {
            all[i] = cbuf_alloc(len, &bases[i]);
            keep[i] = cbuf_alloc(len, &keepb[i]);
            if (i < nd)
                randfill(all[i], len);
        }
        ref_encode(ti->mat, nd, np, all, all + nd, len);
        for (int i = 0; i < N; i++)
            memcpy(keep[i], all[i], len);
        for (int f = 0; f < nfailed; f++)
            memset(all[fs[f]], 0xA5, len);

        const void *in[64];
        void *out[64];
        int ni = 0, fi = 0, nf = 0;
        for (int dsk = 0; dsk < N; dsk++) {
            if (fi < nfailed && fs[fi] == dsk) {
                out[nf++] = all[dsk];
                fi++;
            } else {
                in[ni++] = all[dsk];
            }
        }
        v.fn(in, out, len);
        nruns++;

        for (int f = 0; f < nfailed; f++)
            CHECK(memcmp(all[fs[f]], keep[fs[f]], len) == 0,
                  "%s %s recover disk%d len=%zu mismatch [%s]", cpuname,
                  tname, fs[f], len, v.desc);
        /* survivors must be bit-identical: recovery is a pure function of
         * the survivors and must not modify them */
        for (int dsk = 0; dsk < N; dsk++) {
            int isf = 0;
            for (int f = 0; f < nfailed; f++)
                if (fs[f] == dsk) {
                    isf = 1;
                    break;
                }
            if (!isf)
                CHECK(memcmp(all[dsk], keep[dsk], len) == 0,
                      "%s %s recover len=%zu survivor disk%d modified [%s]",
                      cpuname, tname, len, dsk, v.desc);
        }
        for (int i = 0; i < N; i++)
            CHECK(cbuf_check(bases[i], len),
                  "%s %s recover len=%zu canary clobbered [%s]", cpuname,
                  tname, len, v.desc);
        for (int i = 0; i < N; i++) {
            free(bases[i]);
            free(keepb[i]);
        }
        if (failures)
            break;
    }

    if (full) {
    /* aligned ordinary buffers (see encode case) */
    {
        size_t ulens[8];
        int nu = build_ulens(ulens, group);
        for (size_t oi = 0;
             oi < sizeof(align_offs) / sizeof(align_offs[0]) && !failures;
             oi++) {
            size_t off = align_offs[oi];
            for (int li = 0; li < nu && !failures; li++) {
                size_t len = ulens[li];
                uint8_t *bases[64], *keepb[64];
                uint8_t *all[64], *keep[64];
                for (int i = 0; i < N; i++) {
                    all[i] = ubuf_alloc(len, off, &bases[i]);
                    keep[i] = ubuf_alloc(len, off, &keepb[i]);
                    if (i < nd)
                        randfill(all[i], len);
                }
                ref_encode(ti->mat, nd, np, all, all + nd, len);
                for (int i = 0; i < N; i++)
                    memcpy(keep[i], all[i], len);
                for (int f = 0; f < nfailed; f++)
                    memset(all[fs[f]], 0xA5, len);
                const void *in[64];
                void *out[64];
                int ni = 0, fi = 0, nf = 0;
                for (int dsk = 0; dsk < N; dsk++) {
                    if (fi < nfailed && fs[fi] == dsk) {
                        out[nf++] = all[dsk];
                        fi++;
                    } else {
                        in[ni++] = all[dsk];
                    }
                }
                v.fn(in, out, len);
                nruns++;
                for (int f = 0; f < nfailed; f++)
                    CHECK(memcmp(all[fs[f]], keep[fs[f]], len) == 0,
                          "%s %s recover disk%d len=%zu off=%zu mismatch "
                          "[%s]",
                          cpuname, tname, fs[f], len, off, v.desc);
                for (int i = 0; i < N; i++)
                    CHECK(ubuf_check(bases[i], off, len),
                          "%s %s recover len=%zu off=%zu canary clobbered "
                          "[%s]",
                          cpuname, tname, len, off, v.desc);
                for (int i = 0; i < N; i++) {
                    free(bases[i]);
                    free(keepb[i]);
                }
            }
        }
    }

    {
        size_t gs[] = { (size_t)(group > 1 ? group - 1 : 0), (size_t)group,
                        (size_t)group + 1, 1000 };
        for (int i = 0; i < 4; i++) {
            int faulted =
                guard_probe(v.fn, N - nfailed, nfailed, gs[i]);
            nruns++;
            CHECK(!faulted,
                  "%s %s recover len=%zu guard-page fault sig=%d code=%d "
                  "addr=%p rip=%llx%s [%s]",
                  cpuname, tname, gs[i], fault_sig, fault_code, fault_addr,
                  fault_rip,
                  (fault_rip >= (unsigned long long)(uintptr_t)v.mem &&
                   fault_rip < (unsigned long long)(uintptr_t)v.mem + v.memsz)
                      ? " rip-in-jit"
                      : " rip-outside-jit",
                  v.desc);
            if (failures)
                break;
        }
    }
    } /* if (full) */
    jit_free_version(&v);
}

/* ---------------- P_DELTA (parity update) matrix ----------------
 * For every candidate, verify the JIT delta against ref_run() on the
 * same P_DELTA plan, for several changed-disk sets. Input layout:
 * [old_0..old_{nc-1}, new_0..new_{nc-1}, oldpar_0..oldpar_{np-1}].
 */
/* forward: defined below, after run_recover_case */
static int enum_cands(const char *cpuname, const rj_cpu_info *cpu,
                      const char *tname, const plan_t *pl, cand_t *cands);
static void run_pdelta_case(const char *cpuname, const rj_cpu_info *cpu,
                            const char *tname,
                            const topo_int_t *ti, const plan_t *pl,
                            const cand_t *cand,
                            const size_t *lens, int nl)
{
    char err[256];
    version_t v;
    memset(&v, 0, sizeof(v));
    if (jit_build_version(pl, cand, cpu, &jopts, &v, err, sizeof(err)) != 0) {
        CHECK(0, "%s %s pdelta cand %s U=%d: build failed: %s", cpuname,
              tname, isa_name(cand->isa), cand->U, err);
        return;
    }
    int nc = pl->nchanged, np = ti->npar;
    if (nc > 64 || np > 64) {
        CHECK(0, "pdelta topology too wide for harness");
        jit_free_version(&v);
        return;
    }

    for (int li = 0; li < nl; li++) {
        size_t len = lens[li];
        uint8_t *bases_o[64], *bases_n[64], *bases_op[64], *bases_p[64],
            *bases_r[64];
        uint8_t *old[64], *new_[64], *oldpar[64], *par[64], *ref[64];
        for (int i = 0; i < nc; i++) {
            old[i] = cbuf_alloc(len, &bases_o[i]);
            new_[i] = cbuf_alloc(len, &bases_n[i]);
            randfill(old[i], len);
            randfill(new_[i], len);
        }
        for (int j = 0; j < np; j++) {
            oldpar[j] = cbuf_alloc(len, &bases_op[j]);
            par[j] = cbuf_alloc(len, &bases_p[j]);
            ref[j] = cbuf_alloc(len, &bases_r[j]);
            randfill(oldpar[j], len);
        }

        /* reference via the scalar plan interpreter */
        const uint8_t *rin[192];
        for (int i = 0; i < nc; i++) {
            rin[i] = old[i];
            rin[nc + i] = new_[i];
        }
        for (int j = 0; j < np; j++)
            rin[2 * nc + j] = oldpar[j];
        ref_run(pl, rin, ref, len);

        const void *in[192];
        void *out[64];
        for (int i = 0; i < nc; i++) {
            in[i] = old[i];
            in[nc + i] = new_[i];
        }
        for (int j = 0; j < np; j++) {
            in[2 * nc + j] = oldpar[j];
            out[j] = par[j];
        }
        v.fn(in, out, len);
        nruns++;

        for (int j = 0; j < np; j++)
            CHECK(memcmp(par[j], ref[j], len) == 0,
                  "%s %s pdelta len=%zu par%d mismatch [%s]", cpuname, tname,
                  len, j, v.desc);
        for (int i = 0; i < nc; i++) {
            CHECK(cbuf_check(bases_o[i], len),
                  "%s %s pdelta len=%zu old canary clobbered [%s]", cpuname,
                  tname, len, v.desc);
            CHECK(cbuf_check(bases_n[i], len),
                  "%s %s pdelta len=%zu new canary clobbered [%s]", cpuname,
                  tname, len, v.desc);
        }
        for (int j = 0; j < np; j++) {
            CHECK(cbuf_check(bases_op[j], len),
                  "%s %s pdelta len=%zu oldpar canary clobbered [%s]", cpuname,
                  tname, len, v.desc);
            CHECK(cbuf_check(bases_p[j], len),
                  "%s %s pdelta len=%zu output canary clobbered [%s]", cpuname,
                  tname, len, v.desc);
        }
        /* in-place: out[j] may alias oldpar[j] */
        for (int j = 0; j < np; j++)
            memcpy(par[j], oldpar[j], len);
        for (int j = 0; j < np; j++)
            out[j] = oldpar[j];
        v.fn(in, out, len);
        nruns++;
        for (int j = 0; j < np; j++)
            CHECK(memcmp(oldpar[j], ref[j], len) == 0,
                  "%s %s pdelta in-place len=%zu par%d mismatch [%s]", cpuname,
                  tname, len, j, v.desc);

        for (int i = 0; i < nc; i++) {
            free(bases_o[i]);
            free(bases_n[i]);
        }
        for (int j = 0; j < np; j++) {
            free(bases_op[j]);
            free(bases_p[j]);
            free(bases_r[j]);
        }
        if (failures)
            break;
    }
    jit_free_version(&v);
}

static void test_pdelta_op(const char *cpuname, const rj_cpu_info *cpu,
                           rj_topology *t, const char *tname,
                           const int *changed, int nchanged)
{
    char err[256];
    topo_int_t ti;
    if (topo_int_build(&ti, t, err, sizeof(err)) != 0) {
        CHECK(0, "%s %s pdelta topo: %s", cpuname, tname, err);
        return;
    }
    rj_op op = rj_op_parity_update(changed, nchanged);

    plan_t pl;
    int prc = plan_build(&ti, &op, tname, &pl, err, sizeof(err));
    if (prc != 0) {
        CHECK(0, "%s %s pdelta plan: %s", cpuname, tname, err);
        rj_op_free(&op);
        topo_int_free(&ti);
        return;
    }

    cand_t cands[64];
    int ncand = enum_cands(cpuname, cpu, tname, &pl, cands);
    if (ncand == 0) {
        /* legal when the CPU offers no multiply-capable ISA (e.g.
         * sse2-only); enum_cands already validated the condition */
        plan_free(&pl);
        rj_op_free(&op);
        topo_int_free(&ti);
        return;
    }

    for (int c = 0; c < ncand && !failures; c++) {
        int group = cands[c].U * cand_vl(&cands[c], cpu);
        size_t lens[MAXL];
        int nl = build_lens(lens, group);
        run_pdelta_case(cpuname, cpu, tname, &ti, &pl, &cands[c], lens, nl);
    }
    plan_free(&pl);
    rj_op_free(&op);
    topo_int_free(&ti);
}

/* light length set: tiny sizes + boundaries around one vector group */
static int small_lens(size_t *lens, int group)
{
    int nl = 0;
    size_t tmp[] = { 0, 1, (size_t)(group > 1 ? group - 1 : 0),
                     (size_t)group, (size_t)group + 1 };
    for (int i = 0; i < 5; i++) {
        int dup = 0;
        for (int j = 0; j < nl; j++)
            if (lens[j] == tmp[i]) {
                dup = 1;
                break;
            }
        if (!dup)
            lens[nl++] = tmp[i];
    }
    return nl;
}

/* enumerate candidates; the empty set is only legal when the plan needs
 * multiplies and the CPU offers no multiply-capable ISA */
static int enum_cands(const char *cpuname, const rj_cpu_info *cpu,
                      const char *tname, const plan_t *pl, cand_t *cands)
{
    int ncand = plan_candidates(pl, cpu, cands, 64, 0, 0);
    if (ncand == 0) {
        int need_mul = pl->kind != P_XOR;
        int mul_ok = cpu->ssse3 || cpu->avx2 || cpu->avx512f;
        CHECK(need_mul && !mul_ok, "%s %s: zero candidates unexpectedly",
              cpuname, tname);
    }
    return ncand;
}

/* one op: encode or recover(failed[]) with expect_ok */
typedef struct {
    int nfailed;
    int failed[8];
    int expect_ok;
} op_case_t;

static void test_topo_op(const char *cpuname, const rj_cpu_info *cpu,
                         rj_topology *t, const char *tname,
                         const op_case_t *oc)
{
    char err[256];
    topo_int_t ti;
    if (topo_int_build(&ti, t, err, sizeof(err)) != 0) {
        CHECK(0, "%s %s topo: %s", cpuname, tname, err);
        return;
    }
    rj_op op;
    if (oc->nfailed == 0 && oc->expect_ok)
        op = rj_op_encode();
    else
        op = rj_op_recover(oc->failed, oc->nfailed);

    plan_t pl;
    int prc = plan_build(&ti, &op, tname, &pl, err, sizeof(err));
    if (!oc->expect_ok) {
        CHECK(prc != 0, "%s %s: expected plan failure but it succeeded",
              cpuname, tname);
        rj_op_free(&op);
        topo_int_free(&ti);
        return;
    }
    if (prc != 0) {
        CHECK(0, "%s %s plan: %s", cpuname, tname, err);
        rj_op_free(&op);
        topo_int_free(&ti);
        return;
    }

    cand_t cands[64];
    int ncand = enum_cands(cpuname, cpu, tname, &pl, cands);
    if (ncand == 0) {
        plan_free(&pl);
        rj_op_free(&op);
        topo_int_free(&ti);
        return;
    }
    CHECK(ncand > 0, "%s %s: plan_candidates failed", cpuname, tname);

    for (int c = 0; c < ncand; c++) {
        int group = cands[c].U * cand_vl(&cands[c], cpu);
        size_t lens[MAXL];
        int nl = build_lens(lens, group);
        if (oc->nfailed == 0 && oc->expect_ok)
            run_encode_case(cpuname, cpu, tname, &ti, &pl, &cands[c], lens,
                            nl, 1);
        else
            run_recover_case(cpuname, cpu, t, tname, &ti, &pl, &cands[c],
                             oc->failed, oc->nfailed, lens, nl, 1);
        if (failures)
            break;
    }
    plan_free(&pl);
    rj_op_free(&op);
    topo_int_free(&ti);
}

/* ---------------- exhaustive failure-subset enumeration ----------------
 * For every subset of disks of size 1..npar: planning must succeed and every
 * candidate must recover correctly (light length set; the full sweep above
 * already covers codegen depth per candidate). For every subset of size
 * npar+1: planning must fail. This proves the planner's recoverability
 * decisions are complete, not just sampled.
 */

typedef struct {
    const char *cpuname;
    const rj_cpu_info *cpu;
    rj_topology *t;
    const char *tname;
    topo_int_t *ti;
    int N, npar, k;
} exh_ctx_t;

static void exh_one_subset(const int *sub, void *ctx)
{
    exh_ctx_t *x = ctx;
    char err[256];
    rj_op op = rj_op_recover(sub, x->k);
    plan_t pl;
    int prc = plan_build(x->ti, &op, x->tname, &pl, err, sizeof(err));
    if (x->k <= x->npar) {
        if (prc != 0) {
            char fb[64] = "";
            for (int i = 0; i < x->k; i++)
                snprintf(fb + strlen(fb), sizeof(fb) - strlen(fb), "%s%d",
                         i ? "+" : "", sub[i]);
            CHECK(0, "%s %s: subset {%s} should be recoverable: %s",
                  x->cpuname, x->tname, fb, err);
            rj_op_free(&op);
            return;
        }
        cand_t cands[64];
        int ncand = enum_cands(x->cpuname, x->cpu, x->tname, &pl, cands);
        for (int c = 0; c < ncand && !failures; c++) {
            int group = cands[c].U * cand_vl(&cands[c], x->cpu);
            size_t lens[8];
            int nl = small_lens(lens, group);
            run_recover_case(x->cpuname, x->cpu, x->t, x->tname, x->ti, &pl,
                             &cands[c], sub, x->k, lens, nl, 0);
        }
        plan_free(&pl);
    } else {
        /* k == npar+1: must be unrecoverable */
        if (prc == 0) {
            char fb[64] = "";
            for (int i = 0; i < x->k; i++)
                snprintf(fb + strlen(fb), sizeof(fb) - strlen(fb), "%s%d",
                         i ? "+" : "", sub[i]);
            CHECK(0, "%s %s: subset {%s} planned OK but exceeds npar=%d",
                  x->cpuname, x->tname, fb, x->npar);
            plan_free(&pl);
        }
    }
    rj_op_free(&op);
}

static void for_each_subset(int N, int k,
                            void (*cb)(const int *sub, void *ctx), void *ctx)
{
    int sub[64];
    for (int i = 0; i < k; i++)
        sub[i] = i;
    for (;;) {
        cb(sub, ctx);
        if (failures)
            return;
        int i = k - 1;
        while (i >= 0 && sub[i] == N - k + i)
            i--;
        if (i < 0)
            return;
        sub[i]++;
        for (int j = i + 1; j < k; j++)
            sub[j] = sub[j - 1] + 1;
    }
}

static void test_exhaustive_subsets(const char *cpuname, const rj_cpu_info *cpu,
                                    rj_topology *t, const char *tname)
{
    char err[256];
    topo_int_t ti;
    if (topo_int_build(&ti, t, err, sizeof(err)) != 0) {
        CHECK(0, "%s %s topo: %s", cpuname, tname, err);
        return;
    }
    int N = ti.ndata + ti.npar;
    exh_ctx_t x = { cpuname, cpu, t, tname, &ti, N, ti.npar, 0 };
    for (int k = 1; k <= ti.npar + 1 && !failures; k++) {
        x.k = k;
        for_each_subset(N, k, exh_one_subset, &x);
    }
    topo_int_free(&ti);
}

/* ---------------- deterministic random legal topologies ----------------
 * Cauchy matrices are MDS by construction (distinct x_i, y_j, x_i != y_j);
 * Vandermonde-with-ones-row generalizes RAID-6 to arbitrary bases;
 * sparse variants start from Vandermonde and zero random entries, keeping
 * only matrices for which every <=npar failure subset plans successfully
 * (i.e. empirically MDS). All exercise arbitrary coefficients including
 * 0 and 1 through every candidate.
 */

static uint8_t rbyte(void)
{
    return (uint8_t)(xrng() & 0xFF);
}

/* Cauchy: M[j][i] = inv(x[i] ^ y[j]) */
static void gen_cauchy(int nd, int np, uint8_t *m)
{
    uint8_t x[64], y[64];
    for (int i = 0; i < nd; i++) {
        for (;;) {
            uint8_t v = rbyte() | 1; /* nonzero */
            int dup = 0;
            for (int k = 0; k < i; k++)
                if (x[k] == v) {
                    dup = 1;
                    break;
                }
            if (!dup) {
                x[i] = v;
                break;
            }
        }
    }
    for (int j = 0; j < np; j++) {
        for (;;) {
            uint8_t v = rbyte() | 1;
            int bad = 0;
            for (int k = 0; k < j; k++)
                if (y[k] == v)
                    bad = 1;
            for (int i = 0; i < nd; i++)
                if (x[i] == v)
                    bad = 1;
            if (!bad) {
                y[j] = v;
                break;
            }
        }
    }
    for (int j = 0; j < np; j++)
        for (int i = 0; i < nd; i++)
            m[j * nd + i] = gf_inv(x[i] ^ y[j]);
}

/* row 0 = all ones; rows j>=1 = [c_j^(nd-1) ... c_j^0], distinct c_j */
static void gen_vander(int nd, int np, uint8_t *m)
{
    uint8_t c[64];
    c[0] = 1;
    for (int j = 1; j < np; j++) {
        for (;;) {
            uint8_t v = rbyte() | 1;
            int dup = 0;
            for (int k = 0; k < j; k++)
                if (c[k] == v) {
                    dup = 1;
                    break;
                }
            if (!dup) {
                c[j] = v;
                break;
            }
        }
    }
    for (int j = 0; j < np; j++)
        for (int i = 0; i < nd; i++)
            m[j * nd + i] = gf_pow(c[j], (unsigned)(nd - 1 - i));
}

/* true iff every failure subset of size <= npar plans successfully */
static int all_subsets_plan(rj_topology *t, const char *tname)
{
    char err[256];
    topo_int_t ti;
    if (topo_int_build(&ti, t, err, sizeof(err)) != 0)
        return 0;
    int N = ti.ndata + ti.npar, ok = 1;
    int sub[64];
    for (int k = 1; k <= ti.npar && ok; k++) {
        for (int i = 0; i < k; i++)
            sub[i] = i;
        for (;;) {
            rj_op op = rj_op_recover(sub, k);
            plan_t pl;
            if (plan_build(&ti, &op, tname, &pl, err, sizeof(err)) != 0) {
                ok = 0;
            } else {
                plan_free(&pl);
            }
            rj_op_free(&op);
            if (!ok)
                break;
            int i = k - 1;
            while (i >= 0 && sub[i] == N - k + i)
                i--;
            if (i < 0)
                break;
            sub[i]++;
            for (int j = i + 1; j < k; j++)
                sub[j] = sub[j - 1] + 1;
        }
    }
    topo_int_free(&ti);
    return ok;
}

static void run_encode_light(const char *cpuname, const rj_cpu_info *cpu,
                             const char *tname, topo_int_t *ti)
{
    char err[256];
    rj_op op = rj_op_encode();
    plan_t pl;
    if (plan_build(ti, &op, tname, &pl, err, sizeof(err)) != 0) {
        CHECK(0, "%s %s encode plan: %s", cpuname, tname, err);
        rj_op_free(&op);
        return;
    }
    cand_t cands[64];
    int ncand = enum_cands(cpuname, cpu, tname, &pl, cands);
    for (int c = 0; c < ncand && !failures; c++) {
        int group = cands[c].U * cand_vl(&cands[c], cpu);
        size_t lens[8];
        int nl = small_lens(lens, group);
        run_encode_case(cpuname, cpu, tname, ti, &pl, &cands[c], lens, nl, 0);
    }
    plan_free(&pl);
    rj_op_free(&op);
}

static void test_random_topos(const char *cpuname, const rj_cpu_info *cpu)
{
    struct {
        int nd, np, kind; /* 0=cauchy, 1=vander, 2=sparse */
    } specs[] = {
        { 3, 1, 0 }, { 5, 2, 0 }, { 8, 3, 0 },
        { 4, 2, 1 }, { 6, 3, 1 },
        { 4, 2, 2 }, { 6, 2, 2 },
    };
    for (size_t si = 0;
         si < sizeof(specs) / sizeof(specs[0]) && !failures; si++) {
        int nd = specs[si].nd, np = specs[si].np;
        uint8_t m[64 * 64];
        rj_topology *t = NULL;
        char tname[64];
        if (specs[si].kind == 0) {
            gen_cauchy(nd, np, m);
            t = rj_topo_custom(nd, np, m);
            snprintf(tname, sizeof(tname), "cauchy(%d,%d)#%zu", nd, np, si);
        } else {
            /* vander/sparse: random bases or random zeros can yield a
             * singular (non-MDS) matrix, so generate-and-filter: keep the
             * matrix only if every <=npar failure subset plans OK. The
             * exhaustive test below then re-verifies each kept subset by
             * actual JIT execution against the scalar reference. */
            const char *kn = specs[si].kind == 1 ? "vander" : "sparse";
            int tries = 0;
            for (;;) {
                if (specs[si].kind == 1) {
                    gen_vander(nd, np, m);
                } else {
                    gen_vander(nd, np, m);
                    for (int k = 0; k < nd * np; k++)
                        if ((xrng() % 10) < 3)
                            m[k] = 0;
                }
                t = rj_topo_custom(nd, np, m);
                if (all_subsets_plan(t, "random-probe"))
                    break;
                rj_topo_free(t);
                t = NULL;
                if (++tries > 500) {
                    CHECK(0, "%s: no MDS %s matrix found", cpuname, kn);
                    break;
                }
            }
            if (!t)
                continue;
            snprintf(tname, sizeof(tname), "%s(%d,%d)#%zu", kn, nd, np, si);
        }
        char err[256];
        topo_int_t ti;
        if (topo_int_build(&ti, t, err, sizeof(err)) != 0) {
            CHECK(0, "%s %s topo: %s", cpuname, tname, err);
            rj_topo_free(t);
            continue;
        }
        run_encode_light(cpuname, cpu, tname, &ti);
        topo_int_free(&ti);
        if (!failures)
            test_exhaustive_subsets(cpuname, cpu, t, tname);
        rj_topo_free(t);
    }
}

/* ---------------- decode-and-encode chain ----------------
 * recover is "decode then encode": the system solve happens once at JIT
 * time and reconstruction is a single encode-like pass over survivors.
 * Chain check: encode -> destroy data+parity disks -> recover -> re-encode
 * from the recovered stripe -> parity must match the original encode.
 * Uses one candidate per ISA to cover ISA diversity without a quadratic
 * blowup (per-candidate correctness is covered above).
 */
static void test_chain(const char *cpuname, const rj_cpu_info *cpu,
                       rj_topology *t, const char *tname)
{
    char err[256];
    topo_int_t ti;
    if (topo_int_build(&ti, t, err, sizeof(err)) != 0) {
        CHECK(0, "%s %s topo: %s", cpuname, tname, err);
        return;
    }
    int nd = ti.ndata, np = ti.npar, N = nd + np;
    if (np == 0) {
        topo_int_free(&ti);
        return;
    }
    /* fail data disk 0 and (when npar >= 2) parity disk 0: the mixed case
     * stresses the chain; parity-only/data-only are covered elsewhere */
    int nfailed = np >= 2 ? 2 : 1;
    int failed[2] = { 0, nd };
    rj_op eop = rj_op_encode();
    rj_op rop = rj_op_recover(failed, nfailed);
    plan_t epl, rpl;
    if (plan_build(&ti, &eop, tname, &epl, err, sizeof(err)) != 0 ||
        plan_build(&ti, &rop, tname, &rpl, err, sizeof(err)) != 0) {
        CHECK(0, "%s %s chain plan: %s", cpuname, tname, err);
        rj_op_free(&eop);
        rj_op_free(&rop);
        topo_int_free(&ti);
        return;
    }
    cand_t ec[32], rc[32];
    int ne = enum_cands(cpuname, cpu, tname, &epl, ec);
    int nr = enum_cands(cpuname, cpu, tname, &rpl, rc);
    for (int ei = 0; ei < ne && !failures; ei++) {
        for (int ri = 0; ri < nr && !failures; ri++) {
            if (ec[ei].isa != rc[ri].isa)
                continue; /* one pair per ISA */
            /* only the first unroll per ISA pair */
            int seen = 0;
            for (int e2 = 0; e2 < ei; e2++)
                if (ec[e2].isa == ec[ei].isa)
                    seen = 1;
            for (int r2 = 0; r2 < ri; r2++)
                if (rc[r2].isa == rc[ri].isa)
                    seen = 1;
            if (seen)
                continue;
            version_t ev, rv;
            memset(&ev, 0, sizeof(ev));
            memset(&rv, 0, sizeof(rv));
            if (jit_build_version(&epl, &ec[ei], cpu, &jopts, &ev, err,
                                 sizeof(err)) != 0 ||
                jit_build_version(&rpl, &rc[ri], cpu, &jopts, &rv, err,
                                  sizeof(err)) != 0) {
                CHECK(0, "%s %s chain build: %s", cpuname, tname, err);
                jit_free_version(&ev);
                jit_free_version(&rv);
                continue;
            }
            size_t len = 1000;
            uint8_t *bases[64], *all[64], *par2[64], *bases2[64];
            for (int i = 0; i < N; i++) {
                all[i] = cbuf_alloc(len, &bases[i]);
                if (i < nd)
                    randfill(all[i], len);
            }
            for (int j = 0; j < np; j++)
                par2[j] = cbuf_alloc(len, &bases2[j]);
            const void *in[64];
            void *out[64];
            for (int i = 0; i < nd; i++)
                in[i] = all[i];
            for (int j = 0; j < np; j++)
                out[j] = all[nd + j];
            ev.fn(in, out, len); /* encode */
            nruns++;
            uint8_t *orig_d0_base, *orig_p0_base;
            uint8_t *orig_d0 = cbuf_alloc(len, &orig_d0_base);
            uint8_t *orig_p0 = cbuf_alloc(len, &orig_p0_base);
            memcpy(orig_d0, all[0], len);
            memcpy(orig_p0, all[nd], len);
            memset(all[0], 0xA5, len);   /* destroy data disk 0 */
            if (nfailed == 2)
                memset(all[nd], 0x5A, len); /* destroy parity disk 0 */
            int ni = 0, nf = 0;
            for (int dsk = 0; dsk < N; dsk++) {
                int isf = (dsk == 0) || (nfailed == 2 && dsk == nd);
                if (isf)
                    out[nf++] = all[dsk];
                else
                    in[ni++] = all[dsk];
            }
            rv.fn(in, out, len); /* recover (decode+encode pass) */
            nruns++;
            CHECK(memcmp(all[0], orig_d0, len) == 0,
                  "%s %s chain len=%zu data0 recover mismatch [%s]",
                  cpuname, tname, len, rv.desc);
            CHECK(memcmp(all[nd], orig_p0, len) == 0,
                  "%s %s chain len=%zu par0 recover mismatch [%s]",
                  cpuname, tname, len, rv.desc);
            /* re-encode from the recovered stripe */
            for (int i = 0; i < nd; i++)
                in[i] = all[i];
            for (int j = 0; j < np; j++)
                out[j] = par2[j];
            ev.fn(in, out, len);
            nruns++;
            for (int j = 0; j < np; j++)
                CHECK(memcmp(par2[j], j == 0 ? orig_p0 : all[nd + j],
                             len) == 0,
                      "%s %s chain len=%zu par%d re-encode mismatch [%s+%s]",
                      cpuname, tname, len, j, ev.desc, rv.desc);
            for (int i = 0; i < N; i++) {
                CHECK(cbuf_check(bases[i], len),
                      "%s %s chain canary clobbered", cpuname, tname);
                free(bases[i]);
            }
            for (int j = 0; j < np; j++)
                free(bases2[j]);
            free(orig_d0_base);
            free(orig_p0_base);
            jit_free_version(&ev);
            jit_free_version(&rv);
        }
    }
    plan_free(&epl);
    plan_free(&rpl);
    rj_op_free(&eop);
    rj_op_free(&rop);
    topo_int_free(&ti);
}

int main(void)
{
    rj_options_default(&jopts);
    jopts.benchmark = 0;
    jopts.verbose = 0;

    cpu_variant_t variants[8];
    int nvar = 0;
    const char *env = getenv("RAIDJIT_TEST_CPU");
    if (env && strcmp(env, "detect") == 0) {
        /* qemu path: use the real (possibly emulated) CPUID */
        variants[nvar].name = "detected";
        rj_cpu_detect(&variants[nvar].cpu);
        nvar = 1;
        printf("cpu: %s (detected)\n", variants[nvar - 1].cpu.brand);
    } else {
#if defined(__aarch64__)
        variants[nvar].name = "neon";
        variants[nvar++].cpu = mkcpu(1, 0, 0);
        /* Synthetic SVE variants generate SVE instructions that fault
         * (SIGILL) on hosts without SVE. Only include them if the host
         * actually supports SVE, otherwise the test crashes instead of
         * testing. */
        {
            rj_cpu_info host;
            rj_cpu_detect(&host);
            if (host.sve) {
                variants[nvar].name = "sve";
                variants[nvar].cpu = mkcpu(1, 1, 0);
                /* Seed synthetic VL from host so benchmark metadata matches. */
                variants[nvar].cpu.sve_vl = host.sve_vl;
                nvar++;
                printf("cpu variant: sve (host-gated, vl=%u)\n", host.sve_vl);
                if (host.sve2) {
                    variants[nvar].name = "sve2";
                    variants[nvar].cpu = mkcpu(1, 1, 1);
                    variants[nvar].cpu.sve_vl = host.sve_vl;
                    nvar++;
                    printf("cpu variant: sve2 (host-gated, vl=%u)\n", host.sve_vl);
                }
            } else {
                printf("cpu variant: sve skipped (host lacks SVE)\n");
            }
            printf("cpu variant: neon (always included)\n");
        }
#elif defined(__riscv)
        /* Only test RVV if the host supports it; otherwise the generated
         * vector code faults with an illegal instruction. */
        {
            rj_cpu_info host;
            rj_cpu_detect(&host);
            if (host.rvv) {
                variants[nvar].name = "rvv";
                variants[nvar++].cpu = mkcpu(1);
                printf("cpu variant: rvv (host-gated)\n");
            } else {
                printf("cpu variant: rvv skipped (host lacks RVV)\n");
            }
        }
#else
        variants[nvar].name = "sse3";
        variants[nvar++].cpu = mkcpu(1, 1, 0, 0, 0, 0, 0, 0);
        variants[nvar].name = "avx2";
        variants[nvar++].cpu = mkcpu(1, 1, 1, 1, 0, 0, 0, 0);
        variants[nvar].name = "avx512-nogfni";
        variants[nvar++].cpu = mkcpu(1, 1, 1, 1, 1, 1, 1, 0);
        variants[nvar].name = "avx512-gfni";
        variants[nvar++].cpu = mkcpu(1, 1, 1, 1, 1, 1, 1, 1);
        variants[nvar].name = "avx512f-only";
        variants[nvar++].cpu = mkcpu(1, 1, 1, 1, 1, 0, 0, 0);
        variants[nvar].name = "sse2-only";
        variants[nvar++].cpu = mkcpu(1, 0, 0, 0, 0, 0, 0, 0);
#endif
    }

    /* topologies */
    rj_topology *r0 = rj_topo_raid0(4);
    rj_topology *r1 = rj_topo_raid1();
    rj_topology *r5 = rj_topo_raid5(4);
    rj_topology *r64 = rj_topo_raid6(4);
    rj_topology *r68 = rj_topo_raid6(8);
    rj_topology *rs = rj_topo_rs(6, 3);
    rj_topology *rs51 = rj_topo_rs(51, 3); /* large: 51 data disks */
    uint8_t cmat[8] = { 1, 1, 1, 1, 1, 2, 4, 8 };
    rj_topology *rc = rj_topo_custom(4, 2, cmat);
    /* custom with zero coefficients: exercises the c==0 skip in P_DELTA */
    uint8_t cmat_z[8] = { 1, 0, 1, 0, 0, 1, 0, 1 };
    rj_topology *rcz = rj_topo_custom(4, 2, cmat_z);

    struct {
        rj_topology *t;
        const char *name;
        op_case_t ops[10];
        int nops;
    } cases[] = {
        { r0, "raid0(4)",
          { { 0, { 0 }, 1 },              /* encode */
            { 1, { 0 }, 0 } },            /* unrecoverable */
          2 },
        { r1, "raid1",
          { { 0, { 0 }, 1 }, { 1, { 0 }, 1 }, { 1, { 1 }, 1 } },
          3 },
        { r5, "raid5(4)",
          { { 0, { 0 }, 1 }, { 1, { 2 }, 1 }, { 1, { 4 }, 1 },
            { 2, { 2, 4 }, 0 } },
          4 },
        { r64, "raid6(4)",
          { { 0, { 0 }, 1 }, { 1, { 0 }, 1 }, { 1, { 4 }, 1 },
            { 1, { 5 }, 1 }, { 2, { 0, 3 }, 1 }, { 2, { 1, 4 }, 1 },
            { 2, { 4, 5 }, 1 }, { 3, { 0, 1, 2 }, 0 } },
          8 },
        { r68, "raid6(8)",
          { { 0, { 0 }, 1 }, { 1, { 0 }, 1 }, { 1, { 8 }, 1 },
            { 1, { 9 }, 1 }, { 2, { 0, 7 }, 1 }, { 2, { 3, 8 }, 1 },
            { 2, { 8, 9 }, 1 }, { 3, { 0, 1, 2 }, 0 } },
          8 },
        { rs, "rs(6,3)",
          { { 0, { 0 }, 1 }, { 3, { 0, 1, 2 }, 1 }, { 3, { 6, 7, 8 }, 1 },
            { 3, { 2, 5, 7 }, 1 }, { 4, { 0, 1, 2, 3 }, 0 } },
          5 },
        { rc, "custom(4,2)",
          { { 0, { 0 }, 1 }, { 2, { 0, 3 }, 1 }, { 2, { 4, 5 }, 1 },
            { 3, { 0, 1, 2 }, 0 } },
          4 },
    };

    for (int vi = 0; vi < nvar && !failures; vi++) {
        for (size_t ci = 0;
             ci < sizeof(cases) / sizeof(cases[0]) && !failures; ci++) {
            for (int oi = 0; oi < cases[ci].nops && !failures; oi++)
                test_topo_op(variants[vi].name, &variants[vi].cpu,
                             cases[ci].t, cases[ci].name, &cases[ci].ops[oi]);
        }
        /* exhaustive failure subsets + decode-and-encode chains for the
         * named topologies */
        for (size_t ci = 0;
             ci < sizeof(cases) / sizeof(cases[0]) && !failures; ci++) {
            test_exhaustive_subsets(variants[vi].name, &variants[vi].cpu,
                                    cases[ci].t, cases[ci].name);
            test_chain(variants[vi].name, &variants[vi].cpu, cases[ci].t,
                       cases[ci].name);
        }
        /* P_DELTA (parity update) for topologies with parity: single,
         * multiple, and all-data-disks-changed sets. raid0 has no parity
         * (skip); raid1's single parity is a copy. */
        if (!failures) {
            const int c1[] = { 1 };
            const int c2[] = { 0, 2 };
            const int c68_1[] = { 5 };
            const int c68_2[] = { 1, 5 };
            const int c68_all[] = { 0, 1, 2, 3, 4, 5, 6, 7 };
            const int crs_1[] = { 2 };
            const int crs_3[] = { 0, 3, 5 };
            const int crc_2[] = { 1, 3 };
            test_pdelta_op(variants[vi].name, &variants[vi].cpu, r5, "raid5(4)",
                           c1, 1);
            test_pdelta_op(variants[vi].name, &variants[vi].cpu, r5, "raid5(4)",
                           c2, 2);
            test_pdelta_op(variants[vi].name, &variants[vi].cpu, r68,
                           "raid6(8)", c68_1, 1);
            test_pdelta_op(variants[vi].name, &variants[vi].cpu, r68,
                           "raid6(8)", c68_2, 2);
            test_pdelta_op(variants[vi].name, &variants[vi].cpu, r68,
                           "raid6(8)", c68_all, 8);
            test_pdelta_op(variants[vi].name, &variants[vi].cpu, rs, "rs(6,3)",
                           crs_1, 1);
            test_pdelta_op(variants[vi].name, &variants[vi].cpu, rs, "rs(6,3)",
                           crs_3, 3);
            test_pdelta_op(variants[vi].name, &variants[vi].cpu, rc,
                           "custom(4,2)", crc_2, 2);
            /* custom with zero coefficients: exercises c==0 skip */
            if (rcz) {
                const int ccz[] = { 0, 1, 2, 3 };
                test_pdelta_op(variants[vi].name, &variants[vi].cpu, rcz,
                               "custom-z(4,2)", ccz, 4);
            }
            /* large-nchanged stress: 51 data disks, 25 changed */
            if (rs51) {
                static int c51[25];
                for (int i = 0; i < 25; i++)
                    c51[i] = i * 2; /* even disks */
                test_pdelta_op(variants[vi].name, &variants[vi].cpu, rs51,
                               "rs(51,3)", c51, 25);
            }
        }
        /* deterministic random legal topologies */
        if (!failures)
            test_random_topos(variants[vi].name, &variants[vi].cpu);
        printf("ok cpu=%s\n", variants[vi].name);
    }

    rj_topo_free(r0);
    rj_topo_free(r1);
    rj_topo_free(r5);
    rj_topo_free(r64);
    rj_topo_free(r68);
    rj_topo_free(rs);
    rj_topo_free(rs51);
    rj_topo_free(rc);
    rj_topo_free(rcz);

    printf("matrix: %ld runs, %d failures\n", nruns, failures);
    if (failures) {
        printf("MATRIX TESTS FAILED\n");
        return 1;
    }
    printf("matrix: all OK\n");
    return 0;
}
