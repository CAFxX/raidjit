/* Extended P_DELTA (partial-stripe parity update) edge-case tests.
 *
 * Beyond test_pdelta.c's baseline (API validation, correctness vs
 * re-encode on small topologies, canaries, in-place, benchmark mode),
 * this file covers:
 *  1. nchanged == 51 (maximum data disks for the default RS matrix)
 *  2. 200+ disk custom Cauchy topology (rj_topo_custom)
 *  3. PROT_NONE guard pages around every buffer (no overrun/underrun)
 *  4. byte-for-byte input immutability on all edge paths
 *  5. aliasing contract: exact in-place out[j]==oldpar[j] (supported and
 *     correct) vs out overlapping another input (unsupported by the
 *     documented contract: must not crash; result is unchecked)
 *  6. NT-store twin + caller-side fence (SFENCE / DMB ISHST / none on
 *     RVV), output verified against the scalar reference
 *  7. invalid changed-index shapes (dedup with nchanged > ndata,
 *     boundary indices, oom-flag plumbing)
 *  8. OOM/API: op.oom honored, NULL op, rj_get_fn(NULL), zero length
 *
 * Correctness oracle: full re-encode of the modified stripe through the
 * public API (independent codegen path from the delta update), plus the
 * scalar ref_run() for the NT-twin test.
 */
#include "../include/raidjit.h"
#include "../src/plan.h"
#include "../src/jit.h"
#include "../src/gf.h"
#include "../src/ref.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>
#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>

static int fails;
#define CHECK(c, ...) do {                                             \
    if (!(c)) {                                                        \
        printf("FAIL "); printf(__VA_ARGS__); printf("\n"); fails++;   \
    }                                                                  \
} while (0)

static uint64_t xrng_state = 0x51ED47A11E5ULL;
static uint64_t xrng(void)
{
    uint64_t x = xrng_state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return xrng_state = x;
}
static void rnd_bytes(uint8_t *p, size_t n)
{
    for (size_t i = 0; i < n; i++)
        p[i] = (uint8_t)(xrng() >> 33);
}

/* ---- buffer sources: 64-byte aligned, 64-byte canaries ---- */
static uint8_t *abuf_canary(size_t n)
{
    uint8_t *raw = NULL;
    if (posix_memalign((void **)&raw, 64, n + 128))
        abort();
    memset(raw, 0xA5, 64);
    memset(raw + 64 + n, 0x5A, 64);
    return raw + 64;
}
static void afree_canary(uint8_t *p, size_t n)
{
    uint8_t *raw = p - 64;
    int ok = 1;
    for (int i = 0; i < 64; i++)
        if (raw[i] != 0xA5 || raw[64 + n + i] != 0x5A)
            ok = 0;
    CHECK(ok, "canary corruption (buffer overrun/underrun)");
    free(raw);
}
static int check_canary(const uint8_t *p, size_t n)
{
    const uint8_t *raw = p - 64;
    for (int i = 0; i < 64; i++)
        if (raw[i] != 0xA5 || raw[64 + n + i] != 0x5A)
            return 0;
    return 1;
}

/* ---- buffer source: mmap with PROT_NONE guard pages ----
 * Layout: [guard page][data pages][guard page]. The returned pointer is
 * page-aligned (hence 64-byte aligned). Any vector overrun/underrun by
 * the JIT faults immediately. */
static uint8_t *abuf_guarded(size_t n)
{
    long ps = sysconf(_SC_PAGESIZE);
    size_t pagesz = (size_t)ps;
    size_t npages = (n + pagesz - 1) / pagesz;
    if (npages == 0)
        npages = 1;
    size_t total = (npages + 2) * pagesz;
    uint8_t *base = mmap(NULL, total, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED)
        abort();
    if (mprotect(base, pagesz, PROT_NONE) != 0)
        abort();
    if (mprotect(base + (npages + 1) * pagesz, pagesz, PROT_NONE) != 0)
        abort();
    return base + pagesz;
}
static void afree_guarded(uint8_t *p, size_t n)
{
    long ps = sysconf(_SC_PAGESIZE);
    size_t pagesz = (size_t)ps;
    size_t npages = (n + pagesz - 1) / pagesz;
    if (npages == 0)
        npages = 1;
    uint8_t *base = p - pagesz;
    size_t total = (npages + 2) * pagesz;
    if (munmap(base, total) != 0)
        abort();
}
/* Prove the guard pages are really PROT_NONE: a child that writes one
 * byte into the guard must die with SIGSEGV/SIGBUS. This validates the
 * harness itself (a guard that isn't active would make test 3 vacuous). */
static void verify_guard_active(uint8_t *data, size_t n, const char *tag)
{
    long ps = sysconf(_SC_PAGESIZE);
    size_t pagesz = (size_t)ps;
    size_t npages = (n + pagesz - 1) / pagesz;
    if (npages == 0)
        npages = 1;
    uint8_t *lo_guard = data - pagesz;
    uint8_t *hi_guard = data + npages * pagesz;
    for (int g = 0; g < 2; g++) {
        uint8_t *gp = g ? hi_guard : lo_guard;
        pid_t pid = fork();
        if (pid < 0)
            abort();
        if (pid == 0) {
            /* ASan intercepts SEGV (DEADLYSIGNAL) and aborts instead of
             * dying with SIGSEGV; reset to default so the parent can
             * observe the fault via WIFSIGNALED. */
            signal(SIGSEGV, SIG_DFL);
            signal(SIGBUS, SIG_DFL);
            gp[0] = 0xFF; /* must fault */
            _exit(42);    /* unreachable if guard is active */
        }
        int status = 0;
        if (waitpid(pid, &status, 0) < 0)
            abort();
        int faulted = WIFSIGNALED(status) &&
                      (WTERMSIG(status) == SIGSEGV ||
                       WTERMSIG(status) == SIGBUS);
        CHECK(faulted, "%s: %s guard page is not PROT_NONE", tag,
              g ? "high" : "low");
    }
}

/* sort + dedup a changed list; returns count, writes to out (caller-
 * provided, nchanged capacity) */
static int sort_dedup(const int *changed, int nchanged, int *out)
{
    memcpy(out, changed, (size_t)nchanged * sizeof(int));
    for (int a = 0; a < nchanged; a++)
        for (int b = a + 1; b < nchanged; b++)
            if (out[a] > out[b]) {
                int t = out[a]; out[a] = out[b]; out[b] = t;
            }
    int nc = 0;
    for (int a = 0; a < nchanged; a++)
        if (a == 0 || out[a] != out[a - 1])
            out[nc++] = out[a];
    return nc;
}

/* ------------------------------------------------------------------ */
/* Core delta runner: public API, verified against a full re-encode   */
/* of the modified stripe (independent codegen path). Buffers come    */
/* from the given allocator. If inplace, out[j] aliases oldpar[j].    */
/* ------------------------------------------------------------------ */
typedef uint8_t *(*abuf_fn)(size_t n);
typedef void (*afree_fn)(uint8_t *p, size_t n);

static void run_delta_case(rj_topology *t, int ndata, int npar,
                           const int *changed, int nchanged,
                           size_t len, int inplace,
                           abuf_fn abuf, afree_fn afree,
                           const char *tag)
{
    char err[512];
    int *cs = malloc((size_t)(nchanged ? nchanged : 1) * sizeof(int));
    int nc = sort_dedup(changed, nchanged, cs);

    rj_op op = rj_op_parity_update(changed, nchanged);
    rj_options o; rj_options_default(&o); o.benchmark = 0;
    rj_jit *j = rj_compile(t, &op, &o, err, sizeof(err));
    CHECK(j, "%s: delta compile failed: %s", tag, err);
    if (!j) { free(cs); rj_op_free(&op); return; }
    rj_op eop = rj_op_encode();
    rj_options eo; rj_options_default(&eo); eo.benchmark = 0;
    rj_jit *je = rj_compile(t, &eop, &eo, err, sizeof(err));
    CHECK(je, "%s: encode compile failed: %s", tag, err);
    if (!je) { rj_free(j); free(cs); rj_op_free(&op); rj_op_free(&eop); return; }

    /* original data + encode -> old parities */
    uint8_t **data = malloc(sizeof(*data) * (size_t)ndata);
    for (int i = 0; i < ndata; i++) {
        data[i] = abuf(len);
        rnd_bytes(data[i], len);
    }
    uint8_t **oldp = malloc(sizeof(*oldp) * (size_t)npar);
    uint8_t **newp = malloc(sizeof(*newp) * (size_t)npar);
    uint8_t **wantp = malloc(sizeof(*wantp) * (size_t)npar);
    for (int k = 0; k < npar; k++) {
        oldp[k] = abuf(len); newp[k] = abuf(len); wantp[k] = abuf(len);
    }
    const void **di = malloc(sizeof(*di) * (size_t)ndata);
    for (int i = 0; i < ndata; i++) di[i] = data[i];
    void **oo = malloc(sizeof(*oo) * (size_t)npar);
    for (int k = 0; k < npar; k++) oo[k] = oldp[k];
    rj_get_fn(je)(di, oo, len);

    /* old/new images of the changed disks; apply new data */
    uint8_t **oldc = malloc(sizeof(*oldc) * (size_t)(nc ? nc : 1));
    uint8_t **newc = malloc(sizeof(*newc) * (size_t)(nc ? nc : 1));
    uint8_t **oldc_snap = malloc(sizeof(*oldc_snap) * (size_t)(nc ? nc : 1));
    uint8_t **newc_snap = malloc(sizeof(*newc_snap) * (size_t)(nc ? nc : 1));
    for (int k = 0; k < nc; k++) {
        oldc[k] = abuf(len); newc[k] = abuf(len);
        oldc_snap[k] = abuf(len); newc_snap[k] = abuf(len);
        memcpy(oldc[k], data[cs[k]], len);
        rnd_bytes(newc[k], len);
        memcpy(data[cs[k]], newc[k], len);
        memcpy(oldc_snap[k], oldc[k], len);
        memcpy(newc_snap[k], newc[k], len);
    }
    uint8_t **oldp_snap = malloc(sizeof(*oldp_snap) * (size_t)npar);
    for (int k = 0; k < npar; k++) {
        oldp_snap[k] = abuf(len);
        memcpy(oldp_snap[k], oldp[k], len);
    }

    /* delta inputs: [old, new, oldpar]; outputs: newpar */
    const void **din = malloc(sizeof(*din) * (size_t)(2 * nc + npar));
    void **dout = malloc(sizeof(*dout) * (size_t)npar);
    for (int k = 0; k < nc; k++) din[k] = oldc[k];
    for (int k = 0; k < nc; k++) din[nc + k] = newc[k];
    for (int k = 0; k < npar; k++) {
        if (inplace) {
            memcpy(newp[k], oldp[k], len);
            din[2 * nc + k] = newp[k];
            dout[k] = newp[k];
        } else {
            din[2 * nc + k] = oldp[k];
            dout[k] = newp[k];
        }
    }

    rj_get_fn(j)(din, dout, len);

    /* reference: full encode of the modified stripe */
    for (int i = 0; i < ndata; i++) di[i] = data[i];
    for (int k = 0; k < npar; k++) oo[k] = wantp[k];
    rj_get_fn(je)(di, oo, len);

    for (int k = 0; k < npar; k++) {
        uint8_t *got = (uint8_t *)dout[k];
        if (memcmp(got, wantp[k], len) != 0) {
            size_t bad = 0;
            for (size_t b = 0; b < len; b++)
                if (got[b] != wantp[k][b]) bad++;
            CHECK(0, "%s: par%d mismatch (%zu/%zu bytes)", tag, k, bad, len);
        }
    }
    /* inputs must be byte-identical afterwards (item 4) */
    for (int k = 0; k < nc; k++) {
        CHECK(memcmp(oldc[k], oldc_snap[k], len) == 0,
              "%s: oldc[%d] modified", tag, k);
        CHECK(memcmp(newc[k], newc_snap[k], len) == 0,
              "%s: newc[%d] modified", tag, k);
    }
    if (!inplace) {
        for (int k = 0; k < npar; k++)
            CHECK(memcmp(oldp[k], oldp_snap[k], len) == 0,
                  "%s: oldpar[%d] modified (out-of-place)", tag, k);
    }

    for (int i = 0; i < ndata; i++) afree(data[i], len);
    for (int k = 0; k < npar; k++) {
        afree(oldp[k], len); afree(newp[k], len); afree(wantp[k], len);
        afree(oldp_snap[k], len);
    }
    for (int k = 0; k < nc; k++) {
        afree(oldc[k], len); afree(newc[k], len);
        afree(oldc_snap[k], len); afree(newc_snap[k], len);
    }
    free(data); free(oldp); free(newp); free(wantp);
    free(di); free(oo); free(oldc); free(newc); free(din); free(dout);
    free(oldc_snap); free(newc_snap); free(oldp_snap); free(cs);
    rj_free(j); rj_free(je);
    rj_op_free(&op); rj_op_free(&eop);
}

/* ------------------------------------------------------------------ */
/* 1. nchanged == 51: maximum data disks for the default RS matrix     */
/* ------------------------------------------------------------------ */
static void test_nchanged_51(void)
{
    rj_topology *t = rj_topo_raid6(51);
    CHECK(t, "rj_topo_raid6(51) returned NULL");
    if (!t) return;
    int changed[51];
    for (int i = 0; i < 51; i++) changed[i] = i;
    size_t lens[] = { 1, 64, 1000 };
    for (unsigned li = 0; li < sizeof(lens) / sizeof(lens[0]); li++) {
        char tag[128];
        snprintf(tag, sizeof(tag), "nchanged51 len=%zu oop", lens[li]);
        run_delta_case(t, 51, 2, changed, 51, lens[li], 0,
                       abuf_canary, afree_canary, tag);
        snprintf(tag, sizeof(tag), "nchanged51 len=%zu inplace", lens[li]);
        run_delta_case(t, 51, 2, changed, 51, lens[li], 1,
                       abuf_canary, afree_canary, tag);
    }
    rj_topo_free(t);
    if (!fails) printf("nchanged51: OK\n");
}

/* ------------------------------------------------------------------ */
/* 2. 200+ disk custom Cauchy topology (MDS for ndata+npar <= 255)     */
/* ------------------------------------------------------------------ */
static rj_topology *mk_cauchy_200x4(void)
{
    int ndata = 200, npar = 4;
    uint8_t *mat = malloc((size_t)ndata * (size_t)npar);
    /* Cauchy: M[j][i] = 1/(x_i + y_j), x_i = i, y_j = 200+j.
     * x_i ^ y_j != 0 always (ranges 0..199 and 200..203 are disjoint),
     * so gf_inv is defined and the matrix is MDS. */
    for (int j = 0; j < npar; j++)
        for (int i = 0; i < ndata; i++) {
            uint8_t d = (uint8_t)(i ^ (200 + j));
            if (d == 0) { free(mat); return NULL; } /* cannot happen */
            mat[(size_t)j * (size_t)ndata + (size_t)i] = gf_inv(d);
        }
    rj_topology *t = rj_topo_custom(ndata, npar, mat);
    free(mat);
    return t;
}

static void test_wide_topology(void)
{
    rj_topology *t = mk_cauchy_200x4();
    CHECK(t, "rj_topo_custom(200,4) returned NULL");
    if (!t) return;
    /* change 8 scattered disks; bounded runtime despite 204-disk width */
    int changed[8] = { 0, 1, 50, 99, 100, 150, 199, 7 };
    run_delta_case(t, 200, 4, changed, 8, 1024, 0,
                   abuf_canary, afree_canary, "cauchy200x4 oop");
    run_delta_case(t, 200, 4, changed, 8, 1024, 1,
                   abuf_canary, afree_canary, "cauchy200x4 inplace");
    /* all-200 changed would be 404 input pointers; spot-check 64 */
    int changed64[64];
    for (int i = 0; i < 64; i++) changed64[i] = i * 3;
    run_delta_case(t, 200, 4, changed64, 64, 256, 0,
                   abuf_canary, afree_canary, "cauchy200x4 nc=64");
    rj_topo_free(t);
    if (!fails) printf("wide-topo: OK\n");
}

/* ------------------------------------------------------------------ */
/* 3. PROT_NONE guard pages around every buffer                        */
/* ------------------------------------------------------------------ */
static void test_guard_pages(void)
{
    rj_topology *t = rj_topo_raid6(8);
    int changed[2] = { 0, 1 };
    /* lengths touching page edges: the last vector store/load must not
     * step into the guard page */
    size_t lens[] = { 1, 63, 64, 65, 4095, 4096 };
    for (unsigned li = 0; li < sizeof(lens) / sizeof(lens[0]); li++) {
        size_t len = lens[li];
        char tag[128];
        /* prove the harness first: guards really are PROT_NONE */
        uint8_t *probe = abuf_guarded(len);
        verify_guard_active(probe, len, "guard-probe");
        afree_guarded(probe, len);
        snprintf(tag, sizeof(tag), "guard len=%zu oop", len);
        run_delta_case(t, 8, 2, changed, 2, len, 0,
                       abuf_guarded, afree_guarded, tag);
        snprintf(tag, sizeof(tag), "guard len=%zu inplace", len);
        run_delta_case(t, 8, 2, changed, 2, len, 1,
                       abuf_guarded, afree_guarded, tag);
    }
    rj_topo_free(t);
    if (!fails) printf("guard-pages: OK\n");
}

/* ------------------------------------------------------------------ */
/* 5. Aliasing contract. From raidjit.h: out[j] MAY alias in[2*nc+j]   */
/*    (in-place parity update); no other aliasing is allowed and       */
/*    yields silently wrong results (unchecked for speed). So:         */
/*    - exact in-place must be CORRECT;                                */
/*    - out overlapping any other input must NOT crash (no SIGSEGV/    */
/*      SIGBUS); the numeric result is unchecked, but all non-aliased  */
/*      buffers' canaries must stay intact.                            */
/* ------------------------------------------------------------------ */
static void test_aliasing(void)
{
    char err[512];
    rj_topology *t = rj_topo_raid6(8);
    int changed[2] = { 2, 5 };
    int ndata = 8, npar = 2, nc = 2;
    size_t len = 512;

    rj_op op = rj_op_parity_update(changed, nc);
    rj_options o; rj_options_default(&o); o.benchmark = 0;
    rj_jit *j = rj_compile(t, &op, &o, err, sizeof(err));
    CHECK(j, "alias compile: %s", err);
    if (!j) { rj_op_free(&op); rj_topo_free(t); return; }
    rj_op eop = rj_op_encode();
    rj_jit *je = rj_compile(t, &eop, &o, err, sizeof(err));
    CHECK(je, "alias encode compile: %s", err);
    if (!je) { rj_free(j); rj_op_free(&op); rj_op_free(&eop); rj_topo_free(t); return; }

    /* reference stripe */
    uint8_t **data = malloc(sizeof(*data) * (size_t)ndata);
    for (int i = 0; i < ndata; i++) { data[i] = abuf_canary(len); rnd_bytes(data[i], len); }
    uint8_t **par = malloc(sizeof(*par) * (size_t)npar);
    uint8_t **wantp = malloc(sizeof(*wantp) * (size_t)npar);
    for (int k = 0; k < npar; k++) { par[k] = abuf_canary(len); wantp[k] = abuf_canary(len); }
    const void **di = malloc(sizeof(*di) * (size_t)ndata);
    for (int i = 0; i < ndata; i++) di[i] = data[i];
    void **oo = malloc(sizeof(*oo) * (size_t)npar);
    for (int k = 0; k < npar; k++) oo[k] = par[k];
    rj_get_fn(je)(di, oo, len);

    /* changed-disk images */
    uint8_t *oldc[2], *newc[2];
    for (int k = 0; k < nc; k++) {
        oldc[k] = abuf_canary(len); newc[k] = abuf_canary(len);
        memcpy(oldc[k], data[changed[k]], len);
        rnd_bytes(newc[k], len);
        memcpy(data[changed[k]], newc[k], len);
    }
    /* reference parities of the modified stripe */
    for (int i = 0; i < ndata; i++) di[i] = data[i];
    for (int k = 0; k < npar; k++) oo[k] = wantp[k];
    rj_get_fn(je)(di, oo, len);

    /* (a) exact in-place: out[j] == oldpar[j]; must be correct */
    {
        uint8_t *inp[6], *outp[2], *refp[2];
        for (int k = 0; k < npar; k++) {
            refp[k] = abuf_canary(len);
            memcpy(refp[k], par[k], len);
            inp[2 * nc + k] = refp[k];
            outp[k] = refp[k];
        }
        for (int k = 0; k < nc; k++) { inp[k] = oldc[k]; inp[nc + k] = newc[k]; }
        const void *din[6];
        void *dout[2];
        for (int k = 0; k < 2 * nc + npar; k++) din[k] = inp[k];
        for (int k = 0; k < npar; k++) dout[k] = outp[k];
        rj_get_fn(j)(din, dout, len);
        for (int k = 0; k < npar; k++)
            CHECK(memcmp(outp[k], wantp[k], len) == 0,
                  "alias: exact in-place par%d wrong", k);
        for (int k = 0; k < npar; k++) afree_canary(refp[k], len);
    }

    /* (b) unsupported: out[0] fully overlaps newc[0] (another input).
     * Documented as silently-wrong; the test asserts no crash and that
     * every non-aliased buffer's canary survives. */
    {
        uint8_t *inp[6], *outp[2];
        uint8_t *victim_oldc[2], *victim_par[2];
        for (int k = 0; k < nc; k++) {
            victim_oldc[k] = abuf_canary(len);
            memcpy(victim_oldc[k], oldc[k], len);
            inp[k] = victim_oldc[k];
            inp[nc + k] = abuf_canary(len);
            memcpy(inp[nc + k], newc[k], len);
        }
        for (int k = 0; k < npar; k++) {
            victim_par[k] = abuf_canary(len);
            memcpy(victim_par[k], par[k], len);
            inp[2 * nc + k] = victim_par[k];
        }
        outp[0] = inp[nc + 0]; /* out[0] aliases newc[0]: unsupported */
        outp[1] = abuf_canary(len);
        const void *din[6];
        void *dout[2];
        for (int k = 0; k < 2 * nc + npar; k++) din[k] = inp[k];
        dout[0] = outp[0]; dout[1] = outp[1];
        rj_get_fn(j)(din, dout, len); /* must not crash */
        /* canaries of non-aliased buffers intact */
        for (int k = 0; k < nc; k++)
            CHECK(check_canary(victim_oldc[k], len),
                  "alias: oldc[%d] canary damaged", k);
        for (int k = 0; k < npar; k++)
            CHECK(check_canary(victim_par[k], len),
                  "alias: oldpar[%d] canary damaged", k);
        CHECK(check_canary(outp[1], len), "alias: out[1] canary damaged");
        for (int k = 0; k < nc; k++) { afree_canary(victim_oldc[k], len); afree_canary(inp[nc + k], len); }
        for (int k = 0; k < npar; k++) afree_canary(victim_par[k], len);
        afree_canary(outp[1], len);
    }

    for (int i = 0; i < ndata; i++) afree_canary(data[i], len);
    for (int k = 0; k < npar; k++) { afree_canary(par[k], len); afree_canary(wantp[k], len); }
    for (int k = 0; k < nc; k++) { afree_canary(oldc[k], len); afree_canary(newc[k], len); }
    free(data); free(par); free(wantp); free(di); free(oo);
    rj_free(j); rj_free(je);
    rj_op_free(&op); rj_op_free(&eop); rj_topo_free(t);
    if (!fails) printf("aliasing: OK\n");
}

/* ------------------------------------------------------------------ */
/* 6. NT-store twin + caller-side fence. Enumerate NT twins directly   */
/*    (internal API, real CPU), run each, fence per the ISA contract,  */
/*    verify against the scalar reference. At least one NT twin must   */
/*    be exercised (non-vacuity).                                      */
/* ------------------------------------------------------------------ */
static void caller_fence(void)
{
#if defined(__x86_64__)
    __asm__ volatile("sfence" ::: "memory");
#elif defined(__aarch64__)
    __asm__ volatile("dmb ishst" ::: "memory");
#elif defined(__riscv)
    /* Zihintntl NTL.ALL hint: no ordering change, no fence needed. */
#else
#error "caller_fence: unsupported architecture"
#endif
}

static void test_nt_fence(void)
{
    char err[512];
    rj_cpu_info cpu;
    rj_cpu_detect(&cpu);
    rj_topology *t = rj_topo_raid6(8);
    int ch[2] = { 1, 4 };
    int ndata = 8, npar = 2, nc = 2;
    size_t lens[] = { 64, 4096, 65537 };

    rj_op op = rj_op_parity_update(ch, nc);
    topo_int_t ti;
    memset(&ti, 0, sizeof(ti));
    if (topo_int_build(&ti, t, err, sizeof(err)) != 0) {
        CHECK(0, "nt: topo build: %s", err);
        rj_op_free(&op); rj_topo_free(t);
        return;
    }
    plan_t pl;
    memset(&pl, 0, sizeof(pl));
    if (plan_build(&ti, &op, "raid6", &pl, err, sizeof(err)) != 0) {
        CHECK(0, "nt: plan build: %s", err);
        topo_int_free(&ti);
        rj_op_free(&op); rj_topo_free(t);
        return;
    }
    cand_t cands[128];
    int n = plan_candidates(&pl, &cpu, cands, 128, /*allow_nt=*/1,
                            /*prefetch=*/0);
    CHECK(n > 0, "nt: no candidates");
    rj_options o; rj_options_default(&o); o.benchmark = 0;

    int n_nt = 0;
    for (int ci = 0; ci < n; ci++) {
        if (!cands[ci].nt)
            continue;
        n_nt++;
        version_t v;
        memset(&v, 0, sizeof(v));
        if (jit_build_version(&pl, &cands[ci], &cpu, &o, &v,
                              err, sizeof(err)) != 0) {
            CHECK(0, "nt: build %s: %s", isa_name(cands[ci].isa), err);
            continue;
        }
        for (unsigned li = 0; li < sizeof(lens) / sizeof(lens[0]); li++) {
            size_t len = lens[li];
            /* delta inputs */
            uint8_t *oldc[2], *newc[2], *oldp[2], *outp[2], *exp[2];
            for (int k = 0; k < nc; k++) {
                oldc[k] = abuf_canary(len); newc[k] = abuf_canary(len);
                rnd_bytes(oldc[k], len); rnd_bytes(newc[k], len);
            }
            for (int k = 0; k < npar; k++) {
                oldp[k] = abuf_canary(len); outp[k] = abuf_canary(len);
                exp[k] = abuf_canary(len);
                rnd_bytes(oldp[k], len);
            }
            /* snapshots for input-immutability (item 4 on the NT path) */
            uint8_t *oldc_snap[2], *newc_snap[2], *oldp_snap[2];
            for (int k = 0; k < nc; k++) {
                oldc_snap[k] = abuf_canary(len); newc_snap[k] = abuf_canary(len);
                memcpy(oldc_snap[k], oldc[k], len);
                memcpy(newc_snap[k], newc[k], len);
            }
            for (int k = 0; k < npar; k++) {
                oldp_snap[k] = abuf_canary(len);
                memcpy(oldp_snap[k], oldp[k], len);
            }
            const void *in[6];
            void *out[2];
            for (int k = 0; k < nc; k++) in[k] = oldc[k];
            for (int k = 0; k < nc; k++) in[nc + k] = newc[k];
            for (int k = 0; k < npar; k++) in[2 * nc + k] = oldp[k];
            for (int k = 0; k < npar; k++) out[k] = outp[k];
            v.fn(in, out, len);
            caller_fence(); /* ISA contract: caller fences NT stores */
            ref_run(&pl, (const uint8_t **)in, exp, len);
            for (int k = 0; k < npar; k++)
                CHECK(memcmp(outp[k], exp[k], len) == 0,
                      "nt: [%s] len=%zu par%d mismatch", v.desc, len, k);
            for (int k = 0; k < nc; k++) {
                CHECK(memcmp(oldc[k], oldc_snap[k], len) == 0,
                      "nt: [%s] oldc[%d] modified", v.desc, k);
                CHECK(memcmp(newc[k], newc_snap[k], len) == 0,
                      "nt: [%s] newc[%d] modified", v.desc, k);
            }
            for (int k = 0; k < npar; k++)
                CHECK(memcmp(oldp[k], oldp_snap[k], len) == 0,
                      "nt: [%s] oldpar[%d] modified", v.desc, k);
            for (int k = 0; k < nc; k++) {
                afree_canary(oldc[k], len); afree_canary(newc[k], len);
                afree_canary(oldc_snap[k], len); afree_canary(newc_snap[k], len);
            }
            for (int k = 0; k < npar; k++) {
                afree_canary(oldp[k], len); afree_canary(outp[k], len);
                afree_canary(exp[k], len); afree_canary(oldp_snap[k], len);
            }
        }
        jit_free_version(&v);
        if (fails) break;
    }
    CHECK(n_nt > 0, "nt: no NT twins enumerated (vacuous test)");
    if (!fails) printf("nt-fence: OK (%d NT twins)\n", n_nt);
    plan_free(&pl);
    topo_int_free(&ti);
    rj_op_free(&op);
    rj_topo_free(t);
    (void)ndata;
}

/* ------------------------------------------------------------------ */
/* 7. Invalid changed-index shapes (beyond test_pdelta.c's baseline)   */
/* ------------------------------------------------------------------ */
static void test_invalid_indices(void)
{
    char err[256];
    rj_options o; rj_options_default(&o); o.benchmark = 0;

    /* nchanged > ndata, all duplicates: dedups to a valid set.
     * Verify the deduped set is CORRECT (not just compilable) via the
     * full re-encode oracle. */
    {
        int ch[6] = { 0, 1, 2, 3, 0, 1 }; /* raid6(4): ndata=4 */
        rj_topology *t = rj_topo_raid6(4);
        run_delta_case(t, 4, 2, ch, 6, 64, 0,
                       abuf_canary, afree_canary, "dedup nchanged>ndata oop");
        run_delta_case(t, 4, 2, ch, 6, 64, 1,
                       abuf_canary, afree_canary, "dedup nchanged>ndata inplace");
        rj_topo_free(t);
    }
    /* index == ndata (boundary): out of range */
    {
        int ch[1] = { 4 }; /* raid6(4): valid 0..3 */
        rj_op op = rj_op_parity_update(ch, 1);
        rj_topology *t = rj_topo_raid6(4);
        rj_jit *j = rj_compile(t, &op, &o, err, sizeof(err));
        CHECK(!j, "changed==ndata should fail compile");
        rj_op_free(&op); rj_topo_free(t);
    }
    /* huge nchanged clamps to RJ_MAX_DISKS (256); op is usable */
    {
        int *ch = malloc(1000 * sizeof(int));
        for (int i = 0; i < 1000; i++) ch[i] = i;
        rj_op op = rj_op_parity_update(ch, 1000);
        CHECK(op.nchanged == 256, "nchanged=1000 clamps to 256, got %d",
              op.nchanged);
        CHECK(!op.oom, "clamped op must not set oom");
        rj_op_free(&op);
        free(ch);
    }
    /* single changed disk at the last data index */
    {
        int ch[1] = { 50 }; /* raid6(51): valid 0..50 */
        rj_op op = rj_op_parity_update(ch, 1);
        rj_topology *t = rj_topo_raid6(51);
        rj_jit *j = rj_compile(t, &op, &o, err, sizeof(err));
        CHECK(j, "single last-disk change should compile: %s", err);
        if (j) rj_free(j);
        rj_op_free(&op); rj_topo_free(t);
    }
    if (!fails) printf("invalid-indices: OK\n");
}

/* ------------------------------------------------------------------ */
/* 8. OOM/API behavior                                                */
/* ------------------------------------------------------------------ */
static void test_oom_api(void)
{
    char err[256];
    rj_options o; rj_options_default(&o); o.benchmark = 0;

    /* op.oom honored: constructor-level OOM must fail cleanly */
    {
        int ch[1] = { 0 };
        rj_op op = rj_op_parity_update(ch, 1);
        op.oom = 1; /* simulate constructor allocation failure */
        rj_topology *t = rj_topo_raid6(4);
        rj_jit *j = rj_compile(t, &op, &o, err, sizeof(err));
        CHECK(!j, "op.oom=1 must fail compile cleanly");
        CHECK(err[0] != '\0', "op.oom=1 should produce an error message");
        rj_op_free(&op); rj_topo_free(t);
    }
    /* NULL op */
    {
        rj_topology *t = rj_topo_raid6(4);
        rj_jit *j = rj_compile(t, NULL, &o, err, sizeof(err));
        CHECK(!j, "NULL op should fail compile");
        rj_topo_free(t);
    }
    /* rj_get_fn(NULL) */
    CHECK(rj_get_fn(NULL) == NULL, "rj_get_fn(NULL) should return NULL");
    /* NULL opts -> defaults */
    {
        int ch[1] = { 0 };
        rj_op op = rj_op_parity_update(ch, 1);
        rj_topology *t = rj_topo_raid6(4);
        rj_jit *j = rj_compile(t, &op, NULL, err, sizeof(err));
        CHECK(j, "NULL opts should use defaults: %s", err);
        if (j) rj_free(j);
        rj_op_free(&op); rj_topo_free(t);
    }
    /* zero length through the public path */
    {
        int ch[2] = { 0, 1 };
        rj_topology *t = rj_topo_raid6(8);
        run_delta_case(t, 8, 2, ch, 2, 0, 0,
                       abuf_canary, afree_canary, "zero-len");
        rj_topo_free(t);
    }
    if (!fails) printf("oom-api: OK\n");
}

int main(void)
{
    test_nchanged_51();
    test_wide_topology();
    test_guard_pages();
    test_aliasing();
    test_nt_fence();
    test_invalid_indices();
    test_oom_api();
    if (fails == 0)
        printf("test_pdelta_edge: all OK\n");
    else
        printf("test_pdelta_edge: %d FAILURES\n", fails);
    return fails != 0;
}
