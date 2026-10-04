/* Version selection: benchmark with noise-aware tie-break, or static
 * heuristic when benchmarking is disabled.
 *
 * Protocol:
 *   1. Correctness first: every candidate's output is checked against the
 *      scalar reference (ref_run) at scalar-only, vector-boundary and
 *      multi-group lengths, plus one aligned non-zero-start case. A wrong
 *      candidate fails compilation -- it must never win on speed.
 *   2. Timing: several working-set sizes (bench_bytes/4, bench_bytes,
 *      bench_bytes*4, clamped), warm-up calls, then interleaved rounds
 *      (round-robin over candidates) so thermal/frequency drift cannot
 *      favor early candidates. Median per (candidate, size); the score is
 *      total time across sizes.
 *   3. Selection: select_by_time on the scores -- only candidates within
 *      max(1.5%, measured noise) of the best score are eligible; the noise
 *      estimate is the pooled (median over candidates of median-over-sizes)
 *      relative IQR of the timing samples. Ties break on uops/byte, then
 *      mem/byte, then code size, so candidates tied within measurement
 *      noise resolve deterministically toward the most efficient code.
 *
 * Performance claims are therefore limited to: best among the generated
 * legal candidates under this protocol on this CPU.
 *
 * Note for NT-store twins: the protocol measures steady-state
 * repeated-call throughput on reused buffers, not single-shot cold
 * writes. An NT win therefore means NT wins even under buffer-reuse
 * pressure (where temporal stores have the home-field advantage) — a
 * conservative signal, but it is still a claim about this protocol, not
 * about every possible access pattern. */
#include "bench.h"
#include "ref.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

static uint64_t xrng(uint64_t *s)
{
    uint64_t x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return *s = x;
}

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Sort samples in place; fill median and quartiles (Tukey hinges). */
static void sample_stats(double *ts, int n, double *med, double *q1,
                         double *q3)
{
    for (int i = 0; i < n; i++)
        for (int k = i + 1; k < n; k++)
            if (ts[k] < ts[i]) {
                double t = ts[i];
                ts[i] = ts[k];
                ts[k] = t;
            }
    *med = ts[n / 2];
    if (n >= 4) {
        *q1 = ts[n / 4];
        *q3 = ts[(3 * n) / 4];
    } else {
        *q1 = ts[0];
        *q3 = ts[n - 1];
    }
}

/* Validate every candidate against the scalar reference before timing.
 * Returns 0 on success, -1 with err set on the first mismatch.
 *
 * Length-dispatch twins (version_t.notail) contractually serve only
 * len % group_bytes == 0, so they are validated only at group multiples;
 * every other version keeps the full scalar/boundary/multi-group sweep. */
static int validate_versions(version_t *vs, int nv, const plan_t *pl,
                             char *err, size_t errlen)
{
    if (pl->n_out == 0 || !pl->mat)
        return 0; /* noop plan: nothing to validate */
    int n_in = pl->n_in, n_out = pl->n_out;
    const size_t vmax = 4096;
    /* Length-dispatch twins are validated at 2*group_bytes, which can
     * exceed vmax when the group is large. The old code sized the
     * scratch buffers at exactly vmax and then filled/timed/called at
     * lens[1] = 2*gb -- a heap over-read/over-write past the buffers
     * whenever 2*gb > vmax. Latent today (current maxima: SVE 5*256 =
     * 1280 B at 2048-bit VL, RVV 5*vlenb, x86 512 B -- all under vmax),
     * but group_bytes is runtime-determined on RVV/SVE, so the
     * "2*gb <= vmax" assumption was unchecked. Size the buffers for the
     * largest length any version will actually touch. */
    size_t vbuf = vmax;
    for (int vi = 0; vi < nv; vi++) {
        size_t gb0 = vs[vi].group_bytes
                         ? vs[vi].group_bytes
                         : (size_t)(vs[vi].cand.U * vs[vi].vl);
        if (vs[vi].notail && gb0 > vbuf / 2) {
            if (gb0 > (size_t)-1 / 2) {
                snprintf(err, errlen,
                         "benchmark: candidate %s has absurd group_bytes=%zu",
                         vs[vi].desc, gb0);
                return -1;
            }
            vbuf = 2 * gb0;
        }
    }
    uint8_t **ib = malloc(sizeof(uint8_t *) * (size_t)n_in);
    uint8_t **ob = malloc(sizeof(uint8_t *) * (size_t)n_out);
    uint8_t **rb = malloc(sizeof(uint8_t *) * (size_t)n_out);
    const void **in = malloc(sizeof(void *) * (size_t)n_in);
    void **out = malloc(sizeof(void *) * (size_t)n_out);
    /* offset views for the non-zero-start validation case */
    const void **in_u = malloc(sizeof(void *) * (size_t)n_in);
    void **out_u = malloc(sizeof(void *) * (size_t)n_out);
    const uint8_t **ib_u = malloc(sizeof(uint8_t *) * (size_t)n_in);
    if (!ib || !ob || !rb || !in || !out || !in_u || !out_u || !ib_u) {
        free(ib);
        free(ob);
        free(rb);
        free(in);
        free(out);
        free(in_u);
        free(out_u);
        free(ib_u);
        snprintf(err, errlen, "out of memory");
        return -1;
    }
    for (int i = 0; i < n_in; i++) {
        if (posix_memalign((void **)&ib[i], 64, vbuf))
            abort();
        in[i] = ib[i];
    }
    for (int j = 0; j < n_out; j++) {
        if (posix_memalign((void **)&ob[j], 64, vbuf))
            abort();
        if (posix_memalign((void **)&rb[j], 64, vbuf))
            abort();
        out[j] = ob[j];
    }

    int rc = 0;
    for (int i = 0; i < nv && rc == 0; i++) {
        /* Track A5: a pooled candidate set may mix versions built from
         * different plans (survivor-set selection); each version is
         * validated against the scalar reference of its OWN plan.
         * version_t.plan is set by jit_build_version; NULL (older
         * callers) falls back to the shared plan. n_in/n_out are
         * identical across a pool by construction, so the shared
         * buffers stay valid. */
        const plan_t *vpl = vs[i].plan ? vs[i].plan : pl;
        /* True full-group bytes; group_bytes is set by every backend
         * (U*VL, or U*vlenb on RISC-V). */
        size_t gb = vs[i].group_bytes
                        ? vs[i].group_bytes
                        : (size_t)(vs[i].cand.U * vs[i].vl);
        size_t lens[5];
        int nlens;
        if (vs[i].notail) {
            /* One group, two groups, and a large group multiple. Never
             * at lengths the twin contractually cannot serve. */
            lens[0] = gb;
            lens[1] = gb * 2;
            lens[2] = (vmax / gb) * gb;
            nlens = 3;
        } else {
            /* scalar-only, vector-group boundaries, multi-group */
            int group = vs[i].cand.U * vs[i].vl;
            lens[0] = 1;
            lens[1] = (size_t)group > 1 ? (size_t)group - 1 : 1;
            lens[2] = (size_t)group;
            lens[3] = (size_t)group + 1;
            lens[4] = vmax;
            nlens = 5;
        }
        uint64_t s = 0x12345678abcdefULL + (uint64_t)i;
        for (int li = 0; li < nlens && rc == 0; li++) {
            size_t len = lens[li];
            for (int k = 0; k < n_in; k++)
                for (size_t b = 0; b < len; b++)
                    ib[k][b] = (uint8_t)xrng(&s);
            for (int k = 0; k < n_out; k++)
                memset(ob[k], 0xA5, len);
            vs[i].fn(in, out, len);
            ref_run(vpl, (const uint8_t **)ib, rb, len);
            for (int k = 0; k < n_out && rc == 0; k++)
                if (memcmp(ob[k], rb[k], len) != 0) {
                    snprintf(err, errlen,
                             "benchmark: candidate %s produced wrong output "
                             "at len=%zu (scalar reference mismatch)",
                             vs[i].desc, len);
                    rc = -1;
                }
        }
        /* offset-start case: 64-byte-aligned non-zero buffer starts, so any
         * pointer arithmetic or offset bug in codegen is caught here too.
         * (Deliberately misaligned buffers would violate the API contract
         * and fault on the aligned vector loads; they are not tested.) */
        if (rc == 0) {
            size_t ulen = vmax - 128;
            if (vs[i].notail)
                ulen = (ulen / gb) * gb; /* keep the twin's length contract */
            for (int k = 0; k < n_in; k++) {
                for (size_t b = 0; b < ulen; b++)
                    ib[k][64 + b] = (uint8_t)xrng(&s);
                ib_u[k] = ib[k] + 64;
                in_u[k] = ib[k] + 64;
            }
            for (int k = 0; k < n_out; k++) {
                memset(ob[k] + 64, 0xA5, ulen);
                out_u[k] = ob[k] + 64;
            }
            vs[i].fn(in_u, out_u, ulen);
            ref_run(vpl, ib_u, rb, ulen);
            for (int k = 0; k < n_out && rc == 0; k++)
                if (memcmp((uint8_t *)out_u[k], rb[k], ulen) != 0) {
                    snprintf(err, errlen,
                             "benchmark: candidate %s produced wrong output "
                             "at offset-start len=%zu (scalar reference "
                             "mismatch)",
                             vs[i].desc, ulen);
                    rc = -1;
                }
        }
    }

    for (int i = 0; i < n_in; i++)
        free(ib[i]);
    for (int j = 0; j < n_out; j++) {
        free(ob[j]);
        free(rb[j]);
    }
    free(ib);
    free(ob);
    free(rb);
    free(in);
    free(out);
    free(in_u);
    free(out_u);
    free(ib_u);
    return rc;
}

/* Warm-up calls: fault in pages, fill i-cache/uop-cache for this version. */
static void warm_version(version_t *v, void **in, void **out, size_t bytes)
{
    for (int i = 0; i < 3; i++)
        v->fn((const void **)in, out, bytes);
}

/* One timed call. */
static double timed_call(version_t *v, void **in, void **out, size_t bytes)
{
    double t0 = now_sec();
    v->fn((const void **)in, out, bytes);
    double t1 = now_sec();
    return t1 - t0;
}

/* Length-dispatch twins only admit len % group_bytes == 0: time them at
 * the largest group multiple <= the nominal size (bench sizes are powers
 * of two >= 4096 and groups are small, so this is exact whenever the
 * group divides the size). Every other version runs the nominal size. */
static size_t version_call_bytes(const version_t *v, size_t bytes)
{
    if (!v->notail || v->group_bytes == 0)
        return bytes;
    size_t g = v->group_bytes;
    size_t r = (bytes / g) * g;
    return r >= g ? r : g;
}

/* Pure noise-window computation, factored out for unit testing.
 * csn is nv*nsizes row-major: csn[i*nsizes+si] = relative IQR of candidate
 * i at size si. Per-candidate noise = median over sizes; pooled noise =
 * median over candidates; window = max(1.5%, pooled). If pcand_out is
 * non-NULL it receives the per-candidate noises in candidate order (for
 * diagnostics). On allocation failure returns the 1.5% floor (safe
 * degradation). */
double noise_window(const double *csn, int nv, int nsizes, double *pcand_out)
{
    double *pc = malloc(sizeof(double) * (size_t)nv);
    double *tmp = malloc(sizeof(double) * (size_t)nsizes);
    double *ps = malloc(sizeof(double) * (size_t)nv);
    if (!pc || !tmp || !ps) {
        free(pc);
        free(tmp);
        free(ps);
        return 0.015;
    }
    for (int i = 0; i < nv; i++) {
        for (int si = 0; si < nsizes; si++)
            tmp[si] = csn[(size_t)i * (size_t)nsizes + si];
        double med, q1, q3;
        sample_stats(tmp, nsizes, &med, &q1, &q3);
        pc[i] = med;
    }
    if (pcand_out)
        memcpy(pcand_out, pc, sizeof(double) * (size_t)nv);
    /* median over candidates; sort a copy so pc stays in candidate order */
    memcpy(ps, pc, sizeof(double) * (size_t)nv);
    double pmed, pq1, pq3;
    sample_stats(ps, nv, &pmed, &pq1, &pq3);
    free(pc);
    free(tmp);
    free(ps);
    return pmed > 0.015 ? pmed : 0.015;
}

/* Pure selection from measured times. The argmin-time candidate seeds the
 * winner; only candidates within `window` (relative) of the best time are
 * ever eligible -- a slower candidate must never win on tie-breaks alone.
 * The caller sets window = max(1.5%, measured noise) so that candidates
 * tied within measurement noise fall through to the efficiency tie-break
 * (uops/byte, then mem/byte, then code size). */
int select_by_time(const version_t *vs, int nv, const double *secs,
                   double window)
{
    double best = secs[0];
    int besti = 0;
    for (int i = 1; i < nv; i++)
        if (secs[i] < best) {
            best = secs[i];
            besti = i;
        }

    /* noise tie: within `window` of the best time.
     * NaN-safe: a NaN time must never count as tied. */
    double thresh = best * (1.0 + window);
    int winner = besti;
    for (int i = 0; i < nv; i++) {
        if (i == besti)
            continue;
        if (!(secs[i] <= thresh))
            continue;
        int w = winner;
        if (vs[i].uops_per_byte < vs[w].uops_per_byte - 1e-9)
            winner = i;
        else if (fabs(vs[i].uops_per_byte - vs[w].uops_per_byte) < 1e-9) {
            if (vs[i].mem_per_byte < vs[w].mem_per_byte - 1e-9)
                winner = i;
            else if (fabs(vs[i].mem_per_byte - vs[w].mem_per_byte) < 1e-9 &&
                     vs[i].code_size < vs[w].code_size)
                winner = i;
        }
    }
    return winner;
}

int bench_select(version_t *vs, int nv, const plan_t *pl,
                 const rj_options *opts, int verbose, double *noise_out,
                 char *err, size_t errlen)
{
    if (nv <= 0) {
        snprintf(err, errlen, "no candidates to benchmark");
        return -1;
    }

    /* A notail twin with group_bytes == 0 has no meaningful validation
     * lengths or timing sizes: the dispatch predicate would admit
     * nothing, but the timing loop would silently record garbage at the
     * nominal sizes. Unreachable today (every backend sets group_bytes
     * on success) -- fail loudly rather than rely on backend
     * discipline. */
    for (int i = 0; i < nv; i++)
        if (vs[i].notail && vs[i].group_bytes == 0) {
            snprintf(err, errlen,
                     "benchmark: notail candidate %s has group_bytes==0",
                     vs[i].desc);
            return -1;
        }

    /* 1. correctness before timing: a wrong candidate must never win. */
    if (validate_versions(vs, nv, pl, err, errlen) != 0)
        return -1;

    /* 2. timing over several working-set sizes, interleaved rounds. */
    size_t b = opts->bench_bytes ? opts->bench_bytes : (size_t)1 << 20;
    if (b > ((size_t)16 << 20))
        b = (size_t)16 << 20; /* clamp before b*4: no wrap, sizes cap here */
    int iters = opts->bench_iters > 0 ? opts->bench_iters : 11;
    size_t sizes[3];
    int nsizes = 0;
    size_t cand_sizes[3] = { b / 4, b, b * 4 };
    for (int i = 0; i < 3; i++) {
        size_t s = cand_sizes[i];
        if (s < 4096)
            s = 4096;
        if (s > (size_t)16 << 20)
            s = (size_t)16 << 20;
        int dup = 0;
        for (int k = 0; k < nsizes; k++)
            if (sizes[k] == s)
                dup = 1;
        if (!dup)
            sizes[nsizes++] = s;
    }

    int n_in = pl->n_in, n_out = pl->n_out;
    /* per-candidate score = total median time across sizes;
     * csn[i*nsizes+si] = relative IQR of candidate i at size si */
    double *score = calloc((size_t)nv, sizeof(double));
    double *csn = calloc((size_t)nv * (size_t)nsizes, sizeof(double));
    double *ts = malloc(sizeof(double) * (size_t)nv * (size_t)iters);
    /* Per-version byte totals: notail twins may run slightly smaller
     * (group-rounded) sizes, so each version's GB/s uses its own bytes. */
    double *vtot = calloc((size_t)nv, sizeof(double));
    if (!score || !csn || !ts || !vtot) {
        free(score);
        free(csn);
        free(ts);
        free(vtot);
        snprintf(err, errlen, "out of memory");
        return -1;
    }
    uint64_t cksum = 0x12345678abcdefULL;

    for (int si = 0; si < nsizes; si++) {
        size_t bytes = sizes[si];
        void **in = malloc(sizeof(void *) * (size_t)n_in);
        void **out = malloc(sizeof(void *) * (size_t)n_out);
        uint8_t **ib = malloc(sizeof(uint8_t *) * (size_t)n_in);
        uint8_t **ob = malloc(sizeof(uint8_t *) * (size_t)n_out);
        if (!in || !out || !ib || !ob) {
            free(in);
            free(out);
            free(ib);
            free(ob);
            free(score);
            free(csn);
            free(ts);
            free(vtot);
            snprintf(err, errlen, "out of memory");
            return -1;
        }
        uint64_t s = 0x9e3779b97f4a7c15ULL + (uint64_t)si;
        for (int i = 0; i < n_in; i++) {
            if (posix_memalign((void **)&ib[i], 64, bytes))
                abort();
            for (size_t k = 0; k < bytes; k++)
                ib[i][k] = (uint8_t)xrng(&s);
            in[i] = ib[i];
        }
        for (int j = 0; j < n_out; j++) {
            if (posix_memalign((void **)&ob[j], 64, bytes))
                abort();
            out[j] = ob[j];
        }

        for (int i = 0; i < nv; i++)
            warm_version(&vs[i], in, out,
                         version_call_bytes(&vs[i], bytes));
        /* interleaved rounds: every candidate sees the same machine state
         * per round, so drift cannot favor early candidates */
        for (int r = 0; r < iters; r++)
            for (int i = 0; i < nv; i++)
                ts[(size_t)i * (size_t)iters + r] = timed_call(
                    &vs[i], in, out, version_call_bytes(&vs[i], bytes));

        /* checksum to keep pages hot and prove the calls ran */
        if (n_out > 0)
            for (size_t k = 0; k < bytes; k += 64)
                cksum ^= ((uint8_t *)out[0])[k];

        for (int i = 0; i < nv; i++) {
            double *t = ts + (size_t)i * (size_t)iters;
            double med, q1, q3;
            sample_stats(t, iters, &med, &q1, &q3);
            score[i] += med;
            if (med > 0)
                csn[(size_t)i * (size_t)nsizes + si] = (q3 - q1) / med;
        }
        for (int i = 0; i < nv; i++)
            vtot[i] += (double)(n_in + n_out) *
                       (double)version_call_bytes(&vs[i], bytes);

        for (int i = 0; i < n_in; i++)
            free(ib[i]);
        for (int j = 0; j < n_out; j++)
            free(ob[j]);
        free(in);
        free(out);
        free(ib);
        free(ob);
    }

    /* eligibility window = max(1.5%, pooled measurement noise): candidates
     * tied within measurement noise fall through to the efficiency
     * tie-break. Pooled in noise_window() so it is unit-testable; pnoise
     * keeps per-candidate noise in candidate order for diagnostics. */
    double *pnoise = malloc(sizeof(double) * (size_t)nv);
    double window = pnoise ? noise_window(csn, nv, nsizes, pnoise) : 0.015;
    if (noise_out)
        *noise_out = window;

    /* 3. select on total time; report GB/s over all sizes. Note the score
     * is a sum of per-size medians, so the largest size (~76% of the sum
     * for the default {b/4,b,4b}) dominates the decision; that size has
     * the best timer SNR and is the most DRAM-streaming, i.e. the most
     * representative of bulk RAID throughput. */
    int winner = select_by_time(vs, nv, score, window);
    for (int i = 0; i < nv; i++)
        vs[i].gbps = score[i] > 0 ? vtot[i] / score[i] / 1e9 : 0;

    if (verbose) {
        fprintf(stderr,
                "benchmark: %d candidates, %d sizes, %d interleaved iters, "
                "window=%.1f%%\n",
                nv, nsizes, iters, window * 100.0);
        for (int i = 0; i < nv; i++)
            fprintf(stderr,
                    "  %-42s %8.2f GB/s noise=%.1f%% uops/B=%.2f mem/B=%.2f "
                    "code=%zu%s\n",
                    vs[i].desc, vs[i].gbps, pnoise[i] * 100.0,
                    vs[i].uops_per_byte, vs[i].mem_per_byte, vs[i].code_size,
                    i == winner ? "  <--" : "");
        fprintf(stderr, "  checksum=%016llx\n", (unsigned long long)cksum);
    }
    free(pnoise);
    for (int i = 0; i < nv; i++)
        vs[i].gbps_measured = 1;
    free(score);
    free(csn);
    free(ts);
    free(vtot);
    return winner;
}

/* ---------------- tail duel (track B1x86) ---------------- */

/* Build the duel length lists for a group g. Decisive: tail-dominated
 * lengths (small lens where the tail is most of the work, plus
 * group-relative tails). Large: big multiple-plus-tail lengths (the tail
 * is negligible; the twin must not regress there). Every length is a tail
 * length (len % g != 0): those are the only lengths a masked twin will
 * ever serve (dispatch.h). lens[] must hold 64 entries; the ndec decisive
 * lengths come first, then the nlarge large ones. */
static void duel_lengths(size_t g, size_t *lens, int *ndec, int *nlarge)
{
    long long rcand[] = {
        1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 255,
        (long long)g - 63, (long long)g - 31, (long long)g - 7,
        (long long)g - 1, (long long)g + 1, (long long)g + 4,
        (long long)g + 7, (long long)g + 8, (long long)g + 16,
        (long long)g + 31, (long long)g + 32, (long long)g + 63,
        (long long)g + 64, (long long)2 * g - 1, (long long)2 * g + 1,
    };
    int nd = 0;
    for (size_t i = 0; i < sizeof(rcand) / sizeof(rcand[0]); i++) {
        long long lc = rcand[i];
        if (lc <= 0 || lc > 4096)
            continue;
        size_t L = (size_t)lc;
        if (L % g == 0)
            continue; /* not a tail length */
        int dup = 0;
        for (int k = 0; k < nd; k++)
            if (lens[k] == L)
                dup = 1;
        if (!dup)
            lens[nd++] = L;
    }
    /* Large gate: round 65536 up to a group multiple, then add a small
     * tail. base is a group multiple, so (base+k)%g == k%g: the added
     * tails are genuine tail lengths iff k%g != 0. 63%g != 0 holds for
     * every real group (g = U*VL >= 64 > 63 on AVX-512; the SVE twin's
     * groups are U*vlenb with vlenb a power of two >= 16, hence even,
     * and 63 = 3^2*7 is odd -- the g values that would break it,
     * {1,3,7,9,21,63}, contain no real group). */
    size_t base = (((size_t)65536 + g - 1) / g) * g;
    int nl = 0;
    size_t ltail[] = { 1, 7, 63 };
    for (int i = 0; i < 3; i++)
        lens[nd + nl++] = base + ltail[i];
    *ndec = nd;
    *nlarge = nl;
}

/* Pure duel decision, factored out for unit testing (see bench.h). */
int tail_duel_decide(const double *twin_meds, const double *general_meds,
                     int ndecisive, const double *twin_large,
                     const double *general_large, int nlarge, double window)
{
    double st = 0, sg = 0;
    for (int i = 0; i < ndecisive; i++) {
        st += twin_meds[i];
        sg += general_meds[i];
    }
    /* Strict win: a tie keeps the status-quo general (installing a
     * dispatcher has a cost, so the twin must earn it). NaN-safe: a NaN
     * score fails the strict comparison and drops the twin. */
    if (!(st < sg * (1.0 - window)))
        return 0;
    /* No-regression on the SUM of the large lengths (not per-length): a
     * single noisy length must not veto an otherwise clean duel, while a
     * consistent slowdown across the large set still fails. The floor
     * keeps the gate sane when the measured window is tiny: this is a
     * sanity check (the twins' main loops are identical by construction,
     * so only noise can move it), not a precision instrument. */
    double lt = 0, lg = 0;
    for (int i = 0; i < nlarge; i++) {
        lt += twin_large[i];
        lg += general_large[i];
    }
    double tol = window > 0.05 ? window : 0.05;
    if (!(lt <= lg * (1.0 + tol)))
        return 0; /* large-size regression: drop the twin */
    return 1;
}

int bench_tail_duel(version_t *twin, version_t *general, const plan_t *pl,
                    const rj_options *opts, int verbose,
                    char *err, size_t errlen)
{
    if (!twin || !general || !pl || !opts) {
        snprintf(err, errlen, "tail duel: null argument");
        return -1;
    }
    size_t g = twin->group_bytes;
    if (g == 0) {
        snprintf(err, errlen, "tail duel: twin %s has group_bytes==0",
                 twin->desc);
        return -1;
    }
    if (general->group_bytes != g) {
        snprintf(err, errlen,
                 "tail duel: group mismatch (twin=%zu, general=%zu)",
                 g, general->group_bytes);
        return -1;
    }
    if (pl->n_out == 0 || !pl->mat)
        return 0; /* noop plan: no tail to duel over, nothing to install */

    size_t lens[64];
    int ndec = 0, nlarge = 0;
    duel_lengths(g, lens, &ndec, &nlarge);
    int nlen = ndec + nlarge;
    if (ndec == 0) {
        snprintf(err, errlen, "tail duel: empty decisive set (group=%zu)",
                 g);
        return -1;
    }
    size_t maxlen = lens[nlen - 1]; /* large lengths are last and largest */

    int n_in = pl->n_in, n_out = pl->n_out;
    uint8_t **ib = malloc(sizeof(uint8_t *) * (size_t)n_in);
    uint8_t **ob = malloc(sizeof(uint8_t *) * (size_t)n_out);
    uint8_t **rb = malloc(sizeof(uint8_t *) * (size_t)n_out);
    const void **in = malloc(sizeof(void *) * (size_t)n_in);
    void **out = malloc(sizeof(void *) * (size_t)n_out);
    double *med = malloc(sizeof(double) * 2 * (size_t)nlen);
    double *csn = malloc(sizeof(double) * 2 * (size_t)nlen);
    if (!ib || !ob || !rb || !in || !out || !med || !csn) {
        free(ib);
        free(ob);
        free(rb);
        free(in);
        free(out);
        free(med);
        free(csn);
        snprintf(err, errlen, "out of memory");
        return -1;
    }
    for (int i = 0; i < n_in; i++) {
        if (posix_memalign((void **)&ib[i], 64, maxlen))
            abort();
        in[i] = ib[i];
    }
    for (int j = 0; j < n_out; j++) {
        if (posix_memalign((void **)&ob[j], 64, maxlen))
            abort();
        if (posix_memalign((void **)&rb[j], 64, maxlen))
            abort();
        out[j] = ob[j];
    }

    int rc = 0;
    version_t *vs[2] = { twin, general };

    /* 1. correctness before timing: both versions against the scalar
     * reference at every duel length. A wrong twin fails compilation. */
    for (int vi = 0; vi < 2 && rc == 0; vi++) {
        const plan_t *vpl = vs[vi]->plan ? vs[vi]->plan : pl;
        uint64_t s = 0x7a11d3e1b00b5ULL + (uint64_t)vi;
        for (int li = 0; li < nlen && rc == 0; li++) {
            size_t len = lens[li];
            for (int k = 0; k < n_in; k++)
                for (size_t b = 0; b < len; b++)
                    ib[k][b] = (uint8_t)xrng(&s);
            for (int k = 0; k < n_out; k++)
                memset(ob[k], 0xA5, len);
            vs[vi]->fn(in, out, len);
            ref_run(vpl, (const uint8_t **)ib, rb, len);
            for (int k = 0; k < n_out && rc == 0; k++)
                if (memcmp(ob[k], rb[k], len) != 0) {
                    snprintf(err, errlen,
                             "tail duel: %s produced wrong output at "
                             "len=%zu (scalar reference mismatch)",
                             vs[vi]->desc, len);
                    rc = -1;
                }
        }
    }

    /* 2. timing: interleaved rounds; small lengths are batched (reps
     * consecutive calls per sample) so each sample is ~10us+ and the
     * timer SNR stays sane. The batching loop is identical for both
     * versions, so its overhead cancels in the head-to-head. */
    int iters = opts->bench_iters > 0 ? opts->bench_iters : 11;
    double *ts = NULL;
    if (rc == 0) {
        ts = malloc(sizeof(double) * 2 * (size_t)iters);
        if (!ts) {
            snprintf(err, errlen, "out of memory");
            rc = -1;
        }
    }
    for (int li = 0; li < nlen && rc == 0; li++) {
        size_t len = lens[li];
        size_t reps = 65536 / (len ? len : 1);
        if (reps < 1)
            reps = 1;
        if (reps > 100000)
            reps = 100000;
        for (int vi = 0; vi < 2; vi++)
            for (int r = 0; r < 3; r++)
                for (size_t k = 0; k < reps; k++)
                    vs[vi]->fn(in, out, len);
        for (int r = 0; r < iters; r++)
            for (int vi = 0; vi < 2; vi++) {
                double t0 = now_sec();
                for (size_t k = 0; k < reps; k++)
                    vs[vi]->fn(in, out, len);
                ts[(size_t)vi * (size_t)iters + r] =
                    (now_sec() - t0) / (double)reps;
            }
        for (int vi = 0; vi < 2; vi++) {
            double m, q1, q3;
            sample_stats(ts + (size_t)vi * (size_t)iters, iters, &m, &q1,
                         &q3);
            med[(size_t)vi * (size_t)nlen + li] = m;
            csn[(size_t)vi * (size_t)nlen + li] =
                m > 0 ? (q3 - q1) / m : 0;
        }
    }
    free(ts);

    int install = 0;
    double window = 0.015;
    if (rc == 0) {
        /* 3. decide: strict tail win outside the noise window, plus no
         * large-size regression. */
        window = noise_window(csn, 2, nlen, NULL);
        install = tail_duel_decide(med, med + nlen, ndec, med + ndec,
                                   med + nlen + ndec, nlarge, window);
        /* 4. twin GB/s from the duel (tail-length traffic), for
         * dispatcher ranking. The general version is untouched: its
         * gbps stays the large-size measurement. */
        double tot = 0, ttot = 0;
        for (int li = 0; li < nlen; li++) {
            tot += (double)(n_in + n_out) * (double)lens[li];
            ttot += med[li];
        }
        twin->gbps = ttot > 0 ? tot / ttot / 1e9 : 0;
        twin->gbps_measured = 1;
    }

    if (verbose) {
        fprintf(stderr,
                "tail-duel: %d decisive + %d large lengths, %d interleaved "
                "iters, window=%.1f%%\n",
                ndec, nlarge, iters, window * 100.0);
        for (int li = 0; li < nlen; li++) {
            double bytes = (double)(n_in + n_out) * (double)lens[li];
            /* med[0..nlen) is the twin (vs[0]); med[nlen..) is general */
            double tm = med[li] > 0 ? bytes / med[li] / 1e9 : 0;
            double gm = med[nlen + li] > 0 ? bytes / med[nlen + li] / 1e9
                                           : 0;
            fprintf(stderr, "  len=%-6zu twin %8.2f GB/s  general %8.2f "
                            "GB/s%s\n",
                    lens[li], tm, gm, li == ndec ? "  [large gate]" : "");
        }
        if (rc == 0)
            fprintf(stderr, "tail-duel: %s\n",
                    install ? "INSTALL (strict tail win, no regression)"
                            : "DROP (no strict win or large regression)");
    }

    for (int i = 0; i < n_in; i++)
        free(ib[i]);
    for (int j = 0; j < n_out; j++) {
        free(ob[j]);
        free(rb[j]);
    }
    free(ib);
    free(ob);
    free(rb);
    free(in);
    free(out);
    free(med);
    free(csn);
    return rc ? -1 : install;
}

int heuristic_select(version_t *vs, int nv, const plan_t *pl)
{
    (void)pl;
    /* primary: highest ISA (finer grains, more throughput). Among equal
     * ISA: roofline-ish estimate cycles/byte = max(uops/3.5, mem/64). */
    int winner = 0;
    for (int i = 1; i < nv; i++) {
        int a = vs[i].cand.isa, b = vs[winner].cand.isa;
        if (a != b) {
            if (a > b)
                winner = i;
            continue;
        }
        double sa = fmax(vs[i].uops_per_byte / 3.5, vs[i].mem_per_byte / 64.0);
        double sb = fmax(vs[winner].uops_per_byte / 3.5,
                         vs[winner].mem_per_byte / 64.0);
        if (sa < sb - 1e-9)
            winner = i;
    }
    /* fill a rough GB/s estimate (3 GHz reference) for display */
    for (int i = 0; i < nv; i++) {
        double s = fmax(vs[i].uops_per_byte / 3.5, vs[i].mem_per_byte / 64.0);
        vs[i].gbps = s > 0 ? 3.0 / s : 0;
    }
    return winner;
}
