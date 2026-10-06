/* Calibrated prefetch distance: one-time machine probe + analytical
 * per-kernel distance. See prefetch_calib.h for the design contract. */
#include "prefetch_calib.h"

#include <math.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---------------- one-time probe ---------------- */

/* Chase buffer: 64 MB, well above the guest-visible L3 (the dev VM
 * reports 32 MB), so the pointer chase measures last-level/DRAM
 * latency, not L1/L2. See the virtualization caveat on
 * prefetch_calib_ensure: on a VM the host LLC may still be larger
 * than this buffer. */
#define CHASE_BYTES (64u << 20)
#define CHASE_LINE 64u
/* 4 rounds x 16384 dependent hops; each round is ~2-4 ms of chase.
 * The minimum across rounds rejects descheduling inflation (see
 * probe_latency_ns). */
#define CHASE_HOPS (1u << 14)
#define CHASE_ROUNDS 4
/* Memset sweep: 32 MB streaming writes, 3 timed reps, best taken. */
#define STREAM_BYTES (32u << 20)
#define STREAM_REPS 3

static double now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

/* Volatile sinks: keep the probe loops live (a store to a volatile is
 * observable, so the timed loads/memsets cannot be eliminated).
 * File scope: function-local static volatiles trip
 * -Wunused-but-set-variable under -Werror. */
static volatile size_t chase_sink;
static volatile uint8_t stream_sink;

/* Returns ns per hop, or -1 on failure. Runs several short timed
 * rounds and takes the minimum: on a loaded or shared machine a long
 * single region is inflated by descheduling (CLOCK_MONOTONIC keeps
 * counting while preempted), and only preemption inflates -- the
 * minimum rejects it.
 *
 * The chase needs a long aperiodic sequence of dependent loads
 * spanning the whole 64 MB buffer (defeating prefetchers). Chasing a
 * random *map* (buf[i] = f(i)) is wrong here: iterating a random map
 * falls into a ~sqrt(N)-length cycle (rho), after which every load is
 * an L2 hit and the probe reports cache latency as DRAM latency.
 * Instead the chase advances a full-period 64-bit LCG state
 * arithmetically (Knuth's constants, odd increment -> period 2^64, so
 * the state never short-cycles) and loads one word per state from the
 * line indexed by the state's top 20 bits:
 *   s = buf[top20(s)*stride] + A*s + B
 * The load's address derives from the previous load's value, so the
 * CPU cannot overlap the loads -- this is the load-use latency -- and
 * the loaded word feeds the next state, keeping the chain live. The
 * top bits of an LCG are its highest-quality bits, so consecutive
 * lines are decorrelated and defeat stride prefetchers. The buffer is
 * zero-filled (calloc + one sequential pre-touch, which also faults
 * all pages outside the timed region); the zero word is a legitimate
 * addend, not a shortcut. */
static double probe_latency_ns(void)
{
    /* The top-20-bit indexing below requires exactly 2^20 lines. */
    _Static_assert(CHASE_BYTES / CHASE_LINE == (1u << 20),
                   "chase buffer must hold exactly 2^20 lines");
    const size_t stride = CHASE_LINE / sizeof(uint64_t); /* 8 */
    uint64_t *buf = calloc(CHASE_BYTES, 1);
    if (!buf)
        return -1.0;
    memset(buf, 0, CHASE_BYTES); /* pre-touch all pages */
    double best = -1.0;
    for (int r = 0; r < CHASE_ROUNDS; r++) {
        /* Nonzero seed: the LCG's zero state is a fixed point of the
         * address stream (top20(0) = line 0 forever). */
        uint64_t s = 0x123456789ABCDEF1ull + (uint64_t)r;
        double t0 = now_ns();
        for (uint32_t h = 0; h < CHASE_HOPS; h++)
            s = buf[((size_t)(s >> 44)) * stride] +
                s * 6364136223846793005ull + 1442695040888963407ull;
        double t1 = now_ns();
        chase_sink = s;
        if (t1 > t0) {
            double ns = (t1 - t0) / (double)CHASE_HOPS;
            if (best < 0.0 || ns < best)
                best = ns;
        }
    }
    free(buf);
    return best;
}

/* Returns GB/s, or -1 on failure. */
static double probe_bandwidth_gbs(void)
{
    uint8_t *buf = malloc(STREAM_BYTES);
    if (!buf)
        return -1.0;
    /* Pre-touch pages outside the timed region. */
    memset(buf, 0, STREAM_BYTES);
    double best = -1.0;
    for (int r = 0; r < STREAM_REPS; r++) {
        double t0 = now_ns();
        memset(buf, 0xA5, STREAM_BYTES);
        double t1 = now_ns();
        /* Volatile read forces the memset to have happened. */
        stream_sink = buf[STREAM_BYTES - 1];
        if (t1 > t0) {
            double gbs =
                ((double)STREAM_BYTES / 1e9) / ((t1 - t0) / 1e9);
            if (best < 0.0 || gbs > best)
                best = gbs;
        }
    }
    free(buf);
    return best;
}

/* Sanity envelope for a plausible machine: latency in [1 ns, 100 us],
 * bandwidth in (0, 10 TB/s]. Outside it the probe misfired (broken
 * clock, swapped-in buffer, ...) -- fail closed to invalid rather
 * than feeding garbage into the distance formula. */
static int sane(double lat_ns, double bw_gbs)
{
    return lat_ns >= 1.0 && lat_ns <= 1e5 && bw_gbs > 0.0 &&
           bw_gbs <= 1e4;
}

static prefetch_calib_t run_probe(void)
{
    prefetch_calib_t c;
    c.latency_ns = probe_latency_ns();
    c.bandwidth_gbs = probe_bandwidth_gbs();
    c.valid = sane(c.latency_ns, c.bandwidth_gbs);
    if (!c.valid) {
        c.latency_ns = 0.0;
        c.bandwidth_gbs = 0.0;
    }
    return c;
}

/* ---------------- cached state ---------------- */

static atomic_int g_state = 0; /* 0 = idle, 1 = probing, 2 = done */
static prefetch_calib_t g_calib;

/* Test-only injection (see header). When on, ensure() serves it. */
static int g_inject_on = 0;
static prefetch_calib_t g_inject;

const prefetch_calib_t *prefetch_calib_ensure(void)
{
    if (g_inject_on)
        return &g_inject;
    int st = atomic_load(&g_state);
    if (st == 2)
        return &g_calib;
    int expected = 0;
    if (atomic_compare_exchange_strong(&g_state, &expected, 1)) {
        g_calib = run_probe();
        atomic_store(&g_state, 2);
        return &g_calib;
    }
    /* Another thread is probing (~10 ms); spin until it publishes. */
    while (atomic_load(&g_state) != 2)
        ;
    return &g_calib;
}

void prefetch_calib_inject(double latency_ns, double bandwidth_gbs,
                           int valid)
{
    /* Serves exactly what is injected, including valid=0 (which
     * exercises the fail-closed path in prefetch_calib_distance_for).
     * Sticky for process lifetime: there is no restore-to-real-probe;
     * tests must run real-probe checks before injecting (mirrors the
     * no-clear convention of rj_uarch_override in cpu.h). */
    g_inject.latency_ns = latency_ns;
    g_inject.bandwidth_gbs = bandwidth_gbs;
    g_inject.valid = valid;
    g_inject_on = 1;
}

/* ---------------- analytical distance ---------------- */

size_t prefetch_calib_distance_for(const plan_t *pl, const cand_t *cand,
                                   const rj_cpu_info *cpu)
{
    const prefetch_calib_t *cb = prefetch_calib_ensure();
    if (!cb->valid)
        return 0;
    int U = cand->U;
    int n_streams = pl->n_in;
    if (U <= 0 || n_streams <= 0)
        return 0;
    /* Effective vector bytes for this candidate. VL-agnostic ISAs read
     * the runtime length; the 16-byte fallbacks mirror the CPU
     * detection defaults (riscv/cpu.c falls back to 16 = RVV 1.0
     * minimum; SVE's architectural minimum is 16). */
    int vl = isa_vl(cand->isa);
    if (cand->isa == ISA_SVE)
        vl = cpu->sve_vl > 0 ? cpu->sve_vl : 16;
    else if (cand->isa == ISA_RVV) {
        int lmul = cand->lmul ? cand->lmul : 1;
        int vlen = cpu->rvv_vlen > 0 ? cpu->rvv_vlen : 16;
        vl = lmul * vlen;
    }
    if (vl <= 0)
        vl = 16;
    /* Little's-law heuristic: bytes that must be in flight to hide the
     * calibrated latency at the calibrated bandwidth, expressed as
     * loop iterations at this kernel's all-streams stride. GB/s is
     * bytes/ns, so latency_ns * bandwidth_gbs is bytes. */
    double in_flight = cb->latency_ns * cb->bandwidth_gbs;
    double stride = (double)U * (double)vl * (double)n_streams;
    long iters = (long)ceil(in_flight / stride);
    if (iters < 1)
        iters = 1;
    if (iters > 64)
        iters = 64;
    double dist = (double)iters * (double)U * (double)vl;
    /* ISA encodability (ranges documented on rj_options.prefetch_distance
     * in include/raidjit.h and enforced by the backends at build time;
     * rounding here only avoids emitting doomed twins). */
    switch (cand->isa) {
    case ISA_NEON:
    case ISA_SVE:
        /* PRFM PLDL1STRM: imm12 * 8 = 0..32760. */
        if (dist > 32760.0)
            dist = 32760.0;
        dist = floor(dist / 8.0) * 8.0;
        break;
    case ISA_RVV:
        /* prefetch.r: 32..2016 in 32-byte steps. Below one cache line
         * a prefetch is noise: decline the twin. */
        if (dist < 32.0)
            return 0;
        if (dist > 2016.0)
            dist = 2016.0;
        dist = floor(dist / 32.0) * 32.0;
        break;
    default:
        /* x86 PREFETCHT0: any byte distance encodable. */
        break;
    }
    return (size_t)dist;
}

size_t prefetch_calib_dist_arg(const rj_options *o)
{
    if (!o->benchmark)
        return 0;
    if (o->prefetch_distance)
        return o->prefetch_distance;
    return RJ_PREFETCH_AUTO;
}
