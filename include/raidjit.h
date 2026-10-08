/*
 * raidjit - JIT-compiled RAID encode/decode for the host CPU.
 *
 * Generates machine code specialized to the exact operation (encode or
 * reconstruct-after-failure) and the exact RAID topology, using the best
 * ISA available on the current CPU and the best algorithm for that
 * operation/topology pair. Optionally benchmarks candidate versions and
 * picks the fastest (ties broken by uop-cache / memory efficiency).
 */
#ifndef RAIDJIT_H
#define RAIDJIT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------- CPU detection ---------------- */

typedef struct {
    int sse2;        /* baseline x86-64, always 1 in practice */
    int ssse3;       /* pshufb */
    int avx;
    int avx2;
    int avx512f;
    int avx512bw;    /* byte/word shuffles */
    int avx512vl;
    int vbmi;        /* AVX-512 VBMI (vpermi2b; leaf 7 ECX bit 1) */
    int bitalg;      /* AVX-512 BITAlg (vpshufbitqmb; leaf 7 ECX bit 12).
                      * Distinct from VBMI: e.g. Cannon Lake has VBMI
                      * without BITAlg, so a BITAlg instruction must never
                      * be gated on vbmi alone. Added for track I1: the
                      * VPSHUFBITQMB bit-gather was evaluated against the
                      * SDM and rejected as a GF-multiply twin (it selects
                      * bits but performs no XOR/parity, so it cannot do an
                      * 8x8 bit-matrix multiply; see
                      * hidden_files/brainstorm_isa_uarch_2026-10-06.md
                      * item 1 and the I1/I2 lane's final report). The bit
                      * stays as the correct gate for any future BITAlg
                      * use. */
    int gfni;        /* Galois-field affine instructions (needs AVX512) */
    int pclmulqdq;   /* CLMUL carryless multiply (CRC-32 folding);
                      * CPUID.1:ECX[1]. No XCR0 state needed. */
    int vpclmulqdq;  /* AVX-512 vector carryless multiply (4x64-bit);
                      * CPUID.7:ECX[10]. No XCR0 state needed. */
    int sse42;       /* SSE4.2 (crc32 instruction); CPUID.1:ECX[20] */
    int cet_ss;      /* CPU supports CET shadow stack */
    /* AArch64 */
    int neon;        /* Advanced SIMD, baseline for AArch64 */
    int sve;         /* Scalable Vector Extension */
    int sve2;        /* SVE2 */
    int sve_vl;      /* SVE vector length in bytes (0 if no SVE) */
    int crc32;       /* ARMv8 CRC32 instructions; HWCAP_CRC32 (AT_HWCAP bit 7) */
    int pmull;       /* NEON PMULL (carryless multiply); HWCAP_PMULL (bit 4) */
    int sha3;        /* SHA3 incl. EOR3; HWCAP_SHA3 (AT_HWCAP bit 17) */
    /* RISC-V */
    int rvv;         /* Vector extension */
    int rvv_vlen;    /* VLEN in bytes (0 if no RVV) */
    int rvv_zicbop;  /* Zicbop cache-block prefetch hints (prefetch.r/w/i) */
    int rvv_zvbc;    /* Zvbc vector carryless multiply (vclmul.vv /
                      * vclmulh.vv; SEW=64 only per the ratified Vector
                      * Crypto spec). Informational for the SEW=8 twin:
                      * Zvbc does NOT imply Zvbc32e. */
    int rvv_zvkg;    /* Zvkg vector crypto (vghsh/vgmul). Informational
                      * only -- NOT a gate for the MUL_ZVBC twin:
                      * vclmulh.vv is Zvbc, not Zvkg. */
    int rvv_zvbc32e; /* Zvbc32e: vclmul.vv/vclmulh.vv for SEW=32/16/8
                      * (fast-track vector-crypto-extra §33; independent
                      * of Zvbc, legal standalone). THE gate for the
                      * SEW=8 MUL_ZVBC twin: until a kernel/QEMU reports
                      * the token the twin stays off (fail closed). */
    int rvv_zvbb;    /* Zvbb vector bitmanip (vandn.vv, vbrev8.v, ...;
                      * ratified 1.0). Detected for future use; no
                      * backend path consumes it yet (see I5 analysis:
                      * the nibble split needs 2 instructions minimum --
                      * one per nibble -- and neither vandn nor vbrev8
                      * can fuse them, so no profitable twin exists). */
    int rvv_zicond;  /* Zicond integer conditional (czero.eqz/nez) */
    int zbc;         /* Zbc scalar carryless multiply (clmul/clmulh) */
    int p_ext;       /* Packed SIMD P-extension */
    int p_version;   /* P-extension draft version: 5=0.5, 9=0.9.x, 20=0.20 */
    char brand[64];
} rj_cpu_info;

void rj_cpu_detect(rj_cpu_info *out);

/* ---------------- topology ----------------
 * A topology is ndata data disks + npar parity disks.
 * parity[j] = XOR_i ( data[i] * M[j][i] ), arithmetic in GF(2^8).
 * rj_topo_rs() uses the default Vandermonde matrix in "Horner order":
 *   M[j][i] = g^( j*(ndata-1-i) ), g=2,
 * which lets the JIT use the optimal Horner accumulation.
 *
 * MDS envelope of the default matrix: g=2 has order 51 in GF(2^8)/0x11B,
 * so the matrix is a true Reed-Solomon (every <=npar erasure set
 * recoverable) only for ndata <= 51 and npar <= 3. rj_topo_raid6() and
 * rj_topo_rs() return NULL outside this envelope. Wider MDS topologies
 * (e.g. Cauchy matrices, MDS for ndata+npar <= 255) are available via
 * rj_topo_custom().
 *
 * Codegen limits: the JIT keeps all accumulators in vector registers, so
 * very high parity counts may have no encodable candidate on a given CPU
 * (rj_compile then fails cleanly with "no candidates for this CPU/plan";
 * never wrong code). Worst-case ceilings at unroll U=1: Horner/nibble
 * needs 3*npar+6 <= nregs (npar <= 3 on SSE/AVX2, <= 8 on AVX-512),
 * Horner/GFNI needs 2*npar+2 <= nregs (<= 7 / <= 15), mulacc/nibble
 * needs npar+7 <= nregs (<= 9 / <= 25). Recovery of nf failed disks
 * needs the same with nf in place of npar.
 */

typedef struct rj_topology rj_topology;

rj_topology *rj_topo_raid0(int ndata);
rj_topology *rj_topo_raid1(void);                       /* 1 data + 1 mirror */
rj_topology *rj_topo_raid5(int ndata);                  /* npar = 1 */
rj_topology *rj_topo_raid6(int ndata);  /* npar = 2; dispatches to the
                                         * benchmark-selected default code
                                         * per docs/xor_code_default.md
                                         * (RS at every ndata on the
                                         * measured host). When the winner
                                         * is an XOR code the topology uses
                                         * the expanded row model below. */
rj_topology *rj_topo_raid6_rs(int ndata); /* explicit RS RAID-6, npar = 2:
                                          * the pre-dispatch behavior */
/* XOR array codes for RAID-6 (RDP, EVENODD, Liberation): binary (XOR-only)
 * MDS codes with 2 parity disks. Each data disk holds R rows of bytes and
 * the diagonal parities mix bytes ACROSS rows, so the topology exposes
 * every ROW as a logical disk ("expanded row model"): with k data disks
 * of R rows, ndata = k*R, npar = 2*R; in[d*R + r] is row r of data disk
 * d, out[r] is P parity row r, out[R + r] is Q parity row r. Every parity
 * row is a pure XOR of input rows. (Full pointer contract and the
 * whole-disk-failure index mapping: docs/xor_codes.md §1.1.) Each returns
 * NULL when the parameters are outside the code's validity envelope
 * (docs/xor_codes.md §2) or on OOM. */
rj_topology *rj_topo_rdp(int p);        /* p prime, p >= 3; k = p-1 data disks */
rj_topology *rj_topo_evenodd(int p);    /* p prime, p >= 3; k = p data disks */
rj_topology *rj_topo_liberation(int k); /* k >= 2; w = smallest prime >= max(k,3),
                                         * R = w rows */
/* Rows per data disk R for an XOR-code topology, 0 for anything else
 * (RS/RAID-5/custom/...). Lets callers map whole-disk failures to the
 * expanded index lists rj_op_recover needs. */
int rj_xorcode_rows(const rj_topology *t);
rj_topology *rj_topo_rs(int ndata, int npar);
/* Locally recoverable codes (A6; see docs/lrc.md). Matrix row order:
 * the h global (Vandermonde) rows first, then the g local (XOR) rows.
 * A single failure inside a local group recovers from its group-mates
 * (pure XOR, no multiplies); up to h failures anywhere recover from
 * the global parities.
 *
 * rj_topo_lrc(k, h, r): the general local-parity mechanism -- k data
 *   disks, h global parities, locality r. The data disks are
 *   partitioned into g = ceil(k/r) groups as equal as possible (each
 *   of size <= r), each group with its own XOR local parity; npar =
 *   h + g. Returns NULL unless k >= 2, h >= 1, h <= 50, r >= 2,
 *   k <= 51 (the Vandermonde envelope: 2 has order 51 in GF(2^8)/0x11B,
 *   so the nodes 2^0..2^50 are distinct, and exponents above 50 would
 *   make global row 51 the all-ones row the design skips, then repeat
 *   with period 51) and k + h + g <= RJ_MAX_DISKS, or on OOM.
 * rj_topo_lrc_azure(): the canonical Azure (12,2,2) instance -- 12
 *   data, 2 global parities, 2 local parities with locality 6.
 * rj_topo_lrc_optimal(k, h, r): the distance-optimal
 *   parameterization -- like rj_topo_lrc but requires r | k (NULL
 *   otherwise), under which the code attains the Singleton-type bound
 *   d = n-k-ceil(k/r)+2 = h+2. The k <= 51, h <= 50 Vandermonde
 *   envelope applies as well. Optimality is verified exhaustively per
 *   instance in tests/test_lrc.c (every (h+1)-erasure pattern
 *   recoverable, an (h+2) non-recoverable witness exhibited); the
 *   verified instances are recorded in docs/lrc.md.
 *
 * The global rows are Vandermonde with exponents 1..h (exponent 0
 * skipped: it would duplicate the all-ones local row on single-group
 * column sets). They are deliberately NOT the matrix_search_default
 * matrices rj_topo_rs uses: weight-minimal matrices carry no guarantee
 * that the global rows compose with the all-ones local rows into
 * full-rank joint systems. */
rj_topology *rj_topo_lrc(int k, int h, int r);
rj_topology *rj_topo_lrc_azure(void);
rj_topology *rj_topo_lrc_optimal(int k, int h, int r);
rj_topology *rj_topo_custom(int ndata, int npar,
                            const uint8_t *matrix /* npar*ndata, row-major */);
void rj_topo_free(rj_topology *t);
int  rj_topo_ndata(const rj_topology *t);
int  rj_topo_npar(const rj_topology *t);

/* ---------------- operation ---------------- */

typedef struct {
    int  is_recover;       /* 0 = encode, 1 = recover */
    int  is_parity_update; /* 1 = parity update (partial stripe) */
    int *failed;           /* owned copy; disk indices 0..ndata+npar-1, sorted */
    int  nfailed;
    int *changed;          /* owned copy; data-disk indices 0..ndata-1, sorted */
    int  nchanged;
    int  oom;              /* 1 = constructor failed to allocate (not empty) */
} rj_op;

rj_op rj_op_encode(void);
rj_op rj_op_recover(const int *failed, int nfailed);
/* Partial-stripe parity update: only the data disks in changed[] (indices
 * 0..ndata-1, nchanged >= 1) were rewritten. new_p[j] =
 *   old_p[j] XOR (over i in changed) ((old[i] XOR new[i]) * M[j][changed[i]])
 * in GF(2^8)/0x11B. The op owns a sorted copy of changed[] (duplicates are
 * removed during planning). */
rj_op rj_op_parity_update(const int *changed, int nchanged);
void  rj_op_free(rj_op *op);

/* ---------------- options ---------------- */

typedef struct {
    int    benchmark;         /* run selection benchmarks (default 1) */
    size_t bench_bytes;       /* stripe bytes per buffer in benchmarks;
                               * 0 = default (1 MiB); clamped to 16 MiB */
    int    bench_iters;       /* timed iterations per candidate;
                               * <= 0 = default (11) */
    int    emit_endbr;        /* endbr64 at entry (NOP on non-CET CPUs) */
    int    emit_shadow_stack; /* setssbsy/clrssbsy; needs OS-enabled CET */
    int    loop_align;        /* 0 = auto (64/32/16 by vector width);
                               * otherwise must be a power of two */
    int verbose;           /* print candidate table to stderr */
    size_t nt_threshold;   /* 0 (default) = never use non-temporal stores.
                            * >0 = add an NT-store twin of every candidate
                            * and let the benchmark pick empirically.
                            * x86: MOVNTDQ/VMOVNTDQ, no fence -- the caller
                            * must SFENCE after the call before reading
                            * the outputs. AArch64: STNP/STNT1B, no DMB --
                            * the caller must DMB ISHST. RISC-V: Zihintntl
                            * NTL.ALL hint before each vector store; the
                            * hint does not change memory ordering, so no
                            * caller fence is needed. The magnitude is
                            * currently unused (reserved): it is a boolean
                            * enable, not a length cutoff -- a winning NT
                            * twin applies to every runtime length. NT
                            * stores help only for large, cold, streaming
                            * writes (stripe working set well above the
                            * LLC); for small or immediately-reused stripes
                            * they are slower. This is an enable, not a
                            * guarantee: the benchmark decides per CPU. */
    size_t prefetch_distance; /* 0 (default) = analytical default (I4):
                            * each candidate gets a per-kernel prefetch
                            * distance computed at plan time from the
                            * once-per-machine calibration (see rj_init)
                            * and the kernel's stride (U x vector bytes x
                            * input streams) -- no runtime cost, no
                            * per-kernel search. The distance is a
                            * heuristic, NOT a measured optimum; the
                            * compile-time benchmark remains the final
                            * arbiter, and the distance-0 (no-prefetch)
                            * twin is always enumerated alongside.
                            * >0 = manual uniform override: every prefetch
                            * twin uses this distance (pre-I4 behavior).
                            * x86: PREFETCHT0; AArch64: PRFM PLDL1STRM (the
                            * encodable range is 0..32760, multiple of 8
                            * -- other distances fail the twin build
                            * cleanly); RISC-V:
                            * Zicbop prefetch.r (needs rvv_zicbop; the
                            * encodable range is 32..2016, multiple of 32
                            * -- other distances fail the twin build
                            * cleanly). Useful only when the stripe working
                            * set exceeds what the hardware prefetcher
                            * tracks (many concurrent sequential streams on
                            * large cold buffers); for in-cache or
                            * few-stream workloads it is pure overhead.
                            * Like NT, it participates in benchmarking only
                            * as a separate candidate twin -- measurement
                            * decides. */
} rj_options;

void rj_options_default(rj_options *o);

/* ---------------- JIT ---------------- */

typedef struct rj_jit rj_jit;

/* in[]/out[] layout (caller contract — unchecked for speed):
 *   encode:  in = ndata data pointers; out = npar parity pointers
 *            (out may be NULL when npar == 0; fn is then a no-op).
 *   recover: in = surviving disk pointers, increasing disk-index order;
 *            out = one pointer per failed disk, increasing failed-index order.
 *   parity update: in = [old_0..old_{nc-1}, new_0..new_{nc-1},
 *                  oldpar_0..oldpar_{np-1}] (2*nc+np pointers, nc = nchanged,
 *                  np = npar); old_k/new_k are the pre/post images of
 *                  changed[k] (changed[] sorted by the op), oldpar_j the
 *                  current parity j. out = np new-parity pointers.
 *            out[j] MAY alias in[2*nc+j] (in-place parity update); no
 *            other aliasing is allowed.
 * len = stripe bytes per disk; arbitrary, need not be a vector multiple.
 * Every buffer must be 64-byte aligned and no two buffers may overlap or
 * alias (in[] vs out[], or outputs vs each other), except for the
 * documented in-place parity-update exception above. These conditions are
 * unchecked for speed: a misaligned buffer will normally raise #GP on an
 * aligned vector load/store, and aliasing buffers silently compute wrong
 * results. The 64-byte contract is load-bearing -- codegen uses aligned
 * vector loads/stores (MOVDQA / VMOVDQA / VMOVDQA32) throughout.
 */
typedef void (*rj_fn)(const void *in[], void *out[], size_t len);

/* First-touch process initialization (I4): runs the prefetch
 * calibration probe (~10 ms, once per process; later calls are free).
 * Optional: the probe also runs lazily at the first rj_compile that
 * needs an analytical prefetch distance, so calling rj_init is never
 * required for correctness -- it just moves the probe cost out of the
 * first compile. */
void rj_init(void);

rj_jit *rj_compile(const rj_topology *topo, const rj_op *op,
                   const rj_options *opts, char *err, size_t errlen);
rj_fn   rj_get_fn(const rj_jit *j);
void    rj_free(rj_jit *j);

/* Batched multi-stripe fast path (Tier-1 driver, queue item #9).
 *
 * One call processes S stripes that share j's topology and op (encode,
 * recover with one fixed failed set, or parity update with one fixed
 * changed set -- a dead disk is dead for all stripes, so one failed
 * set per recover batch matches the physical reality). The driver is a
 * C loop calling j's dispatched rj_fn once per stripe: the same jitted
 * code (and the same length dispatch) rj_get_fn(j) returns. Zero new
 * codegen. rj_jit stays specialized on (topology, op, CPU); S is per
 * call, so one compiled unit serves every batch size.
 *
 * Pointer layout: flat, row-major over stripes. Stripe s owns
 *   in[s*nbufs_in .. (s+1)*nbufs_in) and
 *   out[s*nbufs_out .. (s+1)*nbufs_out),
 * where nbufs_in/nbufs_out are exactly the per-stripe counts from the
 * rj_fn contract for j's op (encode: ndata/npar; recover:
 * nsurv/nfailed; parity update: 2*nchanged+npar/npar). Only the pointer
 * arrays are flat; the data buffers keep whatever layout the caller
 * already uses. len is one shared stripe length for the whole batch,
 * same meaning as rj_fn's len (arbitrary, need not be a vector
 * multiple); mixed lengths are handled by grouping equal-len stripes
 * into separate calls.
 *
 * Unchecked contract, exactly like rj_fn: violations (S == 0, NULL in,
 * NULL out when nbufs_out > 0, misaligned buffers, aliasing across
 * stripes, overlapping buffers) are undefined behavior. Stripes
 * complete in index order 0..S-1. void return: the driver loop cannot
 * fail partway (no allocation, no syscalls inside).
 *
 * S sizing guidance (no hard cap in code): the driver chunks internally
 * so its auxiliary footprint stays bounded for any S; batches up to
 * ~2^20 stripes are comfortably handled. Keep each batch's working set
 * LLC-resident when you care about throughput -- stripe-major data
 * layouts stream better than batch-major-per-disk ones past the LLC.
 *
 * NT-store fence contract, inherited from rj_fn: the caller must fence
 * once per batched call (not per stripe) before reading the outputs --
 * SFENCE on x86, DMB ISHST on AArch64; no fence needed on RISC-V.
 * There is no machine-readable NT signal and no batch-aware
 * benchmarking in v1 (the driver inherits the single-stripe benchmark
 * winner), so fence unconditionally whenever the winning kernel may
 * use NT stores.
 *
 * Lifetime: the entry point shares j's lifetime. rj_free releases it; a
 * call through the entry point after rj_free is dangling (the slot may
 * be reused by a newer rj_jit, turning a loud fault into silent wrong
 * output). Always re-fetch via rj_get_fn_batched after re-compiling. */
typedef void (*rj_fn_batched)(const void *const *in, void *const *out,
                              size_t S, size_t len);

/* The batched entry point for j. NULL when j is NULL or on driver
 * allocation/slot failure -- which degrades to single-stripe only:
 * rj_get_fn(j) keeps working. */
rj_fn_batched rj_get_fn_batched(const rj_jit *j);

typedef struct {
    char   desc[192];      /* e.g. "raid6 enc horner avx512+gfni U=4" */
    char   isa_name[32];   /* "sse" / "avx2" / "avx512" */
    char   strategy[32];   /* "xor" / "horner" / "mulacc" / "delta" */
    int    unroll;         /* vector-width unroll factor U */
    int    lmul;           /* RVV LMUL multiplier (1/2/4/8); 1 on other ISAs */
    int    gbps_measured;  /* 1 if gbps came from real benchmarks */
    double gbps;           /* measured or heuristically estimated.
                            * Measured values mean: best among the generated
                            * legal candidates under the compile-time
                            * benchmark protocol on this CPU -- not a
                            * portable or absolute performance claim. */
    double uops_per_byte;  /* static model: vector uops per byte */
    double mem_per_byte;   /* static model: bytes moved per byte of stripe */
    size_t code_size;
    size_t const_size;
} rj_info;

void rj_get_info(const rj_jit *j, rj_info *info);

/* ---------------- stable plan IDs ----------------
 * A plan ID is a structured, human-readable, version-prefixed string
 * naming one candidate spec exactly:
 *
 *   rj<V>:<topo>:<op>:<isa>:<kind>:<mul>:u<U>:<twins>:<tail>
 *
 * e.g. "rj1:rs(6,3)#9e3779b97f4a7c15:enc:avx512:mulacc:gfni:u8:ternlog:scalar"
 *
 * Axes, in order: codegen version ("rj1"); topology ("rs(6,3)" plus a
 * 16-hex-digit FNV-1a fingerprint of ndata/npar/matrix -- the name is
 * for humans, the fingerprint is the identity); op ("enc", "pupd0+2",
 * or "rec6.rows0+1+2" naming failed disks and the equation row set);
 * ISA; plan kind (xor/horner/mulacc/delta); multiply method
 * (nibble/gfni/zvbc); unroll ("u8"); twin axes ("-" or "+"-joined
 * nt/pf<N>/ternlog/remat/xorsched/xorbit/mshift/permi2b/lmul<N>); and
 * the effective stage-2 tail configuration ("scalar", or
 * "notail"/"predtail"/"maskedtail" for installed tail twins).
 *
 * The library keeps no persistent state, ever: the ID is a pure
 * function of the candidate spec and the caller owns storage.
 * Typical flow: rj_compile once -> rj_plan_id -> store the string ->
 * rj_compile_pinned on every later start. rj_list_variants enumerates
 * the full pre-prune candidate space for callers that benchmark with
 * their own methodology and then pin the winner's ID.
 *
 * Reproduce contract: the same build, given the same ID, re-JITs the
 * same candidate spec deterministically. Not byte-identical machine
 * code: the candidate is the contract. Codegen-affecting rj_options
 * fields (loop_align, emit_endbr, emit_shadow_stack) still come from
 * the pinned call's opts -- pass identical opts for identical code.
 * benchmark/nt_threshold/prefetch_distance are inert for pinned
 * compile (the ID already pins nt and the prefetch distance).
 *
 * Strict invalidation: the version prefix is bumped on any release
 * that could affect generated code. rj_compile_pinned rejects IDs
 * from other versions with a version-mismatch error -- stale pins are
 * never honored, forcing re-benchmarking after upgrades. There is no
 * best-effort cross-version mode. */

/* Compile the exact plan named by plan_id, bypassing enumeration,
 * pruning, and benchmarking. Returns NULL with err set on any
 * mismatch: malformed/unknown ID, version mismatch, topology or op
 * mismatch vs the passed topo/op, a plan kind the current build would
 * not derive, or a candidate spec the current build cannot construct
 * (wrong CPU features, unbuildable twin, ...). NEVER falls back to
 * heuristics -- a silent fallback would defeat pinning. The returned
 * unit validates against the scalar reference before it is returned
 * (same bar as rj_compile); tail twins named by the ID are rebuilt
 * deterministically (no duels -- their install decisions are part of
 * the pinned spec). */
rj_jit *rj_compile_pinned(const rj_topology *topo, const rj_op *op,
                          const rj_options *opts, const char *plan_id,
                          char *err, size_t errlen);

/* The stable plan ID of a compiled unit (rj_compile or
 * rj_compile_pinned). Returns NULL for a NULL unit. The string is
 * owned by the unit (valid until rj_free); copy it if it must outlive
 * the unit. */
const char *rj_plan_id(const rj_jit *j);

/* One enumerated candidate variant (rj_list_variants). */
typedef struct {
    char plan_id[4096];   /* stable ID (stage-1 candidate; tail=scalar) */
    char desc[192];       /* human-readable, e.g. "raid6 enc mulacc
                           * avx512+gfni U=8+ternlog" */
    int survivor;         /* 1 = survives the library's static assessment,
                           * 0 = pruned (see reason) */
    char reason[160];     /* "" when survivor; otherwise why pruned:
                           * "build failed: ..." (the candidate does not
                           * build on this CPU), "statically dominated by
                           * +ternlog twin ..." (prune_versions), or
                           * "subset shortlisted out ..." (recover: the
                           * A5 parity-row subset did not survive the
                           * static L2 shortlist) */
    double uops_per_byte; /* static model (0 when the build failed) */
    double mem_per_byte;  /* static model (0 when the build failed) */
} rj_plan_variant;
/* NOTE: named rj_plan_variant (not rj_variant): rj_variant is already
 * taken by the length-dispatcher's internal variant type. */

/* Enumerate the FULL pre-prune candidate space for (topo, op): every
 * candidate spec plan_candidates produces (over every A5 parity-row
 * subset for recover), each tagged with its stable plan ID and the
 * library's static assessment (survivor vs pruned + reason). NT and
 * prefetch twins follow the same rule as rj_compile (only when
 * benchmarking with nt_threshold > 0 / prefetch enabled); the static
 * prune assessment applies only when opts->benchmark is set (rj_compile
 * does not prune on the heuristic path either).
 *
 * Returns the total number of variants. When out != NULL, fills up to
 * nout entries in deterministic enumeration order; when nout is
 * smaller than the total, re-call with a larger buffer. Returns -1 on
 * invalid input (null topo, bad op, no candidates). Enumeration builds
 * every candidate once (no benchmarking) to obtain the exact static
 * cost triple -- the same work rj_compile does before pruning. */
int rj_list_variants(const rj_topology *topo, rj_op op,
                     const rj_options *opts,
                     rj_plan_variant *out, int nout);

#ifdef __cplusplus
}
#endif

#endif /* RAIDJIT_H */
