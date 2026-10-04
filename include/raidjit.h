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
    int gfni;        /* Galois-field affine instructions (needs AVX512) */
    int cet_ss;      /* CPU supports CET shadow stack */
    /* AArch64 */
    int neon;        /* Advanced SIMD, baseline for AArch64 */
    int sve;         /* Scalable Vector Extension */
    int sve2;        /* SVE2 */
    int sve_vl;      /* SVE vector length in bytes (0 if no SVE) */
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
    int rvv_zicond;  /* Zicond integer conditional (czero.eqz/nez) */
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
    size_t prefetch_distance; /* 0 (default) = no software prefetch.
                            * >0 = emit a prefetch for each input stream
                            * this many bytes ahead of the current vector
                            * group, once per group iteration. x86:
                            * PREFETCHT0; AArch64: PRFM PLDL1STRM (the
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
                            * as a separate candidate twin when nonzero --
                            * measurement decides. */
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

rj_jit *rj_compile(const rj_topology *topo, const rj_op *op,
                   const rj_options *opts, char *err, size_t errlen);
rj_fn   rj_get_fn(const rj_jit *j);
void    rj_free(rj_jit *j);

typedef struct {
    char   desc[192];      /* e.g. "raid6 enc horner avx512+gfni U=4" */
    char   isa_name[32];   /* "sse" / "avx2" / "avx512" */
    char   strategy[32];   /* "xor" / "horner" / "mulacc" / "delta" */
    int    unroll;         /* vector-width unroll factor U */
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

#ifdef __cplusplus
}
#endif

#endif /* RAIDJIT_H */
