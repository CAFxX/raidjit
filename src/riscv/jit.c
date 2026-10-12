/* Code generation: plan + candidate -> RISC-V (RV64G + RVV 1.0) machine
 * code. Implements jit_build_version() from ../jit.h.
 *
 * Generated function (LP64D calling convention):
 *   void fn(const void *in[], void *out[], size_t len);
 *   a0 = in, a1 = out, a2 = len
 *
 * Counted full-group loop driven by vsetvli (e8, ta, ma; LMUL from the
 * candidate: m1/m2/m4/m8). A group is U * VLMAX bytes where
 * VLMAX = LMUL * vlenb (SEW=8); each main-loop iteration consumes
 * exactly one group with vl pinned at VLMAX, so the per-iteration vl
 * computation (divu+mul, or srli+slli for U in {1,2,4}) collapses into
 * a single prologue divu for the trip count plus one mul/shift for the
 * group size. A sub-group remainder (remaining < U * VLMAX) is handled
 * by a simple loop that processes up to VLMAX bytes per iteration
 * using the single-unroll body; vsetvli computes vl =
 * min(VLMAX, remaining) in hardware, so no division is needed -- there
 * is NO scalar tail. Arbitrary lengths work, including len < VL and
 * (via an entry branch) len == 0. Note that vsetvli with rs1 == x0
 * selects VLMAX rather than 0, hence the explicit len == 0 guard at
 * entry.
 *
 * GF(2^8) multiplies use the nibble method:
 *   lo = x & 0x0f; hi = x >> 4;
 *   out = vrgather(lotab, lo) ^ vrgather(hitab, hi)
 * The 16-byte tables live in vector registers. Every constant-pool
 * entry holds its table replicated to cover a full VLMAX load
 * (lmul * vlenb bytes on this machine's VLEN), so a vle8.v at
 * vl == VLMAX can never over-read the pool entry; vrgather indices
 * are nibble values (0..15), so only the first 16 bytes are ever read
 * and the replicas are semantically invisible. Table loads therefore
 * run at the body's natural vl with no narrowing: Horner row tables
 * are loaded once in the prologue, mulacc/delta column tables per
 * column from the pool. The vsetivli-16 window survives only in the
 * remainder/tail bodies (win=1), where vl can be < 16 and a short load
 * would leave table elements past vl stale. The "VLMAX >= 16"
 * assumption is VLEN >= 128 (full V extension); cpu.c's "zve*" match
 * is optimistic -- on Zve32x with VLEN < 128 the 16-byte table loads
 * would be short (pre-existing limitation, not introduced here).
 *
 * Accumulators are seeded per loop iteration, never across iterations:
 *   - P_XOR / P_MULACC: first contributing column moves/loads into the
 *     accumulator (xor-with-zero is never emitted); rows with no
 *     contributors are zeroed.
 *   - P_HORNER: the i==0 iteration moves the loaded data straight into
 *     each accumulator (acc = 0*c + t = t for any c).
 *   - P_DELTA: accumulators are seeded from the old parities; the delta
 *     (old ^ new) is formed per changed disk and multiplied in.
 *
 * Non-temporal stores: implemented via the ratified Zihintntl extension.
 * An NT candidate emits NTL.ALL (encoded as ADD x0,x0,x5, a HINT that
 * executes as a NOP on every implementation -- no feature gating
 * needed) immediately before each output vse8.v. Unlike x86 MOVNTDQ /
 * AArch64 STNP, the hint does not change memory ordering ("does not
 * alter the architecturally visible effects of the target
 * instruction"), so the jitted code executes no fence and callers need
 * none either -- there is nothing to fence for.
 *
 * Software prefetch: implemented via the ratified Zicbop extension
 * (prefetch.r, 32-byte granularity, GAS-verified encodable range
 * 32..2016). A prefetch twin emits one prefetch.r per input stream
 * per group iteration, prefetch_dist bytes ahead. Emitted only when
 * the CPU reports Zicbop (rj_cpu_info.rvv_zicbop); the planner only
 * enumerates the twin then, and a force-built candidate without Zicbop
 * degrades to temporal. A distance outside 32..2016 (or not a
 * multiple of 32) fails the candidate build cleanly.
 */
#include "emit.h"
#include "../jit.h"
#include "../gf.h"
#include "../xorsched.h"

#include <sys/mman.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <assert.h>

/* constant-pool entry kinds (16-byte nibble tables) */
enum { CK_NIBLO, CK_NIBHI };

/* nominal vector length (bytes) used only for the static cost model;
 * the generated code is VL-agnostic. */
#define NOMVL 16

typedef struct {
    em_t em;
    const plan_t *pl;
    cand_t cand;
    const rj_options *opts;
    int U;
    int pool_lab;
    uint8_t *pool;
    size_t pool_len, pool_cap;
    struct {
        int kind;
        uint8_t c;
        size_t off;
    } *cent;
    int ncent, ccent;
    /* vector register assignment: all fields below hold PHYSICAL vector
     * register numbers (group heads). Logical register r (as counted by
     * plan_need_regs) maps to head r*lmul, so every head is lmul-aligned
     * as the spec requires for LMUL>1. */
    int lmul;  /* RVV LMUL multiplier (1/2/4/8): logical vector register r
                * maps to the lmul-aligned physical group head r*lmul.
                * 0 in cand->lmul means 1. */
    int vlmul; /* vtype LMUL enum (VLMUL_M1/M2/M4/M8) for vsetvli */
    int vd0; /* first per-lane data group head */
    int vd1; /* XOR-schedule 2nd temp (P_XOR xorsched twin only) */
    /* Software-pipelined loop twin (track I8, cand->pipe): physical
     * head of the Dn dataset (U*lmult regs). D is vd0 itself (reused);
     * Dn holds column 0 of the next iteration. -1 when pipe is off. */
    int pipe_dn;
    int tlo, thi, r1, r2, tbl_lo, tbl_hi, mask;
    /* Bit-program (xorbit) twin registers: input-byte reg, two bit-temp
     * regs (mask via vand.vi immediate, no mask reg needed). */
    int xbt_in, xbt_r1, xbt_r2;
    int r3;    /* Zvbc path 3rd carryless temp (q needs lo/hi live) */
    int cvec;  /* Zvbc path: broadcast GF coefficient (per row for
                * HORNER via hreg[2j]; per column for MULACC/DELTA) */
    int kvec;  /* Zvbc path: broadcast 0x1B reduction constant */
    int zvbc;  /* 1 = MUL_ZVBC: carryless-multiply path (needs
                * rvv_zvbc32e for the SEW=8 vclmul.vv/vclmulh.vv
                * encodings); 0 = nibble-table path */
    int tbl;   /* HORNER remat scratch: pool tables reloaded here.
                * Assigned only in remat mode; 0 (from the cg memset)
                * otherwise and never read there. */
    int hoist_tables; /* MULACC/DELTA nibble path: the whole matrix uses
                * a single nontrivial coefficient, so its table pair is
                * loop-invariant and loaded once in the prologue into
                * tbl_lo/tbl_hi (hoist_c) instead of per column. */
    uint8_t hoist_c;  /* the hoisted coefficient (valid iff hoist_tables) */
    int *hreg; /* HORNER: 2 regs per row, -1 if unused */
    /* static cost model: per one loop iteration (nominal) */
    double uops, mem;
    int need_mask; /* nibble vector path needs the 0x0F mask */
    int emit_pf;   /* validated Zicbop prefetch: emit prefetch.r per input
                    * stream per group iteration (exact gate: desc "+pf"
                    * iff >= 1 prefetch is emitted) */
    unsigned pf_off; /* prefetch.r offset: 32..2016, multiple of 32 */
    int notail;      /* length-dispatch twin: omit the strip-mining /
                    * single-chunk remainder tail entirely; only valid
                    * for len % group == 0 (see docs/length_dispatch.md) */
    int maskedtail;  /* length-dispatch twin: replace the strip-mining
                    * remainder loop with an unrolled full-VL masked tail
                    * (TAIL_MASKED). v0 is reserved as the dedicated mask
                    * register: the whole vector map shifts up by one
                    * LMUL-group (see vreg()). Disjoint from notail (and
                    * from the SVE-only predtail, rejected at build). */
    int masked_mem;  /* in-tail flag: data loads/stores go through the
                    * vm=0 masked forms (v0.t). Table loads (c_ldtab/
                    * c_ldtab1) stay unmasked: pool entries are replicated
                    * to full VLMAX and vl == VLMAX in the tail. */
    int tail_lab;    /* maskedtail only: label at the start of the
                    * unrolled masked tail; the counted loop's
                    * empty-iteration branch targets it (the strip-mining
                    * path uses loop_lab instead). Bound at the top of
                    * gen_tail_masked. */
    size_t vlenb;    /* VLEN in bytes on this machine (vlenb CSR), read
                    * once at build time; sizes the replicated
                    * constant-pool table entries (see pool_get) */
    uint32_t twin_applied; /* BUG-COMBO-STATIC-F16: bitmask of TWIN_APPLIED_*
                            * bits, set when each twin's code path is taken. */
    int oom; /* sticky OOM flag: set by pool_get on realloc failure.
              * jit_build_version checks it after codegen and fails
              * cleanly instead of asserting. */
} cg_t;

/* vlenb CSR (0xC22): VLEN in bytes. A hardware constant, readable from
 * U-mode; needed at build time so the length dispatcher knows the true
 * full-group size (U * VLMAX). */
static size_t read_vlenb(void)
{
#ifdef RJ_WASM
    /* Compile-only (WebAssembly, tools/wasm/SPEC.md §S6.2b): no CSR
     * access on wasm32. The single caller is guarded to use the
     * synthetic CPU bundle instead; this definition only keeps the TU
     * compiling. Native builds never define RJ_WASM. */
    return 16;
#else
    size_t v;
    __asm__ volatile("csrr %0, vlenb" : "=r"(v));
    return v;
#endif
}

/* ---------------- constant pool ---------------- */

static size_t pool_get(cg_t *cg, int kind, uint8_t c)
{
    if (cg->oom) return 0; /* sticky OOM: fail fast */
    for (int i = 0; i < cg->ncent; i++)
        if (cg->cent[i].kind == kind && cg->cent[i].c == c)
            return cg->cent[i].off;
    uint8_t lo[16], hi[16];
    gf_build_nibble(c, lo, hi);
    /* Replicated entry: the 16-byte table repeated enough times to
     * cover a full VLMAX load (lmul * vlenb bytes) on this machine's
     * VLEN, so a vle8.v at vl == VLMAX can never over-read the pool
     * entry and the vsetivli-16 narrowing window is unnecessary in
     * the counted loop. vrgather indices are nibble values (0..15),
     * so only the first 16 bytes are ever read -- the replicas are
     * semantically invisible. (At least 16 bytes even if vlenb < 16;
     * VLEN < 128 is a pre-existing unsupported corner, see the file
     * header.) rep is a multiple of 16: lmul and vlenb are powers of
     * two, so lmul*vlenb >= 16 is a multiple of 16. */
    size_t rep = cg->lmul * cg->vlenb;
    if (rep < 16)
        rep = 16;
    size_t off = (cg->pool_len + 15) & ~(size_t)15;
    if (off + rep > cg->pool_cap) {
        size_t new_cap = (off + rep + 4095) & ~(size_t)4095;
        uint8_t *np = realloc(cg->pool, new_cap);
        if (!np) { cg->oom = 1; return 0; }
        cg->pool = np;
        cg->pool_cap = new_cap;
    }
    for (size_t k = 0; k < rep; k += 16)
        memcpy(cg->pool + off + k, kind == CK_NIBLO ? lo : hi, 16);
    cg->pool_len = off + rep;
    if (cg->ncent == cg->ccent) {
        int new_ccent = cg->ccent ? cg->ccent * 2 : 32;
        void *ncent = realloc(cg->cent, (size_t)new_ccent * sizeof(*cg->cent));
        if (!ncent) { cg->oom = 1; return 0; }
        cg->cent = ncent;
        cg->ccent = new_ccent;
    }
    cg->cent[cg->ncent].kind = kind;
    cg->cent[cg->ncent].c = c;
    cg->cent[cg->ncent].off = off;
    cg->ncent++;
    return off;
}

/* ---------------- cost-counting emit wrappers ---------------- */

/* Logical-to-physical vector register map. Logical register r (as
 * counted by plan_need_regs) maps to the lmul-aligned physical group
 * head r*lmul; the maskedtail twin shifts the whole map up by one
 * LMUL-group to reserve v0 as the dedicated mask register. Every
 * vector register reference in the backend must go through this
 * helper -- raw r*lmul arithmetic would silently alias v0 under
 * maskedtail, and a masked load/store with vd == v0 under ma
 * (mask-agnostic) lets hardware overwrite masked-off lanes of v0 with
 * 1s, destroying the mask mid-chunk. When maskedtail == 0 this is the
 * identity map (r*lmul), so non-maskedtail codegen is byte-identical. */
static int vreg(cg_t *cg, int r)
{
    return r * cg->lmul + (cg->maskedtail ? cg->lmul : 0);
}

static void c_vle(cg_t *cg, int vd, int rs1)
{
    if (cg->masked_mem)
        em_vle8_v_m(&cg->em, vd, rs1); /* v0.t: inactive lanes don't fault */
    else
        em_vle8(&cg->em, vd, rs1);
    cg->uops += 1;
    /* Nominal data movement scales with LMUL: one vector load moves a
     * full vl=VLMAX strip (lmul x NOMVL nominal bytes). Table loads
     * (c_ldtab/c_ldtab1) move a full VLMAX strip of replicated table
     * bytes but count unscaled NOMVL (the 16 useful bytes). */
    cg->mem += NOMVL * cg->lmul;
}

static void c_vse(cg_t *cg, int vs, int rs1)
{
    /* NT twin: Zihintntl NTL.ALL hint for the store that follows (the
     * hint applies to the immediately subsequent instruction). A HINT
     * is architecturally a NOP -- no gating needed -- and does not
     * change memory ordering, so unlike x86/AArch64 NT no caller fence
     * is required. Not counted as a uop: on implementations that
     * ignore the hint it executes as zero uops. */
    if (cg->cand.nt) {
        em_ntl_all(&cg->em);
        /* BUG-COMBO-STATIC-F16: effect marker -- nt hint emitted. */
        cg->twin_applied |= TWIN_APPLIED_NT;
    }
    if (cg->masked_mem)
        em_vse8_v_m(&cg->em, vs, rs1); /* v0.t: inactive lanes not stored */
    else
        em_vse8(&cg->em, vs, rs1);
    cg->uops += 1;
    /* As c_vle: nominal data movement scales with LMUL. */
    cg->mem += NOMVL * cg->lmul;
}

/* Software prefetch for an input stream (Zicbop prefetch.r): prefetch
 * pf_off bytes ahead of ptr_reg. No-op unless the candidate carries a
 * validated prefetch distance (cg->emit_pf). One prefetch per stream
 * per group iteration; the benchmark decides whether it pays off. */
static void c_prefetch_in(cg_t *cg, int ptr_reg)
{
    if (cg->emit_pf) {
        em_prefetch_r(&cg->em, ptr_reg, cg->pf_off);
        cg->uops += 1;
        /* BUG-COMBO-STATIC-F16: effect marker -- prefetch emitted. */
        cg->twin_applied |= TWIN_APPLIED_PREFETCH;
    }
}

static void c_vxor(cg_t *cg, int vd, int vs1, int vs2)
{
    em_vxor_vv(&cg->em, vd, vs1, vs2);
    cg->uops += 1;
}

static void c_vzero(cg_t *cg, int vd)
{
    em_vxor_vv(&cg->em, vd, vd, vd);
    cg->uops += 1;
}

/* Vector register copy; a self-move emits nothing. Used to fold
 * "xor with a compile-time-known zero" (first-touch accumulators) into
 * a single move. */
static void c_vmv(cg_t *cg, int vd, int vs)
{
    if (vd == vs)
        return;
    em_vmv_v_v(&cg->em, vd, vs);
    cg->uops += 1;
}

static void c_vand(cg_t *cg, int vd, int vs1, int vs2)
{
    em_vand_vv(&cg->em, vd, vs1, vs2);
    cg->uops += 1;
}

/* vd = vs2 & 0x0f via the immediate form (SEW=8): replaces the
 * vand.vv with the broadcast 0x0F mask register wherever the mask
 * has no other use, freeing that register. */
static void c_vand_vi(cg_t *cg, int vd, int vs2, unsigned uimm)
{
    em_vand_vi(&cg->em, vd, vs2, uimm);
    cg->uops += 1;
}

static void c_vsrl4(cg_t *cg, int vd, int vs)
{
    em_vsrl_vi(&cg->em, vd, vs, 4);
    cg->uops += 1;
}

static void c_vrg(cg_t *cg, int vd, int vtab, int vidx)
{
    em_vrgather_vv(&cg->em, vd, vtab, vidx);
    cg->uops += 1;
}

/* ---- Bit-program (xorbit) twin helpers ----
 * Mirror the x86 c_srlw/c_and/c_sllw trio for the bit-program body.
 * RVV has 3-operand shifts via immediate forms; no mask register needed
 * for the &1 (vand.vi takes an immediate). */

/* d = s >> k, k in 1..7. Caller skips k=0 (no shift needed). */
static void c_xbt_srl(cg_t *cg, int vd, int vs, unsigned k)
{
    assert(k >= 1 && k <= 7);
    em_vsrl_vi(&cg->em, vd, vs, k);
    cg->uops += 1;
}

/* d = s & 0x01 via vand.vi immediate (no mask reg). */
static void c_xbt_and1(cg_t *cg, int vd, int vs)
{
    em_vand_vi(&cg->em, vd, vs, 1);
    cg->uops += 1;
}

/* d = s << b, b in 1..7. Caller uses c_vmv for b=0. */
static void c_xbt_sll(cg_t *cg, int vd, int vs, unsigned b)
{
    assert(b >= 1 && b <= 7);
    em_vsll_vi(&cg->em, vd, vs, b);
    cg->uops += 1;
}


/* Forward declarations (defined below, used by xorbit code) */
static void c_ldptr(cg_t *cg, int dst, int base, int idx);

static void gen_body_xorbit(cg_t *cg, int nu)
{
    /* BUG-COMBO-STATIC-F16: effect marker. */
    cg->twin_applied |= TWIN_APPLIED_XORBIT;
    em_t *e = &cg->em;
    const plan_t *pl = cg->pl;
    const xorbit_t *s = pl->bit_sched;
    assert(s && s->ntemps > 0);
    assert(s->kind == pl->kind);
    int n_out = pl->n_out, U = cg->U, lmul = cg->lmul;
    int ni_sched = s->n_in_sched;
    int is_delta = (pl->kind == P_DELTA);
    int inreg = cg->xbt_in * lmul, r1 = cg->xbt_r1 * lmul, r2 = cg->xbt_r2 * lmul;

    if (!is_delta) {
        /* zero dead output bytes (no atoms in any of their 8 bits) */
        for (int j = 0; j < n_out; j++) {
            int live = 0;
            for (int b = 0; b < 8; b++)
                if (s->bit_natoms[8 * j + b] > 0) {
                    live = 1;
                    break;
                }
            if (!live)
                for (int u = 0; u < nu; u++)
                    c_vzero(cg, (j * U + u) * lmul);
        }
    } else {
        /* seed accumulators from oldpar */
        int nc = pl->nchanged;
        for (int j = 0; j < n_out; j++) {
            c_ldptr(cg, T2, A0, 2 * nc + j);
            em_add(e, T2, T2, T1); /* T1 = x11 offset */
            for (int u = 0; u < nu; u++) {
                c_vle(cg, (j * U + u) * lmul, T2);
                if (u < nu - 1)
                    em_add(e, T2, T2, T0); /* T0 = VLMAX stride */
            }
        }
    }

    int pf_done[RJ_MAX_DISKS];
    for (int i = 0; i < ni_sched; i++)
        pf_done[i] = 0;
    int touched[RJ_MAX_DISKS];

    for (int u = 0; u < nu; u++) {
        for (int j = 0; j < n_out; j++)
            touched[j] = 0;
        int cur_in = -1;
        /* phase 1: form each temp, fan out immediately */
        for (int m = 0; m < s->ntemps; m++) {
            int a = s->temps[m].a, b = s->temps[m].b;
            int ia = a >> 3, ka = a & 7;
            int ib = b >> 3, kb = b & 7;
            if (cur_in != ia) {
                c_ldptr(cg, T2, A0, ia);
                em_add(e, T2, T2, T1);
                if (!pf_done[ia]) {
                    c_prefetch_in(cg, T2);
                    pf_done[ia] = 1;
                }
                /* advance T2 by u*VLMAX */
                for (int uu = 0; uu < u; uu++)
                    em_add(e, T2, T2, T0);
                c_vle(cg, inreg, T2);
                cur_in = ia;
            }
            if (ka == 0) {
                c_vmv(cg, r1, inreg);
                c_xbt_and1(cg, r1, r1);
            } else {
                c_xbt_srl(cg, r1, inreg, (unsigned)ka);
                c_xbt_and1(cg, r1, r1);
            }
            if (cur_in != ib) {
                c_ldptr(cg, T2, A0, ib);
                em_add(e, T2, T2, T1);
                if (!pf_done[ib]) {
                    c_prefetch_in(cg, T2);
                    pf_done[ib] = 1;
                }
                for (int uu = 0; uu < u; uu++)
                    em_add(e, T2, T2, T0);
                c_vle(cg, inreg, T2);
                cur_in = ib;
            }
            if (kb == 0) {
                c_vmv(cg, r2, inreg);
                c_xbt_and1(cg, r2, r2);
            } else {
                c_xbt_srl(cg, r2, inreg, (unsigned)kb);
                c_xbt_and1(cg, r2, r2);
            }
            c_vxor(cg, r1, r1, r2);
            /* fanout with lmul-scaled acc indices */
            {
                int Uloc = U, w = 0;
                int *users = s->temps[m].users, nusers = s->temps[m].nusers;
                while (w < nusers) {
                    int b = users[w] & 7;
                    int w1 = w;
                    while (w1 < nusers && (users[w1] & 7) == b)
                        w1++;
                    int nj = w1 - w;
                    int j0 = users[w] >> 3;
                    int acc0 = (j0 * Uloc + u) * lmul;
                    if (nj == 1 && !is_delta && !touched[j0]) {
                        if (b == 0)
                            c_vmv(cg, acc0, r1);
                        else
                            c_xbt_sll(cg, acc0, r1, (unsigned)b);
                        touched[j0] = 1;
                    } else if (b == 0) {
                        for (int x = w; x < w1; x++) {
                            int j = users[x] >> 3;
                            int acc = (j * Uloc + u) * lmul;
                            if (!is_delta && !touched[j]) {
                                c_vmv(cg, acc, r1);
                                touched[j] = 1;
                            } else {
                                c_vxor(cg, acc, acc, r1);
                            }
                        }
                    } else {
                        c_xbt_sll(cg, r2, r1, (unsigned)b);
                        for (int x = w; x < w1; x++) {
                            int j = users[x] >> 3;
                            int acc = (j * Uloc + u) * lmul;
                            if (!is_delta && !touched[j]) {
                                c_vmv(cg, acc, r2);
                                touched[j] = 1;
                            } else {
                                c_vxor(cg, acc, acc, r2);
                            }
                        }
                    }
                    w = w1;
                }
            }
        }
        /* phase 2: surviving input bits, fanned out directly */
        for (int q = 0; q < 8 * ni_sched; q++) {
            if (!s->inbit_nusers[q])
                continue;
            int i = q >> 3, k = q & 7;
            if (cur_in != i) {
                c_ldptr(cg, T2, A0, i);
                em_add(e, T2, T2, T1);
                if (!pf_done[i]) {
                    c_prefetch_in(cg, T2);
                    pf_done[i] = 1;
                }
                for (int uu = 0; uu < u; uu++)
                    em_add(e, T2, T2, T0);
                c_vle(cg, inreg, T2);
                cur_in = i;
            }
            if (k == 0) {
                c_vmv(cg, r1, inreg);
                c_xbt_and1(cg, r1, r1);
            } else {
                c_xbt_srl(cg, r1, inreg, (unsigned)k);
                c_xbt_and1(cg, r1, r1);
            }
            /* fanout (same as above, using r1 as vsrc) */
            {
                int Uloc = U, w = 0;
                int *users = s->inbit_users[q], nusers = s->inbit_nusers[q];
                while (w < nusers) {
                    int b = users[w] & 7;
                    int w1 = w;
                    while (w1 < nusers && (users[w1] & 7) == b)
                        w1++;
                    int nj = w1 - w;
                    int j0 = users[w] >> 3;
                    int acc0 = (j0 * Uloc + u) * lmul;
                    if (nj == 1 && !is_delta && !touched[j0]) {
                        if (b == 0)
                            c_vmv(cg, acc0, r1);
                        else
                            c_xbt_sll(cg, acc0, r1, (unsigned)b);
                        touched[j0] = 1;
                    } else if (b == 0) {
                        for (int x = w; x < w1; x++) {
                            int j = users[x] >> 3;
                            int acc = (j * Uloc + u) * lmul;
                            if (!is_delta && !touched[j]) {
                                c_vmv(cg, acc, r1);
                                touched[j] = 1;
                            } else {
                                c_vxor(cg, acc, acc, r1);
                            }
                        }
                    } else {
                        c_xbt_sll(cg, r2, r1, (unsigned)b);
                        for (int x = w; x < w1; x++) {
                            int j = users[x] >> 3;
                            int acc = (j * Uloc + u) * lmul;
                            if (!is_delta && !touched[j]) {
                                c_vmv(cg, acc, r2);
                                touched[j] = 1;
                            } else {
                                c_vxor(cg, acc, acc, r2);
                            }
                        }
                    }
                    w = w1;
                }
            }
        }
    }
}

/* Zvbc carryless multiplies (MUL_ZVBC twin only; the backend gate in
 * jit_build_version guarantees rvv_zvbc32e for both vclmul.vv and
 * vclmulh.vv -- both are Zvbc, and their SEW=8 forms need Zvbc32e --
 * before any of these can be emitted). */
static void c_vclmul(cg_t *cg, int vd, int vs1, int vs2)
{
    em_vclmul_vv(&cg->em, vd, vs1, vs2);
    cg->uops += 1;
}

static void c_vclmulh(cg_t *cg, int vd, int vs1, int vs2)
{
    em_vclmulh_vv(&cg->em, vd, vs1, vs2);
    cg->uops += 1;
}

/* The carryless reduction half of c_gfmul_acc_zvbc: res =
 * clmul_reduce(vs1, vs2) (5 carryless + 2 xor, lo=q=n fold with
 * K=0x1B; see DESIGN.md), left in r1. 7 vector uops. Split out so the
 * HORNER step can fuse its trailing add into one vxor.vv (see
 * gen_body); vs1 may alias the eventual destination (the reduction
 * only reads vs1). r1/r2/r3/kvec are the cg temps. */
static void c_gfmul_red_zvbc(cg_t *cg, int vs1, int vs2)
{
    c_vclmul(cg, cg->r1, vs1, vs2);        /* lo */
    c_vclmulh(cg, cg->r2, vs1, vs2);       /* hi */
    c_vclmul(cg, cg->r3, cg->r2, cg->kvec); /* q */
    c_vclmulh(cg, cg->r2, cg->r2, cg->kvec);/* qh */
    c_vclmul(cg, cg->r2, cg->r2, cg->kvec); /* n */
    c_vxor(cg, cg->r1, cg->r1, cg->r3);     /* lo ^ q */
    c_vxor(cg, cg->r1, cg->r1, cg->r2);     /* res */
}

/* One GF(2^8) multiply-add via the Zvbc carryless path:
 *   res = clmul_reduce(vs1, vs2)   (5 carryless + 2 xor)
 *   vd  = first ? res : vd ^ res   (move / xor, like the nibble path)
 * where clmul_reduce is lo=q=n fold with K=0x1B (see DESIGN.md).
 * vd may alias vs1 (HORNER: acc = acc*c + t); r1/r2/r3/kvec/cvec are
 * the cg temps. 8 vector uops. */
static void c_gfmul_acc_zvbc(cg_t *cg, int vd, int vs1, int vs2, int first)
{
    c_gfmul_red_zvbc(cg, vs1, vs2);
    if (first)
        c_vmv(cg, vd, cg->r1);
    else
        c_vxor(cg, vd, vd, cg->r1);
}

/* ---------------- small GPR helpers ---------------- */

/* rd = v (non-negative int constant) */
static void c_li(cg_t *cg, int rd, int v)
{
    em_t *e = &cg->em;
    assert(v >= 0);
    if (v < 2048) {
        em_addi(e, rd, X0, v);
    } else {
        long hi = ((long)v + 0x800) >> 12;
        long lo = (long)v - (hi << 12);
        em_lui(e, rd, (int)(hi & 0xfffff));
        em_addi(e, rd, rd, (int)lo);
    }
}

/* Broadcast a byte constant into a vector register (2 uops: li +
 * vmv.v.x). Used by per-column call sites in the loop body; the
 * prologue uses raw emitter calls instead (prologue setup is not
 * per-iteration work, same as the nibble mask/table setup which is
 * also uncounted). */
static void c_broadcast(cg_t *cg, int vd, unsigned val)
{
    c_li(cg, T3, (int)val);
    em_vmv_v_x(&cg->em, vd, T3);
    cg->uops += 2;
}

/* dst = base[idx]: in[]/out[] pointer fetch. idx*8 can exceed the 12-bit
 * load displacement for very wide P_DELTA ops, hence the indexed path. */
static void c_ldptr(cg_t *cg, int dst, int base, int idx)
{
    em_t *e = &cg->em;
    long off = (long)idx * 8L;
    if (off < 2048) {
        em_ld(e, dst, base, (int)off);
    } else {
        assert(idx < (1 << 30));
        c_li(cg, T5, idx);
        em_slli(e, T5, T5, 3);
        em_add(e, T5, base, T5);
        em_ld(e, dst, T5, 0);
    }
}

/* T3 = T6 (pool base) + off: address of a constant-pool entry */
static void c_pooladdr(cg_t *cg, size_t off)
{
    em_t *e = &cg->em;
    if (off < 2048) {
        em_addi(e, T3, T6, (int)off);
    } else {
        long o = (long)off;
        long hi = (o + 0x800) >> 12;
        long lo = o - (hi << 12);
        em_lui(e, T3, (int)(hi & 0xfffff));
        em_addi(e, T3, T3, (int)lo);
        em_add(e, T3, T3, T6);
    }
}

/* Load the 16-byte nibble-table pair for coefficient c into
 * (tbl_lo, tbl_hi). The pool entry holds the table replicated to cover
 * a full VLMAX load (see pool_get), so a vle8.v at the body's natural
 * vl can never over-read it. The loads run inside a vsetivli-16 window
 * only when win != 0 (the tail, where vl can be < 16 and a short load
 * would leave table elements past vl stale); in the counted loop
 * (win == 0) vl == VLMAX for the whole body, so the window is
 * provably redundant and elided at any LMUL. */
static void c_ldtab(cg_t *cg, uint8_t c, int win)
{
    em_t *e = &cg->em;
    /* -1-register tripwire (see c_ldtab1): tbl_lo/tbl_hi are only
     * assigned on the nibble path; a -1 reaching the emitter would
     * silently encode v0 and clobber accumulator v0. */
    assert(cg->tbl_lo >= 0 && cg->tbl_hi >= 0);
    size_t olo = pool_get(cg, CK_NIBLO, c);
    size_t ohi = pool_get(cg, CK_NIBHI, c);
    /* T5 saves the vl value (dead at all call sites: T2/T4 hold the
     * live pointers, T5 is only c_ldptr scratch); T3 is c_pooladdr
     * scratch. Only the win != 0 path needs them: the counted loop
     * loads the replicated entry at vl == VLMAX with no narrowing. */
    if (win)
        em_addi(e, T5, T0, 0); /* save vl */
    if (win)
        em_vsetivli(e, X0, 16, VSEW8, cg->vlmul, 1, 1);
    c_pooladdr(cg, olo);
    em_vle8(e, cg->tbl_lo, T3);
    c_pooladdr(cg, ohi);
    em_vle8(e, cg->tbl_hi, T3);
    if (win)
        /* restore vl: min(VLMAX, saved) == saved */
        em_vsetvli(e, X0, T5, VSEW8, cg->vlmul, 1, 1);
    cg->uops += win ? 7 : 4;
    cg->mem += 2 * NOMVL;
}

/* Load one 16-byte nibble table (kind CK_NIBLO/CK_NIBHI) for coefficient
 * c into vector register vd. The HORNER remat path uses this to reload
 * tables from the pool into the tbl scratch on every use; c_ldtab's
 * fixed (tbl_lo, tbl_hi) pair is only assigned in the MULACC/DELTA
 * register map, so it cannot serve the HORNER path. Same window
 * discipline as c_ldtab: the pool entry is replicated to cover a full
 * VLMAX load, so the vsetivli-16 window is elided in the counted
 * full-group loop (win=0) at any LMUL and kept only in the
 * remainder/tail bodies (win=1), where vl can be < 16. */
static void c_ldtab1(cg_t *cg, int vd, int kind, uint8_t c, int win)
{
    em_t *e = &cg->em;
    assert(vd >= 0); /* -1-register tripwire: hreg[] stays -1 in remat
                      * mode; a -1 reaching the emitter would silently
                      * encode v0. */
    size_t off = pool_get(cg, kind, c);
    /* T5 saves the vl value (dead at all call sites: T2 holds the live
     * input pointer in the HORNER body, T5 is only c_ldptr scratch);
     * T3 is c_pooladdr scratch. Only the win != 0 path needs them. */
    if (win)
        em_addi(e, T5, T0, 0); /* save vl */
    if (win)
        em_vsetivli(e, X0, 16, VSEW8, cg->vlmul, 1, 1);
    c_pooladdr(cg, off);
    em_vle8(e, vd, T3);
    if (win)
        /* restore vl: min(VLMAX, saved) == saved */
        em_vsetvli(e, X0, T5, VSEW8, cg->vlmul, 1, 1);
    cg->uops += win ? 5 : 2;
    cg->mem += NOMVL;
}

/* ---------------- software-pipelined loop (track I8) ---------------- */

static void gen_body(cg_t *cg, int nu, int win);
static void mulacc_rows_c(cg_t *cg, int col, int n_in, const int *first_col,
                          uint8_t c, int nu);

/* Columns whose input data the pipe body stages: P_MULACC stages only
 * columns with a nonzero coefficient somewhere (dead columns are
 * skipped, mirroring gen_body); P_HORNER stages every input column.
 * Returns the count (0 for degenerate plans: no loads to pipeline). */
static int pipe_used_columns(cg_t *cg, int *cols)
{
    const plan_t *pl = cg->pl;
    int ncols = 0;
    if (pl->kind == P_MULACC) {
        for (int i = 0; i < pl->n_in; i++) {
            int any = 0;
            for (int j = 0; j < pl->n_out; j++)
                if (pl->mat[(size_t)j * (size_t)pl->n_in + (size_t)i]) {
                    any = 1;
                    break;
                }
            if (any)
                cols[ncols++] = i;
        }
    } else { /* P_HORNER */
        for (int i = 0; i < pl->n_in; i++)
            cols[ncols++] = i;
    }
    return ncols;
}

/* Load U lanes of input column col at byte offset off_reg into the
 * dataset at physical head dst_head. off_reg is T1 (current offset) or
 * T5 (next offset = T1+A3). NOTE: c_ldptr uses T5 as scratch only on
 * the idx*8 >= 2048 path; cols < RJ_MAX_DISKS <= 256 keeps idx*8 <
 * 2048, so T5 survives as off_reg. */
static void pipe_load_column(cg_t *cg, int col, int dst_head, int off_reg)
{
    em_t *e = &cg->em;
    c_ldptr(cg, T2, A0, col);
    em_add(e, T2, T2, off_reg);
    c_prefetch_in(cg, T2);
    for (int u = 0; u < cg->U; u++) {
        c_vle(cg, dst_head + u * cg->lmul, T2);
        if (u < cg->U - 1)
            em_add(e, T2, T2, T0);
    }
}

/* One HORNER column's compute from the staged dataset (pipe twin).
 * The caller rebinds cg->vd0 to the dataset's physical head. Mirrors
 * gen_body's P_HORNER nibble path (non-remat, non-zvbc; both are
 * rejected for the pipe twin). Main loop only (nu == U, win == 0). */
static void pipe_horner_compute(cg_t *cg, int col, int nu)
{
    const plan_t *pl = cg->pl;
    int n_out = pl->n_out, U = cg->U;
    assert(nu == U);
    if (col == 0) {
        /* acc = d0: plain moves from the staged dataset */
        for (int j = 0; j < n_out; j++)
            for (int u = 0; u < nu; u++)
                c_vmv(cg, vreg(cg, j * U + u),
                      cg->vd0 + u * cg->lmul);
        return;
    }
    for (int j = 0; j < n_out; j++) {
        uint8_t c = pl->horner_c[j];
        for (int u = 0; u < nu; u++) {
            int acc = vreg(cg, j * U + u), vd = cg->vd0 + u * cg->lmul;
            if (c == 1) {
                c_vxor(cg, acc, acc, vd);
            } else if (c == 0) {
                c_vmv(cg, acc, vd);
            } else {
                c_vand(cg, cg->tlo, acc, cg->mask);
                c_vsrl4(cg, cg->thi, acc);
                assert(cg->hreg[2 * j] >= 0 &&
                       cg->hreg[2 * j + 1] >= 0);
                c_vrg(cg, cg->r1, cg->hreg[2 * j], cg->tlo);
                c_vrg(cg, cg->r2, cg->hreg[2 * j + 1], cg->thi);
                c_vxor(cg, cg->r1, cg->r1, cg->r2);
                c_vxor(cg, acc, cg->r1, vd);
            }
        }
    }
}

/* Software-pipelined (double-buffered) loop body (track I8, pipe twin).
 *
 * Per-column double buffering: D is the vd0 dataset (U group heads,
 * reused), Dn is the pipe_dn dataset (U group heads) holding column 0
 * of the next iteration. Per (iter, cidx):
 *   - compute(cols[cidx], iter) from (cidx == 0 ? Dn : D), via a
 *     temporary cg->vd0 rebind (mulacc_rows_c / pipe_horner_compute
 *     read the dataset through cg->vd0). Dn holds (cols[0], iter):
 *     the prologue for iter == 0, the boundary load of the previous
 *     iteration otherwise.
 *   - then, if cidx+1 < ncols: load(cols[cidx+1], iter) -> D at T1;
 *     else (last column): if A4 > 1 (a full next iteration exists):
 *     load(cols[0], iter+1) -> Dn at T5 = T1+A3. The guard keeps the
 *     load in-bounds; the last iteration would otherwise over-read up
 *     to group bytes into/past the tail.
 * Every issued load is consumed by the next column step's compute, so
 * the pipeline drains naturally; the "epilogue" is the final
 * iteration's stores. ncols == 0 falls back to gen_body. */
static void gen_body_pipe(cg_t *cg)
{
    /* BUG-COMBO-STATIC-F16: effect marker. */
    cg->twin_applied |= TWIN_APPLIED_PIPE;
    em_t *e = &cg->em;
    const plan_t *pl = cg->pl;
    int n_in = pl->n_in, n_out = pl->n_out, U = cg->U;
    assert(cg->cand.pipe);
    assert(pl->kind == P_MULACC || pl->kind == P_HORNER);
    assert(!cg->cand.remat && !cg->zvbc);

    int cols[RJ_MAX_DISKS];
    int ncols = pipe_used_columns(cg, cols);
    if (ncols == 0) {
        gen_body(cg, U, 0); /* degenerate: no loads to pipeline */
        return;
    }

    int first_col[RJ_MAX_DISKS];
    if (pl->kind == P_MULACC) {
        for (int j = 0; j < n_out; j++) {
            first_col[j] = -1;
            for (int i = 0; i < n_in; i++)
                if (pl->mat[(size_t)j * (size_t)n_in + (size_t)i]) {
                    first_col[j] = i;
                    break;
                }
            if (first_col[j] < 0)
                for (int u = 0; u < U; u++)
                    c_vzero(cg, vreg(cg, j * U + u));
        }
    }

    for (int cidx = 0; cidx < ncols; cidx++) {
        int col = cols[cidx];
        /* compute column col from the staged dataset */
        int save_vd0 = cg->vd0;
        cg->vd0 = (cidx == 0) ? cg->pipe_dn : save_vd0;
        if (pl->kind == P_MULACC) {
            mulacc_rows_c(cg, col, n_in, first_col, 1, U);
            uint8_t seen[256] = { 0 };
            for (int j = 0; j < n_out; j++) {
                uint8_t c =
                    pl->mat[(size_t)j * (size_t)n_in + (size_t)col];
                if (c < 2 || seen[c])
                    continue;
                seen[c] = 1;
                if (!cg->hoist_tables)
                    c_ldtab(cg, c, 0);
                mulacc_rows_c(cg, col, n_in, first_col, c, U);
            }
        } else {
            pipe_horner_compute(cg, col, U);
        }
        cg->vd0 = save_vd0;
        /* issue the next loads into the other dataset */
        if (cidx + 1 < ncols) {
            pipe_load_column(cg, cols[cidx + 1], cg->vd0, T1);
        } else {
            /* boundary: (cols[0], iter+1) -> Dn at T1+A3, iff a full
             * next iteration exists (A4 = N - iter > 1). */
            int skip = em_label(e);
            em_addi(e, T5, A4, -1);
            em_beq(e, T5, X0, skip);
            em_add(e, T5, T1, A3); /* T5 = next offset */
            pipe_load_column(cg, cols[0], cg->pipe_dn, T5);
            em_bind(e, skip);
        }
    }

    /* stores (mirrors gen_body's store loop) */
    for (int j = 0; j < n_out; j++) {
        c_ldptr(cg, T2, A1, j);
        em_add(e, T2, T2, T1);
        for (int u = 0; u < U; u++) {
            c_vse(cg, vreg(cg, j * U + u), T2);
            if (u < U - 1)
                em_add(e, T2, T2, T0);
        }
    }
}

/* ---------------- codegen ---------------- */

static void gen_const_setup(cg_t *cg)
{
    em_t *e = &cg->em;
    const plan_t *pl = cg->pl;
    if (cg->cand.xorbit) {
        /* Bit-program twin: no nibble tables, no 0x0F mask, no kvec,
         * no per-row HORNER coefficient slots (hreg is not allocated
         * on this path). The body is pure bit logic (vand.vi takes an
         * immediate) and runs under the vl=VLMAX that gen_function
         * establishes. Prologue setup is not per-iteration work:
         * uncounted, by the same convention as the nibble path. */
        return;
    }
    /* vl = VLMAX (LMUL-scaled), held in T5 for the table loads below:
     * the replicated pool entries (see pool_get) are exactly VLMAX
     * bytes, so the loads need no narrowing. */
    em_vsetvli(e, T5, X0, VSEW8, cg->vlmul, 1, 1);
    /* The 0x0F mask broadcast is dead in HORNER remat mode: that path
     * uses vand.vi with the immediate and repurposes the mask register
     * as the NIBHI table scratch (see gen_body). */
    if (cg->need_mask && !(pl->kind == P_HORNER && cg->cand.remat)) {
        em_addi(e, T3, X0, 0x0f);
        em_vmv_v_x(e, cg->mask, T3);
    }
    /* Zvbc path: broadcast the 0x1B reduction constant once (vl=VLMAX
     * here, so the full vector is written). Prologue setup is not
     * per-iteration work: uncounted, like the nibble mask/tables. kvec
     * is allocated exactly when the multiply path is live (HORNER, or
     * MULACC/DELTA with a nontrivial coefficient); the all-trivial
     * P_DELTA Zvbc twin has no multiplies and no kvec. */
    if (cg->zvbc &&
        (pl->kind == P_HORNER || plan_any_nontrivial(pl))) {
        c_li(cg, T3, 0x1b);
        em_vmv_v_x(e, cg->kvec, T3);
    }
    if (cg->hoist_tables) {
        /* Single-coefficient table hoist (MULACC/DELTA nibble path):
         * the table pair is loop-invariant, so it is loaded once here
         * into tbl_lo/tbl_hi instead of per column in the body. vl ==
         * VLMAX (T5, set above) and the replicated pool entry is
         * exactly VLMAX bytes, so no narrowing window is needed.
         * Prologue setup is not per-iteration work: uncounted. */
        assert(cg->tbl_lo >= 0 && cg->tbl_hi >= 0);
        size_t olo = pool_get(cg, CK_NIBLO, cg->hoist_c);
        size_t ohi = pool_get(cg, CK_NIBHI, cg->hoist_c);
        c_pooladdr(cg, olo);
        em_vle8(e, cg->tbl_lo, T3);
        c_pooladdr(cg, ohi);
        em_vle8(e, cg->tbl_hi, T3);
    }
    if (pl->kind == P_HORNER) {
        if (cg->zvbc) {
            /* Per-row coefficient broadcasts into the hreg[2j] cvec
             * slots (nontrivial rows only; c in {0,1} needs no
             * multiply). MULACC/DELTA coefficients vary per column, so
             * their cvec broadcast happens per column in
             * mulacc_rows_c. */
            for (int j = 0; j < pl->n_out; j++) {
                int cv = cg->hreg[2 * j];
                if (cv < 0)
                    continue;
                c_li(cg, T3, pl->horner_c[j]);
                em_vmv_v_x(e, cv, T3);
            }
        } else if (!cg->cand.remat) {
            /* In remat mode hreg[] stays -1 (no persistent table regs);
             * tables are reloaded from the pool into tbl on each use. */
            for (int j = 0; j < pl->n_out; j++) {
                uint8_t c = pl->horner_c[j];
                if (c == 0 || c == 1)
                    continue;
                /* -1 tripwire: in remat mode this loop is skipped and
                 * hreg[] stays -1; here the slots must be assigned. */
                assert(cg->hreg[2 * j] >= 0 && cg->hreg[2 * j + 1] >= 0);
                size_t olo = pool_get(cg, CK_NIBLO, c);
                size_t ohi = pool_get(cg, CK_NIBHI, c);
                /* No vl narrowing: the pool entry holds the table
                 * replicated to cover a full VLMAX load (see pool_get),
                 * and vl == VLMAX here (T5, set above), so the vle8.v
                 * reads exactly the entry. */
                c_pooladdr(cg, olo);
                em_vle8(e, cg->hreg[2 * j], T3);
                c_pooladdr(cg, ohi);
                em_vle8(e, cg->hreg[2 * j + 1], T3);
            }
        }
    }
}

/* Multiply-accumulate one matrix column's rows that share coefficient c
 * (c != 0; c == 1 handled by the caller as plain XOR). vd holds the
 * per-lane multiplicands in vd0..vd0+U-1; mat is n_out x ncols
 * row-major. first_col[j] is the first column with a nonzero entry in
 * row j, or -1; pass NULL when accumulators are not provably zero
 * (P_DELTA seeds from oldpar). At the first contributing column the
 * accumulator is known zero, so the update is a move instead of an
 * xor-with-zero. */
static void mulacc_rows_c(cg_t *cg, int col, int ncols, const int *first_col,
                          uint8_t c, int nu)
{
    const plan_t *pl = cg->pl;
    int n_out = pl->n_out, U = cg->U;
    assert(nu >= 1 && nu <= U);
    if (cg->zvbc && c != 1) {
        /* One coefficient broadcast per (column, c) group, shared by
         * all rows below (2 uops: li + vmv.v.x). c==0 never reaches
         * here (the caller filters c < 2); c==1 needs no multiply. */
        c_broadcast(cg, cg->cvec, c);
    }
    for (int j = 0; j < n_out; j++) {
        if (pl->mat[(size_t)j * (size_t)ncols + (size_t)col] != c)
            continue;
        int first = first_col && col == first_col[j];
        for (int u = 0; u < nu; u++) {
            int acc = vreg(cg, j * U + u);
            int vd = cg->vd0 + u * cg->lmul;
            if (c == 1) {
                if (first)
                    c_vmv(cg, acc, vd); /* acc was 0 */
                else
                    c_vxor(cg, acc, acc, vd);
            } else if (cg->zvbc) {
                /* Carryless path: 8 vector uops (5 clmul + 3 xor), no
                 * tables, no mask. */
                c_gfmul_acc_zvbc(cg, acc, vd, cg->cvec, first);
            } else {
                c_vand(cg, cg->tlo, vd, cg->mask);
                c_vsrl4(cg, cg->thi, vd);
                c_vrg(cg, cg->r1, cg->tbl_lo, cg->tlo);
                c_vrg(cg, cg->r2, cg->tbl_hi, cg->thi);
                c_vxor(cg, cg->r1, cg->r1, cg->r2);
                if (first)
                    c_vmv(cg, acc, cg->r1); /* acc was 0 */
                else
                    c_vxor(cg, acc, acc, cg->r1);
            }
        }
    }
}

/* Emit one strip-mining iteration processing nu vector chunks of vl
 * bytes (nu <= U). The main loop passes nu == U; the length tail passes
 * nu == 1. Accumulator register indices keep the full-U stride
 * (j*U+u); the tail simply uses the u == 0 subset. win controls the
 * table-load vsetivli-16 window (see c_ldtab): pass 0 when vl is
 * provably VLMAX for the whole body (the counted full-group loop),
 * 1 otherwise (remainder/tail, where vl can be < 16). */
static void gen_body(cg_t *cg, int nu, int win)
{
    em_t *e = &cg->em;
    const plan_t *pl = cg->pl;
    int n_in = pl->n_in, n_out = pl->n_out, U = cg->U;
    assert(nu >= 1 && nu <= U);

    if (cg->cand.xorbit) {
        /* Bit-program twin: the body follows pl->bit_sched instead of
         * the naive emitter. gen_body_xorbit handles its own
         * accumulator seeding/zeroing; stores below are shared. */
        gen_body_xorbit(cg, nu);
    } else if (pl->kind == P_XOR) {
        if (cg->cand.xorsched) {
        /* BUG-COMBO-STATIC-F16: effect marker -- the xorsched path. */
        cg->twin_applied |= TWIN_APPLIED_XORSCHED;
        /* XOR-schedule twin: the body follows pl->xor_sched (cross-row
         * CSE). Phase 1 forms each temp t_k = in_a ^ in_b into
         * (vd0, vd1) and fans out; phase 2 streams surviving inputs via
         * vd0. Every accumulator is overwritten (move / direct load) on
         * first touch, never XORed with zero; each shared subexpression
         * is computed once. T3/T4 are the two input pointers (free in
         * P_XOR bodies). Like the naive body, this runs inside gen_body
         * and is emitted 3x (nu=U, U, 1). */
        {
            const xorsched_t *xs = pl->xor_sched;
            assert(xs && xs->ntemps > 0);
            int pf_done[RJ_MAX_DISKS];
            for (int i = 0; i < n_in; i++)
                pf_done[i] = 0;
            /* dead rows: zero (mirrors the naive body's prologue) */
            for (int uw = 0; uw < xs->nunique; uw++) {
                if (xs->row_first[uw] >= 0)
                    continue;
                for (int j = 0; j < n_out; j++) {
                    if (xs->urow_of[j] != uw)
                        continue;
                    for (int u = 0; u < nu; u++)
                        c_vzero(cg, vreg(cg, j * U + u));
                }
            }
            /* phase 1: temp formation + fan-out */
            for (int k = 0; k < xs->ntemps; k++) {
                int a = xs->temps[k].a, b = xs->temps[k].b;
                c_ldptr(cg, T3, A0, a);
                em_add(e, T3, T3, T1);
                if (!pf_done[a]) {
                    c_prefetch_in(cg, T3);
                    pf_done[a] = 1;
                }
                c_ldptr(cg, T4, A0, b);
                em_add(e, T4, T4, T1);
                if (!pf_done[b]) {
                    c_prefetch_in(cg, T4);
                    pf_done[b] = 1;
                }
                for (int u = 0; u < nu; u++) {
                    c_vle(cg, cg->vd0, T3);
                    c_vle(cg, cg->vd1, T4);
                    c_vxor(cg, cg->vd0, cg->vd0, cg->vd1);
                    for (int wi = 0; wi < xs->temps[k].nusers; wi++) {
                        int uw = xs->temps[k].users[wi];
                        int acc = vreg(cg, xs->first_row[uw] * U + u);
                        if (xs->row_first[uw] == k)
                            c_vmv(cg, acc, cg->vd0);
                        else
                            c_vxor(cg, acc, acc, cg->vd0);
                    }
                    if (u < nu - 1) {
                        em_add(e, T3, T3, T0);
                        em_add(e, T4, T4, T0);
                    }
                }
            }
            /* phase 2: surviving input atoms */
            for (int i = 0; i < n_in; i++) {
                int nusers = xs->in_nusers[i];
                if (!nusers)
                    continue;
                c_ldptr(cg, T2, A0, i);
                em_add(e, T2, T2, T1);
                if (!pf_done[i]) {
                    c_prefetch_in(cg, T2);
                    pf_done[i] = 1;
                }
                int uw0 = xs->in_users[i][0];
                int is_first =
                    (nusers == 1 && xs->row_first[uw0] == xs->ntemps + i);
                for (int u = 0; u < nu; u++) {
                    if (is_first) {
                        int acc = vreg(cg, xs->first_row[uw0] * U + u);
                        c_vle(cg, acc, T2);
                    } else {
                        c_vle(cg, cg->vd0, T2);
                        for (int wi = 0; wi < nusers; wi++) {
                            int uw = xs->in_users[i][wi];
                            int acc = vreg(cg, xs->first_row[uw] * U + u);
                            if (xs->row_first[uw] == xs->ntemps + i)
                                c_vmv(cg, acc, cg->vd0);
                            else
                                c_vxor(cg, acc, acc, cg->vd0);
                        }
                    }
                    if (u < nu - 1)
                        em_add(e, T2, T2, T0);
                }
            }
            /* duplicate rows: vector moves from the canonical row */
            for (int j = 0; j < n_out; j++) {
                int uw = xs->urow_of[j];
                if (j == xs->first_row[uw] || xs->row_first[uw] < 0)
                    continue;
                for (int u = 0; u < nu; u++)
                    c_vmv(cg, vreg(cg, j * U + u), vreg(cg, xs->first_row[uw] * U + u));
            }
        }
        } else {
        /* first_in[j]: first input with a 1 in row j (-1 if none). The
         * accumulator is provably zero there: load/move, don't xor. */
        int first_in[RJ_MAX_DISKS];
        for (int j = 0; j < n_out; j++) {
            first_in[j] = -1;
            for (int i = 0; i < n_in; i++)
                if (pl->mat[(size_t)j * (size_t)n_in + (size_t)i]) {
                    first_in[j] = i;
                    break;
                }
            if (first_in[j] < 0)
                for (int u = 0; u < nu; u++)
                    c_vzero(cg, vreg(cg, j * U + u));
        }
        for (int i = 0; i < n_in; i++) {
            int users[RJ_MAX_DISKS], nusers = 0;
            for (int j = 0; j < n_out; j++)
                if (pl->mat[(size_t)j * (size_t)n_in + (size_t)i])
                    users[nusers++] = j;
            if (!nusers)
                continue;
            c_ldptr(cg, T2, A0, i);
            em_add(e, T2, T2, T1); /* + byte offset */
            c_prefetch_in(cg, T2);
            for (int u = 0; u < nu; u++) {
                if (nusers == 1 && i == first_in[users[0]]) {
                    /* first touch, single user: load straight into acc */
                    c_vle(cg, vreg(cg, users[0] * U + u), T2);
                } else {
                    c_vle(cg, cg->vd0, T2);
                    for (int k = 0; k < nusers; k++) {
                        int acc = vreg(cg, users[k] * U + u);
                        if (i == first_in[users[k]])
                            c_vmv(cg, acc, cg->vd0);
                        else
                            c_vxor(cg, acc, acc, cg->vd0);
                    }
                }
                if (u < nu - 1)
                    em_add(e, T2, T2, T0); /* += VL */
            }
        }
        } /* end xorsched/naive P_XOR */
    } else if (pl->kind == P_MULACC) {
        int first_col[RJ_MAX_DISKS];
        for (int j = 0; j < n_out; j++) {
            first_col[j] = -1;
            for (int i = 0; i < n_in; i++)
                if (pl->mat[(size_t)j * (size_t)n_in + (size_t)i]) {
                    first_col[j] = i;
                    break;
                }
            if (first_col[j] < 0)
                for (int u = 0; u < nu; u++)
                    c_vzero(cg, vreg(cg, j * U + u));
        }
        for (int i = 0; i < n_in; i++) {
            int any = 0;
            for (int j = 0; j < n_out; j++)
                if (pl->mat[(size_t)j * (size_t)n_in + (size_t)i])
                    any = 1;
            if (!any)
                continue;
            c_ldptr(cg, T2, A0, i);
            em_add(e, T2, T2, T1);
            c_prefetch_in(cg, T2);
            for (int u = 0; u < nu; u++) {
                c_vle(cg, cg->vd0 + u * cg->lmul, T2);
                if (u < nu - 1)
                    em_add(e, T2, T2, T0);
            }
            /* c == 1 rows: plain XOR, no tables */
            mulacc_rows_c(cg, i, n_in, first_col, 1, nu);
            /* one table load per distinct c != 0,1 in this column
             * (nibble path only: the Zvbc path has no tables; an
             * unguarded c_ldtab here would load into the unassigned
             * tbl_lo/tbl_hi (v0), clobbering accumulator v0). Skipped
             * when the single-coefficient hoist keeps the table pair
             * live in tbl_lo/tbl_hi from the prologue. */
            uint8_t seen[256] = { 0 };
            for (int j = 0; j < n_out; j++) {
                uint8_t c = pl->mat[(size_t)j * (size_t)n_in + (size_t)i];
                if (c < 2 || seen[c])
                    continue;
                seen[c] = 1;
                if (!cg->zvbc && !cg->hoist_tables)
                    c_ldtab(cg, c, win);
                mulacc_rows_c(cg, i, n_in, first_col, c, nu);
            }
        }
    } else if (pl->kind == P_DELTA) {
        /* new_p[j] = old_p[j] ^ Σ_i ((old[i] ^ new[i]) * M[j][changed[i]]).
         * in[] = [old_0..old_{nc-1}, new_0..new_{nc-1},
         *         oldpar_0..oldpar_{np-1}]. */
        int nc = pl->nchanged;
        for (int j = 0; j < n_out; j++) {
            c_ldptr(cg, T2, A0, 2 * nc + j);
            em_add(e, T2, T2, T1);
            c_prefetch_in(cg, T2);
            for (int u = 0; u < nu; u++) {
                c_vle(cg, vreg(cg, j * U + u), T2);
                if (u < nu - 1)
                    em_add(e, T2, T2, T0);
            }
        }
        for (int i = 0; i < nc; i++) {
            int any = 0;
            for (int j = 0; j < n_out; j++)
                if (pl->mat[(size_t)j * (size_t)nc + (size_t)i])
                    any = 1;
            if (!any)
                continue;
            c_ldptr(cg, T2, A0, i); /* old */
            em_add(e, T2, T2, T1);
            c_prefetch_in(cg, T2);
            c_ldptr(cg, T4, A0, nc + i); /* new */
            em_add(e, T4, T4, T1);
            c_prefetch_in(cg, T4);
            for (int u = 0; u < nu; u++) {
                /* tlo/thi borrowed as delta temps (reused below) */
                c_vle(cg, cg->tlo, T2);
                c_vle(cg, cg->thi, T4);
                c_vxor(cg, cg->vd0 + u * cg->lmul, cg->tlo, cg->thi);
                if (u < nu - 1) {
                    em_add(e, T2, T2, T0);
                    em_add(e, T4, T4, T0);
                }
            }
            mulacc_rows_c(cg, i, nc, NULL, 1, nu); /* c == 1 rows */
            uint8_t seen[256] = { 0 };
            for (int j = 0; j < n_out; j++) {
                uint8_t c = pl->mat[(size_t)j * (size_t)nc + (size_t)i];
                if (c < 2 || seen[c])
                    continue;
                seen[c] = 1;
                /* Nibble path only: the Zvbc path has no tables (see
                 * the P_MULACC call site for why the guard is
                 * load-bearing). Skipped under the single-coefficient
                 * hoist (tables live from the prologue). */
                if (!cg->zvbc && !cg->hoist_tables)
                    c_ldtab(cg, c, win);
                /* NULL: accs hold oldpar, not zero -- always xor */
                mulacc_rows_c(cg, i, nc, NULL, c, nu);
            }
        }
    } else { /* P_HORNER */
        if (n_in == 0) {
            /* Degenerate: no inputs; accumulators stay zero. */
            for (int j = 0; j < n_out; j++)
                for (int u = 0; u < nu; u++)
                    c_vzero(cg, vreg(cg, j * U + u));
        }
        for (int i = 0; i < n_in; i++) {
            c_ldptr(cg, T2, A0, i);
            em_add(e, T2, T2, T1);
            c_prefetch_in(cg, T2);
            for (int u = 0; u < nu; u++) {
                c_vle(cg, cg->vd0 + u * cg->lmul, T2);
                if (u < nu - 1)
                    em_add(e, T2, T2, T0);
            }
            if (i == 0) {
                /* Accumulators are provably zero: acc = 0*c + t = t
                 * for any c. A plain move per row replaces the whole
                 * multiply-add. */
                for (int j = 0; j < n_out; j++)
                    for (int u = 0; u < nu; u++)
                        c_vmv(cg, vreg(cg, j * U + u), cg->vd0 + u * cg->lmul);
                continue;
            }
            for (int j = 0; j < n_out; j++) {
                uint8_t c = pl->horner_c[j];
                if (cg->cand.remat && c >= 2) {
                    /* BUG-COMBO-STATIC-F16: effect marker. */
                    cg->twin_applied |= TWIN_APPLIED_REMAT;
                    /* Table-load hoisting: both nibble tables depend
                     * only on the row coefficient c, so they are loaded
                     * once per (i,j) -- not once per (i,j,u). The u-loop
                     * below runs inside the tables' live scope: NIBLO
                     * in tbl, NIBHI in the mask register repurposed as
                     * the second table scratch (the mask itself is dead
                     * here -- the u-loop uses vand.vi with the 0x0F
                     * immediate -- so the proven (n_out+1)*U+6 pressure
                     * formula is unchanged). The old comment claiming
                     * "2 loads per use are inherent" was wrong about
                     * the per-u factor: the per-u reloads were never
                     * inherent, only the single extra live table. */
                    c_ldtab1(cg, cg->tbl, CK_NIBLO, c, win);
                    c_ldtab1(cg, cg->mask, CK_NIBHI, c, win);
                }
                for (int u = 0; u < nu; u++) {
                    int acc = vreg(cg, j * U + u), vd = cg->vd0 + u * cg->lmul;
                    if (c == 1) {
                        c_vxor(cg, acc, acc, vd);
                    } else if (c == 0) {
                        /* acc = 0*acc + t = t: a move, not zero+xor */
                        c_vmv(cg, acc, vd);
                    } else if (cg->zvbc) {
                        /* Carryless path: the coefficient broadcast for
                         * this row lives in hreg[2j] (cvec slot;
                         * hreg[2j+1] is unused here).
                         * Horner step: acc = acc*C + t. The product is
                         * formed in r1 and XORed with t in a single
                         * vxor.vv acc,r1,vd -- fusing the "vmv acc,r1 /
                         * vxor acc,acc,vd" pair (RVV allows distinct
                         * rd/rs1/rs2). The fused form is vl-agnostic,
                         * so the tail (win=1) needs no special casing.
                         * acc must be OVERWRITTEN with the step result
                         * (not accumulated: xor-into-acc would compute
                         * acc^product, wrong). */
                        int cv = cg->hreg[2 * j];
                        assert(cv >= 0);
                        c_gfmul_red_zvbc(cg, acc, cv);
                        c_vxor(cg, acc, cg->r1, vd);
                    } else {
                        if (cg->cand.remat) {
                            /* Tables are already live from the per-(i,j)
                             * hoist above (NIBLO in tbl, NIBHI in the
                             * repurposed mask register); the u-loop does
                             * the two gathers and combines. vand.vi
                             * carries the 0x0F mask as an immediate, so
                             * no mask register is needed. */
                            c_vand_vi(cg, cg->tlo, acc, 15);
                            c_vsrl4(cg, cg->thi, acc);
                            c_vrg(cg, cg->r1, cg->tbl, cg->tlo);
                            c_vrg(cg, cg->r2, cg->mask, cg->thi);
                        } else {
                            c_vand(cg, cg->tlo, acc, cg->mask);
                            c_vsrl4(cg, cg->thi, acc);
                            /* -1 tripwire: the remat branch above is the
                             * only other user of this path; here the
                             * slots must hold assigned table regs. */
                            assert(cg->hreg[2 * j] >= 0 &&
                                   cg->hreg[2 * j + 1] >= 0);
                            c_vrg(cg, cg->r1, cg->hreg[2 * j], cg->tlo);
                            c_vrg(cg, cg->r2, cg->hreg[2 * j + 1],
                                  cg->thi);
                        }
                        c_vxor(cg, cg->r1, cg->r1, cg->r2);
                        c_vxor(cg, acc, cg->r1, vd);
                    }
                }
            }
        }
    }

    /* stores */
    for (int j = 0; j < n_out; j++) {
        c_ldptr(cg, T2, A1, j);
        em_add(e, T2, T2, T1);
        for (int u = 0; u < nu; u++) {
            c_vse(cg, vreg(cg, j * U + u), T2);
            if (u < nu - 1)
                em_add(e, T2, T2, T0);
        }
    }
}

/* ---------------- recovery closed-form twin (algos #4) ---------------- */

/* Coefficient setup for one GF(2^8) multiply: broadcast into cvec on
 * the Zvbc path, nibble-table pair load into (tbl_lo, tbl_hi) on the
 * nibble path. The caller performs the actual multiplies. */
static void rec_cf_coef_setup(cg_t *cg, uint8_t c, int win)
{
    if (cg->zvbc)
        c_broadcast(cg, cg->cvec, c);
    else
        c_ldtab(cg, c, win);
}

/* r1 = c * vs (GF(2^8)): the Zvbc carryless reduction, or the nibble
 * split + gathers. The coefficient setup (rec_cf_coef_setup) is the
 * caller's job. Temps: tlo/thi/r1/r2 (+kvec/cvec) like mulacc_rows_c. */
static void rec_cf_gfmul_r1(cg_t *cg, int vs)
{
    if (cg->zvbc) {
        c_gfmul_red_zvbc(cg, vs, cg->cvec);
    } else {
        c_vand(cg, cg->tlo, vs, cg->mask);
        c_vsrl4(cg, cg->thi, vs);
        c_vrg(cg, cg->r1, cg->tbl_lo, cg->tlo);
        c_vrg(cg, cg->r2, cg->tbl_hi, cg->thi);
        c_vxor(cg, cg->r1, cg->r1, cg->r2);
    }
}

/* Accumulate one closed-form syndrome into row k's accumulators:
 * acc_k[u] = XOR_s (ecoef[s] * x_s), u < nu. Input column s is
 * in[A0][s]; vd0+u*L are the data temps. Returns 1 when some
 * coefficient was nonzero (accumulators hold the syndrome), 0 when
 * the syndrome is empty (accumulators are zeroed). The first-touch
 * move (instead of xor-with-zero) mirrors gen_body's P_MULACC. */
static int gen_rec_cf_syndrome(cg_t *cg, int k, const uint8_t *ecoef, int ns,
                               int nu, int win)
{
    em_t *e = &cg->em;
    int U = cg->U, L = cg->lmul;
    int first = -1;
    for (int s = 0; s < ns; s++)
        if (ecoef[s]) {
            first = s;
            break;
        }
    if (first < 0) {
        for (int u = 0; u < nu; u++)
            c_vzero(cg, (k * U + u) * L);
        return 0;
    }
    for (int s = 0; s < ns; s++) {
        uint8_t c = ecoef[s];
        if (!c)
            continue;
        c_ldptr(cg, T2, A0, s);
        em_add(e, T2, T2, T1); /* + byte offset */
        c_prefetch_in(cg, T2);
        if (c != 1)
            rec_cf_coef_setup(cg, c, win);
        for (int u = 0; u < nu; u++) {
            int acc = (k * U + u) * L;
            int vd = cg->vd0 + u * L;
            c_vle(cg, vd, T2);
            if (u < nu - 1)
                em_add(e, T2, T2, T0); /* += vl */
            if (c == 1) {
                if (s == first)
                    c_vmv(cg, acc, vd); /* acc was 0 */
                else
                    c_vxor(cg, acc, acc, vd);
            } else if (cg->zvbc) {
                c_gfmul_acc_zvbc(cg, acc, vd, cg->cvec, s == first);
            } else {
                rec_cf_gfmul_r1(cg, vd);
                if (s == first)
                    c_vmv(cg, acc, cg->r1); /* acc was 0 */
                else
                    c_vxor(cg, acc, acc, cg->r1);
            }
        }
    }
    return 1;
}

/* Closed-form recovery body (algos #4, phase 2). The plan kind is
 * P_MULACC with the baked R matrix, but the twin emits the
 * syndrome+combine form from pl->rec_cf instead:
 *   e=1: x_f = cinv * XOR_s (ecoef[s] * x_s)
 *   e=2: p = XOR_s (ecoef1[s] * x_s), q = XOR_s (ecoef2[s] * x_s),
 *        x1 = (d*p ^ b*q) * detinv, x2 = (c*p ^ a*q) * detinv.
 * Bit-identical to the generic MULACC over the baked R (the closed
 * form is what the planner baked into R; see plan_recover_closed_form).
 *
 * Register use is exactly the P_MULACC map (n_out*U accumulators, U
 * vd0 data temps, tlo/thi/r1/r2/tbl_lo/tbl_hi/mask or
 * r1/r2/r3/cvec/kvec): kind == P_MULACC implies the baked R is
 * nontrivial, which implies the body performs at least one nontrivial
 * multiply (e=1: R[s] = ecoef[s]*cinv; e=2: an all-{0,1} body would
 * bake an all-{0,1} R), so the mask/table (nibble) and kvec/cvec
 * (Zvbc) registers the P_MULACC assignment derived from the baked R
 * are always sufficient -- no extra allocation, LMUL handled by the
 * shared *L stride.
 *
 * e=2 combine layout: p lives in the row-0 accumulators, q in the
 * row-1 accumulators. x1's value is parked in the vd temps: x2's
 * combine still needs both p and q, so x1 cannot go into the row-0
 * accumulators yet (that would destroy p). x2 is combined in place
 * into the row-1 accumulators (destination-as-accumulator, no extra
 * temp row needed). The stores then read x1 from the vd temps and x2
 * from row 1. */
static void gen_body_rec_cf(cg_t *cg, int nu, int win)
{
    /* BUG-COMBO-STATIC-F16: effect marker. */
    cg->twin_applied |= TWIN_APPLIED_REC_CF;
    em_t *e = &cg->em;
    const plan_t *pl = cg->pl;
    const struct rec_cf_data *cfd = pl->rec_cf;
    int U = cg->U, L = cg->lmul;
    assert(nu >= 1 && nu <= U);
    assert(cfd && (cfd->nf == 1 || cfd->nf == 2));
    assert(cfd->ns == pl->n_in);
    /* The twin is only enumerated for P_MULACC (see plan_candidates);
     * a force-built candidate on another kind fails the build via this
     * assert in debug builds. */
    assert(pl->kind == P_MULACC);

    if (cfd->nf == 1) {
        /* x_f = cinv * syndrome(ecoef) */
        gen_rec_cf_syndrome(cg, 0, cfd->ecoef, cfd->ns, nu, win);
        if (cfd->cinv != 1) {
            /* One final multiply by the plan-time constant c_f^-1
             * (strength-reducible: the planner only enumerates the
             * twin when cinv is a power of 2; emitted via the
             * standard multiply path for bit-identity). */
            rec_cf_coef_setup(cg, cfd->cinv, win);
            for (int u = 0; u < nu; u++) {
                int acc = (0 * U + u) * L;
                rec_cf_gfmul_r1(cg, acc);
                c_vmv(cg, acc, cg->r1);
            }
        }
    } else {
        /* p -> row 0, q -> row 1 */
        int tp = gen_rec_cf_syndrome(cg, 0, cfd->ecoef1, cfd->ns, nu, win);
        int tq = gen_rec_cf_syndrome(cg, 1, cfd->ecoef2, cfd->ns, nu, win);
        /* x1 = (d*p ^ b*q) * detinv, parked in the vd temps */
        {
            int tA = tp && cfd->d != 0;
            int tB = tq && cfd->b != 0;
            int have = 0;
            if (tA) {
                if (cfd->d != 1)
                    rec_cf_coef_setup(cg, cfd->d, win);
                for (int u = 0; u < nu; u++) {
                    int tt = cg->vd0 + u * L, ap = (0 * U + u) * L;
                    if (cfd->d == 1)
                        c_vmv(cg, tt, ap);
                    else {
                        rec_cf_gfmul_r1(cg, ap);
                        c_vmv(cg, tt, cg->r1);
                    }
                }
                have = 1;
            }
            if (tB) {
                if (cfd->b != 1)
                    rec_cf_coef_setup(cg, cfd->b, win);
                for (int u = 0; u < nu; u++) {
                    int tt = cg->vd0 + u * L, aq = (1 * U + u) * L;
                    if (cfd->b == 1) {
                        if (have)
                            c_vxor(cg, tt, tt, aq);
                        else
                            c_vmv(cg, tt, aq);
                    } else {
                        rec_cf_gfmul_r1(cg, aq);
                        if (have)
                            c_vxor(cg, tt, tt, cg->r1);
                        else
                            c_vmv(cg, tt, cg->r1);
                    }
                }
                have = 1;
            }
            if (!have) {
                for (int u = 0; u < nu; u++)
                    c_vzero(cg, cg->vd0 + u * L);
            } else if (cfd->detinv != 1) {
                rec_cf_coef_setup(cg, cfd->detinv, win);
                for (int u = 0; u < nu; u++) {
                    int tt = cg->vd0 + u * L;
                    rec_cf_gfmul_r1(cg, tt);
                    c_vmv(cg, tt, cg->r1);
                }
            }
        }
        /* x2 = (c*p ^ a*q) * detinv, combined in place into row 1.
         * a = E[r1][f1] != 0 by pivot choice, so the q-term is always
         * present when q is live; the destination doubles as the
         * accumulator (no extra temp row needed). */
        {
            int tA = tp && cfd->c != 0;
            int tB = tq; /* a != 0 always */
            if (!tA && !tB) {
                for (int u = 0; u < nu; u++)
                    c_vzero(cg, (1 * U + u) * L);
            } else {
                int have = 0;
                if (tB) {
                    if (cfd->a != 1)
                        rec_cf_coef_setup(cg, cfd->a, win);
                    for (int u = 0; u < nu; u++) {
                        int aj = (1 * U + u) * L;
                        if (cfd->a != 1) {
                            /* aj holds q; the multiply reads it
                             * before the move overwrites it */
                            rec_cf_gfmul_r1(cg, aj);
                            c_vmv(cg, aj, cg->r1);
                        }
                        /* a == 1: aj already holds q */
                    }
                    have = 1;
                }
                if (tA) {
                    if (cfd->c != 1)
                        rec_cf_coef_setup(cg, cfd->c, win);
                    for (int u = 0; u < nu; u++) {
                        int aj = (1 * U + u) * L, ap = (0 * U + u) * L;
                        if (cfd->c == 1) {
                            if (have)
                                c_vxor(cg, aj, aj, ap);
                            else
                                c_vmv(cg, aj, ap);
                        } else {
                            rec_cf_gfmul_r1(cg, ap);
                            if (have)
                                c_vxor(cg, aj, aj, cg->r1);
                            else
                                c_vmv(cg, aj, cg->r1);
                        }
                    }
                    have = 1;
                }
                if (cfd->detinv != 1) {
                    rec_cf_coef_setup(cg, cfd->detinv, win);
                    for (int u = 0; u < nu; u++) {
                        int aj = (1 * U + u) * L;
                        rec_cf_gfmul_r1(cg, aj);
                        c_vmv(cg, aj, cg->r1);
                    }
                }
            }
        }
    }

    /* stores */
    if (cfd->nf == 1) {
        assert(pl->n_out == 1);
        c_ldptr(cg, T2, A1, 0);
        em_add(e, T2, T2, T1);
        for (int u = 0; u < nu; u++) {
            c_vse(cg, (0 * U + u) * L, T2);
            if (u < nu - 1)
                em_add(e, T2, T2, T0);
        }
    } else {
        assert(pl->n_out == 2);
        /* x1 parked in the vd temps, x2 in the row-1 accumulators */
        c_ldptr(cg, T2, A1, 0);
        em_add(e, T2, T2, T1);
        for (int u = 0; u < nu; u++) {
            c_vse(cg, cg->vd0 + u * L, T2);
            if (u < nu - 1)
                em_add(e, T2, T2, T0);
        }
        c_ldptr(cg, T2, A1, 1);
        em_add(e, T2, T2, T1);
        for (int u = 0; u < nu; u++) {
            c_vse(cg, (1 * U + u) * L, T2);
            if (u < nu - 1)
                em_add(e, T2, T2, T0);
        }
    }
}

/* Long-range beq: B-type spans only +-4 KiB, so for targets that can
 * lie farther away (anything spanning the unrolled body), branch over
 * a J-type jal instead. The bne itself only skips one instruction. */
static void c_beq_long(cg_t *cg, int rs1, int rs2, int lab)
{
    em_t *e = &cg->em;
    int skip = em_label(e);
    em_bne(e, rs1, rs2, skip);
    em_jal(e, X0, lab);
    em_bind(e, skip);
}

/* Branch to lab if rs <= 0 (signed). The maskedtail twin's per-chunk
 * skip spans the unrolled single-chunk body, which can exceed the
 * B-type +-4 KiB range, so this is unconditionally the long form:
 * blt x0, rs, skip falls through into the chunk when rs > 0, else
 * jal x0, lab skips it. (blez is a pseudo-instruction for bge x0, rs;
 * the blt-over-jal shape keeps one branch on the fall-through path.) */
static void c_blez_long(cg_t *cg, int rs, int lab)
{
    em_t *e = &cg->em;
    int skip = em_label(e);
    em_blt(e, X0, rs, skip);
    em_jal(e, X0, lab);
    em_bind(e, skip);
}

/* Conditional branch to lab if rs1 != rs2. When the target label is
 * already bound (the counted loop's backward edge: cnt_lab is bound
 * before the body is emitted) and lies within the B-type +-4 KiB
 * range, a single bne suffices -- one control-flow instruction per
 * hot-loop iteration instead of two. Otherwise (forward reference,
 * or a body spanning >= 4 KiB) fall back to the beq-over-jal long
 * form, same motivation as c_beq_long. The distance is exact, not
 * optimistic: everything between the label and the branch is already
 * emitted, so no later fixup can move it. */
static void c_bne_long(cg_t *cg, int rs1, int rs2, int lab)
{
    em_t *e = &cg->em;
    if (lab >= 0 && lab < e->nlab && e->lpos[lab] != (size_t)-1) {
        long off = (long)e->lpos[lab] - (long)e->len;
        /* B-type range: 12-bit signed offset in units of 2 bytes
         * (em_resolve asserts the same bounds when patching). */
        if (off >= -4096 && off < 4096 && (off & 1) == 0) {
            em_bne(e, rs1, rs2, lab);
            return;
        }
    }
    int skip = em_label(e);
    em_beq(e, rs1, rs2, skip);
    em_jal(e, X0, lab);
    em_bind(e, skip);
}

/* Masked tail twin (TAIL_MASKED): unrolled full-VL masked tail replacing
 * the strip-mining remainder loop. Entered with A2 = remaining
 * (0..U*VLMAX-1; the counted loop guarantees < U*VLMAX, and 0 when len
 * is an exact multiple) and T1 = byte offset of the tail region.
 *
 *   vsetvli T0, x0, e8, m<lmul>, ta, ma   # T0 = vl = VLMAX (once)
 *   A3 = A2                               # chunk-0 remaining
 *   for u in 0..U-1 (unrolled at JIT time):
 *       blez A3 -> next                   # skip chunks fully past the end
 *       T2 = min(A3, T0)                  # clamp the bound at VLMAX:
 *                                         # A3 reaches U*VLMAX-1, but the
 *                                         # SEW=8 scalar truncates to 8 bits
 *       vid.v vd0                         # vd0[i] = i
 *       vmsltu.vx v0, vd0, T2             # v0[i] = (i <u T2)
 *       <gen_body(cg,1,0), data mem masked by v0>
 *   next:
 *       add T1, T1, T0                    # offset += VLMAX (even if skipped)
 *       sub A3, A3, T0                    # remaining -= VLMAX
 *
 * vl stays VLMAX throughout (no vl reduction); the mask selects active
 * lanes. Table loads stay unmasked at win=0: vl is provably VLMAX and
 * the pool entries are replicated to full VLMAX, so the vsetivli-16
 * window is unnecessary. Only DATA loads/stores are masked
 * (cg->masked_mem). Compute on masked-off lanes is don't-care;
 * fully-masked chunks execute no memory ops, so nothing past the tail
 * is ever touched (no fault, no over-write).
 *
 * vid.v targets vd0, never v0: V spec §16.9 raises illegal-insn when a
 * vid.v destination group overlaps the mask register (v0) at LMUL > 1.
 * vd0 is safe as the temp: every body writes it (data staging load)
 * before any read, or leaves it untouched (P_XOR direct-accumulator
 * path) -- it is never live-in to the body. vmsltu.vx itself is
 * unmasked (vm=1), so reading vd0 there is a pure
 * read-before-body-clobber.
 *
 * A3/A4 are free caller-saved temps across gen_body (it clobbers
 * T2-T5; T0 = vl, T1 = offset, T6 = pool base stay live). T0 still
 * holds VLMAX after each gen_body (win=0 never reseeds vl). */
static void gen_tail_masked(cg_t *cg)
{
    em_t *e = &cg->em;
    int U = cg->U;
    /* The empty-loop branch in gen_function targets this label when the
     * counted loop has zero iterations; the fall-through from the loop
     * reaches it too (a bound label is just a position marker). */
    em_bind(e, cg->tail_lab);
    em_vsetvli(e, T0, X0, VSEW8, cg->vlmul, 1, 1); /* T0 = vl = VLMAX */
    cg->uops += 1;
    em_addi(e, A3, A2, 0); /* A3 = remaining for chunk 0 */
    cg->uops += 1;
    for (int u = 0; u < U; u++) {
        int next_lab = em_label(e);
        c_blez_long(cg, A3, next_lab); /* A3 <= 0: skip the chunk */
        /* Clamp the compare bound at VLMAX: A3 (remaining for this
         * chunk) reaches U*VLMAX-1 on non-final chunks, but the SEW=8
         * scalar operand of vmsltu.vx truncates to 8 bits (V spec
         * §11.1) -- an unclamped A3 >= 256 would wrap the bound and
         * deactivate lanes that are really in range. T2 = min(A3, T0)
         * branchless: T2 = T0 + min(0, A3 - T0). T2/T3 are scratch:
         * gen_body clobbers them anyway, and the mask compare reads T2
         * before the body runs. A3 > 0 here (the blez skip above), and
         * T0 = VLMAX > 0, so the min is exact. */
        em_sub(e, T2, A3, T0);  /* T2 = A3 - VLMAX */
        em_srai(e, T3, T2, 63);  /* T3 = -1 if A3 < VLMAX else 0 */
        em_and(e, T2, T2, T3);   /* T2 = min(0, A3 - VLMAX) */
        em_add(e, T2, T0, T2);   /* T2 = min(A3, VLMAX) */
        cg->uops += 4;
        em_vid_v(e, cg->vd0);          /* vd0[i] = i */
        cg->uops += 1;
        /* v0[i] = (i <u T2): unsigned, exact -- T2 <= VLMAX <= 255
         * (backend gate), so neither vid.v (indexes mod 256) nor the
         * scalar truncation wraps. */
        em_vmsltu_vx(e, 0, cg->vd0, T2);
        cg->uops += 1;
        cg->masked_mem = 1;
        gen_body(cg, 1, 0);
        cg->masked_mem = 0;
        em_bind(e, next_lab);
        em_add(e, T1, T1, T0); /* offset += VLMAX (even when skipped) */
        em_sub(e, A3, A3, T0); /* remaining -= VLMAX */
        cg->uops += 2;
    }
}

static void gen_function(cg_t *cg)
{
    em_t *e = &cg->em;
    int U = cg->U;
    /* Power-of-two U (1,2,4) uses shifts; non-power-of-two (3,5,6,7,8)
     * uses divu/mul from the M extension. The choice is compile-time. */
    int lgu = U == 4 ? 2 : U == 2 ? 1 : 0;
    int use_shift = (U == 1 || U == 2 || U == 4);
    int cnt_lab = em_label(e), loop_lab = em_label(e),
        done_lab = em_label(e);

    /* len == 0 guard: vsetvli with rs1 == x0 selects VLMAX, not 0. */
    c_beq_long(cg, A2, X0, done_lab); /* spans the whole function */
    /* T6 = constant-pool base (PC-relative) */
    em_lla(e, T6, cg->pool_lab, 0);

    if (!cg->pl->noop) {
        gen_const_setup(cg);
        em_addi(e, T1, X0, 0); /* byte offset = 0 */
        unsigned al = cg->opts->loop_align ? (unsigned)cg->opts->loop_align
                                           : 16;
        if (al & (al - 1))
            al = 16; /* not a power of two: fall back */

        /* Counted full-group loop (F6): a group is U * VLMAX bytes and
         * every iteration consumes exactly one group with vl = VLMAX,
         * so the old per-iteration divu+mul (or srli+slli) collapse
         * into one prologue divu for the trip count plus one mul (or
         * shift) for the group size. A3/A4 are free caller-saved temps
         * across gen_body (it clobbers T2-T5; T0 = vl, T1 = offset and
         * T6 = pool base stay live). vl is set to VLMAX once up front;
         * gen_body preserves it. Table loads use a vsetivli-16 window
         * only when win=1 (the remainder path); in the counted loop
         * the replicated pool entries make the window unnecessary at
         * any LMUL. */
        /* T0 = vl = VLMAX (vsetvli with rs1 == x0 selects VLMAX) */
        em_vsetvli(e, T0, X0, VSEW8, cg->vlmul, 1, 1);
        cg->uops += 1;
        if (use_shift) {
            em_slli(e, A3, T0, (unsigned)lgu); /* A3 = U * VLMAX */
        } else {
            em_addi(e, A3, X0, U);
            em_mul(e, A3, T0, A3);             /* A3 = U * VLMAX */
        }
        em_divu(e, A4, A2, A3); /* A4 = niter = len / group */
        /* notail: len > 0 (the len==0 guard above already left for done)
         * and len % group == 0, so niter >= 1 always; retarget the
         * empty-loop branch at done since the remainder path (and its
         * loop_lab) is not emitted. maskedtail: the empty-loop branch
         * targets the masked tail's start label (tail_lab); the
         * strip-mining path's loop_lab is not emitted there. */
        if (cg->maskedtail)
            cg->tail_lab = em_label(e);
        c_beq_long(cg, A4, X0,
                   cg->notail ? done_lab
                              : cg->maskedtail ? cg->tail_lab : loop_lab);
        /* Pipe prologue (track I8): issue the first iteration's
         * first-column loads into Dn. After the empty-loop branch
         * (N >= 1), before the loop label; T1 == 0 here. Uncounted:
         * prologue setup is not per-iteration work. */
        if (cg->cand.pipe) {
            int cols[RJ_MAX_DISKS];
            if (pipe_used_columns(cg, cols) > 0) {
                double su = cg->uops, sm = cg->mem;
                pipe_load_column(cg, cols[0], cg->pipe_dn, T1);
                cg->uops = su;
                cg->mem = sm;
            }
        }
        em_align(e, al);
        em_bind(e, cnt_lab);
        if (cg->cand.rec_cf)
            gen_body_rec_cf(cg, U, 0);
        else if (cg->cand.pipe)
            gen_body_pipe(cg);
        else
            gen_body(cg, U, 0);
        em_add(e, T1, T1, A3);            /* offset += group */
        em_addi(e, A4, A4, -1);           /* niter-- */
        c_bne_long(cg, A4, X0, cnt_lab);  /* spans the body */
        /* notail twin: the counted loop consumed exactly len bytes
         * (dispatcher guarantees len % group == 0), so the remainder
         * computation and any tail are not emitted at all. */
        if (!cg->notail) {
            /* Remainder loop: steady-state-excluded from the uops/mem
             * model, exactly like the scalar tail and AArch64's
             * predicated tail (twins compare on the identical vector
             * body). */
            double save_uops = cg->uops, save_mem = cg->mem;
            em_sub(e, A2, A2, T1);            /* A2 = remaining */
            if (cg->maskedtail)
                gen_tail_masked(cg);
            else {
                /* Remainder: remaining < U * VLMAX.
                 * Process in chunks of up to VLMAX bytes using the
                 * single-unroll body. vsetvli computes vl =
                 * min(VLMAX, remaining) in hardware, so no division is
                 * needed (the old strip-mining approach used divu for
                 * non-power-of-two U, and ran at degraded vl =
                 * remaining / U). vl can be < 16 here, so table loads
                 * keep their vsetivli-16 window (gen_body win=1). */
                em_bind(e, loop_lab);
                em_vsetvli(e, T0, A2, VSEW8, cg->vlmul, 1, 1); /* T0 = vl */
                cg->uops += 1;
                c_beq_long(cg, T0, X0, done_lab); /* vl == 0 => done */
                if (cg->cand.rec_cf)
                    gen_body_rec_cf(cg, 1, 1);
                else
                    gen_body(cg, 1, 1);
                em_add(e, T1, T1, T0);             /* offset += vl */
                em_sub(e, A2, A2, T0);            /* remaining -= vl */
                em_jal(e, X0, loop_lab);          /* loop while remaining > 0 */
            }
        }
    }
    em_bind(e, done_lab);
    em_jalr(e, X0, RA, 0); /* ret */
    /* Trap sled after ret, unconditional: a fall-through past ret is a
     * codegen bug and must trap immediately, never slide through NOPs
     * into the const pool. */
    em_unimp(e);
}

/* ---------------- version build ---------------- */

int jit_build_version(const plan_t *pl, const cand_t *cand,
                      const rj_cpu_info *cpu, const rj_options *opts,
                      version_t *v, char *err, size_t errlen)
{
    (void)cpu;
    memset(v, 0, sizeof(*v));
    v->cand = *cand;
    v->kind = pl->kind;
    v->plan = pl; /* borrowed: bench validates against the owning plan */
    v->gbps = -1;
    v->gbps_measured = 0;
    v->notail = cand->notail;
    v->maskedtail = cand->maskedtail;

    /* This backend implements the RVV nibble path and the Zvbc carryless
     * path; the planner never offers anything else for ISA_RVV. */
    if (cand->isa != ISA_RVV) {
        snprintf(err, errlen, "riscv backend: not an RVV candidate");
        return -1;
    }
    if (cand->mul != MUL_NIBBLE && cand->mul != MUL_ZVBC) {
        snprintf(err, errlen, "riscv backend: only the nibble and Zvbc "
                              "carryless multiply methods are implemented");
        return -1;
    }
    /* Zvbc32e is the only extension that matters here: never emit
     * vclmul.vv/vclmulh.vv at SEW=8 without rvv_zvbc32e (on
     * Zvbc-without-Zvbc32e silicon the encoding is reserved ->
     * illegal-insn). The planner only enumerates MUL_ZVBC when
     * rvv_zvbc32e is present; re-check here so a force-built candidate
     * can never emit the instructions on the wrong CPU (defense in
     * depth). rvv_zvbc/rvv_zvkg are informational only. */
    if (cand->mul == MUL_ZVBC && !cpu->rvv_zvbc32e) {
        snprintf(err, errlen, "riscv backend: Zvbc carryless multiply "
                              "needs rvv_zvbc32e (SEW=8 vclmul.vv/"
                              "vclmulh.vv)");
        return -1;
    }
    /* The planner never generates remat twins for other plan kinds, and
     * the backend only implements remat for P_HORNER. User-reachable
     * via pinned candidates: fail cleanly, never abort. */
    if (cand->remat && pl->kind != P_HORNER) {
        snprintf(err, errlen,
                 "riscv backend: remat twin needs P_HORNER");
        return -1;
    }
    /* Software-pipelined loop twin (track I8): implemented for the
     * P_MULACC/P_HORNER nibble bodies only (non-remat). Anything else
     * fails the build cleanly -- never wrong code; the planner only
     * enumerates the supported combinations. */
    if (cand->pipe &&
        ((pl->kind != P_MULACC && pl->kind != P_HORNER) ||
         cand->mul != MUL_NIBBLE || cand->remat)) {
        snprintf(err, errlen,
                 "riscv backend: pipe twin needs P_MULACC/P_HORNER "
                 "nibble, non-remat");
        return -1;
    }
    /* Recovery closed-form emitter twin (algos #4): validated here, never
     * wrong code for force-built candidates. Requires P_MULACC, rec_cf_data,
     * nf in {1,2} matching n_out/n_in, and no other body twin.
     * BUG-COMBO-STATIC-F03: this gate excluded only pipe and remat, so a
     * pinned rec_cf+xorbit spec passed every gate and the rec_cf dispatch
     * (which runs before gen_body) silently dropped xorbit. Now excludes
     * the full body-twin list, mirroring x86_64/aarch64 and the L1 check. */
    if (cand->rec_cf) {
        const struct rec_cf_data *cfd = pl->rec_cf;
        if (!cfd || pl->kind != P_MULACC ||
            (cfd->nf != 1 && cfd->nf != 2) ||
            pl->n_out != cfd->nf || pl->n_in != cfd->ns ||
            cand->pipe || cand->remat || cand->xorbit || cand->xorsched ||
            cand->permi2b || cand->mshift) {
            snprintf(err, errlen,
                     "riscv backend: rec_cf twin needs P_MULACC with "
                     "rec_cf_data, no other body twins");
            return -1;
        }
    }

    cg_t cg;
    memset(&cg, 0, sizeof(cg));
    /* tbl_lo/tbl_hi are only assigned on the nibble path; -1 marks
     * them unassigned so c_ldtab's tripwire assert fires instead of
     * silently encoding v0 (see c_ldtab). */
    cg.tbl_lo = -1;
    cg.tbl_hi = -1;
    cg.pl = pl;
    cg.cand = *cand;
    cg.opts = opts;
    cg.U = cand->U;
    cg.notail = cand->notail;
    /* LMUL tuning axis (RVV only): validate and derive the vtype enum.
     * 0 (zero-initialized candidate) means 1. */
    {
        int lmul = cand->lmul ? cand->lmul : 1;
        if (lmul != 1 && lmul != 2 && lmul != 4 && lmul != 8) {
            snprintf(err, errlen,
                     "riscv backend: unsupported LMUL=%d (want 1/2/4/8)",
                     lmul);
            return -1;
        }
        cg.lmul = lmul;
        cg.vlmul = lmul == 8   ? VLMUL_M8
                   : lmul == 4 ? VLMUL_M4
                   : lmul == 2 ? VLMUL_M2
                               : VLMUL_M1;
    }
    /* predtail is an SVE-only twin (track B1sve, aarch64 backend):
     * reject cleanly rather than silently building a general twin. */
    if (cand->predtail) {
        snprintf(err, errlen,
                 "predtail twin requires SVE (not implemented for RISC-V)");
        return -1;
    }
    /* maskedtail twin (RVV TAIL_MASKED): disjoint from notail/predtail
     * -- a version is never both. The v0 mask reservation is applied
     * to the register map below (vreg()); the backend's register
     * pressure checks then reject unencodable twins cleanly (defense
     * in depth: the planner never enumerates maskedtail, so api.c only
     * builds it from a winner that already fit without the reservation). */
    if (cand->maskedtail && (cand->notail || cand->predtail)) {
        snprintf(err, errlen, "riscv backend: maskedtail is disjoint from "
                              "notail/predtail");
        return -1;
    }
    cg.maskedtail = cand->maskedtail;
    cg.zvbc = (cand->mul == MUL_ZVBC);
    /* True full-group size for the length dispatcher: U * VLMAX * lmul.
     * vlenb is a hardware constant, so reading it once at build time is
     * sound (also under QEMU: the C code sees the same VLEN as the
     * JITed code's vsetvli). It also sizes the replicated constant-pool
     * table entries (see pool_get). */
#ifdef RJ_WASM
    /* Compile-only (WebAssembly, tools/wasm/SPEC.md §S6.2b): the VLEN
     * comes from the synthetic CPU bundle (cpu->rvv_vlen). */
    cg.vlenb = (size_t)cpu->rvv_vlen;
#else
    cg.vlenb = read_vlenb();
#endif
    /* maskedtail mask math (vmsltu.vx at SEW=8) is exact only when VLMAX
     * fits in 8 bits: vid.v indexes mod 256 and the .vx scalar operand
     * truncates to SEW (V spec §11.1), so lmul*vlenb > 255 would wrap
     * the index on one side and the bound on the other. The twin is
     * simply not built on wider configs; the general strip-mining tail
     * covers all hardware. */
    if (cg.maskedtail && (size_t)cg.lmul * cg.vlenb > 255) {
        snprintf(err, errlen,
                 "riscv backend: maskedtail needs VLMAX <= 255 "
                 "(lmul=%d vlenb=%zu)",
                 cg.lmul, cg.vlenb);
        return -1;
    }
    v->group_bytes = (size_t)cg.U * cg.vlenb * (size_t)cg.lmul;

    /* cand->ternlog: not implemented for RVV; ignored.
     * cand->nt: honored (Zihintntl NTL.ALL before each output store;
     * see c_vse).
     * cand->prefetch_dist: honored when the CPU reports Zicbop. The
     * planner only enumerates prefetch twins then; a force-built
     * candidate without Zicbop degrades to temporal (never wrong
     * code). An unencodable distance fails the build cleanly. A noop
     * plan (n_out == 0) emits no loop at all, so the twin degrades to
     * temporal too -- the "+pf" suffix below therefore means ">= 1
     * prefetch emitted", exactly. */
    if (cand->prefetch_dist && cpu->rvv_zicbop && !pl->noop) {
        size_t d = cand->prefetch_dist;
        if (d < 32 || d > 2016 || (d & 31u) != 0) {
            snprintf(err, errlen, "riscv backend: prefetch distance %zu "
                             "not encodable (Zicbop prefetch.r: 32..2016, "
                             "multiple of 32)", d);
            return -1;
        }
        cg.emit_pf = 1;
        cg.pf_off = (unsigned)d;
    }

    /* The nibble vector path needs the 0x0F mask under the same condition
     * the scalar tail would need log/exp tables (there is no scalar tail
     * here: vsetvli strip-mining handles every length). P_DELTA with an
     * all-{0,1} delta matrix (e.g. RAID-5) needs no mask. The Zvbc
     * carryless path works on whole SEW=8 lanes and needs no mask at
     * all. Single source of truth: plan_any_nontrivial (see plan.h);
     * for P_DELTA it scans the n_out x nchanged delta columns, not
     * n_out x n_in. */
    if (!cg.zvbc && (pl->kind == P_MULACC || pl->kind == P_DELTA ||
                     pl->kind == P_HORNER))
        cg.need_mask = plan_any_nontrivial(pl);

    /* vector register assignment (v0..v31); fail the candidate cleanly if
     * it does not fit -- the planner then tries a smaller unroll. All
     * assigned numbers are PHYSICAL group heads: logical register r maps
     * to r*L, so every head is L-aligned as LMUL>1 requires. */
    int n_out = pl->n_out, U = cand->U;
    const int L = cg.lmul;
    if (cand->xorbit) {
        /* Bit-program twin: n_out*U accumulators + input-byte reg +
         * two bit temps. No mask reg needed (vand.vi takes immediate).
         * Must stay in lockstep with xorbit_need_regs. */
        assert(pl->bit_sched && pl->bit_sched->ntemps > 0);
        assert(pl->kind == P_MULACC || pl->kind == P_DELTA ||
               pl->kind == P_HORNER);
        cg.xbt_in = n_out * U;
        cg.xbt_r1 = n_out * U + 1;
        cg.xbt_r2 = n_out * U + 2;
        int vneed = (n_out * U + 3) * L;
        if (vneed > 32) {
            snprintf(err, errlen, "riscv backend: register pressure "
                                  "(xorbit needs %d regs)", vneed);
            return -1;
        }
        int want_regs = xorbit_need_regs(ISA_RVV, pl, U);
        assert(vneed == want_regs * L);
    } else if (pl->kind == P_XOR) {
        cg.vd0 = vreg(&cg, n_out * U); /* single temp */
        int vneed = cg.vd0 + L;
        if (cand->xorsched) {
            /* XOR-schedule twin: temp formation needs two live vector
             * temps (vd0, vd1); RVV has no vector XOR-with-memory. */
            cg.vd1 = cg.vd0 + L;
            vneed = cg.vd0 + 2 * L;
        }
        if (vneed > 32) {
            snprintf(err, errlen, "riscv backend: register pressure "
                                  "(xor%s needs %d regs)",
                     cand->xorsched ? "+xsched" : "", vneed);
            return -1;
        }
        int want_regs = cand->xorsched ? xorsched_need_regs(ISA_RVV, pl, U)
                                       : plan_need_regs(ISA_RVV, cand->mul,
                                                       pl, U, cand->remat, cand->pipe,
                                                       0, cand->crc_fold);
        /* The maskedtail twin reserves v0 on top of the base map, for
         * either body shape. */
        want_regs += cand->maskedtail ? 1 : 0;
        assert(vneed == want_regs * L);
    } else if (pl->kind == P_HORNER) {
        cg.vd0 = vreg(&cg, n_out * U); /* U data group heads */
        cg.hreg = calloc((size_t)n_out * 2, sizeof(int));
        if (!cg.hreg) {
            snprintf(err, errlen, "out of memory");
            return -1;
        }
        int r;
        if (cg.zvbc) {
            /* Zvbc carryless path: r1/r2/r3 temps, kvec (0x1B reduction
             * constant), and one cvec coefficient broadcast per
             * nontrivial row in hreg[2j] (hreg[2j+1] stays -1; c in
             * {0,1} needs no multiply). No tables, no mask. remat is a
             * nibble-table concept and is meaningless here -- the
             * planner never enumerates zvbc+remat twins. */
            if (cand->remat) {
                snprintf(err, errlen, "riscv backend: remat is not "
                                      "implemented for the Zvbc path");
                free(cg.hreg);
                return -1;
            }
            cg.r1 = cg.vd0 + U * L;
            cg.r2 = cg.r1 + L;
            cg.r3 = cg.r2 + L;
            cg.kvec = cg.r3 + L;
            r = cg.kvec + L;
            for (int j = 0; j < n_out; j++) {
                uint8_t c = pl->horner_c[j];
                if (c == 0 || c == 1) {
                    cg.hreg[2 * j] = cg.hreg[2 * j + 1] = -1;
                    continue;
                }
                cg.hreg[2 * j] = r;
                r += L;
                cg.hreg[2 * j + 1] = -1;
            }
        } else {
            /* Pipe twin (track I8): the Dn dataset (U group heads)
             * sits between vd0/D and the temps. */
            if (cand->pipe)
                cg.pipe_dn = cg.vd0 + U * L;
            else
                cg.pipe_dn = -1;
            cg.tlo = (cand->pipe ? cg.pipe_dn : cg.vd0) + U * L;
            cg.thi = cg.tlo + L;
            cg.r1 = cg.thi + L;
            cg.r2 = cg.r1 + L;
            cg.mask = cg.r2 + L;
            if (cand->remat) {
                /* BUG_t5_remat (lead L5): effect marker at mode entry.
                 * The remat twin's code path starts here in register
                 * allocation (no persistent table regs; hreg[] stays
                 * -1, tables reloaded from the pool into tbl on each
                 * use) -- taken for every remat build, including
                 * all-trivial plans where no per-coefficient pool
                 * reload ever fires in the body. Marking only the body
                 * reload site made the F16 postcondition misfire on
                 * all-trivial plans; the body site remains as the
                 * per-use instance of the same path. */
                cg.twin_applied |= TWIN_APPLIED_REMAT;
                /* Rematerialize: no persistent table regs; hreg[] stays
                 * -1 and is never read (gen_const_setup and the body
                 * both branch on cand.remat, and the asserts at the
                 * hreg[] read sites trip if a -1 ever reached the
                 * emitter). Tables are reloaded from the pool once per
                 * (i,j) into tbl (NIBLO) and the mask register,
                 * repurposed as the NIBHI table scratch (the mask
                 * itself is dead in remat mode: the body uses vand.vi
                 * with the 0x0F immediate, so no extra register is
                 * needed and the allocation below is unchanged).
                 * Pressure proof: n_out*U accumulators
                 * (logical v0..n_out*U-1, i.e. physical heads shifted up
                 * by one LMUL-group under maskedtail's vreg() map), U
                 * data regs
                 * (vd0..vd0+U-1 = n_out*U..(n_out+1)*U-1), tlo, thi, r1,
                 * r2, mask, tbl: exactly (n_out+1)*U+6 vector regs,
                 * matching the plan_need_regs P_HORNER remat branch
                 * (asserted below, scaled by L for physical heads). */
                for (int j = 0; j < n_out; j++)
                    cg.hreg[2 * j] = cg.hreg[2 * j + 1] = -1;
                cg.tbl = cg.mask + L;
                r = cg.tbl + L;
            } else {
                r = cg.mask + L;
                for (int j = 0; j < n_out; j++) {
                    uint8_t c = pl->horner_c[j];
                    if (c == 0 || c == 1) {
                        cg.hreg[2 * j] = cg.hreg[2 * j + 1] = -1;
                        continue;
                    }
                    cg.hreg[2 * j] = r;
                    r += L;
                    cg.hreg[2 * j + 1] = r;
                    r += L;
                }
            }
        }
        if (r > 32) {
            snprintf(err, errlen, "riscv backend: register pressure "
                                  "(horner%s needs %d regs)",
                     cand->remat ? " remat" : "", r);
            free(cg.hreg);
            return -1;
        }
        assert(r == plan_need_regs(ISA_RVV, cand->mul, pl, U, cand->remat, cand->pipe,
                                       cand->maskedtail, cand->crc_fold) * L);
    } else { /* P_MULACC / P_DELTA */
        cg.vd0 = vreg(&cg, n_out * U); /* U data group heads */
        if (cg.zvbc && plan_any_nontrivial(pl)) {
            /* Zvbc carryless path: r1/r2/r3 temps, one cvec coefficient
             * broadcast, kvec (0x1B reduction constant). P_DELTA
             * additionally keeps its tlo/thi delta-formation temps
             * (same role as in the nibble path). No tables, no mask. */
            int r = cg.vd0 + U * L;
            if (pl->kind == P_DELTA) {
                cg.tlo = r;
                cg.thi = r + L;
                r += 2 * L;
            }
            cg.r1 = r;
            cg.r2 = r + L;
            cg.r3 = r + 2 * L;
            cg.cvec = r + 3 * L;
            cg.kvec = r + 4 * L;
            int vneed = cg.kvec + L;
            if (vneed > 32) {
                snprintf(err, errlen, "riscv backend: register pressure "
                                      "(zvbc mulacc/delta needs %d regs)",
                         vneed);
                return -1;
            }
            assert(vneed ==
                   plan_need_regs(ISA_RVV, cand->mul, pl, U, cand->remat, cand->pipe,
                                       cand->maskedtail, cand->crc_fold) *
                       L);
        } else if (cg.zvbc) {
            /* All-trivial coefficients: no multiplies, no kvec/cvec.
             * P_DELTA still needs its tlo/thi delta-formation temps
             * (borrowed by the delta body). */
            int vneed = cg.vd0 + U * L;
            if (pl->kind == P_DELTA) {
                cg.tlo = cg.vd0 + U * L;
                cg.thi = cg.tlo + L;
                vneed = cg.thi + L;
            }
            if (vneed > 32) {
                snprintf(err, errlen, "riscv backend: register pressure "
                                      "(zvbc trivial delta needs %d regs)",
                         vneed);
                return -1;
            }
            assert(vneed ==
                   plan_need_regs(ISA_RVV, cand->mul, pl, U, cand->remat, cand->pipe,
                                       cand->maskedtail, cand->crc_fold) *
                       L);
        } else if (cg.need_mask) {
            /* Pipe twin (track I8): the Dn dataset (U group heads)
             * sits between vd0/D and the temps. */
            if (cand->pipe)
                cg.pipe_dn = cg.vd0 + U * L;
            else
                cg.pipe_dn = -1;
            cg.tlo = (cand->pipe ? cg.pipe_dn : cg.vd0) + U * L;
            cg.thi = cg.tlo + L;
            cg.r1 = cg.thi + L;
            cg.r2 = cg.r1 + L;
            cg.tbl_lo = cg.r2 + L;
            cg.tbl_hi = cg.tbl_lo + L;
            cg.mask = cg.tbl_hi + L;
            /* Table hoisting: when the whole matrix uses a single
             * nontrivial coefficient, its table pair is loop-invariant
             * and can live in tbl_lo/tbl_hi for the entire function
             * (loaded once in gen_const_setup) instead of being
             * reloaded per column. Register pressure is unchanged (the
             * same two registers, now persistent). With >= 2 distinct
             * nontrivial coefficients the per-column reloads stay:
             * keeping a table pair live per coefficient would cost
             * 2*(ndistinct-1) extra registers against the 32-register
             * budget, and with replicated pool entries each reload is
             * only 4 instructions in the hot loop. */
            {
                uint8_t seen[256] = { 0 };
                int ndistinct = 0;
                int ncols =
                    pl->kind == P_DELTA ? pl->nchanged : pl->n_in;
                for (int k = 0; k < n_out * ncols; k++) {
                    uint8_t c = pl->mat[k];
                    if (c < 2 || seen[c])
                        continue;
                    seen[c] = 1;
                    cg.hoist_c = c;
                    ndistinct++;
                }
                cg.hoist_tables = (ndistinct == 1);
            }
            if (cg.mask + L > 32) {
                snprintf(err, errlen, "riscv backend: register pressure "
                                      "(mulacc needs %d regs)",
                         cg.mask + L);
                return -1;
            }
            assert(cg.mask + L ==
                   plan_need_regs(ISA_RVV, cand->mul, pl, U, cand->remat, cand->pipe,
                                       cand->maskedtail, cand->crc_fold) *
                       L);
        } else if (pl->kind == P_DELTA) {
            /* All-{0,1} delta matrix: the nibble tables, r1/r2 and the
             * mask are dead and not allocated. tlo/thi stay live,
             * borrowed as the delta temps (the body loads old/new into
             * them for every column unconditionally). */
            cg.tlo = cg.vd0 + U * L;
            cg.thi = cg.tlo + L;
            if (cg.thi + L > 32) {
                snprintf(err, errlen, "riscv backend: register pressure "
                                      "(delta needs %d regs)",
                         cg.thi + L);
                return -1;
            }
            assert(cg.thi + L ==
                   plan_need_regs(ISA_RVV, cand->mul, pl, U, cand->remat, cand->pipe,
                                       cand->maskedtail, cand->crc_fold) *
                       L);
        } else {
            /* P_MULACC with all-{0,1} coefficients is unreachable from
             * the planner (it would be P_XOR); only the U data regs
             * survive. */
            if (cg.vd0 + U * L > 32) {
                snprintf(err, errlen, "riscv backend: register pressure "
                                      "(mulacc needs %d regs)",
                         cg.vd0 + U * L);
                return -1;
            }
            assert(cg.vd0 + U * L ==
                   plan_need_regs(ISA_RVV, cand->mul, pl, U, cand->remat, cand->pipe,
                                       cand->maskedtail, cand->crc_fold) *
                       L);
        }
    }

    uint8_t *buf = malloc(1 << 20);
    if (!buf) {
        free(cg.hreg);
        snprintf(err, errlen, "out of memory");
        return -1;
    }
    em_init(&cg.em, buf, 1 << 20);
    cg.pool_lab = em_label(&cg.em);
    cg.pool_cap = 8192;
    cg.pool = malloc(cg.pool_cap);
    if (!cg.pool) {
        free(buf);
        free(cg.hreg);
        snprintf(err, errlen, "out of memory");
        return -1;
    }

    gen_function(&cg);

    /* constant pool after the code. The dead space between ret and the
     * pool is trap-filled (never NOPs): see em_trap_align. */
    em_trap_align(&cg.em, 16);
    em_bind(&cg.em, cg.pool_lab);
    size_t code_size = cg.em.len;
    for (size_t i = 0; i < cg.pool_len; i++)
        em_byte(&cg.em, cg.pool[i]);
    em_resolve(&cg.em);
    /* BUG-COMBO-STATIC-F16: twin effect postcondition. If a twin was
     * requested but its code path was never taken, the pin was a no-op:
     * reject cleanly instead of recording a false twin label. Placed
     * before the mem install so the fail path doesn't leak the mapping. */
    {
        struct { int req; uint32_t bit; const char *name; } twins[] = {
            { cand->nt, TWIN_APPLIED_NT, "nt" },
            { cand->prefetch_dist != 0, TWIN_APPLIED_PREFETCH, "prefetch" },
            { cand->remat, TWIN_APPLIED_REMAT, "remat" },
            { cand->xorsched, TWIN_APPLIED_XORSCHED, "xorsched" },
            { cand->xorbit, TWIN_APPLIED_XORBIT, "xorbit" },
            { cand->pipe, TWIN_APPLIED_PIPE, "pipe" },
            { cand->rec_cf, TWIN_APPLIED_REC_CF, "rec_cf" },
        };
        for (size_t ti = 0; ti < sizeof(twins)/sizeof(twins[0]); ti++) {
            if (twins[ti].req && !(cg.twin_applied & twins[ti].bit)) {
                if (err && errlen > 0)
                    snprintf(err, errlen,
                             "%s twin requested but its code path was not "
                             "taken (phantom twin)",
                             twins[ti].name);
                goto fail;
            }
        }
    }
    v->twin_applied = cg.twin_applied;
    /* OOM check: the emitter sets cg.em.err on buffer overflow or
     * label/fixup realloc failure, and pool_get sets cg.oom on realloc
     * failure. Either way the generated code is incomplete -- reject
     * cleanly instead of installing it. Placed before the mem install
     * so the fail path doesn't leak the mmap'd page. */
    if (cg.em.err || cg.oom) {
        snprintf(err, errlen, "riscv backend: out of memory during codegen");
        goto fail;
    }
    size_t total = cg.em.len;

    long ps = sysconf(_SC_PAGESIZE);
    size_t memsz = (total + (size_t)ps - 1) & ~((size_t)ps - 1);
#ifdef RJ_WASM
    /* Compile-only (WebAssembly, tools/wasm/SPEC.md §S6.2): no
     * mmap/mprotect in the browser and no I-cache maintenance to
     * perform. The emitters are position-independent, so the bytes are
     * the contract; keep them in a plain malloc'd block. Native builds
     * never define RJ_WASM. */
    (void)ps;
    void *mem = malloc(total ? total : 1);
    if (!mem) {
        snprintf(err, errlen, "malloc failed");
        goto fail;
    }
    memsz = total;
    memcpy(mem, buf, total);
#else
    void *mem = mmap(NULL, memsz, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) {
        snprintf(err, errlen, "mmap failed");
        goto fail;
    }
    memcpy(mem, buf, total);
    /* RISC-V I-caches are not coherent: flush before executing.
     * (QEMU masks this; real hardware crashes without fence.i.) */
    __builtin___clear_cache((char *)mem, (char *)mem + total);
    if (mprotect(mem, memsz, PROT_READ | PROT_EXEC) != 0) {
        snprintf(err, errlen, "mprotect RX failed");
        munmap(mem, memsz);
        goto fail;
    }
#endif

    v->mem = mem;
    v->memsz = memsz;
    v->fn = (rj_fn)mem;
    /* Nominal strip length scales with LMUL: each vector instruction
     * processes lmul*NOMVL bytes (SEW=8). uops/mem are counted per
     * instruction, so the per-byte model divides by the widened strip. */
    v->vl = NOMVL * cg.lmul; /* nominal VL for strip-mining seam validation in bench */
    v->uops_per_byte = cg.uops / (double)(cg.U * NOMVL * cg.lmul);
    v->mem_per_byte = cg.mem / (double)(cg.U * NOMVL * cg.lmul);
    v->code_size = code_size;
    v->const_size = cg.pool_len;

    const char *strat = pl->kind == P_XOR    ? "xor"
                        : pl->kind == P_HORNER ? "horner"
                        : pl->kind == P_DELTA  ? "delta"
                                               : "mulacc";
    char lmuls[16];
    /* LMUL marker: only shown when >1 (lmul=1 is the historical default
     * and keeps existing descs byte-identical). */
    if (cg.lmul > 1)
        snprintf(lmuls, sizeof(lmuls), " lmul%d", cg.lmul);
    else
        lmuls[0] = '\0';
    if (pl->kind == P_XOR)
    /* %.65s on the plan name: the provable -Wformat-truncation bound
     * for the 128B desc (fixed suffix <= 63B; plan names can reach
     * 127B since the A5 "[rows ...]" suffix). The old name[64]
     * already truncated names at 63B, so this preserves strictly
     * more, never less. */
        snprintf(v->desc, sizeof(v->desc), "%.65s %s %s U=%d%s%s%s%s%s%s",
                 pl->name, strat, isa_name(cand->isa), cand->U, lmuls,
                 cand->xorsched ? "+xsched" : "",
                 cand->nt ? "+nt" : "", cg.emit_pf ? "+pf" : "",
                 cand->notail ? " notail" : "",
                 cand->maskedtail ? " maskedtail" : "");
    else
        snprintf(v->desc, sizeof(v->desc), "%.65s %s %s+%s U=%d%s%s%s%s%s%s%s%s",
                 pl->name, strat, isa_name(cand->isa),
                 cg.zvbc ? "zbc" : "nibble", cand->U, lmuls,
                 cand->remat ? "+remat" : "", cand->nt ? "+nt" : "",
                 cg.emit_pf ? "+pf" : "",
                 cand->pipe ? "+pipe" : "",
                 cand->rec_cf ? "+reccf" : "",
                 cand->notail ? " notail" : "",
                 cand->maskedtail ? " maskedtail" : "");

    free(buf);
    free(cg.pool);
    free(cg.cent);
    free(cg.hreg);
    em_free(&cg.em);
    return 0;

fail:
    free(buf);
    free(cg.pool);
    free(cg.cent);
    free(cg.hreg);
    em_free(&cg.em);
    return -1;
}

void jit_free_version(version_t *v)
{
    if (v->mem) {
#ifdef RJ_WASM
        /* Paired with the malloc pivot in jit_build_version (§S6.2). */
        free(v->mem);
#else
        munmap(v->mem, v->memsz);
#endif
    }
    memset(v, 0, sizeof(*v));
}
