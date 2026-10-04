/* XOR array codes for RAID-6: RDP, EVENODD, Liberation.
 *
 * These are binary (XOR-only) MDS array codes with 2 parity disks. Each
 * data disk holds R rows of bytes; parity rows mix bytes ACROSS rows of
 * the data disks (diagonal parity), so they cannot be expressed as a flat
 * npar x ndata GF(2^8) matrix over disk streams -- the JIT's codegen is
 * offset-uniform (every input is read at the same byte offset).
 *
 * Lowering: each ROW is exposed as a separate logical disk ("expanded"
 * model). With k data disks of R rows:
 *   - expanded ndata = k*R, expanded npar = 2*R
 *   - in[d*R + r]  = row r of data disk d   (d in [0,k), r in [0,R))
 *   - out[r]      = P parity row r
 *   - out[R + r]  = Q parity row r
 * Every parity row is a pure XOR of input rows, so the expanded matrix is
 * binary and plan_build() picks the existing P_XOR path automatically --
 * encode, recovery (via the generic GF(2^8) erasure solver, whose solution
 * of a binary system is binary), everything. No separate plan path.
 *
 * A whole-disk failure of data disk d = failure of the R expanded disks
 * d*R .. d*R+R-1; pass all of them to rj_op_recover. rj_xorcode_rows()
 * reports R for an XOR-code topology (0 for anything else) so callers can
 * build the expanded index lists.
 *
 * Primary sources for the constructions:
 *   RDP:      Corbett et al., "Row-Diagonal Parity for Double Disk Failure
 *             Correction", FAST '04. p prime > 2, k = p-1 data disks,
 *             R = p-1 rows. Q includes the row-parity blocks on the
 *             diagonal ("Each diagonal parity block contains the even
 *             parity of the data and row parity blocks in the same
 *             diagonal", section 4); diagonal p-1 is the missing diagonal
 *             (not stored).
 *   EVENODD:  Blaum et al., "EVENODD: An Efficient Scheme for Tolerating
 *             Double Disk Failures in RAID Architectures", IEEE Trans.
 *             Computers 1995 (as presented in the "Code Design for
 *             Dependable Systems" text, Example 14.2). p prime, k = p data
 *             disks, R = p-1 rows (imaginary row p-1 = 0). Q rows carry the
 *             S ("missing diagonal") adjustment that makes the code MDS.
 *   Liberation: Plank, "The RAID-6 Liberation Codes", FAST '08, section
 *             3.3. w prime > 2, w >= k, k data disks, R = w rows.
 *             Q row r = XOR over disks i of (X_i row r . data disk i),
 *             with X_0 = I_w and X_i = I_w^->i + O_w[y, y+i-1],
 *             y = i*(w-1)/2 mod w.
 */
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Which XOR code (if any) backs an rj_topo_raid6() topology. */
typedef enum {
    XORCODE_RS = 0,      /* classic Reed-Solomon (Horner path) */
    XORCODE_RDP,         /* row-diagonal parity */
    XORCODE_EVENODD,     /* EVENODD */
    XORCODE_LIBERATION   /* Liberation */
} xorcode_choice_t;

/* The four topology constructors (rj_topo_rdp, rj_topo_evenodd,
 * rj_topo_liberation, rj_topo_raid6_rs) and rj_xorcode_rows() are declared
 * in the public header include/raidjit.h; this header keeps the internal
 * declarations. */
struct rj_topology;

/* Benchmark-selected default code for rj_topo_raid6(ndata). Pure table
 * lookup over docs/xor_code_default.md's recorded winners; the table in
 * src/xorcode.c is transcribed from that file and test_xorcode.c checks
 * they agree. Returns XORCODE_RS for ndata with no applicable XOR code. */
xorcode_choice_t xorcode_default_for_ndata(int ndata);

/* Matrix builders: fill m (npar*ndata binary entries, row-major) for the
 * expanded model. rdp_npar(p) = 2*(p-1) etc. Return 0 on success, -1 when
 * the parameters are outside the validity envelope. */
int xorcode_rdp_matrix(int p, uint8_t *m);
int xorcode_evenodd_matrix(int p, uint8_t *m);
int xorcode_liberation_matrix(int k, int w, uint8_t *m);

/* Validity predicates (also the constructor gates). These test the
 * MATHEMATICAL envelope only (primality, k/w relations); the constructors
 * additionally enforce the engineering envelope — the expanded topology
 * must fit RJ_MAX_DISKS disks — so e.g. xorcode_rdp_valid(257) == 1 (257
 * is prime) while rj_topo_rdp(257) == NULL (257^2-1 expanded disks > 256).
 * The two levels disagree by design; do not "align" them. */
int xorcode_rdp_valid(int p);              /* p prime, p >= 3 */
int xorcode_evenodd_valid(int p);          /* p prime, p >= 3 */
int xorcode_liberation_valid(int k, int w);/* w prime, w >= 3, 2 <= k <= w */

/* Smallest prime >= n (n >= 2). Used to pick Liberation's w. */
int xorcode_next_prime(int n);

#ifdef __cplusplus
}
#endif
