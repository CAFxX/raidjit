/* XOR array codes for RAID-6: RDP, EVENODD, Liberation.
 *
 * Each builder fills a BINARY (0/1) expanded parity matrix that
 * plan_build() lowers to the existing P_XOR path automatically.
 *
 * Expanded model: k data disks x R rows. Column (d,r) = row r of data
 * disk d, i.e. col = d*R + r. Output rows 0..R-1 are the P parity rows,
 * rows R..2R-1 the Q parity rows. All arithmetic is XOR.
 */
#include "xorcode.h"

#include <string.h>

static int is_prime(int n)
{
    if (n < 2)
        return 0;
    if (n == 2)
        return 1;
    if ((n & 1) == 0)
        return 0;
    for (int d = 3; (long)d * d <= n; d += 2)
        if (n % d == 0)
            return 0;
    return 1;
}

int xorcode_next_prime(int n)
{
    if (n <= 2)
        return 2;
    if ((n & 1) == 0)
        n++;
    while (!is_prime(n))
        n += 2;
    return n;
}

int xorcode_rdp_valid(int p)
{
    return p >= 3 && is_prime(p);
}

int xorcode_evenodd_valid(int p)
{
    return p >= 3 && is_prime(p);
}

int xorcode_liberation_valid(int k, int w)
{
    return w >= 3 && is_prime(w) && k >= 2 && k <= w;
}

/* RDP (Corbett et al., FAST '04, section 4): p prime > 2, k = p-1 data
 * disks, R = p-1 rows (imaginary row p-1 = 0).
 *   P_j = XOR_{i=0}^{p-2} d[i][j]
 * Diagonal parity set s = { (i,k) : (i+k) mod p = s }; the diagonal disk
 * stores sets s = 0..p-2 (diagonal p-1 is the missing diagonal). Each
 * diagonal parity block is the even parity of the data AND row-parity
 * blocks on its diagonal, so with the row-parity block of diagonal s at
 * row (s-(p-1)) mod p = (s+1) mod p substituted by its definition:
 *   Q_s = XOR_{i=0}^{p-2} d[i][(s-i) mod p]
 *       ^ XOR_{i=0}^{p-2} d[i][(s+1) mod p]
 * with imaginary-row (p-1) terms dropped. The two diagonal terms never
 * hit the same cell ((s-i) = (s+1) mod p would need i = p-1). */
int xorcode_rdp_matrix(int p, uint8_t *m)
{
    if (!xorcode_rdp_valid(p) || !m)
        return -1;
    int R = p - 1, k = p - 1;
    /* size_t dims: k*R overflows int for large-but-valid parameters
     * (e.g. p > 46341); the builders take caller-allocated buffers
     * and must not UB before touching them. */
    size_t nd = (size_t)k * (size_t)R, np = 2u * (size_t)R;
    memset(m, 0, (size_t)np * (size_t)nd);
    for (int j = 0; j < R; j++)
        for (int i = 0; i < k; i++)
            m[(size_t)j * (size_t)nd + (size_t)i * (size_t)R + (size_t)j] =
                1;
    for (int s = 0; s < R; s++) {
        uint8_t *row = m + (size_t)(R + s) * (size_t)nd;
        for (int i = 0; i < k; i++) {
            int r1 = (s - i) % p;
            if (r1 < 0)
                r1 += p;
            if (r1 < R)
                row[(size_t)i * (size_t)R + (size_t)r1] ^= 1;
            int r2 = (s + 1) % p;
            if (r2 < R)
                row[(size_t)i * (size_t)R + (size_t)r2] ^= 1;
        }
    }
    return 0;
}

/* EVENODD (Blaum et al., IEEE Trans. Computers 1995): p prime, k = p data
 * disks, R = p-1 rows (imaginary row p-1 = 0).
 *   P_j = XOR_{i=0}^{p-1} d[i][j]
 *   S   = XOR_{i=1}^{p-1} d[i][(p-1-i) mod p]   (missing diagonal)
 *   Q_j = S ^ XOR_{i=0}^{p-1} d[i][(j-i) mod p]
 * with imaginary-row terms dropped. */
int xorcode_evenodd_matrix(int p, uint8_t *m)
{
    if (!xorcode_evenodd_valid(p) || !m)
        return -1;
    int R = p - 1, k = p;
    /* size_t dims: k*R overflows int for large-but-valid parameters
     * (e.g. p > 46341); the builders take caller-allocated buffers
     * and must not UB before touching them. */
    size_t nd = (size_t)k * (size_t)R, np = 2u * (size_t)R;
    memset(m, 0, (size_t)np * (size_t)nd);
    for (int j = 0; j < R; j++)
        for (int i = 0; i < k; i++)
            m[(size_t)j * (size_t)nd + (size_t)i * (size_t)R + (size_t)j] =
                1;
    for (int j = 0; j < R; j++) {
        uint8_t *row = m + (size_t)(R + j) * (size_t)nd;
        for (int i = 1; i < k; i++) {
            int rs = (p - 1 - i) % p;
            if (rs < 0)
                rs += p;
            /* i in [1,p-1] -> rs in [0,p-2]: always a real row */
            row[(size_t)i * (size_t)R + (size_t)rs] ^= 1;
        }
        for (int i = 0; i < k; i++) {
            int r = (j - i) % p;
            if (r < 0)
                r += p;
            if (r < R)
                row[(size_t)i * (size_t)R + (size_t)r] ^= 1;
        }
    }
    return 0;
}

/* Liberation (Plank, FAST '08, section 3.3): w prime > 2, w >= k,
 * k data disks, R = w rows.
 *   P_j = XOR_{i=0}^{k-1} d[i][j]
 *   Q_r = XOR_{i=0}^{k-1} (X_i row r . d[i]), with X_0 = I_w and
 *         X_i = I_w^->i + O_w[y, y+i-1], y = i*(w-1)/2 mod w.
 * I_w^->i (columns rotated right by i) has its row-r 1 at column
 * (r+i) mod w; O_w[y,y+i-1] adds a 1 at (row y, col (y+i-1) mod w) for
 * i >= 1. The two 1s never coincide (cols (y+i) vs (y+i-1) mod w). */
int xorcode_liberation_matrix(int k, int w, uint8_t *m)
{
    if (!xorcode_liberation_valid(k, w) || !m)
        return -1;
    int R = w;
    /* size_t dims: k*R overflows int for large-but-valid parameters
     * (e.g. p > 46341); the builders take caller-allocated buffers
     * and must not UB before touching them. */
    size_t nd = (size_t)k * (size_t)R, np = 2u * (size_t)R;
    memset(m, 0, (size_t)np * (size_t)nd);
    for (int j = 0; j < R; j++)
        for (int i = 0; i < k; i++)
            m[(size_t)j * (size_t)nd + (size_t)i * (size_t)R + (size_t)j] =
                1;
    for (int r = 0; r < R; r++) {
        uint8_t *row = m + (size_t)(R + r) * (size_t)nd;
        for (int i = 0; i < k; i++) {
            row[(size_t)i * (size_t)R + (size_t)((r + i) % w)] ^= 1;
            if (i > 0) {
                int y = (i * (w - 1) / 2) % w;
                if (r == y)
                    row[(size_t)i * (size_t)R +
                        (size_t)((y + i - 1) % w)] ^= 1;
            }
        }
    }
    return 0;
}

/* Benchmark-selected default for rj_topo_raid6(ndata).
 *
 * Source of truth: docs/xor_code_default.md (machine-readable table of
 * measured winners per ndata). This array is transcribed from it;
 * tests/test_xorcode.c parses that file and checks agreement, so the two
 * cannot drift. The integrator RE-RUNS the code-choice benchmark on the
 * merged tree (with track A3's XOR-schedule/CSE, which is expected to
 * make the XOR codes more competitive) and may flip entries.
 *
 * Measured 2026-10-02 (AMD EPYC 9D25): RS fastest at every ndata 3..51;
 * see docs/xor_codes.md §4 for the numbers. */
xorcode_choice_t xorcode_default_for_ndata(int ndata)
{
    /* ndata 3..51 -> choice; all RS per the 2026-10-02 measurement. */
    static const xorcode_choice_t table[52] = {
        [3] = XORCODE_RS, [4] = XORCODE_RS, [5] = XORCODE_RS,
        [6] = XORCODE_RS, [7] = XORCODE_RS, [8] = XORCODE_RS,
        [9] = XORCODE_RS, [10] = XORCODE_RS, [11] = XORCODE_RS,
        [12] = XORCODE_RS, [13] = XORCODE_RS, [14] = XORCODE_RS,
        [15] = XORCODE_RS, [16] = XORCODE_RS, [17] = XORCODE_RS,
        [18] = XORCODE_RS, [19] = XORCODE_RS, [20] = XORCODE_RS,
        [21] = XORCODE_RS, [22] = XORCODE_RS, [23] = XORCODE_RS,
        [24] = XORCODE_RS, [25] = XORCODE_RS, [26] = XORCODE_RS,
        [27] = XORCODE_RS, [28] = XORCODE_RS, [29] = XORCODE_RS,
        [30] = XORCODE_RS, [31] = XORCODE_RS, [32] = XORCODE_RS,
        [33] = XORCODE_RS, [34] = XORCODE_RS, [35] = XORCODE_RS,
        [36] = XORCODE_RS, [37] = XORCODE_RS, [38] = XORCODE_RS,
        [39] = XORCODE_RS, [40] = XORCODE_RS, [41] = XORCODE_RS,
        [42] = XORCODE_RS, [43] = XORCODE_RS, [44] = XORCODE_RS,
        [45] = XORCODE_RS, [46] = XORCODE_RS, [47] = XORCODE_RS,
        [48] = XORCODE_RS, [49] = XORCODE_RS, [50] = XORCODE_RS,
        [51] = XORCODE_RS,
    };
    if (ndata < 3 || ndata > 51)
        return XORCODE_RS;
    return table[ndata];
}
