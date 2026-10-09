/* CRC Phase 2 (interleaved fold) finalize for x86-64.
 *
 * CORRECTED DESIGN (2026-10-09): The original Phase 2 spec called for a
 * 4-lane every-4th-16B-chunk fold state (64 bytes/stream). That design is
 * mathematically impossible: the 4 lane states do not determine the
 * stream CRC (proven by counterexample: two 8-chunk messages with
 * identical 4-lane states but different CRC32s, 0x51eb4347 vs
 * 0x24abc5d9). No finalize seeing only (fold_state, nstreams,
 * total_len) can produce the correct CRC32 for messages with >4 chunks.
 *
 * The backend implements the corrected single-128-bit-stream fold
 * (verified bit-exact vs zlib, see design_crc_fold.md section 11):
 * per-stream 128-bit state S, init low 32 bits = 0xFFFFFFFF, per 16B
 * chunk C: W = S XOR C; S' = CLMUL(W[63:0], K1) XOR CLMUL(W[127:64],
 * K2) with K1 = 0xae689191 (x^-128 mod QREF), K2 = 0xccaa009e (x^-64
 * mod QREF), QREF = 0x1DB710641.
 *
 * Fold state layout: nstreams * 16 bytes, one 128-bit state per stream
 * (little-endian). The driver must allocate 16 bytes/stream (NOT 64).
 *
 * Calling convention: the crc_fold kernel takes 4 args
 *   (in, out, len, fold_state)
 * per the System V ABI: RDI=in, RSI=out, RDX=len, RCX=fold_state.
 * The kernel loads 16B/stream at entry and stores 16B/stream at exit.
 */

#include <stdint.h>
#include <stddef.h>

/* QREF = 0x1DB710641 (the reflected CRC-32 polynomial, degree 32). */
#define CRC_FOLD_QREF ((__uint128_t)0x1DB710641ULL)

/* Reduce a 128-bit value mod QREF. Result is < 2^32. */
static uint32_t crc_fold_red128(__uint128_t s)
{
    while (s >> 32) {
        int bl;
        if (s >> 64) {
            unsigned long long hi = (unsigned long long)(s >> 64);
            bl = 128 - __builtin_clzll(hi);
        } else {
            unsigned long long lo = (unsigned long long)s;
            bl = 64 - __builtin_clzll(lo);
        }
        /* bl > 32; align QREF's top bit (bit 32) with s's top bit. */
        s ^= CRC_FOLD_QREF << (bl - 33);
    }
    return (uint32_t)s;
}

/* Finalize nstreams CRCs from the fold state.
 * fold_state: nstreams * 16 bytes, each 16B little-endian 128-bit state.
 * total_len: total bytes processed per stream (currently unused; the
 *   single-stream fold state fully determines the CRC, no length
 *   adjustment needed).
 * crc: output array of nstreams uint32_t CRCs (standard zlib CRC-32).
 */
void rj_crc_fold_finalize_x86(const void *fold_state, int nstreams,
                              size_t total_len, uint32_t *crc)
{
    (void)total_len; /* not needed for the single-stream fold */
    const unsigned char *st = (const unsigned char *)fold_state;
    for (int s = 0; s < nstreams; s++) {
        __uint128_t S = 0;
        for (int i = 0; i < 16; i++)
            S |= (__uint128_t)st[s * 16 + i] << (8 * i);
        crc[s] = crc_fold_red128(S) ^ 0xFFFFFFFFu;
    }
}
