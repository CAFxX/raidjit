/* x86-64 CRC Phase 2 (interleaved fold) finalize.
 *
 * Private header. See src/x86_64/crc_fold.c for the design note on the
 * corrected single-128-bit-stream fold (the 4-lane every-4th-chunk
 * structure from the original spec is mathematically impossible).
 */
#pragma once
#include <stddef.h>
#include <stdint.h>

/* Finalize nstreams CRC-32s from the fold state.
 * fold_state: nstreams * 16 bytes, one little-endian 128-bit stream
 *   state per stream (NOT 64 bytes/stream; the 4-lane layout is gone).
 * total_len: total bytes processed per stream (unused; the
 *   single-stream state fully determines the CRC).
 * crc: output array of nstreams uint32_t (standard zlib CRC-32).
 *
 * Calling convention (documented for the fused.c driver): the crc_fold
 * kernel takes 4 args (in, out, len, fold_state) per System V ABI:
 * RDI=in, RSI=out, RDX=len, RCX=fold_state. */
void rj_crc_fold_finalize_x86(const void *fold_state, int nstreams,
                              size_t total_len, uint32_t *crc);
