/* Cache-blocked striping: plan-time tile-size selection (queue items
 * #38/#8/#10).
 *
 * The fused drivers (#38 encode+CRC, #8 generalized fusion) process the
 * stripe in tiles: per tile they run each sub-kernel once, so every
 * byte is loaded from DRAM once and the follow-on passes (CRC over the
 * just-written parities, the second fused op) hit hot cache. The tile
 * size is chosen once at compile time from a static cache heuristic --
 * zero runtime cost for the decision (Carlo's invariant: tiling/fusion
 * decisions at plan time).
 */
#pragma once
#include <stddef.h>

/* Static LLC model for the tile heuristic (est., NOT measured): the
 * heuristic assumes a modest last-level cache and budgets half of it
 * for one tile's working set across all concurrent streams, leaving
 * headroom for kernel traffic the model does not account for. */
#define RJ_STRIPING_LLC_BYTES ((size_t)16 << 20)

/* Tile-size bounds: the floor amortizes per-tile call/dispatch overhead
 * and keeps the kernel's vector main loop dominant; the cap keeps the
 * scheduling quantum sane for very few streams. Both are powers of
 * two, so every tile offset preserves the caller's 64-byte buffer
 * alignment. */
#define RJ_STRIPING_TILE_MIN ((size_t)4096)
#define RJ_STRIPING_TILE_MAX ((size_t)1 << 20)

/* Plan-time tile selection. nstreams = number of concurrently live
 * buffer streams in one tile (e.g. ndata+npar for encode+CRC). Returns
 * a power-of-two tile length in bytes within
 * [RJ_STRIPING_TILE_MIN, RJ_STRIPING_TILE_MAX]: the largest power of
 * two with nstreams*tile <= RJ_STRIPING_LLC_BYTES/2, clamped into the
 * bounds. nstreams == 0 is treated as 1. Pure function of its inputs:
 * no CPUID, no measurement, no per-call work. */
size_t rj_tile_select(size_t nstreams);
