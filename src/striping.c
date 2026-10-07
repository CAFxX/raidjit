/* Cache-blocked striping: plan-time tile-size selection.
 *
 * rj_tile_select is a pure function of nstreams: the largest power of
 * two with nstreams*tile <= LLC/2, clamped to [TILE_MIN, TILE_MAX].
 * Computed once per compile; the drivers below (#38, #8, #10) only
 * read the stored value, so the tiling decision costs nothing at
 * runtime.
 */
#include "striping.h"

size_t rj_tile_select(size_t nstreams)
{
    if (nstreams == 0)
        nstreams = 1;
    size_t budget = RJ_STRIPING_LLC_BYTES / 2;
    size_t per_stream = budget / nstreams; /* nstreams >= 1: no div-by-0 */

    /* Largest power of two <= per_stream. per_stream >= 1 always
     * (budget >> nstreams for any sane nstreams), so the loop
     * terminates with tile >= 1. */
    size_t tile = (size_t)1 << 62;
    while (tile > per_stream)
        tile >>= 1;

    if (tile < RJ_STRIPING_TILE_MIN)
        tile = RJ_STRIPING_TILE_MIN;
    if (tile > RJ_STRIPING_TILE_MAX)
        tile = RJ_STRIPING_TILE_MAX;
    return tile;
}
