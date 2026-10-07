/* CRC-32/ISO-HDLC: scalar reference + slice-by-8 fast path.
 *
 * The 8x256 tables are generated once from the reflected polynomial
 * 0xEDB88320 (no magic constants baked into the source); pthread_once
 * guards the generation for thread safety, matching the library's
 * existing pthread use (src/api_batched.c).
 */
#include "crc.h"

#include <pthread.h>
#include <stdint.h>

#define CRC_POLY_REFLECTED 0xEDB88320u
#define CRC_INIT 0xFFFFFFFFu

static uint32_t crc_tab[8][256];
static pthread_once_t crc_once = PTHREAD_ONCE_INIT;

static void crc_gen_tables(void)
{
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int k = 0; k < 8; k++)
            c = (c & 1) ? (c >> 1) ^ CRC_POLY_REFLECTED : c >> 1;
        crc_tab[0][i] = c;
    }
    for (int s = 1; s < 8; s++)
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = crc_tab[s - 1][i];
            crc_tab[s][i] = (c >> 8) ^ crc_tab[0][c & 0xFF];
        }
}

static void crc_tables_ensure(void)
{
    pthread_once(&crc_once, crc_gen_tables);
}

void rj_crc32_init(uint32_t *crc)
{
    *crc = CRC_INIT;
}

uint32_t rj_crc32_final(uint32_t crc)
{
    return crc ^ CRC_INIT;
}

/* Byte-at-a-time scalar reference: the differential oracle. */
uint32_t rj_crc32_update_scalar(uint32_t crc, const void *data, size_t len)
{
    crc_tables_ensure();
    const uint8_t *p = (const uint8_t *)data;
    for (size_t i = 0; i < len; i++)
        crc = (crc >> 8) ^ crc_tab[0][(crc ^ p[i]) & 0xFF];
    return crc;
}

/* Slice-by-8 fast path. Unaligned head/tail go through the scalar
 * step; the 8-byte body uses explicit little-endian loads (portable,
 * folded into single unaligned loads on LE targets). */
uint32_t rj_crc32_update(uint32_t crc, const void *data, size_t len)
{
    crc_tables_ensure();
    const uint8_t *p = (const uint8_t *)data;

    while (len && ((uintptr_t)p & 7)) {
        crc = (crc >> 8) ^ crc_tab[0][(crc ^ *p++) & 0xFF];
        len--;
    }
    while (len >= 8) {
        /* Explicit little-endian loads: portable, and compilers fold
         * them into single unaligned loads on LE targets. */
        uint32_t lo = (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
                      ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
        uint32_t hi = (uint32_t)p[4] | ((uint32_t)p[5] << 8) |
                      ((uint32_t)p[6] << 16) | ((uint32_t)p[7] << 24);
        lo ^= crc;
        crc = crc_tab[7][lo & 0xFF] ^ crc_tab[6][(lo >> 8) & 0xFF] ^
              crc_tab[5][(lo >> 16) & 0xFF] ^ crc_tab[4][(lo >> 24) & 0xFF] ^
              crc_tab[3][hi & 0xFF] ^ crc_tab[2][(hi >> 8) & 0xFF] ^
              crc_tab[1][(hi >> 16) & 0xFF] ^ crc_tab[0][(hi >> 24) & 0xFF];
        p += 8;
        len -= 8;
    }
    while (len--) {
        crc = (crc >> 8) ^ crc_tab[0][(crc ^ *p++) & 0xFF];
    }
    return crc;
}

uint32_t rj_crc32(const void *data, size_t len)
{
    uint32_t c;
    rj_crc32_init(&c);
    c = rj_crc32_update(c, data, len);
    return rj_crc32_final(c);
}
