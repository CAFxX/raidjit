/* CRC-32 for the integrated-checksum encode path (queue item #38).
 *
 * CRC-32/ISO-HDLC (the "PKZIP" CRC): polynomial 0x04C11DB7 reflected
 * (0xEDB88320), init 0xFFFFFFFF, refin/refout true, xorout 0xFFFFFFFF.
 * Check value: CRC32("123456789") = 0xCBF43926 (external ground truth;
 * tests/test_crc.c asserts it).
 *
 * Two implementations share one generated table:
 *   rj_crc32_update        - slice-by-8 fast path (the production path
 *                            used by the fused drivers).
 *   rj_crc32_update_scalar - byte-at-a-time reference (the differential
 *                            oracle; tests prove the fast path against
 *                            it and the KAT anchors both to the catalog
 *                            check value).
 */
#pragma once
#include <stddef.h>
#include <stdint.h>

void rj_crc32_init(uint32_t *crc); /* *crc = 0xFFFFFFFF */
uint32_t rj_crc32_update(uint32_t crc, const void *data, size_t len);
uint32_t rj_crc32_update_scalar(uint32_t crc, const void *data, size_t len);
uint32_t rj_crc32_final(uint32_t crc); /* ^ 0xFFFFFFFF */
uint32_t rj_crc32(const void *data, size_t len); /* one-shot */
