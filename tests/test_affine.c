/* Validate gf_affine_matrix layout against real VGF2P8AFFINEQB. */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include "../src/x86_64/emit.h"
#include "gf.h"

typedef void (*aff_fn)(const uint8_t *, const uint64_t *, uint8_t *);

int main(void)
{
    /* emit: load matrix qword broadcast manually via loop is overkill;
       single instruction test: vgf2p8affineqb(zmm2=dst, zmm1=matrix, zmm0=data, 0) */
    uint8_t code[64];
    em_t e;
    em_init(&e, code, sizeof code);
    /* we will fill registers manually in asm below instead; just get bytes */
    em_vgf2p8affineqb(&e, 2, 0, 1, 0, VL64);
    printf("affine bytes:");
    for (size_t i = 0; i < e.len; i++)
        printf(" %02x", code[i]);
    printf("\n");

    /* build executable thunk: (rdi=data_in, rsi=matrix_broadcast, rdx=data_out)
       zmm0 = load [rdi], zmm1 = load [rsi], zmm2 = affine, store [rdx], ret */
    uint8_t *buf = mmap(NULL, 4096, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    em_t t;
    em_init(&t, buf, 4096);
    em_vmovdqa_ld(&t, 0, 7, -1, 0, 0, VL64);   /* zmm0 = [rdi] */
    em_vmovdqa_ld(&t, 1, 6, -1, 0, 0, VL64);   /* zmm1 = [rsi] */
    em_vgf2p8affineqb(&t, 2, 0, 1, 0, VL64);   /* zmm2 = affine(data=zmm0, matrix=zmm1) */
    em_vmovdqa_st(&t, 2, -1, 0, 0, 2, VL64);   /* [rdx] = zmm2 */
    em_ret(&t);
    mprotect(buf, 4096, PROT_READ | PROT_EXEC);

    aff_fn fn = (aff_fn)buf;
    int bad = 0, bad_t = 0;
    /* aligned loads/stores now: the thunk uses VMOVDQA32 */
    _Alignas(64) uint64_t mb[8];
    _Alignas(64) uint8_t in[64], out[64];
    for (unsigned c = 1; c < 256; c++) {
        uint64_t m = gf_affine_matrix((uint8_t)c);
        for (int i = 0; i < 8; i++)
            mb[i] = m;
        for (int i = 0; i < 64; i++)
            in[i] = (uint8_t)(i * 37 + c * 11 + 3);
        fn(in, mb, out);
        for (int i = 0; i < 64; i++) {
            uint8_t want = gf_mul((uint8_t)c, in[i]);
            if (out[i] != want && !bad) {
                printf("MISMATCH c=%u i=%d: got %02x want %02x (in=%02x)\n",
                       c, i, out[i], want, in[i]);
                bad = 1;
            }
        }
        /* transpose variant: byte k = c * 2^k */
        uint64_t mt = 0;
        for (int k = 0; k < 8; k++)
            mt |= (uint64_t)gf_mul((uint8_t)c, (uint8_t)(1u << k)) << (8 * k);
        for (int i = 0; i < 8; i++)
            mb[i] = mt;
        fn(in, mb, out);
        for (int i = 0; i < 64; i++) {
            uint8_t want = gf_mul((uint8_t)c, in[i]);
            if (out[i] != want && !bad_t) {
                printf("TRANSPOSE MISMATCH c=%u i=%d\n", c, i);
                bad_t = 1;
            }
        }
    }
    printf("current layout: %s, transpose layout: %s\n",
           bad ? "WRONG" : "OK", bad_t ? "WRONG" : "OK");
    if (bad) {
        printf("affine: FAILED\n");
        return 1;
    }
    printf("affine: all OK\n");
    return 0;
}
