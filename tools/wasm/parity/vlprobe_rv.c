#include <stdio.h>
int main(void){ size_t v; __asm__ volatile("csrr %0, vlenb":"=r"(v)); printf("vlenb=%zu\n", v); return 0; }
