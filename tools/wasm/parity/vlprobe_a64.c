#include <stdio.h>
int main(void){ size_t v; __asm__ volatile(".arch armv8-a+sve\n\tcntb %0":"=r"(v)); printf("cntb=%zu\n", v); return 0; }
