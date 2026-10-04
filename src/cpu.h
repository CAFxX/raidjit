#pragma once
#include "../include/raidjit.h"

/* Fills rj_cpu_info. Never claims an extension the OS hasn't enabled
 * (checks XCR0 via XGETBV for AVX/AVX512 state). */
void rj_cpu_detect(rj_cpu_info *out);
