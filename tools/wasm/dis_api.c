/* dis_api.c — Capstone wrapper for the live kernel explorer.
 *
 * dj_disasm(arch, bytes, len, base_addr) disassembles one kernel blob and
 * returns a malloc'd JSON array string:
 *   [{"addr":1234,"size":4,"bytes":"0f1f00","mnemonic":"nop","op_str":""}, ...]
 * Bytes that do not decode are emitted as {mnemonic:".byte", op_str:"0x.."}
 * so the differential test (SPEC §S9.4) can spot unknowns explicitly.
 *
 * arch: "x86_64" | "aarch64" | "riscv64". x86 uses AT&T syntax to match
 * the repo's objdump reference; AArch64/RISC-V use Capstone's default.
 * RISC-V opens with CS_MODE_RISCV64|CS_MODE_RISCV_V (RVV 1.0 + Zvbc).
 */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <capstone/capstone.h>
#include "vendor/cJSON.h"

/* Capstone 6 requires explicit memory-function setup before cs_open()
 * (CS_OPT_MEM with a NULL handle); otherwise cs_open fails with
 * CS_ERR_MEMSETUP. Idempotent: safe to call per dj_disasm invocation. */
static void capstone_mem_setup(void)
{
    static int done = 0;
    if (done)
        return;
    cs_opt_mem mem;
    mem.malloc = malloc;
    mem.calloc = calloc;
    mem.realloc = realloc;
    mem.free = free;
    mem.vsnprintf = vsnprintf;
    cs_option(0, CS_OPT_MEM, (uintptr_t)&mem);
    done = 1;
}

static void bytes_hex(const uint8_t *b, size_t n, char *out)
{
    static const char *hexd = "0123456789abcdef";
    for (size_t i = 0; i < n; i++) {
        out[2 * i] = hexd[b[i] >> 4];
        out[2 * i + 1] = hexd[b[i] & 15];
    }
    out[2 * n] = '\0';
}

char *dj_disasm(const char *arch, const uint8_t *bytes, size_t len,
                uint64_t base_addr)
{
    cs_arch csarch;
    cs_mode csmode;
    if (!strcmp(arch, "x86_64")) {
        csarch = CS_ARCH_X86;
        csmode = CS_MODE_64;
    } else if (!strcmp(arch, "aarch64")) {
        csarch = CS_ARCH_AARCH64;
        csmode = CS_MODE_ARM;
    } else if (!strcmp(arch, "riscv64")) {
        csarch = CS_ARCH_RISCV;
        csmode = (cs_mode)(CS_MODE_RISCV64 | CS_MODE_RISCV_V);
    } else {
        return NULL;
    }

    capstone_mem_setup();
    csh handle;
    if (cs_open(csarch, csmode, &handle) != CS_ERR_OK)
        return NULL;
    if (csarch == CS_ARCH_X86)
        cs_option(handle, CS_OPT_SYNTAX, CS_OPT_SYNTAX_ATT);

    cJSON *arr = cJSON_CreateArray();
    const uint8_t *code = bytes;
    size_t size = len;
    uint64_t addr = base_addr;
    cs_insn insn;
    /* Cap: 1M instructions; kernels are small, this only bounds
     * hostile input. */
    for (size_t n = 0; n < 1048576 && size > 0; n++) {
        const uint8_t *cur = code;
        uint64_t cur_addr = addr;
        if (cs_disasm_iter(handle, &code, &size, &addr, &insn)) {
            char hb[64];
            bytes_hex(cur, insn.size, hb);
            cJSON *e = cJSON_CreateObject();
            cJSON_AddNumberToObject(e, "addr", (double)cur_addr);
            cJSON_AddNumberToObject(e, "size", (double)insn.size);
            cJSON_AddStringToObject(e, "bytes", hb);
            cJSON_AddStringToObject(e, "mnemonic", insn.mnemonic);
            cJSON_AddStringToObject(e, "op_str", insn.op_str);
            cJSON_AddItemToArray(arr, e);
        } else {
            /* Unknown byte: emit explicitly and advance one. */
            char hb[3];
            bytes_hex(cur, 1, hb);
            char op[8];
            snprintf(op, sizeof(op), "0x%s", hb);
            cJSON *e = cJSON_CreateObject();
            cJSON_AddNumberToObject(e, "addr", (double)cur_addr);
            cJSON_AddNumberToObject(e, "size", (double)1);
            cJSON_AddStringToObject(e, "bytes", hb);
            cJSON_AddStringToObject(e, "mnemonic", ".byte");
            cJSON_AddStringToObject(e, "op_str", op);
            cJSON_AddItemToArray(arr, e);
            code = cur + 1;
            size -= 1;
            addr = cur_addr + 1;
        }
    }
    cs_close(&handle);
    char *out = cJSON_PrintUnformatted(arr);
    cJSON_Delete(arr);
    return out;
}

void dj_free_string(char *s)
{
    free(s);
}
