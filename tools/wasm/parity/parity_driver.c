/* parity_driver.c — native counterpart of the WASM rjw_compile API.
 *
 * Links rjw_api.c WITHOUT -DRJ_WASM (native mmap path) per ARCH, with the
 * matching -DRJW_* module selector. Same JSON in -> native bytes out.
 * Used by run_parity.sh to prove byte-identity between the WASM modules
 * and the native build (SPEC §S9.2).
 *
 * Usage: parity_driver <options.json> <out-prefix>
 *   stdout: the rjw_compile result JSON (one line)
 *   <out-prefix>.code.bin / <out-prefix>.const.bin: the blobs
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

char *rjw_compile(const char *json);
const uint8_t *rjw_code_bytes(size_t *len_out);
const uint8_t *rjw_const_bytes(size_t *len_out);
void rjw_free_string(char *s);

static char *read_file(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "parity_driver: cannot open %s\n", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)n + 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) {
        fclose(f);
        free(buf);
        return NULL;
    }
    buf[n] = '\0';
    fclose(f);
    return buf;
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s <options.json> <out-prefix>\n", argv[0]);
        return 2;
    }
    char *json = read_file(argv[1]);
    if (!json)
        return 2;
    char *res = rjw_compile(json);
    free(json);
    printf("%s\n", res);

    /* Blobs only on success; parse "ok" cheaply. */
    int ok = strstr(res, "\"ok\":true") != NULL;
    rjw_free_string(res);
    if (!ok)
        return 1;

    size_t clen, dlen;
    const uint8_t *code = rjw_code_bytes(&clen);
    const uint8_t *cnst = rjw_const_bytes(&dlen);
    char path[1024];
    snprintf(path, sizeof(path), "%s.code.bin", argv[2]);
    FILE *f = fopen(path, "wb");
    if (!f)
        return 2;
    int rc = 0;
    if (fwrite(code, 1, clen, f) != clen)
        rc = 2;
    fclose(f);
    snprintf(path, sizeof(path), "%s.const.bin", argv[2]);
    f = fopen(path, "wb");
    if (!f)
        return 2;
    if (fwrite(cnst, 1, dlen, f) != dlen)
        rc = 2;
    fclose(f);
    return rc;
}
