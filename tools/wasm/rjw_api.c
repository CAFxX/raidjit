/* rjw_api.c — compile-only JSON API for the raidjit WASM modules.
 *
 * One instance of this file is compiled into each per-ISA module
 * (tools/wasm/SPEC.md §S3-S4), selected by exactly one of -DRJW_X86 /
 * -DRJW_AARCH64 / -DRJW_RISCV.
 *
 * Contract:
 *  - rjw_compile(json) parses ONE option document (schema "v":1, see
 *    tools/wasm/README.md), runs plan + candidate selection + the
 *    emitter, and returns a malloc'd JSON result string. It NEVER
 *    executes generated code, NEVER benchmarks, NEVER probes hardware:
 *    o.benchmark is forced 0, validate_versions/stage-2 are skipped, and
 *    rj_cpu_detect is the synthetic-bundle stub (cpu_wasm_*.c).
 *  - Deterministic: identical JSON in -> identical bytes + metadata out.
 *  - Invalid option combinations fail CLOSED with {ok:false, err}:
 *    planid_cand_valid() plus explicit CPU-feature gates run before
 *    jit_build_version, so no backend assert is reachable from a pinned
 *    candidate (an assert would abort the module).
 *  - "exec_validated":false is always reported: skipping
 *    validate_versions is the one deliberate validation gap; the
 *    byte-identity parity matrix (tools/wasm/parity/) against the
 *    CI-executed native build is the correctness argument.
 *
 * The returned string and the code/const blobs are owned by the module;
 * rjw_free_string() releases a result string, and blobs stay valid
 * until the next rjw_compile call.
 */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>

#include "raidjit.h"
#include "plan.h"
#include "jit.h"
#include "bench.h"
#include "planid.h"
#include "cpu.h"
#include "rjw_wasm.h"
#include "vendor/cJSON.h"

#if !defined(RJW_X86) && !defined(RJW_AARCH64) && !defined(RJW_RISCV)
#error "rjw_api.c needs exactly one of -DRJW_X86/-DRJW_AARCH64/-DRJW_RISCV"
#endif
#if (defined(RJW_X86) + defined(RJW_AARCH64) + defined(RJW_RISCV)) != 1
#error "rjw_api.c needs exactly one of -DRJW_X86/-DRJW_AARCH64/-DRJW_RISCV"
#endif

/* ---------------- synthetic CPU bundles (SPEC §S5.1) ---------------- */

static rj_cpu_info g_cpu; /* installed per rjw_compile call */

const rj_cpu_info *rjw_cpu_current(void)
{
    return &g_cpu;
}

/* Fill *cpu from a preset name. Returns 0 ok, -1 unknown preset or a
 * preset for a different ISA module. */
static int preset_fill(const char *name, rj_cpu_info *cpu)
{
    memset(cpu, 0, sizeof(*cpu));
#if defined(RJW_X86)
    if (!strcmp(name, "avx512-gfni")) {
        cpu->sse2 = cpu->ssse3 = 1;
        cpu->avx = cpu->avx2 = 1;
        cpu->avx512f = cpu->avx512bw = cpu->avx512vl = 1;
        cpu->gfni = 1;
        cpu->vbmi = 1;
    } else if (!strcmp(name, "avx2")) {
        cpu->sse2 = cpu->ssse3 = 1;
        cpu->avx = cpu->avx2 = 1;
    } else if (!strcmp(name, "sse")) {
        cpu->sse2 = cpu->ssse3 = 1;
    } else {
        return -1;
    }
#elif defined(RJW_AARCH64)
    if (!strcmp(name, "sve2")) {
        cpu->neon = 1;
        cpu->sve = 1;
        cpu->sve2 = 1;
        cpu->sve_vl = 16;
    } else if (!strcmp(name, "neon")) {
        cpu->neon = 1;
    } else {
        return -1;
    }
#elif defined(RJW_RISCV)
    if (!strcmp(name, "rvv")) {
        cpu->rvv = 1;
        cpu->rvv_vlen = 16;
    } else if (!strcmp(name, "rvv-zvbc32e")) {
        cpu->rvv = 1;
        cpu->rvv_vlen = 16;
        cpu->rvv_zvbc32e = 1;
    } else if (!strcmp(name, "rvv-zicbop")) {
        cpu->rvv = 1;
        cpu->rvv_vlen = 16;
        cpu->rvv_zicbop = 1;
    } else {
        return -1;
    }
#endif
    snprintf(cpu->brand, sizeof(cpu->brand), "synthetic-%s", name);
    return 0;
}

/* Expert overrides: a flat subset of rj_cpu_info fields. Only
 * codegen-gating flags are honored (SPEC §S5.1); anything else is
 * rejected so callers cannot build a nonsensical CPU. */
static int cpu_override_apply(cJSON *o, rj_cpu_info *cpu, char *err,
                              size_t errlen)
{
    static const char *const bool_fields[] = {
#if defined(RJW_X86)
        "gfni", "vbmi", "avx512bw",
#elif defined(RJW_AARCH64)
        "sve2",
#elif defined(RJW_RISCV)
        "rvv_zvbc32e", "rvv_zicbop",
#endif
        NULL
    };
    for (cJSON *it = o->child; it; it = it->next) {
        const char *k = it->string;
        int known = 0;
        for (int i = 0; bool_fields[i]; i++)
            if (!strcmp(k, bool_fields[i]))
                known = 1;
#if defined(RJW_AARCH64)
        if (!strcmp(k, "sve_vl"))
            known = 2;
#elif defined(RJW_RISCV)
        if (!strcmp(k, "rvv_vlen"))
            known = 2;
#endif
        if (!known) {
            snprintf(err, errlen, "unknown/inapplicable cpu override '%s'",
                     k);
            return -1;
        }
        if (!cJSON_IsNumber(it)) {
            snprintf(err, errlen, "cpu override '%s' must be a number", k);
            return -1;
        }
        int v = it->valueint;
        if (known == 1 && (v < 0 || v > 1)) {
            snprintf(err, errlen, "cpu override '%s' must be 0 or 1", k);
            return -1;
        }
        if (!strcmp(k, "gfni"))
            cpu->gfni = v;
        else if (!strcmp(k, "vbmi"))
            cpu->vbmi = v;
        else if (!strcmp(k, "avx512bw"))
            cpu->avx512bw = v;
        else if (!strcmp(k, "sve2"))
            cpu->sve2 = v;
        else if (!strcmp(k, "rvv_zvbc32e"))
            cpu->rvv_zvbc32e = v;
        else if (!strcmp(k, "rvv_zicbop"))
            cpu->rvv_zicbop = v;
#if defined(RJW_AARCH64)
        else if (!strcmp(k, "sve_vl")) {
            if (v < 16 || v > 256 || (v & (v - 1)) != 0) {
                snprintf(err, errlen,
                         "sve_vl must be a power of two in 16..256");
                return -1;
            }
            cpu->sve_vl = v;
        }
#elif defined(RJW_RISCV)
        else if (!strcmp(k, "rvv_vlen")) {
            if (v < 16 || v > 4096 || (v & (v - 1)) != 0) {
                snprintf(err, errlen,
                         "rvv_vlen must be a power of two in 16..4096");
                return -1;
            }
            cpu->rvv_vlen = v;
        }
#endif
    }
    /* Consistency: derived flags follow their bases. */
    if (cpu->gfni)
        cpu->avx512f = cpu->avx512bw = 1;
    if (cpu->vbmi)
        cpu->avx512f = cpu->avx512bw = 1;
    if (cpu->sve2)
        cpu->sve = 1;
    return 0;
}

/* ---------------- JSON helpers ---------------- */

static cJSON *req_obj(cJSON *parent, const char *key, char *err, size_t errlen)
{
    cJSON *o = cJSON_GetObjectItemCaseSensitive(parent, key);
    if (!cJSON_IsObject(o)) {
        snprintf(err, errlen, "missing/invalid object '%s'", key);
        return NULL;
    }
    return o;
}

static const char *req_str(cJSON *parent, const char *key, char *err,
                           size_t errlen)
{
    cJSON *o = cJSON_GetObjectItemCaseSensitive(parent, key);
    if (!cJSON_IsString(o)) {
        snprintf(err, errlen, "missing/invalid string '%s'", key);
        return NULL;
    }
    return o->valuestring;
}

static int opt_int(cJSON *parent, const char *key, int dflt, char *err,
                   size_t errlen)
{
    cJSON *o = cJSON_GetObjectItemCaseSensitive(parent, key);
    if (!o)
        return dflt;
    if (!cJSON_IsNumber(o)) {
        snprintf(err, errlen, "invalid number '%s'", key);
        return dflt;
    }
    err[0] = '\0';
    return o->valueint;
}

static int opt_bool(cJSON *parent, const char *key, int dflt, char *err,
                    size_t errlen)
{
    cJSON *o = cJSON_GetObjectItemCaseSensitive(parent, key);
    if (!o)
        return dflt;
    if (cJSON_IsTrue(o))
        return 1;
    if (cJSON_IsFalse(o))
        return 0;
    if (cJSON_IsNumber(o))
        return o->valueint != 0;
    snprintf(err, errlen, "invalid bool '%s'", key);
    return dflt;
}

/* ---------------- topology builders (SPEC §S5.2) ---------------- */

static rj_topology *build_topology(cJSON *tj, char *err, size_t errlen)
{
    const char *ctor = req_str(tj, "ctor", err, errlen);
    if (!ctor)
        return NULL;
    /* int params, validated per constructor below */
    int p1 = opt_int(tj, "p1", -1, err, errlen);
    int p2 = opt_int(tj, "p2", -1, err, errlen);
    int p3 = opt_int(tj, "p3", -1, err, errlen);
    if (err[0])
        return NULL;
    rj_topology *t = NULL;
    if (!strcmp(ctor, "raid0")) {
        if (p1 < 1)
            goto badp;
        t = rj_topo_raid0(p1);
    } else if (!strcmp(ctor, "raid1")) {
        t = rj_topo_raid1();
    } else if (!strcmp(ctor, "raid5")) {
        if (p1 < 1)
            goto badp;
        t = rj_topo_raid5(p1);
    } else if (!strcmp(ctor, "raid6")) {
        if (p1 < 1)
            goto badp;
        t = rj_topo_raid6(p1);
    } else if (!strcmp(ctor, "raid6_rs")) {
        if (p1 < 1)
            goto badp;
        t = rj_topo_raid6_rs(p1);
    } else if (!strcmp(ctor, "rdp")) {
        if (p1 < 3)
            goto badp;
        t = rj_topo_rdp(p1);
    } else if (!strcmp(ctor, "evenodd")) {
        if (p1 < 3)
            goto badp;
        t = rj_topo_evenodd(p1);
    } else if (!strcmp(ctor, "liberation")) {
        if (p1 < 2)
            goto badp;
        t = rj_topo_liberation(p1);
    } else if (!strcmp(ctor, "rs")) {
        if (p1 < 1 || p2 < 1)
            goto badp;
        t = rj_topo_rs(p1, p2);
    } else if (!strcmp(ctor, "lrc")) {
        if (p1 < 2 || p2 < 1 || p3 < 2)
            goto badp;
        t = rj_topo_lrc(p1, p2, p3);
    } else if (!strcmp(ctor, "lrc_azure")) {
        t = rj_topo_lrc_azure();
    } else if (!strcmp(ctor, "lrc_optimal")) {
        if (p1 < 2 || p2 < 1 || p3 < 2)
            goto badp;
        t = rj_topo_lrc_optimal(p1, p2, p3);
    } else if (!strcmp(ctor, "custom")) {
        const char *hex = req_str(tj, "matrix_hex", err, errlen);
        if (!hex)
            return NULL;
        if (p1 < 1 || p2 < 1)
            goto badp;
        size_t want = (size_t)p1 * (size_t)p2 * 2;
        if (strlen(hex) != want) {
            snprintf(err, errlen,
                     "matrix_hex length %zu != 2*ndata*npar (%zu)",
                     strlen(hex), want);
            return NULL;
        }
        uint8_t *mat = malloc((size_t)p1 * (size_t)p2);
        if (!mat) {
            snprintf(err, errlen, "out of memory");
            return NULL;
        }
        for (size_t i = 0; i < (size_t)p1 * (size_t)p2; i++) {
            unsigned v;
            if (sscanf(hex + 2 * i, "%2x", &v) != 1) {
                free(mat);
                snprintf(err, errlen, "matrix_hex: bad hex at byte %zu", i);
                return NULL;
            }
            mat[i] = (uint8_t)v;
        }
        t = rj_topo_custom(p1, p2, mat);
        free(mat);
    } else {
        snprintf(err, errlen, "unknown topology ctor '%s'", ctor);
        return NULL;
    }
    if (!t && !err[0])
        snprintf(err, errlen,
                 "topology ctor '%s' rejected its parameters (validity "
                 "envelope)", ctor);
    return t;
badp:
    snprintf(err, errlen, "topology ctor '%s': bad/missing params", ctor);
    return NULL;
}

/* ---------------- op builders (SPEC §S5.3) ---------------- */

static int int_array(cJSON *parent, const char *key, int *out, int maxn,
                     char *err, size_t errlen)
{
    cJSON *a = cJSON_GetObjectItemCaseSensitive(parent, key);
    if (!cJSON_IsArray(a)) {
        snprintf(err, errlen, "missing/invalid array '%s'", key);
        return -1;
    }
    int n = cJSON_GetArraySize(a);
    if (n < 1 || n > maxn) {
        snprintf(err, errlen, "array '%s' length %d out of range", key, n);
        return -1;
    }
    int i = 0;
    for (cJSON *it = a->child; it; it = it->next, i++) {
        if (!cJSON_IsNumber(it) || it->valueint < 0) {
            snprintf(err, errlen, "array '%s': bad index", key);
            return -1;
        }
        out[i] = it->valueint;
    }
    return n;
}

/* ---------------- candidate gates (SPEC §S5.4) ---------------- */

/* planid_cand_valid covers the planner's shape rules. These add the CPU
 * feature gates and ISA-module match, mirroring the planner's
 * enumeration conditions. Fail closed: doubt -> reject. */
static int feature_gates(const plan_t *pl, const cand_t *cand,
                         const rj_cpu_info *cpu, char *err, size_t errlen)
{
#if defined(RJW_X86)
    if (cand->isa != ISA_SSE && cand->isa != ISA_AVX2 &&
        cand->isa != ISA_AVX512) {
        snprintf(err, errlen, "candidate ISA not in the x86 module");
        return -1;
    }
    if (cand->mul == MUL_GFNI && !cpu->gfni) {
        snprintf(err, errlen, "MUL_GFNI needs the gfni CPU flag");
        return -1;
    }
    if (cand->mul == MUL_ZVBC) {
        snprintf(err, errlen, "MUL_ZVBC is RISC-V only");
        return -1;
    }
    if ((cand->permi2b || cand->mshift) && !cpu->vbmi) {
        snprintf(err, errlen, "permi2b/mshift need the vbmi CPU flag");
        return -1;
    }
    if (cand->ternlog && !(cpu->avx512f && cpu->avx512bw)) {
        snprintf(err, errlen, "ternlog needs AVX-512");
        return -1;
    }
    if (cand->maskedtail && cand->isa == ISA_AVX512 &&
        !(cpu->avx512f && cpu->avx512bw)) {
        snprintf(err, errlen, "maskedtail needs AVX-512");
        return -1;
    }
    if (cand->maskedtail && cand->isa != ISA_AVX512 && cand->isa != ISA_RVV) {
        snprintf(err, errlen, "maskedtail needs the avx512 or rvv ISA");
        return -1;
    }
    if (cand->maskedtail && cand->notail) {
        snprintf(err, errlen, "maskedtail and notail are disjoint");
        return -1;
    }
    if (cand->predtail) {
        snprintf(err, errlen, "predtail is SVE only");
        return -1;
    }
    if (cand->lmul != 0 && cand->lmul != 1) {
        snprintf(err, errlen, "lmul is RISC-V only");
        return -1;
    }
#elif defined(RJW_AARCH64)
    if (cand->isa != ISA_NEON && cand->isa != ISA_SVE) {
        snprintf(err, errlen, "candidate ISA not in the aarch64 module");
        return -1;
    }
    if (cand->predtail && !(cand->isa == ISA_SVE && cpu->sve)) {
        snprintf(err, errlen, "predtail needs SVE");
        return -1;
    }
    if (cand->notail && cand->predtail) {
        snprintf(err, errlen, "notail and predtail are disjoint");
        return -1;
    }
    if (cand->mul != MUL_NIBBLE || cand->ternlog || cand->maskedtail ||
        cand->permi2b || cand->mshift || cand->xorbit) {
        snprintf(err, errlen, "x86-only multiply/twin axis on aarch64");
        return -1;
    }
    if (cand->lmul != 0 && cand->lmul != 1) {
        snprintf(err, errlen, "lmul is RISC-V only");
        return -1;
    }
#elif defined(RJW_RISCV)
    if (cand->isa != ISA_RVV) {
        snprintf(err, errlen, "candidate ISA not in the riscv module");
        return -1;
    }
    if (cand->mul == MUL_ZVBC && !cpu->rvv_zvbc32e) {
        snprintf(err, errlen, "MUL_ZVBC needs the rvv_zvbc32e CPU flag");
        return -1;
    }
    if (cand->mul == MUL_GFNI) {
        snprintf(err, errlen, "MUL_GFNI is x86 only");
        return -1;
    }
    if (cand->ternlog || cand->permi2b ||
        cand->mshift || cand->xorbit || cand->predtail) {
        snprintf(err, errlen, "x86/SVE-only twin axis on riscv");
        return -1;
    }
    /* maskedtail is valid on RVV (RVV masked-tail twin) and x86-64 (AVX-512);
     * the ISA check above already gates it. */
    if (cand->lmul != 0 && cand->lmul != 1 && cand->lmul != 2 &&
        cand->lmul != 4 && cand->lmul != 8) {
        snprintf(err, errlen, "lmul must be 1, 2, 4 or 8");
        return -1;
    }
#endif
    /* Cross-ISA shape rules the planner enforces at enumeration. */
    if (cand->xorsched && !pl->xor_sched) {
        snprintf(err, errlen, "xorsched needs a P_XOR plan with xor_sched");
        return -1;
    }
    if (cand->xorbit && !pl->bit_sched) {
        snprintf(err, errlen, "xorbit needs a plan with bit_sched");
        return -1;
    }
    if (cand->prefetch_dist > 0) {
#if defined(RJW_AARCH64)
        if (cand->prefetch_dist > 32760 ||
            (cand->prefetch_dist & 7u) != 0) {
            snprintf(err, errlen,
                     "aarch64 prefetch distance: 0..32760, multiple of 8");
            return -1;
        }
#elif defined(RJW_RISCV)
        if (!cpu->rvv_zicbop) {
            snprintf(err, errlen, "riscv prefetch needs rvv_zicbop");
            return -1;
        }
        if (cand->prefetch_dist < 32 || cand->prefetch_dist > 2016 ||
            (cand->prefetch_dist & 31u) != 0) {
            snprintf(err, errlen,
                     "riscv prefetch distance: 32..2016, multiple of 32");
            return -1;
        }
#endif
    }
    if (cand->nt && pl->kind == P_DELTA) {
        snprintf(err, errlen, "nt twin not offered for P_DELTA");
        return -1;
    }
    return 0;
}

/* isa_name() comes from plan.h (same strings). */
static const char *mul_name(mul_t m)
{
    switch (m) {
    case MUL_NIBBLE:
        return "nibble";
    case MUL_GFNI:
        return "gfni";
    case MUL_ZVBC:
        return "zvbc";
    default:
        return "?";
    }
}

static const char *kind_name(plan_kind_t k)
{
    switch (k) {
    case P_XOR:
        return "xor";
    case P_MULACC:
        return "mulacc";
    case P_HORNER:
        return "horner";
    case P_DELTA:
        return "delta";
    default:
        return "?";
    }
}

/* ---------------- the compile ---------------- */

static uint8_t *g_code;
static size_t g_code_len;
static uint8_t *g_const;
static size_t g_const_len;

static char *result_error(const char *err)
{
    cJSON *r = cJSON_CreateObject();
    cJSON_AddFalseToObject(r, "ok");
    cJSON_AddStringToObject(r, "err", err);
    char *s = cJSON_PrintUnformatted(r);
    cJSON_Delete(r);
    return s;
}

/* Build a pinned cand_t from the "cand" object. seen_auto is set when the
 * value was the string "auto". */
static int build_pinned_cand(cJSON *cj, const plan_t *pl,
                             const rj_cpu_info *cpu, cand_t *cand, int *is_auto,
                             char *err, size_t errlen)
{
    if (cJSON_IsString(cj)) {
        if (strcmp(cj->valuestring, "auto")) {
            snprintf(err, errlen, "cand must be \"auto\" or an object");
            return -1;
        }
        *is_auto = 1;
        return 0;
    }
    if (!cJSON_IsObject(cj)) {
        snprintf(err, errlen, "cand must be \"auto\" or an object");
        return -1;
    }
    *is_auto = 0;
    memset(cand, 0, sizeof(*cand));

    /* isa: default = highest available in this module for the target */
#if defined(RJW_X86)
    isa_t dflt_isa =
        (cpu->avx512f && cpu->avx512bw) ? ISA_AVX512 : (cpu->avx2 ? ISA_AVX2 : ISA_SSE);
#elif defined(RJW_AARCH64)
    isa_t dflt_isa = cpu->sve ? ISA_SVE : ISA_NEON;
#else
    isa_t dflt_isa = ISA_RVV;
#endif
    const char *isa_s = NULL;
    cJSON *o = cJSON_GetObjectItemCaseSensitive(cj, "isa");
    if (o) {
        if (!cJSON_IsString(o)) {
            snprintf(err, errlen, "cand.isa must be a string");
            return -1;
        }
        isa_s = o->valuestring;
        if (!strcmp(isa_s, "sse"))
            cand->isa = ISA_SSE;
        else if (!strcmp(isa_s, "avx2"))
            cand->isa = ISA_AVX2;
        else if (!strcmp(isa_s, "avx512"))
            cand->isa = ISA_AVX512;
        else if (!strcmp(isa_s, "neon"))
            cand->isa = ISA_NEON;
        else if (!strcmp(isa_s, "sve"))
            cand->isa = ISA_SVE;
        else if (!strcmp(isa_s, "rvv"))
            cand->isa = ISA_RVV;
        else {
            snprintf(err, errlen, "unknown cand.isa '%s'", isa_s);
            return -1;
        }
    } else {
        cand->isa = dflt_isa;
    }

    o = cJSON_GetObjectItemCaseSensitive(cj, "mul");
    if (o && cJSON_IsString(o)) {
        /* Default multiply method follows the planner: GFNI when
         * available on AVX-512, ZVBC never by default, else nibble. */
        if (!strcmp(o->valuestring, "nibble"))
            cand->mul = MUL_NIBBLE;
        else if (!strcmp(o->valuestring, "gfni"))
            cand->mul = MUL_GFNI;
        else if (!strcmp(o->valuestring, "zvbc"))
            cand->mul = MUL_ZVBC;
        else {
            snprintf(err, errlen, "unknown cand.mul '%s'", o->valuestring);
            return -1;
        }
    } else {
#if defined(RJW_X86)
        /* P_XOR plans have no GF multiply: the planner never selects
         * GFNI for them, so the default follows suit (nibble). */
        cand->mul = (cand->isa == ISA_AVX512 && cpu->gfni &&
                     pl->kind != P_XOR) ? MUL_GFNI : MUL_NIBBLE;
#else
        cand->mul = MUL_NIBBLE;
#endif
    }

    err[0] = '\0';
    cand->U = opt_int(cj, "U", 0, err, errlen);
    if (err[0])
        return -1;
    if (cand->U < 0 || cand->U > 8) {
        snprintf(err, errlen, "cand.U must be 0 (auto) or 1..8");
        return -1;
    }
    int lmul = opt_int(cj, "lmul", 0, err, errlen);
    if (err[0])
        return -1;
    cand->lmul = lmul; /* 0 = 1; validated in feature_gates */
    cand->nt = opt_bool(cj, "nt", 0, err, errlen);
    if (err[0])
        return -1;
    long pd = opt_int(cj, "prefetch_dist", 0, err, errlen);
    if (err[0])
        return -1;
    if (pd < 0) {
        snprintf(err, errlen, "cand.prefetch_dist must be >= 0");
        return -1;
    }
    cand->prefetch_dist = (size_t)pd;
    cand->ternlog = opt_bool(cj, "ternlog", 0, err, errlen);
    if (err[0])
        return -1;
    cand->remat = opt_bool(cj, "remat", 0, err, errlen);
    if (err[0])
        return -1;
    cand->notail = opt_bool(cj, "notail", 0, err, errlen);
    if (err[0])
        return -1;
    cand->predtail = opt_bool(cj, "predtail", 0, err, errlen);
    if (err[0])
        return -1;
    cand->maskedtail = opt_bool(cj, "maskedtail", 0, err, errlen);
    if (err[0])
        return -1;
    cand->permi2b = opt_bool(cj, "permi2b", 0, err, errlen);
    if (err[0])
        return -1;
    cand->xorsched = opt_bool(cj, "xorsched", 0, err, errlen);
    if (err[0])
        return -1;
    cand->xorbit = opt_bool(cj, "xorbit", 0, err, errlen);
    if (err[0])
        return -1;
    cand->rec_cf = opt_bool(cj, "rec_cf", 0, err, errlen);
    if (err[0])
        return -1;
    cand->mshift = opt_bool(cj, "mshift", 0, err, errlen);
    if (err[0])
        return -1;
    cand->pipe = opt_bool(cj, "pipe", 0, err, errlen);
    if (err[0])
        return -1;

    /* planid_cand_valid rejects stage-2 tail flags (notail/predtail/
     * maskedtail live in the plan-ID <tail> field, never <twins>), but the
     * rjw API pins them directly as candidate axes (SPEC S5.4) and the
     * backends consume cand->predtail etc. Validate the shape with the
     * tail flags masked out, then re-apply: feature_gates + the backend's
     * asserts are the tail-flag validators on this path. */
    int save_notail = cand->notail, save_predtail = cand->predtail;
    int save_maskedtail = cand->maskedtail;
    cand->notail = cand->predtail = cand->maskedtail = 0;
    int shape_ok = planid_cand_valid(pl, cand, err, errlen);
    cand->notail = save_notail;
    cand->predtail = save_predtail;
    cand->maskedtail = save_maskedtail;
    if (shape_ok != 0) {
        /* U=0 is our "auto-U" spelling: validate the shape with a
         * placeholder U (the real U is chosen by the scan in
         * rjw_compile). */
        if (cand->U == 0) {
            cand_t tmp = *cand;
            tmp.U = 8;
            tmp.notail = tmp.predtail = tmp.maskedtail = 0;
            if (planid_cand_valid(pl, &tmp, err, errlen) != 0)
                return -1;
        } else {
            return -1;
        }
    }
    if (feature_gates(pl, cand, cpu, err, errlen) != 0)
        return -1;
    return 0;
}

char *rjw_compile(const char *json)
{
    char err[512];
    err[0] = '\0';
    cJSON *doc = cJSON_Parse(json);
    if (!doc)
        return result_error("invalid JSON");

    /* ---- schema version ---- */
    cJSON *v = cJSON_GetObjectItemCaseSensitive(doc, "v");
    if (!cJSON_IsNumber(v) || v->valueint != 1) {
        cJSON_Delete(doc);
        return result_error("unsupported schema version (want \"v\":1)");
    }

    /* ---- target bundle ---- */
    const char *target = req_str(doc, "target", err, sizeof(err));
    rj_topology *topo = NULL;
    topo_int_t ti;
    int ti_built = 0;
    plan_t pl;
    int pl_built = 0;
    rj_op op;
    int op_built = 0;
    version_t *vs = NULL;
    int nv = 0;
    char *out = NULL;
    memset(&op, 0, sizeof(op));

    if (!target)
        goto fail;
    if (preset_fill(target, &g_cpu) != 0) {
        snprintf(err, sizeof(err),
                 "unknown target '%s' for this ISA module", target);
        goto fail;
    }
    cJSON *cpuo = cJSON_GetObjectItemCaseSensitive(doc, "cpu");
    if (cpuo) {
        if (!cJSON_IsObject(cpuo)) {
            snprintf(err, sizeof(err), "invalid object 'cpu'");
            goto fail;
        }
        if (cpu_override_apply(cpuo, &g_cpu, err, sizeof(err)) != 0)
            goto fail;
    }

    /* ---- topology ---- */
    cJSON *tj = req_obj(doc, "topology", err, sizeof(err));
    if (!tj)
        goto fail;
    topo = build_topology(tj, err, sizeof(err));
    if (!topo)
        goto fail;
    if (topo_int_build(&ti, topo, err, sizeof(err)) != 0)
        goto fail;
    ti_built = 1;

    /* ---- op ---- */
    cJSON *oj = req_obj(doc, "op", err, sizeof(err));
    if (!oj)
        goto fail;
    const char *kind = req_str(oj, "kind", err, sizeof(err));
    if (!kind)
        goto fail;
    int rec_rows_buf[RJ_MAX_DISKS];
    int n_rec_rows = 0;
    int have_rec_rows = 0;
    if (!strcmp(kind, "encode")) {
        op = rj_op_encode();
    } else if (!strcmp(kind, "recover")) {
        int failed[RJ_MAX_DISKS];
        int nf = int_array(oj, "failed", failed, RJ_MAX_DISKS, err,
                           sizeof(err));
        if (nf < 0)
            goto fail;
        op = rj_op_recover(failed, nf);
        cJSON *ra = cJSON_GetObjectItemCaseSensitive(oj, "rows");
        if (ra) {
            n_rec_rows = int_array(oj, "rows", rec_rows_buf, RJ_MAX_DISKS,
                                  err, sizeof(err));
            if (n_rec_rows < 0)
                goto fail;
            have_rec_rows = 1;
        }
    } else if (!strcmp(kind, "parity_update")) {
        int changed[RJ_MAX_DISKS];
        int nc = int_array(oj, "changed", changed, RJ_MAX_DISKS, err,
                           sizeof(err));
        if (nc < 0)
            goto fail;
        op = rj_op_parity_update(changed, nc);
    } else {
        snprintf(err, sizeof(err), "unknown op kind '%s'", kind);
        goto fail;
    }
    if (op.oom) {
        snprintf(err, sizeof(err), "out of memory building op");
        goto fail;
    }
    op_built = 1;

    /* ---- plan ---- */
    if (have_rec_rows) {
        int *F = NULL;
        int nf =
            plan_recover_failed_list(&op, ti.ndata, ti.npar, &F, err,
                                     sizeof(err));
        if (nf < 0) {
            goto fail;
        }
        int rc = plan_build_recover_rows(&ti, F, nf, rec_rows_buf,
                                         n_rec_rows, "rjw", &pl, err,
                                         sizeof(err));
        free(F);
        if (rc != 0)
            goto fail;
    } else if (plan_build(&ti, &op, "rjw", &pl, err, sizeof(err)) != 0) {
        goto fail;
    }
    pl_built = 1;

    /* ---- candidate ---- */
    cJSON *cj = cJSON_GetObjectItemCaseSensitive(doc, "cand");
    if (!cj) {
        snprintf(err, sizeof(err), "missing 'cand'");
        goto fail;
    }
    cand_t pinned;
    int is_auto = 0;
    if (build_pinned_cand(cj, &pl, &g_cpu, &pinned, &is_auto, err,
                          sizeof(err)) != 0)
        goto fail;

    /* ---- options ---- */
    rj_options o;
    rj_options_default(&o);
    o.benchmark = 0; /* compile-only: never benchmark, never execute */
    cJSON *optj = cJSON_GetObjectItemCaseSensitive(doc, "options");
    if (optj) {
        if (!cJSON_IsObject(optj)) {
            snprintf(err, sizeof(err), "invalid object 'options'");
            goto fail;
        }
        err[0] = '\0';
        o.loop_align = opt_int(optj, "loop_align", 0, err, sizeof(err));
        if (err[0])
            goto fail;
        o.emit_endbr = opt_bool(optj, "emit_endbr", 0, err, sizeof(err));
        if (err[0])
            goto fail;
        o.emit_shadow_stack =
            opt_bool(optj, "emit_shadow_stack", 0, err, sizeof(err));
        if (err[0])
            goto fail;
    }

    /* ---- build ---- */
    if (is_auto) {
        cand_t cands[512];
        int ncand = plan_candidates(&pl, &g_cpu, cands, 512, 0, 0);
        if (ncand <= 0) {
            snprintf(err, sizeof(err), "no candidates for this CPU/plan");
            goto fail;
        }
        vs = calloc((size_t)ncand, sizeof(*vs));
        if (!vs) {
            snprintf(err, sizeof(err), "out of memory");
            goto fail;
        }
        for (int i = 0; i < ncand; i++) {
            char e2[256];
            if (jit_build_version(&pl, &cands[i], &g_cpu, &o, &vs[nv], e2,
                                  sizeof(e2)) == 0)
                nv++;
        }
        if (nv == 0) {
            snprintf(err, sizeof(err), "no candidate built");
            goto fail;
        }
        int w = heuristic_select(vs, nv, &pl);
        if (w < 0) {
            snprintf(err, sizeof(err), "heuristic_select failed");
            goto fail;
        }
        /* Move the winner to slot 0 for uniform handling below. */
        if (w != 0) {
            version_t tmp = vs[0];
            vs[0] = vs[w];
            vs[w] = tmp;
        }
    } else {
        vs = calloc(1, sizeof(*vs));
        if (!vs) {
            snprintf(err, sizeof(err), "out of memory");
            goto fail;
        }
        char e2[256];
        /* U=0 means "auto": build every U in 1..8 that passes the gates
         * and pick the heuristic winner — the same static-model rule
         * the planner uses. Deterministic. */
        if (pinned.U == 0) {
            version_t *uvs = calloc(8, sizeof(*uvs));
            if (!uvs) {
                snprintf(err, sizeof(err), "out of memory");
                free(vs);
                vs = NULL;
                goto fail;
            }
            int unv = 0;
            char lasterr[256] = "";
            /* Mask stage-2 tail flags for planid_cand_valid (see
             * build_pinned_cand): the shape check is tail-agnostic. */
            int pt_notail = pinned.notail, pt_predtail = pinned.predtail;
            int pt_maskedtail = pinned.maskedtail;
            /* Largest-first: matches plan_candidates' Us[] order
             * ({8,7,6,5,4,3,2,1}). heuristic_select keeps the earliest
             * candidate on ties (strict <), so this resolves U-ties the
             * same way auto mode does. Smallest-first used to resolve
             * memory-bound ties to U=1 while auto chose U=8. */
            for (int U = 8; U >= 1; U--) {
                pinned.U = U;
                pinned.notail = pinned.predtail = pinned.maskedtail = 0;
                int vok = planid_cand_valid(&pl, &pinned, e2, sizeof(e2));
                pinned.notail = pt_notail;
                pinned.predtail = pt_predtail;
                pinned.maskedtail = pt_maskedtail;
                if (vok != 0) {
                    snprintf(lasterr, sizeof(lasterr), "%s", e2);
                    continue;
                }
                if (feature_gates(&pl, &pinned, &g_cpu, e2, sizeof(e2)) !=
                    0) {
                    snprintf(lasterr, sizeof(lasterr), "%s", e2);
                    continue;
                }
                if (jit_build_version(&pl, &pinned, &g_cpu, &o,
                                      &uvs[unv], e2, sizeof(e2)) != 0) {
                    snprintf(lasterr, sizeof(lasterr), "%s", e2);
                    continue;
                }
                unv++;
            }
            if (unv == 0) {
                snprintf(err, sizeof(err),
                         "no U in 1..8 builds for these pinned axes: %s",
                         lasterr);
                free(uvs);
                free(vs);
                vs = NULL;
                goto fail;
            }
            int w = heuristic_select(uvs, unv, &pl);
            if (w < 0) {
                snprintf(err, sizeof(err), "heuristic_select failed");
                for (int i = 0; i < unv; i++)
                    jit_free_version(&uvs[i]);
                free(uvs);
                free(vs);
                vs = NULL;
                goto fail;
            }
            vs[0] = uvs[w];
            /* Free the losers without touching the winner's slot. */
            for (int i = 0; i < unv; i++) {
                if (i != w)
                    jit_free_version(&uvs[i]);
            }
            free(uvs);
        } else if (jit_build_version(&pl, &pinned, &g_cpu, &o, vs, e2,
                                     sizeof(e2)) != 0) {
            snprintf(err, sizeof(err), "build failed: %s", e2);
            goto fail;
        }
        nv = 1;
    }
    version_t *bw = &vs[0];

    /* ---- snapshot the blobs (versions are freed below) ---- */
    free(g_code);
    free(g_const);
    g_code = malloc(bw->code_size ? bw->code_size : 1);
    g_const = malloc(bw->const_size ? bw->const_size : 1);
    if (!g_code || !g_const) {
        snprintf(err, sizeof(err), "out of memory");
        goto fail;
    }
    g_code_len = bw->code_size;
    g_const_len = bw->const_size;
    memcpy(g_code, bw->mem, bw->code_size);
    memcpy(g_const, (const uint8_t *)bw->mem + bw->code_size,
           bw->const_size);

    /* ---- plan ID ---- */
    planid_opdesc_t opd;
    {
        int rows[RJ_MAX_DISKS];
        int nrows = 0;
        const int *rowsp = NULL;
        if (op.is_recover) {
            if (have_rec_rows) {
                rowsp = rec_rows_buf;
                nrows = n_rec_rows;
            } else {
                for (int r = 0; r < ti.npar; r++)
                    rows[r] = r;
                rowsp = rows;
                nrows = ti.npar;
            }
        }
        if (planid_opdesc_build(&opd, &ti, &op, rowsp, nrows, err,
                                sizeof(err)) != 0)
            goto fail;
    }
    char plan_id[RJ_PLAN_ID_MAX];
    /* Stage-2 tail flags live in the plan-ID <tail> axis, never in the
     * candidate: planid_format rejects them in cand by design ("fail,
     * don't mis-encode"). The rjw API pins them as candidate axes
     * (SPEC S5.4) and the backend consumed them for the build, so move
     * them to the tail struct for ID formatting. */
    planid_tail_t tail = { bw->cand.notail, bw->cand.predtail,
                           bw->cand.maskedtail };
    cand_t idcand = bw->cand;
    idcand.notail = idcand.predtail = idcand.maskedtail = 0;
    if (planid_format(plan_id, sizeof(plan_id), &ti,
                      topo_internal_name(topo), &opd, &pl, &idcand,
                      &tail) != 0) {
        snprintf(err, sizeof(err), "plan ID did not fit");
        goto fail;
    }

    /* ---- result ---- */
    {
        cJSON *r = cJSON_CreateObject();
        cJSON_AddTrueToObject(r, "ok");
        cJSON_AddStringToObject(r, "desc", bw->desc);
        cJSON_AddStringToObject(r, "plan_id", plan_id);
        cJSON_AddStringToObject(r, "target", target);
        cJSON_AddStringToObject(r, "topology", topo_internal_name(topo));
        cJSON_AddStringToObject(r, "operation", kind);
        cJSON_AddStringToObject(r, "strategy", kind_name(pl.kind));
        /* plan flags for client-side twin grey-out */
        cJSON_AddNumberToObject(r, "plan_xor_sched", pl.xor_sched ? 1 : 0);
        cJSON_AddNumberToObject(r, "plan_n_out", pl.n_out);
        cJSON_AddStringToObject(r, "isa", isa_name(bw->cand.isa));
        cJSON_AddStringToObject(r, "mul", mul_name(bw->cand.mul));
        cJSON_AddNumberToObject(r, "unroll", bw->cand.U);
        cJSON_AddNumberToObject(r, "lmul",
                               bw->cand.lmul ? bw->cand.lmul : 1);
        cJSON *tw = cJSON_CreateArray();
        if (bw->cand.nt)
            cJSON_AddItemToArray(tw, cJSON_CreateString("nt"));
        if (bw->cand.prefetch_dist)
            cJSON_AddItemToArray(
                tw, cJSON_CreateString("prefetch"));
        if (bw->cand.ternlog)
            cJSON_AddItemToArray(tw, cJSON_CreateString("ternlog"));
        if (bw->cand.remat)
            cJSON_AddItemToArray(tw, cJSON_CreateString("remat"));
        if (bw->cand.notail)
            cJSON_AddItemToArray(tw, cJSON_CreateString("notail"));
        if (bw->cand.predtail)
            cJSON_AddItemToArray(tw, cJSON_CreateString("predtail"));
        if (bw->cand.maskedtail)
            cJSON_AddItemToArray(tw, cJSON_CreateString("maskedtail"));
        if (bw->cand.permi2b)
            cJSON_AddItemToArray(tw, cJSON_CreateString("permi2b"));
        if (bw->cand.xorsched)
            cJSON_AddItemToArray(tw, cJSON_CreateString("xorsched"));
        if (bw->cand.xorbit)
            cJSON_AddItemToArray(tw, cJSON_CreateString("xorbit"));
        if (bw->cand.mshift)
            cJSON_AddItemToArray(tw, cJSON_CreateString("mshift"));
        if (bw->cand.pipe)
            cJSON_AddItemToArray(tw, cJSON_CreateString("pipe"));
        cJSON_AddItemToObject(r, "twins", tw);
        cJSON_AddNumberToObject(r, "code_size", (double)g_code_len);
        cJSON_AddNumberToObject(r, "const_size", (double)g_const_len);
        cJSON_AddNumberToObject(r, "uops_per_byte", bw->uops_per_byte);
        cJSON_AddNumberToObject(r, "mem_per_byte", bw->mem_per_byte);
        cJSON_AddStringToObject(r, "selection",
                               is_auto ? "heuristic" : "pinned");
        cJSON_AddFalseToObject(r, "exec_validated");
        cJSON_AddNumberToObject(r, "nominal_vl_bytes", (double)bw->vl);
        out = cJSON_PrintUnformatted(r);
        cJSON_Delete(r);
        if (!out) {
            snprintf(err, sizeof(err), "out of memory");
            goto fail;
        }
    }

fail:
    if (vs) {
        for (int i = 0; i < nv; i++)
            jit_free_version(&vs[i]);
        free(vs);
    }
    if (pl_built)
        plan_free(&pl);
    if (op_built)
        rj_op_free(&op);
    if (ti_built)
        topo_int_free(&ti);
    if (topo)
        rj_topo_free(topo);
    cJSON_Delete(doc);
    if (err[0] && !out)
        out = result_error(err);
    if (!out)
        out = result_error("internal error");
    return out;
}

const uint8_t *rjw_code_bytes(size_t *len_out)
{
    *len_out = g_code_len;
    return g_code;
}

const uint8_t *rjw_const_bytes(size_t *len_out)
{
    *len_out = g_const_len;
    return g_const;
}

void rjw_free_string(char *s)
{
    free(s);
}
