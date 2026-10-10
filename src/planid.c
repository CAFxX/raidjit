/* Stable plan IDs (#29). See src/planid.h for the format contract. */
#include "planid.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------------- topology hash ---------------- */

unsigned long long planid_topo_hash(const topo_int_t *ti)
{
    /* FNV-1a 64, fixed algorithm: deterministic across processes and
     * platforms. Feeds ndata/npar as little-endian 32-bit words (no
     * host-endian dependence), then the raw matrix bytes. */
    unsigned long long h = 14695981039346656037ULL;
    unsigned long long words[2];
    words[0] = (unsigned long long)(unsigned int)ti->ndata;
    words[1] = (unsigned long long)(unsigned int)ti->npar;
    const unsigned char *p = (const unsigned char *)words;
    for (size_t i = 0; i < sizeof(words); i++) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    size_t nbytes = (size_t)ti->npar * (size_t)ti->ndata;
    for (size_t i = 0; i < nbytes; i++) {
        h ^= ti->mat[i];
        h *= 1099511628211ULL;
    }
    return h;
}

/* ---------------- format ---------------- */

/* Cursor appender: never truncates, fails loudly instead (the ID
 * contract forbids silent truncation). */
typedef struct {
    char *p;
    char *end; /* one past the last writable byte (NUL reserved) */
} cursor_t;

static int cur_putc(cursor_t *c, char ch)
{
    if (c->p >= c->end)
        return -1;
    *c->p++ = ch;
    return 0;
}

static int cur_puts(cursor_t *c, const char *s)
{
    while (*s)
        if (cur_putc(c, *s++))
            return -1;
    return 0;
}

static int cur_putu(cursor_t *c, unsigned long long v)
{
    char tmp[21]; /* max 20 digits for 64-bit */
    int n = 0;
    if (v == 0)
        tmp[n++] = '0';
    else {
        char rev[21];
        int m = 0;
        while (v > 0) {
            rev[m++] = (char)('0' + v % 10);
            v /= 10;
        }
        while (m > 0)
            tmp[n++] = rev[--m];
    }
    for (int i = 0; i < n; i++)
        if (cur_putc(c, tmp[i]))
            return -1;
    return 0;
}

static int cur_puthex16(cursor_t *c, unsigned long long v)
{
    for (int i = 15; i >= 0; i--) {
        unsigned d = (unsigned)((v >> (4 * i)) & 0xFu);
        if (cur_putc(c, (char)(d < 10 ? '0' + d : 'a' + d - 10)))
            return -1;
    }
    return 0;
}

/* "+"-joined int list (already sorted/deduped by the caller). */
static int cur_put_intlist(cursor_t *c, const int *xs, int n)
{
    for (int i = 0; i < n; i++) {
        if (i > 0 && cur_putc(c, '+'))
            return -1;
        if (xs[i] < 0)
            return -1;
        if (cur_putu(c, (unsigned long long)xs[i]))
            return -1;
    }
    return 0;
}

static const char *kind_token(plan_kind_t k)
{
    switch (k) {
    case P_XOR: return "xor";
    case P_HORNER: return "horner";
    case P_MULACC: return "mulacc";
    case P_DELTA: return "delta";
    }
    return NULL;
}

static const char *mul_token(mul_t m)
{
    switch (m) {
    case MUL_NIBBLE: return "nibble";
    case MUL_GFNI: return "gfni";
    case MUL_ZVBC: return "zvbc";
    }
    return NULL;
}

int planid_format(char *out, size_t outlen, const topo_int_t *ti,
                  const char *toponame, const planid_opdesc_t *opd,
                  const plan_t *pl, const cand_t *cand,
                  const planid_tail_t *tail)
{
    if (!out || outlen == 0 || !ti || !toponame || !opd || !pl || !cand ||
        !tail)
        return -1;
    const char *kt = kind_token(pl->kind);
    const char *mt = mul_token(cand->mul);
    if (!kt || !mt)
        return -1;
    if (cand->U < 1 || cand->U > 999)
        return -1;
    /* The stage-2-only flags are never part of the candidate axes; the
     * tail axis carries the stage-2 decisions. A caller passing them in
     * cand is a programming error: fail, don't mis-encode. */
    if (cand->notail || cand->predtail || cand->maskedtail)
        return -1;
    if (tail->predtail && tail->maskedtail)
        return -1;

    cursor_t c = {out, out + outlen - 1};
    if (cur_puts(&c, "rj"))
        return -1;
    if (cur_putu(&c, RJ_CODEGEN_VERSION))
        return -1;
    if (cur_putc(&c, ':'))
        return -1;
    /* topo axis: "<name>#<mhash16>". The name never contains '#' or ':'
     * (constructors use [a-z0-9(),]); refuse otherwise rather than
     * emitting an unparseable ID. */
    for (const char *s = toponame; *s; s++)
        if (*s == '#' || *s == ':')
            return -1;
    if (cur_puts(&c, toponame))
        return -1;
    if (cur_putc(&c, '#'))
        return -1;
    if (cur_puthex16(&c, planid_topo_hash(ti)))
        return -1;
    if (cur_putc(&c, ':'))
        return -1;
    /* op axis */
    if (opd->is_recover) {
        if (cur_puts(&c, "rec"))
            return -1;
        if (cur_put_intlist(&c, opd->failed, opd->nfailed))
            return -1;
        if (cur_puts(&c, ".rows"))
            return -1;
        if (cur_put_intlist(&c, opd->rows, opd->nrows))
            return -1;
    } else if (opd->is_parity_update) {
        if (cur_puts(&c, "pupd"))
            return -1;
        if (cur_put_intlist(&c, opd->changed, opd->nchanged))
            return -1;
    } else {
        if (cur_puts(&c, "enc"))
            return -1;
    }
    if (cur_putc(&c, ':'))
        return -1;
    if (cur_puts(&c, isa_name(cand->isa)))
        return -1;
    if (cur_putc(&c, ':'))
        return -1;
    if (cur_puts(&c, kt))
        return -1;
    if (cur_putc(&c, ':'))
        return -1;
    if (cur_puts(&c, mt))
        return -1;
    if (cur_putc(&c, ':'))
        return -1;
    if (cur_putc(&c, 'u'))
        return -1;
    if (cur_putu(&c, (unsigned long long)cand->U))
        return -1;
    if (cur_putc(&c, ':'))
        return -1;
    /* twins axis, fixed order; "-" when empty. */
    int any_twin = cand->nt || cand->prefetch_dist || cand->ternlog ||
                   cand->remat || cand->xorsched || cand->xorbit ||
                   cand->mshift || cand->permi2b || cand->pipe || cand->lmul;
    if (!any_twin) {
        if (cur_putc(&c, '-'))
            return -1;
    } else {
        int first = 1;
#define TWIN_TOK(cond, tok)                                                  \
    do {                                                                     \
        if (cond) {                                                          \
            if (!first && cur_putc(&c, '+'))                                 \
                return -1;                                                   \
            first = 0;                                                       \
            if (cur_puts(&c, tok))                                           \
                return -1;                                                   \
        }                                                                    \
    } while (0)
        TWIN_TOK(cand->nt, "nt");
        if (cand->prefetch_dist) {
            if (!first && cur_putc(&c, '+'))
                return -1;
            first = 0;
            if (cur_puts(&c, "pf"))
                return -1;
            if (cur_putu(&c, (unsigned long long)cand->prefetch_dist))
                return -1;
        }
        TWIN_TOK(cand->ternlog, "ternlog");
        TWIN_TOK(cand->remat, "remat");
        TWIN_TOK(cand->xorsched, "xorsched");
        TWIN_TOK(cand->xorbit, "xorbit");
        TWIN_TOK(cand->mshift, "mshift");
        TWIN_TOK(cand->permi2b, "permi2b");
        TWIN_TOK(cand->pipe, "pipe");
        TWIN_TOK(cand->rec_cf, "reccf");
        TWIN_TOK(cand->crc_fold, "crc_fold");
        if (cand->lmul) {
            if (!first && cur_putc(&c, '+'))
                return -1;
            if (cur_puts(&c, "lmul"))
                return -1;
            if (cur_putu(&c, (unsigned long long)cand->lmul))
                return -1;
        }
#undef TWIN_TOK
    }
    if (cur_putc(&c, ':'))
        return -1;
    /* tail axis */
    if (!tail->notail && !tail->predtail && !tail->maskedtail) {
        if (cur_puts(&c, "scalar"))
            return -1;
    } else {
        int first = 1;
        if (tail->notail) {
            if (cur_puts(&c, "notail"))
                return -1;
            first = 0;
        }
        if (tail->predtail) {
            if (!first && cur_putc(&c, '+'))
                return -1;
            if (cur_puts(&c, "predtail"))
                return -1;
            first = 0;
        }
        if (tail->maskedtail) {
            if (!first && cur_putc(&c, '+'))
                return -1;
            if (cur_puts(&c, "maskedtail"))
                return -1;
        }
    }
    *c.p = '\0';
    return 0;
}

/* ---------------- parse ---------------- */

typedef struct {
    const char *p;
    size_t n;
} field_t;

static int field_eq(field_t f, const char *s)
{
    size_t m = strlen(s);
    return f.n == m && memcmp(f.p, s, m) == 0;
}

/* Split id on ':' into exactly 9 fields. */
static int split_fields(const char *id, field_t f[9])
{
    int nf = 0;
    const char *s = id;
    while (nf < 9) {
        const char *e = strchr(s, ':');
        if (!e) {
            /* last field: must be field 8 */
            if (nf != 8)
                return -1;
            f[nf].p = s;
            f[nf].n = strlen(s);
            nf++;
            break;
        }
        f[nf].p = s;
        f[nf].n = (size_t)(e - s);
        nf++;
        s = e + 1;
    }
    if (nf != 9)
        return -1;
    /* no empty fields, no trailing ':' (strchr above would have made an
     * empty 9th field only if the string ends with ':' -- caught by the
     * nf != 9 check... actually "a:...:" gives f[8] = "" which is
     * caught below) */
    for (int i = 0; i < 9; i++)
        if (f[i].n == 0)
            return -1;
    /* extra ':' beyond 9 fields */
    if (strchr(f[8].p + f[8].n, ':') || memchr(f[8].p, ':', f[8].n))
        return -1;
    return 0;
}

static int parse_ull(field_t f, unsigned long long *out)
{
    if (f.n == 0 || f.n > 20)
        return -1;
    unsigned long long v = 0;
    for (size_t i = 0; i < f.n; i++) {
        if (f.p[i] < '0' || f.p[i] > '9')
            return -1;
        unsigned d = (unsigned)(f.p[i] - '0');
        if (v > (unsigned long long)-1 / 10)
            return -1;
        v *= 10;
        if (v > (unsigned long long)-1 - d)
            return -1;
        v += d;
    }
    *out = v;
    return 0;
}

/* Parse a "+"-separated int list into out (max RJ_MAX_DISKS entries);
 * each value in [lo, hi]. Sorts + dedupes (canonical form). */
static int cmp_int_asc(const void *a, const void *b)
{
    int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}

static int parse_int_list(field_t f, int *out, int *n_out, int lo, int hi,
                          char *err, size_t errlen, const char *what)
{
    int n = 0;
    size_t i = 0;
    while (i < f.n) {
        size_t j = i;
        while (j < f.n && f.p[j] != '+')
            j++;
        if (j == i) {
            snprintf(err, errlen, "unknown plan ID: empty %s entry", what);
            return -1;
        }
        field_t tok = {f.p + i, j - i};
        unsigned long long v;
        if (parse_ull(tok, &v) != 0 || v > (unsigned long long)hi) {
            snprintf(err, errlen, "unknown plan ID: bad %s value", what);
            return -1;
        }
        if ((long long)v < lo) {
            snprintf(err, errlen, "unknown plan ID: bad %s value", what);
            return -1;
        }
        if (n >= RJ_MAX_DISKS) {
            snprintf(err, errlen, "unknown plan ID: too many %s entries",
                     what);
            return -1;
        }
        out[n++] = (int)v;
        i = j + 1;
    }
    if (n == 0) {
        snprintf(err, errlen, "unknown plan ID: empty %s list", what);
        return -1;
    }
    qsort(out, (size_t)n, sizeof(int), cmp_int_asc);
    int w = 0;
    for (int k = 0; k < n; k++)
        if (k == 0 || out[k] != out[k - 1])
            out[w++] = out[k];
    *n_out = w;
    return 0;
}

/* Canonicalize a caller-side int array (sort + dedupe) for comparison
 * against a parsed list. */
static int canon_list(const int *in, int n, int *out)
{
    if (n > RJ_MAX_DISKS)
        n = RJ_MAX_DISKS;
    memcpy(out, in, (size_t)n * sizeof(int));
    qsort(out, (size_t)n, sizeof(int), cmp_int_asc);
    int w = 0;
    for (int k = 0; k < n; k++)
        if (k == 0 || out[k] != out[k - 1])
            out[w++] = out[k];
    return w;
}

static int list_eq(const int *a, int na, const int *b, int nb)
{
    if (na != nb)
        return 0;
    return memcmp(a, b, (size_t)na * sizeof(int)) == 0;
}

static int parse_hex16(field_t f, unsigned long long *out)
{
    if (f.n != 16)
        return -1;
    unsigned long long v = 0;
    for (size_t i = 0; i < 16; i++) {
        char ch = f.p[i];
        unsigned d;
        if (ch >= '0' && ch <= '9')
            d = (unsigned)(ch - '0');
        else if (ch >= 'a' && ch <= 'f')
            d = (unsigned)(ch - 'a' + 10);
        else
            return -1;
        v = (v << 4) | d;
    }
    *out = v;
    return 0;
}

int planid_parse(const char *id, const topo_int_t *ti, const char *toponame,
                 const rj_op *op, planid_spec_t *spec, char *err,
                 size_t errlen)
{
    if (!id || !ti || !toponame || !op || !spec) {
        if (err && errlen)
            snprintf(err, errlen, "unknown plan ID: null argument");
        return -1;
    }
    memset(spec, 0, sizeof(*spec));
    field_t f[9];
    if (split_fields(id, f) != 0) {
        snprintf(err, errlen,
                 "unknown plan ID: want 9 colon-separated axes, got %s",
                 id);
        return -1;
    }

    /* axis 0: version. Checked before anything else so a stale ID gets
     * the version-mismatch error (forcing re-benchmarking), not a
     * confusing "unknown ID". */
    if (f[0].n < 3 || f[0].p[0] != 'r' || f[0].p[1] != 'j') {
        snprintf(err, errlen, "unknown plan ID: bad version prefix");
        return -1;
    }
    field_t vf = {f[0].p + 2, f[0].n - 2};
    unsigned long long ver;
    if (parse_ull(vf, &ver) != 0) {
        snprintf(err, errlen, "unknown plan ID: bad version number");
        return -1;
    }
    if (ver != (unsigned long long)RJ_CODEGEN_VERSION) {
        snprintf(err, errlen,
                 "plan ID version mismatch: ID is codegen v%llu, this "
                 "build is v%d -- re-benchmark under the new build; "
                 "stale pins are never honored",
                 ver, RJ_CODEGEN_VERSION);
        return -1;
    }

    /* axis 1: topo "<name>#<mhash16>". */
    const char *hash = memchr(f[1].p, '#', f[1].n);
    if (!hash) {
        snprintf(err, errlen,
                 "unknown plan ID: bad topology axis (want name#hash16)");
        return -1;
    }
    field_t nm = {f[1].p, (size_t)(hash - f[1].p)};
    field_t hx = {hash + 1, (size_t)(f[1].p + f[1].n - hash - 1)};
    if (nm.n == 0 || nm.n >= 64) {
        snprintf(err, errlen, "unknown plan ID: bad topology name");
        return -1;
    }
    char namebuf[64];
    memcpy(namebuf, nm.p, nm.n);
    namebuf[nm.n] = '\0';
    if (strcmp(namebuf, toponame) != 0) {
        snprintf(err, errlen,
                 "plan ID topology mismatch: ID names '%s', caller passed "
                 "'%s'",
                 namebuf, toponame);
        return -1;
    }
    unsigned long long want_hash;
    if (parse_hex16(hx, &want_hash) != 0) {
        snprintf(err, errlen, "unknown plan ID: bad topology hash");
        return -1;
    }
    if (want_hash != planid_topo_hash(ti)) {
        snprintf(err, errlen,
                 "plan ID topology mismatch: matrix fingerprint differs "
                 "(same name '%s', different matrix/dimensions)",
                 namebuf);
        return -1;
    }

    /* axis 2: op. */
    int nd = ti->ndata, np = ti->npar;
    planid_opdesc_t *opd = &spec->opd;
    if (field_eq(f[2], "enc")) {
        if (op->is_recover || op->is_parity_update || op->oom) {
            snprintf(err, errlen,
                     "plan ID op mismatch: ID is encode, caller op is not");
            return -1;
        }
    } else if (f[2].n > 4 && memcmp(f[2].p, "pupd", 4) == 0) {
        if (!op->is_parity_update || op->is_recover || op->oom) {
            snprintf(err, errlen,
                     "plan ID op mismatch: ID is parity-update, caller op "
                     "is not");
            return -1;
        }
        opd->is_parity_update = 1;
        field_t rest = {f[2].p + 4, f[2].n - 4};
        if (parse_int_list(rest, opd->changed, &opd->nchanged, 0, nd - 1,
                           err, errlen, "changed-disk") != 0)
            return -1;
        int caller_changed[RJ_MAX_DISKS];
        int nc = canon_list(op->changed, op->nchanged, caller_changed);
        if (!list_eq(opd->changed, opd->nchanged, caller_changed, nc)) {
            snprintf(err, errlen,
                     "plan ID op mismatch: changed-disk set differs from "
                     "the caller's op");
            return -1;
        }
    } else if (f[2].n > 3 && memcmp(f[2].p, "rec", 3) == 0) {
        if (!op->is_recover || op->is_parity_update || op->oom) {
            snprintf(err, errlen,
                     "plan ID op mismatch: ID is recover, caller op is not");
            return -1;
        }
        opd->is_recover = 1;
        /* "rec<f...>.rows<r...>": split on ".rows" (manual search:
         * memmem is a GNU extension, unavailable under the build's
         * _POSIX_C_SOURCE=200809L -std=c11). */
        const char *dot = NULL;
        for (size_t k = 0; k + 5 <= f[2].n; k++) {
            if (memcmp(f[2].p + k, ".rows", 5) == 0) {
                dot = f[2].p + k;
                break;
            }
        }
        if (!dot) {
            snprintf(err, errlen,
                     "unknown plan ID: recover op wants "
                     "rec<failed>.rows<rows>");
            return -1;
        }
        field_t fl = {f[2].p + 3, (size_t)(dot - f[2].p - 3)};
        field_t rl = {dot + 5, (size_t)(f[2].p + f[2].n - dot - 5)};
        if (parse_int_list(fl, opd->failed, &opd->nfailed, 0, nd + np - 1,
                           err, errlen, "failed-disk") != 0)
            return -1;
        if (parse_int_list(rl, opd->rows, &opd->nrows, 0, np - 1, err,
                           errlen, "equation-row") != 0)
            return -1;
        int caller_failed[RJ_MAX_DISKS];
        int nf = canon_list(op->failed, op->nfailed, caller_failed);
        if (!list_eq(opd->failed, opd->nfailed, caller_failed, nf)) {
            snprintf(err, errlen,
                     "plan ID op mismatch: failed-disk set differs from "
                     "the caller's op");
            return -1;
        }
        if (opd->nrows < 1) {
            snprintf(err, errlen,
                     "unknown plan ID: empty equation-row set");
            return -1;
        }
    } else {
        snprintf(err, errlen, "unknown plan ID: bad op axis");
        return -1;
    }

    /* axis 3: isa. */
    cand_t *c = &spec->cand;
    if (field_eq(f[3], "sse"))
        c->isa = ISA_SSE;
    else if (field_eq(f[3], "avx2"))
        c->isa = ISA_AVX2;
    else if (field_eq(f[3], "avx512"))
        c->isa = ISA_AVX512;
    else if (field_eq(f[3], "neon"))
        c->isa = ISA_NEON;
    else if (field_eq(f[3], "sve"))
        c->isa = ISA_SVE;
    else if (field_eq(f[3], "rvv"))
        c->isa = ISA_RVV;
    else if (field_eq(f[3], "pext"))
        c->isa = ISA_PEXT;
    else {
        snprintf(err, errlen, "unknown plan ID: bad isa axis");
        return -1;
    }

    /* axis 4: kind. */
    if (field_eq(f[4], "xor"))
        spec->kind = P_XOR;
    else if (field_eq(f[4], "horner"))
        spec->kind = P_HORNER;
    else if (field_eq(f[4], "mulacc"))
        spec->kind = P_MULACC;
    else if (field_eq(f[4], "delta"))
        spec->kind = P_DELTA;
    else {
        snprintf(err, errlen, "unknown plan ID: bad kind axis");
        return -1;
    }

    /* axis 5: mul. */
    if (field_eq(f[5], "nibble"))
        c->mul = MUL_NIBBLE;
    else if (field_eq(f[5], "gfni"))
        c->mul = MUL_GFNI;
    else if (field_eq(f[5], "zvbc"))
        c->mul = MUL_ZVBC;
    else {
        snprintf(err, errlen, "unknown plan ID: bad mul axis");
        return -1;
    }

    /* axis 6: u<U>. */
    if (f[6].n < 2 || f[6].p[0] != 'u') {
        snprintf(err, errlen, "unknown plan ID: bad unroll axis");
        return -1;
    }
    field_t uf = {f[6].p + 1, f[6].n - 1};
    unsigned long long uu;
    if (parse_ull(uf, &uu) != 0 || uu < 1 || uu > 8) {
        snprintf(err, errlen,
                 "unknown plan ID: unroll out of range (want u1..u8)");
        return -1;
    }
    c->U = (int)uu;

    /* axis 7: twins. */
    if (!field_eq(f[7], "-")) {
        size_t i = 0;
        while (i < f[7].n) {
            size_t j = i;
            while (j < f[7].n && f[7].p[j] != '+')
                j++;
            if (j == i) {
                snprintf(err, errlen,
                         "unknown plan ID: empty twin token");
                return -1;
            }
            field_t tok = {f[7].p + i, j - i};
            if (field_eq(tok, "nt")) {
                if (c->nt) goto dup_twin;
                c->nt = 1;
            } else if (field_eq(tok, "ternlog")) {
                if (c->ternlog) goto dup_twin;
                c->ternlog = 1;
            } else if (field_eq(tok, "remat")) {
                if (c->remat) goto dup_twin;
                c->remat = 1;
            } else if (field_eq(tok, "xorsched")) {
                if (c->xorsched) goto dup_twin;
                c->xorsched = 1;
            } else if (field_eq(tok, "xorbit")) {
                if (c->xorbit) goto dup_twin;
                c->xorbit = 1;
            } else if (field_eq(tok, "mshift")) {
                if (c->mshift) goto dup_twin;
                c->mshift = 1;
            } else if (field_eq(tok, "permi2b")) {
                if (c->permi2b) goto dup_twin;
                c->permi2b = 1;
            } else if (field_eq(tok, "pipe")) {
                if (c->pipe) goto dup_twin;
                c->pipe = 1;
            } else if (field_eq(tok, "reccf")) {
                if (c->rec_cf) goto dup_twin;
                c->rec_cf = 1;
            } else if (field_eq(tok, "crc_fold")) {
                if (c->crc_fold) goto dup_twin;
                c->crc_fold = 1;
            } else if (tok.n > 2 && memcmp(tok.p, "pf", 2) == 0) {
                field_t vf2 = {tok.p + 2, tok.n - 2};
                unsigned long long d;
                if (c->prefetch_dist || parse_ull(vf2, &d) != 0 || d == 0 ||
                    d == (unsigned long long)(size_t)-1) {
                    snprintf(err, errlen,
                             "unknown plan ID: bad pf<N> twin token");
                    return -1;
                }
                c->prefetch_dist = (size_t)d;
            } else if (tok.n > 4 && memcmp(tok.p, "lmul", 4) == 0) {
                field_t vf2 = {tok.p + 4, tok.n - 4};
                unsigned long long l;
                if (c->lmul || parse_ull(vf2, &l) != 0 || (l != 1 && l != 2 &&
                    l != 4 && l != 8)) {
                    snprintf(err, errlen,
                             "unknown plan ID: bad lmul<N> twin token");
                    return -1;
                }
                c->lmul = (int)l;
            } else {
                snprintf(err, errlen,
                         "unknown plan ID: unknown twin axis '%.*s'",
                         (int)tok.n, tok.p);
                return -1;
            }
            i = j + 1;
            continue;
        dup_twin:
            snprintf(err, errlen,
                     "unknown plan ID: duplicate twin axis");
            return -1;
        }
    }

    /* axis 8: tail. */
    if (field_eq(f[8], "scalar")) {
        /* nothing installed */
    } else {
        size_t i = 0;
        while (i < f[8].n) {
            size_t j = i;
            while (j < f[8].n && f[8].p[j] != '+')
                j++;
            if (j == i) {
                snprintf(err, errlen, "unknown plan ID: empty tail token");
                return -1;
            }
            field_t tok = {f[8].p + i, j - i};
            if (field_eq(tok, "notail")) {
                if (spec->tail.notail) goto dup_tail;
                spec->tail.notail = 1;
            } else if (field_eq(tok, "predtail")) {
                if (spec->tail.predtail) goto dup_tail;
                spec->tail.predtail = 1;
            } else if (field_eq(tok, "maskedtail")) {
                if (spec->tail.maskedtail) goto dup_tail;
                spec->tail.maskedtail = 1;
            } else {
                snprintf(err, errlen,
                         "unknown plan ID: unknown tail axis '%.*s'",
                         (int)tok.n, tok.p);
                return -1;
            }
            i = j + 1;
            continue;
        dup_tail:
            snprintf(err, errlen, "unknown plan ID: duplicate tail axis");
            return -1;
        }
        if (spec->tail.predtail && spec->tail.maskedtail) {
            snprintf(err, errlen,
                     "unknown plan ID: predtail and maskedtail are mutually "
                     "exclusive");
            return -1;
        }
    }
    return 0;
}

/* ---------------- candidate shape validation ---------------- */

int planid_cand_valid(const plan_t *pl, const cand_t *cand, char *err,
                      size_t errlen)
{
#define SHAPE_BAD(...)                            \
    do {                                          \
        snprintf(err, errlen, __VA_ARGS__);       \
        return -1;                                \
    } while (0)
    if (!pl || !cand)
        SHAPE_BAD("unknown plan ID: null plan/candidate");
    if (cand->U < 1 || cand->U > 8)
        SHAPE_BAD("unknown plan ID: unroll %d out of range (1..8)",
                  cand->U);
    /* Stage-2-only flags are never part of a parsed candidate: the
     * twin vocabulary has no tokens for them (the tail axis carries
     * the stage-2 decisions). Defensive: the parser never sets them. */
    if (cand->notail || cand->predtail || cand->maskedtail)
        SHAPE_BAD("unknown plan ID: stage-2 tail flag on the candidate");
    /* The backend asserts below become clean errors here: a bad ID must
     * fail loudly, never abort. Each mirrors the planner's enumeration
     * contract and the backend's shape asserts. */
    if (cand->remat && pl->kind != P_HORNER)
        SHAPE_BAD("unknown plan ID: remat is P_HORNER-only");
    if (cand->xorsched &&
        (pl->kind != P_XOR || cand->ternlog || cand->xorbit))
        SHAPE_BAD("unknown plan ID: xorsched needs a P_XOR non-ternlog "
                  "base");
    if (cand->xorbit) {
        if (pl->kind != P_MULACC && pl->kind != P_DELTA &&
            pl->kind != P_HORNER)
            SHAPE_BAD("unknown plan ID: xorbit needs GF-multiply rows");
        if (cand->mul != MUL_NIBBLE)
            SHAPE_BAD("unknown plan ID: xorbit needs the nibble method");
        if (cand->ternlog || cand->permi2b || cand->remat || cand->xorsched)
            SHAPE_BAD("unknown plan ID: xorbit combines with no other "
                      "body twin");
        /* BUG_t1_529: xorbit is 5-8x worse than nibble on x86. */
        if (cand->isa == ISA_SSE || cand->isa == ISA_AVX2 ||
            cand->isa == ISA_AVX512)
            SHAPE_BAD("unknown plan ID: xorbit not available on x86 "
                      "(BUG_t1_529)");
    }
    if (cand->mshift) {
        if (cand->isa != ISA_AVX512 || cand->mul != MUL_NIBBLE)
            SHAPE_BAD("unknown plan ID: mshift needs AVX-512 nibble");
        if (pl->kind != P_MULACC && pl->kind != P_DELTA &&
            pl->kind != P_HORNER)
            SHAPE_BAD("unknown plan ID: mshift needs GF-multiply rows");
        if (cand->xorbit || cand->permi2b || cand->remat || cand->xorsched)
            SHAPE_BAD("unknown plan ID: mshift combines with no other "
                      "body twin");
    }
    if (cand->permi2b) {
        /* Full eligibility (VBMI host, nontrivial coefficients) is
         * checked by the caller via api_permi2b_duel_eligible; the
         * shape part is checked here. */
        if (cand->isa != ISA_AVX512 || cand->mul != MUL_NIBBLE)
            SHAPE_BAD("unknown plan ID: permi2b needs AVX-512 nibble");
        if (pl->kind != P_MULACC && pl->kind != P_DELTA)
            SHAPE_BAD("unknown plan ID: permi2b needs P_MULACC/P_DELTA");
        if (cand->xorbit || cand->mshift || cand->remat || cand->xorsched)
            SHAPE_BAD("unknown plan ID: permi2b combines with no other "
                      "body twin");
    }
    if (cand->pipe) {
        if (cand->isa != ISA_NEON && cand->isa != ISA_SVE &&
            cand->isa != ISA_RVV)
            SHAPE_BAD("unknown plan ID: pipe needs NEON/SVE/RVV");
        if (cand->mul != MUL_NIBBLE)
            SHAPE_BAD("unknown plan ID: pipe needs the nibble method");
        if (pl->kind != P_MULACC && pl->kind != P_HORNER)
            SHAPE_BAD("unknown plan ID: pipe needs P_MULACC/P_HORNER");
        if (cand->ternlog || cand->permi2b || cand->remat ||
            cand->xorsched || cand->xorbit || cand->mshift)
            SHAPE_BAD("unknown plan ID: pipe combines with no other "
                      "body twin");
    }
    if (cand->ternlog && cand->isa != ISA_AVX512)
        SHAPE_BAD("unknown plan ID: ternlog is AVX-512-only");
    if (cand->lmul) {
        if (cand->isa != ISA_RVV)
            SHAPE_BAD("unknown plan ID: lmul is RVV-only");
        if (cand->lmul != 1 && cand->lmul != 2 && cand->lmul != 4 &&
            cand->lmul != 8)
            SHAPE_BAD("unknown plan ID: bad LMUL %d", cand->lmul);
    }
    if (pl->kind == P_XOR && cand->mul != MUL_NIBBLE)
        SHAPE_BAD("unknown plan ID: P_XOR always uses the nibble method");
#undef SHAPE_BAD
    return 0;
}

/* ---------------- op descriptor ---------------- */

int planid_opdesc_build(planid_opdesc_t *opd, const topo_int_t *ti,
                        const rj_op *op, const int *rows, int nrows,
                        char *err, size_t errlen)
{
    if (!opd || !ti || !op) {
        if (err && errlen)
            snprintf(err, errlen, "null argument");
        return -1;
    }
    memset(opd, 0, sizeof(*opd));
    if (op->oom) {
        snprintf(err, errlen, "operation out of memory");
        return -1;
    }
    if (op->is_parity_update && op->is_recover) {
        snprintf(err, errlen, "malformed operation (recover+parity_update)");
        return -1;
    }
    if (op->is_recover && !op->is_parity_update) {
        opd->is_recover = 1;
        opd->nfailed = canon_list(op->failed, op->nfailed, opd->failed);
        if (opd->nfailed < 1) {
            snprintf(err, errlen, "recover: empty failed-disk list");
            return -1;
        }
        for (int i = 0; i < opd->nfailed; i++)
            if (opd->failed[i] < 0 ||
                opd->failed[i] >= ti->ndata + ti->npar) {
                snprintf(err, errlen, "recover: disk index %d out of range",
                         opd->failed[i]);
                return -1;
            }
        if (!rows || nrows < 1 || nrows > RJ_MAX_DISKS) {
            snprintf(err, errlen, "recover: bad equation-row set");
            return -1;
        }
        opd->nrows = canon_list(rows, nrows, opd->rows);
        for (int i = 0; i < opd->nrows; i++)
            if (opd->rows[i] < 0 || opd->rows[i] >= ti->npar) {
                snprintf(err, errlen, "recover: equation row %d out of "
                         "range",
                         opd->rows[i]);
                return -1;
            }
    } else if (op->is_parity_update) {
        opd->is_parity_update = 1;
        opd->nchanged = canon_list(op->changed, op->nchanged, opd->changed);
        if (opd->nchanged < 1) {
            snprintf(err, errlen, "parity_update: empty changed-disk list");
            return -1;
        }
        for (int i = 0; i < opd->nchanged; i++)
            if (opd->changed[i] < 0 || opd->changed[i] >= ti->ndata) {
                snprintf(err, errlen,
                         "parity_update: data-disk index %d out of range",
                         opd->changed[i]);
                return -1;
            }
    }
    return 0;
}
