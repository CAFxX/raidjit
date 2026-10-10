#include "ports.h"
#include <string.h>
#include <stdio.h>
#include <stdarg.h>

/* Port model data (generated from data/uarch port files).
 * To regenerate: python3 tools/gen_ports_inc.py data/uarch src/ports_data.inc
 * Then rebuild. Adding a uarch = add a .ports file (no C code change).
 */
#include "ports_data.inc"

/* Log-once helper for fallback diagnostics (MAJOR-4: fallbacks must
 * not be silent). */
static void log_once(const char *key, const char *fmt, ...)
{
    /* Simple once-per-key logging using a small static set. */
    static char seen[8][64];
    static int n_seen = 0;
    int i;
    va_list ap;
    char msg[256];

    for (i = 0; i < n_seen; i++) {
        if (strcmp(seen[i], key) == 0)
            return;
    }
    if (n_seen < 8) {
        strncpy(seen[n_seen], key, 63);
        seen[n_seen][63] = '\0';
        n_seen++;
    }
    va_start(ap, fmt);
    vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);
    fprintf(stderr, "raidjit: port-model: %s\n", msg);
}

/* Map uarch aliases to the model that covers them.
 * (The .ports files use canonical names; aliases resolve here.) */
static rj_uarch_t canonical_uarch(rj_uarch_t u)
{
    switch (u) {
    case RJ_UARCH_INTEL:
        return RJ_UARCH_INTEL_ICELAKE;
    case RJ_UARCH_AMD:
    case RJ_UARCH_AMD_ZEN3:
        return RJ_UARCH_AMD_ZEN4;
    default:
        return u;
    }
}

const rj_port_model_t *rj_ports_for_uarch(rj_uarch_t u)
{
    rj_uarch_t c = canonical_uarch(u);
    const rj_port_model_t *m = pm_model_for_uarch(c);
    if (m)
        return m;
    /* Fallback: generic model (matches old I3 UNKNOWN behavior).
     * Logged once (MAJOR-4: fallbacks must not be silent). */
    log_once("generic-fallback", "no port model for uarch %d, using generic",
             (int)u);
    return pm_model_for_uarch(RJ_UARCH_UNKNOWN);
}

const rj_pm_class_t *rj_ports_find_class(const rj_port_model_t *m,
                                         const char *name)
{
    int i;

    if (!m || !name)
        return NULL;
    for (i = 0; i < m->n_classes; i++) {
        if (strcmp(m->classes[i].name, name) == 0)
            return &m->classes[i];
    }
    return NULL;
}

double rj_ports_bottleneck(const rj_port_model_t *m,
                           const rj_pm_class_t **classes,
                           const double *mix, int n)
{
    double pressure[RJ_PM_MAX_PORTS];
    double worst = 0.0;
    int p, i;

    if (!m || m->n_ports <= 0)
        return 0.0;

    for (p = 0; p < m->n_ports; p++)
        pressure[p] = 0.0;

    /* Accumulate port pressure: for each class, distribute its uops
     * across ports per the class's distribution. mix[i] is in
     * instructions/byte; multiply by uops_per_instr to get uops/byte. */
    for (i = 0; i < n; i++) {
        const rj_pm_class_t *c = classes[i];
        double w = mix[i];
        int q;

        if (!c || w <= 0.0)
            continue;
        w *= c->uops_per_instr; /* instr/byte -> uops/byte */
        for (q = 0; q < m->n_ports; q++)
            pressure[q] += w * c->dist[q];
    }

    /* Bottleneck: max over ports of pressure / throughput. */
    for (p = 0; p < m->n_ports; p++) {
        double tput = m->ports[p].throughput;
        double cyc;

        if (tput <= 0.0)
            continue; /* degenerate port; skip */
        cyc = pressure[p] / tput;
        if (cyc > worst)
            worst = cyc;
    }

    return worst;
}
