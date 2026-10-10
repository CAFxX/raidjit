#include "ports.h"
#include <string.h>

/* Port model data (generated from data/uarch port files).
 * To regenerate: python3 tools/gen_ports_inc.py data/uarch src/ports_data.inc
 * Then rebuild. Adding a uarch = add a .ports file (no C code change).
 */
#include "ports_data.inc"

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
    /* Fallback: generic model (matches old I3 UNKNOWN behavior). */
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
     * across ports per the class's distribution. */
    for (i = 0; i < n; i++) {
        const rj_pm_class_t *c = classes[i];
        double w = mix[i];
        int q;

        if (!c || w <= 0.0)
            continue;
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
