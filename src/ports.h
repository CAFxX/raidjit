#pragma once
#include "cpu.h"

/* Per-microarchitecture port-pressure model.
 *
 * Replaces the I3 weighted-average heuristic with a true port-pressure
 * computation: cycles = max over ports (uops_on_port / port_throughput).
 *
 * Port tables live in data files (data/uarch/<name>.ports), not in C
 * code. See hidden_files/port_model_SPEC.md for the model definition,
 * data sources, and validation methodology.
 */

#define RJ_PM_MAX_PORTS 10
#define RJ_PM_MAX_CLASSES 16
#define RJ_PM_MAX_NAME 32

/* A single execution port. */
typedef struct {
    char name[RJ_PM_MAX_NAME]; /* e.g. "p0", "fp0", "b0" */
    double throughput;        /* uops/cycle (sustained) */
} rj_pm_port_t;

/* An instruction class: port distribution for one uop. */
typedef struct {
    char name[RJ_PM_MAX_NAME]; /* e.g. "ternlog", "shuffle" */
    double dist[RJ_PM_MAX_PORTS]; /* fraction of uop on each port;
                                  * sums to 1.0 */
} rj_pm_class_t;

/* Complete port model for one microarchitecture. */
typedef struct {
    rj_uarch_t uarch;
    int n_ports;
    rj_pm_port_t ports[RJ_PM_MAX_PORTS];
    int n_classes;
    rj_pm_class_t classes[RJ_PM_MAX_CLASSES];
    /* Scalar parameters (from the I3 weights, kept for the roofline
     * bandwidth leg and fallbacks): */
    double vec_bw;    /* generic vector-uop dispatch width (uops/cycle) */
    double mem_cpb;   /* cycles per memory byte (load-dominated) */
    double freq512;   /* 512-bit sustained-frequency derate (>= 1.0) */
    double bw_bpc;    /* peak single-core DRAM streaming bandwidth,
                       * bytes/cycle (est., NOT measured) */
} rj_port_model_t;

/* Get the port model for a uarch (never NULL; unknown uarchs get a
 * generic model). The result is cached; the underlying data is loaded
 * from data/uarch/ on first use. Thread-safe after first call. */
const rj_port_model_t *rj_ports_for_uarch(rj_uarch_t u);

/* Find a class by name (NULL if not found). */
const rj_pm_class_t *rj_ports_find_class(const rj_port_model_t *m,
                                         const char *name);

/* Compute the port-pressure bottleneck (cycles/byte) for an
 * instruction mix.
 *
 * mix[i] = uops/byte of class classes[i] (parallel arrays, n entries).
 * Returns max_p ( sum_i mix[i] * dist_i[p] ) / throughput[p].
 *
 * This is the throughput-bound estimate: it assumes perfect OoO
 * scheduling and no frontend/memory bottlenecks (memory is handled
 * by the caller's roofline bandwidth leg).
 */
double rj_ports_bottleneck(const rj_port_model_t *m,
                           const rj_pm_class_t **classes,
                           const double *mix, int n);
