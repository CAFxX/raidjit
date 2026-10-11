/* Tests for the benchmark facility's version-tolerance machinery.
 *
 * These are Python-tool tests driven from CTest (the tools are Python).
 * Each test invokes the tool as a subprocess and checks the exit code
 * and key output markers. No benchmarks are run.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
static int passes = 0;

#define CHECK(cond, msg) do { \
    if (cond) { passes++; } \
    else { failures++; fprintf(stderr, "FAIL: %s\n", msg); } \
} while (0)

/* test_migration_coverage: every schema_version from 1..CURRENT must have
 * a migration path (or explicit quarantine). Enforced via migrate.py. */
static void test_migration_coverage(void)
{
    int rc = system(
        "python3 -c \""
        "import sys; sys.path.insert(0, 'tools/bench');"
        "from migrate import migration_covers, CURRENT_SCHEMA_VERSION;"
        "ok = all(migration_covers(v) for v in range(1, CURRENT_SCHEMA_VERSION + 1));"
        "sys.exit(0 if ok else 1)\"");
    CHECK(rc == 0, "migration covers all versions 1..CURRENT");
}

/* test_empty_state: aggregation on empty results dir must not crash. */
static void test_empty_state(void)
{
    int rc = system(
        "mkdir -p /tmp/bench_emptytest && "
        "python3 -c \""
        "import sys; sys.path.insert(0, 'tools/bench');"
        "import rj_bench_aggregate as agg;"
        "r = agg.aggregate([], None, []);"
        "assert r['ledger']['note'] == 'empty state: no benchmark data yet (valid)';"
        "\"");
    CHECK(rc == 0, "empty aggregation state is valid");
}

/* test_classify_levels: change classifier returns sane levels. */
static void test_classify_levels(void)
{
    int rc = system(
        "python3 -c \""
        "import sys; sys.path.insert(0, 'tools/bench');"
        "from classify import classify, LEVEL_WEIGHTS;"
        "c = {'twins': [], 'isa': 'avx512'};"
        "r = classify(c, 'v1', 'v1', 'a'*64, 'a'*64);"
        "assert r['level'] == 0 and r['weight'] == 1.0, 'identical hashes -> level 0';"
        "r = classify(c, 'v1', 'v2', 'a'*64, 'b'*64);"
        "assert r['level'] == 4, 'unknown change -> level 4';"
        "assert LEVEL_WEIGHTS[4] == 0.1;"
        "\"");
    CHECK(rc == 0, "classifier levels correct");
}

/* test_import_rejects_garbage: invalid JSON must be rejected, not crash. */
static void test_import_rejects_garbage(void)
{
    int rc = system(
        "echo 'not json{{{' > /tmp/bench_garbage.json && "
        "python3 tools/bench/rj_bench_import.py /tmp/bench_garbage.json "
        "2>&1 | grep -q REJECTED");
    CHECK(rc == 0, "garbage input rejected cleanly");
}

/* test_import_quarantines_future: future schema version quarantined. */
static void test_import_quarantines_future(void)
{
    int rc = system(
        "python3 -c \""
        "import json;"
        "d = {'schema_version': 999, 'machine_id': 'x', 'timestamp': '2026-01-01T00:00:00Z'};"
        "json.dump(d, open('/tmp/bench_future.json', 'w'))\" && "
        "python3 tools/bench/rj_bench_import.py /tmp/bench_future.json "
        "2>&1 | grep -q QUARANTINED");
    CHECK(rc == 0, "future schema version quarantined");
}

int main(void)
{
    test_migration_coverage();
    test_empty_state();
    test_classify_levels();
    test_import_rejects_garbage();
    test_import_quarantines_future();

    printf("bench_facility: %d passed, %d failed\n", passes, failures);
    return failures ? 1 : 0;
}
