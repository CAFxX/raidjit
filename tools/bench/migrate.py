#!/usr/bin/env python3
"""Schema migration registry for the raidjit benchmark facility.

Every schema_version bump MUST register a migration function here (or an
explicit quarantine rule). A version bump without a migration entry is a
test failure (see tests/test_bench_facility.c::test_migration_coverage).

Migration functions take (data: dict) -> dict. They must:
  - Not mutate the input (work on a copy).
  - Record provenance: set data["_migrated_from"] = <old version>.
  - Be total: handle every field the old version could contain.

Quarantine rules (for versions we refuse to migrate) are recorded in
QUARANTINED with a human-readable reason.
"""

import copy

# Current schema version. Bump when bench/schema.json changes incompatibly.
CURRENT_SCHEMA_VERSION = 1

# Versions we will not migrate (with reasons). Ingestion quarantines these.
QUARANTINED = {
    # 0: "pre-release draft format, superseded before any results were stored",
}


def _migrate_1_to_1(data):
    """Identity: schema v1 is current."""
    return data


# (from_version, to_version) -> migration function.
# Migrations compose: v1 -> v3 runs v1->v2 then v2->v3.
MIGRATIONS = {
    # (1, 2): _migrate_1_to_2,   # template for the next bump
}


def migrate(data, from_version, to_version=CURRENT_SCHEMA_VERSION):
    """Migrate a result dict from from_version to to_version.

    Raises:
        QuarantineError: no migration path exists (caller quarantines).
        ValueError: from_version is newer than to_version (from the future).
    """
    if from_version > to_version:
        raise ValueError(
            f"result schema_version {from_version} is newer than "
            f"supported {to_version}; quarantine, do not migrate backwards")
    if from_version == to_version:
        return copy.deepcopy(data)
    if from_version in QUARANTINED:
        raise QuarantineError(
            f"schema_version {from_version} is quarantined: "
            f"{QUARANTINED[from_version]}")
    out = copy.deepcopy(data)
    v = from_version
    while v < to_version:
        step = (v, v + 1)
        if step not in MIGRATIONS:
            raise QuarantineError(
                f"no migration path from schema_version {from_version} "
                f"(stuck at {v}); quarantine as legacy")
        out = MIGRATIONS[step](out)
        out["_migrated_from"] = v
        v += 1
    return out


class QuarantineError(Exception):
    """Raised when a result cannot be migrated and must be quarantined."""
    pass


def migration_covers(version):
    """True if `version` can be migrated to CURRENT_SCHEMA_VERSION."""
    if version > CURRENT_SCHEMA_VERSION:
        return False  # from the future: quarantine, not migrate
    if version in QUARANTINED:
        return False
    v = version
    while v < CURRENT_SCHEMA_VERSION:
        if (v, v + 1) not in MIGRATIONS:
            return False
        v += 1
    return True
