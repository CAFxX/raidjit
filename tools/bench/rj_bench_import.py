#!/usr/bin/env python3
"""Import user-submitted benchmark results into the in-repo facility.

Usage:
    rj_bench_import.py [--new-machine] [--force] <file.json> [...]

Pipeline per file:
  1. Parse JSON (invalid JSON -> reject, no traceback).
  2. Schema version check -> migrate forward (migrate.py) or quarantine.
  3. Machine check (must be registered; --new-machine to register).
  4. Sanity checks (ranges, consistency).
  5. Dedup check (same machine_id + timestamp + topology + op).
  6. Store to bench/results/<machine-id>/<timestamp>-<topo>-<op>.json
     with _compat.json sidecar.

Quarantine (bench/incoming/quarantine/) is never deletion: files stay with
a machine-readable reason so future migrations or human review can rescue them.

This tool never runs benchmarks. It only validates and files submitted data.
"""

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from migrate import migrate, QuarantineError, CURRENT_SCHEMA_VERSION  # noqa: E402

# Repo root = two levels up from tools/bench/
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCH_DIR = os.path.join(REPO_ROOT, "bench")
RESULTS_DIR = os.path.join(BENCH_DIR, "results")
INCOMING_DIR = os.path.join(BENCH_DIR, "incoming")
QUARANTINE_DIR = os.path.join(INCOMING_DIR, "quarantine")

MACHINE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
CODE_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$")

# Sanity bounds. Generous: we flag outliers, we don't pretend to know the max.
MAX_GBPS = 2000.0        # no storage kernel does 2 TB/s (yet)
MAX_UOPS_PER_BYTE = 1000.0
MAX_CODE_SIZE = 10 * 1024 * 1024  # 10 MiB of generated code is absurd
MAX_INPUT_BYTES = 64 * 1024 * 1024  # 64 MiB input cap (MAJOR-1 fix)

# Load the authoritative JSON Schema (CRITICAL-2 fix)
with open(os.path.join(BENCH_DIR, "schema.json"), encoding="utf-8") as _sf:
    _SCHEMA = json.load(_sf)
try:
    import jsonschema
    _HAS_JSONSCHEMA = True
except ImportError:
    _HAS_JSONSCHEMA = False


class Reject(Exception):
    """Unrecoverable: the file is not a benchmark result at all."""
    pass


class Quarantine(Exception):
    """Recoverable later: file is plausible but can't be used yet."""
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def _reject_constant(v):
    # MAJOR-1 fix: NaN/Infinity are not valid measurements
    raise Reject(f"non-finite JSON constant: {v}")


def load_json(path):
    # MAJOR-1 fix: size cap before parsing; reject NaN/Infinity; catch RecursionError
    try:
        size = os.path.getsize(path)
    except OSError as e:
        raise Reject(f"cannot stat: {e}")
    if size > MAX_INPUT_BYTES:
        raise Reject(f"input too large ({size} > {MAX_INPUT_BYTES} bytes)")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f, parse_constant=_reject_constant)
    except json.JSONDecodeError as e:
        raise Reject(f"invalid JSON: {e}")
    except RecursionError:
        raise Reject("JSON too deeply nested (recursion limit)")
    except ValueError as e:
        # parse_constant raises Reject (a ValueError subclass path); re-raise as-is
        if isinstance(e, Reject):
            raise
        raise Reject(f"invalid JSON value: {e}")
    except OSError as e:
        raise Reject(f"cannot read: {e}")


def validate_schema(data):
    """CRITICAL-2 fix: validate against the authoritative bench/schema.json.
    Returns a list of error strings (empty = valid)."""
    if not _HAS_JSONSCHEMA:
        return ["jsonschema package not available; skipping schema validation"]
    import jsonschema
    errors = []
    # jsonschema validates; collect all errors (not just the first)
    validator = jsonschema.Draft7Validator(_SCHEMA)
    for e in sorted(validator.iter_errors(data), key=lambda e: list(e.path)):
        loc = ".".join(str(p) for p in e.path) or "(root)"
        errors.append(f"{loc}: {e.message}")
    return errors


def check_schema_version(data):
    """Migrate or quarantine based on schema_version. Returns migrated data."""
    v = data.get("schema_version")
    # MINOR-1 fix: bool is an int subclass; reject True/False explicitly
    if isinstance(v, bool) or not isinstance(v, int):
        raise Reject("missing or non-integer schema_version")
    if v > CURRENT_SCHEMA_VERSION:
        raise Quarantine(
            f"schema_version {v} is newer than supported {CURRENT_SCHEMA_VERSION} "
            f"(from the future)")
    try:
        return migrate(data, v, CURRENT_SCHEMA_VERSION)
    except QuarantineError as e:
        raise Quarantine(str(e))


def check_machine(data, allow_new):
    mid = data.get("machine_id")
    if not isinstance(mid, str) or not MACHINE_ID_RE.match(mid):
        raise Reject(f"invalid machine_id: {mid!r}")
    mdir = os.path.join(RESULTS_DIR, mid)
    mpath = os.path.join(mdir, "machine.json")
    if not os.path.exists(mpath):
        if allow_new:
            # MINOR-1 fix: --new-machine actually creates the stub
            os.makedirs(mdir, exist_ok=True)
            stub = {
                "machine_id": mid,
                "cpu": {"brand": "UNKNOWN (fill in)", "vendor": "unknown",
                        "uarch": "unknown", "features": [],
                        "rvv_vlen_bytes": 0},
                "os": "unknown",
                "compiler": "unknown",
                "registered": data.get("timestamp", "1970-01-01T00:00:00Z"),
                "notes": "STUB created by rj_bench_import.py --new-machine; "
                         "fill in real values.",
                "machine_trust": 1.0,
            }
            with open(mpath, "w", encoding="utf-8") as f:
                json.dump(stub, f, indent=2, sort_keys=True)
                f.write("\n")
            print(f"[NOTE] created stub {mpath}; fill in real values", file=sys.stderr)
            return stub
        raise Quarantine(f"unknown machine_id {mid!r}; use --new-machine to register")
    try:
        with open(mpath, encoding="utf-8") as f:
            mdata = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        raise Reject(f"machine descriptor corrupt: {e}")
    if mdata.get("machine_id") != mid:
        raise Reject("machine.json machine_id mismatch")
    return mdata


def sanity_checks(data):
    """Returns a list of (severity, message). 'error' -> quarantine, 'warn' -> note.
    Note: structural validation is done by validate_schema() (jsonschema);
    these are semantic sanity checks beyond the schema."""
    issues = []

    for i, c in enumerate(data.get("candidates", [])):
        tag = f"candidates[{i}]"
        # measured sanity
        m = c.get("measured", {})
        for j, g in enumerate(m.get("gbps_median", [])):
            if not isinstance(g, (int, float)) or isinstance(g, bool) or g <= 0:
                issues.append(("error", f"{tag}: gbps_median[{j}] not positive"))
            elif g > MAX_GBPS:
                issues.append(("warn", f"{tag}: gbps_median[{j}]={g} implausibly high"))
        # static sanity
        s = c.get("static", {})
        u = s.get("uops_per_byte")
        if u is not None and (not isinstance(u, (int, float)) or isinstance(u, bool) or u <= 0):
            issues.append(("error", f"{tag}: uops_per_byte not positive"))
        elif isinstance(u, (int, float)) and u > MAX_UOPS_PER_BYTE:
            issues.append(("warn", f"{tag}: uops_per_byte={u} implausibly high"))
        cs = c.get("code_size")
        if cs is not None and cs > MAX_CODE_SIZE:
            issues.append(("warn", f"{tag}: code_size={cs} implausibly large"))

    # At least one candidate must be marked selected (bench_select always picks)
    cands = data.get("candidates", [])
    if cands and not any(c.get("selected") for c in cands):
        issues.append(("warn", "no candidate marked selected"))

    # Timestamp must fully match RFC 3339 UTC (MINOR-1 fix: was re.match prefix only)
    ts = data.get("timestamp", "")
    if not isinstance(ts, str) or not TIMESTAMP_RE.fullmatch(ts):
        issues.append(("error", f"unparseable timestamp: {ts!r}"))

    return issues


def dedup_key(data):
    return (data["machine_id"], data["timestamp"],
            data.get("topology", ""), data.get("op", ""))


def _content_hash(data):
    blob = json.dumps(data, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def _find_duplicate(mid, key, content_hash):
    """MINOR-1 fix: dedup by (machine, timestamp, topology, op) key AND by
    content hash (catches same data under different filenames)."""
    mdir = os.path.join(RESULTS_DIR, mid)
    if not os.path.isdir(mdir):
        return None
    for fn in os.listdir(mdir):
        if not fn.endswith(".json") or fn.endswith(".compat.json") or fn == "machine.json":
            continue
        path = os.path.join(mdir, fn)
        compat_path = path + ".compat.json"
        try:
            with open(path, encoding="utf-8") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError, ValueError):
            continue
        if _content_hash(existing) == content_hash:
            return path + " (identical content)"
        try:
            ek = (existing["machine_id"], existing["timestamp"],
                  existing.get("topology", ""), existing.get("op", ""))
        except KeyError:
            continue
        if ek == key:
            return path + " (same dedup key)"
    return None


def store_result(data, issues, force):
    mid = data["machine_id"]
    ts = data["timestamp"]
    # MINOR-1 fix: strict filename sanitization (was prefix-regex + strip)
    # Timestamp is validated by TIMESTAMP_RE.fullmatch in sanity_checks.
    tfn = re.sub(r"[^0-9TZ]", "", ts)  # keep only digits, T, Z
    topo_fn = re.sub(r"[^a-zA-Z0-9()+_-]", "_", data.get("topology", "unknown"))
    op_fn = re.sub(r"[^a-z0-9_]", "_", data.get("op", "unknown"))
    # Belt and suspenders: reject path separators even after sanitization
    for part in (tfn, topo_fn, op_fn):
        if "/" in part or "\\" in part or part in ("", ".", ".."):
            raise Reject(f"unsafe filename component: {part!r}")
    fname = f"{tfn}-{topo_fn}-{op_fn}.json"

    mdir = os.path.join(RESULTS_DIR, mid)
    os.makedirs(mdir, exist_ok=True)

    # MINOR-1 fix: real dedup by key and content hash (was filename-only)
    content_hash = _content_hash(data)
    dup = _find_duplicate(mid, dedup_key(data), content_hash)
    if dup and not force:
        raise Quarantine(f"duplicate of {dup}; use --force to overwrite")

    dest = os.path.join(mdir, fname)
    if os.path.exists(dest) and not force:
        raise Quarantine(f"result file {fname} exists; use --force to overwrite")

    # Sidecar: compatibility axes for fast filtering without re-deriving
    prov = data.get("provenance", {})
    compat = {
        "schema_version": data.get("schema_version"),
        "raidjit_version": prov.get("raidjit_version"),
        "planid_format_version": prov.get("planid_format_version"),
        "static_model_version": prov.get("static_model_version"),
        "twin_set_version": prov.get("twin_set_version"),
        "config_space_version": prov.get("config_space_version"),
        "bench_protocol_version": prov.get("bench_protocol_version"),
        "warnings": [msg for sev, msg in issues if sev == "warn"],
        "imported_with_migration": data.get("_migrated_from"),
    }
    # Per-candidate config_space versions (may differ within a file in theory)
    csvs = sorted({c.get("config_space_version") for c in data.get("candidates", [])})
    compat["candidate_config_space_versions"] = csvs

    # Content hash for dedup across filenames (computed in _find_duplicate)
    compat["content_sha256"] = content_hash

    stored = copy.deepcopy(data)
    stored.pop("_migrated_from", None)  # sidecar holds it; keep stored file clean
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(stored, f, indent=2, sort_keys=True)
        f.write("\n")
    with open(dest + ".compat.json", "w", encoding="utf-8") as f:
        json.dump(compat, f, indent=2, sort_keys=True)
        f.write("\n")
    return dest


def quarantine_file(src_path, reason):
    os.makedirs(QUARANTINE_DIR, exist_ok=True)
    base = os.path.basename(src_path)
    dest = os.path.join(QUARANTINE_DIR, base)
    # Avoid clobbering an existing quarantined file
    n = 1
    while os.path.exists(dest):
        dest = os.path.join(QUARANTINE_DIR, f"{n:02d}-{base}")
        n += 1
    shutil.copy2(src_path, dest)
    with open(dest + ".reason.json", "w", encoding="utf-8") as f:
        json.dump({"reason": reason, "source": src_path}, f, indent=2)
    return dest


def import_one(path, allow_new, force):
    """Returns (status, detail). Status in {imported, quarantined, rejected}."""
    data = load_json(path)  # may raise Reject
    try:
        data = check_schema_version(data)  # may raise Quarantine/Reject
        # CRITICAL-2 fix: validate against authoritative schema.json
        schema_errors = validate_schema(data)
        # Filter out the "jsonschema not available" note (not a real error)
        real_errors = [e for e in schema_errors
                       if "not available" not in e]
        if real_errors:
            raise Quarantine("schema validation failed: " + "; ".join(real_errors[:5]))
        check_machine(data, allow_new)     # may raise Quarantine/Reject
        issues = sanity_checks(data)
        errors = [m for s, m in issues if s == "error"]
        if errors:
            raise Quarantine("sanity errors: " + "; ".join(errors))
        dest = store_result(data, issues, force)  # may raise Quarantine/Reject
        warns = [m for s, m in issues if s == "warn"]
        return ("imported", dest + (f" ({len(warns)} warnings)" if warns else ""))
    except Quarantine as q:
        dest = quarantine_file(path, q.reason)
        return ("quarantined", f"{dest}: {q.reason}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+", help="Result JSON files to import")
    ap.add_argument("--new-machine", action="store_true",
                    help="Allow unregistered machine_id (registers a stub; fill machine.json afterwards)")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite duplicate result files")
    args = ap.parse_args()

    counts = {"imported": 0, "quarantined": 0, "rejected": 0}
    for path in args.files:
        try:
            status, detail = import_one(path, args.new_machine, args.force)
        except Reject as r:
            status, detail = "rejected", f"{path}: {r}"
        counts[status] += 1
        print(f"[{status.upper()}] {detail}")

    print(f"\n{counts['imported']} imported, {counts['quarantined']} quarantined, "
          f"{counts['rejected']} rejected")
    return 1 if counts["rejected"] else 0


if __name__ == "__main__":
    sys.exit(main())
