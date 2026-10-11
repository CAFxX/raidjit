#!/usr/bin/env python3
"""Per-candidate freshness computation (bench/STALENESS.md §3).

Freshness answers: "how much do I trust this candidate's old measurement
when reasoning about target_version?"

    freshness(candidate, target_version) =
        classification_weight(code_change(candidate, measured_version -> target))

Freshness is per-candidate, not per-file. A comparison set where twinA changed
but twinB didn't keeps twinB's signal at full weight.

This module also implements the pairwise usability rule:
    usable(A vs B) iff min(freshness(A), freshness(B)) >= threshold
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classify import classify, _load_annotations  # noqa: E402

USABILITY_THRESHOLD = 0.5  # min(freshness) for a pairwise comparison to count
DECAY_BASE = 0.9           # per-relevant-change decay (STALENESS.md §4)


def build_hash_index(results):
    """Build {(norm_config, raidjit_version): [(code_hash, candidate, result)]}.

    norm_config is a canonical string of the candidate's config fields.
    Multiple entries per key -> nondeterminism (flagged by the caller).
    """
    index = {}
    for r in results:
        v = r["provenance"]["raidjit_version"]
        for c in r["candidates"]:
            key = (norm_config(c), v)
            index.setdefault(key, []).append((c["code_hash"], c, r))
    return index


def norm_config(c):
    """Canonical config string for grouping. Sorts twins for stability."""
    twins = ",".join(sorted(c.get("twins", [])))
    parts = [
        f"topo={c.get('_topology', '?')}",
        f"op={c.get('_op', '?')}",
        f"kind={c.get('plan_kind', '?')}",
        f"isa={c.get('isa', '?')}",
        f"mul={c.get('mul', '?')}",
        f"twins=[{twins}]",
        f"u={c.get('u', '?')}",
    ]
    return "|".join(parts)


def detect_nondeterminism(hash_index):
    """Find (config, version) keys with >1 distinct code_hash.

    Returns list of dicts describing each nondeterminism alert.
    Same config + same version + different hash = codegen nondeterminism BUG.
    """
    alerts = []
    for (cfg, ver), entries in hash_index.items():
        hashes = {h for h, _, _ in entries}
        if len(hashes) > 1:
            alerts.append({
                "type": "NONDETERMINISM",
                "config": cfg,
                "raidjit_version": ver,
                "distinct_hashes": sorted(hashes),
                "count": len(entries),
                "severity": "BUG",
                "message": (
                    f"config {cfg} at {ver[:12]} produced "
                    f"{len(hashes)} distinct code hashes: codegen is "
                    f"nondeterministic (uninitialized memory? ASLR leak?). "
                    f"File a bug; quarantine these measurements."
                ),
            })
    return alerts


def detect_codegen_drift(hash_index):
    """Track each config's code_hash across versions.

    Returns list of drift records: config, v1->v2, hash1->hash2.
    The caller classifies each drift via classify().
    """
    # Group by config -> sorted list of (version, hash)
    by_config = {}
    for (cfg, ver), entries in hash_index.items():
        # Take the most common hash if nondeterministic (alert already filed)
        hashes = {}
        for h, _, _ in entries:
            hashes[h] = hashes.get(h, 0) + 1
        best = max(hashes, key=hashes.get)
        by_config.setdefault(cfg, []).append((ver, best))

    drifts = []
    for cfg, vhs in by_config.items():
        vhs.sort()  # versions sort lexicographically; SHAs don't order chronologically
        # Note: true chronological ordering needs timestamps; the caller
        # should sort by result timestamp. We emit unordered pairs here.
        for i in range(len(vhs)):
            for j in range(i + 1, len(vhs)):
                (v1, h1), (v2, h2) = vhs[i], vhs[j]
                if h1 != h2:
                    drifts.append({
                        "type": "CODEGEN_DRIFT",
                        "config": cfg,
                        "v1": v1, "v2": v2,
                        "hash1": h1, "hash2": h2,
                    })
    return drifts


def freshness_for(candidate, measured_version, measured_hash,
                  target_version, target_hash=None,
                  csv_measured=None, csv_target=None,
                  annotations=None):
    """Compute freshness of one candidate's measurement for a target version.

    If target_hash is known (we've built the target version's kernel):
        - hash match -> 1.0
        - else classify the change.
    If target_hash is unknown (predicting for an unbuilt version):
        - we can't know if the code changed; use the most recent known
          classification, or 0.1 (unknown) if never classified.
    """
    if target_hash is not None:
        cls = classify(candidate, measured_version, target_version,
                       measured_hash, target_hash,
                       csv_measured, csv_target,
                       _annotations=annotations)
        return cls  # dict with level, weight, source, reason
    # Target not built: conservative
    return {"level": 4, "weight": 0.1, "source": "target_unbuilt",
            "reason": "target version's code not available; cannot verify identity"}


def pairwise_usable(fa, fb, threshold=USABILITY_THRESHOLD):
    """Usability rule: min(freshness) >= threshold."""
    return min(fa["weight"], fb["weight"]) >= threshold


def pairwise_weight(fa, fb, protocol_match=1.0, machine_trust=1.0):
    """Comparison weight for a usable pair."""
    return min(fa["weight"], fb["weight"]) * protocol_match * machine_trust


def decay_factor(num_relevant_changes, base=DECAY_BASE):
    """Recency decay driven by change count, not wall-clock time."""
    return base ** num_relevant_changes
