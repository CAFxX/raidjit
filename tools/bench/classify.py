#!/usr/bin/env python3
"""Change classification for the staleness model (bench/STALENESS.md §2).

Given two versions of the same config (identified by code_hash change),
classify the change into levels 0-4:

  0  Identical            weight 1.0  (hashes match; not really a change)
  1  Performance-neutral  weight 0.9
  2  Bounded-relevant     weight 0.5
  3  Structural           weight 0.2
  4  Unknown              weight 0.1  (conservative default; always flagged)

Classification sources, in priority order:
  1. Manual annotations (bench/annotations/) — highest priority.
  2. Disassembly diff (if both versions' disassembly stored).
  3. Config-space signal (a validity bump touching this config -> >= 3).
  4. Default: 4 (unknown). Never assume neutral.

This module is pure logic (no I/O except annotation loading). The aggregation
tool calls classify() per (config, v1, v2) triple.
"""

import json
import os
import re
import sys

LEVEL_WEIGHTS = {0: 1.0, 1: 0.9, 2: 0.5, 3: 0.2, 4: 0.1}
LEVEL_NAMES = {
    0: "identical",
    1: "performance-neutral",
    2: "bounded-relevant",
    3: "structural",
    4: "unknown",
}

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANNOTATIONS_DIR = os.path.join(REPO_ROOT, "bench", "annotations")


def _match_pattern(pattern, candidate):
    """pattern: 'key=value,key2=value2'. All pairs must match candidate fields.
    'twin' matches against the candidate's twins list."""
    for pair in pattern.split(","):
        pair = pair.strip()
        if "=" not in pair:
            continue
        k, v = pair.split("=", 1)
        k, v = k.strip(), v.strip()
        if k == "twin":
            if v not in candidate.get("twins", []):
                return False
        elif str(candidate.get(k, "")) != v:
            return False
    return True


def _load_annotations():
    """Load all manual annotations. Returns list of validated dicts.
    MAJOR-4 fix: validate level ranges at load; skip corrupt files loudly
    (to stderr) instead of silently ignoring or crashing later."""
    out = []
    if not os.path.isdir(ANNOTATIONS_DIR):
        return out
    for fn in sorted(os.listdir(ANNOTATIONS_DIR)):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(ANNOTATIONS_DIR, fn)
        try:
            with open(path, encoding="utf-8") as f:
                a = json.load(f)
        except (json.JSONDecodeError, OSError, ValueError) as e:
            print(f"warning: skipping corrupt annotation {fn}: {e}",
                  file=sys.stderr)
            continue
        # Validate structure: levels must be int 0..4
        ok = True
        for entry in a.get("classifications", []):
            lvl = entry.get("level")
            if not isinstance(lvl, int) or isinstance(lvl, bool) or not 0 <= lvl <= 4:
                print(f"warning: skipping annotation {fn}: invalid level {lvl!r} "
                      f"(must be int 0..4)", file=sys.stderr)
                ok = False
                break
        dl = a.get("default_level", 4)
        if not isinstance(dl, int) or isinstance(dl, bool) or not 0 <= dl <= 4:
            print(f"warning: skipping annotation {fn}: invalid default_level {dl!r}",
                  file=sys.stderr)
            ok = False
        if ok:
            out.append(a)
    return out


def _versions_match(a_from, a_to, v1, v2):
    """MAJOR-4 fix: empty versions never match (was: ''.startswith('') is True).
    Supports abbreviated SHA matching in both directions."""
    if not a_from or not a_to or not v1 or not v2:
        return False
    return ((v1.startswith(a_from) or a_from.startswith(v1)) and
            (v2.startswith(a_to) or a_to.startswith(v2)))


def _classify_manual(candidate, v1, v2, annotations):
    """Check manual annotations. Returns (level, reason) or None.

    MAJOR-4 fix: check ALL patterns across ALL version-matching annotations
    before falling back to defaults (was: first annotation's default won
    even if a later annotation had a matching pattern)."""
    # Phase 1: pattern matches across all version-matching annotations
    for a in annotations:
        if not _versions_match(a.get("from_version", ""), a.get("to_version", ""),
                               v1, v2):
            continue
        for entry in a.get("classifications", []):
            if _match_pattern(entry.get("config_pattern", ""), candidate):
                return (entry["level"],
                        f"manual annotation by {a.get('annotator', '?')}: "
                        f"{entry.get('reason', '')}")
    # Phase 2: defaults (first matching annotation's default wins)
    for a in annotations:
        if not _versions_match(a.get("from_version", ""), a.get("to_version", ""),
                               v1, v2):
            continue
        default = a.get("default_level", 4)
        return (default, f"manual annotation default (no pattern matched)")
    return None
    # NOTE: the old second loop below is superseded by _versions_match above.
def _classify_config_space(candidate, csv1, csv2):
    """If the config-space version bump touched this config's validity,
    the change is at least structural (level 3). Returns (level, reason) or None.
    Conservative: any bump without specific knowledge -> level 4 (unknown),
    not level 3, because we don't know if it touched this config."""
    if csv1 == csv2:
        return None
    # A bump happened. Without per-version restriction data we can't tell
    # whether it touched this candidate -> unknown (level 4), flagged.
    # When bench/config_space/vN.json gains machine-readable restrictions,
    # this becomes precise.
    return (4, f"config_space_version {csv1}->{csv2}; impact on this config unknown")


def classify(candidate, v1, v2, hash1, hash2, csv1=None, csv2=None,
             disasm1=None, disasm2=None, _annotations=None):
    """Classify the change for one candidate between two versions.

    Args:
        candidate: the candidate dict (twins, isa, etc.)
        v1, v2: raidjit version SHAs (strings)
        hash1, hash2: code_hash at v1, v2
        csv1, csv2: config_space_version at v1, v2 (optional)
        disasm1, disasm2: disassembly text or None (optional)
        _annotations: preloaded annotations (for batch use)

    Returns:
        dict(level=int, weight=float, source=str, reason=str)
    """
    if hash1 == hash2:
        return {"level": 0, "weight": 1.0, "source": "hash_match",
                "reason": "code hashes identical"}

    annotations = _annotations if _annotations is not None else _load_annotations()

    # 1. Manual annotations (highest priority)
    manual = _classify_manual(candidate, v1, v2, annotations)
    if manual:
        level, reason = manual
        return {"level": level, "weight": LEVEL_WEIGHTS[level],
                "source": "manual_annotation", "reason": reason}

    # 2. Disassembly diff (if available for both)
    if disasm1 is not None and disasm2 is not None:
        level = _diff_disasm(disasm1, disasm2)
        return {"level": level, "weight": LEVEL_WEIGHTS[level],
                "source": "disasm_diff",
                "reason": f"basic-block diff classified as {LEVEL_NAMES[level]}"}

    # 3. Config-space signal
    if csv1 is not None and csv2 is not None:
        cs = _classify_config_space(candidate, csv1, csv2)
        if cs:
            level, reason = cs
            return {"level": level, "weight": LEVEL_WEIGHTS[level],
                    "source": "config_space", "reason": reason}

    # 4. Default: unknown
    return {"level": 4, "weight": LEVEL_WEIGHTS[4], "source": "default",
            "reason": "no annotation, no disassembly, no config-space info; "
                     "conservative default"}


def _diff_disasm(d1, d2):
    """Coarse basic-block-level diff of two disassembly texts.
    Returns a classification level (1, 2, or 3).

    Heuristic:
      - Normalize: strip addresses, keep mnemonics + operand shapes.
      - Identical normalized text -> 1 (shouldn't happen; hashes differed,
        but maybe only addresses changed -> still performance-neutral).
      - Same mnemonic sequence, different operands -> 2.
      - Different mnemonic sequence -> 3.
    """
    def normalize(text):
        lines = []
        for line in text.splitlines():
            line = line.strip()
            # Strip address prefixes like "0x1234:" or "1234:"
            line = re.sub(r"^[0-9a-fA-Fx]+:\s*", "", line)
            # Strip comments
            line = re.sub(r"[;#].*$", "", line).strip()
            if line:
                lines.append(line)
        return lines

    n1, n2 = normalize(d1), normalize(d2)
    if n1 == n2:
        return 1
    # Compare mnemonic sequences (first token of each line)
    m1 = [l.split()[0] if l.split() else "" for l in n1]
    m2 = [l.split()[0] if l.split() else "" for l in n2]
    if m1 == m2:
        return 2
    return 3


def weight_for_level(level):
    return LEVEL_WEIGHTS.get(level, LEVEL_WEIGHTS[4])
