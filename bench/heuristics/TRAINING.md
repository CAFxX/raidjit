# Heuristic training: output format design

## Purpose

`tools/bench/rj_bench_train.py` (to be implemented) consumes the weighted
pairwise comparisons from `rj_bench_aggregate.py` and produces **heuristic
override tables** the planner can load. This document specifies the output
format — what the planner consumes — not the training algorithm.

## Design principles

1. **Data, not code.** The trainer outputs JSON tables. The planner reads them
   at startup (or links them in). No code generation, no recompilation to
   change heuristics.
2. **Overrides, not replacements.** The tables express *deltas* against the
   static model (`heuristic_select` in `bench.c`). Where the table is silent,
   the static model decides. This keeps the system working with zero training
   data (empty table = static model everywhere).
3. **Uncertainty is load-bearing.** Every override carries its confidence and
   provenance. The planner applies an override only if confidence exceeds a
   threshold (default 0.7, configurable). Low-confidence overrides are logged,
   not applied.
4. **Versioned and auditable.** Each table records exactly which benchmark data
   produced it (content hashes). Reproducible: re-running the trainer on the
   same inputs yields the same table.

## Output format: `bench/heuristics/<uarch>.json`

```json
{
  "format_version": "heuristics/1",
  "uarch": "zen4",
  "generated_at": "2026-10-20T12:00:00Z",
  "generated_by": "rj_bench_train.py",
  "target_raidjit_version": "abc1234...",
  "target_config_space_version": 2,
  "inputs": {
    "num_results": 47,
    "num_comparisons": 312,
    "total_weight": 245.7,
    "content_hashes": ["sha256:...", "..."]
  },
  "confidence_threshold": 0.7,
  "overrides": [
    {
      "id": "ovr-001",
      "match": {
        "topology": "rs(6,3)",
        "op": "encode",
        "isa": "avx512"
      },
      "prefer": {
        "twins": ["ternlog"],
        "over_twins": [[]],
        "reason": "ternlog beats baseline in 12/12 comparisons (weight 10.8)"
      },
      "confidence": 0.93,
      "evidence": {
        "wins": 12,
        "losses": 0,
        "ties": 0,
        "total_weight": 10.8,
        "sources": [
          {"raidjit_version": "abc1234", "weight": 9.0, "change_level": 0},
          {"raidjit_version": "def5678", "weight": 1.8, "change_level": 1,
           "note": "reporting-only change"}
        ],
        "flags": ["CROSS_VERSION"]
      }
    },
    {
      "id": "ovr-002",
      "match": {
        "topology": "rs(*)",
        "op": "recover",
        "isa": "rvv"
      },
      "avoid": {
        "twins": ["xorbit"],
        "reason": "xorbit never wins on RVV recover (0/8, weight 6.2); prefer nibble path"
      },
      "confidence": 0.81,
      "evidence": { "...": "..." }
    }
  ],
  "not_overridden": [
    {
      "match": {"topology": "raid5(*)", "op": "encode"},
      "reason": "insufficient evidence (total_weight 0.3 < minimum 1.0)"
    }
  ]
}
```

### Match semantics

- `match` fields support glob patterns (`*`) for topology names.
- All specified fields must match; unspecified fields match anything.
- More specific overrides win over general ones (longest match).
- `prefer.twins` vs `prefer.over_twins`: when both candidate sets are present,
  boost the preferred set's static score by a confidence-scaled margin.
- `avoid.twins`: when present, demote by a confidence-scaled margin.
- Overrides never *force* a selection — they adjust the static-model ranking.
  The final pick is still the argmax of (static_score + override_adjustment).
  This preserves the "no data → static model" fallback and prevents a bad
  table from breaking correctness (only performance).

### Confidence computation

```
confidence = total_weight / (total_weight + skepticism_prior)
```

- `skepticism_prior` = 2.0 (higher than the aggregation default: training
  needs stronger evidence to change planner behavior).
- An override is emitted only if `confidence >= confidence_threshold`
  (default 0.7) AND `total_weight >= min_weight` (default 1.0).
- Overrides below threshold go in `not_overridden` with the reason —
  visible, auditable, not silently dropped.

### Staleness in training

The trainer consumes the aggregation's weighted comparisons directly:
- Each comparison already carries `weight` (freshness × protocol × trust).
- Comparisons flagged `RETIRED_CONFIG` are excluded from training for the
  current config-space version (they're history, not guidance).
- Comparisons flagged `STALE_UNKNOWN` contribute at most 10% of their
  nominal weight (defense in depth: even if the classifier was optimistic).
- The trainer refuses to emit overrides when `CROSS_VERSION` data is the
  *only* source and total weight is low — it says so in `not_overridden`.

## Planner integration (future work, not implemented here)

1. At startup, the planner loads `bench/heuristics/<uarch>.json` for the
   detected uarch (falls back to `default.json`, then to empty).
2. `heuristic_select()` applies overrides as score adjustments before argmax.
3. An environment variable (`RJ_HEURISTICS=off`) disables overrides entirely.
4. The planner logs which overrides fired (for debugging and for feeding
   back into the facility: "override ovr-001 applied 1,247 times").

## What the trainer does NOT do

- It does not *run* benchmarks (that's out of scope; data arrives via import).
- It does not *predict* unmeasured configs (no extrapolation; the static model
  handles the unknown).
- It does not modify the static model (uop tables stay in `ports.c`; the
  trainer only learns *preferences* between candidates the static model
  already ranks).
- It does not auto-deploy: generating a table is explicit (`rj_bench_train.py`
  is run by a human), and the table is committed to the repo for review.

## Implementation sketch (for the future implementer)

```
1. Load aggregation report (JSON).
2. Filter to usable comparisons with weight >= min_comparison_weight.
3. Group by (match_key): topology pattern, op, isa.
4. Within each group, for each twin-pair (A, B):
     wins_A = sum(weight where A won and not tied)
     total  = sum(weight)
     if wins_A / total > dominance_threshold (0.8) and total >= min_weight:
         emit prefer override (A over B).
     similarly for avoid (A never wins).
5. Compute confidence per override; filter by threshold.
6. Write bench/heuristics/<uarch>.json with full evidence.
```

The dominance threshold (0.8) means we only override when the data is
decisive, not when it's 55/45. The static model is the prior; overrides need
strong evidence to move it.
