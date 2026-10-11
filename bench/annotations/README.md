# Manual change annotations

When a human understands a code change better than the automatic classifier
(`tools/bench/classify.py`), they can annotate the transition here.

## Format

`bench/annotations/<from_sha7>-<to_sha7>.json`:

```json
{
  "from_version": "abc1234",
  "to_version": "def5678",
  "annotator": "carlo",
  "date": "2026-10-15",
  "classifications": [
    {
      "config_pattern": "twin=xorbit,isa=x86",
      "level": 3,
      "reason": "xorbit restricted to non-x86; x86 xorbit candidates no longer generated"
    },
    {
      "config_pattern": "isa=avx512",
      "level": 1,
      "reason": "only twin reporting changed in ui_meta.c; codegen untouched (verified by hash)"
    }
  ],
  "default_level": 4
}
```

- `config_pattern`: comma-separated `key=value` pairs matched against candidate
  fields. `twin` matches against the candidate's `twins` list. All pairs must
  match. First matching pattern wins.
- `level`: 0–4 per `bench/STALENESS.md` §2.
- `default_level`: applied to configs matching no pattern (default: 4).

## Rules

1. Annotations override automatic classification. They are the highest-priority
   source.
2. Annotations must be **falsifiable**: the `reason` should say how the claim
   could be checked (e.g. "verified by hash", "disassembly diff shows only
   register renaming").
3. Annotations are versioned with the facility (committed). The aggregation
   report lists which annotations influenced each inference (`ANNOTATED` flag).
4. When in doubt, use level 4. A wrong level-1 annotation silently inflates
   trust; a wrong level-4 annotation only loses some signal.
5. Annotate **transitions**, not versions. A single annotation covers one
   (from, to) pair. Chains compose automatically.
