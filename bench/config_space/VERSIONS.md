# Config-space versions

The **config space** is the set of (twin × ISA × plan-kind × option) combinations
the library considers legal. It changes over time: twins are added, restricted,
or retired; ISAs gain extensions; gates are fixed.

Each benchmark result tags its candidates with the `config_space_version`
under which they were legal. This file is the registry: what changed, when,
and what it means for old data.

## Version history

### v1 (2026-10-11, initial)
- **raidjit_version**: (initial facility commit)
- **Twins** (12): nt, ternlog, remat, notail, predtail, maskedtail, permi2b,
  xorsched, xorbit, mshift, pipe, rec_cf
- **ISAs**: sse, avx2, avx512, neon, sve, rvv
- **Notable rules**:
  - xorbit: legal on **all** ISAs (x86 restriction not yet applied)
  - ternlog: AVX-512 only
  - permi2b/mshift: x86 only
  - predtail: SVE only; maskedtail: AVX-512 or RVV
  - rec_cf: P_MULACC recover only

### v2 (TEMPLATE — copy for the next bump)
- **raidjit_version**: <sha>
- **Changes**: <what changed, e.g. "xorbit restricted to NEON/SVE/RVV; x86+xorbit now rejected">
- **Retired**: <list of (twin/ISA/option) that were legal in v1 but not v2>
- **Migration note**: <how old v1 results referencing retired configs should be interpreted>

## Rules for bumping

Bump `config_space_version` (and add a section above) when **any** of these change:
1. A twin is added, removed, renamed, or has its meaning changed.
2. A twin's ISA/plan-kind gating changes (e.g. xorbit x86 restriction).
3. An ISA is added or gains/loses a relevant extension gate.
4. A backend starts rejecting (or accepting) a previously accepted (or rejected) combination.

Do **not** bump for:
- Pure codegen changes that don't affect legality (the `code_hash` captures those).
- Static-model changes (that's `static_model_version`).
- Benchmark protocol changes (that's `bench_protocol_version`).

## Query semantics

- "Show me all xorbit results" → must specify or group by `config_space_version`.
  Results from v1 (xorbit-on-x86 legal) and v2 (xorbit x86-restricted) are
  **different populations** and must never be silently merged.
- A candidate tagged v1 referencing a combination illegal in v2 is marked
  **retired config** in reports — valid historical data, clearly labeled.
- Aggregation across a config-space break for the affected axis is forbidden;
  the tool must either filter to one version or report per-version.

## Deriving from raidjit SHA

The `raidjit_version` (git SHA) in each result is the ground truth: the
config space for any SHA can be derived by checking out that tree and
querying the library's own validation (`rj_ui_twin_valid`, `planid_cand_valid`).
The `config_space_version` integer is a **cache** of that derivation for
offline queries — it must always agree with what the SHA's library would say.
If they disagree, the SHA wins and the version tag is treated as corrupt
(quarantine the candidate).
