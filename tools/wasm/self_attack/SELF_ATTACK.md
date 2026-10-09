# S9.8 self-attack — WASM explorer (2026-10-09)

Harness: `tools/wasm/self_attack/attack.js` (Node, drives the actual
`tools/wasm/out/*.js` bundles exactly as the page's `compileKernel()` /
`disassemble()` wrappers do). Full log: `attack.log`.

Verdict rule per case: the module must NEVER trap (no exception escaping
`_rjw_compile` / `_dj_disasm`), must return well-formed JSON with
`ok:false` + a non-empty string `err` on bad input, and must still
compile a good document afterwards (state not corrupted).

## A. `_rjw_compile` adversarial option documents (29 cases + 29 recovery probes)

- Non-JSON: garbage, empty string, 1MB of `x`, 5000-deep `[` nesting —
  all clean `ok:false` ("invalid JSON"). No hang on the 1MB input.
- JSON but wrong shape: `null`, array, number, missing `v`, `v:0/2/"1"` —
  all rejected with schema-version or field errors.
- Unknown/missing `target`, unknown `ctor` ("raid99"), missing `topology` —
  clean errors.
- Malformed `matrix_hex`: non-hex chars, odd length — clean errors.
- Numeric abuse: `p1:100000`, `p1:-5`, `U:1000000`, `U:-3`,
  `prefetch_dist` huge, `failed:[999]` on recover — all clean errors
  (no OOM, no hang; the planner rejects out-of-range shapes).
- Bogus enums: `mul:"quantum"`, `isa:"bogus"`, unknown op kind,
  `nt:"yes"` (wrong type), `cpu` as string, `cpu` with unknown keys —
  clean errors.
- All 11 twin flags true at once (including pairwise-exclusive body
  twins): clean error via `feature_gates` validation (the S9.7 fix).
- `cand:null`, `options` as array, duplicate JSON keys, NUL/unicode in
  target string — clean errors.
- After EVERY attack, a good document still compiled `ok:true` —
  module state never corrupted.

**58/58 pass. No traps, no hangs, no state corruption.**

## B. Deep-link hash parser `readHash()` (20 cases, functions copied verbatim from the page)

- Empty/missing/malformed hashes, invalid base64, base64-of-non-JSON,
  JSON array/null/string, wrong pane shape, `isa` not a string,
  `ctor` not a string, unknown key `c=`, missing `=`, double `=`,
  control characters, second pane invalid, extra unknown fields,
  `isa` as object — all ignored safely, no throw.
- `{"__proto__":{"polluted":1}}` pane: ignored by shape validation;
  verified `({}).polluted === undefined` afterwards — no prototype pollution.
- 5MB hash: parsed without throw (returns empty; shape validation fails).
- Valid pane round-trips: `a` and `b` both load correctly.

**22/22 pass. No throws, no pollution.**

## C. `_dj_disasm` adversarial inputs (12 cases)

- Empty byte arrays on all three arches: clean (empty array or NULL).
- 256 deterministic-random bytes per arch, single `0xff` byte,
  4KB all-zeros, 4KB all-`0xff`: no trap; Capstone decodes what it can.
- Bogus arch name, empty arch name, 1MB arch name: clean (no trap).

**12/12 pass. No traps.**

## Total: 102/102 pass, 0 failures. No findings requiring code changes.
