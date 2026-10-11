# LRC topologies (A6 step 1)

Locally recoverable codes, supported as a topology option (approved
2026-10-06). An LRC has **k** data disks, **h** global parities, and one
**local parity** per local group (**g = ceil(k/r)** groups of size ≤ r,
as equal as possible). A single failure inside a group recovers from
its group-mates with a pure-XOR equation; up to **h** failures anywhere
recover from the global parities. The point of the family is repair
cost: local recovery reads **r** disks instead of **k**.

## Constructors ([`src/topo.c`](../src/topo.c), declared in [`include/raidjit.h`](../include/raidjit.h))

| Constructor | Meaning |
|---|---|
| `rj_topo_lrc(k, h, r)` | General local-parity mechanism: k data, h global parities, locality r. `g = ceil(k/r)` groups, one XOR local parity each; npar = h+g. |
| `rj_topo_lrc_azure(void)` | The canonical Azure (12,2,2): 12 data, 2 global parities, 2 local parities, locality 6. |
| `rj_topo_lrc_optimal(k, h, r)` | Distance-optimal parameterization: like `rj_topo_lrc` but requires **r \| k** (NULL otherwise), under which the code attains the Singleton-type bound d = h+2 (§4). |

Matrix row order: the h global rows first (rows 0..h−1), then the g
local rows (rows h..h+g−1). All three return NULL on bad parameters
(k<2, h<1, h>50, r<2, k>51, `r ∤ k` for the optimal form, k+h+g >
`RJ_MAX_DISKS`) or OOM.

## Matrix design decisions

- **Global rows are Vandermonde with exponents 1..h** (row j:
  coefficient 2^((j+1)·i) on data disk i). Exponent 0 is deliberately
  skipped: the all-ones row would duplicate the local XOR row on any
  single-group column set, collapsing the "(h+1) unknowns in one group"
  solve from full rank.
- **The globals are NOT the `matrix_search_default` matrices**
  `rj_topo_rs` uses. Weight-minimal matrices carry no guarantee that
  the global rows compose with the all-ones local rows into full-rank
  joint systems, which is exactly what distance-optimality needs
  (every (h+1)-erasure pattern recoverable). The Vandermonde global
  part does compose — verified exhaustively per instance (§4).
- **Local rows are pure XOR** (0/1). This is the Azure choice and the
  whole point of LRC: local repair costs no multiplies at all.
- **k ≤ 51, h ≤ 50 envelope.** 2 has order 51 in GF(2^8)/0x11B
  (verified computationally), so the Vandermonde nodes 2^0..2^50 are
  distinct — the same envelope the header documents for the default
  RS matrix. The order-51 fact bounds the exponents too: at h = 51 the
  global row with exponent 51 has entries 2^(i·51) = 1 for every
  column — the all-ones row the design deliberately skips as exponent
  0 (it would collapse the "(h+1) unknowns in one group" solve from
  full rank), and at h ≥ 52 the global rows repeat with period 51.
  `lrc_new` rejects h > 50 for exactly this reason.

## Local vs global recovery: the existing machinery already picks local

No planner/survivor change was needed (Step 2 verdict: **not needed**).
For a recover op the A5 survivor path enumerates parity-row subsets and
scores each with the **exact static signal** — the canonical version's
real `(uops_per_byte, mem_per_byte)` from `jit_build_version`, no
duplicated model. A single failure in a group offers the h global-row
subsets (GF-multiply plans over ~k streams) against the one local-row
subset (pure-XOR plan over r streams). The local subset strictly
dominates on both axes, so the pareto shortlist keeps it and the
static pick (`heuristic_select`, benchmarking off) prefers it. The
measured pick is expected to prefer it as well, but benchmarks are
postponed per the A6 brief, so that preference is unverified. The
`is_today` anchor (a global row) stays shortlisted
as the no-regression anchor; it only loses genuinely.

Static cost (canonical version, single failure of disk 0; from
[`tests/test_lrc.c`](../tests/test_lrc.c), benchmark-off — benchmarks are postponed per the
A6 brief, so these are static-model numbers, not measurements):

`lrc-azure(12,2,2)`:

| subset rows | plan kind | input cols | uops/byte | mem byte/byte |
|---|---|---|---|---|
| {0} global | mulacc | 12 | 0.5977 | 14.48 |
| {1} global | mulacc | 12 | 0.5977 | 14.48 |
| {2} local  | **xor** | 6 | **0.1113** | **7.02** |

`lrc-opt(20,2,4)`:

| subset rows | plan kind | input cols | uops/byte | mem byte/byte |
|---|---|---|---|---|
| {0} global | mulacc | 20 | 1.0039 | 23.61 |
| {1} global | mulacc | 20 | 1.0039 | 23.61 |
| {2} local  | **xor** | 4 | **0.0801** | **5.02** |

The test pins the mechanism end to end: `survivor_select` (static
path) reports `win_rows == {local row}`; the `rj_compile` plan ID names
it (`...:rec0.rows2:...`); one failure per group wins both local rows
(`rec0+6.rows2+3`); two failures in one group win local+global; and a
tampered topology with the local row zeroed falls back to a global row
while staying bit-exact (proving the local pick is caused by the local
row, not the test scaffolding). Under the `RJ_SUBSET_PRUNE_INVERT`
test hook the local-pick assertions fail loudly (red-green).

## Distance-optimality

When r | k the Singleton-type bound for (n,k) LRCs with locality r is
d ≤ n−k−⌈k/r⌉+2 = h+2. The construction attains it. Verified
exhaustively per instance with `survivor_enumerate` as the
recoverability oracle (all (h+1)-erasure patterns recoverable; a
concrete (h+2)-pattern not recoverable, hence d = h+2 exactly):

| instance | n | (h+1)-patterns checked | (h+2) witness |
|---|---|---|---|
| lrc-azure(12,2,2) | 16 | 560 | (0,1,2,3) |
| lrc(8,2,4) | 12 | 220 | (0,1,2,3) |
| lrc-opt(9,3,3) | 15 | 1365 | (0,1,2,9,10) |

`rj_topo_lrc_optimal` enforces r | k (the bound-tight case) at the API
level; `rj_topo_lrc` with r ∤ k is still a valid LRC, just not
bound-tight. New parameter choices should extend the table above, not
assume the bound.

## Failure capability (azure (12,2,2))

- 1 failure in a group → local XOR recovery (6 streams).
- 1 failure per group across groups → per-group local recovery.
- 2 failures in one group → local + 1 global row.
- 3 failures in one group → local + 2 global rows.
- 4 failures in one group → **not recoverable** (compile fails loudly;
  only 3 equations cover the group).
- Failed parity disks (global or local) → recomputed from survivors.
