# Persistent autotune DB ("wisdom") — design

**Status:** DESIGN ONLY — no implementation. Item #29 (PARKED), strongest
promotion candidate from the SOTA batch. Implementation requires Carlo's
decision; the open questions in §12 are the decision points.

**Scope note:** this doc designs the persistent store only. It does not
change the planner, the benchmark protocol, or any codegen. All
performance effects are described qualitatively — no timings are claimed.

---

## 1. Background and motivation

1. `rj_compile` empirically selects kernels at plan time: it enumerates
   candidates (`plan_candidates`, src/plan.c), JIT-builds every one,
   prunes statically dominated twins, and picks the winner with
   `bench_select` (wall-clock timing over several working-set sizes,
   src/bench.c). Stage-2 twins (`notail`, `predtail`, `maskedtail`,
   `permi2b`) are then duelled off the winner (src/api.c).
2. Every `rj_compile` call on the same machine repeats this work, even
   for identical (topology, op) inputs. The winner is a pure function of
   (machine, plan, benchmark regime) — it does not depend on process
   state. Re-measuring it per process is wasted work.
3. FFTW solved the analogous problem with *wisdom*: a persistent,
   per-machine database of planner decisions, with import/export/merge
   primitives. This doc adapts that model to raidjit.

## 2. Non-goals

1. **Not a replacement for empirical selection.** The DB is a cache of
   decisions, never a substitute for the benchmark. Every miss, every
   mismatch, and every explicitly requested fresh run goes through the
   full `bench_select` protocol unchanged.
2. **Not a code cache.** The DB stores *which candidate won*, never
   machine code. JIT emission always runs fresh in-process; there is no
   deserialization of executable bytes, ever.
3. **Not a tuning-parameter store.** `bench_bytes`, `bench_iters`,
   `allow_nt`, `prefetch_dist` remain caller options. The DB records the
   regime a decision was measured under and refuses to apply decisions
   measured under a different regime (§7.1).
4. **Not a cross-machine oracle by default.** Entries are keyed by full
   machine fingerprint. Sharing wisdom across machines is an explicit,
   separate operation (§7.4), default off.
5. **No learning, no model.** The DB is a lookup table, not a predictor.
   A cost-model-driven selector is a different project.
6. **No change to the static (non-benchmark) path's selection logic.**
   The heuristic path may *consult* the DB (see OQ-5) but never *writes*
   heuristic picks into it — only `bench_select` measurements are
   recorded.

## 3. Terminology

1. **Fingerprint** — canonical string identifying a machine for tuning
   purposes (§4), plus its hash.
2. **Plan key** — canonical string identifying *what* was tuned:
   `(kind, n_in, n_out, nchanged, coefficient-hash, option-bits)`.
   Two compiles with equal plan keys on equal fingerprints must select
   the same winner, or the key is under-specified (a bug).
3. **Candidate spec** — the winning `cand_t` serialized axis by axis
   (isa, mul, U, nt, prefetch_dist, ternlog, remat, xorsched, lmul),
   plus the stage-2 twin decisions. Never a pointer, never code.
4. **Wisdom entry** — `(fingerprint, plan key, candidate spec, twin
   decisions, measurement provenance)`. The unit of storage.
5. **Wisdom** — the on-disk collection of entries for this user.

## 4. Machine fingerprint

### 4.1 Included signals

The fingerprint must capture everything that can change the benchmark
winner. It is built from the same `rj_cpu_info` the planner already
consumes (include/raidjit.h), plus build identity:

1. **CPU identity:** vendor string, family/model/stepping (CPUID leaf 1
   EAX on x86-64; MIDR_EL1 on AArch64; `marchid`/`mimpid` on RISC-V),
   normalized microarchitecture name when known, and the brand string.
2. **Feature flags:** the complete `rj_cpu_info` flag set —
   `avx512f/bw/vl/vbmi/gfni`, `sve/sve2`, `rvv_zvbc32e`, `rvv_zicbop`,
   etc. These change the candidate set (`plan_candidates` gates on
   them), so any flag flip must invalidate.
3. **Vector length:** `sve_vl` (SVE, runtime-determined) and `rvv_vlen`.
   Fixed-VL ISAs contribute their `isa_vl` implicitly via the ISA name.
4. **Library build identity:** raidjit version + build id (git hash when
   available). Codegen changes must invalidate — this is the primary
   library-upgrade invalidation signal (§6).
5. **OS family** (`linux`/`windows`/`macos`) — the bench harness and
   memory-mapping behavior differ per OS. Not the kernel release (see
   §4.2).

### 4.2 Deliberately excluded signals (with rationale)

1. **Kernel release / patchlevel.** The winner is dominated by
   microarchitecture and codegen, not by the kernel. Including it would
   expire the DB on every routine update and defeat the purpose.
2. **CPU frequency, turbo/DVFS state, governor.** Not stable across
   runs; the benchmark's interleaved rounds already average over this.
   Baking it into the key would fragment the DB without improving
   decisions.
3. **Core count / memory size / bandwidth class.** `bench_select`
   measures single-threaded, mostly L1/L2-resident kernels; the
   candidate *ranking* is insensitive to these. (If a future
   multi-threaded benchmark path appears, revisit.)
4. **Compiler version that built libraidjit.** The JIT emits machine
   code directly; the C compiler only affects the harness and dispatch
   overhead, second-order effects on ranking. The library build id (§4.1.4)
   is the coarser, sufficient signal. (OQ-3 revisits this.)
5. **Container/VM identity.** A virtualized CPUID is still
   self-consistent within the VM; the fingerprint is *descriptive*, not
   *authoritative* about bare metal. Document, don't special-case.

### 4.3 Canonical encoding, hashing, collisions, mismatches

1. The fingerprint is a canonical `key=value;` string with fixed field
   order, then hashed (FNV-1a-64 or SHA-256 truncated to 64 bits — an
   implementation detail; the string is the source of truth).
2. **Lookup compares the full canonical string, never just the hash.**
   Hash collisions must not cause cross-machine misattribution.
3. **Any fingerprint mismatch is a miss, never an error.** Entries whose
   fingerprint differs from the current machine are inert: skipped on
   lookup, eligible for garbage collection, never applied.
4. A DB file may therefore contain entries from several machines (e.g.
   NFS-shared `$HOME`, or after a CPU upgrade) — the file is namespaced
   per entry, not per file.

## 5. Storage

### 5.1 Entry schema

Each entry records:

1. Format version (u32) and entry length (u32, length-prefix framing).
2. Fingerprint: full canonical string + hash.
3. Plan key: `kind` (P_XOR/MULACC/HORNER/DELTA), `n_in`, `n_out`,
   `nchanged` (P_DELTA), SHA-256 (or FNV-1a-64) of the coefficient
   bytes (`mat` for MULACC/DELTA, `horner_c` for HORNER; empty for
   P_XOR), and the option bits that change enumeration
   (`allow_nt`, `prefetch_dist`).
4. Measurement regime: `bench_bytes`, `bench_iters` used. Lookup
   requires regime equality with current options (§7.1).
5. Candidate spec: all `cand_t` axes as explicit integers
   (isa, mul, U, nt, prefetch_dist, ternlog, remat, xorsched, lmul).
6. Stage-2 twin decisions: per twin kind (`notail`, `predtail`,
   `maskedtail`, `permi2b`) — installed/skipped/not-applicable — plus
   the measured GB/s that decided each duel.
7. Provenance: library build id, timestamp, measured winner GB/s and
   the `bench_select` noise window, static `uops_per_byte` /
   `mem_per_byte` / `code_size` of the winner.
8. Per-entry CRC32 covering fields 2–7. Any CRC failure → entry
   discarded (fail closed, §9).

### 5.2 File format

1. Binary, length-prefixed records: magic `"RJWD"`, u32 format
   version, u32 entry count, then entries (§5.1), then a file-level
   CRC32. Strict parser: any truncation, bad magic, unknown version, or
   CRC failure → the whole file is treated as empty (fail closed), never
   partially trusted.
2. Rationale for binary over text: fixed, auditable validation surface;
   no parser differential risk. A `rj_wisdom_dump` debug tool (future
   work, not this design) can render entries as text.
3. Rejected alternative (SQLite): a whole dependency for a lookup table
   with tens of entries; file locking + atomic rename is sufficient
   (§8). Recorded in Appendix B.

### 5.3 Location

1. `$XDG_CACHE_HOME/raidjit/wisdom.db`, falling back to
   `~/.cache/raidjit/wisdom.db` when `XDG_CACHE_HOME` is unset/empty.
   Rationale: it is a cache — loss is always recoverable by
   re-benchmarking; the XDG cache dir has exactly these semantics.
2. Override: `RJ_WISDOM_PATH` names an explicit file (used by tests and
   by fleet tooling that stages wisdom files).
3. Disable: `RJ_NO_WISDOM` (any non-empty value other than exactly
   `"0"`) disables both lookup and recording, following the existing
   `RJ_NO_PRUNE` / `RJ_NO_SURVIVOR_SELECT` convention. Disabling never
   changes selection semantics — it just forces the empirical path.
4. Portability note: on Windows use `%LOCALAPPDATA%\raidjit\wisdom.db`
   via the equivalent known-folder lookup.

### 5.4 Versioning

1. **Format version** in the file header: bump on any schema change;
   old readers reject new files outright (fail closed, §9).
2. **Library build id** per entry (§5.1.7): a codegen change produces a
   new build id, so old entries stop matching without any explicit
   migration. No migration logic is ever needed — staleness is by
   mismatch, not by upgrade script.
3. A reader must accept (skip, not fail on) *newer per-entry fields it
   does not understand* only within the same format version — achieved
   by the length-prefix framing. Unknown format version → reject file.

### 5.5 Size bounds and eviction

1. Hard caps checked *before* parsing: file size ≤ 4 MiB, entries ≤
   4096. Beyond either → treat as empty and (on next write) rewrite.
   Real-world usage is tens of entries; the caps are abuse guardrails.
2. On insert past the entry cap: evict least-recently-hit entries (each
   entry carries a last-hit timestamp, updated on lookup hit). LRU is
   simple and matches the access pattern (hot topologies stay).
3. No unbounded growth by construction: one entry per (fingerprint,
   plan key); re-recording the same key overwrites in place.

## 6. Invalidation

Entries go stale — i.e. are ignored, never applied — when:

1. **Library upgrade:** the entry's build id differs from the running
   library's. This is the main codegen-change signal; no version
   comparison logic, just string inequality.
2. **Fingerprint change:** any included signal flips (CPU swap, feature
   flag change, SVE VL change, OS family change). The entry remains in
   the file but never matches; a periodic GC pass (on write) drops
   entries whose fingerprint has not matched in N days.
3. **Regime change:** current `bench_bytes`/`bench_iters` differ from
   the entry's recorded regime → miss, re-benchmark, overwrite.
4. **Planner change that alters enumeration:** covered by (1), since the
   planner ships in the library. If `plan_candidates` ever becomes
   data-driven (external config), its config hash must join the build
   id — note for the future, not a current requirement.
5. **Explicit flush:** `rj_wisdom_forget()` API + `RJ_WISDOM_FLUSH=1`
   env (one-shot: flush on next `rj_compile`, then continue normally).
   Flush deletes the file (or truncates to an empty valid DB), never
   leaves a half-written file (§8 atomic write).
6. **Recorded candidate no longer enumerates:** if the stored candidate
   spec does not match any candidate `plan_candidates` produces today
   (e.g. an axis was removed), the hit is discarded and the full
   empirical path runs. This is the backstop for every case the
   fingerprint missed.

There is deliberately **no TTL-based expiry**: a correct entry does not
rot. Re-validation policy (periodic re-benchmarking of trusted entries)
is OQ-4.

## 7. Lookup semantics and precedence

### 7.1 Precedence ladder

For each `rj_compile`, in order:

1. **Explicit fresh run** (`RJ_NO_WISDOM` set, or a future
   `RJ_WISDOM_FRESH=1`, or the API flag): skip lookup entirely, run the
   full empirical path, and (unless disabled) record the result.
2. **DB hit** (fingerprint match + plan-key match + regime match +
   candidate still enumerates): build *only* the recorded winner
   candidate (+ recorded twin decisions), run correctness validation
   (`validate_versions` — the same gate `bench_select` applies first),
   skip all timing. On any validation failure → fall through to (3)
   and delete the entry (see §7.3).
3. **DB miss / hit rejected:** today's flow unchanged — enumerate,
   JIT-build all candidates, prune, `bench_select` (or
   `heuristic_select` on the static path), stage-2 duels — then record
   the outcome if the path was empirical (see §7.2).
4. **Static default:** the non-benchmark path (`o.benchmark == 0`)
   keeps its `heuristic_select` semantics. A DB hit on this path means
   "build only the recorded empirical winner and validate it" — cheaper
   than building every candidate for static scoring, and at least as
   good a pick. Whether the static path consults the DB at all is OQ-5.

Precedence is total and deterministic: exactly one of the above fires
per compile. There is no "DB says X but benchmark says Y" conflict by
construction — the DB is only consulted when no fresh measurement is
taken.

### 7.2 What gets recorded, and when

1. **Only the benchmark path writes.** `heuristic_select` outcomes are
   never recorded (non-goal 6): the DB stores measurements, not guesses.
2. **What is recorded:** the stage-1 winner's candidate spec + the
   stage-2 twin decisions (installed/skipped per twin kind with their
   duel GB/s) + provenance (§5.1.7). Recording happens once per
   `rj_compile`, after the dispatcher is installed.
3. **Overwrite, never duplicate:** the key is (fingerprint, plan key);
   re-recording replaces the entry and refreshes its timestamp.
4. **No recording on failure:** if compilation fails, nothing is
   written. A failed compile must not poison the DB.

### 7.3 Confidence and re-validation

1. **Correctness is re-validated on every hit** (`validate_versions`
   over the single rebuilt winner). This is non-negotiable and cheap:
   it is the same gate `bench_select` runs before timing. A DB entry can
   therefore never install a *wrong* kernel — at worst a *slower* one.
2. **Performance is trusted, not re-measured, on hit.** Re-measuring on
   every hit would defeat the purpose. The trust rests on: fingerprint
   equality (same machine, same codegen, same regime) + the backstop in
   §6.6.
3. **Suspect marking:** if a hit fails validation (case (2) in §7.1),
   the entry is deleted and the empirical path runs. Two consecutive suspect hits for the same
   key are a bug signal worth logging verbosely, not a normal event.
4. Periodic re-benchmarking of long-trusted entries (to catch silent
   perf drift, e.g. microcode updates) is OQ-4.

### 7.4 FFTW-style import / export / merge / forget

Public API surface (names provisional, C linkage, in a new
`wisdom.h` — no changes to `raidjit.h`'s core API):

1. `int rj_wisdom_export(const char *path)` — write the in-memory
   wisdom (this process's view) to `path`. Same format, same validation.
2. `int rj_wisdom_import(const char *path)` — read `path`, validate
   strictly (§9), merge into the in-memory wisdom. Entries whose
   fingerprint matches this machine are usable immediately; entries
   from other machines are kept inert (lookups always filter by
   fingerprint, §4.3.4) so the file round-trips without loss.
3. **Merge policy:** keyed overwrite by (fingerprint, plan key); on key
   collision the imported entry wins *iff* its timestamp is newer AND
   its build id equals the running library's — otherwise the existing
   entry stands. Never merge entries with an unknown format version.
4. `int rj_wisdom_forget(void)` — drop all entries (memory + disk).
   `int rj_wisdom_forget_key(...)` — drop a single plan key (useful
   when a user suspects one bad entry).
5. Fleet scenario (OQ-2): export on a canary machine, distribute the
   file, import on fleet machines with identical fingerprints. The
   fingerprint-equality rule makes this safe by construction — a file
   from a different SKU simply never matches.

## 8. Thread-safety and multi-process access

1. **In-process:** a single `pthread_mutex_t` guards the in-memory
   wisdom (parsed entries + dirty flag). Lookup parses the file once per
   process (lazily, on first `rj_compile` needing it); the parsed view
   is refreshed when the file's mtime changes. All public wisdom APIs
   take the mutex. No lock is held across JIT compilation or
   benchmarking — only across parse/lookup/insert of the table.
2. **Cross-process:** POSIX `flock()` on the DB file — `LOCK_SH` for
   read/lookup, `LOCK_EX` for write. The lock is held only for the
   duration of the read or the atomic replace, never across a benchmark.
3. **Atomic writes:** serialize to a temp file in the same directory
   (`wisdom.db.tmp.<pid>`), `fsync`, then `rename(2)` over the target.
   Readers never see a torn file; a crashed writer leaves at most an
   orphan temp file (cleaned on next write — or left; it is harmless).
4. **Lock ordering:** in-process mutex → file lock. Never the reverse.
   No nested `rj_compile` calls exist, so no re-entrancy concern; the
   mutex is non-recursive by design and an assert guards misuse in
   debug builds.
5. **Windows portability:** `LockFileEx` for the cross-process lock;
   `MoveFileEx(MOVEFILE_REPLACE_EXISTING)` for atomic replace. The
   design isolates these in two small platform functions.
6. **Stale-lock safety:** `flock` is kernel-managed — a crashed holder
   releases automatically. No lock files, no PID-based schemes, no
   timeouts.

## 9. Security

Threat model: the DB lives in the user's own cache dir, but it is
parsed by a library that JITs code — the parser is therefore
security-relevant. Additionally, imported wisdom files (§7.4) are
explicitly user-supplied and may come from third parties.

1. **Fail-closed parsing** (§5.2.1): bad magic, unknown version, size
   over cap, truncation, CRC mismatch → treat as empty. The parser never
   recovers partial entries.
2. **Data only, never code:** the DB contains candidate *specs*, not
   machine code. Nothing from the DB is ever executed, `mmap`ed as
   executable, or passed to a JIT emitter as bytes. The emitter rebuilds
   the kernel from the spec through the normal `plan_candidates` →
   backend path.
3. **Correctness gate on every hit** (§7.3.1): even a maliciously
   crafted entry can only select a *slower, still-correct* kernel,
   because the winner is re-validated before installation. The residual
   risk is performance manipulation, not wrong output.
4. **File permissions:** create with `0600` (and the directory `0700`
   if created). Do not rely on umask. On open, verify with `fstat`: it
   is a regular file (`O_NOFOLLOW` on open to defeat symlink swaps),
   owned by the effective UID. Refuse otherwise → treat as empty.
5. **Size caps before parse** (§5.5.1): prevents memory-exhaustion via a
   bloated file. All length fields are range-checked before allocation;
   no `malloc(len)` on an untrusted `len` without the cap check.
6. **No network, no setuid interaction:** the library never fetches
   wisdom over the network itself; fleet distribution (OQ-2) is an
   out-of-band operator action. The DB is never consulted in a setuid
   context — if `geteuid() != getuid()`, disable silently.
7. **Imported files get identical treatment:** `rj_wisdom_import`
   applies the same validation, caps, and fingerprint filtering as the
   default path. There is no "trusted import" mode.

## 10. Interaction with the planner, plan cache, and dispatch

### 10.1 `rj_compile` flow changes

Today: `plan_build` → `plan_candidates` → build all versions →
`prune_versions` → `bench_select`/`heuristic_select` → winner →
stage-2 twins → dispatcher. With wisdom, two insertion points:

1. **After `plan_build`, before candidate JIT builds:** compute the
   plan key (§3) and consult the DB. On hit: reconstruct the single
   winning `cand_t` from the stored spec (match it against a fresh
   `plan_candidates` enumeration by axis equality — this both validates
   the spec and reuses the existing candidate-construction code),
   JIT-build just that version, `validate_versions`, apply recorded
   twin decisions (build each installed twin from the winner as
   `api.c` does today, validate, skip the duels), install the
   dispatcher. Everything else — enumeration, pruning, benchmarking —
   is skipped.
2. **After selection, on the benchmark path only:** serialize the
   winner + twin decisions + provenance into an entry and merge into
   the in-memory wisdom; mark dirty. The file write happens lazily
   (on process exit via `atexit`, or on the next lookup — TBD in
   implementation), under `LOCK_EX` with atomic rename (§8).

No changes to `plan_build`, `plan_candidates`, `prune_versions`,
`bench_select`, or any backend. The DB is a layer *around* the planner,
not inside it.

### 10.2 Stage-2 twins and length dispatch

1. Twin decisions are part of the entry (§5.1.6) because they are
   empirical outcomes of the same compile. On a hit, twins are rebuilt
   from the winner exactly as `api.c` builds them today — only the
   *duels* are skipped, not the construction or validation.
2. The dispatcher's length-routing (`dispatch.c`: `TAIL_SCALAR` /
   `TAIL_MASKED` / `TAIL_NONE`, fastest-first ordering) is unaffected:
   it operates on validated, built variants regardless of how the
   winner was chosen.
3. If a recorded twin kind is not applicable today (e.g. `permi2b`
   recorded but the current run lacks `cpu->vbmi` — impossible under
   fingerprint equality, but defense in depth), that twin is dropped
   and the general variant covers its lengths, exactly as if the duel
   had rejected it.

### 10.3 Relation to existing caches; provenance reporting

1. **Matrix-search cache** (`matrix_search.c`, per `(ndata,npar)`
   in-process): orthogonal and untouched. It caches *which matrix* to
   use; wisdom caches *which kernel* won for a plan. They compose:
   matrix cache feeds `plan_build`, wisdom short-circuits selection.
2. **No cross-`rj_compile` in-process plan cache exists today** — each
   compile re-runs selection from scratch. The in-memory wisdom table
   (§8.1) incidentally provides this: a second compile of the same plan
   in one process hits the parsed table without touching disk.
3. **Provenance:** `rj_get_info` should report how the winner was
   chosen — proposed new `rj_info` fields: `wisdom_hit` (0/1),
   `wisdom_disabled` (0/1). This makes "did the DB fire?" observable in
   benchmarks and bug reports, and gives the perf-guard tooling a way
   to force the empirical path when validating.

## 11. Operational knobs

Following the existing `RJ_NO_PRUNE` convention (env name, "any
non-empty value other than exactly `0`" disables):

1. `RJ_NO_WISDOM` — disable lookup and recording entirely.
2. `RJ_WISDOM_PATH` — explicit DB file path (tests, fleet staging).
3. `RJ_WISDOM_FLUSH` — one-shot flush on next compile (see §6.5).
4. `RJ_WISDOM_VERBOSE` — log hits/misses/records to stderr (parallels
   the `o.verbose` compile logging).
5. Proposed API (provisional names, new `wisdom.h`): `rj_wisdom_export`,
   `rj_wisdom_import`, `rj_wisdom_forget`, `rj_wisdom_forget_key`,
   `rj_wisdom_stats` (entry count, hits/misses this process).
6. Default posture (lookup on / recording on / opt-out vs opt-in) is
   OQ-1.

## 12. Open questions for Carlo

- **OQ-1 — Default posture: opt-in or opt-out?** Arguments for
  opt-out (on by default): the DB is a pure cache — correctness is
  re-validated on every hit, loss is recoverable, and the win (no
  re-benchmarking across processes) accrues silently. Arguments for
  opt-in: any persistent state is a surprise in a library; benchmark
  harnesses and CI want hermetic runs; a stale-entry class of bug
  reports appears. Proposed default: **on, with `RJ_NO_WISDOM` opt-out**
  — but this is his call.
- **OQ-2 — Fleet / network sharing of wisdom?** The design makes
  same-fingerprint sharing safe by construction (§7.4.5), but does he
  want a supported workflow (export on canary → distribute → import on
  fleet), or should import stay a power-user escape hatch? Any
  *automatic* network fetch is out (non-goal, §9.6) — the question is
  only how first-class the manual flow should be.
- **OQ-3 — Fingerprint: include the compiler that built libraidjit?**
  Design says no (§4.2.4) — the build id covers codegen changes. Counter:
  two builds from the same source with different compilers/flags share
  a build id only if the id includes them; if the build id is just the
  git hash, `-O2` vs `-O3` libraidjit builds would share entries.
  Options: (a) keep git-hash-only build id and document "same flags"
  as an assumption; (b) fold compiler+flags into the build id.
- **OQ-4 — Re-validation cadence for trusted entries?** Design trusts
  a hit's performance indefinitely (correctness is always re-validated).
  Silent drift sources: microcode updates, kernel scheduler changes,
  firmware. Options: (a) trust indefinitely (simplest; drift only
  costs performance, never correctness); (b) re-benchmark entries
  older than N days in the background / on a slow path; (c) re-duel on
  every K-th hit. Any choice here is a policy knob, default (a).
- **OQ-5 — Should the static (non-benchmark) path consult the DB?**
  Design proposes yes for lookup (build-only-the-winner is cheaper and
  no worse than the heuristic pick) but never for recording. Alternative:
  keep the static path fully hermetic and DB-free. Note the static path
  exists for environments where benchmarking is impossible/undesired —
  consulting the DB there is arguably exactly the right fallback.
- **OQ-6 — Binary format, or human-readable text?** Design picks a
  strict binary format (§5.2) for a minimal validation surface, with a
  future dump tool for debuggability. Alternative: a text format
  (one entry per line, `key=value`) that operators can read and
  hand-edit — at the cost of a more complex parser and hand-edit
  footguns.
- **OQ-7 — Eviction policy.** Design picks LRU with a 4096-entry cap
  (§5.5). Real usage is tens of entries; is a cap + LRU over-engineering?
  Alternative: no cap (entries are ~200 bytes; even 10k entries is 2 MB),
  GC only fingerprint-stale entries.

## Appendix A — example entry (illustrative, not normative)

```
magic:            "RJWD"  fmt=1
fingerprint:      "os=linux;vendor=GenuineIntel;fam=6;model=143;step=8;
                   uarch=sapphirerapids;avx512f=1;avx512bw=1;avx512vl=1;
                   vbmi=1;gfni=1;sve_vl=0;rvv_vlen=0;build=3a4d232d"
                  hash=9f2c…e1
plan key:         kind=P_MULACC n_in=6 n_out=3 nchanged=0
                  coeff_hash=7b44…09 nt=0 prefetch=0
regime:           bench_bytes=1048576 bench_iters=11
winner:           isa=avx512 mul=gfni U=8 nt=0 prefetch=0 ternlog=1
                  remat=0 xorsched=0 lmul=1
twins:            notail=installed(41.2 GB/s) predtail=n/a
                  maskedtail=skipped(38.9 GB/s) permi2b=n/a
provenance:       build=3a4d232d ts=2026-10-06T06:30:00Z
                  gbps=40.8 noise=1.5% uops/B=0.31 mem/B=2.0 code=4812
crc32:            8a1f…c4
```

## Appendix B — rejected alternatives

1. **SQLite / embedded KV store.** A dependency and a write-ahead log
   for a table with tens of entries and a trivial access pattern.
   `flock` + atomic rename covers the concurrency requirement (§8).
2. **Storing compiled machine code.** Rejected absolutely: executing
   deserialized bytes breaks the "JIT emits fresh" invariant, explodes
   the validation surface (relocations, constant pools, W^X
   implications), and buys little — JIT emission is cheap relative to
   benchmarking.
3. **In-`bench_select` memoization keyed by candidate desc string.**
   Fragile: `desc` is human-readable, not canonical; the plan key +
   candidate-spec encoding (§5.1) is exact where `desc` is approximate.
4. **Kernel-level (sysfs/debugfs) store.** Requires privileges, is
   OS-specific, and outlives user intent. The XDG cache dir has the
   right lifetime and permission semantics.
5. **Daemon / service holding wisdom.** A process boundary for a
   lookup table adds failure modes (socket, lifecycle) for no gain;
   `flock` gives adequate multi-process safety.
