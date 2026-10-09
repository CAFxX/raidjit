#!/usr/bin/env python3
"""gen_live_kernels_page.py — build the WASM live kernel explorer page.

Reads the four SINGLE_FILE emscripten bundles from tools/wasm/out/
(raidjit_x86.js, raidjit_aarch64.js, raidjit_riscv.js, disasm.js) and
inlines them into a single self-contained raidjit_kernels.html that
works from file:// (no server, no fetch, no threads).

Usage: python3 tools/gen_live_kernels_page.py [out.html]
Default output: tools/wasm/out/raidjit_kernels.html
"""
import os
import sys
import html

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "wasm", "out")
MODULES = ["raidjit_x86.js", "raidjit_aarch64.js", "raidjit_riscv.js",
           "disasm.js"]

PAGE_JS = r"""
"use strict";
/* ---- module loading (MODULARIZE factories, lazy per ISA) ---- */
const MODULE_FACTORY = {
  x86:     () => RaidjitX86(),
  aarch64: () => RaidjitAArch64(),
  riscv:   () => RaidjitRiscv(),
};
const DISASM_FACTORY = () => Disasm();
const modCache = {};
const disasmCache = { mod: null };
async function getModule(isa) {
  if (!modCache[isa]) modCache[isa] = await MODULE_FACTORY[isa]();
  return modCache[isa];
}
async function getDisasm() {
  if (!disasmCache.mod) disasmCache.mod = await DISASM_FACTORY();
  return disasmCache.mod;
}

/* ---- rjw_compile wrapper ---- */
function compileKernel(mod, optsObj) {
  const json = JSON.stringify(optsObj);
  const p = mod.stringToNewUTF8(json);
  let rp;
  try { rp = mod._rjw_compile(p); }
  catch (e) { mod._free(p); return { ok: false, err: "trap: " + e.message }; }
  let res;
  try {
    res = JSON.parse(mod.UTF8ToString(rp));
  } catch (e) {
    mod._rjw_free_string(rp); mod._free(p);
    return { ok: false, err: "bad JSON from rjw_compile: " + e.message };
  }
  mod._rjw_free_string(rp); mod._free(p);
  if (!res || typeof res !== "object" || !res.ok) return res || { ok: false, err: "bad result" };
  const lenp = mod._malloc(8);
  const cp = mod._rjw_code_bytes(lenp);
  const clen = mod.getValue(lenp, "i32");
  const code = mod.HEAPU8.slice(cp, cp + clen);
  const dp = mod._rjw_const_bytes(lenp);
  const dlen = mod.getValue(lenp, "i32");
  const cnst = mod.HEAPU8.slice(dp, dp + dlen);
  mod._free(lenp);
  res.code = code; res.cnst = cnst;
  return res;
}

/* ---- disassembly ---- */
async function disassemble(arch, bytes) {
  let d;
  try { d = await getDisasm(); }
  catch (e) { return { error: "disasm module load failed: " + e.message }; }
  const p = d._malloc(bytes.length || 1);
  if (bytes.length) d.HEAPU8.set(bytes, p);
  // arch is const char*: allocate; base_addr is uint64_t: pass BigInt.
  const ap = d.stringToNewUTF8(arch);
  let rp;
  try { rp = d._dj_disasm(ap, p, bytes.length, 0n); }
  catch (e) { d._free(ap); d._free(p); return { error: "trap: " + e.message }; }
  if (!rp) { d._free(ap); d._free(p); return { error: "dj_disasm returned NULL" }; }
  let arr;
  try {
    arr = JSON.parse(d.UTF8ToString(rp));
  } catch (e) {
    d._dj_free_string(rp); d._free(ap); d._free(p);
    return { error: "bad JSON from dj_disasm: " + e.message };
  }
  d._dj_free_string(rp); d._free(ap); d._free(p);
  if (!Array.isArray(arr)) return { error: "dj_disasm did not return an array" };
  return { insns: arr };
}
/* objdump-style normalization (SPEC: byte equality is the invariant) */
function fmtHex(bytes) {
  return Array.from(bytes, b => b.toString(16).padStart(2, "0")).join("");
}
function renderDisasm(insns) {
  const tbl = el("table", { class: "disasm" });
  const head = el("tr");
  for (const h of ["offset", "bytes", "instruction"])
    head.appendChild(el("th", {}, h));
  tbl.appendChild(head);
  for (const e of insns) {
    const tr = el("tr");
    tr.appendChild(el("td", { class: "off" },
      e.addr.toString(16).padStart(4, "0")));
    tr.appendChild(el("td", { class: "bytes" }, e.bytes));
    tr.appendChild(el("td", { class: "insn" },
      e.mnemonic + (e.op_str ? " " + e.op_str : "")));
    tbl.appendChild(tr);
  }
  return tbl;
}

/* ---- option model ---- */
const TARGETS = {
  x86:     ["avx512-gfni", "avx2", "sse"],
  aarch64: ["sve2", "neon"],
  riscv:   ["rvv", "rvv-zvbc32e", "rvv-zicbop"],
};
const DISASM_ARCH = { x86: "x86_64", aarch64: "aarch64", riscv: "riscv64" };
const CTORS = {
  raid0: ["ndata"], raid1: [], raid5: ["ndata"], raid6: ["ndata"],
  raid6_rs: ["ndata"], rdp: ["p"], evenodd: ["p"], liberation: ["k"],
  rs: ["ndata", "npar"], lrc: ["k", "h", "r"], lrc_azure: [],
  lrc_optimal: ["k", "h", "r"], custom: ["ndata", "npar", "matrix_hex"],
};
/* Per-ctor default parameters. Every entry must satisfy the backend
 * validity envelope (rjw_api.c), so that switching constructors always
 * lands on a compilable configuration — never a rejected one.
 * rdp/evenodd need p prime >= 3 (p=5 -> 4 data disks, as in the archived
 * doc's rdp-4 sample); rs needs npar >= 1; lrc/lrc_optimal need
 * k>=2,h>=1,r>=2 (doc's lrc-8-2-2 / lrc-opt-20-2-4 samples); custom needs
 * a matrix_hex of exactly 2*ndata*npar hex chars (any bytes accepted). */
const CTOR_DEFAULTS = {
  raid0: { p1: 4 }, raid1: {}, raid5: { p1: 4 }, raid6: { p1: 4 },
  raid6_rs: { p1: 4 }, rdp: { p1: 5 }, evenodd: { p1: 5 }, liberation: { p1: 4 },
  rs: { p1: 4, p2: 2 }, lrc: { p1: 8, p2: 2, p3: 2 }, lrc_azure: {},
  lrc_optimal: { p1: 20, p2: 2, p3: 4 },
  custom: { p1: 6, p2: 2, matrix_hex: "010101020103010401050106" },
};
function ctorDefaults(ctor) {
  return Object.assign({ ctor: ctor }, CTOR_DEFAULTS[ctor] || { p1: 4 });
}
function isPrime(n) {
  n = n | 0;
  if (n < 2) return false;
  for (let i = 2; i * i <= n; i++) if (n % i === 0) return false;
  return true;
}
/* Repair a topology object so it satisfies the backend validity envelope
 * (rjw_api.c). Stale deep links — e.g. saved when the page defaulted p=4
 * for every ctor, or hand-crafted with an unknown ctor — self-heal to the
 * per-ctor defaults instead of showing a backend rejection for a
 * configuration the user never chose. */
function sanitizeTopology(t) {
  const d = ctorDefaults(t && t.ctor);
  if (!CTORS[d.ctor]) return ctorDefaults("raid5");
  if (!t || typeof t !== "object") return d;
  const p1 = t.p1 | 0, p2 = t.p2 | 0, p3 = t.p3 | 0;
  let ok;
  switch (d.ctor) {
    case "rdp": case "evenodd": ok = isPrime(p1); break;
    case "liberation": ok = p1 >= 2; break;
    case "rs": ok = p1 >= 1 && p2 >= 1; break;
    case "lrc": case "lrc_optimal": ok = p1 >= 2 && p2 >= 1 && p3 >= 2; break;
    case "custom":
      ok = p1 >= 1 && p2 >= 1 && typeof t.matrix_hex === "string" &&
           t.matrix_hex.length === 2 * p1 * p2;
      break;
    case "raid1": case "lrc_azure": ok = true; break;
    default: ok = p1 >= 1; break; /* raid0/raid5/raid6/raid6_rs */
  }
  if (!ok) return d;
  const out = { ctor: d.ctor };
  const par = CTORS[d.ctor];
  if (par[0] !== "matrix_hex") out.p1 = p1;
  if (par[1] && par[1] !== "matrix_hex") out.p2 = p2;
  if (par[2] && par[2] !== "matrix_hex") out.p3 = p3;
  if (par.includes("matrix_hex")) out.matrix_hex = t.matrix_hex;
  return out;
}
const TWINS = ["nt", "ternlog", "remat", "notail", "predtail", "maskedtail",
               "permi2b", "xorsched", "xorbit", "mshift", "pipe"];
/* pairwise-exclusive body twins */
const BODY_TWINS = ["xorsched", "xorbit", "permi2b", "mshift", "pipe", "remat"];
/* pairwise-exclusive tail twins (backend: "maskedtail and notail are disjoint",
 * predtail disjoint from notail) */
const TAIL_TWINS = ["notail", "predtail", "maskedtail"];

/* Plan-kind gates for twins. The strategy (plan kind) is cached per
 * (topology, operation); a probe compile with auto candidate populates it
 * when unknown. Only these plan-deterministic gates are greyed out.
 * s.lastPlan holds {strategy, xor_sched, n_out} from the last compile. */
const TWIN_PLAN_OK = {
  remat:    (p) => p.strategy === "horner",
  xorbit:   (p) => p.strategy === "mulacc" || p.strategy === "delta" ||
    p.strategy === "horner",
  xorsched: (p) => p.strategy === "xor" && p.xor_sched === 1,
  permi2b:  (p) => p.strategy === "mulacc" || p.strategy === "delta",
  mshift:   (p) => p.strategy === "mulacc" || p.strategy === "delta" ||
    p.strategy === "horner",
  pipe:     (p) => p.strategy === "mulacc" || p.strategy === "horner",
  ternlog:  (p) => (p.strategy === "xor" && p.n_out >= 2) ||
    p.strategy === "mulacc" || p.strategy === "delta",
};
const strategyCache = {};
function strategyKey(s) {
  return JSON.stringify([s.topology, s.op.kind]);
}
/* Probe the plan kind with an auto compile (cheap, compile-only). Returns
 * {strategy, xor_sched, n_out} or null on failure. */
async function probeStrategy(P) {
  const s = P.state;
  const key = strategyKey(s);
  if (strategyCache[key]) return strategyCache[key];
  let mod;
  try { mod = await getModule(s.isa); } catch (e) { return null; }
  const doc = buildDoc({ ...s, candAuto: true });
  const r = compileKernel(mod, doc);
  if (r.ok && r.strategy) {
    const plan = { strategy: r.strategy, xor_sched: r.plan_xor_sched | 0,
      n_out: r.plan_n_out | 0 };
    strategyCache[key] = plan;
    return plan;
  }
  return null;
}

/* Effective ISA for a pane: the pinned isaVer wins when set, otherwise the
 * target implies it. Used to grey out options that are ISA-invalid. */
function effIsa(s) {
  if (!s.candAuto && s.cand.isaVer) return s.cand.isaVer;
  const t = s.target;
  if (t === "avx512-gfni") return "avx512";
  if (t === "avx2") return "avx2";
  if (t === "sse") return "sse";
  if (t === "sve2") return "sve";
  if (t === "neon") return "neon";
  return "rvv";
}
/* ISA-deterministic twin gates (from src/plan.c enumeration and the per-ISA
 * backend asserts in src/.../jit.c). Only these are greyed out client-side;
 * plan-kind gates (P_XOR vs P_MULACC etc.) stay backend-validated. */
const TWIN_ISA_OK = {
  ternlog:    (isa) => isa === "avx512",
  permi2b:    (isa) => isa === "avx512",
  mshift:     (isa) => isa === "avx512",
  maskedtail: (isa) => isa === "avx512",
  pipe:       (isa) => isa === "neon" || isa === "sve" || isa === "rvv",
  predtail:   (isa) => isa === "sve",
};
/* ISA-deterministic mul gates. nibble is valid everywhere; gfni needs the
 * avx512-gfni target (cpu->gfni) at full AVX-512; zvbc needs rvv-zvbc32e. */
function mulIsaOk(m, s) {
  if (m === "gfni") return s.isa === "x86" && s.target === "avx512-gfni" &&
    effIsa(s) === "avx512";
  if (m === "zvbc") return s.isa === "riscv" && s.target === "rvv-zvbc32e";
  return true;
}
/* Uncheck twins / reset mul that the new ISA or target invalidates, so a
 * configuration change can never leave a selected-but-impossible option. */
function sanitizeIsaGated(s) {
  const isa = effIsa(s);
  for (const t of TWINS) {
    if (s.twins[t] && TWIN_ISA_OK[t] && !TWIN_ISA_OK[t](isa)) s.twins[t] = false;
  }
  if (s.cand.mul && s.cand.mul !== "auto" && !mulIsaOk(s.cand.mul, s)) {
    s.cand.mul = "auto";
  }
}
/* Uncheck twins that the plan kind invalidates (needs s.lastPlan). */
function sanitizePlanGated(s) {
  if (!s.lastPlan) return;
  for (const t of TWINS) {
    if (s.twins[t] && TWIN_PLAN_OK[t] && !TWIN_PLAN_OK[t](s.lastPlan)) {
      s.twins[t] = false;
    }
  }
}

function defaultPane() {
  return {
    isa: "x86", target: "avx512-gfni", cpu: {},
    topology: ctorDefaults("raid5"),
    op: { kind: "encode" },
    candAuto: true,
    cand: { mul: "auto", U: 0, lmul: 1, prefetch_dist: 0 },
    twins: {},
    lastPlan: null,
    options: { loop_align: 0, emit_endbr: false, emit_shadow_stack: false },
  };
}

/* Build the rjw_compile option document from pane state. */
function buildDoc(p) {
  const top = { ctor: p.topology.ctor };
  const par = CTORS[p.topology.ctor];
  ["p1", "p2", "p3"].forEach((k, i) => {
    if (i < par.length && par[i] !== "matrix_hex") top[k] = p.topology[k] | 0;
  });
  if (par.includes("matrix_hex")) top.matrix_hex = p.topology.matrix_hex || "";
  const op = { kind: p.op.kind };
  if (p.op.kind === "recover") {
    op.failed = parseIdxList(p.op.failed);
    if (!op.failed.length) throw new Error("recover needs a non-empty 'failed' disk list");
    if (p.op.rows) op.rows = parseIdxList(p.op.rows);
  } else if (p.op.kind === "parity_update") {
    op.changed = parseIdxList(p.op.changed);
    if (!op.changed.length) throw new Error("parity_update needs a non-empty 'changed' disk list");
  }
  let cand = "auto";
  if (!p.candAuto) {
    cand = { mul: p.cand.mul === "auto" ? undefined : p.cand.mul,
             U: p.cand.U | 0,
             /* lmul is RVV-only (backend rejects it elsewhere): only send
              * it for riscv; 0 means "default" to the backend. The lmul
              * selector is hidden for non-RVV ISAs, so sending the stored
              * default 1 would reject a configuration the user never chose. */
             lmul: p.isa === "riscv" ? (p.cand.lmul | 0) : 0,
             prefetch_dist: p.cand.prefetch_dist | 0 };
    for (const t of TWINS) cand[t] = !!p.twins[t];
    if (p.cand.isaVer) cand.isa = p.cand.isaVer;
  }
  const doc = { v: 1, target: p.target, topology: top, op: op, cand: cand,
                options: p.options };
  if (Object.keys(p.cpu).length) doc.cpu = p.cpu;
  return doc;
}
function parseIdxList(s) {
  return (s || "").split(/[, ]+/).filter(x => x !== "").map(x => parseInt(x, 10))
    .filter(x => Number.isInteger(x) && x >= 0);
}

/* ---- deep links: #a=<b64url(JSON)>[&b=<b64url(JSON)>] ---- */
function b64urlEncode(s) {
  return btoa(unescape(encodeURIComponent(s)))
    .replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}
function b64urlDecode(s) {
  s = s.replace(/-/g, "+").replace(/_/g, "/");
  while (s.length % 4) s += "=";
  return decodeURIComponent(escape(atob(s)));
}
/* Normalize a deep-linked pane state: fill every sub-object from the
 * defaults so a partially-specified link can never crash page init
 * (buildDoc/buildPane assume complete state), and repair the topology
 * against the backend validity envelope. Unknown ISA/ctor/op fall back
 * to defaults. */
const OP_KINDS = ["encode", "recover", "parity_update"];
function normalizeState(o) {
  const d = defaultPane();
  if (!o || typeof o !== "object" || !TARGETS[o.isa]) return d;
  const target = TARGETS[o.isa].includes(o.target) ? o.target : TARGETS[o.isa][0];
  const op = (o.op && typeof o.op === "object") ? o.op : {};
  return {
    isa: o.isa,
    target: target,
    cpu: (o.cpu && typeof o.cpu === "object") ? o.cpu : {},
    topology: sanitizeTopology(o.topology),
    op: {
      kind: OP_KINDS.includes(op.kind) ? op.kind : "encode",
      failed: op.failed, rows: op.rows, changed: op.changed,
    },
    candAuto: o.candAuto !== false,
    cand: Object.assign({}, d.cand, (o.cand && typeof o.cand === "object") ? o.cand : {}),
    twins: (o.twins && typeof o.twins === "object") ? o.twins : {},
    options: Object.assign({}, d.options, (o.options && typeof o.options === "object") ? o.options : {}),
  };
}
function readHash() {
  const h = location.hash.replace(/^#/, "");
  const out = {};
  for (const part of h.split("&")) {
    const [k, v] = part.split("=");
    if ((k === "a" || k === "b") && v) {
      try {
        out[k] = normalizeState(JSON.parse(b64urlDecode(v)));
      } catch (e) { /* ignore malformed links */ }
    }
  }
  return out;
}
function writeHash(panes) {
  const parts = ["a=" + b64urlEncode(JSON.stringify(panes[0]))];
  if (panes.length > 1) parts.push("b=" + b64urlEncode(JSON.stringify(panes[1])));
  history.replaceState(null, "", "#" + parts.join("&"));
}

/* ---- UI ---- */
function el(tag, attrs, text) {
  const e = document.createElement(tag);
  if (attrs) for (const k in attrs) {
    if (k === "class") e.className = attrs[k]; else e.setAttribute(k, attrs[k]);
  }
  if (text !== undefined) e.textContent = text;
  return e;
}
function select(opts, val, isDisabled) {
  const s = el("select");
  for (const o of opts) {
    const op = el("option", { value: o }, o);
    if (o === val) op.selected = true;
    if (isDisabled && isDisabled(o)) op.disabled = true;
    s.appendChild(op);
  }
  return s;
}

const panes = [];   // {state, root, ...}
let abMode = false;

function buildPane(state) {
  const root = el("div", { class: "pane" });
  const P = { state: state, root: root };
  /* ISA + target */
  const row1 = el("div", { class: "row" });
  row1.appendChild(el("label", {}, "ISA "));
  P.isaSel = select(["x86", "aarch64", "riscv"], state.isa);
  row1.appendChild(P.isaSel);
  row1.appendChild(el("label", {}, " Target "));
  P.targetSel = select(TARGETS[state.isa], state.target);
  row1.appendChild(P.targetSel);
  root.appendChild(row1);
  /* topology */
  const rowT = el("div", { class: "row" });
  rowT.appendChild(el("label", {}, "Topology "));
  P.ctorSel = select(Object.keys(CTORS), state.topology.ctor);
  rowT.appendChild(P.ctorSel);
  P.topoParams = el("span");
  rowT.appendChild(P.topoParams);
  root.appendChild(rowT);
  /* op */
  const rowO = el("div", { class: "row" });
  rowO.appendChild(el("label", {}, "Operation "));
  P.opSel = select(["encode", "recover", "parity_update"], state.op.kind);
  rowO.appendChild(P.opSel);
  P.opParams = el("span");
  rowO.appendChild(P.opParams);
  root.appendChild(rowO);
  /* candidate */
  const rowC = el("div", { class: "row" });
  P.autoChk = el("input", { type: "checkbox" });
  P.autoChk.checked = state.candAuto;
  rowC.appendChild(P.autoChk);
  rowC.appendChild(el("label", {}, " auto candidate (heuristic)"));
  root.appendChild(rowC);
  P.candBox = el("div", { class: "cand" });
  root.appendChild(P.candBox);
  /* twins */
  P.twinBox = el("div", { class: "twins" });
  root.appendChild(P.twinBox);
  /* output */
  P.status = el("div", { class: "status" });
  root.appendChild(P.status);
  P.cfgTable = el("table", { class: "cfg" });
  root.appendChild(P.cfgTable);
  P.disasm = el("div", { class: "disasm" });
  root.appendChild(P.disasm);

  P.isaSel.onchange = () => {
    state.isa = P.isaSel.value;
    state.target = TARGETS[state.isa][0];
    sanitizeIsaGated(state);
    refreshPane(P); scheduleCompile(P);
  };
  P.targetSel.onchange = () => { state.target = P.targetSel.value;
    sanitizeIsaGated(state);
    refreshPane(P); scheduleCompile(P); };
  P.ctorSel.onchange = () => { state.topology = ctorDefaults(P.ctorSel.value);
    state.lastPlan = null;
    refreshPane(P); scheduleCompile(P);
    probeStrategy(P).then(pl => { if (pl) { state.lastPlan = pl;
      sanitizePlanGated(state); refreshPane(P); } }); };
  P.opSel.onchange = () => { state.op = { kind: P.opSel.value };
    state.lastPlan = null;
    refreshPane(P); scheduleCompile(P);
    probeStrategy(P).then(pl => { if (pl) { state.lastPlan = pl;
      sanitizePlanGated(state); refreshPane(P); } }); };
  P.autoChk.onchange = () => { state.candAuto = P.autoChk.checked;
    refreshPane(P); scheduleCompile(P);
    if (!state.candAuto && !state.lastPlan) {
      probeStrategy(P).then(pl => { if (pl) { state.lastPlan = pl;
        sanitizePlanGated(state); refreshPane(P); } });
    } };
  refreshPane(P);
  return P;
}

function refreshPane(P) {
  const s = P.state;
  /* target options follow ISA */
  P.targetSel.innerHTML = "";
  for (const t of TARGETS[s.isa]) {
    const o = el("option", { value: t }, t);
    if (t === s.target) o.selected = true;
    P.targetSel.appendChild(o);
  }
  /* topology params */
  P.topoParams.innerHTML = "";
  const par = CTORS[s.topology.ctor];
  ["p1", "p2", "p3"].forEach((k, i) => {
    if (i < par.length && par[i] !== "matrix_hex") {
      P.topoParams.appendChild(el("label", {}, " " + par[i] + " "));
      const inp = el("input", { type: "number", value: s.topology[k] || 0,
        min: 0, style: "width:5em" });
      inp.onchange = () => { s.topology[k] = parseInt(inp.value, 10) || 0;
        scheduleCompile(P); };
      P.topoParams.appendChild(inp);
    }
  });
  if (par.includes("matrix_hex")) {
    P.topoParams.appendChild(el("label", {}, " matrix_hex "));
    const ta = el("textarea", { rows: 2, cols: 40 });
    ta.value = s.topology.matrix_hex || "";
    ta.onchange = () => { s.topology.matrix_hex = ta.value.trim();
      scheduleCompile(P); };
    P.topoParams.appendChild(ta);
  }
  /* op params */
  P.opParams.innerHTML = "";
  if (s.op.kind === "recover") {
    P.opParams.appendChild(el("label", {}, " failed "));
    const fi = el("input", { value: s.op.failed || "0", style: "width:8em" });
    fi.onchange = () => { s.op.failed = fi.value; scheduleCompile(P); };
    P.opParams.appendChild(fi);
    P.opParams.appendChild(el("label", {}, " rows (opt) "));
    const ri = el("input", { value: s.op.rows || "", style: "width:8em" });
    ri.onchange = () => { s.op.rows = ri.value; scheduleCompile(P); };
    P.opParams.appendChild(ri);
  } else if (s.op.kind === "parity_update") {
    P.opParams.appendChild(el("label", {}, " changed "));
    const ci = el("input", { value: s.op.changed || "0", style: "width:8em" });
    ci.onchange = () => { s.op.changed = ci.value; scheduleCompile(P); };
    P.opParams.appendChild(ci);
  }
  /* candidate box */
  P.candBox.innerHTML = "";
  P.twinBox.innerHTML = "";
  if (!s.candAuto) {
    const c = s.cand;
    const r1 = el("div", { class: "row" });
    const isaVers = s.isa === "x86" ? ["avx512", "avx2", "sse"] :
      s.isa === "aarch64" ? ["sve", "neon"] : [];
    if (isaVers.length) {
      r1.appendChild(el("label", {}, "isa "));
      const iv = select(["auto"].concat(isaVers),
        c.isaVer || "auto");
      iv.onchange = () => { c.isaVer = iv.value === "auto" ? undefined
        : iv.value; sanitizeIsaGated(s); refreshPane(P); scheduleCompile(P); };
      r1.appendChild(iv);
    }
    r1.appendChild(el("label", {}, "mul "));
    const mul = select(["auto", "nibble", "gfni", "zvbc"], c.mul || "auto",
      (o) => !mulIsaOk(o, s));
    mul.onchange = () => { c.mul = mul.value; scheduleCompile(P); };
    r1.appendChild(mul);
    r1.appendChild(el("label", {}, " U (0=auto) "));
    const u = el("input", { type: "number", value: c.U || 0, min: 0, max: 8,
      style: "width:4em" });
    u.onchange = () => { c.U = parseInt(u.value, 10) || 0; scheduleCompile(P); };
    r1.appendChild(u);
    if (s.isa === "riscv") {
      r1.appendChild(el("label", {}, " lmul "));
      const l = select(["1", "2", "4", "8"], String(c.lmul || 1));
      l.onchange = () => { c.lmul = parseInt(l.value, 10); scheduleCompile(P); };
      r1.appendChild(l);
    }
    r1.appendChild(el("label", {}, " prefetch_dist "));
    const pf = el("input", { type: "number", value: c.prefetch_dist || 0,
      min: 0, style: "width:6em" });
    pf.onchange = () => { c.prefetch_dist = parseInt(pf.value, 10) || 0;
      scheduleCompile(P); };
    r1.appendChild(pf);
    P.candBox.appendChild(r1);
    /* twin checkboxes with mutual-exclusion grey-out */
    for (const t of TWINS) {
      const lab = el("label", { class: "twin" });
      const cb = el("input", { type: "checkbox" });
      cb.checked = !!s.twins[t];
      cb.onchange = () => { s.twins[t] = cb.checked; refreshPane(P);
        scheduleCompile(P); };
      /* grey out twins excluded by another selected twin in the same
       * exclusivity group, by ISA gate, or by plan-kind gate */
      const isaOk = !TWIN_ISA_OK[t] || TWIN_ISA_OK[t](effIsa(s));
      const planOk = !TWIN_PLAN_OK[t] || !s.lastPlan ||
        TWIN_PLAN_OK[t](s.lastPlan);
      let excl = false;
      if (BODY_TWINS.includes(t)) {
        excl = BODY_TWINS.some(o => o !== t && s.twins[o]);
      } else if (TAIL_TWINS.includes(t)) {
        excl = TAIL_TWINS.some(o => o !== t && s.twins[o]);
      }
      cb.disabled = (excl && !s.twins[t]) || !isaOk || !planOk;
      lab.appendChild(cb); lab.appendChild(document.createTextNode(" " + t));
      P.twinBox.appendChild(lab);
    }
  }
}

let compileTimer = {};
function scheduleCompile(P) {
  const idx = panes.indexOf(P);
  clearTimeout(compileTimer[idx]);
  compileTimer[idx] = setTimeout(() => runPane(P), 350);
}

async function runPane(P) {
  const s = P.state;
  P.status.textContent = "compiling…";
  P.status.className = "status busy";
  let doc;
  try { doc = buildDoc(s); }
  catch (e) { P.status.textContent = "option error: " + e.message;
    P.status.className = "status err"; return; }
  let mod;
  try { mod = await getModule(s.isa); }
  catch (e) { P.status.textContent = "module load failed: " + e.message;
    P.status.className = "status err"; return; }
  const r = compileKernel(mod, doc);
  if (!r.ok) {
    P.status.textContent = "build failed: " + r.err;
    P.status.className = "status err";
    P.cfgTable.innerHTML = ""; P.disasm.innerHTML = "";
    writeHash(panes.map(q => q.state));
    return;
  }
  /* sha256 via SubtleCrypto (async) — fall back to "…" */
  let hex = "…";
  try {
    const h = await crypto.subtle.digest("SHA-256", r.code);
    hex = Array.from(new Uint8Array(h),
      b => b.toString(16).padStart(2, "0")).join("");
  } catch (e) { /* file:// contexts without crypto.subtle */ }
  const rows = [
    ["desc", r.desc], ["plan_id", r.plan_id],
    ["target", r.target], ["topology", r.topology],
    ["operation", r.operation], ["strategy", r.strategy],
    ["isa", r.isa], ["mul", r.mul],
    ["unroll", r.unroll], ["lmul", r.lmul],
    ["twins", (r.twins && r.twins.join(" ")) || "(none)"],
    ["code_size", r.code_size], ["const_size", r.const_size],
    ["uops/byte (static)", r.uops_per_byte],
    ["mem/byte (static)", r.mem_per_byte],
    ["code sha256", hex],
    ["selection", r.selection],
    ["exec_validated", "false (compile-only; byte-identity vs native is the proof)"],
  ];
  P.cfgTable.innerHTML = "";
  for (const [k, v] of rows) {
    const tr = el("tr");
    tr.appendChild(el("td", { class: "k" }, k));
    tr.appendChild(el("td", {}, String(v)));
    P.cfgTable.appendChild(tr);
  }
  P.status.textContent = "ok";
  P.status.className = "status ok";
  /* capture plan kind for twin grey-out; refresh if it changed */
  if (r.strategy) {
    const plan = { strategy: r.strategy, xor_sched: r.plan_xor_sched | 0,
      n_out: r.plan_n_out | 0 };
    const key = strategyKey(s);
    if (!s.lastPlan || s.lastPlan.strategy !== plan.strategy ||
        s.lastPlan.xor_sched !== plan.xor_sched) {
      s.lastPlan = plan;
      strategyCache[key] = plan;
      sanitizePlanGated(s);
      refreshPane(P);
    }
  }
  const d = await disassemble(DISASM_ARCH[s.isa], r.code);
  P.disasm.innerHTML = "";
  if (d.error) P.disasm.textContent = d.error;
  else P.disasm.appendChild(renderDisasm(d.insns));
  writeHash(panes.map(q => q.state));
}

function init() {
  const fromHash = readHash();
  const initial = [];
  initial.push(fromHash.a || defaultPane());
  abMode = !!fromHash.b;
  if (fromHash.b) initial.push(fromHash.b);
  document.getElementById("ab-toggle").checked = abMode;
  const wrap = document.getElementById("panes");
  for (const st of initial) {
    const P = buildPane(st);
    panes.push(P); wrap.appendChild(P.root);
    runPane(P);
  }
  document.getElementById("ab-toggle").onchange = e => {
    abMode = e.target.checked;
    if (abMode && panes.length < 2) {
      const P = buildPane(defaultPane());
      panes.push(P); wrap.appendChild(P.root); runPane(P);
    } else if (!abMode && panes.length > 1) {
      const P = panes.pop();
      P.root.remove();
    }
    document.getElementById("panes").classList.toggle("ab", abMode);
    writeHash(panes.map(q => q.state));
  };
  document.getElementById("panes").classList.toggle("ab", abMode);
  document.getElementById("copy-link").onclick = () => {
    navigator.clipboard.writeText(location.href).catch(() => {});
  };
}
document.addEventListener("DOMContentLoaded", init);
"""

HTML_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>raidjit live kernel explorer</title>
<style>
body{font-family:system-ui,sans-serif;margin:1em;color:#111;background:#fff}
h1{font-size:1.3em}
.row{margin:.3em 0}
label{margin-right:.2em}
.pane{border:1px solid #ccc;padding:.8em;margin:.5em 0;max-width:72em}
#panes.ab{display:flex;gap:1em;align-items:flex-start}
#panes.ab .pane{flex:1;min-width:0}
table.cfg{border-collapse:collapse;margin:.5em 0;font-size:.85em}
table.cfg td{border:1px solid #ddd;padding:.15em .5em;vertical-align:top}
td.k{font-weight:bold;white-space:nowrap}
div.disasm{overflow:auto;max-height:40em}
table.disasm{border-collapse:collapse;font-family:monospace;font-size:.8em;
  background:#f6f6f6}
table.disasm th,table.disasm td{padding:.1em .7em;vertical-align:top;
  text-align:left;white-space:pre}
table.disasm th{border-bottom:1px solid #bbb}
table.disasm td.off{color:#666;text-align:right}
table.disasm td.bytes{color:#064}
.status{font-weight:bold;margin:.3em 0}
.status.busy{color:#a60}.status.ok{color:#070}.status.err{color:#a00}
.twin{display:inline-block;margin:.15em .6em .15em 0;font-size:.85em}
.twin input:disabled{cursor:not-allowed;opacity:.45}
.twin:has(input:disabled){color:#999;cursor:not-allowed}
select option:disabled{color:#999}
input:disabled{cursor:not-allowed}
.cand{margin:.3em 0}
.toolbar{margin:.5em 0}
.note{font-size:.85em;color:#555;max-width:72em}
code{background:#f0f0f0;padding:0 .2em}
</style>
</head>
<body>
<h1>raidjit live kernel explorer</h1>
<p class="note">page build <code>2026-10-09h</code> (per-ctor topology defaults;
stale deep links self-heal). Compiles kernels live in your browser via WebAssembly
(planner + emitters for x86-64 / AArch64 / RISC-V, plus a Capstone
disassembler). Nothing is executed: the modules are compile-only, and
correctness is argued by byte-identity with the native build
(<code>tools/wasm/parity/</code>). <code>uops/byte</code> figures are
static-model estimates, not measurements.</p>
<div class="toolbar">
<label><input type="checkbox" id="ab-toggle"> side-by-side A/B comparison</label>
<button id="copy-link">copy link</button>
</div>
<div id="panes"></div>
<script>
"""

HTML_TAIL = """
</script>
</body>
</html>
"""


def main():
    out_path = (sys.argv[1] if len(sys.argv) > 1
                else os.path.join(OUT_DIR, "raidjit_kernels.html"))
    parts = [HTML_HEAD]
    for m in MODULES:
        p = os.path.join(OUT_DIR, m)
        if not os.path.exists(p):
            sys.exit(f"missing module {p}: build with make -f tools/wasm/Makefile.wasm")
        with open(p, "r", encoding="utf-8") as f:
            js = f.read()
        # Guard against an accidental </script> inside the bundle.
        js = js.replace("</script", "<\\/script")
        parts.append(js + "\n")
    parts.append(PAGE_JS)
    parts.append(HTML_TAIL)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("".join(parts))
    print(f"wrote {out_path} ({os.path.getsize(out_path)} bytes)")


if __name__ == "__main__":
    main()
