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
  const res = JSON.parse(mod.UTF8ToString(rp));
  mod._rjw_free_string(rp); mod._free(p);
  if (!res.ok) return res;
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
  const d = await getDisasm();
  const p = d._malloc(bytes.length || 1);
  if (bytes.length) d.HEAPU8.set(bytes, p);
  let rp;
  try { rp = d._dj_disasm(arch, p, bytes.length, 0); }
  catch (e) { d._free(p); return { error: "trap: " + e.message }; }
  const arr = JSON.parse(d.UTF8ToString(rp));
  d._dj_free_string(rp); d._free(p);
  return { insns: arr };
}
/* objdump-style normalization (SPEC: byte equality is the invariant) */
function fmtHex(bytes) {
  return Array.from(bytes, b => b.toString(16).padStart(2, "0")).join("");
}
function renderDisasm(insns) {
  return insns.map(e => {
    const addr = e.addr.toString(16).padStart(4, "0");
    let line = addr + ":\t" + e.bytes + "\t" + e.mnemonic;
    if (e.op_str) line += "\t" + e.op_str;
    return line;
  }).join("\n");
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
const TWINS = ["nt", "ternlog", "remat", "notail", "predtail", "maskedtail",
               "permi2b", "xorsched", "xorbit", "mshift", "pipe"];
/* pairwise-exclusive body twins */
const BODY_TWINS = ["xorsched", "xorbit", "permi2b", "mshift", "pipe", "remat"];

function defaultPane() {
  return {
    isa: "x86", target: "avx512-gfni", cpu: {},
    topology: { ctor: "raid5", p1: 4 },
    op: { kind: "encode" },
    candAuto: true,
    cand: { mul: "auto", U: 0, lmul: 1, prefetch_dist: 0 },
    twins: {},
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
    if (p.op.rows) op.rows = parseIdxList(p.op.rows);
  } else if (p.op.kind === "parity_update") {
    op.changed = parseIdxList(p.op.changed);
  }
  let cand = "auto";
  if (!p.candAuto) {
    cand = { mul: p.cand.mul === "auto" ? undefined : p.cand.mul,
             U: p.cand.U | 0, lmul: p.cand.lmul | 0,
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
function readHash() {
  const h = location.hash.replace(/^#/, "");
  const out = {};
  for (const part of h.split("&")) {
    const [k, v] = part.split("=");
    if ((k === "a" || k === "b") && v) {
      try { out[k] = JSON.parse(b64urlDecode(v)); } catch (e) { /* ignore */ }
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
function select(opts, val) {
  const s = el("select");
  for (const o of opts) {
    const op = el("option", { value: o }, o);
    if (o === val) op.selected = true;
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
  P.disasm = el("pre", { class: "disasm" });
  root.appendChild(P.disasm);

  P.isaSel.onchange = () => {
    state.isa = P.isaSel.value;
    state.target = TARGETS[state.isa][0];
    refreshPane(P); scheduleCompile(P);
  };
  P.targetSel.onchange = () => { state.target = P.targetSel.value;
    scheduleCompile(P); };
  P.ctorSel.onchange = () => { state.topology = { ctor: P.ctorSel.value,
    p1: 4 }; refreshPane(P); scheduleCompile(P); };
  P.opSel.onchange = () => { state.op = { kind: P.opSel.value };
    refreshPane(P); scheduleCompile(P); };
  P.autoChk.onchange = () => { state.candAuto = P.autoChk.checked;
    refreshPane(P); scheduleCompile(P); };
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
        : iv.value; scheduleCompile(P); };
      r1.appendChild(iv);
    }
    r1.appendChild(el("label", {}, "mul "));
    const mul = select(["auto", "nibble", "gfni", "zvbc"], c.mul || "auto");
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
      /* grey out body twins excluded by another selected body twin */
      if (BODY_TWINS.includes(t)) {
        const other = BODY_TWINS.some(o => o !== t && s.twins[o]);
        cb.disabled = other && !s.twins[t];
      }
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
    P.cfgTable.innerHTML = ""; P.disasm.textContent = "";
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
  const d = await disassemble(DISASM_ARCH[s.isa], r.code);
  P.disasm.textContent = d.error ? d.error : renderDisasm(d.insns);
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
pre.disasm{background:#f6f6f6;padding:.8em;overflow:auto;font-size:.8em;
  max-height:40em}
.status{font-weight:bold;margin:.3em 0}
.status.busy{color:#a60}.status.ok{color:#070}.status.err{color:#a00}
.twin{display:inline-block;margin:.15em .6em .15em 0;font-size:.85em}
.cand{margin:.3em 0}
.toolbar{margin:.5em 0}
.note{font-size:.85em;color:#555;max-width:72em}
code{background:#f0f0f0;padding:0 .2em}
</style>
</head>
<body>
<h1>raidjit live kernel explorer</h1>
<p class="note">Compiles kernels live in your browser via WebAssembly
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
