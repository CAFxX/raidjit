"use strict";
/* S9.8 self-attack on the WASM modules + page deep-link parser.
 * Attacks the compiled modules directly (Node), which is exactly what
 * the page's compileKernel()/disassemble() wrappers call.
 * Verdict per case: the module must NEVER trap or corrupt state;
 * it must return well-formed JSON (or a clean error object).
 */
const fs = require("fs");
const OUT = "tools/wasm/out/";

let pass = 0, fail = 0;
const findings = [];
function check(name, cond, detail) {
  if (cond) { pass++; }
  else { fail++; findings.push(name + (detail ? " :: " + detail : "")); }
  console.log((cond ? "PASS " : "FAIL ") + name);
}

async function load(factoryFile, factoryName) {
  global.self = global;
  // Run the bundle with a CommonJS-like scope (the factory name is a
  // top-level binding inside the bundle) and pull the factory out.
  const code = fs.readFileSync(OUT + factoryFile, "utf8");
  const fn = new Function("require", "module", "exports", "__dirname",
    "__filename", "self", code + `\n;return ${factoryName};`);
  const f = fn(require, { exports: {} }, {}, __dirname, __filename, global);
  if (typeof f !== "function") throw new Error(factoryName + " not exported");
  return await f();
}

function callCompile(mod, input) {
  // input: string (may be garbage). Returns {trapped, json|null, err}
  let p = 0;
  try {
    p = mod.stringToNewUTF8(input);
  } catch (e) {
    return { trapped: true, where: "stringToNewUTF8: " + e.message };
  }
  let rp = 0;
  try {
    rp = mod._rjw_compile(p);
  } catch (e) {
    try { mod._free(p); } catch (_) {}
    return { trapped: true, where: "_rjw_compile: " + e.message };
  }
  if (!rp) { mod._free(p); return { trapped: false, nullResult: true }; }
  let s;
  try { s = mod.UTF8ToString(rp); }
  catch (e) { mod._free(p); return { trapped: true, where: "UTF8ToString: " + e.message }; }
  let j = null, parseErr = null;
  try { j = JSON.parse(s); } catch (e) { parseErr = e.message; }
  try { mod._rjw_free_string(rp); mod._free(p); } catch (_) {}
  return { trapped: false, json: j, parseErr, raw: s.slice(0, 200) };
}

function wellFormedError(r) {
  return !r.trapped && !r.nullResult && !r.parseErr &&
    r.json && typeof r.json === "object" &&
    r.json.ok === false && typeof r.json.err === "string" && r.json.err.length > 0;
}

const GOOD = JSON.stringify({ v: 1, target: "avx512-gfni",
  topology: { ctor: "raid5", p1: 4 }, op: { kind: "encode" },
  cand: "auto", options: {} });

async function attackCompile() {
  const mod = await load("raidjit_x86.js", "RaidjitX86");
  const cases = [
    ["garbage", "not json at all {{{"],
    ["empty", ""],
    ["null-json", "null"],
    ["array-json", "[1,2,3]"],
    ["number-json", "42"],
    ["missing-v", JSON.stringify({ target: "avx512-gfni" })],
    ["v-zero", JSON.stringify({ v: 0 })],
    ["v-two", JSON.stringify({ v: 2 })],
    ["v-string", JSON.stringify({ v: "1" })],
    ["missing-target", JSON.stringify({ v: 1, topology: { ctor: "raid5", p1: 4 }, op: { kind: "encode" } })],
    ["unknown-target", JSON.stringify({ v: 1, target: "quantum", topology: { ctor: "raid5", p1: 4 }, op: { kind: "encode" } })],
    ["missing-topology", JSON.stringify({ v: 1, target: "avx512-gfni", op: { kind: "encode" } })],
    ["unknown-ctor", JSON.stringify({ v: 1, target: "avx512-gfni", topology: { ctor: "raid99", p1: 4 }, op: { kind: "encode" } })],
    ["malformed-matrix-hex", JSON.stringify({ v: 1, target: "avx512-gfni", topology: { ctor: "custom", ndata: 2, npar: 1, matrix_hex: "zzzz!!" }, op: { kind: "encode" } })],
    ["matrix-hex-odd-len", JSON.stringify({ v: 1, target: "avx512-gfni", topology: { ctor: "custom", ndata: 2, npar: 1, matrix_hex: "abc" }, op: { kind: "encode" } })],
    ["huge-ndata", JSON.stringify({ v: 1, target: "avx512-gfni", topology: { ctor: "raid5", p1: 100000 }, op: { kind: "encode" } })],
    ["negative-p1", JSON.stringify({ v: 1, target: "avx512-gfni", topology: { ctor: "raid5", p1: -5 }, op: { kind: "encode" } })],
    ["unknown-op", JSON.stringify({ v: 1, target: "avx512-gfni", topology: { ctor: "raid5", p1: 4 }, op: { kind: "frob" } })],
    ["recover-failed-huge", JSON.stringify({ v: 1, target: "avx512-gfni", topology: { ctor: "raid5", p1: 4 }, op: { kind: "recover", failed: [999] } })],
    ["cand-U-huge", JSON.stringify({ v: 1, target: "avx512-gfni", topology: { ctor: "raid5", p1: 4 }, op: { kind: "encode" }, cand: { U: 1000000 } })],
    ["cand-U-negative", JSON.stringify({ v: 1, target: "avx512-gfni", topology: { ctor: "raid5", p1: 4 }, op: { kind: "encode" }, cand: { U: -3 } })],
    ["cand-mul-bogus", JSON.stringify({ v: 1, target: "avx512-gfni", topology: { ctor: "raid5", p1: 4 }, op: { kind: "encode" }, cand: { mul: "quantum" } })],
    ["cand-isa-bogus", JSON.stringify({ v: 1, target: "avx512-gfni", topology: { ctor: "raid5", p1: 4 }, op: { kind: "encode" }, cand: { isa: "bogus" } })],
    ["cand-all-twins", JSON.stringify({ v: 1, target: "avx512-gfni", topology: { ctor: "raid5", p1: 4 }, op: { kind: "encode" }, cand: { U: 4, nt: true, ternlog: true, remat: true, notail: true, predtail: true, maskedtail: true, permi2b: true, xorsched: true, xorbit: true, mshift: true, pipe: true } })],
    ["cand-twin-wrong-type", JSON.stringify({ v: 1, target: "avx512-gfni", topology: { ctor: "raid5", p1: 4 }, op: { kind: "encode" }, cand: { nt: "yes" } })],
    ["cpu-bogus-keys", JSON.stringify({ v: 1, target: "avx512-gfni", cpu: { flux_capacitor: true }, topology: { ctor: "raid5", p1: 4 }, op: { kind: "encode" } })],
    ["cpu-wrong-type", JSON.stringify({ v: 1, target: "avx512-gfni", cpu: "x", topology: { ctor: "raid5", p1: 4 }, op: { kind: "encode" } })],
    ["1MB-string", "x".repeat(1024 * 1024)],
    ["deep-nesting", "[".repeat(5000)],
    ["duplicate-keys", '{"v":1,"v":2,"target":"avx512-gfni","topology":{"ctor":"raid5","p1":4},"op":{"kind":"encode"}}'],
    ["unicode-target", JSON.stringify({ v: 1, target: "avx512-gfni\u0000\u2028", topology: { ctor: "raid5", p1: 4 }, op: { kind: "encode" } })],
    ["null-doc-fields", JSON.stringify({ v: 1, target: null, topology: null, op: null })],
    ["cand-null", JSON.stringify({ v: 1, target: "avx512-gfni", topology: { ctor: "raid5", p1: 4 }, op: { kind: "encode" }, cand: null })],
    ["options-array", JSON.stringify({ v: 1, target: "avx512-gfni", topology: { ctor: "raid5", p1: 4 }, op: { kind: "encode" }, options: [1] })],
  ];
  for (const [name, input] of cases) {
    const r = callCompile(mod, input);
    check("compile/" + name, wellFormedError(r),
      r.trapped ? "TRAPPED: " + r.where
      : r.nullResult ? "returned NULL"
      : r.parseErr ? "bad JSON out: " + r.parseErr + " :: " + r.raw
      : "json=" + JSON.stringify(r.json).slice(0, 160));
    // state must not be corrupted: a good compile still works after each attack
    const g = callCompile(mod, GOOD);
    check("compile/" + name + "+recovery", !g.trapped && g.json && g.json.ok === true,
      g.trapped ? "TRAPPED after attack: " + g.where : JSON.stringify(g.json).slice(0, 120));
  }
}

/* ---- deep-link hash attacks: functions copied VERBATIM from
 * tools/gen_live_kernels_page.py PAGE_JS ---- */
function b64urlDecode(s) {
  s = s.replace(/-/g, "+").replace(/_/g, "/");
  while (s.length % 4) s += "=";
  return decodeURIComponent(escape(atob(s)));
}
function readHash(hash) {
  const h = hash.replace(/^#/, "");
  const out = {};
  for (const part of h.split("&")) {
    const [k, v] = part.split("=");
    if ((k === "a" || k === "b") && v) {
      try {
        const o = JSON.parse(b64urlDecode(v));
        if (o && typeof o === "object" && typeof o.isa === "string" &&
            o.topology && typeof o.topology.ctor === "string" &&
            o.op && typeof o.op.kind === "string") {
          out[k] = o;
        }
      } catch (e) { /* ignore */ }
    }
  }
  return out;
}
function b64urlEncode(s) {
  return Buffer.from(s, "utf8").toString("base64")
    .replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function attackHash() {
  const goodPane = JSON.stringify({ isa: "x86", topology: { ctor: "raid5" }, op: { kind: "encode" } });
  const cases = [
    ["empty", ""],
    ["bare-hash", "#"],
    ["a-empty", "#a="],
    ["invalid-b64", "#a=!!!"],
    ["b64-not-json", "#a=" + b64urlEncode("not json")],
    ["b64-array", "#a=" + b64urlEncode("[1,2]")],
    ["b64-null", "#a=" + b64urlEncode("null")],
    ["b64-string", "#a=" + b64urlEncode('"x"')],
    ["wrong-shape", "#a=" + b64urlEncode('{"foo":1}')],
    ["isa-not-string", "#a=" + b64urlEncode('{"isa":123,"topology":{"ctor":"raid5"},"op":{"kind":"encode"}}')],
    ["proto-pollution", "#a=" + b64urlEncode('{"__proto__":{"polluted":1},"isa":"x86","topology":{"ctor":"raid5"},"op":{"kind":"encode"}}')],
    ["ctor-not-string", "#a=" + b64urlEncode('{"isa":"x86","topology":{"ctor":5},"op":{"kind":"encode"}}')],
    ["unknown-key", "#c=" + b64urlEncode(goodPane)],
    ["no-equals", "#a"],
    ["double-equals", "#a==" + b64urlEncode(goodPane)],
    ["giant-hash", "#a=" + "A".repeat(5 * 1024 * 1024)],
    ["control-chars", "#a=\x00\x01\x02"],
    ["two-panes-second-bad", "#a=" + b64urlEncode(goodPane) + "&b=" + b64urlEncode('{"isa":1}')],
    ["extra-fields", "#a=" + b64urlEncode('{"isa":"x86","topology":{"ctor":"raid5"},"op":{"kind":"encode"},"evil":"x"}')],
    ["nested-isa-object", "#a=" + b64urlEncode('{"isa":{"toString":"x"},"topology":{"ctor":"raid5"},"op":{"kind":"encode"}}')],
  ];
  for (const [name, h] of cases) {
    let out, threw = null;
    try { out = readHash(h); } catch (e) { threw = e.message; }
    const keys = out ? Object.keys(out) : [];
    const okShape = !threw && keys.every(k => k === "a" || k === "b");
    check("hash/" + name, okShape, threw ? "THREW: " + threw : "keys=" + JSON.stringify(keys));
  }
  // prototype pollution must not have happened
  check("hash/proto-not-polluted", ({}).polluted === undefined,
    "Object.prototype.polluted = " + ({}).polluted);
  // round trip of a valid pane
  const rt = readHash("#a=" + b64urlEncode(goodPane) + "&b=" + b64urlEncode(goodPane));
  check("hash/roundtrip", rt.a && rt.b && rt.a.isa === "x86" && rt.b.isa === "x86",
    JSON.stringify(rt).slice(0, 120));
}

async function attackDisasm() {
  const d = await load("disasm.js", "Disasm");
  function callDis(arch, bytes) {
    const p = d._malloc(bytes.length || 1);
    if (bytes.length) d.HEAPU8.set(bytes, p);
    const ap = d.stringToNewUTF8(arch);
    let rp, trapped = null;
    try { rp = d._dj_disasm(ap, p, bytes.length, 0n); }
    catch (e) { trapped = e.message; }
    let arr = null, parseErr = null;
    if (!trapped && rp) {
      try { arr = JSON.parse(d.UTF8ToString(rp)); }
      catch (e) { parseErr = e.message; }
      try { d._dj_free_string(rp); } catch (_) {}
    }
    try { d._free(ap); d._free(p); } catch (_) {}
    return { trapped, nullResult: !trapped && !rp, arr, parseErr };
  }
  // deterministic PRNG for the random-bytes case
  let seed = 0x12345678;
  const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
  const cases = [
    ["empty-x86", "x86_64", new Uint8Array(0)],
    ["empty-a64", "aarch64", new Uint8Array(0)],
    ["empty-rv", "riscv64", new Uint8Array(0)],
    ["random-x86", "x86_64", Uint8Array.from({ length: 256 }, () => (rnd() * 256) | 0)],
    ["random-a64", "aarch64", Uint8Array.from({ length: 256 }, () => (rnd() * 256) | 0)],
    ["random-rv", "riscv64", Uint8Array.from({ length: 256 }, () => (rnd() * 256) | 0)],
    ["single-byte", "x86_64", new Uint8Array([0xff])],
    ["bogus-arch", "x86_99", new Uint8Array([0x90, 0x90])],
    ["empty-arch", "", new Uint8Array([0x90])],
    ["arch-1MB", "x".repeat(1024 * 1024), new Uint8Array([0x90])],
    ["all-zeros-4k", "aarch64", new Uint8Array(4096)],
    ["all-ff-4k", "riscv64", new Uint8Array(4096).fill(0xff)],
  ];
  for (const [name, arch, bytes] of cases) {
    const r = callDis(arch, bytes);
    const ok = !r.trapped && !r.parseErr &&
      (r.nullResult || Array.isArray(r.arr));
    check("disasm/" + name, ok,
      r.trapped ? "TRAPPED: " + r.trapped
      : r.parseErr ? "bad JSON: " + r.parseErr
      : r.nullResult ? "returned NULL (clean)" : "insns=" + r.arr.length);
  }
}

(async () => {
  await attackCompile();
  attackHash();
  await attackDisasm();
  console.log(`\n==== ${pass} passed, ${fail} failed ====`);
  if (findings.length) {
    console.log("FINDINGS:");
    for (const f of findings) console.log("  - " + f);
  }
  process.exit(fail ? 1 : 0);
})().catch(e => { console.error("HARNESS ERROR", e); process.exit(2); });
