#!/usr/bin/env python3
"""gen_kernels_page.py <kernels.md> <kernels.html>

Build the interactive kernels explorer page from the disassembled-kernels
reference document (tools/gen_kernels_md.py output).

The page is a single self-contained HTML file (embedded JSON data + JS, no
network): dropdowns for target/ISA, topology, operation, strategy and twin
flags plus a text search, showing one kernel's metadata table and disassembly
at a time. It works from file://.

This script is the ONLY way raidjit_kernels.html is produced -- never hand-edit
the .html. It is wired into tools/finalize_kernels.sh so the page refreshes
whenever `make kernels` regenerates the document.

Failure policy: any structural surprise in the input (missing "How to read",
missing/duplicate sections, missing metadata fields, missing asm fence,
section id not matching its table's topology/target, section count below the
sanity floor) is a loud non-zero exit. A changed doc format must break the
build, never silently desync the page.
"""

import html
import json
import re
import sys

MIN_SECTIONS = 100  # sanity floor: today's doc has 150 kernel sections
CORE_FIELDS = ("target", "topology", "operation", "strategy")


def die(msg):
    sys.stderr.write("gen_kernels_page.py: FATAL: %s\n" % msg)
    sys.exit(1)


def parse(md_path):
    text = open(md_path, encoding="utf-8").read()

    m = re.search(r"^Generated (\S+) by tools/gen_kernels\.c", text, re.M)
    if not m:
        die("doc header lacks the 'Generated <ts> by tools/gen_kernels.c' line")
    generated_ts = m.group(1)
    if not text.startswith("# raidjit kernels"):
        die("doc does not start with '# raidjit kernels'")

    parts = text.split("\n## ")
    if len(parts) < 3:
        die("expected header + 'How to read' + kernel sections, got %d parts"
            % len(parts))
    howto_head = parts[1].split("\n", 1)[0].strip()
    if howto_head != "How to read":
        die("second section is %r, expected 'How to read'" % howto_head)
    howto_body = parts[1].split("\n", 1)[1] if "\n" in parts[1] else ""

    kernels = []
    seen_ids = set()
    for part in parts[2:]:
        sid = part.split("\n", 1)[0].strip()
        if not sid:
            die("empty section id")
        if sid in seen_ids:
            die("duplicate section id %r" % sid)
        seen_ids.add(sid)

        # Prose: first non-blank line after the heading that is not a table
        # row or a fence.
        prose = ""
        for line in part.split("\n")[1:]:
            s = line.strip()
            if not s or s.startswith("|") or s.startswith("```"):
                continue
            prose = s
            break
        if not prose:
            die("section %r has no prose line" % sid)

        # Metadata table.
        rows = re.findall(r"^\| ([^|]+?) \| (.*?) \|$", part, re.M)
        fields = {}
        for k, v in rows:
            k = k.strip()
            if k in ("field", "-------"):
                continue
            v = v.strip()
            if v.startswith("`") and v.endswith("`") and len(v) >= 2:
                v = v[1:-1]
            fields[k] = v
        for f in CORE_FIELDS:
            if f not in fields or not fields[f]:
                die("section %r: metadata table missing field %r" % (sid, f))

        # Twin flags from the id: <topo>_<target>[_<twin>]. The table's
        # topology carries a "(N)" suffix the id omits.
        topo_base = re.sub(r"\(.*\)$", "", fields["topology"])
        prefix = topo_base + "_" + fields["target"]
        if sid == prefix:
            twins = []
        elif sid.startswith(prefix + "_"):
            twins = [sid[len(prefix) + 1:]]
        else:
            die("section id %r does not match table topology %r / target %r"
                % (sid, fields["topology"], fields["target"]))

        # Exactly one ```asm fenced block, non-empty.
        fences = re.findall(r"```asm\n(.*?)```", part, re.S)
        if len(fences) != 1:
            die("section %r: expected 1 asm fence, found %d" % (sid, len(fences)))
        asm = fences[0]
        if not asm.strip():
            die("section %r: empty disassembly" % sid)

        kernels.append({
            "id": sid,
            "prose": prose,
            "fields": fields,
            "field_order": [k for k, _ in rows
                            if k.strip() not in ("field", "-------")],
            "twins": twins,
            "asm": asm,
        })

    if len(kernels) < MIN_SECTIONS:
        die("only %d kernel sections parsed, below sanity floor %d"
            % (len(kernels), MIN_SECTIONS))
    return generated_ts, howto_body, kernels


PAGE_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>raidjit kernels &mdash; interactive explorer</title>
<style>
:root { color-scheme: light; }
body { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
       margin: 0; padding: 0 24px 48px; max-width: 1100px; }
header h1 { font-size: 1.4em; margin: 20px 0 4px; }
header .sub { color: #555; font-size: 0.9em; margin-bottom: 16px; }
#controls { display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-end;
            padding: 12px; background: #f4f4f4; border-radius: 8px; }
#controls label { display: flex; flex-direction: column; font-size: 0.8em;
                  color: #333; }
#controls select, #controls input[type=search] { font-size: 1.05em;
                  padding: 4px 6px; margin-top: 2px; max-width: 240px; }
#status { margin: 14px 0 6px; font-size: 0.95em; color: #333; }
#list { margin: 8px 0; }
#list button { display: block; margin: 2px 0; font-family: monospace;
               font-size: 0.9em; padding: 3px 8px; cursor: pointer; }
#kernel h2 { font-family: monospace; font-size: 1.15em; margin: 18px 0 6px; }
#kernel .prose { color: #333; margin: 0 0 10px; }
#kernel table.meta { border-collapse: collapse; margin: 0 0 14px; font-size: 0.9em; }
#kernel table.meta td { border: 1px solid #ccc; padding: 3px 10px; }
#kernel table.meta td:first-child { color: #555; white-space: nowrap; }
#kernel table.meta td:last-child { font-family: monospace; }
#kernel pre.asm { background: #1e1e1e; color: #d4d4d4; padding: 14px;
                  border-radius: 8px; overflow-x: auto; font-size: 0.82em;
                  line-height: 1.35; }
#kernel .meta2 { font-size: 0.85em; color: #555; margin: 4px 0 8px; }
#kernel .actions { margin: 6px 0 0; }
#kernel .actions button { font-size: 0.85em; padding: 3px 10px; cursor: pointer; }
details.howto { margin: 0 0 4px; font-size: 0.9em; color: #444; }
details.howto summary { cursor: pointer; }
details.howto p { max-width: 1000px; }
.no-match { color: #a00; font-weight: bold; }
</style>
</head>
<body>
<header>
<h1>raidjit kernels &mdash; interactive explorer</h1>
<div class="sub">__COUNT__ kernels &middot; document generated __GEN_TS__ by
tools/gen_kernels.c (<code>make kernels</code>) &middot; page generated __PAGE_TS__</div>
<details class="howto"><summary>How to read</summary>__HOWTO__</details>
</header>
<div id="controls">
<label>Target / ISA<select id="f-target"></select></label>
<label>Topology<select id="f-topology"></select></label>
<label>Operation<select id="f-operation"></select></label>
<label>Strategy<select id="f-strategy"></select></label>
<label>Twin flags<select id="f-twins"></select></label>
<label>Search<input id="q" type="search" placeholder="id, text, sha&hellip;" autocomplete="off"></label>
</div>
<div id="status" aria-live="polite"></div>
<div id="list"></div>
<div id="kernel"></div>
<script>
"use strict";
const KERNELS = __KERNELS_JSON__;
const META = { generated: __GEN_TS_JSON__, count: KERNELS.length };

const FACETS = [
  { key: "target",    label: "Target / ISA" },
  { key: "topology",  label: "Topology" },
  { key: "operation", label: "Operation" },
  { key: "strategy",  label: "Strategy" },
];
const ANY = "";

function esc(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;")
                  .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function twinsMatch(k, sel) {
  if (sel === ANY) return true;
  if (sel === "none") return k.twins.length === 0;
  return k.twins.indexOf(sel) !== -1;
}

/* Does kernel k satisfy selection sel, ignoring facet `except`? */
function matchesExcept(k, sel, except) {
  for (const f of FACETS) {
    if (f.key !== except && sel[f.key] !== ANY &&
        k.fields[f.key] !== sel[f.key]) return false;
  }
  if (except !== "twins" && !twinsMatch(k, sel.twins)) return false;
  return true;
}

function matchesQuery(k, q) {
  if (!q) return true;
  q = q.toLowerCase();
  if ((k.id + " " + k.prose).toLowerCase().indexOf(q) !== -1) return true;
  for (const key of k.field_order) {
    if (String(k.fields[key]).toLowerCase().indexOf(q) !== -1) return true;
  }
  return false;
}

function filterKernels(sel, q) {
  return KERNELS.filter(k => matchesExcept(k, sel, null) && matchesQuery(k, q));
}

/* Valid option values for one facet given the other facets' selections. */
function validOptions(key, sel) {
  const vals = [];
  const seen = new Set();
  if (key === "twins") {
    let anyBase = false;
    for (const k of KERNELS) {
      if (!matchesExcept(k, sel, "twins")) continue;
      if (k.twins.length === 0) anyBase = true;
      for (const t of k.twins) {
        if (!seen.has(t)) { seen.add(t); vals.push(t); }
      }
    }
    vals.sort();
    if (anyBase) vals.unshift("none");
    return vals;
  }
  for (const k of KERNELS) {
    if (!matchesExcept(k, sel, key)) continue;
    const v = k.fields[key];
    if (!seen.has(v)) { seen.add(v); vals.push(v); }
  }
  vals.sort();
  return vals;
}

function optionLabel(key, v) {
  if (v === ANY) return "(any)";
  if (key === "twins" && v === "none") return "(base kernel, no twins)";
  return v;
}

function renderKernelHTML(k) {
  let rows = "";
  for (const key of k.field_order) {
    rows += "<tr><td>" + esc(key) + "</td><td>" + esc(k.fields[key]) + "</td></tr>\n";
  }
  const twins = k.twins.length ? k.twins.map(esc).join(", ") : "(base kernel)";
  const nlines = k.asm.split("\n").length;
  return (
    "<h2 id=\"k-" + esc(k.id) + "\">" + esc(k.id) + "</h2>\n" +
    "<p class=\"prose\">" + esc(k.prose) + "</p>\n" +
    "<div class=\"meta2\">twin flags: <b>" + twins + "</b> &middot; " +
    nlines + " disassembly lines</div>\n" +
    "<table class=\"meta\">" + rows + "</table>\n" +
    "<div class=\"actions\"><button id=\"copy-asm\" type=\"button\">copy disassembly</button> " +
    "<button id=\"copy-id\" type=\"button\">copy section id</button></div>\n" +
    "<pre class=\"asm\"><code id=\"asm-code\">" + esc(k.asm) + "</code></pre>\n"
  );
}

/* ---- DOM wiring (skipped under node test harness) ---- */
function currentSelection() {
  const sel = { twins: ANY };
  for (const f of FACETS) sel[f.key] = document.getElementById("f-" + f.key).value;
  sel.twins = document.getElementById("f-twins").value;
  return sel;
}

function rebuildSelect(key, opts, keep) {
  const el = document.getElementById("f-" + key);
  const cur = keep && opts.indexOf(keep) !== -1 ? keep : ANY;
  let h = "";
  h += "<option value=\"\">" + esc(optionLabel(key, ANY)) + "</option>";
  for (const v of opts) {
    h += "<option value=\"" + esc(v) + "\"" + (v === cur ? " selected" : "") +
         ">" + esc(optionLabel(key, v)) + "</option>";
  }
  el.innerHTML = h;
  return cur;
}

function setStatus(msg, isErr) {
  const el = document.getElementById("status");
  el.textContent = msg;
  el.className = isErr ? "no-match" : "";
}

function refresh() {
  const sel = currentSelection();
  for (const f of FACETS) sel[f.key] = rebuildSelect(f.key, validOptions(f.key, sel), sel[f.key]);
  sel.twins = rebuildSelect("twins", validOptions("twins", sel), sel.twins);
  const q = document.getElementById("q").value.trim();
  const hits = filterKernels(sel, q);
  const listEl = document.getElementById("list");
  const kEl = document.getElementById("kernel");
  listEl.innerHTML = "";
  if (hits.length === 0) {
    setStatus("No kernel matches this combination.", true);
    kEl.innerHTML = "";
    return;
  }
  if (hits.length === 1) {
    const k = hits[0];
    setStatus("Showing 1 kernel.", false);
    kEl.innerHTML = renderKernelHTML(k);
    if (location.hash !== "#" + k.id) history.replaceState(null, "", "#" + k.id);
    document.getElementById("copy-asm").addEventListener("click", () => {
      navigator.clipboard.writeText(k.asm);
    });
    document.getElementById("copy-id").addEventListener("click", () => {
      navigator.clipboard.writeText(k.id);
    });
    return;
  }
  setStatus(hits.length + " kernels match \u2014 refine the selection or pick one:", false);
  kEl.innerHTML = "";
  for (const k of hits) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = k.id;
    b.addEventListener("click", () => {
      document.getElementById("q").value = k.id;
      refresh();
    });
    listEl.appendChild(b);
  }
}

function init() {
  for (const f of FACETS.concat([{ key: "twins" }])) {
    document.getElementById("f-" + f.key).addEventListener("change", refresh);
  }
  document.getElementById("q").addEventListener("input", refresh);
  // Populate every select once (unconstrained) so deep-link values below
  // have matching options to land on.
  const empty = { target: ANY, topology: ANY, operation: ANY, strategy: ANY, twins: ANY };
  for (const key of ["target", "topology", "operation", "strategy", "twins"]) {
    rebuildSelect(key, validOptions(key, empty), ANY);
  }
  // Deep link: #<section-id> pre-selects that kernel's facets.
  const h = location.hash ? decodeURIComponent(location.hash.slice(1)) : "";
  const k = KERNELS.find(x => x.id === h);
  if (k) {
    for (const f of FACETS) document.getElementById("f-" + f.key).value = k.fields[f.key];
    document.getElementById("f-twins").value = k.twins.length ? k.twins[0] : "none";
  }
  refresh();
}

if (typeof document !== "undefined" && document.addEventListener) {
  document.addEventListener("DOMContentLoaded", init);
}
</script>
</body>
</html>
"""

HOWTO_PARA_SPLIT = re.compile(r"\n\s*\n")


def howto_html(body):
    paras = [p.strip() for p in HOWTO_PARA_SPLIT.split(body.strip()) if p.strip()]
    return "".join("<p>%s</p>" % html.escape(p) for p in paras)


def main():
    if len(sys.argv) != 3:
        sys.stderr.write("usage: gen_kernels_page.py <kernels.md> <kernels.html>\n")
        sys.exit(2)
    md_path, html_path = sys.argv[1], sys.argv[2]

    generated_ts, howto_body, kernels = parse(md_path)

    import datetime
    page_ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    payload = json.dumps(kernels, ensure_ascii=False, separators=(",", ":"))
    # Never allow a literal </script> inside the embedded JSON.
    payload = payload.replace("</", "<\\/")

    page = PAGE_TEMPLATE
    page = page.replace("__KERNELS_JSON__", payload)
    page = page.replace("__GEN_TS_JSON__", json.dumps(generated_ts))
    page = page.replace("__GEN_TS__", html.escape(generated_ts))
    page = page.replace("__PAGE_TS__", html.escape(page_ts))
    page = page.replace("__COUNT__", str(len(kernels)))
    page = page.replace("__HOWTO__", howto_html(howto_body))

    # Self-check: no placeholder left unreplaced.
    for marker in ("__KERNELS_JSON__", "__GEN_TS_JSON__", "__GEN_TS__",
                   "__PAGE_TS__", "__COUNT__", "__HOWTO__"):
        if marker in page:
            die("template placeholder %s not replaced" % marker)

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(page)
    sys.stderr.write("gen_kernels_page.py: wrote %s (%d kernels, doc %s)\n"
                     % (html_path, len(kernels), generated_ts))


if __name__ == "__main__":
    main()
