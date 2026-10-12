#!/usr/bin/env python3
"""Mechanical broken-link checker for markdown docs.

Walks the repo for *.md files and verifies every markdown link:
  - relative file/dir links resolve to an existing path (rendered by GitHub
    relative to the containing file);
  - #anchor links match a heading in the target file (GitHub slug rules);
  - absolute http(s) links return success (best-effort, short timeout).

Links inside fenced code blocks and HTML comments are skipped (not rendered).

Exit 0 when clean, 1 with a report when broken links are found.
Usage: check_markdown_links.py [--root DIR] [--no-external]
"""

import os
import re
import sys
import urllib.request
import urllib.error

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv"}
# Vendored third-party trees are not ours to fix: skip any directory
# named "vendor" at any depth.
SKIP_DIRNAMES = {"vendor"}

# Inline: [text](url)  [text](url "title")  [text](<url>)
INLINE_RE = re.compile(r"!?\[[^\]]*\]\(\s*(<[^>]+>|[^)\s]+)(?:\s+\"[^\"]*\")?\s*\)")
# Reference: [text][ref]  and definitions  [ref]: url
REFUSE_RE = re.compile(r"!?\[[^\]]*\]\[([^\]]*)\]")
REFDEF_RE = re.compile(r"^\s{0,3}\[([^\]]+)\]:\s*(<[^>]+>|[^)\s]+)")
# Autolink: <https://...>
AUTO_RE = re.compile(r"<(https?://[^>]+)>")
# Fenced code blocks
FENCE_RE = re.compile(r"^\s*(```|~~~)")


def github_slug(heading):
    """Approximate GitHub heading-anchor slugification."""
    text = re.sub(r"<[^>]+>", "", heading)  # strip inline HTML
    text = text.strip().lower()
    text = re.sub(r"[^\w\s\-]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", "-", text)
    return text


def headings_of(path):
    """Map of slug -> heading text for a markdown file (ATX headings only)."""
    slugs = {}
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            in_fence = False
            for line in fh:
                if FENCE_RE.match(line):
                    in_fence = not in_fence
                    continue
                if in_fence:
                    continue
                m = re.match(r"^(#{1,6})\s+(.*)", line.rstrip())
                if m:
                    slugs[github_slug(m.group(2))] = m.group(2)
    except OSError:
        pass
    return slugs


def strip_unrendered(text):
    """Remove fenced code blocks and HTML comments (not rendered by GitHub)."""
    out = []
    in_fence = False
    in_comment = False
    for line in text.splitlines(keepends=True):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if "<!--" in line:
            in_comment = True
        if in_comment:
            if "-->" in line:
                in_comment = False
            continue
        out.append(line)
    return "".join(out)


def extract_links(text):
    """Yield (url, line_number) for every markdown link outside code/comments."""
    text = strip_unrendered(text)
    # collect reference definitions first
    refdefs = {}
    lines = text.splitlines()
    for i, line in enumerate(lines, 1):
        m = REFDEF_RE.match(line)
        if m:
            refdefs[m.group(1).strip().lower()] = (m.group(2).strip(" <>"), i)
    # Bare timing annotations like [XOR](1u) are not links (delay units).
    NOT_A_URL = re.compile(r"^\d+[a-zA-Z]*$")
    def usable(url):
        url = url.strip(" <>")
        return url and not NOT_A_URL.match(url)
    for i, line in enumerate(lines, 1):
        if REFDEF_RE.match(line):
            continue  # definition line itself, not a use
        for m in INLINE_RE.finditer(line):
            url = m.group(1)
            if usable(url):
                yield url.strip(" <>"), i
        for m in REFUSE_RE.finditer(line):
            ref = m.group(1).strip().lower()
            if ref in refdefs and usable(refdefs[ref][0]):
                yield refdefs[ref][0].strip(" <>"), i
        for m in AUTO_RE.finditer(line):
            yield m.group(1), i


def check_external(url, cache):
    if url in cache:
        return cache[url]
    ok, note = True, ""
    try:
        req = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": "mix-linkcheck"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status >= 400:
                ok, note = False, f"HTTP {resp.status}"
    except urllib.error.HTTPError as e:
        # some servers reject HEAD; retry with GET before calling it broken
        if e.code not in (405, 501):
            ok, note = False, f"HTTP {e.code}"
        else:
            try:
                req = urllib.request.Request(url, method="GET",
                                             headers={"User-Agent": "mix-linkcheck"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if resp.status >= 400:
                        ok, note = False, f"HTTP {resp.status}"
            except Exception as e2:
                ok, note = False, str(e2)[:80]
    except Exception as e:
        ok, note = False, str(e)[:80]
    cache[url] = (ok, note)
    return ok, note


def main():
    root = sys.argv[sys.argv.index("--root") + 1] if "--root" in sys.argv else "."
    check_ext = "--no-external" not in sys.argv
    root = os.path.abspath(root)
    broken = []
    ext_cache = {}
    md_files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in SKIP_DIRS and d not in SKIP_DIRNAMES]
        for fn in filenames:
            if fn.endswith(".md"):
                md_files.append(os.path.join(dirpath, fn))

    for path in sorted(md_files):
        rel = os.path.relpath(path, root)
        with open(path, encoding="utf-8", errors="replace") as fh:
            content = fh.read()
        for url, lineno in extract_links(content):
            if not url or url.startswith("mailto:"):
                continue
            if url.startswith(("http://", "https://")):
                if check_ext:
                    ok, note = check_external(url, ext_cache)
                    if not ok:
                        broken.append((rel, lineno, url, note))
                continue
            # split off fragment
            if "#" in url:
                filepart, frag = url.split("#", 1)
            else:
                filepart, frag = url, None
            # Bare words with no path characters (e.g. [b](mem), [XOR](1u))
            # are usually annotations, not links. Only treat as a link if
            # the target exists; otherwise skip to avoid false positives.
            looks_like_path = any(c in filepart for c in "/.") or filepart.startswith(".")
            target = path if not filepart else os.path.normpath(
                os.path.join(os.path.dirname(path), filepart))
            exists = os.path.exists(target)
            if not looks_like_path and not exists:
                continue
            if not exists:
                broken.append((rel, lineno, url, "target does not exist"))
                continue
            if frag:
                slugs = headings_of(target)
                if github_slug(frag) not in slugs and frag not in slugs:
                    broken.append((rel, lineno, url,
                                   f"no heading #{frag} in {os.path.relpath(target, root)}"))

    if broken:
        print(f"{len(broken)} broken link(s) in {len(md_files)} markdown files:")
        for rel, lineno, url, note in broken:
            print(f"  {rel}:{lineno}: [{url}] {note}")
        return 1
    print(f"OK: {len(md_files)} markdown files, no broken links.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
