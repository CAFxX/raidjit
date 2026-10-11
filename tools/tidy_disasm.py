#!/usr/bin/env python3
"""Merge objdump's lone-byte continuation lines back into their instruction.

objdump renders the scaled disp8 of EVEX-encoded instructions (e.g.
`vmovdqa32 0x40(%rsi,%r11,1),%zmm1` = 62 b1 7d 48 6f 4c 1e 01) with the
displacement byte on its own line:

    47:\t62 b1 7d 48 6f 4c 1e \tvmovdqa32 0x40(%rsi,%r11,1),%zmm1
    4e:\t01

The byte stream is correct (gas assembles the identical bytes and the
kernels are execution-verified); only the rendering is noisy. This filter
folds such lines back into the previous instruction line. A line is merged
only if ALL hold:
  - it carries exactly one byte and no disassembly text (a real 1-byte
    instruction would show its mnemonic; undecodable data would show
    `.byte`),
  - the previous line carries bytes (with or without mnemonic text --
    this also covers objdump splitting a long NOP it could not decode,
    e.g. `0f 1f 84 00 00 00 00` followed by a lone `00`),
  - address continuity holds: prev_addr + prev_nbytes == this addr.

Reads stdin, writes stdout. Exits 0 always (never fails the pipeline).
"""
import re
import sys

LINE = re.compile(r"^(\s*)([0-9a-f]+):\t((?:[0-9a-f]{2} )+)(.*)$")


def main() -> None:
    prev = None  # [indent, addr, nbytes, rest, bytes_s] or None

    def flush() -> None:
        nonlocal prev
        if prev is not None:
            indent, addr, _nbytes, rest, bytes_s = prev
            sys.stdout.write("%s%x:\t%s%s\n" % (indent, addr, bytes_s, rest))
            prev = None

    for raw in sys.stdin:
        line = raw.rstrip("\n")
        m = LINE.match(line)
        if not m:
            flush()
            sys.stdout.write(line + "\n")
            continue
        indent, addr_s, bytes_s, rest = m.groups()
        addr = int(addr_s, 16)
        nbytes = len(bytes_s.split())
        if (
            nbytes == 1
            and not rest.strip()
            and prev is not None
            and prev[1] + prev[2] == addr
        ):
            # Fold into the buffered instruction line.
            prev[2] += 1
            prev[4] += bytes_s
            continue
        flush()
        # Any addr:bytes line (decoded instruction or objdump's undecodable
        # fragment) is a merge target for a following lone byte; only
        # non-matching lines (blank, comments, labels) break the run.
        prev = [indent, addr, nbytes, rest, bytes_s]
    flush()


if __name__ == "__main__":
    main()
