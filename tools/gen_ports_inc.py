#!/usr/bin/env python3
"""Generate src/ports_data.inc from data/uarch/*.ports.

The .ports files are the source of truth (human-readable, reviewable).
This script embeds them as static C structs so the library has no
runtime file dependency. Adding a uarch = add a .ports file (no C
code change), then rebuild.

Usage: gen_ports_inc.py <data-dir> <output-inc>
"""

import os
import sys
import re


def parse_ports_file(path):
    """Parse a .ports file. Returns dict with uarch, ports, classes, scalars."""
    # Limits (must match src/ports.h RJ_PM_MAX_*)
    MAX_PORTS = 10
    MAX_CLASSES = 16
    MAX_NAME = 31
    IDENT_RE = re.compile(r'^[A-Za-z0-9_]+$')

    def check_name(name, what, lineno):
        if len(name) > MAX_NAME:
            raise ValueError(
                f"{path}:{lineno}: {what} name '{name}' exceeds {MAX_NAME} chars")
        if not IDENT_RE.match(name):
            raise ValueError(
                f"{path}:{lineno}: {what} name '{name}' must match [A-Za-z0-9_]+")

    uarch = None
    version = None
    ports = []  # list of (name, throughput)
    classes = []  # list of (name, uops_per_instr, [(port_idx, frac), ...])
    scalars = {}  # name -> value
    port_index = {}  # name -> idx
    seen_uarch = set()

    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('uarch:'):
                uarch = line.split(':', 1)[1].strip()
                check_name(uarch, 'uarch', lineno)
            elif line.startswith('version:'):
                version = int(line.split(':', 1)[1].strip())
            elif line.startswith('ports:'):
                for tok in line.split(':', 1)[1].strip().split():
                    name, val = tok.split('=')
                    check_name(name, 'port', lineno)
                    if name in port_index:
                        raise ValueError(
                            f"{path}:{lineno}: duplicate port '{name}'")
                    tput = float(val)
                    if tput <= 0.0:
                        raise ValueError(
                            f"{path}:{lineno}: port '{name}' throughput "
                            f"must be > 0 (got {tput})")
                    if len(ports) >= MAX_PORTS:
                        raise ValueError(
                            f"{path}:{lineno}: too many ports "
                            f"(max {MAX_PORTS})")
                    port_index[name] = len(ports)
                    ports.append((name, tput))
            elif line.startswith('class '):
                rest = line[len('class '):].strip()
                # Format: class <name> [uops=<n>]: <port>=<frac> ...
                if ':' not in rest:
                    raise ValueError(
                        f"{path}:{lineno}: class line missing ':'")
                head, distrs = rest.split(':', 1)
                head = head.strip()
                # Parse optional uops=<n> from head
                uops_per_instr = 1.0
                name_parts = []
                for tok in head.split():
                    if tok.startswith('uops='):
                        uops_per_instr = float(tok[5:])
                        if uops_per_instr <= 0.0:
                            raise ValueError(
                                f"{path}:{lineno}: uops must be > 0")
                    else:
                        name_parts.append(tok)
                name = ' '.join(name_parts).strip()
                check_name(name, 'class', lineno)
                # Check duplicate class names
                if any(c[0] == name for c in classes):
                    raise ValueError(
                        f"{path}:{lineno}: duplicate class '{name}'")
                if len(classes) >= MAX_CLASSES:
                    raise ValueError(
                        f"{path}:{lineno}: too many classes "
                        f"(max {MAX_CLASSES})")
                dist = []
                for tok in distrs.strip().split():
                    pname, frac = tok.split('=')
                    if pname not in port_index:
                        raise ValueError(
                            f"{path}:{lineno}: unknown port '{pname}'")
                    fval = float(frac)
                    if fval < 0.0 or fval > 1.0:
                        raise ValueError(
                            f"{path}:{lineno}: fraction for '{pname}' "
                            f"must be in [0,1] (got {fval})")
                    dist.append((port_index[pname], fval))
                # Validate fractions sum to 1.0
                total = sum(f for _, f in dist)
                if abs(total - 1.0) > 1e-6:
                    raise ValueError(
                        f"{path}:{lineno}: class '{name}' fractions sum to "
                        f"{total}, not 1.0")
                classes.append((name, uops_per_instr, dist))
            elif line.startswith('scalar '):
                rest = line[len('scalar '):].strip()
                name, val = rest.split('=')
                name = name.strip()
                check_name(name, 'scalar', lineno)
                scalars[name] = float(val)
            else:
                raise ValueError(f"{path}:{lineno}: unrecognized line: {line}")

    if uarch is None:
        raise ValueError(f"{path}: missing 'uarch:' line")
    if version != 1:
        raise ValueError(f"{path}: unsupported version {version}")
    if not ports:
        raise ValueError(f"{path}: no ports defined")
    return {
        'uarch': uarch,
        'ports': ports,
        'classes': classes,
        'scalars': scalars,
        'source': os.path.basename(path),
    }


def c_identifier(s):
    """Make a C identifier from a uarch name."""
    return re.sub(r'[^A-Za-z0-9]', '_', s).lower()


def generate(models, out_path):
    with open(out_path, 'w') as out:
        out.write("/* Auto-generated by tools/gen_ports_inc.py. DO NOT EDIT.\n")
        out.write(" * Source: data/uarch port files.\n")
        out.write(" */\n\n")

        for m in models:
            ident = c_identifier(m['uarch'])
            out.write(f"/* {m['uarch']} (from {m['source']}) */\n")
            # Ports
            out.write(f"static const rj_pm_port_t pm_ports_{ident}[] = {{\n")
            for name, tput in m['ports']:
                out.write(f'    {{"{name}", {tput}}},\n')
            out.write("};\n")
            # Classes (with uops_per_instr)
            out.write(f"static const rj_pm_class_t pm_classes_{ident}[] = {{\n")
            for cname, uops, dist in m['classes']:
                # Build dist array in port order (RJ_PM_MAX_PORTS=10)
                darr = [0.0] * 10
                for pidx, frac in dist:
                    darr[pidx] = frac
                dstr = ", ".join(f"{v:.4f}" for v in darr)
                out.write(f'    {{"{cname}", {uops:.1f}, {{{dstr}}}}},\n')
            out.write("};\n")
            # Model struct
            n_ports = len(m['ports'])
            n_classes = len(m['classes'])
            s = m['scalars']
            out.write(f"static const rj_port_model_t pm_model_{ident} = {{\n")
            out.write(f"    RJ_UARCH_{m['uarch']},  /* uarch */\n")
            out.write(f"    {n_ports},  /* n_ports */\n")
            # Ports array: need to copy into fixed-size array
            out.write("    {\n")
            for name, tput in m['ports']:
                out.write(f'        {{"{name}", {tput}}},\n')
            # Pad to RJ_PM_MAX_PORTS (10)
            for _ in range(10 - n_ports):
                out.write('        {"", 0.0},\n')
            out.write("    },\n")
            out.write(f"    {n_classes},  /* n_classes */\n")
            out.write("    {\n")
            for cname, uops, dist in m['classes']:
                darr = [0.0] * 10
                for pidx, frac in dist:
                    darr[pidx] = frac
                dstr = ", ".join(f"{v:.4f}" for v in darr)
                out.write(f'        {{"{cname}", {uops:.1f}, {{{dstr}}}}},\n')
            for _ in range(16 - n_classes):
                out.write('        {"", 0.0, {0}},\n')
            out.write("    },\n")
            out.write(f"    {s.get('vec_bw', 3.5)},  /* vec_bw */\n")
            out.write(f"    {s.get('mem_cpb', 0.015625)},  /* mem_cpb */\n")
            out.write(f"    {s.get('freq512', 1.0)},  /* freq512 */\n")
            out.write(f"    {s.get('bw_bpc', 8.0)},  /* bw_bpc */\n")
            out.write("};\n\n")

        # Registry: map uarch enum to model
        out.write("static const rj_port_model_t *pm_model_for_uarch"
                  "(rj_uarch_t u) {\n")
        out.write("    switch (u) {\n")
        for m in models:
            ident = c_identifier(m['uarch'])
            out.write(f"    case RJ_UARCH_{m['uarch']}:\n")
            out.write(f"        return &pm_model_{ident};\n")
        out.write("    default:\n")
        out.write("        return NULL;\n")
        out.write("    }\n")
        out.write("}\n")


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <data-dir> <output-inc>", file=sys.stderr)
        sys.exit(1)
    data_dir, out_path = sys.argv[1], sys.argv[2]
    models = []
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith('.ports'):
            models.append(parse_ports_file(os.path.join(data_dir, fname)))
    if not models:
        print(f"No .ports files in {data_dir}", file=sys.stderr)
        sys.exit(1)
    generate(models, out_path)
    print(f"Generated {out_path} from {len(models)} models")


if __name__ == '__main__':
    main()
