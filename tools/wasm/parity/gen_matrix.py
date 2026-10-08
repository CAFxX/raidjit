#!/usr/bin/env python3
"""Generate tools/wasm/parity/parity_matrix.json — the byte-identity
matrix (SPEC §S9.2). Each case: {name, module, expect, options}.

Conventions:
 - "module": x86 | aarch64 | riscv (selects the wasm module AND the
   native per-ARCH parity driver).
 - "expect": "ok" (both sides must succeed, bytes+metadata identical)
   or "fail" (both sides must fail; only the failure itself is compared).
 - aarch64 cases pin sve_vl=64 to match qemu-aarch64 -cpu max
   (cntb=64, verified at harness bring-up; see run_parity.sh).
 - riscv cases use rvv_vlen=16 = qemu-riscv64 -cpu max default.
"""
import json
import os

CASES = []


def case(name, module, options, expect="ok"):
    CASES.append({"name": name, "module": module, "expect": expect,
                  "options": options})


def opts(target, ctor, p1=-1, p2=-1, p3=-1, op="encode", cand="auto",
         cpu=None, options=None, matrix_hex=None):
    o = {"v": 1, "target": target,
         "topology": {"ctor": ctor}, "op": {"kind": op}, "cand": cand,
         "options": options or {}}
    t = o["topology"]
    if p1 >= 0:
        t["p1"] = p1
    if p2 >= 0:
        t["p2"] = p2
    if p3 >= 0:
        t["p3"] = p3
    if matrix_hex:
        t["matrix_hex"] = matrix_hex
    if cpu:
        o["cpu"] = cpu
    return o


def pinned(mul="nibble", U=0, **twins):
    c = {"mul": mul, "U": U}
    c.update(twins)
    return c


MULACC_MAT = "010000000000010204081020"  # 6x2 from gen_kernels mk_custom_mulacc

# ---------------- x86-64 ----------------
for tgt in ("avx512-gfni", "avx2", "sse"):
    case(f"x86/{tgt}/raid5-4/enc/auto", "x86",
         opts(tgt, "raid5", 4))
    case(f"x86/{tgt}/raid6-5/enc/auto", "x86",
         opts(tgt, "raid6", 5))

for ctor, p in (("raid6_rs", 5), ("rdp", 5), ("evenodd", 5),
                ("liberation", 4), ("lrc", None), ("lrc_azure", None)):
    if ctor == "lrc":
        o = opts("avx512-gfni", "lrc", 8, 2, 2)
    else:
        o = opts("avx512-gfni", ctor, p) if p else opts("avx512-gfni", ctor)
    case(f"x86/avx512-gfni/{ctor}/enc/auto", "x86", o)

case("x86/avx512-gfni/rs63/enc/auto", "x86", opts("avx512-gfni", "rs", 6, 3))
case("x86/avx512-gfni/custom-mulacc/enc/auto", "x86",
     opts("avx512-gfni", "custom", 6, 2, matrix_hex=MULACC_MAT))
case("x86/avx512-gfni/raid6-5/rec0/auto", "x86",
     opts("avx512-gfni", "raid6", 5, op="recover",
          cand="auto") | {"op": {"kind": "recover", "failed": [0]}})
case("x86/avx512-gfni/raid6-5/pupd0/auto", "x86",
     opts("avx512-gfni", "raid6", 5) |
     {"op": {"kind": "parity_update", "changed": [0]}})

# pinned twins (avx512-gfni)
case("x86/twin/xorsched", "x86",
     opts("avx512-gfni", "rdp", 5, cand=pinned(xorsched=True)))
case("x86/twin/nt", "x86",
     opts("avx512-gfni", "raid5", 4, cand=pinned(nt=True)))
# notail/maskedtail are stage-2 length-dispatch flags: not pinnable via
# the candidate API. Both sides must reject them identically.
case("x86/twin/notail", "x86",
     opts("avx512-gfni", "raid5", 4, cand=pinned(notail=True)),
     expect="fail")
case("x86/twin/maskedtail", "x86",
     opts("avx512-gfni", "raid5", 4, cand=pinned(maskedtail=True)),
     expect="fail")
case("x86/twin/prefetch", "x86",
     opts("avx512-gfni", "raid5", 4, cand=pinned(prefetch_dist=512)))
case("x86/twin/ternlog", "x86",
     opts("avx512-gfni", "rs", 6, 3, cand=pinned(mul="nibble",
                                                  ternlog=True)))
case("x86/twin/permi2b", "x86",
     opts("avx512-gfni", "rs", 6, 3, cand=pinned(permi2b=True)))
case("x86/twin/mshift", "x86",
     opts("avx512-gfni", "rs", 6, 3, cand=pinned(mshift=True)))
case("x86/twin/xorbit", "x86",
     opts("avx512-gfni", "raid6_rs", 5, cand=pinned(xorbit=True)))
case("x86/twin/remat", "x86",
     opts("avx512-gfni", "custom", 6, 4,
          matrix_hex="0101010101012010080402016c1b401004012fab36400801",
          cand=pinned(remat=True)))
case("x86/twin/U2", "x86",
     opts("avx512-gfni", "raid5", 4, cand=pinned(U=2)))
case("x86/twin/mul-gfni", "x86",
     opts("avx512-gfni", "rs", 6, 3, cand=pinned(mul="gfni")))
case("x86/twin/endbr", "x86",
     opts("avx512-gfni", "raid5", 4, cand=pinned(),
          options={"emit_endbr": True}))
# expected failures (both sides must fail cleanly)
case("x86/err/unknown-target", "x86", opts("nope", "raid5", 4),
     expect="fail")
case("x86/err/U9", "x86",
     opts("avx512-gfni", "raid5", 4, cand=pinned(U=9)), expect="fail")
case("x86/err/ternlog-on-sse", "x86",
     opts("sse", "rs", 6, 3, cand=pinned(ternlog=True)), expect="fail")
case("x86/err/maskedtail-xor-only", "x86",
     opts("avx512-gfni", "rs", 6, 3, cand=pinned(maskedtail=True)),
     expect="fail")

# ---------------- AArch64 ----------------
A64CPU = {"sve_vl": 64}
for ctor_args in (("raid5", 4), ("raid6", 5), ("rs", 6, 3)):
    ctor = ctor_args[0]
    o = opts("sve2", ctor, *ctor_args[1:], cpu=A64CPU)
    case(f"a64/sve2/{ctor}/enc/auto", "aarch64", o)
case("a64/neon/raid5-4/enc/auto", "aarch64", opts("neon", "raid5", 4))
case("a64/sve2/raid6-5/rec0/auto", "aarch64",
     opts("sve2", "raid6", 5, cpu=A64CPU) |
     {"op": {"kind": "recover", "failed": [0]}})
case("a64/twin/pipe", "aarch64",
     opts("sve2", "raid6_rs", 5, cand=pinned(pipe=True), cpu=A64CPU))
case("a64/twin/predtail", "aarch64",
     opts("sve2", "raid5", 4, cand=pinned(predtail=True), cpu=A64CPU))
case("a64/twin/remat", "aarch64",
     opts("sve2", "custom", 6, 4,
          matrix_hex="0101010101012010080402016c1b401004012fab36400801",
          cand=pinned(remat=True), cpu=A64CPU))
case("a64/twin/nt", "aarch64",
     opts("sve2", "raid5", 4, cand=pinned(nt=True), cpu=A64CPU))
case("a64/twin/prefetch", "aarch64",
     opts("sve2", "raid5", 4, cand=pinned(prefetch_dist=128), cpu=A64CPU))

# ---------------- RISC-V ----------------
for ctor, p in (("raid5", 4), ("raid6", 5)):
    case(f"rv/rvv/{ctor}/enc/auto", "riscv", opts("rvv", ctor, p))
case("rv/rvv/raid6-5/rec0/auto", "riscv",
     opts("rvv", "raid6", 5) | {"op": {"kind": "recover", "failed": [0]}})
for lmul in (1, 2, 4, 8):
    case(f"rv/twin/lmul{lmul}", "riscv",
         opts("rvv", "raid5", 4, cand=pinned(lmul=lmul)))
case("rv/twin/pipe", "riscv",
     opts("rvv", "raid6_rs", 5, cand=pinned(pipe=True)))
case("rv/twin/nt", "riscv",
     opts("rvv", "raid5", 4, cand=pinned(nt=True)))
case("rv/twin/prefetch", "riscv",
     opts("rvv-zicbop", "raid5", 4, cand=pinned(prefetch_dist=64)))
case("rv/twin/zvbc", "riscv",
     opts("rvv-zvbc32e", "rs", 6, 3, cand=pinned(mul="zvbc")))


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "parity_matrix.json")
    with open(out, "w") as f:
        json.dump(CASES, f, indent=1)
    print(f"wrote {len(CASES)} cases to {out}")


if __name__ == "__main__":
    main()
