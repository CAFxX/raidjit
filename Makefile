CC = gcc
CFLAGS = -O2 -std=c11 -Wall -Wextra -Werror -D_POSIX_C_SOURCE=200809L \
         -Iinclude -Isrc
LDFLAGS = -lm

# Architecture selection: ARCH=x86_64 (default, from uname -m), aarch64, riscv64.
# For cross builds: make ARCH=aarch64 CROSS_COMPILE=aarch64-linux-gnu-
ARCH ?= $(shell uname -m)
CROSS_COMPILE ?=
CC := $(CROSS_COMPILE)$(CC)

# Architecture-specific build directory: prevents "file in wrong format"
# errors when switching ARCH without a full clean. All objects, libraries,
# and test binaries live under $(BUILDDIR).
BUILDDIR = build-$(ARCH)

COMMON_SRCS = src/gf.c src/plan.c src/topo.c src/bench.c src/ref.c src/api.c \
        src/prune.c src/dispatch.c src/xorsched.c src/matrix_search.c \
        src/xorcode.c src/survivor.c src/prefetch_calib.c src/planid.c \
        src/crc.c src/striping.c src/api_crc.c src/fused.c \
        src/api_batched.c
ifeq ($(ARCH),x86_64)
ARCH_SRCS = src/x86_64/cpu.c src/x86_64/emit.c src/x86_64/jit.c
else ifeq ($(ARCH),aarch64)
ARCH_SRCS = src/aarch64/cpu.c src/aarch64/emit.c src/aarch64/jit.c
else ifeq ($(ARCH),riscv64)
ARCH_SRCS = src/riscv/cpu.c src/riscv/emit.c src/riscv/jit.c
else
$(error Unsupported ARCH=$(ARCH); want x86_64, aarch64, or riscv64)
endif

SRCS = $(COMMON_SRCS) $(ARCH_SRCS)
# Objects live in the arch-specific build dir, mirroring source paths.
OBJS = $(addprefix $(BUILDDIR)/,$(SRCS:.c=.o))
HDRS = $(wildcard include/*.h src/*.h src/aarch64/*.h src/riscv/*.h src/x86_64/*.h)

# Portable tests run on all architectures; x86-only tests (instruction
# encodings, CET, GFNI affine, alignment scan) only on x86_64.
# test_aarch64_emit is pure C (byte-level encoding checks) and runs on any host.
PORTABLE_TESTS = $(BUILDDIR)/test_correct $(BUILDDIR)/test_benchreject \
        $(BUILDDIR)/test_select $(BUILDDIR)/test_matrix \
        $(BUILDDIR)/test_nt $(BUILDDIR)/test_pdelta $(BUILDDIR)/test_large_mds \
        $(BUILDDIR)/test_aarch64_emit $(BUILDDIR)/test_pressure \
        $(BUILDDIR)/test_sve_pdelta $(BUILDDIR)/test_aarch64_remat \
        $(BUILDDIR)/test_sve_predtail \
        $(BUILDDIR)/test_riscv_remat $(BUILDDIR)/test_sve_eor3 \
        $(BUILDDIR)/test_riscv_prefetch \
        $(BUILDDIR)/test_riscv_zvbc \
        $(BUILDDIR)/test_riscv_zvbb \
        $(BUILDDIR)/test_rvv_f6 $(BUILDDIR)/test_rvv_lmul \
        $(BUILDDIR)/test_pdelta_edge \
        $(BUILDDIR)/test_boundaries $(BUILDDIR)/test_aarch64_shape \
        $(BUILDDIR)/test_prune $(BUILDDIR)/test_dispatch \
        $(BUILDDIR)/test_xorsched $(BUILDDIR)/test_xorsched_e2e \
        $(BUILDDIR)/test_matrix_search $(BUILDDIR)/test_xorcode \
        $(BUILDDIR)/test_notail_xorsched $(BUILDDIR)/test_survivor \
        $(BUILDDIR)/test_tailduel $(BUILDDIR)/test_b4_prefetch \
        $(BUILDDIR)/test_prefetch_calib \
        $(BUILDDIR)/test_traps $(BUILDDIR)/test_planner_fuzz \
        $(BUILDDIR)/test_recover_closedform \
        $(BUILDDIR)/test_pipe $(BUILDDIR)/test_crc $(BUILDDIR)/test_fused $(BUILDDIR)/test_striping $(BUILDDIR)/test_planid \
        $(BUILDDIR)/test_batched $(BUILDDIR)/test_lrc \
        $(BUILDDIR)/test_twin_audit
X86_ONLY_TESTS = $(BUILDDIR)/test_emit $(BUILDDIR)/test_affine $(BUILDDIR)/test_cet \
        $(BUILDDIR)/test_scan_align $(BUILDDIR)/test_remat \
        $(BUILDDIR)/test_codegen_shape $(BUILDDIR)/test_pxor_ternlog \
        $(BUILDDIR)/test_maskedtail $(BUILDDIR)/test_permi2b \
        $(BUILDDIR)/test_xorbit $(BUILDDIR)/test_mshift
ifeq ($(ARCH),x86_64)
TESTS = $(PORTABLE_TESTS) $(X86_ONLY_TESTS)
else
TESTS = $(PORTABLE_TESTS)
endif

all: $(BUILDDIR)/libraidjit.a $(TESTS) $(BUILDDIR)/demo

# QEMU test targets for cross builds (run on x86_64 host). The portable
# tests run under qemu; x86-only tests are skipped. These targets enforce
# their ARCH via sub-make so plain `make test-qemu-aarch64` does the right
# thing even when invoked without ARCH/CROSS_COMPILE.
QEMU_aarch64 = qemu-aarch64
QEMU_riscv64 = qemu-riscv64
# QEMU 8.2.2's -cpu max advertises Zvbc/Zvkg in the guest /proc/cpuinfo
# "isa" line but cannot execute vclmul.vv/vclmulh.vv (illegal
# instruction; the vector-crypto TCG support postdates 8.2.2). It does
# not know Zvbc32e at all (no token; the property is rejected as
# unknown), so the rvv_zvbc32e gate keeps the MUL_ZVBC twin off under
# QEMU regardless. The zvbc=false,zvkg=false flags are retained to
# document the advertise-but-cannot-execute mismatch. Zicond executes
# correctly and stays enabled. Append ,vlen=N for the multi-VLEN runs.
QEMU_RVV_CPU = max,zvbc=false,zvkg=false

test-qemu-aarch64:
	$(MAKE) ARCH=aarch64 CROSS_COMPILE=aarch64-linux-gnu- _test-qemu-aarch64

_test-qemu-aarch64: $(TESTS)
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max ./$(BUILDDIR)/test_aarch64_emit
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max ./$(BUILDDIR)/test_nt
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max ./$(BUILDDIR)/test_b4_prefetch
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max ./$(BUILDDIR)/test_traps
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max ./$(BUILDDIR)/test_pdelta
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max ./$(BUILDDIR)/test_pdelta_edge
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max ./$(BUILDDIR)/test_boundaries
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max ./$(BUILDDIR)/test_large_mds
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max ./$(BUILDDIR)/test_select
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max ./$(BUILDDIR)/test_benchreject
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max ./$(BUILDDIR)/test_correct
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max ./$(BUILDDIR)/test_matrix
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max ./$(BUILDDIR)/test_aarch64_remat
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max ./$(BUILDDIR)/test_pipe
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max ./$(BUILDDIR)/test_aarch64_shape
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max ./$(BUILDDIR)/test_dispatch
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max,sve-default-vector-length=256 ./$(BUILDDIR)/test_dispatch
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max ./$(BUILDDIR)/test_xorsched
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max ./$(BUILDDIR)/test_xorsched_e2e
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max ./$(BUILDDIR)/test_matrix_search
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max ./$(BUILDDIR)/test_xorcode
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max ./$(BUILDDIR)/test_prune
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max ./$(BUILDDIR)/test_notail_xorsched
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max ./$(BUILDDIR)/test_survivor

# SVE/SVE2 testing under QEMU: -cpu max enables SVE, but the default
# vector length is 64 bits (too small; the test skips SVE). Set an
# explicit VL. (A historical note here claimed test_matrix segfaulted
# with SVE enabled. That does not reproduce on the current tree
# (128k+ runs at VL128/256/512, 0 failures); a peer-reviewed experiment
# refuted the suspected root cause (see docs/limitations.md §12), so the
# historical crash's true cause is unknown — likely a flawed observation
# (stale binary / misreported SIGILL). test_matrix is back in the matrix.)
test-qemu-aarch64-sve:
	$(MAKE) ARCH=aarch64 CROSS_COMPILE=aarch64-linux-gnu- _test-qemu-aarch64-sve

_test-qemu-aarch64-sve: $(TESTS)
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max,sve-default-vector-length=128 ./$(BUILDDIR)/test_aarch64_emit
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max,sve-default-vector-length=128 ./$(BUILDDIR)/test_correct
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max,sve-default-vector-length=128 ./$(BUILDDIR)/test_sve_pdelta
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max,sve-default-vector-length=128 ./$(BUILDDIR)/test_sve_predtail
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max,sve-default-vector-length=128 ./$(BUILDDIR)/test_sve_eor3
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max,sve-default-vector-length=128 ./$(BUILDDIR)/test_aarch64_remat
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max,sve-default-vector-length=128 ./$(BUILDDIR)/test_aarch64_shape
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max,sve-default-vector-length=128 ./$(BUILDDIR)/test_pdelta_edge
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max,sve-default-vector-length=128 ./$(BUILDDIR)/test_boundaries
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max,sve-default-vector-length=128 ./$(BUILDDIR)/test_matrix
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max,sve-default-vector-length=128 ./$(BUILDDIR)/test_dispatch
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max,sve-default-vector-length=128 ./$(BUILDDIR)/test_xorsched
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max,sve-default-vector-length=128 ./$(BUILDDIR)/test_xorsched_e2e
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max,sve-default-vector-length=128 ./$(BUILDDIR)/test_matrix_search
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max,sve-default-vector-length=128 ./$(BUILDDIR)/test_xorcode
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max,sve-default-vector-length=128 ./$(BUILDDIR)/test_prune
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max,sve-default-vector-length=128 ./$(BUILDDIR)/test_notail_xorsched
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max,sve-default-vector-length=128 ./$(BUILDDIR)/test_survivor
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max,sve-default-vector-length=256 ./$(BUILDDIR)/test_aarch64_emit
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max,sve-default-vector-length=256 ./$(BUILDDIR)/test_correct
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max,sve-default-vector-length=256 ./$(BUILDDIR)/test_sve_pdelta
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max,sve-default-vector-length=256 ./$(BUILDDIR)/test_sve_predtail
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max,sve-default-vector-length=256 ./$(BUILDDIR)/test_sve_eor3
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max,sve-default-vector-length=256 ./$(BUILDDIR)/test_aarch64_remat
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max,sve-default-vector-length=256 ./$(BUILDDIR)/test_aarch64_shape
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max,sve-default-vector-length=256 ./$(BUILDDIR)/test_pdelta_edge
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max,sve-default-vector-length=256 ./$(BUILDDIR)/test_boundaries
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max,sve-default-vector-length=512 ./$(BUILDDIR)/test_boundaries
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max,sve-default-vector-length=256 ./$(BUILDDIR)/test_matrix
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max,sve-default-vector-length=256 ./$(BUILDDIR)/test_dispatch
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max,sve-default-vector-length=512 ./$(BUILDDIR)/test_dispatch
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max,sve-default-vector-length=256 ./$(BUILDDIR)/test_xorsched
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max,sve-default-vector-length=512 ./$(BUILDDIR)/test_xorsched
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max,sve-default-vector-length=256 ./$(BUILDDIR)/test_xorsched_e2e
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max,sve-default-vector-length=256 ./$(BUILDDIR)/test_matrix_search
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max,sve-default-vector-length=256 ./$(BUILDDIR)/test_xorcode
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max,sve-default-vector-length=512 ./$(BUILDDIR)/test_xorcode
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max,sve-default-vector-length=512 ./$(BUILDDIR)/test_sve_predtail
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max,sve-default-vector-length=512 ./$(BUILDDIR)/test_survivor
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max,sve-default-vector-length=256 ./$(BUILDDIR)/test_prune
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max,sve-default-vector-length=256 ./$(BUILDDIR)/test_notail_xorsched
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu $(QEMU_aarch64) -cpu max,sve-default-vector-length=256 ./$(BUILDDIR)/test_survivor
	# The test itself also covers the non-SVE2 baseline path via a
	# synthetic cpu (sve2 force-disabled); QEMU's neoverse-n1 model does
	# not advertise SVE hwcaps, so it cannot serve as the baseline host.

test-qemu-riscv64:
	$(MAKE) ARCH=riscv64 CROSS_COMPILE=riscv64-linux-gnu- _test-qemu-riscv64

_test-qemu-riscv64: $(TESTS)
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU) ./$(BUILDDIR)/test_nt
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU) ./$(BUILDDIR)/test_b4_prefetch
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU) ./$(BUILDDIR)/test_traps
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU) ./$(BUILDDIR)/test_pdelta
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU) ./$(BUILDDIR)/test_pdelta_edge
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU) ./$(BUILDDIR)/test_boundaries
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU),vlen=256 ./$(BUILDDIR)/test_boundaries
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU),vlen=512 ./$(BUILDDIR)/test_boundaries
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU) ./$(BUILDDIR)/test_large_mds
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU) ./$(BUILDDIR)/test_correct
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU) ./$(BUILDDIR)/test_matrix
	# Multi-VLEN RVV: the default VLEN is 128; also test 256 and 512.
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU),vlen=256 ./$(BUILDDIR)/test_correct
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU),vlen=256 ./$(BUILDDIR)/test_matrix
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU),vlen=512 ./$(BUILDDIR)/test_correct
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU),vlen=512 ./$(BUILDDIR)/test_matrix
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU) ./$(BUILDDIR)/test_riscv_remat
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU),vlen=256 ./$(BUILDDIR)/test_riscv_remat
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU),vlen=512 ./$(BUILDDIR)/test_riscv_remat
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU) ./$(BUILDDIR)/test_pipe
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU),vlen=256 ./$(BUILDDIR)/test_pipe
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU),vlen=512 ./$(BUILDDIR)/test_pipe
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU) ./$(BUILDDIR)/test_riscv_prefetch
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU),vlen=256 ./$(BUILDDIR)/test_riscv_prefetch
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU),vlen=512 ./$(BUILDDIR)/test_riscv_prefetch
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU) ./$(BUILDDIR)/test_riscv_zvbc
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU),vlen=256 ./$(BUILDDIR)/test_riscv_zvbc
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU),vlen=512 ./$(BUILDDIR)/test_riscv_zvbc
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU),zvbb=true ./$(BUILDDIR)/test_riscv_zvbb
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU),zvbb=true,vlen=256 ./$(BUILDDIR)/test_riscv_zvbb
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU),zvbb=true,vlen=512 ./$(BUILDDIR)/test_riscv_zvbb
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU) ./$(BUILDDIR)/test_rvv_f6
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU),vlen=256 ./$(BUILDDIR)/test_rvv_f6
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU),vlen=512 ./$(BUILDDIR)/test_rvv_f6
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU) ./$(BUILDDIR)/test_rvv_lmul
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU),vlen=256 ./$(BUILDDIR)/test_rvv_lmul
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU),vlen=512 ./$(BUILDDIR)/test_rvv_lmul
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU) ./$(BUILDDIR)/test_dispatch
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU),vlen=256 ./$(BUILDDIR)/test_dispatch
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU),vlen=512 ./$(BUILDDIR)/test_dispatch
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU) ./$(BUILDDIR)/test_xorsched
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU) ./$(BUILDDIR)/test_xorsched_e2e
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU),vlen=256 ./$(BUILDDIR)/test_xorsched
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU),vlen=256 ./$(BUILDDIR)/test_xorsched_e2e
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU),vlen=512 ./$(BUILDDIR)/test_xorsched
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU),vlen=512 ./$(BUILDDIR)/test_xorsched_e2e
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU) ./$(BUILDDIR)/test_matrix_search
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU),vlen=256 ./$(BUILDDIR)/test_matrix_search
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU),vlen=512 ./$(BUILDDIR)/test_matrix_search
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU) ./$(BUILDDIR)/test_xorcode
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU),vlen=256 ./$(BUILDDIR)/test_xorcode
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU),vlen=512 ./$(BUILDDIR)/test_xorcode
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU) ./$(BUILDDIR)/test_prune
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU),vlen=256 ./$(BUILDDIR)/test_prune
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU),vlen=512 ./$(BUILDDIR)/test_prune
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU) ./$(BUILDDIR)/test_notail_xorsched
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU),vlen=256 ./$(BUILDDIR)/test_notail_xorsched
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU),vlen=512 ./$(BUILDDIR)/test_notail_xorsched
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU) ./$(BUILDDIR)/test_survivor
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU),vlen=256 ./$(BUILDDIR)/test_survivor
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu $(QEMU_riscv64) -cpu $(QEMU_RVV_CPU),vlen=512 ./$(BUILDDIR)/test_survivor

$(OBJS): $(HDRS)

# Pattern rule: compile src/foo.c -> $(BUILDDIR)/src/foo.o, creating
# subdirectories as needed.
$(BUILDDIR)/%.o: %.c | $(BUILDDIR)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILDDIR)/libraidjit.a: $(OBJS) | $(BUILDDIR)
	ar rcs $@ $(OBJS)

$(BUILDDIR)/test_correct: tests/test_correct.c $(BUILDDIR)/libraidjit.a | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_correct.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)

$(BUILDDIR)/test_emit: tests/test_emit.c src/x86_64/emit.c $(HDRS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_emit.c src/x86_64/emit.c $(LDFLAGS)

$(BUILDDIR)/test_aarch64_emit: tests/test_aarch64_emit.c src/aarch64/emit.c $(HDRS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_aarch64_emit.c src/aarch64/emit.c $(LDFLAGS)

$(BUILDDIR)/test_affine: tests/test_affine.c $(BUILDDIR)/libraidjit.a | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_affine.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)

$(BUILDDIR)/test_select: tests/test_select.c $(BUILDDIR)/libraidjit.a | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_select.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)

$(BUILDDIR)/test_benchreject: tests/test_benchreject.c $(BUILDDIR)/libraidjit.a | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_benchreject.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)

$(BUILDDIR)/test_matrix: tests/test_matrix.c $(BUILDDIR)/libraidjit.a | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_matrix.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)

$(BUILDDIR)/test_pressure: tests/test_pressure.c $(BUILDDIR)/libraidjit.a | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_pressure.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)

$(BUILDDIR)/test_remat: tests/test_remat.c $(BUILDDIR)/libraidjit.a | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_remat.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)

$(BUILDDIR)/test_cet: tests/test_cet.c $(BUILDDIR)/libraidjit.a | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_cet.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)

$(BUILDDIR)/demo: examples/demo.c $(BUILDDIR)/libraidjit.a | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ examples/demo.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

clean:
	rm -rf build build-x86_64 build-aarch64 build-riscv64
	rm -f src/*.o src/aarch64/*.o src/riscv/*.o

test: $(TESTS)
	$(if $(CROSS_COMPILE),$(error cannot run cross-compiled tests natively; use test-qemu-aarch64 or test-qemu-riscv64))
	./$(BUILDDIR)/test_aarch64_emit
ifeq ($(ARCH),x86_64)
	./$(BUILDDIR)/test_emit
	./$(BUILDDIR)/test_affine
	./$(BUILDDIR)/test_cet
	./$(BUILDDIR)/test_scan_align
	./$(BUILDDIR)/test_remat
	./$(BUILDDIR)/test_codegen_shape
	./$(BUILDDIR)/test_pxor_ternlog
	./$(BUILDDIR)/test_maskedtail
	./$(BUILDDIR)/test_permi2b
	./$(BUILDDIR)/test_xorbit
	./$(BUILDDIR)/test_mshift
endif
	./$(BUILDDIR)/test_nt
	./$(BUILDDIR)/test_pdelta
	./$(BUILDDIR)/test_pdelta_edge
	./$(BUILDDIR)/test_boundaries
	./$(BUILDDIR)/test_large_mds
	./$(BUILDDIR)/test_pressure
	./$(BUILDDIR)/test_select
	./$(BUILDDIR)/test_benchreject
	./$(BUILDDIR)/test_correct
	./$(BUILDDIR)/test_matrix
	./$(BUILDDIR)/test_prune
	./$(BUILDDIR)/test_dispatch
	./$(BUILDDIR)/test_xorsched
	./$(BUILDDIR)/test_xorsched_e2e
	./$(BUILDDIR)/test_matrix_search
	./$(BUILDDIR)/test_xorcode
	./$(BUILDDIR)/test_notail_xorsched
	./$(BUILDDIR)/test_survivor
	./$(BUILDDIR)/test_tailduel
	./$(BUILDDIR)/test_b4_prefetch
	./$(BUILDDIR)/test_prefetch_calib
	./$(BUILDDIR)/test_traps
	./$(BUILDDIR)/test_planner_fuzz
	./$(BUILDDIR)/test_recover_closedform
	./$(BUILDDIR)/test_pipe
	./$(BUILDDIR)/test_crc
	./$(BUILDDIR)/test_fused
	./$(BUILDDIR)/test_striping
	./$(BUILDDIR)/test_planid
	./$(BUILDDIR)/test_batched
	./$(BUILDDIR)/test_lrc
	./$(BUILDDIR)/test_twin_audit

# Sanitizer build: all tests under ASan+UBSan (leaks, OOB, UB).
test-asan: clean
	$(MAKE) CC="$(CC)" CFLAGS="-O1 -g -std=c11 -Wall -Wextra -Werror \
	    -D_POSIX_C_SOURCE=200809L -Iinclude -Isrc \
	    -fsanitize=address,undefined -fno-omit-frame-pointer" \
	    LDFLAGS="-lm -fsanitize=address,undefined" test

# Foreign-CPU coverage: run the matrix under qemu-x86_64 with emulated
# CPU models (needs qemu-user installed). Uses real CPUID detection.
# Builds a dedicated non-sanitizer binary: LSan cannot initialize under
# qemu-user (ptrace restriction), so the test-asan artifacts are unusable
# here. Note: qemu 8.2 TCG does not implement AVX-512/GFNI; QEMU warns and
# drops those features, so the Skylake/Icelake-Server models exercise
# detection plus AVX2-and-below execution paths (AVX-512 execution itself
# is covered by the native matrix on AVX-512 hosts).
test-qemu: clean
	$(MAKE) $(BUILDDIR)/test_matrix $(BUILDDIR)/test_xorcode
	tests/qemu_matrix.sh
	tests/qemu_matrix.sh $(BUILDDIR)/test_xorcode

.PHONY: all clean test test-asan test-qemu test-qemu-aarch64 test-qemu-aarch64-sve test-qemu-riscv64 _test-qemu-aarch64 _test-qemu-aarch64-sve _test-qemu-riscv64 kernels kernels-gens kernels-x86_64 kernels-aarch64 kernels-riscv64 kernels-redgreen

# Prebuilt kernels (user-facing artifacts in
# ~/workspace/your_files/raidjit/kernels/).
#
# Private build dirs (build-kernels-*): this never touches
# build-x86_64/-aarch64/-riscv64 and never runs `clean`, so it is safe to
# run while other work happens in this tree. Generation order is x86_64,
# aarch64, then riscv64 LAST (the riscv backend may still be in flux, so
# its kernels are generated immediately before packaging and verified
# then). The full pipeline is one command: `make kernels`.
KERNELS_STAGE = /tmp/kernels-stage
# Note: no locking on $(KERNELS_STAGE); two concurrent `make kernels` runs
# would collide there and on the install DEST. Single use only.

$(BUILDDIR)/gen_kernels: tools/gen_kernels.c $(BUILDDIR)/libraidjit.a $(HDRS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tools/gen_kernels.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)

# Note: `CC=` on the make command line would override the
# CC := $(CROSS_COMPILE)$(CC) computation in the sub-makes below (same
# pre-existing pattern as the qemu targets); don't do that.

# Fail fast if the cross toolchain or qemu-user is missing, before any
# bins are staged.
kernels-gens:
	command -v aarch64-linux-gnu-gcc riscv64-linux-gnu-gcc \
	    qemu-aarch64 qemu-riscv64 >/dev/null
	$(MAKE) ARCH=x86_64 CROSS_COMPILE= BUILDDIR=build-kernels-x86_64 build-kernels-x86_64/gen_kernels
	$(MAKE) ARCH=aarch64 CROSS_COMPILE=aarch64-linux-gnu- BUILDDIR=build-kernels-aarch64 build-kernels-aarch64/gen_kernels

kernels-x86_64: kernels-gens
	mkdir -p $(KERNELS_STAGE)
	./build-kernels-x86_64/gen_kernels avx512-gfni $(KERNELS_STAGE)

# Chained prerequisites (not just order) so the x86_64 -> aarch64 ->
# riscv64 order holds under -j as well.
kernels-aarch64: kernels-x86_64
	mkdir -p $(KERNELS_STAGE)
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu qemu-aarch64 -cpu max,sve-default-vector-length=128 ./build-kernels-aarch64/gen_kernels sve2 $(KERNELS_STAGE)

# The riscv64 generator binary is rebuilt here, immediately before its run,
# because the riscv backend may still be in flux (another agent is
# surveying it in this tree concurrently).
kernels-riscv64: kernels-aarch64
	$(MAKE) ARCH=riscv64 CROSS_COMPILE=riscv64-linux-gnu- BUILDDIR=build-kernels-riscv64 build-kernels-riscv64/gen_kernels
	mkdir -p $(KERNELS_STAGE)
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu qemu-riscv64 -cpu max ./build-kernels-riscv64/gen_kernels rvv $(KERNELS_STAGE)

# Red-green proof: corruption must be detected. Runs in /tmp only; the
# outputs are never shipped.
kernels-redgreen: kernels-gens
	$(MAKE) ARCH=riscv64 CROSS_COMPILE=riscv64-linux-gnu- BUILDDIR=build-kernels-riscv64 build-kernels-riscv64/gen_kernels
	mkdir -p /tmp/kernels-rg
	./build-kernels-x86_64/gen_kernels avx512-gfni /tmp/kernels-rg --redgreen
	QEMU_LD_PREFIX=/usr/aarch64-linux-gnu qemu-aarch64 -cpu max,sve-default-vector-length=128 ./build-kernels-aarch64/gen_kernels sve2 /tmp/kernels-rg --redgreen
	QEMU_LD_PREFIX=/usr/riscv64-linux-gnu qemu-riscv64 -cpu max ./build-kernels-riscv64/gen_kernels rvv /tmp/kernels-rg --redgreen

kernels: kernels-x86_64 kernels-aarch64 kernels-riscv64
	tools/finalize_kernels.sh $(KERNELS_STAGE)

$(BUILDDIR)/test_scan_align: tests/test_scan_align.c $(BUILDDIR)/libraidjit.a $(HDRS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_scan_align.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)

$(BUILDDIR)/test_codegen_shape: tests/test_codegen_shape.c $(BUILDDIR)/libraidjit.a $(HDRS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_codegen_shape.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)

$(BUILDDIR)/test_pxor_ternlog: tests/test_pxor_ternlog.c $(BUILDDIR)/libraidjit.a $(HDRS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_pxor_ternlog.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)

$(BUILDDIR)/test_maskedtail: tests/test_maskedtail.c $(BUILDDIR)/libraidjit.a $(HDRS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_maskedtail.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)

$(BUILDDIR)/test_permi2b: tests/test_permi2b.c $(BUILDDIR)/libraidjit.a $(HDRS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_permi2b.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)

$(BUILDDIR)/test_tailduel: tests/test_tailduel.c $(BUILDDIR)/libraidjit.a $(HDRS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_tailduel.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)

$(BUILDDIR)/test_b4_prefetch: tests/test_b4_prefetch.c $(BUILDDIR)/libraidjit.a $(HDRS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_b4_prefetch.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)

$(BUILDDIR)/test_prefetch_calib: tests/test_prefetch_calib.c $(BUILDDIR)/libraidjit.a $(HDRS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_prefetch_calib.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)

$(BUILDDIR)/test_traps: tests/test_traps.c $(BUILDDIR)/libraidjit.a $(HDRS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_traps.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)

$(BUILDDIR)/test_nt: tests/test_nt.c $(BUILDDIR)/libraidjit.a $(HDRS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_nt.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)

$(BUILDDIR)/test_large_mds: tests/test_large_mds.c $(BUILDDIR)/libraidjit.a $(HDRS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_large_mds.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)

$(BUILDDIR)/test_pdelta: tests/test_pdelta.c $(BUILDDIR)/libraidjit.a $(HDRS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_pdelta.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)

$(BUILDDIR)/test_pdelta_edge: tests/test_pdelta_edge.c $(BUILDDIR)/libraidjit.a $(HDRS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_pdelta_edge.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)

$(BUILDDIR)/test_boundaries: tests/test_boundaries.c $(BUILDDIR)/libraidjit.a $(HDRS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_boundaries.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)

$(BUILDDIR)/test_aarch64_shape: tests/test_aarch64_shape.c $(BUILDDIR)/libraidjit.a $(HDRS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_aarch64_shape.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)

$(BUILDDIR)/test_prune: tests/test_prune.c $(BUILDDIR)/libraidjit.a $(HDRS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_prune.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)

# Perf guard tool (tools/perf_guard.c): steady-state throughput +
# compile-time report over the config corpus, with a --compare mode for
# before/after runs (see the file header). Not part of `all`.
$(BUILDDIR)/perf_guard: tools/perf_guard.c $(BUILDDIR)/libraidjit.a $(HDRS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tools/perf_guard.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)

$(BUILDDIR)/test_matrix_search: tests/test_matrix_search.c $(BUILDDIR)/libraidjit.a $(HDRS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_matrix_search.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)

$(BUILDDIR)/test_sve_pdelta: tests/test_sve_pdelta.c $(BUILDDIR)/libraidjit.a $(HDRS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_sve_pdelta.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)

$(BUILDDIR)/test_sve_predtail: tests/test_sve_predtail.c $(BUILDDIR)/libraidjit.a $(HDRS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_sve_predtail.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)

$(BUILDDIR)/test_survivor: tests/test_survivor.c $(BUILDDIR)/libraidjit.a $(HDRS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_survivor.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)

$(BUILDDIR)/test_sve_eor3: tests/test_sve_eor3.c $(BUILDDIR)/libraidjit.a $(HDRS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_sve_eor3.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)

$(BUILDDIR)/test_aarch64_remat: tests/test_aarch64_remat.c $(BUILDDIR)/libraidjit.a $(HDRS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_aarch64_remat.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)

$(BUILDDIR)/test_riscv_remat: tests/test_riscv_remat.c $(BUILDDIR)/libraidjit.a $(HDRS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_riscv_remat.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)

$(BUILDDIR)/test_riscv_prefetch: tests/test_riscv_prefetch.c $(BUILDDIR)/libraidjit.a $(HDRS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_riscv_prefetch.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)
$(BUILDDIR)/test_riscv_zvbc: tests/test_riscv_zvbc.c $(BUILDDIR)/libraidjit.a $(HDRS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_riscv_zvbc.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)
$(BUILDDIR)/test_riscv_zvbb: tests/test_riscv_zvbb.c src/riscv/emit.c $(BUILDDIR)/libraidjit.a $(HDRS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_riscv_zvbb.c src/riscv/emit.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)
$(BUILDDIR)/test_rvv_f6: tests/test_rvv_f6.c $(BUILDDIR)/libraidjit.a $(HDRS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_rvv_f6.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)
$(BUILDDIR)/test_rvv_lmul: tests/test_rvv_lmul.c $(BUILDDIR)/libraidjit.a $(HDRS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_rvv_lmul.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)

$(BUILDDIR)/test_dispatch: tests/test_dispatch.c $(BUILDDIR)/libraidjit.a | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_dispatch.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)

$(BUILDDIR)/test_xorsched: tests/test_xorsched.c $(BUILDDIR)/libraidjit.a $(HDRS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_xorsched.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)

$(BUILDDIR)/test_xorbit: tests/test_xorbit.c $(BUILDDIR)/libraidjit.a $(HDRS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_xorbit.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)

$(BUILDDIR)/test_mshift: tests/test_mshift.c $(BUILDDIR)/libraidjit.a $(HDRS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_mshift.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)

$(BUILDDIR)/test_xorsched_e2e: tests/test_xorsched_e2e.c $(BUILDDIR)/libraidjit.a $(HDRS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_xorsched_e2e.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)

$(BUILDDIR)/test_xorcode: tests/test_xorcode.c $(BUILDDIR)/libraidjit.a $(HDRS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_xorcode.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)

$(BUILDDIR)/test_notail_xorsched: tests/test_notail_xorsched.c $(BUILDDIR)/libraidjit.a $(HDRS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_notail_xorsched.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)

$(BUILDDIR)/test_planner_fuzz: tests/test_planner_fuzz.c $(BUILDDIR)/libraidjit.a $(HDRS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_planner_fuzz.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)

$(BUILDDIR)/test_recover_closedform: tests/test_recover_closedform.c $(BUILDDIR)/libraidjit.a $(HDRS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_recover_closedform.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)

$(BUILDDIR)/test_pipe: tests/test_pipe.c $(BUILDDIR)/libraidjit.a $(HDRS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_pipe.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)

$(BUILDDIR)/test_crc: tests/test_crc.c $(BUILDDIR)/libraidjit.a $(HDRS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_crc.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)

$(BUILDDIR)/test_fused: tests/test_fused.c $(BUILDDIR)/libraidjit.a $(HDRS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_fused.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)

$(BUILDDIR)/test_striping: tests/test_striping.c $(BUILDDIR)/libraidjit.a $(HDRS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_striping.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)

$(BUILDDIR)/test_planid: tests/test_planid.c $(BUILDDIR)/libraidjit.a $(HDRS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_planid.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)

$(BUILDDIR)/test_batched: tests/test_batched.c $(BUILDDIR)/libraidjit.a $(HDRS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_batched.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)

$(BUILDDIR)/test_lrc: tests/test_lrc.c $(BUILDDIR)/libraidjit.a $(HDRS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_lrc.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)

$(BUILDDIR)/test_twin_audit: tests/test_twin_audit.c $(BUILDDIR)/libraidjit.a $(HDRS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -o $@ tests/test_twin_audit.c $(BUILDDIR)/libraidjit.a $(LDFLAGS)
