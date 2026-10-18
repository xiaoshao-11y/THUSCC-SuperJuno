# SuperJUNO optimization candidates — build & local verification.
#
# Flags are deliberately conservative: bit-exactness against numpy forbids
# -ffast-math, -Ofast, -funsafe-math-associations and per-host tuning like
# -march=native. C++17 per the team convention.
#
# -pthread is explicit in BOTH compile and link: the rng core spawns
# std::thread workers. A newer host glibc may link silently without it, but
# older cluster distros can fail or misbehave at load/run time — and the
# .so built here must NOT be assumed portable anyway: rebuild on the target
# node (see README "Deployment").

CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -fPIC -fno-fast-math -fno-unsafe-math-optimizations -pthread
LDFLAGS  ?= -shared -pthread

CPP_DIR := cpp
SLURM_DIR := slurm
SO      := $(CPP_DIR)/_sjrng_core.so
SO_FUSED := $(CPP_DIR)/_vmove1_fused_ref.so
UPSTREAM_SRC := upstream/src

.PHONY: all test rng-test fused-test install-test slurm-test clean install

all: $(SO) $(SO_FUSED)

$(SO_FUSED): $(CPP_DIR)/vmove1_fused_ref.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $<

$(SO): $(CPP_DIR)/sjrng_core.cpp $(CPP_DIR)/sjrng_ziggurat_tables.hpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $(CPP_DIR)/sjrng_core.cpp

# Install the opt-in pieces into the upstream tree (additive only; the
# default compute path stays upstream). Idempotent.
install: $(SO)
	install -m 0644 $(SO) $(UPSTREAM_SRC)/_sjrng_core.so
	install -m 0644 $(CPP_DIR)/sjrng.py $(UPSTREAM_SRC)/sjrng.py

# Full local verification of the C++ candidate against local numpy.
# LOCAL_PASS here is a unit-test result, not an official PE/tvE comparison.
test: rng-test fused-test install-test slurm-test

rng-test: $(SO)
	python3 $(CPP_DIR)/test_sjrng.py

# Formula check of the (unwired) fused vmove1 tail candidate: host mirror
# vs a numpy transcription of upstream move.py. Formula-level only.
fused-test: $(SO_FUSED)
	python3 $(CPP_DIR)/test_fused_formula.py

install-test: install
	python3 $(CPP_DIR)/test_install.py

# Local control-flow regression of slurm/submit.sh and run.slurm argument
# validation using explicit sbatch/squeue stubs. No cluster is contacted.
slurm-test:
	bash $(SLURM_DIR)/test_submit_controlflow.sh

clean:
	rm -f $(SO) $(SO_FUSED) $(UPSTREAM_SRC)/_sjrng_core.so
