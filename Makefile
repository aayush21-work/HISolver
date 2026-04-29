# Makefile
# ========
# Builds the Horndeski background and perturbation solvers.
#
# Usage:
#   make              # build both
#   make background   # build background solver only
#   make perturb      # build perturbation solver only
#   make run          # build and run full pipeline
#   make clean        # remove binaries and output files

CXX      = g++
CXXFLAGS = -O3 -Wall -Wextra -march=native

# Headers (background.cpp and perturbations.cpp include these)
HEADERS  = model_generated.hpp model_potential.hpp

.PHONY: all background perturb run clean

all: background perturb

background: background.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o background background.cpp
	@echo "Built: background"

perturb: perturbation.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o perturbations perturbation.cpp
	@echo "Built: perturbations"

run: all
	@echo "━━━  Running background solver  ━━━"
	./background
	@echo ""
	@echo "━━━  Running perturbation solver  ━━━"
	./perturbations
	@echo ""
	@echo "━━━  Results  ━━━"
	@cat spectra_summary.dat

clean:
	rm -f background perturbations
	rm -f background.dat scalar_spectrum.dat tensor_spectrum.dat spectra_summary.dat
