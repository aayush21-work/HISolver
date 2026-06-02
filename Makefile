CXX      = g++
CXXFLAGS = -O3 -Wall -Wextra -march=native

HEADERS  = model_generated.hpp model_potential.hpp


BG_SRC   := background.cpp
PT_SRC   := perturbation.cpp
RS_SRC   := rescale.cpp
BG_BIN   := background
PT_BIN   := perturbation
RS_BIN   := rescale

SPECTRA  := scalar_spectrum.dat tensor_spectrum.dat spectra_summary.dat
RESCALED := scalar_spectrum_rescaled.dat tensor_spectrum_rescaled.dat

.PHONY: all background perturb run model calibrate clean

all: $(BG_BIN) $(PT_BIN) $(RS_BIN)
 

$(BG_BIN): $(BG_SRC) $(HEADERS)
	$(CXX) $(CXXFLAGS) -ffast-math -o $@ $(BG_SRC)
	@echo "Built: $@"
 
$(PT_BIN): $(PT_SRC) $(HEADERS)
	$(CXX) $(CXXFLAGS) -ffast-math -o $@ $(PT_SRC)
	@echo "Built: $@"
 
$(RS_BIN): $(RS_SRC)
	$(CXX) $(CXXFLAGS) -o $@ $(RS_SRC)
	@echo "Built: $@"
 

$(HEADERS):
	@if [ ! -f "$@" ]; then \
	  echo "ERROR: $@ missing. Generate it first:"; \
	  echo "   - model_generated.hpp : run emit_model(...) in the notebook, or 'make model'"; \
	  echo "   - model_potential.hpp : define V_phi/dV_phi (edit the file)"; \
	  exit 1; \
	fi
 

	PYTHONPATH=. python -m horndeski.codegen.cpp_emitter
	@echo "Regenerated model_generated.hpp"
 

run: $(BG_BIN) $(PT_BIN)
	@echo "---  background  ---"
	./$(BG_BIN)
	@echo "---  perturbation  ---"
	./$(PT_BIN)
	@echo "---  summary  ---"
	@cat spectra_summary.dat
 

	./$(RS_BIN) scalar_spectrum.dat
	./$(RS_BIN) tensor_spectrum.dat
	@echo "Rescaled -> $(RESCALED)"


scale: $(RS_BIN)
	./$(RS_BIN) scalar_spectrum.dat
	./$(RS_BIN) tensor_spectrum.dat
	@echo "Rescaled -> $(RESCALED)"
 

pipeline: run scale
	@echo ""
	@echo "Pipeline complete. Plot-ready files:"
	@echo "   scalar_spectrum_rescaled.dat   (k [Mpc^-1]  P_s)"
	@echo "   tensor_spectrum_rescaled.dat   (k [Mpc^-1]  P_t)"
 

	
default:
	PYTHONPATH=. python -m horndeski.codegen.cpp_emitter
	@echo "Regenerated model_generated.hpp from cpp_emitter __main__"
 

calibrate:
	@cp -f model_generated.hpp model_generated.hpp.userbak 2>/dev/null || true
	PYTHONPATH=. python -c "from horndeski.symbols import phi_sym,X_sym,Mpl; from sympy import Function; from horndeski.codegen.cpp_emitter import emit_model; V=Function('V')(phi_sym); emit_model(X_sym-V,0*X_sym,Mpl**2/2,0*X_sym, model_name='SLOWROLL', V_expr=V, dV_expr=V.diff(phi_sym), output_path='model_generated.hpp', params={'Mpl':1.0})"
	$(CXX) $(CXXFLAGS) -o $(BG_BIN) $(BG_SRC) && ./$(BG_BIN) >/dev/null
	$(CXX) $(CXXFLAGS) -o $(PT_BIN) $(PT_SRC) && ./$(PT_BIN) | grep -E "A_s|n_s|^  r |r  "
	@awk 'END{print "last-row check: epsilon="$$6"  Qs="$$9"  (should match)"}' background.dat
	@echo "Validation: r above should equal 16*epsilon at the pivot."
	@mv -f model_generated.hpp.userbak model_generated.hpp 2>/dev/null \
	  && echo "Restored your model_generated.hpp" \
	  || echo "(no user header to restore)"


check:
	@echo "Dry-run of 'make pipeline' (must contain NO cpp_emitter call):"
	@$(MAKE) -n pipeline | grep -q cpp_emitter \
	  && echo "  WARNING: pipeline would regenerate the header!" \
	  || echo "  OK: pipeline uses the existing header, no regeneration."
 

clean:
	rm -f $(BG_BIN) $(PT_BIN) $(RS_BIN)
 
distclean: clean
	rm -f $(SPECTRA) $(RESCALED) *.dat model_generated.hpp.bak