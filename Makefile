PY ?= python3
export PYTHONPATH := src
.PHONY: install test fundamentals data baseline clean
install:        ## pip install -e . with dev + torch extras
	$(PY) -m pip install -e ".[dev,torch]"
test:           ## unit tests (torch cross-check is skipped automatically if torch is unavailable)
	$(PY) -m pytest -q
fundamentals:   ## steps 3-6: all demos + ablations -> out/figures, out/fundamentals_results.json
	$(PY) -m medimg fundamentals all --out out
data:           ## synthetic data + dedup + leakage-safe split -> data/synth
	$(PY) -m medimg data --root data/synth --n 300
baseline: test fundamentals data   ## everything that runs without torch
	@echo "BASELINE OK"
clean:
	rm -rf out data runs build *.egg-info .pytest_cache
