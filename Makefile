.PHONY: install test lint audit eda benchmark compare forecast uncertainty sensitivity layers results all

install:
	pip install -r requirements.txt

lint:
	ruff check src/ scripts/ tests/

test:
	pytest tests/ -v

audit:
	python scripts/audit.py

eda:
	python scripts/run_eda.py

benchmark:
	python scripts/run_benchmark.py

compare:
	python scripts/run_benchmark.py --compare

forecast:
	python scripts/run_final.py

uncertainty:
	python scripts/run_uncertainty.py

sensitivity:
	python scripts/run_sensitivity.py

layers:
	python scripts/build_layers.py

results:
	python scripts/build_results.py

# Pipeline completo. Delega ao run_all.py em vez de listar os alvos: assim a
# ordem das etapas é definida num lugar só, e quem está no Windows — onde não
# há `make` — roda exatamente a mesma sequência chamando o script direto.
all:
	python scripts/run_all.py
