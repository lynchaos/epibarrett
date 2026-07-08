.PHONY: install demo test lint clean real

install:      ## install package + dev extras (editable)
	pip install -e ".[dev,interpret]"

demo:         ## run the offline, simulated-data pipeline end-to-end
	python scripts/run_demo.py --outdir results --seed 7

test:         ## run the CI test suite
	pytest

lint:         ## static checks
	ruff check src tests scripts

real:         ## run on real GEO cohorts (needs `pip install .[real]` + network)
	python scripts/run_real.py --discovery GSE81334 --external GSE104707 --outdir results_real

clean:
	rm -rf results/*.json results/*.md results/figures/*.png .pytest_cache \
	       **/__pycache__ src/**/__pycache__
