.PHONY: install pipeline test quality clean serve-mlflow tensorboard

install:
	python -m pip install --upgrade pip
	python -m pip install -e ".[dev]"

pipeline:
	dvc repro

test:
	pytest --cov=ecg_denoising --cov-report=term-missing

quality:
	ruff check src tests
	ruff format --check src tests
	mypy src/ecg_denoising

serve-mlflow:
	mlflow ui --backend-store-uri sqlite:///mlflow.db

tensorboard:
	tensorboard --logdir logs/tensorboard

clean:
	python -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache','.ruff_cache','.mypy_cache','htmlcov']]"
