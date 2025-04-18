PYTHON := python -m amd_dca.cli

.PHONY: env run_preprocessing run_ml_baseline_1 run_ml_baseline_2 run_dl_autoencoder run_dl_vae run_dge run_evaluation lock

env:
	conda env create -f environment.yaml

run_preprocessing:
	$(PYTHON) run_preprocessing

run_ml_baseline_1:
	$(PYTHON) ml_baseline_1

run_ml_baseline_2:
	$(PYTHON) ml_baseline_2

run_dl_autoencoder:
	$(PYTHON) train_ae

run_dl_vae:
	$(PYTHON) train_vae

run_dge:
	$(PYTHON) run_dge --input $(input)

run_evaluation:
	$(PYTHON) run_evaluation

lock:
	conda list --explicit > env.lock.txt