# Makefile for AMD Denoising Pipeline
PYTHON := python -m amd_dca.cli

.PHONY: env \
        run_preprocessing \
        run_ml_baseline_1 \
        run_ml_baseline_2 \
        run_dl_autoencoder \
        run_dl_vae \
        infer_ae \
        infer_vae \
        run_dge \
        run_evaluation \
        lock

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

# NOTE: invoke the bare `infer_ae` and `infer_vae` commands, not “run_…”
infer_ae:
	$(PYTHON) infer_ae

infer_vae:
	$(PYTHON) infer_vae

# usage: make run_dge input=<raw|combat|knn|rf|nb_ae|nb_vae>
run_dge:
	$(PYTHON) run_dge --input $(input)

# usage: make run_evaluation input=<knn|rf|nb_ae|nb_vae|combat>
run_evaluation:
	$(PYTHON) run_evaluation --input $(input)

lock:
	conda list --explicit > env.lock.txt