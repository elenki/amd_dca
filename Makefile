PYTHON := python -m amd_dca.cli  

.PHONY: env preprocess train evaluate lock

env:
	conda env create -f environment.yaml

preprocess:
	$(PYTHON) preprocess

preprocess-combat:
	$(PYTHON) preprocess --combat

train:
	$(PYTHON) train

dge:
	$(PYTHON) dge

evaluate:
	$(PYTHON) evaluate

lock:
	conda list --explicit > env.lock.txt