PYTHON := python -m amd_dca.cli  

.PHONY: env preprocess train evaluate lock

env:
	conda env create -f environment.yaml

preprocess:
	$(PYTHON) preprocess

train:
	$(PYTHON) train

evaluate:
	$(PYTHON) evaluate

lock:
	conda list --explicit > env.lock.txt