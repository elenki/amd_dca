#!/usr/bin/env python
"""
Run PCA+KNN baseline denoiser on log1p raw counts.
"""
import logging, datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor

from amd_dca.utils.helpers import find_repo_root, load_config, set_seed

# Paths
REPO = find_repo_root()
CFG  = REPO / "config.yaml"
PROC = REPO / "data/processed"
LOG   = REPO / "logs"

def setup_logging():
    LOG.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    f = LOG / f"run_ml_baseline_1_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(f), logging.StreamHandler()]
    )
    logging.info("Log file: %s", f)

def main():
    setup_logging()
    cfg = load_config(CFG)
    set_seed(cfg["random_seed"])

    # Load DL inputs
    data = np.load(PROC / "preprocessed_data.npz")
    X_train = data["X_train"]
    X_test  = data["X_test"]

    # Pull gene count dimension
    genes = pd.read_csv(PROC / "genes.txt", header=None)[0].tolist()
    n_genes = len(genes)

    # Log‐counts only (drop covariates)
    Xtr_log = X_train[:, :n_genes]
    Xte_log = X_test[:,  :n_genes]

    # Baseline config (default if missing)
    bk = cfg.get("baseline_knn", {})
    k = bk.get("n_neighbors", 5)
    pcs = bk.get("pca_components", 50)
    logging.info(f"PCA+KNN denoising: k={k}, pcs={pcs}")

    # PCA reduction
    pca = PCA(n_components=pcs, random_state=cfg["random_seed"])
    Ztr = pca.fit_transform(Xtr_log)
    Zte = pca.transform(Xte_log)

    # KNN regression in log‐space
    knn = KNeighborsRegressor(n_neighbors=k, weights="distance")
    knn.fit(Ztr, Xtr_log)
    denoised_log = knn.predict(Zte)

    # Back‐transform to counts
    denoised = np.expm1(denoised_log)
    denoised[denoised < 0] = 0
    denoised_int = np.rint(denoised).astype(int)

    # Save only test‐set denoised counts
    out = PROC / "knn_denoised_test.npy"
    np.save(out, denoised_int)
    logging.info("Saved KNN‐denoised test counts to %s", out)

if __name__ == "__main__":
    main()