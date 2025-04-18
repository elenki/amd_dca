#!/usr/bin/env python
from __future__ import annotations
import logging, datetime, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt

from amd_dca.utils.helpers import find_repo_root, load_config, set_seed
from amd_dca.model.autoencoder import CountAutoencoder

# — paths & constants —
REPO        = find_repo_root()
CFG_PATH    = REPO / "config.yaml"
PROC_DIR    = REPO / "data" / "processed"
RESULTS_DIR = REPO / "results" / "evaluation"
MODEL_DIR   = REPO / "results" / "models"
LOG_DIR     = REPO / "logs"
SCRIPT      = Path(__file__).stem

def setup_logging():
    LOG_DIR.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    lf = LOG_DIR / f"{SCRIPT}_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [%(module)s] %(message)s",
        handlers=[logging.FileHandler(lf), logging.StreamHandler()],
    )
    logging.info("Log file: %s", lf)

def make_scatter(emb, labels, title, out_png, out_svg):
    plt.figure()
    scatter = plt.scatter(emb[:,0], emb[:,1], c=labels, cmap="Spectral", s=5)
    plt.title(title)
    plt.colorbar(scatter, label="mgs_level")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.savefig(out_svg)
    plt.close()

def main():
    setup_logging()
    cfg = load_config(CFG_PATH)
    set_seed(cfg["random_seed"])

    # load test split
    npz = np.load(PROC_DIR / "preprocessed_data.npz")
    X_test = npz["X_test"]       # log1p(counts) + covariates
    Y_test = npz["Y_test"]       # integer raw counts
    # load sample IDs & metadata for coloring
    test_ids = pd.read_csv(PROC_DIR / "test_ids.txt", header=None)[0].astype(str)
    # Re-load metadata so we can grab mgs_level
    raw_meta = pd.read_csv(REPO / "data" / "raw" / cfg["datasets"][list(cfg["datasets"])[0]]["metadata_file"])
    meta = raw_meta.set_index("r_id")  # or however you index
    # extract mgs_level per test sample.  Here we assume r_id == linking_id
    mgs = meta.loc[test_ids, cfg["preprocessing"]["stratify_on"]].astype(int).values

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # — 1) Load AE model & denoise —
    ae_cfg = cfg["model_ae"]
    model_path = MODEL_DIR / f"{ae_cfg['type']}_best.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CountAutoencoder(
        input_dim       = X_test.shape[1],
        encoder_layers  = ae_cfg["encoder_layers"],
        bottleneck_dim  = ae_cfg["bottleneck_size"],
        decoder_layers  = ae_cfg["decoder_layers"],
        output_dim      = Y_test.shape[1],
        distribution    = ae_cfg["distribution"],
        activation      = ae_cfg["activation"],
        dropout         = cfg["training"].get("dropout_rate",0.0),
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        X_t = torch.FloatTensor(X_test).to(device)
        outs = model(X_t)
        mu_hat = outs[0].cpu().numpy()  # NB → (batch, genes)

    # save denoised matrix
    np.save(RESULTS_DIR / "ae_denoised_test.npy", mu_hat)
    logging.info("Saved AE-denoised test matrix")

    # — 2) PCA & UMAP on raw vs denoised —
    pca_n = cfg["evaluation"]["pca_components"]
    umap_n = cfg["evaluation"]["umap_components"]
    formats = cfg["evaluation"]["plot_formats"]
    comps = {"raw": Y_test, "ae": mu_hat}

    for name, mat in comps.items():
        # PCA
        pca = PCA(n_components=pca_n)
        pc = pca.fit_transform(mat)
        for fmt in formats:
            plt.figure()
            plt.scatter(pc[:,0], pc[:,1], c=mgs, cmap="Spectral", s=5)
            plt.title(f"{name.upper()} PCA ({name})")
            plt.tight_layout()
            out = RESULTS_DIR / f"{name}_pca.{fmt}"
            plt.savefig(out)
            plt.close()
        # UMAP
        um = umap.UMAP(n_components=umap_n, random_state=cfg["random_seed"]).fit_transform(mat)
        for fmt in formats:
            plt.figure()
            plt.scatter(um[:,0], um[:,1], c=mgs, cmap="Spectral", s=5)
            plt.title(f"{name.upper()} UMAP ({name})")
            plt.tight_layout()
            out = RESULTS_DIR / f"{name}_umap.{fmt}"
            plt.savefig(out)
            plt.close()

    logging.info("Evaluation plots saved in %s", RESULTS_DIR)

if __name__ == "__main__":
    main()