#!/usr/bin/env python
# src/amd_dca/scripts/run_evaluation.py

from __future__ import annotations
import argparse, logging, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt

from amd_dca.utils.helpers import find_repo_root, load_config, set_seed
from amd_dca.model.autoencoder import CountAutoencoder

# ── Paths & Constants ────────────────────────────────────────────────────────
REPO        = find_repo_root()
CFG_PATH    = REPO / "config.yaml"
PROC_DIR    = REPO / "data" / "processed"
MODEL_DIR   = REPO / "results" / "models"
OUTDIR      = REPO / "results" / "evaluation"
LOG_DIR     = REPO / "logs"
SCRIPT      = Path(__file__).stem

def setup_logging():
    LOG_DIR.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    lf = LOG_DIR / f"{SCRIPT}_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [%(module)s] - %(message)s",
        handlers=[logging.FileHandler(lf, "w"), logging.StreamHandler()]
    )
    logging.info("Log file: %s", lf)

def load_matrix(method: str, Y_test: np.ndarray) -> np.ndarray:
    """
    Given the method, return the raw or denoised test‐matrix.
    """
    if method == "raw":
        return Y_test
    if method in ("knn", "rf", "nb_ae", "nb_vae", "combat"):
        path = PROC_DIR / (
            "combat_counts.npy" if method == "combat"
            else f"{method}_denoised_test.npy"
        )
        return np.load(path)
    raise ValueError(f"Unknown evaluation method: {method!r}")

def load_dl_model(cfg: dict, method: str, input_dim: int, output_dim: int, device: torch.device):
    """
    Instantiate and load AE or VAE model weights.
    """
    mcfg = cfg["model_ae"] if method == "nb_ae" else cfg["model_vae"]
    ModelClass = CountAutoencoder  # you could swap in a VAE class here
    model = ModelClass(
        input_dim=input_dim,
        encoder_layer_dims=mcfg["encoder_layers"],
        bottleneck_dim=mcfg.get("bottleneck_size", mcfg.get("latent_dim")),
        decoder_layer_dims=mcfg["decoder_layers"],
        output_dim=output_dim,
        distribution=mcfg["distribution"],
        activation_fn=torch.nn.ReLU() if mcfg["activation"]=="relu" else torch.nn.SELU(),
        dropout_rate=cfg["training"].get("dropout_rate",0.0),
    ).to(device)
    ckpt = MODEL_DIR / f"{mcfg['type']}_best.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model

def denoise_with_model(model: torch.nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    """
    Predict the NB mean parameter for each sample.
    """
    ds = torch.utils.data.DataLoader(
        torch.FloatTensor(X),
        batch_size=cfg["training"]["batch_size"],
        shuffle=False
    )
    all_mu = []
    with torch.no_grad():
        for xb in ds:
            xb = xb.to(device)
            out = model(xb)
            all_mu.append(out[0].cpu().numpy())
    return np.vstack(all_mu)

def make_scatter(emb: np.ndarray, labels: np.ndarray, title: str, save_png: Path, save_svg: Path):
    plt.figure(figsize=(6,5))
    plt.scatter(emb[:,0], emb[:,1], c=labels, cmap="Spectral", s=5)
    plt.title(title)
    plt.colorbar(label="mgs_level")
    plt.tight_layout()
    plt.savefig(save_png)
    plt.savefig(save_svg)
    plt.close()

def main(argv=None):
    setup_logging()
    p = argparse.ArgumentParser(prog="run_evaluation")
    p.add_argument(
        "--input",
        choices=["knn","rf","nb_ae","nb_vae","combat"],
        required=True,
        help="Which denoiser to compare vs raw"
    )
    args = p.parse_args(argv)

    cfg = load_config(CFG_PATH)
    set_seed(cfg["random_seed"])

    # load test split
    npz     = np.load(PROC_DIR / "preprocessed_data.npz")
    X_test  = npz["X_test"]
    Y_test  = npz["Y_test"]

    # sample IDs and metadata
    ids_t   = pd.read_csv(PROC_DIR / "test_ids.txt", header=None, dtype=str)[0].tolist()
    raw_meta = pd.read_csv(REPO / "data" / "raw" / cfg["datasets"]["gse115828"]["metadata_file"])
    raw_meta["linking_id"] = raw_meta["r_id"].astype(str).str.split("_").str[0]
    meta    = raw_meta.set_index("linking_id").loc[ids_t]
    labels  = meta[cfg["preprocessing"]["stratify_on"]].astype(int).values

    OUTDIR.mkdir(parents=True, exist_ok=True)

    # 1) get denoised Y_hat
    if args.input in ("nb_ae","nb_vae"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = load_dl_model(cfg, args.input, X_test.shape[1], Y_test.shape[1], device)
        Y_hat  = denoise_with_model(model, X_test, device)
    else:
        Y_hat = load_matrix(args.input, Y_test)

    # save out
    np.save(OUTDIR / f"{args.input}_denoised_test.npy", Y_hat)
    logging.info("Saved %s‐denoised test matrix", args.input)

    # 2) PCA & UMAP
    pca_n    = cfg["evaluation"]["pca_components"]
    umap_n   = cfg["evaluation"]["umap_components"]
    formats  = cfg["evaluation"]["plot_formats"]

    # raw vs method
    mats = {"raw": Y_test, args.input: Y_hat}

    for name, mat in mats.items():
        # PCA
        pca = PCA(n_components=pca_n).fit_transform(np.log1p(mat))
        for ext in formats:
            make_scatter(
                pca, labels,
                title=f"{name.upper()} PCA",
                save_png=OUTDIR / f"{name}_pca.{ext}",
                save_svg=OUTDIR / f"{name}_pca.{ext.replace('png','svg')}"
            )
        # UMAP
        umap_emb = umap.UMAP(n_components=umap_n, random_state=cfg["random_seed"])\
                        .fit_transform(mat)
        for ext in formats:
            make_scatter(
                umap_emb, labels,
                title=f"{name.upper()} UMAP",
                save_png=OUTDIR / f"{name}_umap.{ext}",
                save_svg=OUTDIR / f"{name}_umap.{ext.replace('png','svg')}"
            )

    logging.info("Evaluation complete. Plots in %s", OUTDIR)

if __name__ == "__main__":
    main()