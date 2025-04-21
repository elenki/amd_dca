#!/usr/bin/env python
# src/amd_dca/scripts/run_evaluation.py

from __future__ import annotations
import argparse
import logging
import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt

from amd_dca.utils.helpers import find_repo_root, load_config, set_seed
from amd_dca.model.autoencoder import CountAutoencoder
from amd_dca.model.vae import CountVAE

# ── Paths & Constants ────────────────────────────────────────────────────────
REPO      = find_repo_root()
CFG_PATH  = REPO / "config.yaml"
PROC_DIR  = REPO / "data" / "processed"
MODEL_DIR = REPO / "results" / "models"
OUTDIR    = REPO / "results" / "evaluation"
LOG_DIR   = REPO / "logs"
SCRIPT    = Path(__file__).stem


def setup_logging():
    LOG_DIR.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    lf = LOG_DIR / f"{SCRIPT}_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [%(module)s] %(message)s",
        handlers=[logging.FileHandler(lf, "w"), logging.StreamHandler()]
    )
    logging.info("Log file: %s", lf)


def load_matrix(method: str,
                Y_test: np.ndarray,
                ids_t: list[str],
                manifest: dict
               ) -> np.ndarray:
    """
    Return raw or denoised test‐matrix, always shaped (n_test × n_genes).
    """
    if method == "raw":
        return Y_test

    if method in ("knn", "rf", "nb_ae", "nb_vae"):
        return np.load(PROC_DIR / f"{method}_denoised_test.npy")

    if method == "combat":
        full = np.load(PROC_DIR / "combat_counts.npy")
        all_ids = manifest["train"] + manifest["validation"] + manifest["test"]
        df_full = pd.DataFrame(full, index=all_ids)
        return df_full.loc[ids_t].values

    raise ValueError(f"Unknown method: {method!r}")


def make_scatter(emb: np.ndarray,
                 labels: np.ndarray,
                 title: str,
                 save_png: Path,
                 save_svg: Path):
    """
    Discrete‐hue scatter so each mgs_level is a category.
    """
    df = pd.DataFrame({
        "Dim1": emb[:, 0],
        "Dim2": emb[:, 1],
        "mgs_level": labels.astype(str)
    })
    plt.figure(figsize=(6, 5))
    ax = sns.scatterplot(
        data=df,
        x="Dim1", y="Dim2",
        hue="mgs_level",
        palette="tab10",
        s=20,
        legend="full"
    )
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_png, dpi=150)
    plt.savefig(save_svg, dpi=150)
    plt.close()


def main(argv=None):
    setup_logging()

    p = argparse.ArgumentParser(prog="run_evaluation")
    p.add_argument(
        "--input",
        choices=["knn", "rf", "nb_ae", "nb_vae", "combat"],
        required=True,
        help="Which denoiser to compare vs raw"
    )
    args = p.parse_args(argv)

    cfg = load_config(CFG_PATH)
    set_seed(cfg["random_seed"])

    # ── load test split & metadata ───────────────────
    npz    = np.load(PROC_DIR / "preprocessed_data.npz")
    X_test = npz["X_test"]
    Y_test = npz["Y_test"]

    ids_t    = pd.read_csv(PROC_DIR / "test_ids.txt", header=None, dtype=str)[0].tolist()
    manifest = json.loads((PROC_DIR / "split_manifest.json").read_text())

    raw_meta = pd.read_csv(
        REPO / "data" / "raw" / cfg["datasets"]["gse115828"]["metadata_file"]
    )
    raw_meta["linking_id"] = raw_meta["r_id"].astype(str).str.split("_").str[0]
    meta = raw_meta.set_index("linking_id").loc[ids_t]
    labels = meta[cfg["preprocessing"]["stratify_on"]].astype(int).values

    OUTDIR.mkdir(parents=True, exist_ok=True)

    # ── 1) Generate or load denoised Y_hat ──────────
    if args.input in ("nb_ae", "nb_vae"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if args.input == "nb_ae":
            mcfg  = cfg["model_ae"]
            Model = CountAutoencoder
            model = Model(
                input_dim      = X_test.shape[1],
                encoder_layers = mcfg["encoder_layers"],
                bottleneck_dim = mcfg["bottleneck_size"],
                decoder_layers = mcfg["decoder_layers"],
                output_dim     = Y_test.shape[1],
                distribution   = mcfg["distribution"],
                activation     = mcfg["activation"],
                dropout        = cfg["training"].get("dropout_rate", 0.0),
            ).to(device)
            ckpt = MODEL_DIR / f"{mcfg['type']}_best.pt"

        else:  # nb_vae
            mcfg  = cfg["model_vae"]
            Model = CountVAE
            model = Model(
                input_dim      = X_test.shape[1],
                encoder_layers = mcfg["encoder_layers"],
                latent_dim     = mcfg["latent_dim"],
                decoder_layers = mcfg["decoder_layers"],
                output_dim     = Y_test.shape[1],
                distribution   = mcfg["distribution"],
                activation  = torch.nn.ReLU() if mcfg["activation"] == "relu"
                                 else torch.nn.SELU(),
                dropout   = cfg["training"].get("dropout_rate", 0.0),
            ).to(device)
            ckpt = MODEL_DIR / f"{mcfg['type']}_best.pt"

        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()

        loader = torch.utils.data.DataLoader(
            torch.FloatTensor(X_test),
            batch_size=cfg["training"]["batch_size"],
            shuffle=False
        )
        all_mu = []
        with torch.no_grad():
            for xb in loader:
                xb = xb.to(device)
                out = model(xb)
                mu  = out[0].cpu().numpy()
                all_mu.append(mu)
        Y_hat = np.vstack(all_mu)

    else:
        Y_hat = load_matrix(args.input, Y_test, ids_t, manifest)

    # save denoised matrix
    np.save(OUTDIR / f"{args.input}_denoised_test.npy", Y_hat)
    logging.info(f"Saved {args.input}‐denoised test matrix → {OUTDIR}")

    # ── 2) PCA & UMAP on raw vs. denoised ──────────
    pca_n   = cfg["evaluation"]["pca_components"]
    umap_n  = cfg["evaluation"]["umap_components"]
    formats = cfg["evaluation"]["plot_formats"]

    mats = {
        "raw":      Y_test,
        args.input: Y_hat
    }

    for name, mat in mats.items():
        # PCA on log1p
        pca_emb = PCA(n_components=pca_n).fit_transform(np.log1p(mat))
        for ext in formats:
            make_scatter(
                pca_emb, labels,
                title=f"{name.upper()} PCA",
                save_png=OUTDIR / f"{name}_pca.{ext}",
                save_svg=OUTDIR / f"{name}_pca.{ext.replace('png', 'svg')}"
            )

        # UMAP on counts
        umap_emb = umap.UMAP(
            n_components=umap_n,
            random_state=cfg["random_seed"]
        ).fit_transform(mat)
        for ext in formats:
            make_scatter(
                umap_emb, labels,
                title=f"{name.upper()} UMAP",
                save_png=OUTDIR / f"{name}_umap.{ext}",
                save_svg=OUTDIR / f"{name}_umap.{ext.replace('png', 'svg')}"
            )

    logging.info("Evaluation complete. Plots in %s", OUTDIR)


if __name__ == "__main__":
    main()