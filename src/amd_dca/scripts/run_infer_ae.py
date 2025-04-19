#!/usr/bin/env python
from __future__ import annotations
import logging, datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from amd_dca.utils.helpers import find_repo_root, load_config, set_seed
from amd_dca.model.autoencoder import CountAutoencoder

# ── Paths & Constants ───────────────────────────────────────────────────────
ROOT      = find_repo_root()
CFG_PATH  = ROOT / "config.yaml"
PROC_DIR  = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "results" / "models"
LOG_DIR   = ROOT / "logs"
SCRIPT    = Path(__file__).stem

def setup_logging():
    LOG_DIR.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    logfile = LOG_DIR / f"{SCRIPT}_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [run_infer_ae] %(message)s",
        handlers=[logging.FileHandler(logfile), logging.StreamHandler()],
    )
    logging.info("Log file: %s", logfile)

def main():
    setup_logging()
    cfg = load_config(CFG_PATH)
    set_seed(cfg["random_seed"])

    # 1) load processed test split
    data = np.load(PROC_DIR / "preprocessed_data.npz")
    X_test = data["X_test"]      # (n_test, n_features)
    Y_test = data["Y_test"]      # (n_test, n_genes) — for output_dim

    # 2) instantiate AE with positional args matching your signature
    ae_cfg = cfg["model_ae"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CountAutoencoder(
        X_test.shape[1],                     # input_dim
        ae_cfg["encoder_layers"],            # encoder_layers
        ae_cfg["bottleneck_size"],           # bottleneck_dim
        ae_cfg["decoder_layers"],            # decoder_layers
        Y_test.shape[1],                     # output_dim
        ae_cfg["distribution"],              # distribution
        ae_cfg["activation"],                # activation
        cfg["training"].get("dropout_rate", 0.0),  # dropout
    ).to(device)

    # 3) load weights
    ckpt = MODEL_DIR / f"{ae_cfg['type']}_best.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # 4) DataLoader for X_test
    ds = TensorDataset(torch.from_numpy(X_test).float())
    loader = DataLoader(ds,
                        batch_size=cfg["training"]["batch_size"],
                        shuffle=False)

    # 5) forward & collect μ
    all_mu = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            out = model(xb)   # (mu, theta) or (mu, theta, pi)
            all_mu.append(out[0].cpu().numpy())

    Y_hat = np.vstack(all_mu)  # (n_test, n_genes)

    # 6) save
    out_path = PROC_DIR / "nb_ae_denoised_test.npy"
    np.save(out_path, Y_hat)
    logging.info("Saved AE denoised test matrix → %s", out_path)

if __name__ == "__main__":
    main()