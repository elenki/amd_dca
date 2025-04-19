#!/usr/bin/env python
from __future__ import annotations
import logging, datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from amd_dca.utils.helpers import find_repo_root, load_config, set_seed
from amd_dca.model.vae import CountVAE

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
        format="%(asctime)s - %(levelname)s - [run_infer_vae] %(message)s",
        handlers=[logging.FileHandler(logfile), logging.StreamHandler()],
    )
    logging.info("Log file: %s", logfile)

def main():
    setup_logging()
    cfg = load_config(CFG_PATH)
    set_seed(cfg["random_seed"])

    # 1) load processed test split
    data = np.load(PROC_DIR / "preprocessed_data.npz")
    X_test = data["X_test"]
    Y_test = data["Y_test"]

    # 2) instantiate VAE with positional args matching your signature
    vae_cfg = cfg["model_vae"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CountVAE(
        X_test.shape[1],                # input_dim
        vae_cfg["encoder_layers"],      # encoder_layers
        vae_cfg["latent_dim"],          # latent_dim
        vae_cfg["decoder_layers"],      # decoder_layers
        Y_test.shape[1],                # output_dim
        vae_cfg["distribution"],        # distribution
        torch.nn.ReLU() if vae_cfg["activation"] == "relu"
                     else torch.nn.SELU(),  # activation_fn
        cfg["training"].get("dropout_rate", 0.0),  # dropout_rate
    ).to(device)

    # 3) load weights
    ckpt = MODEL_DIR / f"{vae_cfg['type']}_best.pt"
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
            out = model(xb)  
            # NB-VAE returns (mu, theta, mu_z, logvar_z)
            # ZINB-VAE returns (mu, theta, pi, mu_z, logvar_z)
            all_mu.append(out[0].cpu().numpy())

    Y_hat = np.vstack(all_mu)

    # 6) save
    out_path = PROC_DIR / "nb_vae_denoised_test.npy"
    np.save(out_path, Y_hat)
    logging.info("Saved VAE denoised test matrix → %s", out_path)

if __name__ == "__main__":
    main()