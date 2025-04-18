#!/usr/bin/env python
from __future__ import annotations
import logging, datetime, json
from pathlib import Path

import numpy as np

from amd_dca.utils.helpers import find_repo_root, load_config, set_seed
from amd_dca.training.train_vae import train_vae

REPO       = find_repo_root()
CFG_PATH   = REPO / "config.yaml"
PROC_DIR   = REPO / "data" / "processed"
MODELS_DIR = REPO / "results" / "models"
LOG_DIR    = REPO / "logs"
SCRIPT     = Path(__file__).stem

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

def main():
    setup_logging()
    cfg = load_config(CFG_PATH)
    set_seed(cfg["random_seed"])

    # load processed arrays
    data = np.load(PROC_DIR / "preprocessed_data.npz")
    Xtr, Ytr = data["X_train"], data["Y_train"]
    Xva, Yva = data["X_val"],   data["Y_val"]

    vae_cfg = cfg["model_vae"]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MODELS_DIR / f"{vae_cfg['type']}_best.pt"

    model, hist = train_vae(
        Xtr, Ytr, Xva, Yva,
        model_cfg    = vae_cfg,
        training_cfg = cfg["training"],
        save_path    = str(save_path),
    )

    json.dump(hist, open(MODELS_DIR / f"{vae_cfg['type']}_history.json", "w"), indent=2)
    logging.info("VAE training complete. Model saved to %s", save_path)

if __name__=="__main__":
    main()