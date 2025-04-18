#!/usr/bin/env python
from __future__ import annotations
import logging, datetime, json
from pathlib import Path
import numpy as np

from amd_dca.utils.helpers import find_repo_root, load_config, set_seed
from amd_dca.training.train import train_model

# paths
REPO       = find_repo_root()
CFG_PATH   = REPO / "config.yaml"
DATA_DIR   = REPO / "data" / "processed"
RESULT_DIR = REPO / "results" / "models"
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

    # load preprocessed
    npz = np.load(DATA_DIR / "preprocessed_data.npz")
    Xtr, Ytr = npz["X_train"], npz["Y_train"]
    Xva, Yva = npz["X_val"],   npz["Y_val"]

    # train
    ae_cfg = cfg["model_ae"]
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULT_DIR / f"{ae_cfg['type']}_best.pt"
    model, history = train_model(
        Xtr, Ytr, Xva, Yva,
        model_config   = ae_cfg,
        training_config   = cfg["training"],
        save_path   = str(out_path),
    )

    # save history
    with open(RESULT_DIR / f"{ae_cfg['type']}_history.json","w") as f:
        json.dump(history, f, indent=2)
    logging.info("AE training complete. Model + history saved.")