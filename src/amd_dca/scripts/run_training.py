from __future__ import annotations
import logging, datetime, json
from pathlib import Path

import numpy as np
from amd_dca.utils.helpers import find_repo_root, load_config, set_seed
from amd_dca.training import train as train_mod

REPO_ROOT = find_repo_root()
CONFIG_PATH = REPO_ROOT / "config.yaml"
DATA_DIR   = REPO_ROOT / "data"
LOG_DIR    = REPO_ROOT / "logs"
RESULTS_DIR = REPO_ROOT / "results"
SCRIPT_NAME = Path(__file__).stem

def setup_logging() -> None:
    LOG_DIR.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    logfile = LOG_DIR / f"{SCRIPT_NAME}_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [%(module)s.%(funcName)s] - %(message)s",
        handlers=[logging.FileHandler(logfile, mode="w"), logging.StreamHandler()]
    )
    logging.info("Log file: %s", logfile)

def main() -> None:
    setup_logging()
    logging.info("Starting training…")

    cfg = load_config(CONFIG_PATH)
    set_seed(cfg["random_seed"])

    npz = np.load(DATA_DIR / "processed" / "preprocessed_data.npz")
    X_train, Y_train = npz["X_train"], npz["Y_train"]
    X_val,   Y_val   = npz["X_val"],   npz["Y_val"]

    model_dir = RESULTS_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{cfg['model']['distribution']}_autoencoder_best.pt"

    model, hist = train_mod.train_model(
        X_train, Y_train, X_val, Y_val,
        model_config=cfg["model"],
        training_config=cfg["training"],
        save_path=model_path,
    )

    json.dump(hist, open(model_dir / f"{cfg['model']['distribution']}_training_history.json", "w"), indent=2)
    logging.info("Training finished.  Best model saved to %s", model_path)

if __name__ == "__main__":
    main()