from __future__ import annotations
import logging, datetime
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from amd_dca.utils.helpers import find_repo_root, load_config, set_seed

# --------------------------------------------------------------------- #
def setup_logging(logdir: Path, name: str) -> None:
    logdir.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    logfile = logdir / f"{name}_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(logfile), logging.StreamHandler()],
    )
    logging.info("Log file: %s", logfile)

# --------------------------------------------------------------------- #
def main(argv=None) -> None:
    # --- logging & config ---
    repo = find_repo_root()
    LOGDIR = repo / "logs"
    setup_logging(LOGDIR, "run_ml_baseline_2")
    cfg = load_config(repo / "config.yaml")
    set_seed(cfg["random_seed"])

    # --- paths & data load ---
    PROC = repo / "data" / "processed"
    npz = np.load(PROC / "preprocessed_data.npz")
    X_train, Y_train = npz["X_train"], npz["Y_train"]
    X_test  = npz["X_test"]

    # --- hyperparameters (fall back to defaults if not in config) ---
    rf_cfg = cfg.get("baseline_rf", {})
    n_estimators = rf_cfg.get("n_estimators", 10) # number of trees
    max_depth    = rf_cfg.get("max_depth", None)
    logging.info(f"RandomForest denoising: n_estimators={n_estimators}, max_depth={max_depth}")

    # --- fit multi-output RF ---
    base_rf = RandomForestRegressor(
        n_estimators = n_estimators,
        max_depth    = max_depth,
        random_state = cfg["random_seed"],
        n_jobs       = -1,
    )
    model = MultiOutputRegressor(base_rf)
    model.fit(X_train, Y_train)

    # --- predict & save ---
    Y_pred = model.predict(X_test)
    out_f  = PROC / "rf_denoised_test.npy"
    np.save(out_f, Y_pred)
    logging.info("Saved RF‐denoised test counts to %s", out_f)

if __name__ == "__main__":
    main()