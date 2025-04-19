#!/usr/bin/env python
# src/amd_dca/scripts/run_ml_baseline_2.py

from __future__ import annotations
import logging
import datetime
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from amd_dca.utils.helpers import find_repo_root, load_config, set_seed

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

def main(argv=None) -> None:
    # — Logging & config —
    repo   = find_repo_root()
    LOGDIR = repo / "logs"
    setup_logging(LOGDIR, "run_ml_baseline_2")

    cfg = load_config(repo / "config.yaml")
    set_seed(cfg["random_seed"])

    # — Load processed data —
    PROC = repo / "data" / "processed"
    data = np.load(PROC / "preprocessed_data.npz")
    X_train, Y_train = data["X_train"], data["Y_train"]
    X_test          = data["X_test"]

    # — Pull RF params from config —
    try:
        rf_cfg      = cfg["baseline_rf"]
        n_estimators = rf_cfg["n_estimators"]
        max_depth    = rf_cfg["max_depth"]
        n_jobs       = rf_cfg["n_jobs"]
    except KeyError as e:
        raise KeyError(f"Missing baseline_rf parameter in config.yaml: {e}")

    logging.info(
        "RandomForest denoising: n_estimators=%s, max_depth=%s, n_jobs=%s",
        n_estimators, max_depth, n_jobs
    )

    # — Fit a multi‑output RandomForestRegressor —
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=n_jobs,
        random_state=cfg["random_seed"],
        verbose=0
    )
    model.fit(X_train, Y_train)

    # — Predict and save —
    Y_pred = model.predict(X_test)
    out_f  = PROC / "rf_denoised_test.npy"
    np.save(out_f, Y_pred)
    logging.info("Saved RF‑denoised test counts to %s", out_f)

if __name__ == "__main__":
    main()