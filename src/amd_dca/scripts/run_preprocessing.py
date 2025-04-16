from __future__ import annotations
# ---------------------------------------------------------------------------
#  RUN_PREPROCESSING  —  create train/val/test NPZ with integer targets
# ---------------------------------------------------------------------------
import logging, datetime, json, pickle
from pathlib import Path

import numpy as np
from amd_dca.utils.helpers import (
    find_repo_root,
    load_config,
    set_seed,
)
from amd_dca.data import preprocess

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
REPO_ROOT: Path = find_repo_root()          # repo top‑level (contains config.yaml)
CONFIG_PATH: Path = REPO_ROOT / "config.yaml"
DATA_DIR: Path = REPO_ROOT / "data"
LOG_DIR: Path = REPO_ROOT / "logs"
SCRIPT_NAME = Path(__file__).stem           # 'run_preprocessing'

# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
def main() -> None:
    setup_logging()
    logging.info("Starting preprocessing script…")

    cfg = load_config(CONFIG_PATH)
    set_seed(cfg["random_seed"])

    raw_dir       = DATA_DIR / "raw"
    processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    dataset = next(iter(cfg["datasets"].keys()))          # e.g. 'gse115828'
    counts_path = raw_dir / cfg["datasets"][dataset]["counts_file"]
    meta_path   = raw_dir / cfg["datasets"][dataset]["metadata_file"]

    # ------------------------------------------------------------------ #
    #  Load & ID‑map
    # ------------------------------------------------------------------ #
    raw_counts_df, metadata_df = preprocess.load_data(counts_path, meta_path)
    combined = preprocess.map_and_combine(raw_counts_df, metadata_df)
    if combined is None:
        logging.error("ID mapping failed, aborting.")
        return

    combined_qc = preprocess.filter_samples(combined, cfg)

    gene_prefix = cfg["preprocessing"]["gene_cols_prefix"]
    gene_cols = [c for c in combined_qc.columns if c.startswith(gene_prefix)]
    meta_cols = [c for c in cfg["preprocessing"]["metadata_cols"] if c in combined_qc.columns]

    counts_qc = combined_qc[gene_cols]
    metadata_qc = combined_qc[meta_cols]

    # ------------------------------------------------------------------ #
    #  Gene filter  ➜  counts_final_df   (still floats)
    # ------------------------------------------------------------------ #
    counts_final_df = preprocess.filter_genes(counts_qc, cfg)
    metadata_final_df = metadata_qc.loc[counts_final_df.index]

    # ------------------------------------------------------------------ #
    #  **Create integer copy for NB loss**
    # ------------------------------------------------------------------ #
    counts_final_int = counts_final_df.round().astype(int)

    # ------------------------------------------------------------------ #
    #  Train/Val/Test split
    # ------------------------------------------------------------------ #
    train_ids, val_ids, test_ids = preprocess.split_data(
        counts_final_df.index, metadata_final_df, cfg
    )
    split_manifest = {
        "train": train_ids.tolist(),
        "validation": val_ids.tolist(),
        "test": test_ids.tolist(),
    }
    (processed_dir / "train_val_test_split.json").write_text(
        json.dumps(split_manifest, indent=2)
    )
    logging.info("Saved train/val/test split manifest.")

    # ------------------------------------------------------------------ #
    #  Covariates
    # ------------------------------------------------------------------ #
    covariate_cols = cfg["preprocessing"].get("covariates", [])
    if covariate_cols:
        cov_all, enc, scl = preprocess.prepare_covariates(
            metadata_final_df, covariate_cols, train_ids
        )
        if enc:
            pickle.dump(enc, open(processed_dir / "fitted_encoder.pkl", "wb"))
        if scl:
            pickle.dump(scl, open(processed_dir / "fitted_scaler.pkl", "wb"))
    else:
        cov_all = metadata_final_df.loc[:, []]  # empty DF

    # ------------------------------------------------------------------ #
    #  Assemble model inputs  (X: floats)  and targets  (Y: integers)
    # ------------------------------------------------------------------ #
    y_train = counts_final_int.loc[train_ids].values
    y_val   = counts_final_int.loc[val_ids].values
    y_test  = counts_final_int.loc[test_ids].values

    x_train = np.concatenate([
        np.log1p(counts_final_df.loc[train_ids].values),
        cov_all.loc[train_ids].values], axis=1)
    x_val   = np.concatenate([
        np.log1p(counts_final_df.loc[val_ids].values),
        cov_all.loc[val_ids].values], axis=1)
    x_test  = np.concatenate([
        np.log1p(counts_final_df.loc[test_ids].values),
        cov_all.loc[test_ids].values], axis=1)

    np.savez_compressed(
        processed_dir / "preprocessed_data.npz",
        X_train=x_train, Y_train=y_train,
        X_val=x_val, Y_val=y_val,
        X_test=x_test, Y_test=y_test,
    )

    counts_final_df.columns.to_series().to_csv(
        processed_dir / "final_gene_list.txt", index=False, header=False
    )
    train_ids.to_series().to_csv(processed_dir / "train_sample_ids.txt", index=False, header=False)
    val_ids.to_series().to_csv(processed_dir / "val_sample_ids.txt",   index=False, header=False)
    test_ids.to_series().to_csv(processed_dir / "test_sample_ids.txt", index=False, header=False)

    logging.info("Preprocessing complete.  Outputs written to %s", processed_dir)

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()