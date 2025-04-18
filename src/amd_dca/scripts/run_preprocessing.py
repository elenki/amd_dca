from __future__ import annotations
import argparse
import logging
import datetime
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from amd_dca.utils.helpers import find_repo_root, load_config, set_seed
from amd_dca.data import preprocess
from amd_dca.r.combat import correct as combat_correct

# ────────────────────────────────────────────────────────────────────────────────
# Constants & Paths
# ────────────────────────────────────────────────────────────────────────────────
REPO_ROOT   = find_repo_root()
CONFIG_PATH = REPO_ROOT / "config.yaml"
RAW_DIR     = REPO_ROOT / "data" / "raw"
PROC_DIR    = REPO_ROOT / "data" / "processed"
LOG_DIR     = REPO_ROOT / "logs"
SCRIPT      = Path(__file__).stem

# ────────────────────────────────────────────────────────────────────────────────
def setup_logging() -> None:
    LOG_DIR.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    logfile = LOG_DIR / f"{SCRIPT}_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [%(module)s] %(message)s",
        handlers=[logging.FileHandler(logfile), logging.StreamHandler()],
    )
    logging.info("Log file: %s", logfile)

# ────────────────────────────────────────────────────────────────────────────────
def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run full preprocessing pipeline")
    args = parser.parse_args(argv)

    setup_logging()
    cfg = load_config(CONFIG_PATH)
    set_seed(cfg.get("random_seed", 42))

    # ensure output dir exists
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    # fetch file paths from config
    ds = next(iter(cfg["datasets"]))  # e.g. 'gse115828'
    paths = cfg["datasets"][ds]
    counts_path  = RAW_DIR / paths["counts_file"]
    meta_path    = RAW_DIR / paths["metadata_file"]
    mapping_path = RAW_DIR / paths["mapping_file"]

    # 1) LOAD raw counts, metadata, mapping sheet
    counts_df, meta_df, mapping_df = preprocess.load_data(
        counts_path, meta_path, mapping_path
    )

    # 2) MAP & COMBINE → one DataFrame: samples×genes + metadata columns
    combined = preprocess.map_and_combine(counts_df, meta_df, mapping_df)
    if combined is None:
        logging.error("Failed to map & combine. Aborting.")
        return

    # 3) SAMPLE QC (RIN filter)
    qc_samples = preprocess.filter_samples(combined, cfg)

    # split gene vs meta columns
    meta_cols  = [c for c in meta_df.columns]  # original metadata column names
    gene_cols  = [c for c in qc_samples.columns if c not in meta_cols]
    counts_qc  = qc_samples[gene_cols]
    meta_qc    = qc_samples[meta_cols]

    # 4) GENE QC (min counts & pct)
    counts_filt = preprocess.filter_genes(counts_qc, cfg)
    meta_filt   = meta_qc.loc[counts_filt.index]

    # 5) SIZE FACTORS & LOG‑CPM
    size_factors = preprocess.calculate_size_factors(counts_filt)
    np.save(PROC_DIR / "size_factors.npy", size_factors.values)
    logcpm = np.log2(counts_filt.div(size_factors, axis=0) * 1e6 + 1.0)
    np.save(PROC_DIR / "logcpm.npy", logcpm.values)

    # 6) COMBAT CORRECTION
    batch_col = cfg["preprocessing"]["stratify_on"]
    combat_counts = combat_correct(counts_filt, meta_filt[batch_col])
    np.save(PROC_DIR / "combat_counts.npy", combat_counts.values)

    # 7) RAW COUNTS as integers
    raw_int = counts_filt.round().astype(int)
    np.save(PROC_DIR / "raw_counts_int.npy", raw_int.values)

    # 8) TRAIN/VAL/TEST SPLIT
    train_ids, val_ids, test_ids = preprocess.split_data(
        raw_int.index, meta_filt, cfg
    )
    manifest = {"train": train_ids.tolist(),
                "validation": val_ids.tolist(),
                "test": test_ids.tolist()}
    (PROC_DIR / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )

    # 9) COVARIATE PREPARATION
    covs, encoder, scaler = preprocess.prepare_covariates(
        meta_filt, cfg["preprocessing"]["covariates"], train_ids
    )
    # pickle out the fitted objects
    covs.to_pickle(PROC_DIR / "covariates_df.pkl")
    if encoder:
        pickle.dump(encoder, open(PROC_DIR / "encoder.pkl", "wb"))
    if scaler:
        pickle.dump(scaler, open(PROC_DIR / "scaler.pkl", "wb"))

    # 10) ASSEMBLE X & Y for DL
    def _assemble(ids):
        X = np.concatenate([
            np.log1p(counts_filt.loc[ids].values),
            covs.loc[ids].values
        ], axis=1)
        Y = raw_int.loc[ids].values
        return X, Y

    X_train, Y_train = _assemble(train_ids)
    X_val,   Y_val   = _assemble(val_ids)
    X_test,  Y_test  = _assemble(test_ids)

    np.savez_compressed(
        PROC_DIR / "preprocessed_data.npz",
        X_train=X_train, Y_train=Y_train,
        X_val  =X_val,   Y_val=Y_val,
        X_test =X_test,  Y_test=Y_test
    )

    # 11) WRITE OUT GENE & SAMPLE ID LISTS
    pd.Series(counts_filt.columns).to_csv(
        PROC_DIR / "genes.txt", index=False, header=False
    )
    pd.Series(train_ids).to_csv(
        PROC_DIR / "train_ids.txt", index=False, header=False
    )
    pd.Series(val_ids).to_csv(
        PROC_DIR / "val_ids.txt", index=False, header=False
    )
    pd.Series(test_ids).to_csv(
        PROC_DIR / "test_ids.txt", index=False, header=False
    )

    logging.info("Preprocessing complete. Outputs written to %s", PROC_DIR)


if __name__ == "__main__":
    main()