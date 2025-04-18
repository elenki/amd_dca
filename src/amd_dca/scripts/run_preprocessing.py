from __future__ import annotations
import argparse, logging, datetime, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd

from amd_dca.utils.helpers import find_repo_root, load_config, set_seed
from amd_dca.data import preprocess

# --------------------------------------------------------------------- #
REPO_ROOT   = find_repo_root()
CONFIG_PATH = REPO_ROOT / "config.yaml"
RAW_DIR     = REPO_ROOT / "data" / "raw"
PROC_DIR    = REPO_ROOT / "data" / "processed"
LOG_DIR     = REPO_ROOT / "logs"
SCRIPT      = Path(__file__).stem
# --------------------------------------------------------------------- #

def setup_logging():
    LOG_DIR.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    logfile = LOG_DIR / f"{SCRIPT}_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [%(module)s] %(message)s",
        handlers=[logging.FileHandler(logfile), logging.StreamHandler()],
    )
    logging.info("Log file: %s", logfile)


def main(argv=None):
    parser = argparse.ArgumentParser()
    # no flags: always compute full set
    args = parser.parse_args(argv)

    setup_logging()
    cfg = load_config(CONFIG_PATH)
    set_seed(cfg["random_seed"] if "random_seed" in cfg else cfg.get("random_seed", 42))

    PROC_DIR.mkdir(parents=True, exist_ok=True)

    ds = next(iter(cfg["datasets"]))  # e.g. 'gse115828'
    paths = cfg["datasets"][ds]
    counts_f    = RAW_DIR / paths["counts_file"]
    meta_f      = RAW_DIR / paths["metadata_file"]
    map_f       = RAW_DIR / paths["mapping_file"]

    # 1) load + map to 'linking_id'
    counts_df, meta_df = preprocess.load_data(counts_f, meta_f, map_f)

    # 2) QC: sample & gene filters
    df = preprocess.filter_samples(counts_df.join(meta_df), cfg)
    gene_cols = [c for c in df.columns if c not in meta_df.columns]
    meta_cols = [c for c in df.columns if c in meta_df.columns]
    counts_qc = df[gene_cols]
    meta_qc   = df[meta_cols]
    counts_filt = preprocess.filter_genes(counts_qc, cfg)
    meta_filt   = meta_qc.loc[counts_filt.index]

    # 3) size factors & log-CPM & ComBat
    sf = preprocess.calculate_size_factors(counts_filt)
    np.save(PROC_DIR / "size_factors.npy", sf.values)
    logcpm = np.log2((counts_filt.div(sf, axis=0) * 1e6) + 1.0)
    np.save(PROC_DIR / "logcpm.npy", logcpm.values)
    # ComBat
    combat = preprocess.run_combat(counts_filt, meta_filt[ cfg["preprocessing"]["stratify_on"] ])
    np.save(PROC_DIR / "combat_counts.npy", combat.values)

    # 4) integer raw counts
    raw_int = counts_filt.round().astype(int)
    np.save(PROC_DIR / "raw_counts_int.npy", raw_int.values)

    # 5) splits
    train_ids, val_ids, test_ids = preprocess.split_data(raw_int.index, meta_filt, cfg)
    manifest = {"train": train_ids.tolist(),
                "validation": val_ids.tolist(),
                "test": test_ids.tolist()}
    (PROC_DIR / "split_manifest.json").write_text(json.dumps(manifest, indent=2))

    # 6) covariates
    covs, encoder, scaler = preprocess.prepare_covariates(meta_filt,
        cfg["preprocessing"]["covariates"], train_ids)
    pickle.dump(covs, open(PROC_DIR / "covariates_df.pkl", "wb"))
    if encoder: pickle.dump(encoder, open(PROC_DIR / "encoder.pkl", "wb"))
    if scaler: pickle.dump(scaler, open(PROC_DIR / "scaler.pkl", "wb"))

    # 7) assemble DL inputs (log1p + cov → X; raw_int → Y)
    def _assemble(ids):
        X = np.concatenate([np.log1p(counts_filt.loc[ids].values), covs.loc[ids].values], axis=1)
        Y = raw_int.loc[ids].values
        return X, Y
    Xtr, Ytr = _assemble(train_ids)
    Xva, Yva = _assemble(val_ids)
    Xte, Yte = _assemble(test_ids)
    np.savez_compressed(PROC_DIR / "preprocessed_data.npz",
                        X_train=Xtr, Y_train=Ytr,
                        X_val=Xva,   Y_val=Yva,
                        X_test=Xte,  Y_test=Yte)

    # 8) gene/sample lists
    pd.Series(counts_filt.columns).to_csv(PROC_DIR / "genes.txt",
        index=False, header=False)
    pd.Series(train_ids).to_csv(PROC_DIR / "train_ids.txt", index=False, header=False)
    pd.Series(val_ids).to_csv(PROC_DIR / "val_ids.txt", index=False, header=False)
    pd.Series(test_ids).to_csv(PROC_DIR / "test_ids.txt", index=False, header=False)

    logging.info("Preprocessing complete. Files in %s", PROC_DIR)

if __name__ == "__main__":
    main()