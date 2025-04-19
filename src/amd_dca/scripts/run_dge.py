# src/amd_dca/scripts/run_dge.py

from __future__ import annotations
import argparse
import logging
import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

from amd_dca.utils.helpers import find_repo_root, load_config, set_seed
from amd_dca.r.dge import deseq2
from amd_dca.evaluation.metrics import volcano, venn_two

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT   = find_repo_root()
DATA   = ROOT / "data" / "processed"
RAW    = ROOT / "data" / "raw"
OUTDIR = ROOT / "results" / "dge"
LOGDIR = ROOT / "logs"
SCRIPT = Path(__file__).stem

# ── Logging ────────────────────────────────────────────────────────────────
def setup_logging() -> None:
    LOGDIR.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    logfile = LOGDIR / f"{SCRIPT}_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [%(module)s] - %(message)s",
        handlers=[logging.FileHandler(logfile, "w"), logging.StreamHandler()]
    )
    logging.info("Log file: %s", logfile)

# ── Helpers ────────────────────────────────────────────────────────────────
def _load_identifiers() -> tuple[list[str], list[str]]:
    genes = pd.read_csv(DATA / "genes.txt", header=None, dtype=str)[0].tolist()
    ids_t = pd.read_csv(DATA / "test_ids.txt", header=None, dtype=str)[0].tolist()
    return genes, ids_t

def load_counts(input_type: str,
                genes:      list[str],
                ids_t:      list[str],
                manifest:   dict
               ) -> pd.DataFrame:
    """
    Load test‐set counts for DGE:
      • raw  ← Y_test from preprocessed_data.npz
      • knn  ← knn_denoised_test.npy
      • rf   ← rf_denoised_test.npy
      • combat ← full combat_counts.npy subset to test IDs
    Returns an integer DataFrame shape (n_test × n_genes).
    """
    if input_type == "raw":
        arr = np.load(DATA / "preprocessed_data.npz")["Y_test"]
        return pd.DataFrame(arr, index=ids_t, columns=genes).astype(int)

    if input_type in ("knn", "rf"):
        arr = np.load(DATA / f"{input_type}_denoised_test.npy")
        return pd.DataFrame(arr, index=ids_t, columns=genes).clip(0).round().astype(int)

    if input_type == "combat":
        full = np.load(DATA / "combat_counts.npy")
        all_ids = manifest["train"] + manifest["validation"] + manifest["test"]
        df_full = pd.DataFrame(full, index=all_ids, columns=genes)
        return df_full.loc[ids_t].clip(0).round().astype(int)

    raise ValueError(f"Unsupported DGE input: {input_type!r}")

def load_metadata(cfg: dict, ids_t: list[str]) -> pd.DataFrame:
    """
    Load and filter metadata so it has exactly the contrast column
    (e.g. 'mgs_level') for the test samples, as a categorical.
    """
    meta = pd.read_csv(RAW / cfg["datasets"]["gse115828"]["metadata_file"])
    meta["linking_id"] = meta["r_id"].astype(str).str.split("_").str[0]
    key = cfg["dge"]["contrast_variable"]
    df = meta.set_index("linking_id").loc[ids_t, [key]]
    df[key] = df[key].astype(str).astype("category")
    return df

# ── Main ───────────────────────────────────────────────────────────────────
def main(argv=None):
    setup_logging()

    p = argparse.ArgumentParser(prog="run_dge")
    p.add_argument(
        "--input", choices=["raw", "combat", "knn", "rf"],
        required=True, help="Which test‐set to run DESeq2 on"
    )
    args = p.parse_args(argv)

    cfg = load_config(ROOT / "config.yaml")
    set_seed(cfg.get("random_seed", 42))

    genes, ids_t = _load_identifiers()
    manifest    = json.loads((DATA / "split_manifest.json").read_text())

    counts_df = load_counts(args.input, genes, ids_t, manifest)
    meta_df   = load_metadata(cfg, ids_t)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    dge_cfg = cfg["dge"]
    contrast = (
        dge_cfg["contrast_variable"],
        str(dge_cfg["contrast_level_1"]),
        str(dge_cfg["contrast_level_2"]),
    )
    alpha = float(dge_cfg.get("padj_threshold", 0.05))

    # ---- DESeq2 ----
    logging.info(f"Running DESeq2 on `{args.input}`…")
    res = deseq2(counts_df, meta_df,
                 group=contrast[0],
                 contrast=contrast)
    out_tsv = OUTDIR / f"{args.input}_deseq2.tsv"
    res.to_csv(out_tsv, sep="\t")
    logging.info("Wrote DESeq2 results → %s", out_tsv)

    # ---- Volcano ----
    volcano(res,
            title=f"{args.input.capitalize()} DESeq2",
            save_path=str(OUTDIR / f"volcano_{args.input}.png"))

    # ---- Venn: raw vs. combat (only when input=combat) ----
    if args.input == "combat":
        logging.info("Building Venn overlap (raw vs. combat)…")
        raw_res = deseq2(
            load_counts("raw", genes, ids_t, manifest),
            meta_df,
            group=contrast[0],
            contrast=contrast
        )
        venn_two(raw_res, res,
                 label_a="raw", label_b="combat",
                 alpha=alpha,
                 save_path=str(OUTDIR / "venn_raw_vs_combat.png"))

    logging.info("run_dge complete for `%s`. Results in %s", args.input, OUTDIR)

if __name__ == "__main__":
    main()