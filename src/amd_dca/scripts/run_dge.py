from __future__ import annotations
import logging, datetime, json
from pathlib import Path
import numpy as np, pandas as pd

from amd_dca.utils.helpers import find_repo_root, load_config, set_seed
from amd_dca.r import dge as r_dge
from amd_dca.evaluation import metrics

# --------------------------------------------------------------------- #
ROOT   = find_repo_root()
DATA   = ROOT / "data/processed"
RAW    = ROOT / "data/raw"
OUTDIR = ROOT / "results/dge"
LOGDIR = ROOT / "logs"
SCRIPT = Path(__file__).stem
# --------------------------------------------------------------------- #
def setup_logging():
    LOGDIR.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    logf = LOGDIR / f"{SCRIPT}_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [%(module)s] - %(message)s",
        handlers=[logging.FileHandler(logf, "w"), logging.StreamHandler()],
    )
    logging.info("Log file: %s", logf)

# --------------------------------------------------------------------- #
def load_matrices() -> tuple[pd.DataFrame, pd.DataFrame | None, list[str]]:
    npz    = np.load(DATA / "preprocessed_data.npz")
    genes  = pd.read_csv(DATA / "final_gene_list.txt", header=None,
                         dtype=str)[0].tolist()
    ids_t  = pd.read_csv(DATA / "test_sample_ids.txt", header=None,
                         dtype=str)[0].tolist()

    raw_df = pd.DataFrame(npz["Y_test"], index=ids_t, columns=genes)

    combat = None
    cpath = DATA / "combat_counts.npy"
    if cpath.exists():
        split = json.loads((DATA / "train_val_test_split.json").read_text())
        all_ids = [str(x) for x in split["train"] + split["validation"] + split["test"]]
        combat_full = pd.DataFrame(np.load(cpath), index=all_ids, columns=genes)
        combat = combat_full.loc[ids_t]
    return raw_df, combat, ids_t

# --------------------------------------------------------------------- #
def load_meta(cfg, ids_t: list[str]) -> pd.DataFrame:
    meta = pd.read_csv(RAW / cfg["datasets"]["gse115828"]["metadata_file"])
    meta["linking_id"] = meta["r_id"].astype(str).str.split("_").str[0]
    meta = meta.set_index("linking_id").loc[ids_t, ["mgs_level"]]
    meta["mgs_level"] = meta["mgs_level"].astype(str)          # pass as strings
    return meta

# --------------------------------------------------------------------- #
def main(argv=None):
    setup_logging()
    cfg = load_config(ROOT / "config.yaml")
    set_seed(cfg["random_seed"])

    raw_df, combat_df, ids_t = load_matrices()
    meta = load_meta(cfg, ids_t)

    OUTDIR.mkdir(parents=True, exist_ok=True)

    # ---------------- DESeq2 on raw -----------------
    logging.info("Running DESeq2 on raw counts…")
    res_raw = r_dge.deseq2(raw_df, meta)
    res_raw.to_csv(OUTDIR / "raw_deseq2.tsv", sep="\t")
    metrics.volcano(res_raw, "Raw DESeq2", OUTDIR / "volcano_raw.png")

    # -------------- DESeq2 on ComBat ----------------
    if combat_df is not None:
        logging.info("Running DESeq2 on ComBat counts…")
        combat_int = combat_df.clip(lower=0).round().astype(int)  # safety
        res_combat = r_dge.deseq2(combat_int, meta)
        res_combat.to_csv(OUTDIR / "combat_deseq2.tsv", sep="\t")
        metrics.volcano(res_combat, "ComBat DESeq2", OUTDIR / "volcano_combat.png")
        metrics.venn_two(res_raw, res_combat, "Raw", "ComBat",
                         save_path=OUTDIR / "venn_raw_vs_combat.png")

    # ---------------- edgeR voom --------------------
    # logging.info("Running edgeR‑voom on raw counts…")
    # res_edger = r_dge.edger_voom(raw_df, meta)
    # res_edger.to_csv(OUTDIR / "raw_edger.tsv", sep="\t")

    logging.info("DGE finished. Outputs in %s", OUTDIR)

# --------------------------------------------------------------------- #
if __name__ == "__main__":
    main()