from __future__ import annotations
import logging, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

from amd_dca.utils.helpers import find_repo_root, load_config, set_seed
from amd_dca.evaluation import evaluate
from amd_dca.model import autoencoder as ae

REPO_ROOT = find_repo_root()
CONFIG_PATH = REPO_ROOT / "config.yaml"
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
LOG_DIR = REPO_ROOT / "logs"
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
    logging.info("Starting evaluation…")

    cfg = load_config(CONFIG_PATH)
    set_seed(cfg["random_seed"])

    processed = np.load(DATA_DIR / "processed" / "preprocessed_data.npz")
    X_test, Y_test = processed["X_test"], processed["Y_test"]
    test_ids = pd.read_csv(DATA_DIR / "processed" / "test_sample_ids.txt", header=None)[0].tolist()
    gene_list = pd.read_csv(DATA_DIR / "processed" / "final_gene_list.txt", header=None)[0].tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = RESULTS_DIR / "models" / f"{cfg['model']['distribution']}_autoencoder_best.pt"

    model = ae.CountAutoencoder(
        input_dim=X_test.shape[1],
        output_dim=Y_test.shape[1],
        encoder_layer_dims=cfg["model"]["encoder_layers"],
        bottleneck_dim=cfg["model"]["bottleneck_size"],
        decoder_layer_dims=cfg["model"]["decoder_layers"],
        distribution=cfg["model"]["distribution"],
        dropout_rate=0.0,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    loader = DataLoader(TensorDataset(torch.FloatTensor(X_test)),
                        batch_size=cfg["training"]["batch_size"] * 2,
                        shuffle=False)
    denoised = evaluate.get_denoised_data(model, loader, device)

    denoised_df = pd.DataFrame(denoised, index=test_ids, columns=gene_list)
    den_dir = RESULTS_DIR / "denoised_data"
    den_dir.mkdir(parents=True, exist_ok=True)
    denoised_df.to_csv(den_dir / f"denoised_{cfg['model']['distribution']}_counts.tsv", sep="\t")

    logging.info("Saved denoised matrix; generating PCA plot…")
    evaluate.run_pca_analysis(
        raw_data=Y_test,
        denoised_data=denoised,
        metadata=pd.DataFrame(index=test_ids),  # replace when metadata held‑out
        config=cfg,
        title_prefix="TestSet"
    )

    logging.info("Evaluation complete.")

if __name__ == "__main__":
    main()