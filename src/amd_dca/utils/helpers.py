import yaml
import random
import numpy as np
import torch
import os
import logging

def load_config(config_path='config.yaml'):
    """Loads the YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def set_seed(seed_value):
    """Sets random seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    logging.info(f"Set random seed to {seed_value}")

# --- PROJECT‑ROOT DISCOVERY --------------------------------------------------
from pathlib import Path

def find_repo_root(marker: str = "config.yaml") -> Path:
    """
    Walks up the directory tree from this file until it finds `marker`
    (defaults to 'config.yaml').  Returns that directory path.

    Raises RuntimeError if the marker isn't found.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / marker).exists():
            return parent
    raise RuntimeError(f"Could not locate project root containing '{marker}'")