import sys
import os
import json
import numpy as np
import pandas as pd
import pickle

# Add src directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.training import train
from src.utils import helpers
import logging

def main():
    # --- Setup ---
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    log_dir_abs = os.path.join(project_root, "logs")
    helpers.setup_logging(log_dir_abs, script_name) # Use setup_logging if defined in helpers

    logging.info("Starting training script...")
    config = helpers.load_config(os.path.join(project_root, 'config.yaml'))
    helpers.set_seed(config['random_seed'])

    # --- Load Processed Data ---
    processed_dir = os.path.join(project_root, 'data', 'processed')
    logging.info(f"Loading preprocessed data from {processed_dir}")
    try:
        data = np.load(os.path.join(processed_dir, 'preprocessed_data.npz'))
        X_train = data['X_train']
        Y_train = data['Y_train']
        X_val = data['X_val']
        Y_val = data['Y_val']
        logging.info("Loaded train/validation data arrays.")
    except FileNotFoundError:
        logging.error(f"Processed data file not found in {processed_dir}. Run preprocessing script first.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading processed data: {e}")
        sys.exit(1)

    # --- Train Model ---
    model_save_dir = os.path.join(project_root, config['results_dir'], 'models')
    model_filename = f"{config['model']['distribution']}_autoencoder_best.pt" # Use .pt for PyTorch state_dict
    model_save_path = os.path.join(model_save_dir, model_filename)

    model, history = train.train_model(
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val,
        model_config=config['model'],
        training_config=config['training'],
        save_path=model_save_path
    )

    # --- Save Training History ---
    history_save_path = os.path.join(model_save_dir, f"{config['model']['distribution']}_training_history.json")
    try:
        with open(history_save_path, 'w') as f:
            json.dump(history, f)
        logging.info(f"Saved training history to {history_save_path}")
    except Exception as e:
        logging.error(f"Could not save training history: {e}")


    logging.info(f"Training complete. Best model saved to {model_save_path}")

if __name__ == "__main__":
    main()
