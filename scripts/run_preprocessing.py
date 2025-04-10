import sys
import os
import pandas as pd
import numpy as np
import json
import pickle
import logging # Import logging
import datetime # Import datetime

# Add src directory to Python path to allow imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.data import preprocess
from src.utils import helpers

def setup_logging(log_dir: str, script_name: str):
    """Configures logging to file and console."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log_filename = os.path.join(log_dir, f"{script_name}_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(module)s.%(funcName)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, mode='w'), # Log to file (overwrite)
            logging.StreamHandler(sys.stdout) # Also log to console
        ]
    )
    logging.info(f"Logging initialized. Log file: {log_filename}")
    return log_filename

def main():
    # --- Configure Logging ---
    script_name = os.path.splitext(os.path.basename(__file__))[0] # Gets 'run_preprocessing'
    log_dir_relative = "logs" # Relative to project root
    log_dir_abs = os.path.join(project_root, log_dir_relative)
    setup_logging(log_dir_abs, script_name)

    logging.info("Starting preprocessing script...")

    # --- Load Config and Set Seed ---
    config = helpers.load_config(os.path.join(project_root,'config.yaml')) # Use absolute path
    helpers.set_seed(config['random_seed'])

    # Log config details
    logging.info("--- Configuration ---")
    logging.info(f"Config loaded: {config}")
    logging.info("---------------------")

    # --- Define Paths ---
    raw_dir = os.path.join(project_root, 'data', 'raw')
    processed_dir = os.path.join(project_root, 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    # --- Load Raw Data ---
    dataset_name = list(config['datasets'].keys())[0]
    counts_path = os.path.join(raw_dir, config['datasets'][dataset_name]['counts_file'])
    meta_path = os.path.join(raw_dir, config['datasets'][dataset_name]['metadata_file'])
    raw_counts_df, metadata_df = preprocess.load_data(counts_path, meta_path)

    # --- Map IDs and Combine ---
    combined_data = preprocess.map_and_combine(raw_counts_df, metadata_df)
    if combined_data is None:
        logging.error("Failed to map and combine data. Exiting.")
        sys.exit(1)

    # --- Filter Samples (e.g., by RIN) ---
    combined_data_qc = preprocess.filter_samples(combined_data, config)

    # --- Separate Genes and Metadata ---
    gene_cols = [col for col in combined_data_qc.columns if col.startswith(config['preprocessing']['gene_cols_prefix'])]
    metadata_cols = config['preprocessing']['metadata_cols']
    metadata_cols = [col for col in metadata_cols if col in combined_data_qc.columns] # Ensure they exist

    counts_qc_df = combined_data_qc[gene_cols]
    metadata_qc_df = combined_data_qc[metadata_cols]

    # --- Filter Genes ---
    counts_final_df = preprocess.filter_genes(counts_qc_df, config)
    metadata_final_df = metadata_qc_df.loc[counts_final_df.index] # Keep metadata for remaining samples

    # --- Split Data ---
    train_ids, val_ids, test_ids = preprocess.split_data(
        counts_final_df.index,
        metadata_final_df,
        config
    )
    split_info = {'train': train_ids.tolist(), 'validation': val_ids.tolist(), 'test': test_ids.tolist()}
    split_save_path = os.path.join(processed_dir, 'train_val_test_split.json')
    with open(split_save_path, 'w') as f:
        json.dump(split_info, f)
    logging.info(f"Saved train/val/test split sample IDs to {split_save_path}")

    # --- Prepare Covariates ---
    covariate_cols = config['preprocessing'].get('covariates', [])
    if covariate_cols:
        processed_covariates_all, encoder, scaler = preprocess.prepare_covariates(
            metadata_final_df,
            covariate_cols,
            train_indices=train_ids
        )
        # Save the fitted encoder and scaler
        if encoder:
            with open(os.path.join(processed_dir, 'fitted_encoder.pkl'), 'wb') as f:
                pickle.dump(encoder, f)
        if scaler:
             with open(os.path.join(processed_dir, 'fitted_scaler.pkl'), 'wb') as f:
                pickle.dump(scaler, f)
        logging.info("Prepared and saved covariate scaler/encoder.")
    else:
        logging.info("No covariates specified in config. Skipping covariate preparation.")
        processed_covariates_all = pd.DataFrame(index=metadata_final_df.index)

    # --- Prepare Final Model Inputs (X) and Targets (Y) ---

    # Target = Raw Counts (Y)
    Y_train = counts_final_df.loc[train_ids].values
    Y_val = counts_final_df.loc[val_ids].values
    Y_test = counts_final_df.loc[test_ids].values
    logging.info("Prepared raw count matrices (Y) for loss function.")

    # Input = Scaled Counts + Processed Covariates (X)
    counts_train_scaled = np.log1p(counts_final_df.loc[train_ids].values)
    counts_val_scaled = np.log1p(counts_final_df.loc[val_ids].values)
    counts_test_scaled = np.log1p(counts_final_df.loc[test_ids].values)
    logging.info("Applied log1p scaling to counts for network input.")

    covariates_train = processed_covariates_all.loc[train_ids].values
    covariates_val = processed_covariates_all.loc[val_ids].values
    covariates_test = processed_covariates_all.loc[test_ids].values

    X_train = np.concatenate([counts_train_scaled, covariates_train], axis=1)
    X_val = np.concatenate([counts_val_scaled, covariates_val], axis=1)
    X_test = np.concatenate([counts_test_scaled, covariates_test], axis=1)
    logging.info(f"Combined scaled counts and covariates for model input (X). Final X_train shape: {X_train.shape}")

    # --- Save Processed Data ---
    np.savez_compressed(os.path.join(processed_dir, 'preprocessed_data.npz'),
                        X_train=X_train, Y_train=Y_train,
                        X_val=X_val, Y_val=Y_val,
                        X_test=X_test, Y_test=Y_test)

    final_gene_list = counts_final_df.columns.tolist()
    pd.Series(final_gene_list).to_csv(os.path.join(processed_dir, 'final_gene_list.txt'), index=False, header=False)
    # Also save sample IDs corresponding to the rows in the saved numpy arrays
    pd.Series(train_ids).to_csv(os.path.join(processed_dir, 'train_sample_ids.txt'), index=False, header=False)
    pd.Series(val_ids).to_csv(os.path.join(processed_dir, 'val_sample_ids.txt'), index=False, header=False)
    pd.Series(test_ids).to_csv(os.path.join(processed_dir, 'test_sample_ids.txt'), index=False, header=False)


    logging.info(f"Processed data saved to {processed_dir}")
    logging.info("Preprocessing script finished.")

if __name__ == "__main__":
    main()
