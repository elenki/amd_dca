import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle

# Add src directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.evaluation import evaluate
from src.utils import helpers, plotting
from src.model import autoencoder # Need model class definition to load state_dict
import logging

def main():
     # --- Setup ---
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    log_dir_abs = os.path.join(project_root, "logs")
    helpers.setup_logging(log_dir_abs, script_name) # Use setup_logging if defined in helpers

    logging.info("Starting evaluation script...")
    config = helpers.load_config(os.path.join(project_root, 'config.yaml'))
    helpers.set_seed(config['random_seed']) # For reproducible evaluation steps if any

    # --- Define Paths ---
    processed_dir = os.path.join(project_root, 'data', 'processed')
    results_dir = os.path.join(project_root, config['results_dir'])
    model_save_dir = os.path.join(results_dir, 'models')
    plot_save_dir = os.path.join(results_dir, 'plots')
    denoised_data_dir = os.path.join(results_dir, 'denoised_data')
    os.makedirs(plot_save_dir, exist_ok=True)
    os.makedirs(denoised_data_dir, exist_ok=True)

    # --- Load Test Data ---
    logging.info(f"Loading preprocessed test data from {processed_dir}")
    try:
        data = np.load(os.path.join(processed_dir, 'preprocessed_data.npz'))
        # IMPORTANT: Use X_test and Y_test for final evaluation
        X_test = data['X_test']
        Y_test_raw = data['Y_test'] # Raw counts for comparison

        # Load corresponding sample IDs and metadata
        test_ids = pd.read_csv(os.path.join(processed_dir, 'test_sample_ids.txt'), header=None)[0].tolist()
        # Need to load the full filtered metadata and select test samples
        # This assumes metadata_final_df was saved during preprocessing
        # metadata_final_path = os.path.join(processed_dir, 'metadata_final.csv')
        # metadata_final_df = pd.read_csv(metadata_final_path, index_col=0) # Assuming index was saved
        # test_metadata = metadata_final_df.loc[test_ids]
        # Placeholder if metadata saving wasn't implemented:
        test_metadata = pd.DataFrame(index=test_ids) # Minimal placeholder
        logging.info(f"Loaded test data: X_test shape {X_test.shape}, Y_test_raw shape {Y_test_raw.shape}")

        # Load gene list
        gene_list = pd.read_csv(os.path.join(processed_dir, 'final_gene_list.txt'), header=None)[0].tolist()

    except FileNotFoundError:
        logging.error(f"Processed data file not found in {processed_dir}. Run preprocessing script first.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading processed data: {e}")
        sys.exit(1)

    # --- Load Trained Model ---
    model_filename = f"{config['model']['distribution']}_autoencoder_best.pt"
    model_path = os.path.join(model_save_dir, model_filename)
    logging.info(f"Loading best trained model from {model_path}")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Re-initialize model structure
        input_dim = X_test.shape[1]
        output_dim = Y_test_raw.shape[1] # Number of genes
        model = autoencoder.CountAutoencoder(
            input_dim=input_dim,
            output_dim=output_dim,
            encoder_layer_dims=config['model']['encoder_layers'],
            bottleneck_dim=config['model']['bottleneck_size'],
            decoder_layer_dims=config['model']['decoder_layers'],
            distribution=config['model']['distribution'],
            activation_fn=nn.ReLU() if config['model']['activation'] == 'relu' else nn.SELU(),
            dropout_rate=0.0 # Typically no dropout during evaluation
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval() # Set to evaluation mode
        logging.info("Model loaded successfully.")
    except FileNotFoundError:
        logging.error(f"Model file not found: {model_path}. Run training script first.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        sys.exit(1)

    # --- Generate Denoised Data ---
    # Create DataLoader for test data (only need X)
    test_dataset = TensorDataset(torch.FloatTensor(X_test)) # No Y needed for prediction
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size']*2, shuffle=False) # Use larger batch size for inference

    denoised_test_data = evaluate.get_denoised_data(model, test_loader, device)
    # Convert to DataFrame
    denoised_counts_df = pd.DataFrame(denoised_test_data, index=test_ids, columns=gene_list)
    raw_counts_df = pd.DataFrame(Y_test_raw, index=test_ids, columns=gene_list)

    # Save denoised data
    denoised_save_path = os.path.join(denoised_data_dir, f"denoised_{config['model']['distribution']}_counts.tsv")
    denoised_counts_df.to_csv(denoised_save_path, sep='\t')
    logging.info(f"Saved denoised counts to {denoised_save_path}")

    # --- Perform Evaluations ---
    logging.info("Performing evaluations...")

    # Example: PCA Plot colored by MGS level
    # Ensure test_metadata has the necessary columns and correct index
    # You might need to reload and filter the full metadata based on test_ids
    if 'mgs_level' in test_metadata.columns:
         evaluate.run_pca_analysis(
             raw_data=Y_test_raw, # Raw counts (samples x genes)
             denoised_data=denoised_test_data, # Denoised means (samples x genes)
             metadata=test_metadata, # Metadata for test samples
             config=config, # Pass config for paths etc.
             title_prefix="TestSet"
             # combat_data=... # Add if you have ComBat results for test set
         )
    else:
         logging.warning("Cannot generate PCA colored by MGS level: column not found in test metadata.")

    # Add calls to other evaluation functions (UMAP, DGE comparison placeholder)
    # evaluate.run_umap_analysis(...)
    # evaluate.run_dge_comparison(...)

    logging.info(f"Evaluation complete. Plots saved to {plot_save_dir}")


if __name__ == "__main__":
    main()

