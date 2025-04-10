import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, UMAP # Import UMAP if installed
import logging
import os
from typing import Dict, Any, Optional
# Assume plotting functions are in src.utils.plotting
from ..utils import plotting

logger = logging.getLogger(__name__)

def get_denoised_data(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device) -> np.ndarray:
    """
    Generates denoised data (predicted means) using the trained model.

    Args:
        model: The trained autoencoder model.
        dataloader: DataLoader for the input data (X_scaled).
        device: CPU or CUDA device.

    Returns:
        np.ndarray: Denoised data matrix (samples x genes).
    """
    model.eval() # Set model to evaluation mode
    all_means = []
    logger.info("Generating denoised data (predicting means)...")
    with torch.no_grad():
        for x_batch, _ in dataloader: # Only need X for prediction
            x_batch = x_batch.to(device)
            outputs = model(x_batch)
            # Mean is the first output for both NB and ZINB
            mean_batch = outputs[0]
            all_means.append(mean_batch.cpu().numpy())

    denoised_means = np.concatenate(all_means, axis=0)
    logger.info(f"Generated denoised data with shape: {denoised_means.shape}")
    return denoised_means

def run_pca_analysis(raw_data: np.ndarray,
                     denoised_data: np.ndarray,
                     metadata: pd.DataFrame,
                     config: Dict[str, Any],
                     title_prefix: str = "",
                     combat_data: Optional[np.ndarray] = None):
    """Runs PCA and generates comparison plots."""
    logger.info(f"Running PCA analysis for {title_prefix}...")
    plot_dir = os.path.join(config['results_dir'].format(project_root=config.get('project_root','.')), config['paths']['plot_save_dir']) # Construct path

    # Ensure dataframes have matching indices if using metadata for coloring
    metadata_indexed = metadata.set_index(pd.Index(range(metadata.shape[0]))) # Use simple range index if needed

    # --- PCA on Raw Data ---
    pca_raw = PCA(n_components=2)
    # Use log1p for visualization stability
    raw_log1p = np.log1p(raw_data)
    pca_raw_result = pca_raw.fit_transform(raw_log1p)
    pca_raw_df = pd.DataFrame(pca_raw_result, columns=['PC1', 'PC2'], index=metadata_indexed.index)
    pca_raw_df = pca_raw_df.join(metadata_indexed)

    # --- PCA on Denoised Data ---
    pca_denoised = PCA(n_components=2)
    # Denoised data (means) might already be somewhat scaled, log1p might still help visualization
    denoised_log1p = np.log1p(denoised_data)
    pca_denoised_result = pca_denoised.fit_transform(denoised_log1p)
    pca_denoised_df = pd.DataFrame(pca_denoised_result, columns=['PC1', 'PC2'], index=metadata_indexed.index)
    pca_denoised_df = pca_denoised_df.join(metadata_indexed)

    # --- PCA on ComBat Data (Optional) ---
    if combat_data is not None:
        pca_combat = PCA(n_components=2)
        combat_log1p = np.log1p(combat_data)
        pca_combat_result = pca_combat.fit_transform(combat_log1p)
        pca_combat_df = pd.DataFrame(pca_combat_result, columns=['PC1', 'PC2'], index=metadata_indexed.index)
        pca_combat_df = pca_combat_df.join(metadata_indexed)
        num_plots = 3
        figsize=(24, 7)
    else:
        num_plots = 2
        figsize=(16, 7)

    # --- Plotting ---
    fig, axes = plt.subplots(1, num_plots, figsize=figsize, sharey=True, sharex=True)
    color_by = config['preprocessing'].get('stratify_on', 'mgs_level') # Use stratification col or mgs_level
    if color_by not in pca_raw_df.columns:
        logger.warning(f"Coloring column '{color_by}' not found in metadata. Skipping coloring.")
        color_by = None

    # Plot Raw
    ax_raw = axes[0]
    sns.scatterplot(data=pca_raw_df, x='PC1', y='PC2', hue=color_by, ax=ax_raw, alpha=0.7, s=50, legend='full' if num_plots==2 else False)
    ax_raw.set_title(f"{title_prefix} Raw Data PCA")
    ax_raw.set_xlabel(f"PC1 ({pca_raw.explained_variance_ratio_[0]*100:.1f}%)")
    ax_raw.set_ylabel(f"PC2 ({pca_raw.explained_variance_ratio_[1]*100:.1f}%)")

    # Plot Denoised
    ax_denoised = axes[1]
    sns.scatterplot(data=pca_denoised_df, x='PC1', y='PC2', hue=color_by, ax=ax_denoised, alpha=0.7, s=50, legend=False)
    ax_denoised.set_title(f"{title_prefix} Denoised Data PCA")
    ax_denoised.set_xlabel(f"PC1 ({pca_denoised.explained_variance_ratio_[0]*100:.1f}%)")
    ax_denoised.set_ylabel("") # Remove redundant label

    # Plot ComBat (TBD, need to check if combat_data is provided)
    if combat_data is not None:
        ax_combat = axes[2]
        sns.scatterplot(data=pca_combat_df, x='PC1', y='PC2', hue=color_by, ax=ax_combat, alpha=0.7, s=50, legend='full')
        ax_combat.set_title(f"{title_prefix} ComBat Data PCA")
        ax_combat.set_xlabel(f"PC1 ({pca_combat.explained_variance_ratio_[0]*100:.1f}%)")
        ax_combat.set_ylabel("") # Remove redundant label
        ax_combat.legend(title=color_by, bbox_to_anchor=(1.05, 1), loc='upper left')
        # Move raw legend if needed
        ax_raw.get_legend().remove()


    fig.suptitle(f"PCA Comparison ({title_prefix})", fontsize=16, y=1.02)
    plt.tight_layout()
    plotting.save_plot(fig, os.path.join(plot_dir, f"{title_prefix}_PCA_comparison_{color_by}.png"))
    plt.show() # Show in notebook if run interactively

    # Add similar function for UMAP: run_umap_analysis(...)


def run_dge_comparison(raw_counts_df: pd.DataFrame,
                       denoised_counts_df: pd.DataFrame,
                       metadata_df: pd.DataFrame,
                       config: Dict[str, Any],
                       combat_counts_df: Optional[pd.DataFrame] = None):
    """
    Placeholder function to orchestrate DGE analysis on different count matrices
    and compare results.
    """
    logger.warning("DGE comparison logic not implemented.")
    # 1. Define contrasts (e.g., MGS4 vs MGS1) based on metadata_df
    # 2. Run DESeq2/edgeR on raw_counts_df (using size factors from raw)
    # 3. Run DESeq2/edgeR on denoised_counts_df (recalculate size factors or use model info?)
    # 4. Run DESeq2/edgeR on combat_counts_df (if available)
    # 5. Compare results: number of DEGs, overlap (Venn diagrams), significance levels, pathway enrichment.
    pass

