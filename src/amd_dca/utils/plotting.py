import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from typing import Optional, List, Union

logger = logging.getLogger(__name__)

def save_plot(fig: plt.Figure, save_path: Optional[str] = None, default_filename: str = "plot.png"):
    """Helper function to save plots."""
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save plot to {save_path}: {e}")
    # plt.show() # Decide if you want plots to display interactively
    plt.close(fig) # Close figure to free memory


def plot_library_size(counts_df: pd.DataFrame, metadata_df: Optional[pd.DataFrame] = None, color_by: Optional[str] = None, save_path: Optional[str] = None):
    """Plots the distribution of library sizes (total counts per sample)."""
    logger.info(f"Plotting library size distribution (color by: {color_by})...")
    if not isinstance(counts_df, pd.DataFrame):
        logger.error("Input counts must be a pandas DataFrame.")
        return

    # Assume samples are rows if metadata is provided, otherwise assume samples are columns
    if metadata_df is not None:
        library_sizes = counts_df.sum(axis=1)
        plot_df = pd.DataFrame({'library_size': library_sizes}).join(metadata_df)
        x_col = color_by if color_by and color_by in plot_df.columns else None
        y_col = 'library_size'
    else: # Assume samples are columns if no metadata
        library_sizes = counts_df.sum(axis=0)
        plot_df = pd.DataFrame({'library_size': library_sizes})
        x_col = None # Cannot color without metadata
        y_col = 'library_size'


    fig, ax = plt.subplots(figsize=(10, 6))
    if x_col:
        sns.boxplot(data=plot_df, x=x_col, y=y_col, ax=ax, showfliers=False)
        sns.stripplot(data=plot_df, x=x_col, y=y_col, ax=ax, alpha=0.5, color='black', size=3)
        ax.set_xlabel(color_by)
    else:
        sns.histplot(data=plot_df, x=y_col, kde=True, ax=ax)
        ax.set_xlabel("Library Size (Total Counts)")

    ax.set_title("Library Size Distribution per Sample")
    ax.set_ylabel("Total Counts")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    save_plot(fig, save_path, "library_size_distribution.png")


def plot_detected_genes(counts_df: pd.DataFrame, threshold: int = 0, metadata_df: Optional[pd.DataFrame] = None, color_by: Optional[str] = None, save_path: Optional[str] = None):
    """Plots the distribution of detected genes per sample."""
    logger.info(f"Plotting detected genes distribution (threshold > {threshold}, color by: {color_by})...")
    if not isinstance(counts_df, pd.DataFrame):
        logger.error("Input counts must be a pandas DataFrame.")
        return

    # Assume samples are rows if metadata is provided
    if metadata_df is not None:
        detected_genes = (counts_df > threshold).sum(axis=1)
        plot_df = pd.DataFrame({'detected_genes': detected_genes}).join(metadata_df)
        x_col = color_by if color_by and color_by in plot_df.columns else None
        y_col = 'detected_genes'
    else: # Assume samples are columns
        detected_genes = (counts_df > threshold).sum(axis=0)
        plot_df = pd.DataFrame({'detected_genes': detected_genes})
        x_col = None
        y_col = 'detected_genes'

    fig, ax = plt.subplots(figsize=(10, 6))
    if x_col:
        sns.boxplot(data=plot_df, x=x_col, y=y_col, ax=ax, showfliers=False)
        sns.stripplot(data=plot_df, x=x_col, y=y_col, ax=ax, alpha=0.5, color='black', size=3)
        ax.set_xlabel(color_by)
    else:
        sns.histplot(data=plot_df, x=y_col, kde=True, ax=ax)
        ax.set_xlabel("Number of Detected Genes")

    ax.set_title(f"Detected Genes per Sample (Count > {threshold})")
    ax.set_ylabel("Number of Genes")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    save_plot(fig, save_path, "detected_genes_distribution.png")


def plot_gene_mean_variance(counts_df: pd.DataFrame, save_path: Optional[str] = None):
    """Plots mean vs variance for genes."""
    logger.info("Plotting gene mean vs variance...")
    if not isinstance(counts_df, pd.DataFrame):
        logger.error("Input counts must be a pandas DataFrame.")
        return

    # Assume samples are rows, genes are columns
    means = counts_df.mean(axis=0)
    variances = counts_df.var(axis=0)

    plot_df = pd.DataFrame({'mean': means, 'variance': variances})

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=plot_df, x='mean', y='variance', ax=ax, alpha=0.3, s=10, edgecolor=None)
    ax.set_xlabel("Mean Gene Expression (Counts)")
    ax.set_ylabel("Variance Gene Expression")
    ax.set_title("Mean-Variance Relationship")
    ax.set_xscale('log')
    ax.set_yscale('log')
    # Add line y=x for Poisson comparison
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='Variance = Mean (Poisson)')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.legend()
    plt.tight_layout()
    save_plot(fig, save_path, "mean_variance_plot.png")


def plot_mean_vs_zeros(counts_df: pd.DataFrame, save_path: Optional[str] = None):
    """Plots mean expression vs. fraction of zeros per gene."""
    logger.info("Plotting mean expression vs zero fraction...")
    if not isinstance(counts_df, pd.DataFrame):
        logger.error("Input counts must be a pandas DataFrame.")
        return

    # Assume samples are rows, genes are columns
    means = counts_df.mean(axis=0)
    zero_fraction = (counts_df == 0).mean(axis=0)

    plot_df = pd.DataFrame({'log10_mean_plus_1': np.log10(means + 1), 'zero_fraction': zero_fraction})

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=plot_df, x='log10_mean_plus_1', y='zero_fraction', ax=ax, alpha=0.3, s=10, edgecolor=None)
    ax.set_xlabel("Log10 (Mean Count + 1)")
    ax.set_ylabel("Fraction of Zero Counts")
    ax.set_title("Mean Expression vs. Zero Fraction per Gene")
    plt.tight_layout()
    save_plot(fig, save_path, "mean_vs_zeros.png")


def plot_size_factor_distribution(size_factors: pd.Series, metadata_df: Optional[pd.DataFrame] = None, color_by: Optional[str] = None, save_path: Optional[str] = None):
    """Plots the distribution of calculated size factors."""
    logger.info(f"Plotting size factor distribution (color by: {color_by})...")
    if not isinstance(size_factors, pd.Series):
        logger.error("Input size_factors must be a pandas Series.")
        return

    plot_df = pd.DataFrame({'size_factor': size_factors})
    if metadata_df is not None:
        plot_df = plot_df.join(metadata_df)
        x_col = color_by if color_by and color_by in plot_df.columns else None
        y_col = 'size_factor'
    else:
        x_col = None
        y_col = 'size_factor'

    fig, ax = plt.subplots(figsize=(10, 6))
    if x_col:
        sns.boxplot(data=plot_df, x=x_col, y=y_col, ax=ax, showfliers=False)
        sns.stripplot(data=plot_df, x=x_col, y=y_col, ax=ax, alpha=0.5, color='black', size=3)
        ax.set_xlabel(color_by)
    else:
        sns.histplot(data=plot_df, x=y_col, kde=True, ax=ax)
        ax.set_xlabel("Size Factor")

    ax.set_title("Distribution of Size Factors per Sample")
    ax.set_ylabel("Size Factor")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    save_plot(fig, save_path, "size_factor_distribution.png")


def plot_count_transformations(raw_counts_series: pd.Series, size_factor: Optional[float] = None, mean_size_factor: Optional[float] = None, title: str = "Count Transformations", save_path: Optional[str] = None):
    """Plots distributions of raw, log1p, and optionally size-factor normalized counts."""
    logger.info(f"Plotting count transformations for: {raw_counts_series.name}")
    if not isinstance(raw_counts_series, pd.Series):
        logger.error("Input must be a pandas Series (counts for one gene or sample).")
        return

    data_to_plot = {'Raw Counts': raw_counts_series}
    data_to_plot['Log1p(Counts)'] = np.log1p(raw_counts_series)

    if size_factor is not None and mean_size_factor is not None and size_factor > 0:
         # Example: DESeq2-like normalized log1p
         norm_counts = raw_counts_series / size_factor * mean_size_factor
         data_to_plot['Log1p(SizeFactorNorm)'] = np.log1p(norm_counts)

    n_plots = len(data_to_plot)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    fig.suptitle(title)

    for i, (label, data) in enumerate(data_to_plot.items()):
        ax = axes[i] if n_plots > 1 else axes
        sns.histplot(data, kde=True, ax=ax, bins=30)
        ax.set_title(label)
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")

    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust for suptitle
    save_plot(fig, save_path, f"{title.replace(' ','_')}_transformations.png")



