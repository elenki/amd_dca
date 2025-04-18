from matplotlib import pyplot as plt
from matplotlib_venn import venn2
import numpy as np
import pandas as pd
import logging, os

logger = logging.getLogger(__name__)

def venn_two(res_a: pd.DataFrame, res_b: pd.DataFrame,
             label_a: str, label_b: str,
             alpha: float = 0.05, save_path: str | None = None) -> None:
    """Venn diagram of padj<alpha genes."""
    set_a = set(res_a[res_a["padj"] < alpha].index)
    set_b = set(res_b[res_b["padj"] < alpha].index)
    fig, ax = plt.subplots(figsize=(4,4))
    venn2([set_a, set_b], (label_a, label_b), ax=ax)
    plt.title(f"DE overlap (padj < {alpha})")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Saved venn diagram to %s", save_path)
    plt.close(fig)

def volcano(res: pd.DataFrame, title: str, save_path: str) -> None:
    """Basic volcano plot."""
    fig, ax = plt.subplots(figsize=(5,4))
    ax.scatter(res["log2FC"], -np.log10(res["pvalue"]), s=8, alpha=0.4)
    ax.set_xlabel("log2 FC")
    ax.set_ylabel("-log10 p")
    ax.set_title(title)
    ax.axhline(-np.log10(0.05), color="red", ls="--", lw=0.8)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)