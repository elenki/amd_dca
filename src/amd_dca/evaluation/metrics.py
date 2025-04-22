from matplotlib import pyplot as plt
from matplotlib_venn import venn2
import numpy as np
import pandas as pd
import logging, os
from pathlib import Path
from typing import Union

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
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Saved venn diagram to %s", save_path)
    plt.close(fig)


def volcano(
    res: pd.DataFrame,
    ann_path: Union[str, Path],
    title: str,
    save_path: str,
    fc_thresh: float = 1.0,
    pval_thresh: float = 0.05,
    top_n: int = 10
) -> None:
    """
    Enhanced volcano plot:
      • colors points by significance/FC thresholds
      • annotates top_n most significant up/down genes by name
      • uses an annotation TSV (with columns GeneID, Symbol)
    """
    # 1) load annotation
    ann = (
        pd.read_csv(ann_path, sep="\t", usecols=["GeneID", "Symbol"], dtype=str)
          .drop_duplicates("GeneID")
          .rename(columns={"GeneID": "gene_id", "Symbol": "gene_name"})
          .set_index("gene_id")
    )

    # 2) prepare DataFrame
    df = res.copy()
    df["neg_log10_p"] = -np.log10(df["pvalue"])
    df = df.join(ann, how="left")

    # 3) categorize by thresholds
    df["sig"] = "Not sig"
    up   = (df["padj"] < pval_thresh) & (df["log2FC"] >=  fc_thresh)
    down = (df["padj"] < pval_thresh) & (df["log2FC"] <= -fc_thresh)
    df.loc[ up,  "sig"] = "Up"
    df.loc[down, "sig"] = "Down"

    colors = {"Not sig":"lightgrey", "Up":"red", "Down":"blue"}

    # 4) plot
    fig, ax = plt.subplots(figsize=(6,6))
    for cat, col in colors.items():
        sub = df[df["sig"] == cat]
        ax.scatter(
            sub["log2FC"], sub["neg_log10_p"],
            c=col, label=cat, alpha=0.6, edgecolors="none", s=20
        )

    # draw threshold lines
    ax.axhline(-np.log10(pval_thresh), color="black", linestyle="--", lw=1)
    ax.axvline( fc_thresh, color="black", linestyle="--", lw=1)
    ax.axvline(-fc_thresh, color="black", linestyle="--", lw=1)

    # 5) annotate top N genes in each category
    def _annotate(subdf, offset: float):
        top = subdf.nsmallest(top_n, "padj")
        for gid, row in top.iterrows():
            name = row["gene_name"] if pd.notna(row["gene_name"]) else gid
            ax.text(
                row["log2FC"],
                row["neg_log10_p"] + offset,
                name,
                fontsize=6,
                ha="center",
                va="bottom" if offset>0 else "top"
            )

    _annotate(df[df["sig"]=="Up"],    offset=0.02)
    _annotate(df[df["sig"]=="Down"],  offset=-0.02)

    ax.set_xlabel("log₂(Fold Change)")
    ax.set_ylabel("-log₁₀(p‑value)")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)

    # save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved volcano plot to %s", save_path)