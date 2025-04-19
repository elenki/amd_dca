# src/amd_dca/r/dge.py

from __future__ import annotations
import pandas as pd
import logging

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, vectors
from rpy2.robjects.packages import importr

logger = logging.getLogger(__name__)

# activate pandas<->R conversions
pandas2ri.activate()

# import R libraries
importr("BiocGenerics")  # needed for mcols
_DESEQ2 = importr("DESeq2")
_base   = importr("base")

# define the R helper
ro.r(r'''
run_deseq2_wrapper <- function(countData, colData, design_formula,
                               contrast=NULL, skipNorm=FALSE) {
  library(DESeq2)
  dds <- DESeqDataSetFromMatrix(
    countData = countData,
    colData   = colData,
    design    = as.formula(design_formula)
  )
  if (skipNorm) {
    # prevent DESeq2 from re-estimating size factors
    sizeFactors(dds) <- rep(1, ncol(dds))
  }
  # three‐stage dispersion fitting
  dds <- tryCatch(
    DESeq(dds),
    error = function(e1) {
      message("Parametric fit failed: ", e1$message)
      tryCatch(
        DESeq(dds, fitType="local"),
        error = function(e2) {
          message("Local fit failed: ", e2$message)
          # gene-wise fallback
          dds3 <- estimateDispersionsGeneEst(dds)
          dispersions(dds3) <- mcols(dds3)$dispGeneEst
          nbinomWaldTest(dds3)
        }
      )
    }
  )
  # extract results
  if (!is.null(contrast)) {
    res <- results(dds, contrast=contrast)
  } else {
    res <- results(dds)
  }
  as.data.frame(res)
}
''')
_run_deseq2 = ro.globalenv["run_deseq2_wrapper"]


def _prep_meta(meta: pd.DataFrame, group: str | None) -> tuple[pd.DataFrame,str]:
    df = meta.copy()
    if group is None or group not in df.columns:
        group = next((c for c in ["mgs_level"] if c in df.columns), df.columns[0])
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype("category")
    if df[group].dtype.name != "category":
        df[group] = df[group].astype("category")
    cols = [group] + [c for c in df.columns if c != group]
    return df[cols], group


def deseq2(
    counts:   pd.DataFrame,
    meta:     pd.DataFrame,
    group:    str | None = None,
    contrast: tuple[str,str,str] | None = None,
    skip_norm: bool = False
) -> pd.DataFrame:
    """
    Run DESeq2 with a 3‐stage dispersion fallback.
    If skip_norm=True, pre‐sets all sizeFactors=1.
    """
    # prepare metadata
    meta2, gcol = _prep_meta(meta, group)
    design = f"~ {gcol}"

    # to R
    r_counts   = pandas2ri.py2rpy(counts.T.astype(int))
    r_colData  = pandas2ri.py2rpy(meta2)
    r_contrast = vectors.StrVector(contrast) if contrast is not None else ro.NULL
    r_skip     = ro.BoolVector([skip_norm])[0]

    # call R helper
    res_r = _run_deseq2(r_counts, r_colData, design, r_contrast, r_skip)

    # back to pandas
    df = pandas2ri.rpy2py(res_r)
    df.index = counts.columns
    return (
        df
        .rename(columns={
            "log2FoldChange": "log2FC",
            "pvalue":         "pvalue",
            "padj":           "padj"
        })
        [["log2FC","pvalue","padj"]]
        .astype(float)
    )