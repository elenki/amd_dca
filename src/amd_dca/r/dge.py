"""
Differential‑expression wrappers via rpy2:
  • DESeq2
  • edgeR + voom (+ limma)

Always returns a pandas DataFrame (index = gene) with exactly:
    log2FC, pvalue, padj
"""

from __future__ import annotations
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, vectors
from rpy2.rinterface import NULLType

# Activate the automatic pandas <-> R conversions
pandas2ri.activate()

# ── load R libraries & grab the functions we need ───────────────────────── #
ro.r("library(DESeq2)")
r_DESeqDataSet = ro.r("DESeqDataSetFromMatrix")
r_DESeq        = ro.r("DESeq")
r_results      = ro.r("results")
r_as_df        = ro.r("as.data.frame")

ro.r("library(edgeR); library(limma)")
r_DGEList       = ro.r("DGEList")
r_calcNorm      = ro.r("calcNormFactors")
r_voom          = ro.r("voom")
r_lmFit         = ro.r("lmFit")
r_eBayes        = ro.r("eBayes")
r_makeContrasts = ro.r("makeContrasts")
r_contrastsFit  = ro.r("contrasts.fit")
r_topTable      = ro.r("topTable")
r_modelMatrix   = ro.r("model.matrix")


# ── helper to prep metadata for R ────────────────────────────────────────── #
def _prep_meta(meta: pd.DataFrame, group: str | None) -> tuple[pd.DataFrame, str]:
    meta = meta.copy()
    # pick grouping column
    if not group or group not in meta.columns:
        group = "mgs_level" if "mgs_level" in meta.columns else meta.columns[0]
    # convert any object‐dtype columns to categorical
    for c in meta.select_dtypes(include="object").columns:
        meta[c] = meta[c].astype("category")
    if meta[group].dtype.name != "category":
        meta[group] = meta[group].astype("category")
    # ensure that group is first
    cols = [group] + [c for c in meta.columns if c != group]
    return meta[cols], group


# ── DESeq2 wrapper ───────────────────────────────────────────────────────── #
def deseq2(counts: pd.DataFrame,
           meta:   pd.DataFrame,
           group:  str | None = None,
           contrast: tuple[str,str,str] | None = None
          ) -> pd.DataFrame:
    """
    DESeq2 on raw integer counts (samples×genes).
    If `contrast` is provided, it should be (factor, level1, level2).
    """
    meta, g = _prep_meta(meta, group)

    dds = r_DESeqDataSet(
        countData = pandas2ri.py2rpy(counts.T.astype(int)),
        colData   = pandas2ri.py2rpy(meta),
        design    = ro.Formula(f"~ {g}")
    )
    dds = r_DESeq(dds)

    if contrast:
        res_r = r_results(dds, contrast=vectors.StrVector(contrast))
    else:
        res_r = r_results(dds)

    df = pandas2ri.rpy2py(r_as_df(res_r))
    df.index = counts.columns

    return (
        df
        .rename(columns={
            "log2FoldChange": "log2FC",
            "pvalue":         "pvalue",
            "padj":           "padj"
        })
        [["log2FC", "pvalue", "padj"]]
        .astype(float)
    )


# ── edgeR + voom (+ limma) wrapper ────────────────────────────────────────── #
# def edger_voom(counts:   pd.DataFrame,
#                meta:     pd.DataFrame,
#                group:    str | None = None,
#                contrast: str | None = None
#               ) -> pd.DataFrame:
#     """
#     edgeR TMM → limma‑voom pipeline.
#     `contrast` should be a string like "Batch2 - Batch1". If omitted,
#     we default to the second level minus the first.
#     """
#     meta, g = _prep_meta(meta, group)

#     # 1) Build + normalize DGEList
#     dge = r_DGEList(
#         counts  = pandas2ri.py2rpy(counts.T.astype(int)),
#         samples = pandas2ri.py2rpy(meta)
#     )
#     dge = r_calcNorm(dge)

#     # 2) Design matrix without intercept
#     meta_r   = pandas2ri.py2rpy(meta)
#     design_r = r_modelMatrix(
#         ro.Formula(f"~ 0 + {g}"),
#         meta_r
#     )

#     # 3) voom → fit
#     v   = r_voom(dge, design=design_r, plot=False)
#     fit = r_lmFit(v, design_r)

#     # 4) Try to pull the column names from the R object; fallback to pandas categories
#     colnames_r = ro.r("colnames")(design_r)
#     if isinstance(colnames_r, NULLType):
#         levels = list(meta[g].cat.categories)
#     else:
#         levels = list(colnames_r)

#     # 5) Default contrast if none given
#     if contrast is None:
#         if len(levels) < 2:
#             raise ValueError(f"Need at least two levels in '{g}' to define a contrast")
#         contrast = f"{levels[1]} - {levels[0]}"

#     # 6) Build contrast matrix, refit, empirical Bayes
#     cmat = r_makeContrasts(contrasts=vectors.StrVector([contrast]), levels=colnames_r if not isinstance(colnames_r, NULLType) else vectors.StrVector(levels))
#     fit  = r_contrastsFit(fit, cmat)
#     fit  = r_eBayes(fit)

#     # 7) Pull out top table for our contrast (coef=1)
#     top_r = r_topTable(fit, coef=1, number=ro.r("Inf"))
#     df    = pandas2ri.rpy2py(top_r)
#     df.index = counts.columns

#     return (
#         df
#         .rename(columns={
#             "logFC":     "log2FC",
#             "P.Value":   "pvalue",
#             "adj.P.Val": "padj"
#         })
#         [["log2FC", "pvalue", "padj"]]
#         .astype(float)
#     )