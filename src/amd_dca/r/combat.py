"""
ComBat batch‑effect correction via rpy2
– returns an integer count matrix with negatives clipped to 0.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

pandas2ri.activate()
ro.r("library(sva)")

# ------------------------------------------------------------------ #
def correct(counts: pd.DataFrame, batch: pd.Series) -> pd.DataFrame:
    """
    Parameters
    ----------
    counts : DataFrame  (samples × genes)  raw integer counts
    batch  : Series     (samples,)         batch labels

    Returns
    -------
    DataFrame (samples × genes)  ComBat‑corrected integer counts
    """
    # ---------- 1) log2‑CPM ------------------------------------------
    lib_sizes = counts.sum(axis=1)                        # per sample
    cpm       = counts.div(lib_sizes, axis=0) * 1e6
    logcpm    = np.log2(cpm + 1.0)
    logcpm_t  = logcpm.T                                  # genes × samples (for R)

    # ---------- 2) run ComBat in R ----------------------------------
    combat = ro.r["ComBat"]
    corrected_r = combat(
        pandas2ri.py2rpy(logcpm_t),
        pandas2ri.py2rpy(batch.astype(str))
    )

    # ---------- 3) R matrix → NumPy → DataFrame ---------------------
    corrected_np = np.asarray(corrected_r)                # genes × samples
    corrected_df = (
        pd.DataFrame(
            corrected_np,
            index=logcpm_t.index,                         # genes
            columns=logcpm_t.columns                      # samples
        )
        .T                                               # back to samples × genes
    )

    # ---------- 4) back‑transform to counts -------------------------
    cpm_adj    = (2.0 ** corrected_df) - 1.0
    counts_adj = cpm_adj.mul(lib_sizes, axis=0) / 1e6

    # ---------- 5) clip negatives, round to int --------------------
    counts_adj[counts_adj < 0] = 0
    counts_int = counts_adj.round().astype(int)

    return counts_int