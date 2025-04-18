# src/amd_dca/data/preprocess.py
import pandas as pd
import numpy as np
import re
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Tuple, Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


def load_data(counts_path: str,
              metadata_path: str,
              mapping_path: str
             ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load raw counts, GEO metadata, and GSM->Title mapping.
    Returns (counts_df, metadata_df, mapping_df).
    """
    logger.info(f"[preprocess] Reading raw counts: {counts_path}")
    counts_df = pd.read_csv(counts_path, sep='\t', index_col=0)
    logger.info(f"[preprocess] Counts shape: {counts_df.shape}")

    logger.info(f"[preprocess] Reading metadata: {metadata_path}")
    metadata_df = pd.read_csv(metadata_path)
    logger.info(f"[preprocess] Metadata shape: {metadata_df.shape}")

    logger.info(f"[preprocess] Reading mapping: {mapping_path}")
    mapping_df = pd.read_csv(mapping_path, usecols=['Accession','Title'])
    logger.info(f"[preprocess] Mapping shape: {mapping_df.shape}")

    return counts_df, metadata_df, mapping_df

def map_and_combine(
    counts_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    mapping_df: pd.DataFrame
) -> Optional[pd.DataFrame]:
    """
    Maps raw-counts columns (GSM accessions) to metadata rows via an
    intermediate 'linking_id' extracted both from metadata_df['r_id']
    and mapping_df['Title'], then returns samples×genes joined to metadata.

    counts_df:   genes × samples (columns are GSM IDs)
    metadata_df: must have 'r_id' of the form '123_4'
    mapping_df:  must have 'Accession' (GSM...) and 'Title'
                 e.g. 'R42015-419pf_1-IR_L7' → linking_id='419'
    """
    logger.info("Attempting to map and combine counts, metadata, and mapping sheet…")

    # 1) Build linking_id in metadata_df
    if 'r_id' not in metadata_df.columns:
        logger.error("metadata_df missing required 'r_id' column.")
        return None
    md = metadata_df.copy()
    md['r_id'] = md['r_id'].astype(str)
    md = md.dropna(subset=['r_id'])
    md['linking_id'] = md['r_id'].str.split('_').str[0]
    # drop duplicates so each linking_id maps to one metadata record
    md = md.drop_duplicates(subset=['linking_id'])
    logger.info(f"[preprocess] Metadata: {len(md)} rows with {md['linking_id'].nunique()} unique linking_id.")

    # 2) Build linking_id in mapping_df
    if 'Title' not in mapping_df.columns or 'Accession' not in mapping_df.columns:
        logger.error("mapping_df must contain 'Title' and 'Accession' columns.")
        return None
    mp = mapping_df.copy()
    # capture the digits before 'pf_' (e.g. 'R42015-419pf_' → '419')
    pattern = re.compile(r'-(\d+)pf_')
    mp['linking_id'] = mp['Title'].str.extract(pattern, expand=False)
    n_missing = mp['linking_id'].isna().sum()
    if n_missing:
        logger.warning(f"[preprocess] {n_missing} mapping rows did not match pattern '-(\\d+)pf_'.")
    mp = mp.dropna(subset=['linking_id'])
    mp['linking_id'] = mp['linking_id'].astype(str)
    # keep only one Accession per linking_id
    mp = mp.drop_duplicates(subset=['linking_id'])
    logger.info(f"[preprocess] Mapping sheet: {len(mp)} rows after extracting linking_id & deduplication.")

    # 3) Build dict from GSM accession → linking_id
    acc2link = dict(zip(mp['Accession'], mp['linking_id']))

    # 4) Filter & rename counts columns
    valid_gsms = [gsm for gsm in counts_df.columns if gsm in acc2link]
    if not valid_gsms:
        logger.error("No GSM IDs in counts_df matched any Accession in mapping_df.")
        return None
    counts_subset = counts_df[valid_gsms]
    counts_renamed = counts_subset.rename(columns=acc2link)
    logger.info(f"[preprocess] Kept {len(valid_gsms)} of {len(counts_df.columns)} samples in counts.")

    # 5) Transpose so samples are rows, index is linking_id
    counts_t = counts_renamed.transpose()
    counts_t.index.name = 'linking_id'
    logger.info(f"[preprocess] Transposed counts shape: {counts_t.shape}")

    # 6) Join with metadata on linking_id
    md_indexed = md.set_index('linking_id')
    combined = counts_t.join(md_indexed, how='inner')
    logger.info(f"[preprocess] Combined data shape: {combined.shape}")

    if combined.empty:
        logger.error("Joining counts and metadata resulted in an empty DataFrame. Check your linking_id logic.")
        return None

    return combined


def filter_samples(combined_df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Filter samples by RIN threshold."""
    rin_thr = config['preprocessing']['rin_threshold']
    logger.info(f"[preprocess] Filtering samples with RIN ≥ {rin_thr}")
    df = combined_df.copy()
    if 'rin' not in df.columns:
        logger.warning("No 'rin' column found; skipping sample filter")
        return df
    df['rin'] = pd.to_numeric(df['rin'], errors='coerce')
    df = df.dropna(subset=['rin'])
    return df[df['rin'] >= rin_thr]


def filter_genes(counts_df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Keep genes with ≥ min_counts in ≥ min_pct of samples."""
    min_counts = config['preprocessing']['min_counts_per_gene']
    min_pct    = config['preprocessing']['min_samples_per_gene_pct']
    logger.info(f"[preprocess] Filtering genes: count>={min_counts} in ≥{min_pct*100:.0f}% samples")
    n = counts_df.shape[0]
    mask = (counts_df > min_counts).sum(axis=0) >= int(min_pct * n)
    out = counts_df.loc[:, mask]
    logger.info(f"[preprocess] Genes retained: {out.shape[1]} of {counts_df.shape[1]}")
    return out


def split_data(sample_ids: pd.Index, meta_df: pd.DataFrame, config: Dict[str, Any]
) -> Tuple[pd.Index,pd.Index,pd.Index]:
    """Train/val/test split stratified on config['preprocessing']['stratify_on']."""
    strat = config['preprocessing'].get('stratify_on')
    seed  = config['random_seed']
    test_sz = config['preprocessing']['split_ratios']['test']
    val_sz  = config['preprocessing']['split_ratios']['validation']
    logger.info(f"[preprocess] Splitting: test={test_sz}, val={val_sz}, stratify={strat}")

    stratify_vals = None
    if strat and strat in meta_df.columns:
        stratify_vals = meta_df.loc[sample_ids, strat]
    tv, test = train_test_split(
        sample_ids, test_size=test_sz, random_state=seed,
        stratify=stratify_vals
    )
    val_adj = val_sz / (1 - test_sz)
    stratify_tv = stratify_vals.loc[tv] if stratify_vals is not None else None
    train, val = train_test_split(
        tv, test_size=val_adj, random_state=seed,
        stratify=stratify_tv
    )
    logger.info(f"[preprocess] Split sizes: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


def prepare_covariates(
    meta_df: pd.DataFrame,
    covs: List[str],
    train_idx: pd.Index
) -> Tuple[pd.DataFrame, Optional[OneHotEncoder], Optional[StandardScaler]]:
    """One‐hot encode cats, standardize nums, fit only on train_idx."""
    df = meta_df.copy()
    avail = [c for c in covs if c in df.columns]
    if not avail:
        logger.warning("[preprocess] No covariates found → returning empty DF")
        return pd.DataFrame(index=df.index), None, None

    df_sel = df[avail].copy()
    # impute
    for c in avail:
        if df_sel[c].isna().any():
            if pd.api.types.is_numeric_dtype(df_sel[c]):
                fill = df_sel.loc[train_idx, c].mean()
            else:
                fill = df_sel.loc[train_idx, c].mode()[0]
            df_sel[c] = df_sel[c].fillna(fill)

    cats = df_sel.select_dtypes(include=['object','category']).columns.tolist()
    nums = df_sel.select_dtypes(include=[np.number]).columns.tolist()

    parts = []
    enc, scl = None, None

    if nums:
        scl = StandardScaler()
        scl.fit(df_sel.loc[train_idx, nums])
        parts.append(pd.DataFrame(
            scl.transform(df_sel[nums]),
            index=df_sel.index, columns=nums
        ))
    if cats:
        enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        enc.fit(df_sel.loc[train_idx, cats])
        cols = enc.get_feature_names_out(cats)
        parts.append(pd.DataFrame(
            enc.transform(df_sel[cats]),
            index=df_sel.index, columns=cols
        ))

    covariate_df = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df_sel.index)
    logger.info(f"[preprocess] Covariates shape: {covariate_df.shape}")
    return covariate_df, enc, scl


def calculate_size_factors(counts_df: pd.DataFrame) -> pd.Series:
    """
    Calculate size factors using the median-of-ratios method.
    Handles zeros by ignoring genes with zero geometric mean.

    Args:
        counts_df (pd.DataFrame): Raw counts matrix (samples x genes).

    Returns:
        pd.Series: Size factors per sample.
    """
    logger.info("Calculating size factors using median-of-ratios method...")
    # Ensure input is samples x genes
    if not isinstance(counts_df, pd.DataFrame):
        raise TypeError("Input counts must be a pandas DataFrame.")
    if counts_df.shape[0] < counts_df.shape[1]:
        logger.warning("Input DataFrame has more genes than samples. Ensure samples are rows.")

    # Calculate geometric mean per gene, ignoring zeros
    # Replace 0s with NaN temporarily to calculate geometric mean correctly
    counts_no_zero = counts_df.replace(0, np.nan)
    log_counts = np.log(counts_no_zero)
    log_geo_means = log_counts.mean(axis=0) # Log of geometric means per gene

    # Filter out genes where geometric mean is NaN (e.g., all zeros in a gene)
    valid_geo_means = log_geo_means.notna()
    if not valid_geo_means.any():
        logger.error("Could not calculate geometric mean for any gene. Check input data.")
        # Return default size factors of 1? Or raise error?
        return pd.Series(1.0, index=counts_df.index)

    # Calculate ratio of counts to geometric mean for each sample/gene
    # Use broadcasting: counts_df (samples x genes) / exp(log_geo_means (genes))
    # Need to handle division by zero or issues if geo_mean is 0 (log_geo_mean is -inf)
    # Work in log space: log(counts) - log_geo_means
    # Filter counts and log_geo_means to only valid genes
    log_ratios = log_counts.loc[:, valid_geo_means] - log_geo_means[valid_geo_means]

    # Calculate median of these ratios for each sample (log scale)
    log_median_ratios = log_ratios.median(axis=1) # Median per sample

    # Convert back to linear scale: exp(log_median_ratios)
    size_factors = np.exp(log_median_ratios)

    # Handle potential NaN/Inf values in size factors (e.g., if a sample has all zeros)
    size_factors = size_factors.fillna(1.0).replace([np.inf, -np.inf], 1.0)
    # Ensure no zero size factors
    size_factors[size_factors == 0] = 1.0

    logger.info(f"Calculated size factors for {len(size_factors)} samples.")
    return size_factors


def get_zero_inflation_stats(counts_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculates basic statistics related to zero counts.

    Args:
        counts_df (pd.DataFrame): Raw counts matrix (samples x genes).

    Returns:
        Dict[str, Any]: Dictionary containing zero statistics.
    """
    logger.info("Calculating zero inflation statistics...")
    if not isinstance(counts_df, pd.DataFrame):
        raise TypeError("Input counts must be a pandas DataFrame.")

    total_elements = counts_df.size
    total_zeros = (counts_df == 0).sum().sum()
    overall_sparsity = total_zeros / total_elements if total_elements > 0 else 0

    # Per-gene zero fraction
    zero_fraction_per_gene = (counts_df == 0).mean(axis=0) # Mean over samples for each gene

    stats = {
        "total_samples": counts_df.shape[0],
        "total_genes": counts_df.shape[1],
        "total_counts": total_elements,
        "total_zeros": total_zeros,
        "overall_sparsity": overall_sparsity,
        "zero_fraction_per_gene_stats": zero_fraction_per_gene.describe().to_dict()
        # Optionally return the full series: 'zero_fraction_per_gene': zero_fraction_per_gene
    }
    logger.info(f"Overall sparsity: {overall_sparsity:.3f}")
    return stats