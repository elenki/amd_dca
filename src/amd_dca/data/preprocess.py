from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import logging
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)

# --- I: DATA LOAD & MAPPING --- #

def load_data(counts_path: Path, metadata_path: Path, mapping_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      counts_df (samples x genes)  indexed by linking_id
      meta_df   (samples x covariates) indexed by linking_id
    """
    logger.info(f"Reading raw counts: {counts_path}")
    cnt = pd.read_csv(counts_path, sep="\t", index_col=0)
    logger.info(f"Counts shape: {cnt.shape}")
    logger.info(f"Reading metadata: {metadata_path}")
    meta = pd.read_csv(metadata_path)
    logger.info(f"Metadata shape: {meta.shape}")
    logger.info(f"Reading mapping: {mapping_path}")
    mp = pd.read_csv(mapping_path)
    # derive linking_id
    mp['linking_id'] = mp['Title'].str.split('_').str[0]
    mp = mp.set_index('Accession')
    # rename counts columns GSM -> linking_id
    rename_map = mp['linking_id'].to_dict()
    cnt = cnt.rename(columns=rename_map)
    cnt = cnt.loc[:, cnt.columns.notna()]
    counts = cnt.T
    counts.index.name = 'linking_id'
    # attach metadata
    if 'Accession' in meta.columns:
        meta2 = meta.set_index('Accession').join(mp[['linking_id']])
        meta2 = meta2.set_index('linking_id', drop=True)
    else:
        # fallback: metadata already has Title
        meta['linking_id'] = meta['Title'].str.split('_').str[0]
        meta2 = meta.set_index('linking_id')
    return counts, meta2

# --- II: FILTERING --- #

def filter_samples(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    thr = cfg['preprocessing']['rin_threshold']
    if 'rin' in df.columns:
        df = df[df['rin'].astype(float) >= thr]
        logger.info(f"Samples after RIN>={thr}: {df.shape[0]}")
    return df


def filter_genes(counts: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    min_cnt = cfg['preprocessing']['min_counts_per_gene']
    pct     = cfg['preprocessing']['min_samples_per_gene_pct']
    nc = int(counts.shape[0]*pct)
    keep = (counts>min_cnt).sum(axis=0)>=nc
    logger.info(f"Genes after filter: {keep.sum()} / {len(keep)}")
    return counts.loc[:, keep]

# --- III: SIZE FACTORS --- #

def calculate_size_factors(counts: pd.DataFrame) -> pd.Series:
    return _median_of_ratios(counts)

def _median_of_ratios(counts: pd.DataFrame) -> pd.Series:
    logc = np.log(counts.replace(0, np.nan))
    geo = logc.mean(axis=0)
    valid = geo.notna()
    lr = logc.loc[:, valid] - geo[valid]
    sf = np.exp(lr.median(axis=1))
    return sf.fillna(1.0)

# --- IV: ComBat wrapper --- #
from amd_dca.r.combat import correct
def run_combat(counts: pd.DataFrame, batch: pd.Series) -> pd.DataFrame:
    return correct(counts, batch)

# --- V: SPLITTING --- #

def split_data(sample_ids, meta_df, cfg):
    strat = cfg['preprocessing']['stratify_on']
    s = meta_df.loc[sample_ids, strat] if strat in meta_df.columns else None
    t = cfg['preprocessing']['split_ratios']
    tr, te = train_test_split(sample_ids, test_size=t['test'], stratify=s)
    val_frac = t['validation']/(1-t['test'])
    str2 = meta_df.loc[tr, strat] if s is not None else None
    tr2, va = train_test_split(tr, test_size=val_frac, stratify=str2)
    return tr2, va, te

# --- VI: COVARIATES --- #

def prepare_covariates(meta: pd.DataFrame, covs: list, train_ids) -> Tuple[pd.DataFrame, Optional[OneHotEncoder], Optional[StandardScaler]]:
    df = meta[covs].copy()
    df = df.fillna(method='ffill')
    num = df.select_dtypes(include=float).columns.tolist()
    cat = df.select_dtypes(include=object).columns.tolist()
    scaler, encoder = None, None
    parts = []
    if num:
        scaler = StandardScaler().fit(df.loc[train_ids, num])
        parts.append(pd.DataFrame(scaler.transform(df[num]), index=df.index, columns=num))
    if cat:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoder.fit(df.loc[train_ids, cat])
        arr = encoder.transform(df[cat])
        parts.append(pd.DataFrame(arr, index=df.index, columns=encoder.get_feature_names_out()))
    covs_df = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)
    return covs_df, encoder, scaler