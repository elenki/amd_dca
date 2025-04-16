import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import logging # Keep the import
import re
from typing import List, Tuple, Dict, Optional, Any

# Use this for simple logging configuration (just displays messages to console)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get logger instance - it will inherit configuration from the main script
logger = logging.getLogger(__name__) # Use __name__ for module-specific logger

def load_data(counts_path: str, metadata_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads counts and metadata."""
    logger.info(f"Loading counts from: {counts_path}") # Use logger instance
    # ... (rest of the function using logger.info, logger.warning, logger.error) ...
    try:
        counts_df = pd.read_csv(counts_path, sep='\t', index_col=0)
        logger.info(f"Loaded counts data with shape: {counts_df.shape}")
    except Exception as e:
        logger.error(f"Failed to load counts data: {e}")
        raise

    logger.info(f"Loading metadata from: {metadata_path}")
    try:
        metadata_df = pd.read_csv(metadata_path)
        logger.info(f"Loaded metadata with shape: {metadata_df.shape}")
        if 'r_id' not in metadata_df.columns:
             logger.warning("Metadata missing expected 'r_id' column for mapping.")
    except Exception as e:
        logger.error(f"Failed to load metadata: {e}")
        raise

    return counts_df, metadata_df

def map_and_combine(counts_df: pd.DataFrame, metadata_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Maps counts columns to metadata rows and combines them."""
    logger.info("Attempting to map and combine counts and metadata...")

    # --- 1. Create linking_id in metadata ---
    if 'r_id' not in metadata_df.columns:
        logger.error("Metadata requires 'r_id' column for mapping based on current strategy.")
        return None
    try:
        metadata_df['r_id'] = metadata_df['r_id'].astype(str)
        metadata_df = metadata_df.dropna(subset=['r_id'])
        metadata_df['linking_id'] = metadata_df['r_id'].str.split('_').str[0]
        metadata_df = metadata_df.drop_duplicates(subset=['linking_id'])
        metadata_indexed = metadata_df.set_index('linking_id')
        logger.info(f"Created 'linking_id' index for metadata. Shape: {metadata_indexed.shape}")
    except Exception as e:
        logger.error(f"Failed to create linking_id from metadata 'r_id': {e}")
        return None

    # --- 2. Create linking_id map from counts columns ---
    rsem_columns = counts_df.columns.tolist()
    rsem_id_map = {}
    pattern = re.compile(r'pf_(\d+)-IR_')
    unique_extracted_ids = set()
    duplicate_check = {}

    for col_name in rsem_columns:
        match = pattern.search(col_name)
        if match:
            extracted_id = match.group(1)
            if extracted_id in duplicate_check:
                logger.warning(f"Duplicate linking_id '{extracted_id}' found in RSEM columns: '{col_name}' and '{duplicate_check[extracted_id]}'. Skipping duplicate.")
                continue
            duplicate_check[extracted_id] = col_name
            rsem_id_map[col_name] = extracted_id
            unique_extracted_ids.add(extracted_id)
        else:
            logger.warning(f"Could not extract linking ID from RSEM column: {col_name}")

    logger.info(f"Extracted {len(unique_extracted_ids)} unique linking IDs from {len(rsem_columns)} RSEM columns.")

    # --- 3. Rename counts columns and Transpose ---
    valid_rename_map = {k: v for k, v in rsem_id_map.items() if v is not None and v in unique_extracted_ids}
    counts_subset = counts_df[list(valid_rename_map.keys())]
    counts_renamed = counts_subset.rename(columns=valid_rename_map)
    counts_t = counts_renamed.transpose()
    counts_t.index.name = 'linking_id'
    logger.info(f"Transposed counts shape: {counts_t.shape}")

    # --- 4. Join ---
    combined_data = counts_t.join(metadata_indexed, how='inner')
    logger.info(f"Combined data shape after inner join: {combined_data.shape}")

    if combined_data.empty:
        logger.error("Joining counts and metadata resulted in an empty DataFrame. Check ID mapping.")
        return None

    logger.info(f"Example combined data index: {combined_data.index[:5].tolist()}")
    logger.info(f"Example combined data columns (start): {combined_data.columns[:5].tolist()}")
    logger.info(f"Example combined data columns (end): {combined_data.columns[-5:].tolist()}")

    return combined_data

# --- Other functions (filter_samples, filter_genes, split_data, prepare_covariates) ---
# --- Make sure they also use 'logger.info/warning/error' instead of print ---

def filter_samples(combined_df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Filters samples based on metadata criteria (e.g., RIN)."""
    rin_threshold = config['preprocessing']['rin_threshold']
    logger.info(f"Filtering samples by RIN >= {rin_threshold}...") # Use logger
    initial_count = combined_df.shape[0]
    if 'rin' not in combined_df.columns:
        logger.warning("'rin' column not found in combined data. Skipping RIN filtering.") # Use logger
        return combined_df

    combined_df['rin'] = pd.to_numeric(combined_df['rin'], errors='coerce')
    filtered_df = combined_df.dropna(subset=['rin'])
    removed_nan = initial_count - filtered_df.shape[0]
    if removed_nan > 0:
        logger.warning(f"Removed {removed_nan} samples due to missing RIN values.") # Use logger

    filtered_df = filtered_df[filtered_df['rin'] >= rin_threshold].copy()
    final_count = filtered_df.shape[0]
    logger.info(f"Samples remaining after RIN filter: {final_count} (removed {initial_count - final_count})") # Use logger
    return filtered_df

def filter_genes(counts_df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Filters genes based on minimum counts in a minimum percentage of samples."""
    min_counts = config['preprocessing']['min_counts_per_gene']
    min_samples_pct = config['preprocessing']['min_samples_per_gene_pct']
    logger.info(f"Filtering genes (min_count={min_counts}, min_samples_pct={min_samples_pct})...") # Use logger

    n_samples = counts_df.shape[0]
    min_samples = int(n_samples * min_samples_pct)

    genes_to_keep_mask = (counts_df > min_counts).sum(axis=0) >= min_samples
    filtered_counts_df = counts_df.loc[:, genes_to_keep_mask]

    logger.info(f"Genes remaining after filtering: {filtered_counts_df.shape[1]} (removed {counts_df.shape[1] - filtered_counts_df.shape[1]})") # Use logger
    return filtered_counts_df

def split_data(sample_ids: pd.Index, metadata_df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.Index, pd.Index, pd.Index]:
    """Splits sample IDs into train, validation, and test sets."""
    test_ratio = config['preprocessing']['split_ratios']['test']
    val_ratio = config['preprocessing']['split_ratios']['validation']
    stratify_col = config['preprocessing'].get('stratify_on', None)
    random_state = config['random_seed']
    logger.info(f"Splitting data (stratify by: {stratify_col})...") # Use logger

    stratify_array = metadata_df.loc[sample_ids, stratify_col].values if stratify_col and stratify_col in metadata_df.columns else None
    if stratify_array is None and stratify_col:
        logger.warning(f"Stratification column '{stratify_col}' not found in metadata. Performing random split.") # Use logger

    try:
        train_val_ids, test_ids = train_test_split(
            sample_ids, test_size=test_ratio, random_state=random_state,
            stratify=(stratify_array if stratify_array is not None else None)
        )
        val_ratio_adj = val_ratio / (1.0 - test_ratio)
        stratify_array_train_val = metadata_df.loc[train_val_ids, stratify_col].values if stratify_array is not None else None
        train_ids, val_ids = train_test_split(
            train_val_ids, test_size=val_ratio_adj, random_state=random_state,
            stratify=(stratify_array_train_val if stratify_array_train_val is not None else None)
        )
        logger.info(f"Split sizes: Train={len(train_ids)}, Validation={len(val_ids)}, Test={len(test_ids)}") # Use logger
        return pd.Index(train_ids), pd.Index(val_ids), pd.Index(test_ids)

    except Exception as e:
         logger.error(f"Error during data splitting (check stratification column '{stratify_col}' values and ratios): {e}") # Use logger
         logger.warning("Falling back to random split due to error.") # Use logger
         train_val_ids, test_ids = train_test_split(sample_ids, test_size=test_ratio, random_state=random_state)
         val_ratio_adj = val_ratio / (1.0 - test_ratio)
         train_ids, val_ids = train_test_split(train_val_ids, test_size=val_ratio_adj, random_state=random_state)
         logger.info(f"Fallback split sizes: Train={len(train_ids)}, Validation={len(val_ids)}, Test={len(test_ids)}") # Use logger
         return pd.Index(train_ids), pd.Index(val_ids), pd.Index(test_ids)


def prepare_covariates(metadata_df: pd.DataFrame, covariate_list: List[str], train_indices: pd.Index) -> Tuple[pd.DataFrame, Optional[OneHotEncoder], Optional[StandardScaler]]:
    """
    Selects, encodes categorical, scales continuous covariates, handles missing values.
    Fits scaler/encoder only on train_indices. Returns processed covariates for ALL samples.
    """
    logger.info(f"Preparing covariates: {covariate_list}") # Use logger

    available_covariates = [col for col in covariate_list if col in metadata_df.columns]
    missing_req_covariates = set(covariate_list) - set(available_covariates)
    if missing_req_covariates:
        logger.warning(f"Requested covariates not found in metadata: {missing_req_covariates}") # Use logger
    if not available_covariates:
        logger.warning("No available covariates found to prepare.") # Use logger
        return pd.DataFrame(index=metadata_df.index), None, None

    covariates_df = metadata_df[available_covariates].copy()

    # --- Handle Missing Values ---
    imputation_values = {}
    for col in covariates_df.columns:
        if covariates_df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(covariates_df[col]):
                fill_value = covariates_df.loc[train_indices, col].mean()
                imputation_values[col] = fill_value
                logger.info(f"Imputing missing numeric '{col}' with mean: {fill_value:.2f}") # Use logger
            else:
                fill_value = covariates_df.loc[train_indices, col].mode()[0]
                imputation_values[col] = fill_value
                logger.info(f"Imputing missing categorical '{col}' with mode: {fill_value}") # Use logger
            covariates_df[col] = covariates_df[col].fillna(fill_value)

    # --- Identify Column Types ---
    categorical_cols = covariates_df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = covariates_df.select_dtypes(include=np.number).columns.tolist()
    logger.info(f"Categorical covariates: {categorical_cols}") # Use logger
    logger.info(f"Numerical covariates: {numerical_cols}") # Use logger

    # --- Fit and Transform ---
    encoder = None
    scaler = None
    processed_parts = []

    if numerical_cols:
        scaler = StandardScaler()
        scaler.fit(covariates_df.loc[train_indices, numerical_cols])
        scaled_data = scaler.transform(covariates_df[numerical_cols])
        scaled_df = pd.DataFrame(scaled_data, index=covariates_df.index, columns=numerical_cols)
        processed_parts.append(scaled_df)

    if categorical_cols:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoder.fit(covariates_df.loc[train_indices, categorical_cols])
        encoded_data = encoder.transform(covariates_df[categorical_cols])
        encoded_df = pd.DataFrame(encoded_data, index=covariates_df.index, columns=encoder.get_feature_names_out())
        processed_parts.append(encoded_df)

    if processed_parts:
        final_covariates = pd.concat(processed_parts, axis=1)
        logger.info(f"Processed covariates shape: {final_covariates.shape}") # Use logger
        return final_covariates, encoder, scaler
    else:
        logger.warning("No covariates processed.") # Use logger
        return pd.DataFrame(index=metadata_df.index), None, None


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
