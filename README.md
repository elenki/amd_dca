# Deep Count Autoencoder for Bulk RNA-Seq Denoising in AMD Studies


## Project Overview

This project implements and evaluates a **deep count autoencoder**, a specialized neural network architecture, designed to denoise bulk RNA-sequencing (RNA-seq) data from human retina samples in the context of Age-related Macular Degeneration (AMD). The primary objective is to improve the separation of subtle biological signals related to AMD status and progression from confounding technical noise inherent in high-throughput sequencing experiments. By leveraging a model tailored for the statistical properties of count data (using Negative Binomial or Zero-Inflated Negative Binomial distributions), this work aims to provide a cleaner expression matrix, thereby enhancing the potential for discovering robust AMD biomarkers and understanding disease mechanisms compared to analyses using raw or conventionally corrected data.

## Background & Motivation

Age-related Macular Degeneration (AMD) is a leading cause of irreversible vision loss, particularly in older populations. Understanding its molecular pathogenesis is crucial for developing effective diagnostics and therapies. Retinal transcriptomics via RNA-seq provides valuable data, but its interpretation is often complicated by technical artifacts (e.g., batch effects, sample quality variations (RIN, PMI), library preparation and sequencing depth differences). These noise sources can introduce substantial variance, potentially masking the subtle biological expression changes associated with disease progression across different AMD stages (e.g., MGS1 through MGS4).

While standard computational methods like ComBat or Limma's `removeBatchEffect` can address linear batch effects, they may be less effective against complex, non-linear noise structures often present in biological data and may risk removing genuine biological signal alongside noise. This project explores a deep learning approach, adapting denoising autoencoder techniques from single-cell RNA-seq (e.g., Eraslan et al., 2019) to the specific challenges of bulk retinal RNA-seq data.

## Objectives

* Design and implement a deep autoencoder architecture optimized for denoising bulk RNA-seq count data, incorporating relevant covariates.
* Employ statistically appropriate count-based loss functions (Negative Binomial - NB, or Zero-Inflated Negative Binomial - ZINB) that model data overdispersion.
* Train and validate the model using the publicly available EyeGEx dataset (GSE115828), focusing on the provided RSEM expected counts.
* Systematically evaluate the autoencoder's denoising performance by comparing its output to the original raw data and data corrected using a conventional method (e.g., ComBat).
* Assess whether denoising via the autoencoder improves biological interpretability, specifically focusing on dimensionality reduction results and the outcomes of downstream differential gene expression (DGE) analysis between AMD stages.

## Methodology

The workflow utilizes a deep count autoencoder implemented in Python using **PyTorch**. Key components include:

1.  **Data Acquisition & Preprocessing:**
    * Input Data: RSEM expected gene counts (`GSE115828_RSEM_gene_counts.tsv.gz`) and associated sample metadata from the GSE115828 dataset.
    * ID Mapping: Resolve differences between sample identifiers in the count matrix (e.g., `R42015-490pf_100-IR_L2`) and the metadata (e.g., `r_id` like `100_2` or `GSM_ID`).
    * Filtering: Remove low-quality samples (e.g., based on RIN threshold) and low-expression genes (e.g., based on minimum counts in a minimum percentage of samples).
    * Covariate Preparation: Select, encode (e.g., one-hot for `sex`, `batch`), scale (e.g., standard scaling for `age`, `RIN`, `PMI`), and handle missing values for relevant covariates identified in the metadata. Fit scalers/encoders only on the training set.
    * Data Splitting: Partition samples into training, validation, and test sets, stratified by `MGS_level` (AMD stage) where possible.
    * Model Input Preparation: Create the input matrix (`X`) for the network by applying a `log1p` transformation to the filtered counts (for numerical stability) and concatenating the prepared covariates. The target (`Y`) for the loss function remains the raw, filtered counts.

2.  **Model Architecture (Deep Count Autoencoder):**
    * **Encoder:** Multi-layer fully connected network that compresses the high-dimensional input (`X`) into a lower-dimensional latent representation (bottleneck layer). Uses non-linear activations (e.g., ReLU). Aims to capture biological variance while factoring out noise informed by covariates.
    * **Bottleneck:** Dense layer representing the compressed biological state (dimensionality is a hyperparameter).
    * **Decoder:** Multi-layer fully connected network that reconstructs the expression profile from the bottleneck representation. Uses non-linear activations.
    * **Output Heads:** Final linear layer(s) predicting the parameters of the chosen count distribution (NB determined via EDA) for each gene. Uses activations ensuring parameter constraints (e.g., `softplus` or `exp` for positivity of NB mean `μ` and dispersion `θ`).

3.  **Training:**
    * Optimize model weights by minimizing the Negative Binomial negative log-likelihood loss between the raw target counts (`Y`) and the distribution defined by the predicted parameters (`μ`, `θ`).
    * Utilize the Adam/AdamW optimizer, train in batches using PyTorch `DataLoader`.
    * Monitor validation loss for hyperparameter tuning, early stopping, and model checkpointing (saving the best performing model).

4.  **Evaluation:** Assess denoising efficacy on the held-out test set:
    * **Denoised Data:** Generate the denoised matrix using the predicted mean (`μ`) from the trained model.
    * **Visualization:** Compare PCA/UMAP projections of raw, ComBat-corrected (if generated), and autoencoder-denoised data. Color points by biological factors (`MGS_level`) to assess separation and by technical factors (`batch`, `RIN`, `PMI`) to assess noise reduction (expect less clustering by technical factors after denoising).
    * **Quantitative Metrics (TBD - Optional):** Evaluate cluster separation (e.g., silhouette scores) or perform variance component analysis.
    * **Downstream Analysis:** Conduct DGE analysis (e.g., MGS4 vs MGS1) on raw, ComBat-corrected, and denoised count matrices. Compare the resulting DEG lists in terms of number, statistical significance, overlap, and biological pathway enrichment. The hypothesis is that denoising enhances the detection of relevant biological pathways.

## Expected Outcomes

* A trained PyTorch deep count autoencoder model optimized for the GSE115828 retinal RNA-seq dataset.
* A comparative analysis (quantitative and qualitative) of the autoencoder's denoising capabilities versus raw data and standard batch correction (ComBat).
* The generated denoised expression matrix for the test dataset samples.
* An analysis report summarizing the findings, particularly regarding the impact of denoising on downstream biological interpretation (dimensionality reduction, DGE, pathway analysis).
* A reusable, modular Python codebase implementing the described pipeline.

## Repository Structure
```
├── .gitignore           # Specifies intentionally untracked files that Git should ignore.
├── README.md            # This file: Project description, setup, usage.
├── config.yaml          # Configuration file (paths, hyperparameters, dataset info).
├── environment.yaml     # Conda environment definition for dependencies.
├── notebooks/           # Jupyter notebooks for exploratory data analysis, prototyping.
├── data/                # Data storage (raw, processed - large files may be gitignored).
├── src/                 # Source code for the project, organized into modules.
│   ├── data/            # Data loading, preprocessing, splitting functions.
│   ├── model/           # Autoencoder model definition, custom loss functions.
│   ├── training/        # Training loops, callbacks, model saving logic.
│   ├── evaluation/      # Evaluation metrics, plotting functions, DGE comparison logic.
│   └── utils/           # Common utility functions (config loading, plotting helpers).
├── scripts/             # Executable scripts to run main workflow stages (preprocess, train, eval).
└── logs/                # Directory for storing log files from runs.
```
## Setup

1.  **Clone:** `git clone <repository-url>`
2.  **Environment:** Create and activate the Conda environment:
    ```bash
    conda env create -f environment.yaml
    conda activate amd_denoiser # Or the name specified in environment.yaml
    ```
3.  **Data:** Download required raw data files (e.g., RSEM count matrix `GSE115828_RSEM_gene_counts.tsv.gz` and associated metadata) from the sources listed below. Place them in the directory specified by `raw_data_dir` in `config.yaml` (default: `data/raw/`).
4.  **Configuration:** Review and modify `config.yaml` to ensure paths are correct for your system and adjust parameters as needed.

## Usage

Execute the main stages of the pipeline using the scripts provided in the `scripts/` directory. Ensure the Conda environment is activated.

```bash
# 1. Preprocess the data (generates files in data/processed/)
python scripts/run_preprocessing.py

# 2. Train the autoencoder model (saves model to results/models/)
python scripts/run_training.py

# 3. Evaluate the trained model (generates plots/results in results/)
python scripts/run_evaluation.py
(More detailed command-line arguments or options may be added later).
```

Data Sources:

* GSE115828: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE115828 (Primary dataset: ~453-523 retinal samples)

References:
* Ratnapriya, R., Sosina, O.A., Starostik, M.R. et al. Retinal transcriptome and eQTL analyses identify genes associated with age-related macular degeneration. Nat Genet 51, 606–610 (2019). https://doi.org/10.1038/s41588-019-0351-9

* Eraslan G, Simon LM, Mircea M, Mueller NS, Theis FJ. Single-cell RNA-seq denoising using a deep count autoencoder. Nat Commun. 2019 Jan 23;10(1):390. https://doi.org/10.1038/s41467-018-07931-2. PMID: 30674886; PMCID: PMC6344535.