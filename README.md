# Comparative Evaluation of Denoising Methods for Bulk Retinal RNA-Seq in AMD Studies

## Project Overview

This project implements and evaluates various computational methods for denoising bulk RNA-sequencing (RNA-seq) data from human retina samples in the context of Age-related Macular Degeneration (AMD). The primary objective is to compare the effectiveness of deep learning models (Negative Binomial Autoencoder - NB-AE, Variational Autoencoder - NB-VAE) against standard batch correction (ComBat) and machine learning baselines (PCA+kNN, Random Forest) in improving the separation of biological signals (AMD stage) from technical noise. The evaluation focuses on downstream differential gene expression (DGE) analysis and dimensionality reduction visualizations.

## Background & Motivation

Age-related Macular Degeneration (AMD) is a leading cause of irreversible vision loss. Understanding its molecular pathogenesis through transcriptomics is crucial but often hampered by technical noise inherent in RNA-seq experiments (e.g., batch effects, sample quality variations like RIN/PMI, sequencing depth). This noise can obscure subtle biological signals related to disease progression across different AMD stages (Macular Degeneration Severity Scale - MGS levels 1-4).

While standard methods like ComBat address linear batch effects, they might be less effective against complex noise or risk removing biological signals. Deep learning approaches, particularly autoencoders adapted for count data, offer a potential alternative for learning complex data structures and disentangling biological variation from noise, inspired by successful applications in single-cell transcriptomics. This project performs a comparative assessment of several such methods on bulk retinal RNA-seq data.

## Methodology

The project employs a systematic workflow comparing multiple denoising approaches implemented in Python using PyTorch, Scikit-learn, and R (via `rpy2`).

### 1. Data Acquisition & Preprocessing

* **Input Data:**
    * Raw integer gene counts (`GSE115828_raw_counts_GRCh38.p13_NCBI.tsv`) from the NCBI GEO dataset GSE115828 (genes x samples format).
    * Sample metadata (`GSE115828_metadata.csv`) containing clinical information (age, sex, MGS level, RIN, PMI, etc.) and technical details (batch).
    * Sample mapping file (`GSE115828_sample_mapping.csv`) linking GSM Accession IDs (from counts) to sample Titles (used for linking to metadata).
* **ID Mapping:** A multi-step process links samples across the three files: Counts(GSM) -> Mapping(GSM->Title) -> Metadata(Title->linking\_id<-r\_id).
* **Quality Control:** Samples are filtered based on RNA Integrity Number (RIN ≥ 6.0). Genes are filtered based on minimum expression levels (≥ 10 counts in ≥ 10% of samples).
* **Exploratory Data Analysis (EDA):** Performed in notebooks (`notebooks/01_*.ipynb`, `notebooks/02_*.ipynb`) to assess data quality, library sizes, gene detection rates, mean-variance relationships, and justify the use of the Negative Binomial distribution for count modeling.
* **Covariate Preparation:** Relevant covariates (`age`, `sex`, `rin`, `postmortem_interval_hrs`, `lib_prep_batch`, `library_prepper`) are selected, imputed (using training set mean/mode), scaled (StandardScaler for numerical), and one-hot encoded (for categorical). Scalers/encoders are fit *only* on the training set and saved.
* **Data Splitting:** Samples are deterministically split into training (80%), validation (10%), and test (10%) sets, stratified by `MGS_level`. Split manifest and sample IDs are saved.
* **Output Generation:** The preprocessing script (`scripts/run_preprocessing.py`) generates and saves multiple data formats required by downstream models in `data/processed/`:
    * Filtered raw integer counts (Y - split into train/val/test).
    * `log1p`-transformed counts (X_log - split).
    * Processed covariates (Cov - split).
    * Combined DL input matrices (`X = X_log + Cov` - split).
    * ComBat-corrected integer counts (split).
    * Final gene list, sample ID lists (train/val/test).
    * Test set metadata.
    * Fitted scalers/encoders.

### 2. Denoising Models Implemented

* **Raw Data:** The filtered, raw integer counts serve as the baseline reference.
* **ComBat:** Standard batch correction applied using the `lib_prep_batch` variable via `rpy2` and the `sva` R package.
* **PCA + kNN (Baseline 1):** Log1p counts are projected onto principal components (PCs). k-Nearest Neighbors regression is performed in PC space to predict denoised log1p counts, which are then back-transformed. Implemented in `scripts/run_ml_baseline_1.py`.
* **Random Forest (Baseline 2):** A multi-output Random Forest regressor is trained on combined input features (log1p counts + covariates) to predict raw integer counts. Implemented in `scripts/run_ml_baseline_2.py`.
* **Negative Binomial Autoencoder (NB-AE):** A deep autoencoder implemented in PyTorch (`src/amd_dca/model/autoencoder.py`) taking combined features (log1p counts + covariates) as input. The decoder predicts parameters (mean $\mu$, dispersion $\theta$) of a Negative Binomial distribution, trained by minimizing the NB negative log-likelihood against the raw integer counts. Trained via `scripts/run_dl_autoencoder.py` (using HPO results) and inference via `scripts/run_infer_ae.py`.
* **Negative Binomial Variational Autoencoder (NB-VAE):** A VAE (`src/amd_dca/model/vae.py`) with a similar input/output structure to the NB-AE but incorporating a probabilistic latent space (learning mean $\mu_z$ and variance $\sigma_z^2$). Trained by minimizing a combination of the NB reconstruction loss and the KL divergence between the learned latent distribution and a prior (standard Gaussian). Trained via `scripts/run_dl_vae.py` (using HPO results) and inference via `scripts/run_infer_vae.py`.
* **Hyperparameter Optimization (HPO):** Optuna-based randomized search is implemented (`scripts/run_hpo_ae.py`, `scripts/run_hpo_vae.py`) to tune key hyperparameters for AE and VAE models based on validation set performance.

### 3. Evaluation Strategy

* **Dimensionality Reduction:** PCA and UMAP are applied to the test set counts from all methods (Raw, ComBat, KNN, RF, AE, VAE). Embeddings are visualized and colored by `MGS_level` and technical factors (e.g., batch) to assess separation of biological signal and removal of technical noise.
* **Differential Gene Expression (DGE):** DESeq2 is used via `rpy2` (`src/amd_dca/r/dge.py`) to perform DGE analysis comparing advanced AMD (MGS4) vs. early AMD (MGS1) on the test set counts produced by *each* method (Raw, ComBat, KNN-denoised, RF-denoised, AE-denoised, VAE-denoised). This is run via `scripts/run_dge.py`.
* **Comparative Analysis:** The number of DEGs, their overlap (Venn diagrams), significance levels, and potentially pathway enrichment results are compared across methods using `scripts/run_evaluation.py` and `src/amd_dca/evaluation/`. Quantitative metrics like Silhouette scores on embeddings may also be calculated.

## Repository Structure

```

amd_dca/
├── README.md                     # This file
├── config.yaml                   # Configuration (paths, params)
├── environment.yaml              # Conda environment definition
├── Makefile                      # Workflow execution commands
├── pyproject.toml                # Project metadata (for packaging)
├── \*.sh                         # SLURM submission scripts (optional)
├── data/
│   ├── raw/                      # Raw input data files (needs download)
│   └── processed/                # Generated data matrices, splits, etc.
├── notebooks/
│   ├── 01_eda_metadata.ipynb              # EDA on metadata
│   ├── 02_eda_counts_preprocessing.ipynb  # EDA on counts & justification
│   └── eda_plots/                         # Plots generated by notebooks
├── results/
│   ├── dge/                      # DGE result tables and plots
│   ├── evaluation/               # Comparative evaluation plots (PCA/UMAP)
│   ├── hpo/                      # Hyperparameter optimization results
│   └── models/                   # Saved trained model checkpoints (AE/VAE)
├── logs/                         # Log files from script executions
└── src/
    └── amd_dca/                     # Source code package

    ├── cli.py                        # Command line interface dispatcher
    ├── baseline/                     # Baseline ML model implementations
    │   └── pca_knn.py                # PCA+KNN logic (RF in script)
    ├── data/                         # Data loading and preprocessing
        └── preprocess.py
    ├── evaluation/                   # Evaluation metrics and plotting
        ├── evaluate.py               # Main evaluation logic (plots)
        └── metrics.py                # DGE metrics (volcano, venn)
    ├── model/                        # Model definitions
    │   ├── autoencoder.py            # NB-AE definition
    │   ├── losses.py                 # NB/ZINB/VAE loss functions
    │   └── vae.py                    # NB-VAE definition
    ├── r/                            # R integration via rpy2
        ├── combat.py                 # ComBat wrapper
    │   └── dge.py                    # DESeq2 wrapper
    ├── scripts/                      # Main executable scripts for pipeline stages
    │   ├── run_preprocessing.py
    │   ├── run_ml_baseline_1.py      # PCA+KNN
    │   ├── run_ml_baseline_2.py      # RF
    │   ├── run_hpo_ae.py             # HPO for AE
    │   ├── run_hpo_vae.py            # HPO for VAE
    │   ├── run_dl_autoencoder.py     # Train AE with best HPO params
    │   ├── run_dl_vae.py             # Train VAE with best HPO params
    │   ├── run_infer_ae.py           # Generate denoised data from AE
    │   ├── run_infer_vae.py          # Generate denoised data from VAE
    │   ├── run_dge.py                # Run DGE analysis on specified input
    │   └── run_evaluation.py         # Generate final comparative plots
    ├── training/                     # Training loops
       ├── train.py                  # Training loop for AE
       └── train_vae.py              # Training loop for VAE
    └── utils/                        # Utility functions
        ├── helpers.py                # Config loading, seeding, path finding
        └── plotting.py               # Plotting helper functions

````

## Setup

1.  **Clone Repository:**
    ```bash
    git clone <repository-url>
    cd amd_dca
    ```

2.  **Create Conda Environment:**
    ```bash
    conda env create -f environment.yaml
    conda activate amd_denoiser # Or the name specified in environment.yaml
    ```
    *(Optional but Recommended)* Install the project package in editable mode:
    ```bash
    pip install -e .
    ```

3.  **Download Data:**
    * Obtain the following files from GSE115828 or other sources:
        * `GSE115828_raw_counts_GRCh38.p13_NCBI.tsv` (Raw Counts)
        * `GSE115828_metadata.csv` (Metadata)
        * `GSE115828_sample_mapping.csv` (Sample Mapping)
        * `Human.GRCh38.p13.annot.tsv` (Gene Annotation - for volcano plots)
    * Place these files inside the `data/raw/` directory.

4.  **Configure:**
    * Review `config.yaml`. Ensure file names under `datasets:gse115828` match the downloaded files.
    * Adjust preprocessing parameters, model architectures, or training hyperparameters as needed. Ensure the `preprocessing:batch_column_for_combat` and `dge:contrast_variable` parameters point to the correct columns in your metadata.

## Usage Workflow

The main pipeline stages are executed using `make` commands, which call the underlying Python scripts via `src/amd_dca/cli.py`. Ensure the `amd_denoiser` conda environment is activated.

1.  **Preprocess Data:** Generates all necessary processed files in `data/processed/`.
    ```bash
    make run_preprocessing
    ```

2.  **Run Baseline Models:** These train and save denoised test set counts.
    ```bash
    make run_ml_baseline_1 # Runs PCA+KNN
    make run_ml_baseline_2 # Runs RandomForest
    ```

3.  **Run Deep Learning Models (Optional HPO first):**
    * *(Optional)* Run hyperparameter optimization (may take significant time):
        ```bash
        # python -m amd_dca.cli train_ae # using Optuna HPO defined in the script
        # python -m amd_dca.cli train_vae # using Optuna HPO defined in the script
        # Note: HPO scripts save best params to results/hpo/. Update config.yaml manually if needed.
        ```
    * Train AE and VAE using parameters defined in `config.yaml`:
        ```bash
        make run_dl_autoencoder # Trains AE
        make run_dl_vae        # Trains VAE
        ```
    * Generate denoised test matrices from trained DL models:
        ```bash
        make infer_ae
        make infer_vae
        ```

4.  **Run DGE Analysis:** Execute DESeq2 for each count type.
    *Requires specifying the input type.*
    ```bash
    make run_dge input=raw
    make run_dge input=combat
    make run_dge input=knn
    make run_dge input=rf
    make run_dge input=nb_ae
    make run_dge input=nb_vae
    ```
    *(Results are saved in `results/dge/`)*

5.  **Run Final Evaluation:** Generate comparative PCA/UMAP plots and potentially aggregate DGE results. *Note: The current `run_evaluation.py` primarily generates PCA/UMAP for one specified input vs raw; it may need expansion for full DGE comparison visualization.*
    ```bash
    # Example: Evaluate NB-AE results vs Raw
    make run_evaluation input=nb_ae
    # Repeat for other inputs as needed: input=combat, input=knn, etc.
    ```
    *(Results are saved in `results/evaluation/`)*

### HPC / SLURM Usage (Optional)

SLURM submission scripts (`*.sh`) are provided for running the baseline and DL training steps on a high-performance computing cluster. Modify account/partition details within the scripts as needed. Submit using `sbatch`:
```bash
sbatch 1_submit_ml_baseline_1.sh
sbatch 2_submit_ml_baseline_2.sh
sbatch 3_submit_dl_autoencoder.sh # Check config before running
sbatch 4_submit_dl_vae.sh        # Check config before running
````

## References

  * [1] Eraslan, G., Simon, L.M., Mircea, M. et al. Single-cell RNA-seq denoising using a deep count autoencoder. *Nat Commun* 10, 390 (2019). https://doi.org/10.1038/s41467-018-07931-2 [cite: 17]
  * [2] Ratnapriya R, Sosina OA, Starostik MR, et al. Retinal transcriptome and eQTL analyses identify genes associated with age-related macular degeneration. *Nat Genet*. 2019;51(4):606-610. doi:10.1038/s41588-019-0351-9 [cite: 21, 22]
  * [3] Lähnemann, D., Köster, J., Szczurek, E. et al. Eleven grand challenges in single-cell data science. *Genome Biol* 21, 31 (2020). https://doi.org/10.1186/s13059-020-1926-6 [cite: 18]
  * [4] Wang JH, Wong RCB, Liu GS. Retinal aging transcriptome and cellular landscape in association with the progression of age-related macular degeneration. *Invest Ophthalmol Vis Sci.* 2023;64(4):32. https://www.google.com/search?q=https://doi.org/10.1167/iovs.64.4.32 [cite: 19, 20]
  * [5] Fritsche LG, et al. Age-related macular degeneration: genetics and biology coming together. *Annu Rev Genomics Hum Genet* 15, 151–171 (2014). [PubMed: 24773320] [cite: 15, 16]
