#!/bin/bash
#SBATCH --job-name=ml_baseline_1
#SBATCH --partition=notchpeak-shared-short
#SBATCH --qos=notchpeak-shared-short
#SBATCH --account=notchpeak-shared-short
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/ml_baseline_1_%j.out
#SBATCH --error=logs/ml_baseline_1_%j.err

cd $SLURM_SUBMIT_DIR
make run_ml_baseline_1