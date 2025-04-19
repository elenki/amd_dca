#!/bin/bash
#SBATCH --job-name=ml_baseline_2
#SBATCH --partition=notchpeak-shared-short
#SBATCH --qos=notchpeak-shared-short
#SBATCH --account=notchpeak-shared-short
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ml_baseline_2_%j.out
#SBATCH --error=logs/ml_baseline_2_%j.err

cd $SLURM_SUBMIT_DIR
make run_ml_baseline_2