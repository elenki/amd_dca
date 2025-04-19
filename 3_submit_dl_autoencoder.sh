#!/bin/bash
#SBATCH --job-name=ae_train
#SBATCH --partition=notchpeak-shared-freecycle
#SBATCH --qos=notchpeak-freecycle
#SBATCH --account=bmi6021
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=1            # single‐task array (could expand for e.g. multiple seeds)
#SBATCH --output=logs/ae_train_%A_%a.out
#SBATCH --error=logs/ae_train_%A_%a.err
#SBATCH --requeue

echo "Running on host $(hostname)"
echo "Working directory: $SLURM_SUBMIT_DIR"
cd $SLURM_SUBMIT_DIR

# (Optional) if you want to vary seed per array index:
# export SEED=$(( 42 + SLURM_ARRAY_TASK_ID ))
# python -m amd_dca.cli train_ae --seed $SEED

python -m amd_dca.cli train_ae