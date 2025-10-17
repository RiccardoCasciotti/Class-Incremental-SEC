#!/bin/bash
#SBATCH --job-name=cnn14_FSD50k50_training
#SBATCH --account=project_462000765
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --gpus-per-node=1
#SBATCH --partition=small-g
#SBATCH --output=outputs/train_outputs/fsd50k_50_out.txt
#SBATCH --error=outputs/train_outputs/fsd50k_50_err.txt

cd /pfs/lustrep2/projappl/project_462000765/matias/scripts
SCRIPT_NAME='model_training_script.py'

EPOCHS=100
NR_OF_CLASSES=50
DATASET='fsd50k'
PATH_TO_DATA='/pfs/lustrep2/scratch/project_462000765/matias/data_cl_complete.hdf5'
NR_OF_WORKERS=6
BATCH_SIZE=32
CHECKPOINT_INTERVAL=10
LOG_INTERVAL=10
PATH_TO_MODEL_STATE='/pfs/lustrep2/scratch/project_462000765/matias/blank_models/blank_cnn14_50.pt'

# MIOPEN needs some initialisation for the cache as the default location
# does not work on LUMI as Lustre does not provide the necessary features.
export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-$SLURM_NODEID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH

# Load the modules
module use /appl/local/csc/modulefiles/
module load pytorch/2.7

srun python3 "$SCRIPT_NAME" --epochs "$EPOCHS" \
--nr_of_classes "$NR_OF_CLASSES" \
--dataset "$DATASET" \
--path_to_data "$PATH_TO_DATA" \
--nr_of_workers "$NR_OF_WORKERS" \
--batch_size "$BATCH_SIZE" \
--checkpoint_interval "$CHECKPOINT_INTERVAL" \
--log_interval "$LOG_INTERVAL" \
--path_to_model_state "$PATH_TO_MODEL_STATE" \
--use_amp
