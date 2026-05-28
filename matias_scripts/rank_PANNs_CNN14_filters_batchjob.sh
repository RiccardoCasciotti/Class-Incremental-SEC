#!/bin/bash
#SBATCH --job-name=rank_CNN14_filters
#SBATCH --account=project_462000765
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --gpus-per-node=1
#SBATCH --partition=small-g
#SBATCH --output=outputs/filter_ranking_outputs/PANNs_CNN14_filter_ranking_out.txt
#SBATCH --error=outputs/filter_ranking_outputs/PANNs_CNN14_filter_ranking_err.txt

cd /pfs/lustrep2/projappl/project_462000765/matias/scripts
SCRIPT_NAME='rank_PANNs_CNN14_filters.py'

NR_OF_CLASSES=30
PATH_TO_MODEL_STATE='/pfs/lustrep2/scratch/project_462000765/matias/trained_models/trained_model_audioset_30.pt'
model_name="$(basename $PATH_TO_MODEL_STATE .pt)"
DEST_OF_FILTER_SCORES="/pfs/lustrep2/projappl/project_462000765/matias/scripts/outputs/PANNs_CNN14_filter_scores_per_layer/${model_name}/"
mkdir -p "$DEST_OF_FILTER_SCORES"

# Load the modules
module use /appl/local/csc/modulefiles/
module load pytorch/2.7

# MIOPEN needs some initialisation for the cache as the default location
# does not work on LUMI as Lustre does not provide the necessary features.
export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-$SLURM_NODEID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH

srun python3 "$SCRIPT_NAME" \
--nr_of_classes "$NR_OF_CLASSES" \
--path_to_model_state "$PATH_TO_MODEL_STATE" \
--dest_of_filter_scores "$DEST_OF_FILTER_SCORES"
