#!/bin/bash
#SBATCH --job-name=fsd50k_model_30_ft_head_eval_on_as
#SBATCH --account=project_462000765
#SBATCH --time=00:5:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --gpus-per-node=1
#SBATCH --partition=small-g
#SBATCH --output=outputs/eval_FSD50k_30_FT_head_30e_on_AS_out.txt
#SBATCH --error=outputs/eval_FSD50k_30_FT_head_30e_on_AS_err.txt

cd /pfs/lustrep2/projappl/project_462000765/matias/scripts
SCRIPT_NAME='model_evaluation_script.py'
MODEL='trained_model_fsd50k_30_FT_head_on_audioset_30epochs.pt'

NR_OF_CLASSES=30
DATASET='audioset'
PATH_TO_DATA='/pfs/lustrep2/scratch/project_462000765/matias/data_cl.hdf5'
NR_OF_WORKERS=6
BATCH_SIZE=32
LOG_INTERVAL=1
PATH_TO_MODEL_STATE="/pfs/lustrep2/scratch/project_462000765/matias/trained_models/${MODEL}"

# Load the modules
module use /appl/local/csc/modulefiles/
module load pytorch

srun python3 "$SCRIPT_NAME" --nr_of_classes "$NR_OF_CLASSES" \
--dataset "$DATASET" --path_to_data "$PATH_TO_DATA" --nr_of_workers "$NR_OF_WORKERS" \
--batch_size "$BATCH_SIZE" \
--log_interval "$LOG_INTERVAL" --path_to_model_state "$PATH_TO_MODEL_STATE" --use_amp
