#!/bin/bash
#SBATCH --job-name=as_model_30_ft_full_eval_on_fsd50k
#SBATCH --account=project_462000765
#SBATCH --time=00:5:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --gpus-per-node=1
#SBATCH --partition=small-g
#SBATCH --output=outputs/eval_AS_30_FT_full_60e_on_FSD50k_out.txt
#SBATCH --error=outputs/eval_AS_30_FT_full_60e_on_FSD50k_err.txt

cd /pfs/lustrep2/projappl/project_462000765/matias/scripts
SCRIPT_NAME='model_evaluation_script.py'
MODEL='trained_model_audioset_30_FT_full_on_fsd50k_60epochs.pt'

NR_OF_CLASSES=30
DATASET='fsd50k'
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
