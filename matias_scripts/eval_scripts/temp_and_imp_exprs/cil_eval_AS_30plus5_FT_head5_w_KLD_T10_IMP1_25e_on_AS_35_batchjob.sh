#!/bin/bash
#SBATCH --job-name=cil_model_eval_AS30-5_ft_head_w_KLD_on_AS35
#SBATCH --account=project_462000765
#SBATCH --time=00:5:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --gpus-per-node=1
#SBATCH --partition=small-g
#SBATCH --output=outputs/eval_outputs/cil_eval_AS_30plus5_FT_head5_w_KLD_T10_IMP1_25e_on_AS_35_out.txt
#SBATCH --error=outputs/eval_outputs/cil_eval_AS_30plus5_FT_head5_w_KLD_T10_IMP1_25e_on_AS_35_err.txt

cd /pfs/lustrep2/projappl/project_462000765/matias/scripts
SCRIPT_NAME='cil_model_evaluation_script.py'

NR_OF_CLASSES=35
DATASET='audioset'
PATH_TO_DATA='/pfs/lustrep2/scratch/project_462000765/matias/data_cl_complete.hdf5'
NR_OF_WORKERS=6
BATCH_SIZE=32
LOG_INTERVAL=1
PATH_TO_MODEL_STATE='/pfs/lustrep2/scratch/project_462000765/matias/trained_models/cil_models/trained_cil_model_audioset_30plus5_FT_head_on_audioset_5_w_KLD_T10_IMP1_25epochs.pt'

# Load the modules
module use /appl/local/csc/modulefiles/
module load pytorch/2.7

srun python3 "$SCRIPT_NAME" \
--nr_of_classes "$NR_OF_CLASSES" \
--dataset "$DATASET" \
--path_to_data "$PATH_TO_DATA" \
--nr_of_workers "$NR_OF_WORKERS" \
--batch_size "$BATCH_SIZE" \
--log_interval "$LOG_INTERVAL" \
--path_to_model_state "$PATH_TO_MODEL_STATE" \
--use_amp
