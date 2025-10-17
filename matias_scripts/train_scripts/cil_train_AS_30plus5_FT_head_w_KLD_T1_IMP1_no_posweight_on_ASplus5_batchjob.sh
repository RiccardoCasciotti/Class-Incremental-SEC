#!/bin/bash
#SBATCH --job-name=cil_AS30plus5_on_ASplus5_w_KLD_no_posweight
#SBATCH --account=project_462000765
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --gpus-per-node=1
#SBATCH --partition=small-g
#SBATCH --output=outputs/train_outputs/cil_train_AS30plus5_FT_head_w_KLD_T1_IMP1_no_posweight_on_ASplus5_out.txt
#SBATCH --error=outputs/train_outputs/cil_train_AS30plus5_FT_head_w_KLD_T1_IMP1_no_posweight_on_ASplus5_err.txt

cd /pfs/lustrep2/projappl/project_462000765/matias/scripts
SCRIPT_NAME='cil_model_training_script.py'
MODEL='audioset'

EPOCHS=50
NR_OF_CLASSES=30
CIL_NR_OF_CLASSES=5
DATASET='audioset'
PATH_TO_DATA='/pfs/lustrep2/scratch/project_462000765/matias/data_cl_complete.hdf5'
NR_OF_WORKERS=6
BATCH_SIZE=32
CHECKPOINT_INTERVAL=10
LOG_INTERVAL=1
T=1
CLASS_IMPACT=1
PATH_TO_MODEL_STATE="/pfs/lustrep2/scratch/project_462000765/matias/trained_models/trained_model_${MODEL}_${NR_OF_CLASSES}.pt"
PATH_TO_COMPARISON_MODEL_STATE="/pfs/lustrep2/scratch/project_462000765/matias/trained_models/trained_model_${MODEL}_${NR_OF_CLASSES}.pt"
MODEL_NAME="trained_cil_model_${MODEL}_${NR_OF_CLASSES}plus${CIL_NR_OF_CLASSES}_FT_head_on_${DATASET}_${CIL_NR_OF_CLASSES}_w_KLD_T${T}_IMP${CLASS_IMPACT}_no_posweight_${EPOCHS}epochs.pt"

# Load the modules
module use /appl/local/csc/modulefiles/
module load pytorch/2.7

srun python3 "$SCRIPT_NAME" \
--epochs "$EPOCHS" \
--nr_of_classes "$NR_OF_CLASSES" \
--cil_nr_of_classes "$CIL_NR_OF_CLASSES" \
--dataset "$DATASET" \
--path_to_data "$PATH_TO_DATA" \
--nr_of_workers "$NR_OF_WORKERS" \
--batch_size "$BATCH_SIZE" \
--checkpoint_interval "$CHECKPOINT_INTERVAL" \
--log_interval "$LOG_INTERVAL" \
--path_to_model_state "$PATH_TO_MODEL_STATE" \
--path_to_comparison_model_state "$PATH_TO_COMPARISON_MODEL_STATE" \
--model_name "$MODEL_NAME" \
--T "$T" \
--class_impact "$CLASS_IMPACT" \
--finetune_classifier \
--use_amp \
--use_kld \
--validate_w_map \
--save_latest_epoch_model \
--no_pos_weight
