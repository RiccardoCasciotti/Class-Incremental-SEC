#!/bin/bash
#SBATCH --job-name=rank_filt_cil_rf_AS45plus5_on_ASplus5_w_KLD
#SBATCH --account=project_462000765
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --gpus-per-node=1
#SBATCH --partition=small-g
#SBATCH --output=outputs/train_outputs/rank_filt_F8_cil_train_rf_AS45plus5_FT_full_w_KLDandCOS_T10_IMP-23_no_posweight_on_ASplus5_out.txt
#SBATCH --error=outputs/train_outputs/rank_filt_F8_cil_train_rf_AS45plus5_FT_full_w_KLDandCOS_T10_IMP-23_no_posweight_on_ASplus5_err.txt

cd /pfs/lustrep2/projappl/project_462000765/matias/scripts
# Default modified convolutional layers are 7-12
SCRIPT_NAME='rank_filt_cil_model_training_script.py'
MODEL='audioset'

EPOCHS=100
NR_OF_CLASSES=45
CIL_NR_OF_CLASSES=5
DATASET='audioset'
PATH_TO_DATA='/pfs/lustrep2/scratch/project_462000765/matias/data_cl_complete.hdf5'
NR_OF_WORKERS=6
BATCH_SIZE=32
CHECKPOINT_INTERVAL=10
LOG_INTERVAL=1
T=10
CLASS_IMPACT=-23
FILTER_NR=8
PATH_TO_MODEL_STATE="/pfs/lustrep2/scratch/project_462000765/matias/trained_models/cil_models/trained_rank_filt_F8_cil_model_rf_audioset_40plus5_FT_full_on_audioset_5_w_KLDandCOS_T10_IMP-23_no_posweight_100epochs.pt"
PATH_TO_COMPARISON_MODEL_STATE="/pfs/lustrep2/scratch/project_462000765/matias/trained_models/cil_models/trained_rank_filt_F8_cil_model_rf_audioset_40plus5_FT_full_on_audioset_5_w_KLDandCOS_T10_IMP-23_no_posweight_100epochs.pt"
filter_dir_name="$(basename $PATH_TO_MODEL_STATE .pt)"
PATH_TO_FILTER_SCORE_DIR="/pfs/lustrep2/projappl/project_462000765/matias/scripts/outputs/PANNs_CNN14_filter_scores_per_layer/${filter_dir_name}/"
MODEL_NAME="trained_rank_filt_F${FILTER_NR}_cil_model_rf_${MODEL}_${NR_OF_CLASSES}plus${CIL_NR_OF_CLASSES}_FT_full_on_${DATASET}_${CIL_NR_OF_CLASSES}_w_KLDandCOS_T${T}_IMP${CLASS_IMPACT}_no_posweight_${EPOCHS}epochs.pt"

# Load the modules
module use /appl/local/csc/modulefiles/
module load pytorch/2.7

# MIOPEN needs some initialisation for the cache as the default location
# does not work on LUMI as Lustre does not provide the necessary features.
export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-$SLURM_NODEID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH

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
--filter_nr "$FILTER_NR" \
--path_to_filter_score_dir "$PATH_TO_FILTER_SCORE_DIR" \
--use_amp \
--use_kld \
--save_latest_epoch_model \
--validate_w_map \
--use_cosine_kd \
--no_pos_weight \
--weight_decay 0
