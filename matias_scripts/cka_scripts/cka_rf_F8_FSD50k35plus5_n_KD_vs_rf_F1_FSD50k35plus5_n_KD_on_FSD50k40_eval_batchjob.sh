#!/bin/bash
#SBATCH --job-name=cka_comparison_rf_F8_FSD50k35plus5_n_KD_vs_rf_F1_FSD50k35plus5_n_KD_on_FSD50k40_eval
#SBATCH --account=project_462000765
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --gpus-per-node=1
#SBATCH --partition=small-g
#SBATCH --output=outputs/cka_outputs/cka_rf_F8_FSD50k35plus5_n_KD_vs_rf_F1_FSD50k35plus5_n_KD_on_FSD50k40_eval_out.txt
#SBATCH --error=outputs/cka_outputs/cka_rf_F8_FSD50k35plus5_n_KD_vs_rf_F1_FSD50k35plus5_n_KD_on_FSD50k40_eval_err.txt

cd /pfs/lustrep2/projappl/project_462000765/matias/scripts
SCRIPT_NAME='cka_script.py'
MODEL_1='cil_models/trained_rank_filt_F8_cil_model_rf_fsd50k_35plus5_FT_full_on_fsd50k_5_n_KLDorCOS_T10_IMP-23_no_posweight_100epochs.pt'
MODEL_2='cil_models/trained_rank_filt_F1_cil_model_rf_fsd50k_35plus5_FT_full_on_fsd50k_5_n_KLDorCOS_T10_IMP-23_no_posweight_100epochs.pt'

NR_OF_CLASSES=40
DATASET='fsd50k'
DATASET_SPLIT='eval'
PATH_TO_DATA='/pfs/lustrep2/scratch/project_462000765/matias/data_cl_complete.hdf5'
NR_OF_WORKERS=6
BATCH_SIZE=32
PATH_TO_MODEL_STATE_1="/pfs/lustrep2/scratch/project_462000765/matias/trained_models/${MODEL_1}"
PATH_TO_MODEL_STATE_2="/pfs/lustrep2/scratch/project_462000765/matias/trained_models/${MODEL_2}"
MODEL_NAME_1='rf_F8_FSD50k35plus5_n_KD'
MODEL_NAME_2='rf_F1_FSD50k35plus5_n_KD'
CKA_PLOT_SAVE_PATH='outputs/CKA_plots'

# Load the modules
module use /appl/local/csc/modulefiles/
module load pytorch

# MIOPEN needs some initialisation for the cache as the default location
# does not work on LUMI as Lustre does not provide the necessary features.
export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-$SLURM_NODEID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH

srun python3 "$SCRIPT_NAME"  \
--nr_of_classes "$NR_OF_CLASSES" \
--dataset "$DATASET" \
--dataset_split "$DATASET_SPLIT" \
--path_to_data "$PATH_TO_DATA" \
--nr_of_workers "$NR_OF_WORKERS" \
--batch_size "$BATCH_SIZE" \
--path_to_model_state_1 "$PATH_TO_MODEL_STATE_1" \
--path_to_model_state_2 "$PATH_TO_MODEL_STATE_2" \
--model_name_1 "$MODEL_NAME_1" \
--model_name_2 "$MODEL_NAME_2" \
--cka_plot_save_path "$CKA_PLOT_SAVE_PATH"
