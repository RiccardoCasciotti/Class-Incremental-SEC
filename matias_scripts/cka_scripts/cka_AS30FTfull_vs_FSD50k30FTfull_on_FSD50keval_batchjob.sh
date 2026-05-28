#!/bin/bash
#SBATCH --job-name=cka_comparison_as_30_FT_full_vs_fsd50k_30_ft_full_on_fsd50k
#SBATCH --account=project_462000765
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --gpus-per-node=1
#SBATCH --partition=small-g
#SBATCH --output=outputs/cka_AS30FTfull_vs_FSD50k30FTfull_on_FSD50keval_out.txt
#SBATCH --error=outputs/cka_AS30FTfull_vs_FSD50k30FTfull_on_FSD50keval_err.txt

cd /pfs/lustrep2/projappl/project_462000765/matias/scripts
SCRIPT_NAME='cka_script.py'
MODEL_1='trained_model_audioset_30_FT_full_on_fsd50k_60epochs.pt'
MODEL_2='trained_model_fsd50k_30_FT_full_on_audioset_60epochs.pt'

NR_OF_CLASSES=30
DATASET='fsd50k'
DATASET_SPLIT='eval'
PATH_TO_DATA='/pfs/lustrep2/scratch/project_462000765/matias/data_cl.hdf5'
NR_OF_WORKERS=6
BATCH_SIZE=32
PATH_TO_MODEL_STATE_1="/pfs/lustrep2/scratch/project_462000765/matias/trained_models/${MODEL_1}"
PATH_TO_MODEL_STATE_2="/pfs/lustrep2/scratch/project_462000765/matias/trained_models/${MODEL_2}"
MODEL_NAME_1='AS30FTfull'
MODEL_NAME_2='FSD50k30FTfull'
CKA_PLOT_SAVE_PATH='outputs/CKA_plots'

# Load the modules
module use /appl/local/csc/modulefiles/
module load pytorch

srun python3 "$SCRIPT_NAME"  --nr_of_classes "$NR_OF_CLASSES" \
--dataset "$DATASET" --dataset_split "$DATASET_SPLIT" --path_to_data "$PATH_TO_DATA" --nr_of_workers "$NR_OF_WORKERS" \
--batch_size "$BATCH_SIZE" --path_to_model_state_1 "$PATH_TO_MODEL_STATE_1" \
--path_to_model_state_2 "$PATH_TO_MODEL_STATE_2" --model_name_1 "$MODEL_NAME_1" \
--model_name_2 "$MODEL_NAME_2" --cka_plot_save_path "$CKA_PLOT_SAVE_PATH"
