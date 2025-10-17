#!/bin/bash
#SBATCH --job-name=as_train_file_extraction
#SBATCH --account=project_462000765
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=small
#SBATCH --output=out.txt
#SBATCH --error=err.txt

cd /pfs/lustrep2/projappl/project_462000765/matias/scripts
SCRIPT_NAME='extract_and_process_valid_audiofiles.py'

BLOB_DIR='/pfs/lustrep2/scratch/project_462000765/manjunath/Audioset/hub/datasets--confit--audioset-full/blobs'
VALID_FNAMES='/pfs/lustrep2/scratch/project_462000765/matias/audioset_filename_lists/audioset_train_filenames_top50.txt'
HDF5_FILEPATH='/pfs/lustrep2/scratch/project_462000765/matias/cl_data.hdf5'
HDF5_GROUP='audioset_strong_train_50'
PCKL_DICT_PATH='/pfs/lustrep2/projappl/project_462000765/matias/scripts/pickle_objects/audioset_train_labels_50.pckl'

# Load the modules
module use /appl/local/csc/modulefiles/
module load pytorch

srun python3 "$SCRIPT_NAME" --blob_dir "$BLOB_DIR" \
 --valid_filenames "$VALID_FNAMES" --hdf5_filepath "$HDF5_FILEPATH" \
--hdf5_group "$HDF5_GROUP" --pckl_label_dict "$PCKL_DICT_PATH"
