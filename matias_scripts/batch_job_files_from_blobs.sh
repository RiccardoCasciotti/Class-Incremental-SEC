#!/bin/bash
#SBATCH --job-name=as_train_file_extraction
#SBATCH --account=project_462000765
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=small
#SBATCH --out=out.txt
#SBATCH --error=err.txt

SCRIPT_LOC='/pfs/lustrep2/projappl/project_462000765/matias/scripts/extract_valid_AS_files/extract_valid_audiofiles.py'

BLOB_DIR='/pfs/lustrep2/scratch/project_462000765/manjunath/Audioset/hub/datasets--confit--audioset-full/blobs'
VALID_FNAMES='/pfs/lustrep2/scratch/project_462000765/matias/audioset_filename_lists/audioset_train_filenames_top50.txt'
TARGET_DIR='/pfs/lustrep2/scratch/project_462000765/matias/audioset_strong_train'

# Load the modules
module use /appl/local/csc/modulefiles/
module load pytorch

srun python3 extract_valid_audiofiles.py --blob_dir "$BLOB_DIR" --valid_filenames "$VALID_FNAMES" --target_dir "$TARGET_DIR" 
