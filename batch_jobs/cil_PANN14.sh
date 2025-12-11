#!/bin/bash
#SBATCH --job-name=rank_filt_cil
#SBATCH --account=project_462000765
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --gpus-per-node=1
#SBATCH --partition=standard-g
#SBATCH --output=out.txt
#SBATCH --error=err.txt




# Load the modules
module use /appl/local/csc/modulefiles/
module load pytorch/2.7





cd /projappl/project_462000765/casciott/continual_learning && srun python3 rank_cil.py --config $1 --model_name $2
