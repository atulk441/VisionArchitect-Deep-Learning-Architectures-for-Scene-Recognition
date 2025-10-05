#!/bin/bash
#SBATCH --job-name=se_att
#SBATCH --output=se_att.out
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL    
#SBATCH --mail-user=atulkum@uwindsor.ca

VENV_PATH=/home/atulk441/scratch/VENV

module load python/3.11.5
module load cuda/12.6
module load StdEnv/2023
module load cudacore/.12.6.2
module load cudnn/9.2.1.18

source $VENV_PATH/bin/activate

python se_7.py