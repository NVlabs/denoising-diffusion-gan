#!/bin/bash -x
#SBATCH --account=zam
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpus
source set_torch_distributed_vars.sh
source scripts/init.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "Job id: $SLURM_JOB_ID"
export TOKENIZERS_PARALLELISM=false
srun python -u $*
