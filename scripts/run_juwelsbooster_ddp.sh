#!/bin/bash -x
#SBATCH --account=covidnetx
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:4
#SBATCH --partition=booster
source set_torch_distributed_vars.sh
#source scripts/init_2022.sh
#source scripts/init_2020.sh
source scripts/init.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "Job id: $SLURM_JOB_ID"
export TOKENIZERS_PARALLELISM=false
export NCCL_ASYNC_ERROR_HANDLING=1
srun python -u $*
