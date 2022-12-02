#!/bin/bash -x
#SBATCH --account=cstdl
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --gres=gpu
#SBATCH --partition=batch
ml purge
ml use $OTHERSTAGES
ml Stages/2022
ml GCC/11.2.0
ml OpenMPI/4.1.2
ml CUDA/11.5
ml cuDNN/8.3.1.22-CUDA-11.5
ml NCCL/2.12.7-1-CUDA-11.5
ml PyTorch/1.11-CUDA-11.5
ml Horovod/0.24
ml torchvision/0.12.0
source envs/hdfml/bin/activate
export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "Job id: $SLURM_JOB_ID"
export TOKENIZERS_PARALLELISM=false
export NCCL_ASYNC_ERROR_HANDLING=1
srun python -u $*
