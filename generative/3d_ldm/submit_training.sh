#!/bin/bash
#SBATCH --partition=hai-lo
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=80
#SBATCH --mem=500G
#SBATCH --gres=gpu:1
#SBATCH --job-name="cycle_classifier_guided"
#SBATCH --output=/hai/scratch/fangruih/monai-tutorials/generative/3d_ldm/slurm_outputs/cycle_train_%j.out
#SBATCH --mail-user=fangruih@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --account=mind

# Accept command-line argument for config file path
CONFIG_FILE=$1

# Print some information about the job
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"
echo "Config file: $CONFIG_FILE"

# Load bash configurations and initialize conda
source ~/.bashrc

# Activate your Monai conda environment
conda activate monai

# Change to the directory containing your script
cd /hai/scratch/fangruih/monai-tutorials/generative/3d_ldm

# Create outputs directory if it doesn't exist
mkdir -p slurm_outputs

# Run the Python script
srun python train_timestep_cycle_classifier_guided.py \
    -c $CONFIG_FILE \
    -e ./config/environment_config/environment_t1_all_cycle.json \
    -g 1

# Run the Python script
# srun torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_port=29500 \
#     train_timestep_cycle_classifier_guided.py \
#     -c $CONFIG_FILE \
#     -e ./config/environment_config/environment_t1_all_cycle.json \
#     -g 4

# Print completion message
echo "Job finished at: $(date)"

# Deactivate the conda environment
conda deactivate