#!/bin/bash -l
#SBATCH --job-name=TokenHSI-test
#SBATCH --output=output_slurm/log.txt
#SBATCH --error=output_slurm/error.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20g
#SBATCH --gres=gpu:1
#SBATCH --time=20:00
#SBATCH --account=shdpm0
#SBATCH --partition=spgpu
##### END preamble
##### Run in MotionBert dir

my_job_header

echo "=== NVIDIA SMI ==="
nvidia-smi


conda activate tokenhsi # need python 3.8, so you cant load python3.10-anaconda etc, or used the module load pytorch


echo "=== NVCC Version ==="
nvcc --version

echo "=== Python Version ==="
python --version

echo "=== PyTorch Version ==="
python -c "import torch; print('Torch version:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"


# module list

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/home/wenleyan/projects/isaacgym/python/isaacgym/_bindings/linux-x86_64:$LD_LIBRARY_PATH"

# > output_slurm/mem_usage.log # truncate the file
# while true; do
#     free -h >> output_slurm/mem_usage.log
#     sleep 60
# done &
# monitor_pid=$!

# > output_slurm/gpu_usage.log  # truncate or create a fresh log file
# while true; do
#     nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv >> output_slurm/gpu_usage.log
#     sleep 60
# done &
# gpu_monitor_pid=$!

# export MAX_JOBS=1

# sh tokenhsi/scripts/single_task/carry_test.sh
python ./tokenhsi/run.py --task HumanoidCarry \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task.yaml \
    --cfg_env tokenhsi/data/cfg/basic_interaction_skills/amp_humanoid_carry.yaml \
    --motion_file tokenhsi/data/dataset_carry/dataset_carry.yaml \
    --checkpoint output/single_task/ckpt_carry.pth \
    --output_path /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/carry1/ \
    --test \
    --headless \
    --record_headless \
    --num_envs 16



# python lpanlib/others/video.py --imgs_dir output/imgs/example_path --delete_imgs





# kill $monitor_pid
# kill $gpu_monitor_pid