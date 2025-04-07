# conda activate tokenhsi # need python 3.8, so you cant load python3.10-anaconda etc, or used the module load pytorch

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/home/wenleyan/projects/isaacgym/python/isaacgym/_bindings/linux-x86_64:$LD_LIBRARY_PATH"

CUDA_LAUNCH_BLOCKING=1 python ./tokenhsi/run.py --task HumanoidCarry \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task.yaml \
    --cfg_env tokenhsi/data/cfg/basic_interaction_skills/amp_humanoid_carry.yaml \
    --motion_file tokenhsi/data/dataset_carry/dataset_carry.yaml \
    --checkpoint output/single_task/ckpt_carry.pth \
    --output_path /scratch/shdpm_root/shdpm0/wenleyan/tokenhsi/carry1/ \
    --test \
    --record_headless \
    --num_envs 1 > gui_out.txt

# sh tokenhsi/scripts/single_task/traj_test.sh
