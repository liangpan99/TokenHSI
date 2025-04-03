#!/bin/bash

target_task=$1 # sit, carry, climb, traj

python ./tokenhsi/run.py --task HumanoidTrajSitCarryClimb \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task.yaml \
    --cfg_env tokenhsi/data/cfg/multi_task/amp_humanoid_traj_sit_carry_climb.yaml \
    --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
    --checkpoint output/tokenhsi/ckpt_stage1.pth \
    --test \
    --num_envs 512 \
    --headless \
    --eval \
    --eval_task $target_task \
    --seed 0
