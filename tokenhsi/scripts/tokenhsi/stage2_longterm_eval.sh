#!/bin/bash

python ./tokenhsi/run.py --task HumanoidLongTerm4BasicSkills \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_longterm.yaml \
    --cfg_env tokenhsi/data/cfg/longterm_task_completion/amp_humanoid_longterm_4basicskills_0.yaml \
    --cfg_task_plan tokenhsi/data/dataset_longterm_task_completion/task_plans/4_basic_skills_0/cfg.yaml \
    --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
    --hrl_checkpoint output/tokenhsi/ckpt_stage1.pth \
    --checkpoint output/tokenhsi/ckpt_stage2_longterm.pth \
    --test \
    --num_envs 512 \
    --headless \
    --eval \
    --seed 0