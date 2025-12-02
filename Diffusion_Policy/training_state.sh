#!/bin/bash

seed=3


python Diffusion_Policy/train.py --env-id "PlugCharger-v1"  --batch_size 64   --demo-path demos/PlugCharger-v1/motionplanning/plugcharger.state.pd_ee_delta_pose.physx_cpu.h5 \
  --control-mode "pd_ee_delta_pose" --sim-backend "physx_cpu"  --max_episode_steps 300 --total_iters 30000  --exp-name diffusion_policy-PlugCharger-v1-state-${demos}_motionplanning_demos-${seed} \
  --ckpt_path runs/diffusion_policy-PlugCharger-v1-state-_motionplanning_demos-2/checkpoints/best_eval_success_at_end.pt