#!/bin/bash

seed=1
demos=100

python diffusion_policy/train_rgbd.py --env-id "PlugCharger-v1" --obs-mode "rgb" --batch_size 8   --demo-path demos/PlugCharger-v1/motionplanning/Large_trajs.rgb.pd_ee_delta_pose.physx_cpu.h5 \
  --control-mode "pd_ee_delta_pose" --sim-backend "physx_cpu" --num-demos ${demos} --max_episode_steps 300 --total_iters 60000  --exp-name Large_trajs_PlugCharger+rgb-${demos}