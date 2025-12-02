
python run_with_local_maniskill.py  ManiSkill/mani_skill/trajectory/replay_trajectory.py \
  --traj-path demos/PlugCharger-v1/mp-v2-basecam-rep_lock/trajs_test_256_cpu.h5 \
  -c pd_ee_delta_pose -o rgb \
  --save-traj  --verbose