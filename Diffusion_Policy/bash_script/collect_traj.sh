#python -m mani_skill.examples.motionplanning.panda.run -e 'PlugCharger-v1' -n 300 --only-count-success --traj-name Large_trajs

#python ManiSkill/mani_skill/examples/motionplanning/panda/run.py -e 'PlugCharger-v1' -n 300 --only-count-success --traj-name Large_trajs


# 运行您的命令（保持原有参数）
python mani_skill/examples/motionplanning/panda/run.py \
  -e 'LiftPegUpright-v1' \
  -n 200 \
  --only-count-success  \
  --traj-name 256Large_trajs \
  #--sim-backend gpu #--vis 