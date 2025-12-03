#python -m mani_skill.examples.motionplanning.panda.run -e 'PlugCharger-v1' -n 300 --only-count-success --traj-name Large_trajs

#python ManiSkill/mani_skill/examples/motionplanning/panda/run.py -e 'PlugCharger-v1' -n 300 --only-count-success --traj-name Large_trajs

# 获取当前脚本所在目录（即工作目录）
WORK_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# 关键设置：将工作目录下的 ManiSkill 目录添加到 PYTHONPATH 最前面
export PYTHONPATH="$WORK_DIR:$PYTHONPATH"

python -c "import mani_skill; print('Imported from:', mani_skill.__file__)"

# 运行您的命令（保持原有参数）
python mani_skill/examples/motionplanning/panda/run.py \
  -e 'StackCube-v1' \
  -n 10 \
  --only-count-success  \
  --traj-name traj_debug \
  -o rgb
  #--sim-backend gpu #--vis 