#!/bin/bash
# 获取当前脚本所在目录（即工作目录）
WORK_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# 关键设置：将工作目录下的 ManiSkill 目录添加到 PYTHONPATH 最前面
export PYTHONPATH="$WORK_DIR/ManiSkill:$PYTHONPATH"

# 验证路径（可选，调试时保留）
echo "Using PYTHONPATH: $PYTHONPATH"
python -c "import mani_skill; print('Imported from:', mani_skill.__file__)"