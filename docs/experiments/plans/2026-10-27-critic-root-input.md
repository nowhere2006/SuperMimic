# Critic相对目标输入计划

**Goal:** 补充4维目标相对信息并保留actor输入梯度。

**Baseline:** main/unborn当前源码精确快照 `.cache/change-baselines/2026-10-27-critic-root-input/`；原critic输入784维。

**Primary Metric:** 已知几何/可区分性/梯度/固定targets/模块隔离/checkpoint续训检查，不比较短跑回报。

**Budget:** Mac CPU2环境、2轮+续1轮、单次600秒上限。

---

1. 新增 `models/critic_features.py`，提供参考航向系根位置差及最短yaw差，保留梯度。
2. 扩展 `models/networks.py::Critic` 的显式输入与NetworkConfig；更新训练每步/末端/拟合调用及初始化检查入口。
3. 更新并新增单元测试，运行 `.venv/bin/python -m unittest discover -s tests -v`。
4. `.venv/bin/python scripts/train.py --iterations 2 --max-seconds 600 --output output/training-runs/critic-root-input-check`。
5. 从完成checkpoint续1轮到新目录 `output/training-runs/critic-root-input-resume`，核对输入定义、分组loss校准恢复及probe时间对应。
6. 记录source/config/seed/data/checkpoint/失败与成功回执，更新未决清单；不叠加参考时钟或动作噪声语义改动。
