# CPU Training Loop Plan

**Goal:** 验证在线数据、世界模型拟合、想象和策略/价值更新完整执行。

**Baseline:** `scripts/check_initialization.py`，forward only；main unborn，无提交。

**Primary Metric:** `output/training-smoke/<run>/report.json` 中阶段更新、有限梯度、冻结权重及数据时序检查全部通过；不以 reward 上升定义成功。

**Budget:** CPU / 2 envs / 2 iterations / 600 s deadline；不启动正式训练。

1. 新增 `models/kinematics.py`、`heads.py`、`world_model.py`，共享解析 observation/reward 与硬终止，世界模型学习软终止；用 native MuJoCo 随机姿态/速度对照可微 FK。
2. 新增 `training/replay.py`、`history.py`、`sampler.py`，测试 ring 覆盖、done 边界、term-major 历史与 return sampler。
3. 新增 `envs/tracking_task.py`，用 mjlab 动作/物理/DR，post-forward 状态计算任务信号；参考状态重置后初始化历史。真实 next_state 入库后才 reset。
4. 新增 `training/loop.py`、`config.py` 与 `scripts/train_smoke.py`，先真实采集，8 步自回归 WM loss，再冻结模型进行 32 步 actor BPTT，最后对同一批固定 GAE targets 更新 critic。
5. `.venv/bin/python -m unittest discover -s tests -v`；阶段性便宜检查先于完整 smoke。
6. `.venv/bin/python scripts/train_smoke.py`；保存配置、seed、依赖、数据 manifest digest、源文件 digest、每阶段损失/梯度/更新证据、检查点和 sampler 状态。
7. 更新 `docs/architecture.md`、`docs/mac-feasibility.md`、`docs/training-loop.md` 与 README，报告实测限制。保留失败日志，不以脚本 exit 0 代替梯度和语义核查。
