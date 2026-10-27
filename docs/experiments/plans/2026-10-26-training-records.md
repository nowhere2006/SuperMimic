# Training Records Plan

**Goal:** 训练能持续、原子地保存记录和模型，用户可启动小规模试跑并在重建环境后继续训练。

**Baseline:** `scripts/train_smoke.py`结束时单checkpoint，无持续落盘/恢复入口。

**Primary Metric:** 日志/转移/probe/checkpoint完整可读，开关记录不改变固定seed的训练权重，加载后全局iteration继续。

**Budget:** CPU / 2 envs / 2轮检查 + 1轮resume / 每次600s上限。

1. `training/artifacts.py`：原子写checkpoint/chunks，JSONL持续flush，checkpoint retention和latest。
2. `training/loop.py`：不新增采样/损失，利用现有batch和rollout记录；阶段事件、完整轮次、异常标记、周期保存；checkpoint包含进度和RNG；load显式重建环境/replay。
3. `scripts/train_smoke.py`：CLI试跑参数、初始report及source snapshot、stdout/stderr日志、Ctrl-C落盘、resume和checkpoint安全加载。
4. `docs/evaluation-backlog.md`记录尚未执行的检测；`docs/training-records.md`说明保留/恢复能力、磁盘策略、试跑命令。
5. 单元测试、真实两轮run与一轮resume，以及同seed记录开关一致性核查；完成后交付用户启动命令，不代为启动长训练。
