# Critic相对根位姿：实现与局部验证

状态：按用户选择保留critic额外4维信息。组件/真实CPU闭环/恢复通过，未做学习效果实验。

## 实现及基线

默认critic从784维扩到788维，追加参考航向系根位置差3和最短yaw差1，隐藏层不变；235,009参数。每个V_t与参考t对应，V_H与参考H对应；actor阶段特征可微，critic拟合阶段特征detach。policy、奖励帧对应、动作噪声、上一轮分组L1没有同时修改。

原始文件及SHA保存在 `.cache/change-baselines/2026-10-27-critic-root-input/`；Git仍main/unborn，无提交。当前代码根据用户授权保留，不用一次短跑决定科学优劣。

## 条件与命令

项目隔离环境；Mac CPU、2环境、seed0、完整隐藏层、k8/H32。固定G1 LAFAN1 `ce1572906efe6157840e8474d5a0d7aa87481e74`，walk1/dance1采集、全库加载，无评测划分。

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/train.py --iterations 2 --max-seconds 600 --output output/training-runs/critic-root-input-check
.venv/bin/python scripts/train.py --resume output/training-runs/critic-root-input-check/checkpoints/final-iteration-000002.pt --iterations 1 --max-seconds 600 --output output/training-runs/critic-root-input-resume
.venv/bin/python scripts/check_initialization.py --steps 3 --output output/initialization/critic-root-input/check.json
```

## 核查结果

- 63项组件检查通过：已知坐标/yaw差、共同平移及yaw变换不变、四元数反号、identity/竖直退化梯度、数值差分、同goal但不同目标位置可区分、构造只依赖新增输入的冻结critic能传state梯度、旧配置拒绝等。
- 真实2轮闭环完成192条转移；续1轮增加160条转移，进度2→3。每阶段参数更新及冻结隔离、GAE固定、动作→state、末端state及root-error梯度检查通过。
- 读取首轮probe的32个V_t参考帧和V_H参考帧，逐项核对原始processed NPZ骨盆位姿，确认current/terminal引用正确；从保存state/reference重算4维，与probe逐项一致。
- checkpoint保留`critic_root_error=True`和788维输入层。续训前后世界模型的固定误差校准buffer完全相同，校准event计数分别1次/0次。
- 更新后的初始化入口3步前向通过，参数保持不变；schema新增四维定义并hash `critic_features.py`。
- 回执 `output/training-runs/critic-root-input-check/verification.json`，组件日志 `unit-tests.log`；两次run的config/report/source/checkpoint均保留。
- 每份run的source归档与各自report SHA一致。初始化脚本的hash/schema补充发生在首个训练run之后，所以该run归档的初始化脚本与当前版本不同；核心训练源码一致，续训与初始化检查使用补充后的版本。此差异在回执显式记录。

## 发现、限制与下一步

- 首次有限差分测试中，参考四元数直接用了输入state的detach切片，仍共享底层存储；数值扰动同时改变了参考，导致测试Jacobian不符。将测试参考clone后63项通过；保留初次失败日志 `unit-tests-initial-failure.log`。训练参考来自动作库，不共享预测state存储。
- 真实短跑日志出现`solver iterations limit reached`，原样保留。既有检查未出现非有限loss/梯度，仍不据此保证物理求解收敛；本次没有更改solver_iterations=10。长跑前需关注警告频率及其对动力学标签的影响。
- 原始state分支与新增4维都参与估值；非零梯度不代表价值拟合准确。这个补充也不解决所有历史依赖或部署策略的全局反馈。
- 没有启动100轮试跑或付费GPU训练。奖励参考时序已明确为同一时刻直接对应，由助手负责正确性检查；后续只讨论干净动作与执行动作在平滑惩罚等接口中的含义。
