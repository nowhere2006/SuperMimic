# 世界模型预测误差：定义与尺度来源

记录日期：2026-10-27。用户先授权明确误差定义，随后明确选择参考 SuperTrack 修改损失。当前默认是下述分组 L1 与初始贡献校准；本轮仅做本机小规模验证。

## 比较对象与数据流

世界模型预测误差比较的是**预测状态与真实仿真状态**。参考动作是任务目标，不是动力学拟合标签。

从 replay 的真实 `s_t` 开始，使用记录的实际执行动作及物理上下文，自回归得到 `s_hat_(t+j)`；每一步与同一时间索引、同一环境、同一回合的真实 `s_(t+j)` 比较。沿用现有8步窗口，不能用重置后的状态作为终止步标签。

本次定义适配当前71维状态：根位置3、根四元数4、关节角29、根线速度3、根角速度3、关节速度29。根是自由关节对应的骨盆；不增加逐刚体或接触力预测头。

## 六组物理误差

下表是当前 `supertrack_l1` 损失使用的误差语义。

| 组 | 定义 | 维度与单位 | 坐标与边界 |
|---|---|---|---|
| 根位置 | 预测位置减真实位置 | 3，m | 相对同一环境原点、轴向与世界系一致；不为减小误差而重新对齐平移或航向 |
| 根旋转 | 真实姿态到预测姿态的最短相对旋转向量 | 3，rad | 四元数先归一化；相对旋转 `conj(q_true) ⊗ q_pred`，选择最短旋转分支；向量在真实骨盆坐标系表示，长度为夹角 |
| 根线速度 | 预测速度减真实速度 | 3，m/s | 两者都用世界系分量 |
| 根角速度 | 预测角速度减真实角速度 | 3，rad/s | 沿用状态约定，比较各自骨盆本体系下的分量，不混用世界系角速度 |
| 关节角 | 对应关节的预测角减真实角 | 29，rad | 保持同一关节名称、顺序及零位；当前 G1 有限位关节不按连续转轴任意绕回角度 |
| 关节速度 | 对应关节的预测速度减真实速度 | 29，rad/s | 与关节角保持相同的关节顺序 |

拼接为70维误差 `[dp, rotation_vector, dv, domega, dq, dqdot]`。状态仍然是71维；旋转误差用3维表示不意味着删除状态中的四元数。

旋转定义使 `q` 和 `-q` 得到相同误差。零夹角需要稳定的小角度处理；恰好180度时旋转轴正负不唯一，是后续实现需明确处理的边界。角速度分量的坐标约定是工程选择，不声称为论文明确规定。

## 尺度与权重：初始贡献校准后固定

每组用同单位的参考尺度除误差，即表达“误差是参考值的多少份”。名义误差尺度不是物理常数，也不是失败阈值。平方误差中尺度减半，同一误差的贡献变为4倍；本轮采用的 L1 中则变为2倍。

用户选择借鉴 [SuperTrack](https://theorangeduck.com/media/uploads/other_stuff/SuperTrack.pdf#page=5) §3.4/Algorithm 1：物理/几何误差分组 L1，并设置权重使训练初期贡献接近。SuperTrack 没有给出可直接复制的自动校准程序；它预测各刚体的加速度，本项目保留 G1 的71维状态和六组误差，属于本项目适配。

具体流程：

1. 首次拟合时，直接使用已抽中的真实 replay 窗口（默认8个窗口、每个8步）。保持自回归预测，不另抽校准数据，不用参考动作替换真实标签。
2. 对每组误差取 batch×time×分量的绝对值均值，得到6个初始误差。分母设为 `max(initial_error, world_model_scale_floor)`；默认数值下限 `1e-6` 防除零，触底组明确记录，不把下限解释成物理容差。
3. 这6个分母只校准一次，之后固定且不带梯度。每步状态损失为六组 `group_weight * mean(abs(error)) / scale` 之和，时间上仍求和。默认额外组权重全1，未触底时每组初期贡献约1，总状态损失约 `6*K`。
4. 终止 BCE 保持 batch均值、时间求和，独立系数默认1；它不参与自动校准。状态组贡献接近不等于状态与 BCE 已平衡，更不等于各组对网络参数的梯度接近。
5. raw L1、加权贡献、分母、触底标志、校准窗口身份与来源写入日志和checkpoint；续训读取已保存的分母，不根据新环境/replay重算。只从未拟合的initial checkpoint开始时才允许首次校准。

初始单批可能不具代表性，这是已知限制；本轮验证只检查实现及恢复，不证明权重最优。各组额外权重可配置，但调整属于新方法配置，不在已有checkpoint上静默切换。

## 第三方固定尺度的依据核查

核查固定版本：`CuboiLeo/SuperMimic-Reproduced@d9be385b3eeb13a638fb33fb200d3bfd9be9839f`。

| 误差组 | 该版本 `err_scale` |
|---|---:|
| 根位置 | 0.05 m（5 cm） |
| 根旋转 | 0.1 rad（约5.73度） |
| 根线速度 | 0.5 m/s |
| 根角速度 | 1 rad/s |
| 关节角 | 0.1 rad（约5.73度） |
| 关节速度 | 2 rad/s |

- [代码 `StateSpecCfg` 与 `error_vector`](https://github.com/CuboiLeo/SuperMimic-Reproduced/blob/d9be385b3eeb13a638fb33fb200d3bfd9be9839f/src/supermimic/world_model/state_spec.py#L46) 确认数值及训练误差归一化实现。输入尺度、输出尺度、误差尺度是三套不同配置。
- [复现计划 D3](https://github.com/CuboiLeo/SuperMimic-Reproduced/blob/d9be385b3eeb13a638fb33fb200d3bfd9be9839f/docs/REPRODUCTION_PLAN.md#L168) 给出的理由是避免原始单位下关节速度主导损失；没有逐个解释六个具体数值的推导。计划提到 running-std 备选，不等于实际实现或实验验证了备选。
- [审查记录第87行](https://github.com/CuboiLeo/SuperMimic-Reproduced/blob/d9be385b3eeb13a638fb33fb200d3bfd9be9839f/docs/REVIEW_LOG.md#L84) 明确将 D3 名义尺度列为论文未指明处的自选值。
- 完整阅读该版本 [实验日志](https://github.com/CuboiLeo/SuperMimic-Reproduced/blob/d9be385b3eeb13a638fb33fb200d3bfd9be9839f/docs/EXPERIMENTS.md)，未找到专门改变这些尺度、控制其他条件的敏感性实验或搜索记录。不能据此断言维护者从未私下调参，也不能称其经过精细优化或证明任意尺度都可用。
- 计划称根部误差在航向系比较，但代码位置直接相减、旋转用真实姿态的相对旋转；具体实现以代码为准。各组三轴采用同一尺度时，位置误差的平方和对共同坐标旋转不变。

## 与当前实现的关系

`training/model_loss.py` 提供物理误差、分组L1、一次校准与保存校验；`training/loop.py::fit_world_model` 使用该损失并保留8步梯度路径。默认 `world_model_loss="supertrack_l1"`。`raw_mse` 模式保留旧71维原始平方差及四元数同半球对齐，可作为后续受控对照。

命令：`scripts/train.py --world-model-loss raw_mse` 新开旧损失基线；`--world-model-group-weights POS ROT VEL OMEGA JOINT JOINT_VEL` 设置六个正数乘子，默认全1。续训要求模式/配置一致，旧checkpoint缺少新损失定义时不自动迁移。

这改变了原始 SuperMimic 状态 MSE 的实现选择，不能标为论文明示方法。新L1 loss值也不能与历史raw MSE或第三方固定尺度的数值直接横比。设计/预算/证据见 [实验卡](experiments/specs/2026-10-27-supertrack-loss.md)。
