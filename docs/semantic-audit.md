# 集中语义核对：当前闭环的含义、依据与边界

2026-10-27，按用户“修复然后集中核对”的指令完成。范围为当前本机复现工程的采集、参考数据、模型、训练与恢复；不等于正式论文成绩复现或GPU验收。

## 本轮结论

已修复动作含义混用和近同四元数插值问题。当前核心路径在本轮审查与定向检查范围内，没有新增确认的语义缺陷；这不构成所有代码已被形式证明正确。源码无Git commit（main/unborn），保留改前快照、各运行source副本及SHA。

## 动作现在如何流转

| 名称 | 含义 | 使用者 |
|---|---|---|
| `policy_action` | 加噪前的策略输出a_t，29维归一化指令 | 平滑惩罚、s_(t+1)观测中的上一动作、policy动作记录 |
| `executed_action` | 实际执行u_t=a_t+epsilon_t | mjlab动作/物理、WM动力学监督输入、执行动作记录 |
| `previous_policy_action` | 前一步加噪前输出a_(t-1) | 当前观测最后29维、动作差分惩罚、想象起点 |
| `previous_executed_action` | 前一步实际指令u_(t-1) | 记录、连续性与诊断，不进入平滑项 |

replay的兼容别名`action=executed_action`，`previous_action=previous_policy_action`，入库强制相等；核心训练使用显式字段。`TrackingTask.step`必须同时明确两种动作。想象时不加探索噪声，当前策略动作同时是本步动力学执行动作；起点的上一动作仍从真实replay读取clean policy版本。

这一选择的依据：当前噪声用于真实采集探索，平滑项约束策略自身输出；固定mjlab的`action_rate_l2`定义为raw policy output差分，参考复现也将clean与executed分开。论文明确执行噪声和日志动作，但没有充分列出所有上一动作接口的细节，因此记录为已授权工程修正，不冒称恢复作者私有实现。

## 逐项核对矩阵

| 边界 | 当前明确含义 | 核查依据与证据 | 结论/限制 |
|---|---|---|---|
| 原始动作→参考状态 | 源30Hz、root xyzw、29关节按名字映射；输出50Hz及wxyz | 源清单SHA、官方CSV关节列顺序、全部40条441,140帧独立SciPy对照 | version2已激活，旧version1保留；不是动力学可行性证明 |
| 插值→速度 | float64网格、单位四元数最短弧SLERP/近同nlerp；输出网格中心差分，根角速度先世界系再转MuJoCo本体系 | 恒角速度/反号/近同旋转测试；全帧SciPy旋转和np.interp/差分对照 | 非均匀末尾不额外补点，首尾角速度沿用邻近中心估计 |
| 参考→G1身体状态 | 同一编译模型位置/速度FK，无物理步进 | 40条mjlab运行时加载，120个首/中/末关键帧Warp重建 | 关键帧检查不是所有帧的物理运动验证 |
| 执行动作→真实状态 | executed_action经原生PD/encoder bias补偿，.02s控制步 | TrackingTask边界定向测试、真实转移；原生动作缩放/补偿源码核对 | 实际施加命令与clean policy分开 |
| state/context坐标 | p减各env origin；线速度世界系，角速度骨盆系；context33读实际摩擦/bias/COM | read_model_state、原生观测与FK对拍；COM DR按默认值重新采样源码核对 | COM不累加；部署噪声/随机化是既定配置 |
| 重置/终止步入库 | auto_reset=False；保存post-step/pre-reset next_state，再按failure/timeout/clip尾重置 | g1配置与mjlab手动重置源码；replay episode/clip/state/action连续性测试 | 终止标签与truncated分开，时间上限不训练成失败 |
| 参考时钟与奖励 | s_t执行u_t后，s_(t+1)对reference_(t+1)；不因踉跄重同步 | 真实/想象代码索引；critic32步及末端参考与NPZ逐帧对拍 | 直接的正确性约定，不再作为待用户选择的问题 |
| 奖励几何及聚合 |14体、torso anchor、6指数项+clean action-rate+限位，乘dt | 新增测试直接调用官方MotionCommand对齐方法及六个reward函数；native action_rate作独立参照 | 纸面“root”未注明奖励body；torso由固定mjlab G1配置直接支持，失败root仍按论文用pelvis |
| FK模型适用范围 | 固定G1零关节anchor、标量关节，30body | 原生模型全部jnt_pos=0；随机姿态/速度全body MuJoCo对拍 | 新增非零anchor拒绝保护，不声称支持任意机器人/ball joints |
| WM拟合标签 | s0起步、8步执行日志动作自回归，逐步对实际next_state；物理context来自同episode | loop/replay审查与连续性测试，执行动作探针 | 不拿参考动作当动力学标签，不teacher-force中间状态 |
| WM损失 | 六组几何L1，首批校准固定尺度；终止BCE，解析头不拟合 | 用户选择SuperTrack思路；校准、反号、几何/梯度测试 | 对原论文MSE的显式方法选择；初期贡献相等不等于梯度已平衡或尺度最优 |
| 策略观测 |93维，10帧term-major；传感器uniform noise，最后29维为clean上一步输出 | history测试、native初始观测对拍、每条已保存转移最后槽位核对 | 基础mjlab diagnostic obs仍见执行动作，训练policy使用wrapper维护的明确历史 |
| policy/critic参考输入 | policy未来10帧68维；critic另有相对根位置3/yaw1 | critic可区分性/有限差分/冻结梯度测试及probe参考对拍 | goal具体拆分、当前+9和critic4维不是论文公布的唯一实现；policy仍无额外全局定位反馈 |
| actor梯度 | 冻结WM/critic参数，保留输入图；H32软生存回报+末端V | 原文Eq.2、GAE Eq.3；动作→state及末端V→state/root-error梯度检查 | 有限非零梯度不能证明其方向适合长期控制 |
| critic拟合 | GAE一次生成detach；同批detached输入，5epochs×4minibatches | 固定targets、参数隔离检查；无EMA target network | 符合当前论文递推与训练结构，不等于价值已拟合准确 |
| 训练记录/恢复 | 原子snapshot、完整迭代可恢复、partial仅诊断；恢复权重/Adam/sampler/RNG，重建env/replay | 原子保存测试及真实2→3续训 | warm restart，不是精确仿真现场恢复；旧动作语义或不同manifest拒绝续训 |

## 审查中排除的疑点

- **torso reward是否应改为pelvis**：固定mjlab `tracking/config/g1/env_cfgs.py` 指定torso anchor，`tracking_env_cfg.py`的global_root名称实际调用anchor奖励。没有据名称猜测改成pelvis。论文终止明确pelvis，现实现保留这一区别。
- **COM偏移是否每次累加**：mjlab `dr.add` 的`uses_defaults=True`，重采样基于编译默认值；不是当前值继续加。无需修复。
- **G1关节anchor是否被FK漏算**：当前模型全部为零，随机全body对拍通过；为未来非零anchor显式拒绝，不把模型范围外能力写成已支持。

## 仍未验证/未收敛的事项

1. 短跑出现solver迭代上限提示，记录未屏蔽；当前10/50设置未改。有限loss/对拍通过不证明每次接触求解充分收敛，长跑需观察其频率与标签影响。
2. 当前CPU debug环境数、batch、每外层更新采集步数与正式论文UTD不同；GPU/CUDA路径、并行replay性能及正式规模配置未验证。
3. reset扰动、COM重采样频率、回报bin的EMA/历史权重、截断回报估计等为已记录工程差异；回报bin目前用实际回合后缀、不对时间截断bootstrap，可能造成尾部估计偏低。正式协议前应明确，不在本轮无依据叠加新算法。
4. 网络激活、状态/目标编码、残差积分、终止头结构/初值、损失校准等有论文未指明的工程选择；已列出不是逐指令作者代码复原。
5. 未做正式训练/评测划分、多seed效果比较或论文结果复现；未自动开始100轮或GPU计费训练。

本轮证据、命令与保存位置见 [修复结果](experiments/results/2026-10-27-action-data-audit.md)。
