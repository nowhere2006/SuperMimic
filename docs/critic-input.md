# Critic相对目标输入与时间对应

2026-10-27用户选择给critic补足参考目标信息。当前默认增加4维相对根位姿；其余尚未讨论的动作噪声语义保持原实现。

## 输入和职责

critic输入是：原始机器人state71、未来参考goal680、物理context33，以及相对参考根位置3/yaw差1，共788维。隐藏层仍256/128，输出一个价值数值。默认参数量235,009。policy仍读取930维观测历史和680维参考目标，这4维只供训练的critic使用。

`models/critic_features.py::root_tracking_features` 从机器人状态与**同一时刻**的参考骨盆位置/四元数计算：

- 位置差：机器人骨盆位置减参考骨盆位置，再转到参考的水平航向坐标系；3维，单位m。
- 朝向差：机器人yaw减参考yaw，取最短有符号角差；1维，单位rad。
- 两者均使用相同环境原点；训练库参考位置和state都为env-local。初始化入口的command参考位置含scene origin，因此在该入口扣除一次。
- 输入暂用1m/1rad的单位尺度，无额外归一化；它与世界模型误差校准无关。yaw的前向水平投影退化时用零yaw约定并关闭该退化方向的梯度，避免atan2(0,0)。这不意味着竖直姿态有唯一可观测的yaw，最短角差在±π也有分支边界。

它使仅参考全局平移/朝向变化、原goal特征不变的情况可以被critic区分。不能据此说critic输入已经包含所有历史信息，或部署策略会自动获得全局定位能力。

## 调用时序与梯度

1. `V_t` 用预测 `s_t`、`g_t` 和 `reference_t` 计算四维误差。
2. 执行动作后预测 `s_(t+1)`，奖励/终止仍对 `reference_(t+1)` 计算。
3. 在H=32的末端，`V_H` 用 `s_H/g_H/reference_H`，相对误差由预测state可微计算，不能detach。actor更新时critic权重冻结，但梯度能沿这4维及原state输入回到预测状态、世界模型、policy。
4. 拟合critic时，使用同批已经detach的state/goal/context/root-error，GAE目标仍只算一次；只更新critic参数。

“奖励差一帧”指代码索引对应错误，例如误把s101对ref100，而不是机器人摔倒后自动暂停/重同步参考。当前固定时钟每控制步推进.02秒；跟不上产生跟踪误差，满足失败条件则终止。是否要引入可变参考相位属于另外的方法改变，当前没有引入。

## 来源、兼容性与记录

- 参考固定d9be385版本的 `policy/nets.py::Critic`、`world_model/heads.py::root_error_features`。本项目保留raw state，不同时改成参考实现的canonical state。
- 论文§3.3仅写V(s,g,xi)，没有列全s/g内容；该4维属于显式工程补充，不称为论文作者给出的输入表。
- `NetworkConfig.critic_root_error=True` 是默认；程序级设False可检查旧784维基线。默认开启时必须显式提供4维输入，不以全零静默补缺。
- 新网络配置随checkpoint保存；旧配置/输入层维度与新模型不兼容，不自动迁移权重。恢复入口继承checkpoint的网络配置并严格核对。
- imagination probe保存逐步四维误差、参考帧与参考根位姿，以及末端参考帧/root-error。metrics新增 `bootstrap_root_error_gradient_norm` 检查输入梯度；该值不是控制性能指标。

证据见 [本轮结果](experiments/results/2026-10-27-critic-root-input.md)。
