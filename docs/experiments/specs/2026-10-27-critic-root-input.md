# Critic相对目标输入：局部实现检查

用户于2026-10-27选择给critic补足相对目标信息。本轮落实先前讨论的根位置差3+yaw差1，不扩展部署policy观测。用户同时询问奖励的帧对应：当前保持 `s_(t+1)` 对 `reference_(t+1)`，不实现摔倒后参考重同步。

- 假设：增加4维相对根位姿输入，critic可区分仅参考全局位置/朝向不同的情况，并在actor末端bootstrap中保留该输入对预测state的梯度；不以短检查证明回报提高。
- 基线：当前raw state71+goal680+context33的critic。改前源码与SHA保存在 `.cache/change-baselines/2026-10-27-critic-root-input/`；Git仍main/unborn，按项目禁自动提交规则原位保存精确快照。
- 设计：默认追加 `[reference_heading^-1*(robot_root_position-reference_root_position), wrap(robot_yaw-reference_yaw)]`，单位m/m/m/rad，额外输入尺度暂为1m/1rad；保留旧raw state，MLP仍256/128。该4维是派生的训练输入，不增加数据采集字段，不传给policy。
- 时序：每个V_t使用预测s_t、g_t及参考frame_t；末端V_H使用s_H、g_H、reference_(frame+H)。critic拟合使用同一批detached输入和固定GAE targets；actorbootstrap通过root-error到state的梯度保留。yaw投影退化时明确用零yaw约定防非有限梯度，不声称该处朝向可观测。
- 成功标准：已知位置/旋转误差正确，共同平移/yaw变换下4维不变；目标平移但goal编码不变时新增输入可区分；构造critic依赖新增4维可得到有限非零state梯度；真实闭环模块隔离/GAE固定/新输入bootstrap梯度通过；probe/checkpoint保存并可续训。新网络配置与旧checkpoint严格区分，不自动迁移旧输入层权重。
- 数据：既有固定G1 LAFAN1全库加载、walk1/dance1采集，seed0，无验证/测试划分。
- 预算：Mac CPU2环境，默认完整网络，2轮真实闭环+续1轮，每次600秒上限；不启动正式训练或其他方法改动。
- 保持：上一轮SuperTrack分组L1/校准、奖励、执行噪声、采样、网络隐藏层、actor/critic目标、优化器不变。critic可选关闭新增输入仅供局部基线检查；默认开启。
- 来源：参考实现固定d9be385中 `policy/nets.py::Critic`、`world_model/heads.py::root_error_features`。论文§3.3仅给V(s,g,xi)，未指明此4维；这是用户选择的工程补充，不是作者公布的精确输入。
- 停止条件：任何输入时序、梯度、checkpoint兼容性或模块更新隔离检查失败，先修局部，不开展学习效果实验。存在更多历史信息/Markov状态不完整问题，不把追加4维称为完整解决所有可观测性。
