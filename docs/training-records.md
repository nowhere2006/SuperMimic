# 先训练、保留记录，之后分析

当前任务是让用户能先试跑，并保留后续诊断所需的数据。评测方案在 [evaluation-backlog.md](evaluation-backlog.md)，尚未加入训练循环。

## 直接开始本机试跑

在项目根目录执行：

```bash
# 在 clone 后的 SuperMimic 仓库根目录执行
.venv/bin/python scripts/train.py --iterations 100
```

仍为CPU、2环境、walk1_subject1和dance1_subject1两条动作中的随机参考起点，完整40条动作库会加载。沿用完整MLP宽度、k8/H32、小batch、物理随机化和噪声；100轮是本地试跑配置，不承诺学会动作。默认进程预算600秒。先快速检查入口可使用 `--iterations 2`。

输出目录自动为 `output/training-runs/<UTC timestamp>/`，首次启动时打印路径。也可用 `--output output/training-runs/my-first-run` 指定**不存在**的新目录，避免混入旧运行。

- 默认每10轮保存一次周期checkpoint、保留最近3份；另存初始权重/状态，正常结束时如未刚保存本轮，再存final。
- 默认保存第1轮以及之后每10轮的少量已有WM窗口和想象轨迹probe，最多2个窗口/rollout样本，不额外抽训练样本。
- 默认保存全部本次CPU采集的真实转移。用 `--no-record-transitions` 可关闭磁盘上的转移存档，训练replay仍正常工作。
- `--all-clips` 可选择全部40条作为采集来源；这会改变本次数据范围，初次试跑建议先保留默认两条。`--clips NAME ...` 可明确选择子集。
- `--checkpoint-every 10 --probe-every 10 --keep-checkpoints 3 --max-seconds 600` 都可显式设置；这些保存参数不改变loss或梯度路径。

2026-10-27 起默认世界模型损失为 `supertrack_l1`：六组物理误差取L1，以首次拟合窗口的初始误差校准一次分母，之后固定。可用 `--world-model-loss raw_mse` 新开原始损失对照；`--world-model-group-weights 1 1 1 1 1 1` 显式给出额外乘子，顺序为根位置、根旋转、根线速度、根角速度、关节角、关节速度。方法来源、边界及数据流见 [误差与校准说明](world-model-errors.md)。

写盘会增加耗时。包含Adam状态的完整checkpoint单份约200MiB，初始快照较小；默认周期保留策略控制磁盘占用，final/initial/emergency独立保留。转移与probe按块保存。没有将完整数据集复制进每次运行。

## 每次运行留下什么

| 文件/目录 | 内容 | 之后能用来做什么 |
| --- | --- | --- |
| `config.json`, `report.json` | 配置、seed、依赖版本、机器、数据revision/manifest hash、命令、运行状态/异常、resume来源 | 确认数字来自什么条件；失败保留原因 |
| `source/`, `data-manifest.json` | 源码/测试/项目规则/依赖快照的副本、原始摘要 | 即使本地文件继续改，也能查看当时实现与关节/身体顺序 |
| `metrics.jsonl` | 每个完成迭代的WM state loss/BCE、actor objective、critic loss、学习率、既有梯度摘要和阶段更新检查 | 后续绘训练曲线；不是正式性能评测 |
| `events.jsonl` | 各阶段开始结束、episode结束、clip/起止帧、失败/截断、checkpoint写入耗时与字节数 | 定位哪一步停下、哪些动作失败、保存开销 |
| `transitions/chunk-*.pt` | 真实s/action/next_state、previous_action、观测历史与无噪帧、context、clip/frame、env/episode、全局控制步、policy迭代版本、failure/time-limit/clip-end标记 | 按动作/时间重建采集序列，抽取世界模型诊断窗口 |
| `probes/iteration-*.pt` | 已实际选中的少量WM窗口；想象起点/历史/context、预测状态、动作、奖励、终止概率、V值、传感器噪声和固定GAE targets | 查看当时模型在想象什么，之后设计离线诊断 |
| `checkpoints/` | policy、完整world model、critic、三个Adam状态、sampler统计、完成迭代、配置、Python/NumPy/Torch随机状态、来源摘要 | 安全加载模型或继续训练 |
| `run.log` | Python与原生库stdout/stderr，包括求解器警告 | 查看运行现场；不屏蔽数值警告 |

动作修复后转移/probe显式保存policy_action、executed_action、previous_policy_action、previous_executed_action。WM读取executed_action，平滑项与观测读取clean policy版本；兼容action字段是executed别名、previous_action是previous_policy别名。配置含`action_semantics=clean-policy_executed-dynamics-v2`。旧动作含义checkpoint拒绝直接续训，不凭字段同名混用。

新增损失记录：`metrics.jsonl` 的 `group_l1_mean` 是各组原单位下的平均绝对误差，`group_contribution_mean` 是除固定尺度并乘额外权重后的时间平均贡献；`state_loss` 仍按时间求和。`loss_definition` 保存分母/组顺序/触底标志。`events.jsonl` 的 `world_model_loss_calibrated` 保存首次校准窗口的clip/frame/env/episode起点，checkpoint的 `world_model_loss` 保存分母、初始误差、校准状态及来源。初始贡献接近不能解释为参数梯度已平衡。

critic新增输入后，imagination probe还保存每步相对根位姿4维、所用参考帧及参考根位姿，末端bootstrap也保存对应字段。checkpoint的network_config记录`critic_root_error`，旧784维与新788维输入层配置不自动混用。当前接口及梯度约定见 [critic说明](critic-input.md)。

`metrics.jsonl` 和 `events.jsonl` 逐条flush；转移每32个向量步或每个采集阶段结束落盘。正常Ctrl-C会尽力flush；kill -9可能丢失最后尚未落盘的小块，突然断电还受文件系统缓存影响，不能保证最后写入完整。

梯度字段中 `actor_gradient_norm` / WM梯度norm是裁剪前全局norm；encoder/decoder摘要是现有裁剪后梯度norm。没有保存完整参数梯度。Probe保留detached CPU tensor，不保留计算图；记录开关的固定seed检查已经证明短程训练最终state_dict逐张量相同。保存仍有CPU/磁盘耗时，不能理解为零运行成本。

## 停止与继续

终端按一次 **Ctrl-C**：报告标记interrupted，尽力写入emergency诊断快照。它不覆盖 `checkpoints/latest.json` 指向的最后完整checkpoint；中间阶段快照不允许直接续训。

`latest.json` 中的 `path` 是相对本次运行目录的路径。找到例如 `checkpoints/iteration-000010.pt` 后：

```bash
.venv/bin/python scripts/train.py \
  --resume output/training-runs/<原运行目录>/checkpoints/iteration-000010.pt \
  --iterations 100
```

`--iterations` 表示**本次额外执行的轮数**。默认继承checkpoint训练配置，允许改变本次轮数、设备、时间预算；参考数据和方法配置必须匹配。新运行写到新目录，保留父checkpoint路径及hash。

恢复模型、Adam、采样器和全局调度后，会重新创建仿真环境、重置参考状态并预热新replay。**这是继续训练，不是逐步精确恢复原仿真现场。** 不保存完整物理求解器/接触内部状态，也不从历史转移文件自动重建原replay RNG序列。旧format-v1初始化/smoke快照不支持该续训入口。

损失模式、额外权重、数值下限及BCE系数也是方法配置，续训必须一致；已校准分母直接恢复，不重新校准。2026-10-27之前的旧format-v2快照没有显式损失状态，也会拒绝自动续训，避免将旧MSE训练静默变成新方法。其历史文件保留，可用对应旧source副本检查。新的initial快照在尚未拟合时可未校准，新的完整迭代快照必须包含校准状态。

当前默认数据是preprocess_version2；旧数据保留`data/processed/lafan1-v1/`。恢复还要求manifest SHA完全一致；动作修复前checkpoint和旧数据checkpoint不会自动迁移到新方法/新数据。

## 记录能力的边界

保存了checkpoint和实际转移，之后可以补做跟踪评测、模型误差分析和新的梯度诊断。尚未保存的完整历史梯度、每个优化阶段的权重、接触力不能凭摘要精确恢复。若以后要研究这些更细的问题，应先调整专门的记录方案；目前不为预备检测增加训练分支或额外模型更新。

## 2026-10-26 记录能力验证（旧raw损失）

- `.venv/bin/python -m unittest discover -s tests -v`：43项通过，包括原子保存失败保护、retention/latest、RNG不变、原生stdout/stderr日志、checkpoint恢复校验。
- `output/training-runs/records-final-check/`：两轮真实训练，192条转移分块记录，逐轮metrics、少量32步想象probe、initial/final快照均可用 `weights_only=True` 读取；source副本与摘要核对一致。
- `output/training-runs/recording-check/recording-parity.json`：同seed记录开启/关闭的两轮CPU检查，policy/world_model/critic的全部state_dict张量完全一致。
- `output/training-runs/resume-check/`：从完成2轮的周期checkpoint继续第3轮，进度与学习率延续。
- `output/training-runs/interrupt-check/interruption-verification.json`：完成首轮后发送SIGINT，退出码130、report标记interrupted、emergency保留，latest仍指首轮完整checkpoint。

这些检查只验证记录和训练执行，不报告动作学习效果。用户的100轮试跑尚未由助手启动。

## 2026-10-27 分组损失验证

55项组件检查通过；`output/training-runs/supertrack-loss-check-v2/` 的两轮真实闭环及 `supertrack-loss-resume/` 的续一轮均完成。六组初始贡献均1.0、无下限触底，校准事件分别1次/0次；checkpoint的分母、初始误差和来源逐项一致。回执为原运行 `verification.json`，详见 [本轮结果记录](experiments/results/2026-10-27-supertrack-loss.md)。首次校准来源索引错误的失败目录也保留；这组证据仅证明实现和续训有效。

最新动作/数据修复与集中核对：72项检查通过，`action-data-v2-check/`两轮及`action-data-v2-resume/`续一轮完成；详细证据和版本保护见 [修复结果](experiments/results/2026-10-27-action-data-audit.md)。早期记录的source与data-manifest是历史快照，不能据其路径名将新版数据当成旧版。
