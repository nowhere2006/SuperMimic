# SuperTrack 分组 L1 与初始贡献校准：实现验收

状态：按用户“直接参考 SuperTrack 调整”的指令保留实现，作为默认试跑配置。检查目标已达到；学习效果未评估，尚未完成正式对照实验。无需为同一已授权实现重复确认保留。

## 干预与基线

- 原始基线：71维原单位平方差，四元数标签同半球对齐，时间求和，加终止BCE。
- 当前默认：六组几何误差的L1，首次真实replay窗口校准固定分母，使未触底各组初始贡献为1；终止BCE不参加自动校准，系数仍1、时间仍求和。保留 `raw_mse` 对照模式。
- 这借鉴SuperTrack的分组L1/初始平衡思想，六组状态和自动校准是工程适配；不能称为原作者公布的SuperMimic损失。输入、采集、奖励、参考时间、网络及actor/critic目标保持原有实现。
- Git：main/unborn，没有commit。改前精确源码及SHA保存在 `.cache/change-baselines/2026-10-27-supertrack-loss/`；两次有效运行各自保存全部运行源码及摘要，经回执核对与当前代码一致。未改写Git历史。

## 条件与命令

沿用项目 `.venv`、Mac CPU、2环境、seed0、完整宽度网络、k8/H32、walk1/dance1采集；固定G1 LAFAN1来源 `lvhaidong/LAFAN1_Retargeting_Dataset@ce1572906efe6157840e8474d5a0d7aa87481e74`，无验证/测试划分。实际依赖、硬件和数据manifest摘要在各run的report/config/source中。

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/train.py --iterations 2 --max-seconds 600 --output output/training-runs/supertrack-loss-check-v2
.venv/bin/python scripts/train.py --resume output/training-runs/supertrack-loss-check-v2/checkpoints/final-iteration-000002.pt --iterations 1 --max-seconds 600 --output output/training-runs/supertrack-loss-resume
```

## 证据

| 检查 | 结果 |
|---|---|
| 单元/组件检查 | 55项通过，含旋转log有限差分、反号与小角度、初始贡献校准、固定分母、下限、raw基线一致性及恢复guard |
| 两轮真实闭环 | completed，192条真实转移；世界模型/policy/critic更新与冻结隔离检查通过 |
| 首次六组归一化贡献 | 全部1.0（按校准构造预期如此，不是性能指标） |
| 数值下限 | 六组均未触底 |
| 校准频率 | 原运行1次；续训0次 |
| 续训 | 从完成2轮到完成3轮，新增160条真实转移；loss全部buffer及校准来源与父checkpoint逐项完全相同 |
| 解析头与梯度 | 实际采集奖励对拍、动作→状态梯度、critic末端状态梯度、固定GAE检查通过 |
| 持久化 | checkpoint `weights_only=True`可读；source与当前源码SHA一致 |

运行目录：

- `output/training-runs/supertrack-loss-check-v2/`，包含 `verification.json` 和 `unit-tests.log`；loop记录时间约5.58秒（不含全部导入/库加载开销，不能推算正式训练性能）。
- `output/training-runs/supertrack-loss-resume/`，loop约4.75秒，同样不代表学习速度。

首批自动得到的尺度（B×K×分量平均绝对误差）：根位置0.00793291 m、根旋转0.04183361 rad、根线速度0.13567103 m/s、根角速度0.65564090 rad/s、关节角0.08698917 rad、关节速度1.29919565 rad/s。这是本次seed0/初始模型/采样窗口的结果，不是通用推荐物理精度。

## 失败与修复记录

- 首次组件检查发现新增checkpoint测试的fixture状态在两个断言之间未恢复；整理测试后55项通过，不涉及改变训练方法。
- 首次真实检查 `output/training-runs/supertrack-loss-check/` 在第一次优化前，因校准来源记录将 `[B]` 的env_id误当成 `[B,K]` 索引而失败。已按replay实际接口修复；该目录的失败report、source、initial/emergency与已采集转移保留。随后新目录v2两轮及续训均通过。
- 该失败不说明分组损失假设失败；也不能用修复后的短跑说明控制性能改善。不重复该失败实现，除非replay字段形状接口发生变化。

## 限制与下一步

初始一批可能不具代表性；贡献相等不等于参数梯度平衡；状态损失相对BCE的最佳比例未知。L1与历史MSE的数值不可直接比较。后续训练若显示某组长期拟合不足，先按固定协议观察原单位误差和加权贡献，再讨论权重或校准数据，不自动在线更新尺度。本次没有启动用户100轮试跑、付费GPU训练或学习效果实验。

旧checkpoint缺少本轮损失语义与校准状态，拒绝自动迁移；保留历史source/数据以便核查。当前工作区保留本次实现，其他待讨论差异未同时修改。
