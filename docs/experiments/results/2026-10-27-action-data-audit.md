# 动作/插值修复与集中核对结果

按用户授权保留修复，当前默认路径已生效。两个已知缺陷已通过定向与集成检查；集中核对未发现其他已确认的核心语义错误。学习效果与GPU规模不在本次结论范围内。

## 改动与依据

- clean policy动作、executed动作、二者各自的previous版本显式分开；物理/WM用执行动作，正则化/观测用clean动作，兼容别名入库校验。世界模型组合forward也要求显式给出两种角色。动作语义配置版本为`clean-policy_executed-dynamics-v2`。
- float64时间网格、归一化最短弧插值、近同旋转nlerp替代左端提前返回；输出preprocess_version=2。保留源CSV、关节映射、50Hz帧数与差分约定。
- 只读审查覆盖模型/回报/梯度与环境/数据/恢复两条链，主助手核对关键发现并增加官方奖励oracle及FK适用范围保护；详细矩阵在 [semantic-audit.md](../../semantic-audit.md)。

## 命令与数据版本

Git main/unborn，无commit。改前精确文件/manifest SHA在 `.cache/change-baselines/2026-10-27-action-data-audit/`；各训练run存source与SHA。全40条来源revision仍为ce1572906efe6157840e8474d5a0d7aa87481e74。

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/train.py --iterations 2 --max-seconds 600 --output output/training-runs/action-semantics-v1-data-check
.venv/bin/python scripts/prepare_lafan1.py --local-only --output-dir data/processed/lafan1-v2-building
.venv/bin/python scripts/check_resampling.py --directory data/processed/lafan1-v2-building --baseline data/processed/lafan1 --output output/lafan1/resampling-v2-check.json
.venv/bin/python scripts/check_lafan1.py --directory data/processed/lafan1-v2-building --output output/lafan1/v2-load-check.json
```

上述三条数据检查在激活前执行；随后新目录改名为`data/processed/lafan1`，旧目录保留`data/processed/lafan1-v1`。原始CSV没有更改。激活映射在`output/lafan1/v2-activation.json`。当前若重跑独立检查，使用canonical新库与`--baseline data/processed/lafan1-v1`。

```bash
.venv/bin/python scripts/train.py --iterations 2 --max-seconds 600 --output output/training-runs/action-data-v2-check
.venv/bin/python scripts/train.py --resume output/training-runs/action-data-v2-check/checkpoints/final-iteration-000002.pt --iterations 1 --max-seconds 600 --output output/training-runs/action-data-v2-resume
```

训练均为Mac CPU2环境、seed0、walk1/dance1采集，完整40条库加载，无评测划分；未用短跑回报挑选结果。

## 验证结果

| 检查 | 结果 |
|---|---|
| 单元/组件 | 72项通过；包含4项独立SciPy插值测试、4项动作边界检查、官方mjlab奖励oracle |
| 旧数据动作修复隔离运行 | 2轮完成，192条实际转移 |
| 新数据动作修复运行 | 2轮完成，192条转移；续1轮增加160条，2→3 |
| 记录动作含义 | 所有三次运行的转移逐项核对：action别名=executed，previous别名=previous clean；观测/clean观测最后29维=previous clean；探索动作实际不同于policy |
| 实际奖励重建 | 新库192条真实next_state与clean动作重建reward最大绝对差1.12e-8；故意改用executed差分的负对照最大差约0.00219，证明检查能识别错误角色 |
| 新库独立插值 | 40条441,140帧：根旋转对SciPy最大角差1.13e-7 rad；旧库相同对照最大0.003406 rad（约0.195度） |
| 新库独立位置/速度 | 根位置最大4.77e-7 m、关节角1.20e-7 rad；根角速度1.64e-6 rad/s；其余速度误差<1e-6 |
| mjlab接入 | 40条runtime MotionLoader完整读取一致，120个首/中/末帧Warp重建通过 |
| 更新/恢复 | WM/policy/critic更新与冻结隔离、固定GAE、梯度检查通过；续训不重新校准WM分母（事件1次/0次） |
| 版本保护 | 旧动作语义checkpoint拒绝；新旧数据manifest不一致拒绝；final新库两次run源码与当前SHA一致 |

新manifest SHA：`8def4b2d2544a8d6cb6a3a6284bcb0f5f04a16fe79c32e8ea8fd1cf4933f1ca8`。

旧manifest SHA（已归档）：`137940504e40648d7b5ea8529057a75c47ce0692661070f62e7c41a26fd6db66`。

主回执：`output/training-runs/action-data-v2-check/verification.json`；unit日志在同目录`unit-tests.log`。两个独立数据报告为`output/lafan1/resampling-v2-check.json`与`v2-load-check.json`。

## 失败/边界与代码保留

- 子agent最初用pytest运行4个插值函数测试通过；主工程unittest入口最初发现0项（退出5），因此增加`load_tests`桥接，随后全套包含这4项，72项通过。不是将“未运行”计为通过。
- 旧数据stage与新数据stage短跑均记录solver迭代上限提示，续训此回未出现；没有改求解器或隐藏警告，不能因此声称接触求解都充分收敛。
- 数据version2与旧version1的差异包括修复近同旋转，以及float64时间/线性重采样/速度差分的精度改善；不能把所有速度变化都归因于某一个分支。数据版本/manifest明确隔离。
- 不保留未证实的“COM累加”或“torso奖励一定应换pelvis”猜测为缺陷，原生源码核查排除；无依据不改变方法。
- 保留全部本轮实现与证据。旧源码/数据没有删除，没有Git提交、远程传输、正式训练或付费GPU动作。下一步可在此修复版本做本机试跑；正式GPU协议和剩余方法工程差异需按矩阵收敛。
