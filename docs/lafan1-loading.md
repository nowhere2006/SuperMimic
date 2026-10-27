# 完整 G1 LAFAN1 动作库

2026-10-26：用户明确授权完整加载 G1 LAFAN1。本模块完成参考数据的下载、预处理、整库载入与 mjlab 互操作检查；没有初始化 policy、world model、critic 或 replay buffer。

后续状态：用户另行授权的[参考接口与模型初始化](initialization-contract.md)已经接入该动作库。下文保留本数据加载模块原先的验证范围。

## 当前关系

```text
40 条 G1 参考 CSV
    ↓ 30 → 50 Hz；计算速度和身体状态
40 条 mjlab 格式 NPZ + 数据清单
    ↓ MotionLibrary.from_directory
整库 CPU 张量，可按“动作编号 + 段内帧编号”读取
```

参考动作库保存希望机器人怎么动。它尚未作为 command 挂入我们的 G1 诊断环境，后续仍需接入目标生成和跟踪任务；这些文件不会自行控制机器人。

## 完整性的范围

- 数据源：[lvhaidong/LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/tree/ce1572906efe6157840e8474d5a0d7aa87481e74/g1)，固定提交 `ce1572906efe6157840e8474d5a0d7aa87481e74` 的全部 40 条 G1 CSV。没有导入 H1/H1_2，也没有重新从原始人体 BVH 做重定向。
- 原始 CSV 共 90,487,276 字节，每个文件均验证字节数和该提交的 Git blob 摘要，并记录 SHA-256；保留原件，不筛除或手动裁短动作。
- 预处理后共 441,140 帧，50 Hz；各段时长按 `(帧数−1)/50` 求和为 8,822 秒，即 2.450556 小时。
- 六字段 float32 张量共 790,522,880 字节，约 754 MiB；NPZ 文件及元数据共 790,865,840 字节。这是参考数据存储量，不是训练总内存需求。
- 按 mjlab 官方重采样规则，输出最后时刻取不超过源时长的 50 Hz 网格点；源末帧不落在网格上时，末尾不足一个控制步的区间不会新增一个非均匀采样点。没有把输入 CSV 截成上轮的一秒样本。
- 此处“完整”仅指该版本公开 G1 动作集，不等于已确认与论文的全部数据处理、评测协议完全一致。当前未定义正式训练/验证划分，也没有据此报告学习效果。

## 文件位置

| 内容 | 目录或文件 |
| --- | --- |
| 原始 CSV、每条来源记录 | `data/raw/lafan1/g1/` |
| 原始全量清单 | `data/raw/lafan1/source-index.json` |
| 转换后的 40 条 NPZ | `data/processed/lafan1/` |
| 动作库清单、版本及各条摘要 | `data/processed/lafan1/manifest.json` |
| 完整加载检查 | `output/lafan1/load-check.json` |
| 下载、转换和加载日志 | `output/lafan1/` |

`data/`、`output/` 均由 Git 忽略。数据集来源、版本和必要归属信息保存在本地来源记录；未上传或发布数据。

## 如何读一帧

```python
from supermimic_study.data import MotionLibrary

motions = MotionLibrary.from_directory(
    "data/processed/lafan1",  # 在仓库根目录执行
    device="cpu",
)
reference = motions.get_frames(clip_ids=[0], frame_ids=[0])
joint_angles = reference["joint_pos"]   # [1, 29]
```

动作编号对应 `motions.clip_names` 的顺序。帧编号是该动作内的编号，不能越过该段末尾进入下一条动作；调用越界时会报错。返回字典仍包含关节角/速度、身体位置/姿态和线/角速度六类字段，且是独立副本。

每条动作的源与转换摘要、时长、名称和关节/身体顺序均从清单校验。类中的完整张量只在当前 Python 进程存在；进程退出后内存释放，文件仍保存在磁盘，之后再调用上面的入口加载。

## 预处理与验证依据

2026-10-27起`motion_conversion.py`使用本地float64网格及归一化最短弧SLERP/近同nlerp，修复固定mjlab近同四元数提前返回问题；速度仍按输出网格差分约定计算，数据版本为preprocess_version=2。批量离线FK使用同一个mjlab场景编译出的G1模型，通过原生MuJoCo CPU位置/速度运动学计算；没有替换训练仿真器或进行物理rollout。输入关节顺序来自固定版本官方转换器，输出顺序来自mjlab G1 entity。

当前`data/processed/lafan1/`已是修复后version2，旧version1完整保留在`data/processed/lafan1-v1/`。新旧manifest不同，训练续训不能混用。独立全帧SciPy/差分对照、120个关键帧Warp重建及激活记录分别为`output/lafan1/resampling-v2-check.json`、`v2-load-check.json`、`v2-activation.json`；见 [集中核对](semantic-audit.md)。

以下51帧对拍是2026-10-26旧version1的历史证据：与上一轮mjlab Warp转换样本六字段逐项比较通过，最大绝对差约`7.2×10⁻⁷`。当前version2改由独立插值对照及重建检查验证，不再把旧转换器作为重采样正确性的oracle。全量加载检查包括：

1. 原始清单、转换清单和动作库均覆盖同一 40 条动作，逐条重新核对源 SHA-256 和 NPZ SHA-256。
2. 所有帧的数组形状、float32 类型、有限值和转换后的姿态四元数检查通过；时间和关节/身体元数据一致。
3. 全部 40 条文件分别用 mjlab 原生 `MotionLoader` 加载，并与整库索引结果比较通过。
4. 每条的首、中、末帧，共 120 帧，重新写入 mjlab 的 G1 并运行 Warp `forward`。六字段与准备结果在 `atol=rtol=5e-4` 下通过；实际最大绝对差约 `4.1×10⁻⁶`。
5. 所有动作末端的越界读取均拒绝。另有五项小型单元测试验证跨动作和段内索引、返回副本、异常格式/fps/摘要/数值、时长和目录边界。

这些验证支持参考文件与加载接口可用，不能证明策略能跟踪动作、动作动力学可执行，或 CUDA 训练路径已通过。

## 重复执行

```bash
.venv/bin/python scripts/prepare_lafan1.py --direct
.venv/bin/python scripts/check_lafan1.py
.venv/bin/python -m unittest discover -s tests -v
```

`--direct` 只改变该准备命令的网络路由，不修改系统代理。已存在的原始文件会核对后复用；已转换文件只有在来源、预处理版本和摘要一致时才复用，不会默默采用不明产物。如果中断恰好留下没有转换记录的最终 NPZ，准备命令会停止，需检查该单个产物后再恢复。

已有本地源清单可用`--local-only`完全离线校验/转换，仍核对40条源文件的大小和git blob摘要；无网络下载。重建时选新的`--output-dir`保留旧数据，不将不同version覆盖混写。
