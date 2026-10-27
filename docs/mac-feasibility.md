# Mac 调试可行性与运行记录

核查及实测日期：2026-10-26（Asia/Hong_Kong）。底座、G1接口、数据加载、初始化之后，已在项目隔离环境跑通小规模CPU训练闭环。**这验证执行/数据/梯度，不验证动作学习效果或CUDA。**

## 最新：CPU训练闭环

`scripts/train_smoke.py` 在2个CPU环境采集192条真实转移，进行两轮世界模型、policy、critic更新，保持完整MLP宽度、k8/H32。实际传感器噪声、每episode物理随机化、参考状态重置、任务奖励/失败终止和回报bin采样均已接入。FK/观测/真实转移奖励parity、冻结权重、非零有限梯度、固定GAE targets及检查点读取检查通过。

本次证据在 `output/training-smoke/cpu-loop-seed0-final/`。初次集成的terrain body映射失败和初次通过运行的求解器line-search20上限提示也已保留；修正FK子树并在训练profile显式使用line-search50后，最终运行无该上限提示。细节、CPU缩减及尚未验证项见 [训练闭环](training-loop.md)。以下章节保留各历史阶段的检查边界。

## 参考接口与模型初始化

后续授权的初始化检查已在本机通过：两个 G1 环境读取真实观测历史及行走/舞蹈参考序列，policy 产生动作并执行 3 步；动力学核心与 critic 的前向输出有限，模型权重未更新。初始权重和接口记录保存在 `output/initialization/`，细节及与论文未明确项的区分见 `docs/initialization-contract.md`。奖励与终止头、完整跟踪任务及训练尚未完成。

## 完整 LAFAN1 加载实测

用户随后明确授权完整 G1 LAFAN1。40 条动作已在本机完成预处理和整库 CPU 加载，共 441,140 帧、50 Hz、约 2.45 小时；参考张量约 754 MiB。全部 40 条分别通过 mjlab `MotionLoader` 读取，并抽取首/中/末共 120 帧进行 G1 Warp 运动学复核。详细边界和证据见 `docs/lafan1-loading.md` 与 `output/lafan1/load-check.json`。

这证明参考数据层可在本机使用，尚未接入参考目标、跟踪奖励或学习模型。原始下载在当前代理上遇到 TLS 连接错误后，使用只作用于准备进程的直连成功；没有修改系统网络设置。

## G1 基础接口实测

`scripts/check_g1.py` 已在两个 CPU 实例上通过检查：29 个关节与 29 维动作映射、默认姿态目标、左肘动作缩放、8 个控制步的有限状态、独立状态快照、选择性及全量重置、手动超时状态保留。控制步长为 0.02 s，物理步长 0.005 s。

诊断环境没有参考动作、跟踪奖励、随机化或策略网络；这不是论文训练配置。当前通过的选择性重置检查只覆盖本次 G1 诊断状态和配置，不能推广为上游 CPU 后端所有重置字段都已正确。

模块说明见 `docs/g1-interface.md`，证据见 `output/g1-interface.json` 和 `output/g1-interface.log`。运行包版本未改变，本项目包切换为 `src/` 布局的 editable 安装。

## 最新实测结果

- 项目环境：`.venv`，使用用户已有的独立 Python 3.12.12 建立；没有将 Codex 内置 Python 作为项目解释器。
- mjlab 来自下文固定提交，安装归档 SHA-256 为 `c6da219f4ba25551aee34558b2364500174dfc6af4a951f6955f9a770383cf5a`。
- 实际运行包版本：mjlab 1.6.0、MuJoCo 3.16.0、MuJoCo Warp 3.16.0、Warp 1.18.0、PyTorch 2.14.1、NumPy 2.5.4。
- `pip check` 通过；运行包版本快照在 `requirements/macos-arm64-py312.txt`，完整环境清单在 `output/environment.json`。快照限定当前平台，未锁定全部构建工具与下载哈希。
- 两个 CPU 仿真实例、每实例 16 步、每步 0.005 s：位置与速度有限；方块高度由 1.0 m 变为约 0.966646 m；位置状态形状为 `[2,7]`，速度状态为 `[2,6]`。
- 与相同场景、参数和初始状态的原生 MuJoCo 轨迹比较通过（位置状态 `atol=1e-4, rtol=1e-4`），全量重置后状态恢复检查通过。
- PyTorch 已知标量导数检查通过，所得梯度 `[4,-6]`；这不是世界模型或策略训练梯度验证。
- CPU 仿真未使用 CUDA graph。当前检查不覆盖控制输入、G1、复杂接触、选择性重置或正式训练。
- 初次运行触发字体缓存路径不可写提示，已把该检查的 XDG/Matplotlib/Warp 缓存放到项目 `.cache/` 并重新运行通过；仍有上游 `torch.jit.script` 弃用提示，未改第三方代码。

复查命令（在项目根目录）：

```bash
.venv/bin/python scripts/check_platform.py --device cpu --num-envs 2 --steps 16
```

最新结构化报告：`output/platform-check.json`；修复缓存路径后的完整日志：`output/platform-check-clean.log`；首次运行日志：`output/platform-check.log`。这些生成文件保留在本地并由 Git 忽略。

下载时发现系统及进程均已使用 `127.0.0.1:7897` 代理；该源码归档经代理约 67 KB/s，短时直连测试约 2 MB/s。仅对本次安装进程绕过相关下载域名的代理后，源码归档以约 3.1 MB/s 下载完成，没有改变系统 VPN/代理设置。

## 结论

这台 Mac 有可用于学习和组件调试的条件；mjlab 官方提供 macOS 的 CPU 评估路径。不能原样把第三方训练入口当作 Mac 训练入口：其中存在无条件 CUDA 调用。正式训练应使用 Linux + NVIDIA GPU；具体租卡规格和预算尚未确定。

| 工作内容 | 当前判断 | 证据与限制 |
| --- | --- | --- |
| 阅读、推导、源码追踪 | 可立即进行 | 本轮已核查论文方法部分及上游源码 |
| PyTorch CPU 反向传播 | 最小标量检查已通过 | 尚未验证运动学、世界模型损失或策略 BPTT |
| mjlab CPU 仿真与 G1 基础交互 | 简单场景、G1 动作映射/短时步进/重置已通过 | 参考动作跟踪、回放、长时稳定性及训练仍需验证 |
| 第三方完整训练入口直接改为 CPU | 不能直接视为可用 | 训练循环无条件调用 torch.cuda.synchronize()；还需逐项检查其他 CUDA 假设 |
| 论文规模训练及耗时结论 | 本机不适用 | 官方不推荐 macOS 训练；Apple GPU 不能运行 CUDA 路径 |

当前实测证明上述简单场景及 G1 诊断交互可以在本机通过 mjlab CPU 路径运行；没有训练闭环、学习效果或论文性能结论。

## 最初的本机检查（安装之前）

- MacBook Air，Apple M4，10 核 CPU、10 核 Apple GPU，24 GB 统一内存。
- macOS 15.7.7，arm64。
- 当前工作目录最初为空，尚无项目代码、Git 仓库或项目虚拟环境。
- Shell 默认 Python：3.13.7；Codex 自带检查用 Python：3.12.14。
- 对上述两个解释器用 importlib.util.find_spec 检查：numpy 可找到，torch、mujoco、warp、mjlab 均未找到。这不代表整台机器的所有其他环境都没有这些包。
- 第三方仓库要求 Python >=3.12,<3.13，因此不能直接使用当前 shell 默认的 Python 3.13；正式项目环境尚未建立。
- 此初查时尚未安装包；后续安装及实测见本文开头。整个过程中未下载动作数据或执行训练。

## 固定的核查快照

### mjlab

本轮 main 提交为 `8bb2bc9e187797689ec82c78c35019afd2fd0376`，也正是第三方仓库固定的 mjlab 提交。此处仅记录核查快照，尚未为本项目选定依赖锁文件。

- [官方 FAQ](https://github.com/mujocolab/mjlab/blob/8bb2bc9e187797689ec82c78c35019afd2fd0376/docs/source/faq.rst)：macOS 通过 MuJoCo Warp 使用 CPU；评估可用但慢，训练不推荐。
- [安装要求](https://github.com/mujocolab/mjlab/blob/8bb2bc9e187797689ec82c78c35019afd2fd0376/docs/source/installation.rst)：正式训练平台为 Linux + NVIDIA GPU。
- [架构](https://github.com/mujocolab/mjlab/blob/8bb2bc9e187797689ec82c78c35019afd2fd0376/docs/source/architecture_overview.rst)：仿真层与定义强化学习任务的 manager 层分离。
- [Simulation](https://github.com/mujocolab/mjlab/blob/8bb2bc9e187797689ec82c78c35019afd2fd0376/src/mjlab/sim/sim.py)：非 CUDA 设备不捕获 CUDA graph，step 可走直接调用；不能仅凭架构页强调 GPU 就断言完全不支持 CPU。
- [play 入口](https://github.com/mujocolab/mjlab/blob/8bb2bc9e187797689ec82c78c35019afd2fd0376/src/mjlab/scripts/play.py)：无可用 CUDA 时默认选择 cpu，并支持零动作与随机动作策略。

### 第三方参考实现

本轮核查快照：`d9be385b3eeb13a638fb33fb200d3bfd9be9839f`。这是独立复现仓库，不是论文作者代码。

- [pyproject.toml](https://github.com/CuboiLeo/SuperMimic-Reproduced/blob/d9be385b3eeb13a638fb33fb200d3bfd9be9839f/pyproject.toml)：Python 3.12 范围、固定 mjlab 提交；Darwin 依赖有单独 marker，但这不构成完整训练兼容声明。
- [training/loop.py](https://github.com/CuboiLeo/SuperMimic-Reproduced/blob/d9be385b3eeb13a638fb33fb200d3bfd9be9839f/src/supermimic/training/loop.py)：175、189、194 行有无条件 CUDA 同步，构成 CPU 训练入口的明确障碍。
- [README](https://github.com/CuboiLeo/SuperMimic-Reproduced/blob/d9be385b3eeb13a638fb33fb200d3bfd9be9839f/README.md)：明确声明部分成绩使用额外 9 维锚点相对观测，偏离论文口径；不能把这部分成绩作为忠实复现成功标准。仓库成绩为作者自述，本项目未复跑。

## 论文依据与学习边界

用户提供的 PDF 共 19 页，标题为 SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model，首页标记 arXiv:2610.28480v1（2026-10-20）。

原件：用户提供的 `SuperMimic(1).pdf`。公开仓库省略私人收件路径，不分发 PDF；下方摘要用于识别版本。

SHA-256：`fe213017ab062f4acc3965da89abd3c29969163e144a2628a662b8ce46691009`。未修改原件；已提取文本并查看首页、方法第 4–5 页的渲染图。

第 3 节及 Figure 2 的核心循环是：当前策略采集转移 → 用 replay 中的窗口拟合世界模型 → 固定世界模型参数，通过想象轨迹的 BPTT 更新策略并训练 critic。论文默认拟合窗口 k=8、想象长度 H=32。这些是后续要理解和核对的机制，不是当前已经实现或运行的内容。

本轮不决定完整复现实验矩阵。用户随后已确认：以架构与完整数据流理解为主，助手可以直接实现算法和代码，按用户要求逐模块推进；独立组织本项目模块，以 mjlab 为底座，参考或复用第三方实现。

阶段顺序已确定为：先在 MacBook 上跑通最小训练闭环，检查数据与梯度，不要求学会动作；再迁移至用户租用的 NVIDIA GPU，尝试 LAFAN1 训练。当前完成底座和 G1 基础接口实测，尚未达到最小训练闭环标准。
