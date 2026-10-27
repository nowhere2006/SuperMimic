# 公开验证摘要

2026-10-27 为首次 public 发布整理。这里只包含小型 JSON 摘要与本次重新执行的组件检查文本，不包含数据集、权重、真实转移或完整原生日志。原始运行仍保存在开发机 `output/`；以下材料是开发者留存证据的选摘，不是独立复跑证明。

| 文件 | 来源与含义 |
| --- | --- |
| [unit-tests.txt](unit-tests.txt) | 发布准备时重新执行 `.venv/bin/python -m unittest discover -s tests -v`，72 项通过；依赖警告中的本机安装路径替换为占位符 |
| [action-data-audit.json](action-data-audit.json) | 从历史集中核对回执复制，私人绝对路径改为仓库相对路径；增加摘要时间说明 |
| [training-runs-summary.json](training-runs-summary.json) | 从新库两轮闭环与续一轮的 report 选取状态、完整训练配置、依赖/硬件、数据和 Python 源码 SHA；删除本机路径。整理时重新核对两次运行全部 Python 源码与待发布代码相同 |
| [resampling-v2.json](resampling-v2.json) | 历史全 40 条 / 441,140 帧独立 SciPy 插值/速度对照结果 |
| [motion-load-v2.json](motion-load-v2.json) | 历史整库读取与 120 个关键帧 Warp 运动学重建结果 |
| [data-provenance.json](data-provenance.json) | 固定数据来源、预处理版本和原开发 manifest 摘要；发布准备时再次验证 40 个本地 NPZ 的 SHA |

历史运行发生在首次 Git commit 前，运行 report 中没有 commit SHA，以源码副本及逐文件 SHA 追溯。`AGENTS.md` 在公开准备时更新了发布授权和路径说明，因此不要把“所有历史文件等于当前源码”的旧审查结论外推到本次文档更新；这里重新核对的是 Python 源码。

数据 manifest 含转换耗时、运行依赖等元数据，换机器重新生成的 manifest SHA 可能不同；不能仅因不同就断言数据值不一致。来源 revision、预处理版本、源文件校验和转换记录共同定义可追溯性。训练续训入口仍要求与对应 checkpoint 的原 manifest 完全一致。

复核命令（已安装依赖并准备数据，在仓库根目录执行）：

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/check_resampling.py
.venv/bin/python scripts/check_lafan1.py
.venv/bin/python scripts/train.py --iterations 2
```

这些命令检查数据、局部计算及短程训练执行，不能代替长程学习、跟踪指标、多 seed 比较或论文结果复现。求解器警告及方法差异见 [已知问题](../known-issues.md)。
