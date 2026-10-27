# SuperTrack 分组损失适配计划

**Goal:** 验证首次误差校准能平衡六组 L1 初期贡献，并保持训练闭环及恢复正确。

**Baseline:** `.cache/change-baselines/2026-10-27-supertrack-loss/` 的原始71维MSE与本轮保留的 `raw_mse` 模式。

**Primary Metric:** 未触底组校准贡献为1；几何/反传/模块隔离/checkpoint恢复检查通过；不按回报挑选模型。

**Budget:** Mac CPU2环境，一次2轮及一次续1轮，每次600秒上限；无正式GPU训练。

---

1. 新增 `src/supermimic_study/training/model_loss.py`：六组几何误差、稳定相对旋转log、初始分母一次校准与固定权重、raw基线入口。
2. 修改 `training/config.py`、`training/loop.py`、`scripts/train_smoke.py`：保存损失模式/权重，使用首次已采样窗口校准，输出分项贡献与校准来源，严格恢复校准状态。
3. 新增 `tests/test_model_loss.py` 并补充checkpoint guard测试；运行 `.venv/bin/python -m unittest discover -s tests -v`，包含几何已知值、符号不变性、梯度差分、校准冻结/下限、checkpoint安全往返与旧配置拒绝。
4. 用 `.venv/bin/python scripts/train.py --iterations 2 --max-seconds 600 --output output/training-runs/supertrack-loss-check` 接通真实采集、拟合与actor/critic更新，保存既有全部记录。
5. 从其完成checkpoint续1轮，新输出目录 `output/training-runs/supertrack-loss-resume`；核查scale完全一致、校准event不重复。
6. 写验证回执及文档，明确L1/初始平衡是本轮方法选择，旧raw loss数值不能与新loss横比。失败不启动更长训练；仅用改前精确文件作人工可审查回退，不覆盖其他文件。
