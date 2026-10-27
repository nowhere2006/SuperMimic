# 动作、数据修复与集中核对计划

**Goal:** 分离策略/执行动作、修复SLERP，并核对闭环科学语义。

**Baseline:** main/unborn源码与旧数据manifest快照 `.cache/change-baselines/2026-10-27-action-data-audit/`。

**Primary Metric:** 定向数值/字段/时间/梯度及版本恢复检查；无学习效果指标。

**Budget:** Mac CPU，转换<=20分钟；动作修复旧库2轮，新库2轮+续1轮，各600秒。

---

1. 主助手修动作env/replay/WM采集边界；独立子agent只改motion_conversion和插值测试。
2. 单元/边界检查后，在旧processed库运行2轮，隔离动作修复证据。
3. 全40条写入新的version2目录，保留旧库；独立插值对照、库加载和MuJoCo/Warp关键帧检查通过后切换canonical默认路径。
4. 用新库2轮及恢复1轮，确认动作定义、数据manifest、loss校准/critic输入、记录与旧checkpoint保护。
5. 两个只读agent分别审核心算法及仿真数据合同，主助手复核实质问题，写集中核对矩阵和结果。
6. 运行source/config/hash及验证回执，更新文档。已知负结果不删除；不以smoke报告论文复现完成。
