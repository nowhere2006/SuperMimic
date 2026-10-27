# 只看一段 LAFAN1 参考动作

2026-10-26：本次只检查一条已重定向到 G1 的行走轨迹，并转换开头约一秒。尚未建立正式动作库或训练数据管线，没有训练、replay buffer 或远程上传。

后续状态：用户之后另行授权并完成了[完整 40 条 G1 LAFAN1 的准备与加载验证](lafan1-loading.md)。下文保留本次一秒样本的历史检查范围。

## 这一帧是什么

来源为 [LAFAN1_Retargeting_Dataset 的 G1/walk1_subject1.csv](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/blob/ce1572906efe6157840e8474d5a0d7aa87481e74/g1/walk1_subject1.csv)。一行对应一个参考时刻，共 36 个数：

| 列（从 0 开始） | 含义 |
| --- | --- |
| 0–2 | 根部位置 x,y,z |
| 3–6 | 根部姿态四元数 x,y,z,w |
| 7–35 | 按约定顺序排列的 29 个关节角 |

第一帧根部位置约为 `[0.000480, -0.000023, 0.796553]`，左肘参考角为 `1.405656 rad`。这是数据给出的参考姿态，不是刚才诊断环境的默认姿态，也不是策略的动作输出。

原始 CSV 保留在 `data/raw/lafan1/g1/walk1_subject1.csv`，大小 2,670,866 字节。数据集提交为 `ce1572906efe6157840e8474d5a0d7aa87481e74`；文件 SHA-256 为 `68c7887438e94f1f3d9ebacd3a8ce59251a22c7b05129a0f5d7cacaedc48c28d`，相邻 `.source.json` 保存来源记录。没有下载全量数据集。

## 转换增加什么

```text
G1 参考姿态 CSV
    ↓ 对齐时间、计算速度、根据机器人结构计算各部位状态
mjlab 可以读取的参考轨迹 NPZ
```

本次使用原始第 1–31 帧，30 FPS，覆盖 t=0 到 t=1 秒；转换到 50 FPS 后共 51 帧，同样包括两个端点。

| NPZ 字段 | 本次形状 | 内容 |
| --- | --- | --- |
| `joint_pos` | `[51,29]` | 每帧的关节角 |
| `joint_vel` | `[51,29]` | 根据相邻时刻计算的关节角速度 |
| `body_pos_w` | `[51,30,3]` | G1 的 30 个身体部位在每帧的位置 |
| `body_quat_w` | `[51,30,4]` | 每个身体部位的姿态，顺序 w,x,y,z |
| `body_lin_vel_w` | `[51,30,3]` | 每个部位的线速度 |
| `body_ang_vel_w` | `[51,30,3]` | 每个部位的角速度 |

根部和关节轨迹的重采样及速度计算使用固定版本 mjlab 的 `csv_to_npz.MotionLoader`。关节输入顺序读取自该版本官方转换器，并通过关节名称映射到 G1。每帧将参考状态写入相同 G1 模型并调用 `sim.forward()` 计算身体状态；没有调用物理步进来验证机器人能否执行这段动作，也没有运行官方脚本的上传流程。

## 验证到哪里

已检查数组形状、有限值、身体姿态四元数范数、关节首尾帧与输入对应、首帧根部位置及四元数顺序。生成结果已用 mjlab 运行时的 `MotionLoader` 重新加载，读到 51 帧，关节与身体位置数组一致。

这些检查证明该一秒样本可用于当前版本的格式接入验证，不证明动作可被动力学准确执行，也不构成完整动作库、目标生成或跟踪任务的验证。

本地结果：

- 样本 NPZ：`output/motion-inspection/walk1_subject1-first-1s.npz`
- 结构化报告：`output/motion-inspection/inspection.json`
- 本次诊断脚本：`output/motion-inspection/inspect_one_clip.py`
- 运行日志：`output/motion-inspection/inspection.log`

运行命令为 `.venv/bin/python output/motion-inspection/inspect_one_clip.py`。该脚本是此次单片段诊断记录，保留在被 Git 忽略的 `output/` 中；尚未提取成正式通用转换模块。

参考轨迹表示希望机器人怎么动；之后 replay buffer 保存策略控制机器人时实际发生的转移。当前只处理前者。
