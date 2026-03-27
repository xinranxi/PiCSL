# 毕业论文实现任务清单 (Task List)

本文档旨在规划“面向移动端的复杂环境中文手语识别研究 (Mobile-TFNet)”的详细实施步骤。遵循此路线图，可确保代码开发、实验验证与论文撰写同步推进。

## 第一阶段：环境搭建与基准复现 (基础准备)
**目标**：跑通现有代码，确保 CE-CSL 数据能正常读取，评价指标计算正确。

- [√] **1.1 环境配置**
    - [√] 安装 Python 3.8+, PyTorch 1.7+, torchvision, opencv-python, numpy。
    - [√] 解决 `ctcdecode` 库在 Windows 下的安装问题（已采用 PyTorch 自带的 `ctc_loss` 进行训练）。
    - [√] 确认显卡驱动与 CUDA 版本匹配。

- [√] **1.2 数据准备 (CE-CSL)**
    - [√] 下载 CE-CSL 数据集（train/dev/test 子集）。
    - [√] 运行 `DataProcessMoudle.py` 中的 `Word2Id` 函数，生成词表字典 `word2idx`（仅支持 CE-CSL）。
    - [√] 检查 `config.ini` 中的 CE-CSL 路径配置，确保指向正确位置。

- [√] **1.3 代码流程验证**
    - [√] 使用 `LightTFNet` 配置 (MobileNetV2 + BiGRU) 运行 `Train.py`。
    - [√] 确保训练 1-2 个 epoch 不报错，观察 Loss 是否下降。
    - [√] 验证 `WER.py` 的计算逻辑，确保能输出 WER 分数。

---

## 第二阶段：核心模型开发 (LightTFNet)
**目标**：验证 MobileNetV2 + BiGRU 架构正确性，确保推理效率和识别准确率。

- [√] **2.1 引入 MobileNetV2 骨干**
    - [√] 在 `Net.py` 头部导入 `torchvision.models`。
    - [√] 在 `moduleNet` 类的 `__init__` 中已实现 `if "LightTFNet" == self.moduleChoice:` 分支。
    - [√] 加载预训练权重：`self.conv2d = models.mobilenet_v2(pretrained=True)`。
    - [√] **关键点**：MobileNetV2 分类头已改为 `nn.Identity()`，保留 1280 维原始特征。维度对齐由后续 Conv1D 处理，避免随机初始化线性层破坏预训练权重（"灾难性遗忘"）。
    - [√] **优化**：在 `forward` 中，先根据 `dataLen` 筛选出有效帧（去除 padding），再分块（`chunk_size=64`）输入 MobileNetV2 以防止 OOM，并减少冗余计算。
    - [√] 确保输入维度匹配与归一化：MobileNetV2 期望输入为 `(N, 3, H, W)` 且范围 `[0, 1]`，已在 `forward` 中做数据流管道适配与 ImageNet 标准化。
    - [√] **技术实现细节**：
        - 移除 `SEN.py` 依赖，使用 `torchvision.models.mobilenet_v2(pretrained=True)`。
        - 分类层改为 `nn.Identity()`，保留特征完整性，避免灾难性遗忘。
        - 前向传播优化：
          1. 提取有效帧：根据 `dataLen` 过滤出实际含有信息的帧。
          2. 分块 CNN 推理：`chunk_size=64`，逐块调用 MobileNetV2。
          3. 序列重建：用 `torch.zeros` 补全 padding，Stack 回 `(B, T, 1280)`。
          4. 数据预处理：先 `(x+1)/2` 映射到 `[0,1]`，再按 ImageNet mean/std 标准化。
- [√] **验证**：
      - 需要将以下文件从 `TFNet-main` 复制到 `TEST`，以便单独运行模型：
        - `Net.py`（已复制并修改）
        - `Module.py`, `BiLSTM.py`, `Transformer.py`。
        - **注意**：`SEN.py` 已从 TEST 目录移除，LightTFNet 不依赖此文件。
      - 运行全流程验证脚本 `verify_pipeline.py`：
        ```bash
        cd D:\Graduation_Thesis\TEST
        python verify_pipeline.py
        ```
        期望看到“Input Video Batch Shape: ... Output Shape: ... Pipeline verification successful!”。

- [√] **2.2 实现 BiGRU 时序模块**
    - [√] **技术决策**：使用现有的 `BiLSTM.py` 作为基础，利用其通用的 RNN 封装能力（支持 `rnn_type='GRU'`）。
    - [√] **文件精简**：
        - `Transformer.py` 是用于 MSTNet 的，LightTFNet 不需要，`TEST` 目录下已移除。
        - `Net.py`：已移除对 `Transformer` 和 `SEN` 等的导入与引用，清理了仅用于 `MSTNet` 等复杂模型的分支，保持代码专注于 CE-CSL + LightTFNet。
    - [√] **BiGRU 实现细节**：
        - 在 `moduleNet` 初始化中，已实例化 `BiLSTMLayer` 时指定参数 `rnn_type='GRU'`，`bidirectional=True`。
        - `hidden_size` 参数正确传递（默认为 1024，可调整）。
    - [√] **前向传播对接**：
        - `BiLSTMLayer` 的输入为 `(Time, Batch, InputSize)`，输出为 `(Time, Batch, HiddenSize*2)`。
        - 在 `Net.py` 的 `LightTFNet` 分支中，已调用 `self.temporal_model(x, lgt)`，其中 `lgt` 是序列长度 Tensor。

- [√] **2.3 调整前向传播 (Forward) 并极度优化**
    - [√] **原项目痛点**：在原 `TFNet-main/Net.py` 的设计中（如 `MSTNet`），前向传播包含了庞大的 3D 或深层 2D 网络（如 `ResNet34`），且保留了巨量因 Batch 对齐而产生的 Padding 帧（黑边零向量），导致 `(B, T, C, H, W)` 维度的大部分无效计算。同时输出了 `logProbs1` 到 `logProbs5` 用于知识蒸馏，模型极其笨重。
    - [√] **调整目的**：为了落地移动端部署（LightTFNet），已抛弃冗余的特征流与中间约束分支，仅针对"有效帧"进行运算，压缩 FLOPs 确保线性推理结构。
    - [√] **具体实现（`LightTFNet` 的 `forward` 逻辑重构）**：
        1. **展平输入并提取有效帧**：将输入序列 `(B, T, C, H, W)` 基于 `len_x` 提取为 `valid_frames_list`。
        2. **分块 CNN 推理**：应用 `MobileNetV2`（`chunk_size=64`），避免 OOM，缩小空域特征吞吐量。
        3. **序列重建**：逐帧特征重新 Pad 还原至原最大时间步 `temp`，通过 `torch.stack` 重组为 `(B, T, 1280)`。
        4. **时序与分类映射**：`Conv1D` 压缩时序后，`(T, B, hidden_size)` 送入 `BiGRU`，单一分类器 `self.classifier` 输出 `logProbs1`，简化 CTC Loss 计算。


---

## 第三阶段：鲁棒性增强与数据加载优化
**目标**：修改数据加载逻辑为 CE-CSL 单数据集模式，保留完整的数据增强策略。

- [√] **3.1 增强数据预处理（CE-CSL 专用）**
    - [√] 从 TFNet-main 风格对齐增强策略：保留 **RandomCrop、RandomHorizontalFlip、ColorJitter、TemporalRescale** 完整组合。
    - [√] 删除 CE-CSL 之外的数据集分支（RWTH、CSL-Daily）。
    - [√] `ColorJitter` 实现确保适用视频帧序列，参数遵循 TFNet-main 标准（`p=0.5` 概率）。
    - [√] 验证与 `RandomCrop、CenterCrop、RandomHorizontalFlip` 的数据流兼容性。

- [√] **3.2 集成到训练流（CE-CSL 适配）**
    - [√] `Train.py` 已强制 CE-CSL 仅模式：配置检验入口直接报错非 CE-CSL。
    - [√] 数据增强 `transform` 已恢复 TFNet-main 完整风格（RandomCrop、RandomHorizontalFlip、ColorJitter、TemporalRescale）。
    - [√] 若 dataSetName != "CE-CSL"，`DataProcessMoudle.py` 中的 `Word2Id` 和 `MyDataset` 直接报错，防止误用其他分支。

- [√] **3.3 优化 CTC Loss 计算与梯度管理**
    - [√] `Train.py` 中的 CTC Loss 已简化为纯 `nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)`。
    - [√] 移除了知识蒸馏（KD Loss）和其他辅助损失，专注于 CE-CSL 的核心任务。
    - [√] 加入梯度累积（`gradAccumSteps`）、梯度裁剪（`maxGradNorm`）、backbone 冻结策略（`freezeBackboneEpochs`）、差分学习率（`backboneLrScale`），提升训练稳定性。
    - [√] 增加诊断指标：blank_prob、pred_len 分布、invalid_ctc 计数。

---

## 第四阶段：实验训练与数据记录 (云端进行)
**目标**：获取论文所需的对比数据与图表。

- [ ] **4.1 实验 A：轻量化效能验证 (LightTFNet on CE-CSL)**
    - [ ] 训练 `LightTFNet` 模型到收敛，记录：
        - 最佳 dev WER (Word Error Rate) 及其对应 epoch。
        - 最佳 test WER。
        - 训练一个 epoch 的平均耗时（秒）。
        - 模型权重文件 (.pth) 的大小 (MB)。
    - [ ] 对比与参考 ResNet-34 + BiLSTM 模型的性能（从 TFNet-main 或论文数据），包括参数量、FPS、WER。
    - [ ] **重点**：虽然 WER 可能持平或略高，但参数量应显著下降（~70-80%），推理速度提升 2-3 倍。
    - [ ] **产出**：论文中的"轻量化模型性能对比表"。

- [ ] **4.2 实验 B：增强策略消融实验 (Full Augmentation vs Conservative)**
    - [ ] **实验 B1**：关闭颜色抖动+水平翻转，仅保留随机裁剪和时序缩放，训练一个 LightTFNet 版本。
    - [ ] **实验 B2**：保留完整增强（ColorJitter + Flip + RandomCrop + TemporalRescale），训练最终版本。
    - [ ] 在 dev set 和 test set 上对比两者的 WER，重点关注光照/背景复杂的样本。
    - [ ] **预期结论**：完整增强应显著降低 WER（尤其在复杂环境），证明鲁棒性增强的必要性。
    - [ ] **产出**：论文中的"消融实验对比表及分析"。

- [ ] **4.3 诊断分析与可视化**
    - [ ] 绘制 Loss 下降曲线（TensorBoard 或 Matplotlib，包括 train/dev 曲线）。
    - [ ] 绘制 WER 曲线（按 epoch）。
    - [ ] 分析关键诊断指标：
        - `blank_prob_mean` 随 epoch 的变化（应逐步降低）。
        - `pred_len` 分布（应与 `lgt` 接近）。
        - `invalid_ctc` 比例（应接近 0）。
    - [ ] 截取几张测试视频的识别结果截图（原视频帧 + 预测手语文本 + 真实标注 + WER 分数）。

---

## 第五阶段：论文撰写与答辩准备
**目标**：将代码成果转化为学术文档。

- [ ] **5.1 撰写第三章：系统设计**
    - [ ] 绘制 Mobile-TFNet 整体架构图 (Visio/PPT)。
    - [ ] 绘制 MobileNetV2 倒残差结构图。
    - [ ] 描述数据预处理流程。

- [ ] **5.2 撰写第四章：实验分析**
    - [ ] 整理实验数据，制作三线表。
    - [ ] 对比分析参数量、FPS 和 WER。
    - [ ] 重点强调：虽然 WER 可能略高或持平，但**计算效率提升巨大**，符合移动端定位。

- [ ] **5.3 答辩演示 (Demo)**
    - [ ] 编写 `demo.py`：加载训练好的模型，读取本地一个视频文件，输出识别文字。
    - [ ] 录制演示视频，防止答辩现场环境配置出问题。

---
*注：遇到报错优先检查 `config.ini` 中的路径和 `Net.py` 中的维度匹配问题 (Shape Mismatch)。*
