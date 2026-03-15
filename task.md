# 毕业论文实现任务清单 (Task List)

本文档旨在规划“面向移动端的复杂环境中文手语识别研究 (Mobile-TFNet)”的详细实施步骤。遵循此路线图，可确保代码开发、实验验证与论文撰写同步推进。

## 第一阶段：环境搭建与基准复现 (基础准备)
**目标**：跑通现有代码，确保数据能正常读取，评价指标计算正确。

- [ ] **1.1 环境配置**
    - [ ] 安装 Python 3.8+, PyTorch 1.7+, torchvision, opencv-python, numpy。
    - [ ] 解决 `ctcdecode` 库在 Windows 下的安装问题（难点）。
        - *备选方案*：如果 `ctcdecode` 编译失败，暂时使用 PyTorch 自带的 `ctc_loss` 进行训练，推理时仅输出 Greedy Search 结果（取概率最大的索引）用于调试，后续在 Linux 服务器上进行 Beam Search 评估。
    - [ ] 确认显卡驱动与 CUDA 版本匹配。

- [ ] **1.2 数据准备 (CE-CSL / CSL-Daily)**
    - [ ] 下载数据集（建议先下载 5-10GB 的子集用于调试）。
    - [√ ] 运行 `DataProcessMoudle.py` 中的 `Word2Id` 函数，生成词表字典 `word2idx`。
    - [ ] 检查 `config.ini` 中的路径配置，确保 `trainDataPath`, `trainLabelPath` 指向正确位置。

- [ ] **1.3 基准代码跑通**
    - [ ] 使用原版 `TFNet` 配置 (ResNet34 + BiLSTM) 运行 `Train.py`。
    - [ ] 确保训练 1-2 个 epoch 不报错，观察 Loss 是否下降。
    - [ ] 验证 `WER.py` 的计算逻辑，确保能输出 WER 分数。

---

## 第二阶段：核心模型开发 (Mobile-TFNet)
**目标**：在 `Net.py` 中实现 MobileNetV2 + BiGRU 架构，替换原有重型网络。

- [ ] **2.1 引入 MobileNetV2 骨干**
    - [ ] 在 `Net.py` 头部导入 `torchvision.models`。
    - [ ] 在 `moduleNet` 类的 `__init__` 中新增 `if "LightTFNet" == self.moduleChoice:` 分支。
    - [ ] 加载预训练权重：`self.conv2d = models.mobilenet_v2(pretrained=True)`。
    - [√] **关键点**：修改 MobileNetV2 的分类头 (Classifier)，将 1280 维特征映射到 512 维（或与 hiddenSize 一致），作为时序网络的输入。
    - [√] **优化**：在 `forward` 中，先根据 `dataLen` 筛选出有效帧（去除 padding），再输入 MobileNetV2以减少计算量。
    - [√] 确保输入维度匹配：MobileNetV2 期望输入为 `(N, 3, H, W)`，需将 `(B, T, C, H, W)` 展平并按需提取有效帧进行重组。
    - [√] **技术实现细节**：
        - 移除 `SEN.py` 依赖，替换为 `torchvision.models.mobilenet_v2`。
        - 在 `LightTFNet` 分支下，使用 `models.mobilenet_v2(pretrained=True)`。
        - 自定义分类层：`nn.Linear(1280, 512)`。
        - 前向传播优化：利用 `dataLen` 计算有效帧索引，仅对有效帧执行卷积操作，随后使用 `torch.zeros` 补全 Padding，最后 Stack 回 `(Batch, Time, Feature)`。
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

- [ ] **2.2 实现 BiGRU 时序模块**
    - [√] **技术决策**：使用现有的 `BiLSTM.py` 作为基础，利用其通用的 RNN 封装能力（支持 `rnn_type='GRU'`）。
    - [√] **文件精简**：
        - `Transformer.py` 是用于 MSTNet 的，LightTFNet 不需要，应从 `TEST` 目录移除。
        - **修改** `Net.py`：移除对 `Transformer` 的引用，清理仅用于 `MSTNet` 或其他复杂模型的代码分支，保持代码库针对 LightTFNet 的专注与简洁。
    - [√] **BiGRU 实现细节**：
        - 在 `moduleNet` 初始化中，实例化 `BiLSTMLayer` 时指定参数 `rnn_type='GRU'`，`bidirectional=True`。
        - 确保 `hidden_size` 参数传递正确（通常为 512）。
    - [√] **前向传播对接**：
        - 确认 `BiLSTMLayer` 的输入为 `(Time, Batch, InputSize)`，输出为 `(Time, Batch, HiddenSize*2)`。
        - 在 `Net.py` 的 `LightTFNet` 分支中，调用 `self.temporal_model(x, lgt)`，其中 `lgt` 是序列长度列表（Tensor）。

- [√] **2.3 调整前向传播 (Forward) 并极度优化**
    - [√] **原项目痛点**：在原 `TFNet-main/Net.py` 的设计中（如 `MSTNet`），前向传播包含了庞大的 3D 或深层 2D 网络（如 `ResNet34`），且保留了巨量因 Batch 对齐而产生的 Padding 帧（黑边零向量），导致 `(B, T, C, H, W)` 维度的大部分无效计算。它同时输出了 `logProbs1` 到 `logProbs5` 用于复杂的知识蒸馏损失（KD Loss），使得模型极其笨重。
    - [√] **调整目的**：为了能够实现移动端部署（Mobile-TFNet），我们必须抛弃冗余的特征流和中间层约束分支，仅针对“具有实际信息的有效帧”进行运算，彻底压缩 FLOPs 并确保推理呈线性结构。
    - [√] **具体实现（`LightTFNet` 的 `forward` 逻辑重构）**：
        1. **展平输入并提取有效帧**：将输入序列 `(B, T, C, H, W)` 基于实际长度列表 `len_x` 用局部拼接的方式提取为真实含动作数据的 `valid_frames_list`。
        2. **绕过 Padding 计算的 CNN 推理**：应用 `MobileNetV2`（即 `self.conv2d`），大幅缩小前向传播过程中的空域特征吞吐量。
        3. **序列重建 (Sequence Reconstruction)**：计算完逐帧特征后，将一维数组基于各自的时间步长截取，并使用 `torch.zeros` 将短特征序列重新 Pad 还原至原最大时间步 `temp`，再通过 `torch.stack` 重组为 `(B, T, Feature)`。
        4. **时序与分类映射**：经过 `Conv1D` 压缩时序后，将 `(T, B, Feature)` 送入 `BiGRU`，不再像原版维护多个分类器输出，直接由一层全连接 `self.classifier` 映射到词表并提取出 `logProbs1`，简化了下游 `ctc_loss` 的计算。


---

## 第三阶段：鲁棒性增强与数据加载优化
**目标**：修改数据加载逻辑，引入抗干扰策略。

- [√] **3.1 增强数据预处理 (videoAugmentation.py)**
    - [√] 确认并完善 `ColorJitter` 类的实现。原有的 `ColorJitter` 需要兼容对视频帧序列（Tensor形状为 `(T, C, H, W)` 或 List of Numpy Arrays）的批量增强操作。同时保证同一个视频序列（同一Batch）内的颜色抖动参数一致，或者使用统一的 Transform。
    - [√] 确保与原有库函数（如 `RandomCrop`, `CenterCrop`, `RandomHorizontalFlip`）的数据流兼容，通常输入输出都是 Numpy list 或 Tensor。
    - [√] 保留并测试其他原有的数据增强方法，如 `WERAugment` (时序增强)。
    - [√] 编写独立的测试脚本：读取图片或生成包含几帧连续画面的假 Tensor，应用各种数据增强操作，验证无报错并观察输出形状和效果。

- [√] **3.2 集成到训练流 (Train.py)**
    - [√] 在 `Train.py` 的 `transform` 定义中，确保加入了 `videoAugmentation.ColorJitter` 和 `videoAugmentation.RandomCrop`。
    - [√] 设置合理的增强概率（如 `p=0.5`），避免训练数据过于失真导致难以收敛。

- [√] **3.3 优化 CTC Loss 计算**
    - [√] 检查 `Train.py` 中的 Loss 计算部分。
    - [√] 针对 `LightTFNet`，只保留最纯粹的 `ctc_loss`，去除原项目中针对多流架构设计的 KD Loss (知识蒸馏损失) 和 MSE Loss，简化训练目标，加速收敛。

---

## 第四阶段：实验训练与数据记录 (云端进行)
**目标**：获取论文所需的对比数据与图表。

- [ ] **4.1 实验 A：轻量化优势验证 (Mobile-TFNet vs ResNet-TFNet)**
    - [ ] 训练 `Mobile-TFNet` 模型，记录：
        - 最佳 WER (Word Error Rate)。
        - 训练一个 epoch 的耗时。
        - 模型权重文件 (.pth) 的大小 (MB)。
    - [ ] 训练基准 `ResNet` 模型 (或直接使用原论文数据)，记录相同指标。
    - [ ] **产出**：论文中的“模型性能对比表”。

- [ ] **4.2 实验 B：鲁棒性验证 (Augmentation vs No-Augmentation)**
    - [ ] 关闭数据增强，训练一个 Mobile-TFNet 版本。
    - [ ] 开启数据增强，训练最终版本。
    - [ ] 在验证集（选取部分光照复杂的样本）上测试两者的 WER。
    - [ ] **产出**：论文中的“消融实验分析图”。

- [ ] **4.3 实验可视化**
    - [ ] 绘制 Loss 下降曲线 (TensorBoard 或 Matplotlib)。
    - [ ] 截取几张测试视频的识别结果截图（显示：原视频帧 + 预测文本 + 真实文本）。

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
