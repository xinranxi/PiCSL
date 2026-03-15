# 面向移动端的复杂环境中文手语识别研究 (Mobile-TFNet)

## 1. 研究背景与意义
针对现有手语识别模型（如基于ResNet/3D-CNN架构）参数量大、计算资源消耗高、难以在移动端或嵌入式设备部署的问题，以及复杂环境下（光照变化、杂乱背景）识别率下降的痛点，本项目提出了一种**轻量级且高鲁棒性**的中文手语识别模型 —— **Mobile-TFNet**。

本研究旨在在保证识别准确率的前提下，显著降低模型参数量与推理延迟，使其更适合在资源受限的设备（如手机、平板、嵌入式开发板）上运行，推动无障碍沟通技术的落地应用。

## 2. 核心技术创新点
1.  **架构轻量化设计 (Lightweight Architecture)**：
    使用 **MobileNetV2** 作为视觉特征提取的骨干网络，替代传统的 ResNet-34。利用其核心的**深度可分离卷积 (Depthwise Separable Convolution)** 和 **倒残差结构 (Inverted Residuals)**，在大幅降低模型参数量与计算复杂度的同时，保持了强大的特征提取能力。

2.  **高效时序建模 (Efficient Temporal Modeling)**：
    采用 **BiGRU (双向门控循环单元)** 替代传统的 LSTM 或复杂的 Transformer 结构。GRU 相比 LSTM 少了一个门控单元，参数更少，收敛更快，更适合轻量级任务，同时双向结构保证了对上下文语义的完整捕捉。

3.  **环境鲁棒性增强 (Environmental Robustness)**：
    针对 RGB 摄像头易受环境干扰的问题，在数据预处理阶段引入 **ColorJitter (颜色抖动)** 和 **RandomCrop (随机裁剪)** 等数据增强策略。通过模拟真实场景下的光照波动、色彩偏差与背景噪声，强制模型关注手部运动特征，提升模型在复杂环境下的泛化能力。

4.  **端到端单流架构 (End-to-End Single Stream)**：
    摒弃了复杂的多流（RGB + 光流 + 骨架点）融合策略，采用端到端的 **RGB 单流 + CTC (Connectionist Temporal Classification)** 联合训练。这种设计避免了预先提取光流或骨架点的高昂计算开销，简化了系统部署流程。

## 3. 技术路线与系统架构

### 3.1 整体架构流程
```mermaid
graph LR
    A[视频输入] --> B[数据增强模块]
    B --> C[空间特征提取 (MobileNetV2)]
    C --> D[维度对齐与降维 (FC Layer)]
    D --> E[时序建模 (BiGRU)]
    E --> F[序列对齐与分类 (CTC Loss)]
    F --> G[文本结果输出]
```

### 3.2 模块详解

#### A. 数据增强模块 (Data Augmentation)
*   **目的**：解决复杂背景与光照干扰，提升模型“抗噪”能力。
*   **关键技术**：
    *   **RandomCrop (随机裁剪)**: 在视频帧中随机裁剪出 224x224 区域，模拟摄像头抖动与不同拍摄距离。
    *   **ColorJitter (颜色抖动)**: 随机调整视频帧的亮度(Brightness)、对比度(Contrast)、饱和度(Saturation)，模拟不同光照环境（如逆光、暗光）。
    *   **TemporalRescale (时序重缩放)**: 随机丢帧或插帧，模拟不同手语使用者的语速差异。

#### B. 空间特征提取 (Spatial Feature Extraction)
*   **模型选型**：**MobileNetV2** (预训练于 ImageNet)。
*   **输入处理**：输入 RGB 视频帧序列 $X = \{x_1, x_2, ..., x_T\}$，维度 `(B, T, C, H, W)`。将 Batch 和 Time 维度合并，作为 `(B*T, C, H, W)` 输入 CNN。
*   **特征提取**：
    1.  图像经过 MobileNetV2 的 19 个瓶颈层 (Bottleneck Layers)。
    2.  利用深度可分离卷积提取空间特征。
    3.  输出特征图经过全局平均池化 (GAP)。
    4.  原始输出维度：`(B*T, 1280)`。

#### C. 时序建模 (Temporal Modeling)
*   **维度适配**：
    *   MobileNetV2 输出的 1280 维特征对于时序模型来说过大。
    *   引入全连接层 (Linear) + Dropout，将维度从 `1280` 降维至 `512`。
*   **BiGRU 网络**：
    *   **输入**：`(B, T, 512)`。
    *   **结构**：2层堆叠的双向 GRU。
    *   **输出**：隐层状态序列 $H = \{h_1, h_2, ..., h_T\}$，包含上下文语义信息。

#### D. 序列解码 (Sequence Decoding)
*   **损失函数**：**CTC Loss**。
    *   解决输入视频帧序列长度与输出文本序列长度不一致的问题（无需对齐标注）。
*   **解码策略**：训练时计算 Loss，推理时使用 Greedy Search 或 Beam Search 生成最终文本句子。

## 4. 实验设计与评估

### 4.1 数据集
*   **数据集名称**：CE-CSL (Complex Environment Chinese Sign Language) 或 CSL-Daily。
*   **特点**：包含日常高频词汇与句子，涵盖多种复杂背景（如超市、街道等）。

### 4.2 评价指标
1.  **WER (Word Error Rate)**：词错误率（核心准确性指标，越低越好）。
    $$ WER = \frac{S + D + I}{N} $$
    (S: 替换, D: 删除, I: 插入, N: 总词数)
2.  **FPS (Frames Per Second)**：每秒处理帧数（实时性指标，越高越好）。
3.  **Params (Parameters)**：模型参数量（轻量化指标，越小越好）。

### 4.3 对比实验方案 (论文核心数据来源)

| 实验组别 | 骨干网络 (Backbone) | 时序模型 (Temporal) | 增强策略 (Augmentation) | 预期优势 |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (基准)** | ResNet-34 / ResNet-18 | BiLSTM | 基础翻转/裁剪 | 准确率尚可，但参数量大，推理慢 |
| **Ours (本方案)** | **MobileNetV2** | **BiGRU** | **ColorJitter + Crop** | **参数量下降 ~80%，FPS 提升 ~3倍，WER 相当或略优** |
| **Ablation (消融)** | MobileNetV2 | BiGRU | 无 ColorJitter | 在复杂背景测试集上 WER 显著上升 (证明增强策略有效) |

## 5. 快速开始 (复现指南)

### 5.1 环境配置
确保安装以下依赖库：
```bash
pip install torch torchvision numpy opencv-python
```

### 5.2 配置文件修改
修改 `params/config.ini` 文件，指定模型为 `LightTFNet`：
```ini
[Params]
moduleChoice = LightTFNet   ; 选择轻量化模型分支
dataSetName = CE-CSL        ; 数据集名称
hiddenSize = 1024           ; 或根据显存调整为 512
```

### 5.3 启动训练
在微调好配置文件后，运行以下命令开始训练：
```bash
python Train.py
```
训练日志将显示 Loss 下降曲线及每个 Epoch 的验证集 WER。

---
*Created by AI Assistant for Undergraduate Thesis Project*
