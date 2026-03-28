# ResNet18 + BiLSTM 模型替换 + CSL 数据集支持

## 背景

- **模型**：MobileNetV2+TemporalConv+BiGRU → **ResNet18 + BiLSTM**
- **数据集**：在 CE-CSL 基础上增加 **CSL** 数据集支持

### 两种数据集格式对比

| | CE-CSL | CSL |
|---|---|---|
| 视频格式 | `.mp4` | `.avi` |
| 目录结构 | `train/A/train-00001.mp4` | `color/000000/P01_s1_00_0_color.avi` |
| 标签格式 | CSV，gloss 用 `/` 分隔 | `corpus.txt`，每行 `{id} {句子}`，每个汉字为一个 gloss |
| 划分 | train/dev/test 三个文件夹 + 三个 CSV | 单个 color 文件夹 + 单个 corpus.txt |

## Proposed Changes

### 1. [Net.py](file:///d:/Code/CEL/Net.py) — ResNet18 + BiLSTM

#### [MODIFY] [Net.py](file:///d:/Code/CEL/Net.py)

- Backbone: MobileNetV2 → **ResNet18(pretrained)**，输出 512-d
- 新增特征投影: [Linear(512→hidden) + ReLU + Dropout(0.3)](file:///d:/Code/CEL/Module.py#72-81)
- 移除 [TemporalConv](file:///d:/Code/CEL/Module.py#25-71)（避免时间步缩短导致 CTC 对齐困难）
- 时序建模: BiGRU → **BiLSTM** 2层, dropout=0.3
- 分类头 [NormLinear](file:///d:/Code/CEL/Module.py#72-81) 不变

---

### 2. [DataProcessMoudle.py](file:///d:/Code/CEL/DataProcessMoudle.py) — CSL 数据集支持

#### [MODIFY] [DataProcessMoudle.py](file:///d:/Code/CEL/DataProcessMoudle.py)

**Word2Id()**：
- CE-CSL：读 CSV，`/` 分隔 gloss（现有逻辑）
- CSL：读 `corpus.txt`，每行 `{id} {句子}`，每个汉字拆为独立 gloss

**MyDataset.__init__()**：
- CE-CSL：扫描 `ImagePath/子文件夹/*.mp4`（现有逻辑）
- CSL：扫描 `ImagePath/子文件夹/*.avi`（`ImagePath` = `color` 目录，子文件夹 = `000000`, `000001`...）

**MyDataset.__getitem__()**：
- 两者都用 `cv2.VideoCapture` 读视频帧，`.avi` 和 `.mp4` 逻辑一致，无需改

---

### 3. [Train.py](file:///d:/Code/CEL/Train.py)

#### [MODIFY] [Train.py](file:///d:/Code/CEL/Train.py)

- 移除 L105-106 的 `CE-CSL only` 检查，改为支持 `CE-CSL` 和 `CSL`

---

### 4. 配置

#### [MODIFY] [config.ini](file:///d:/Code/CEL/params/config.ini)

- `cnnChunkSize`: 16 → **8**
- 新增 `hiddenSize = 512`

#### [MODIFY] [ReadConfig.py](file:///d:/Code/CEL/ReadConfig.py)

- 默认 `hiddenSize`: 1024 → 512
- 默认 `cnnChunkSize`: 64 → 8

---

### 不需要改动的文件

[BiLSTM.py](file:///d:/Code/CEL/BiLSTM.py) [Module.py](file:///d:/Code/CEL/Module.py) [SLR.py](file:///d:/Code/CEL/SLR.py) [decode.py](file:///d:/Code/CEL/decode.py) [WER.py](file:///d:/Code/CEL/WER.py) [videoAugmentation.py](file:///d:/Code/CEL/videoAugmentation.py)

## 8G 显存适配

batchSize=1, gradAccumSteps=4, cnnChunkSize=8, frameSampleStride=2, useAmp=1, hiddenSize=512

## Verification

1. `python -c "import Net; import Train"` — 无导入报错
2. CPU 前向传播测试
3. 需要数据集: `python SLR.py` 在 8G 显存测试
