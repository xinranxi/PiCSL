# Colab 上传说明

这个项目已经整理为适合上传到 Colab 的最小代码包，推荐直接使用 [`prepare_colab_bundle.py`](prepare_colab_bundle.py) 生成压缩包。

## 1. 建议上传到 Colab 的文件

训练主入口：

- [`SLR.py`](SLR.py)
- [`Train.py`](Train.py)
- [`ReadConfig.py`](ReadConfig.py)
- [`params/config.ini`](params/config.ini)

模型与数据处理依赖：

- [`Net.py`](Net.py)
- [`Module.py`](Module.py)
- [`BiLSTM.py`](BiLSTM.py)
- [`DataProcessMoudle.py`](DataProcessMoudle.py)
- [`videoAugmentation.py`](videoAugmentation.py)
- [`decode.py`](decode.py)
- [`WER.py`](WER.py)

评估与导出：

- [`export_test_predictions.py`](export_test_predictions.py)
- [`evaluation/`](evaluation)
- [`evaluationT/`](evaluationT)

环境文件：

- [`requirements.txt`](requirements.txt)
- [`README.md`](README.md)
- [`preprocess_csl_videos.py`](preprocess_csl_videos.py)

数据与权重不要直接打进代码包：

- 本地数据目录如 `CSL/color/`
- 大模型权重如 `module/*.pth`
- 中间结果目录如 `test_reports/`、`wer/`

## 2. 一键生成 Colab 上传包

在项目根目录运行：

```bash
python prepare_colab_bundle.py
```

生成结果：

- `colab_bundle/`：整理后的上传目录
- `colab_bundle.zip`：可直接上传到 Colab 的压缩包

## 3. Colab 中的推荐步骤

### 3.1 上传代码包

把 `colab_bundle.zip` 上传到 Colab，然后执行：

```python
!unzip -q colab_bundle.zip -d /content/project
%cd /content/project
```

### 3.2 安装依赖

```python
!pip install -r requirements.txt
```

[`decode.py`](decode.py) 现在优先使用 `torchaudio` 的 beam decoder；如果当前环境缺少对应依赖或初始化失败，会自动回退到 greedy decode。

### 3.3 上传数据和权重

你需要额外上传以下内容到 Colab：

1. `CSL/corpus.txt`
2. `CSL/splits/train_split.txt`
3. `CSL/splits/valid_split.txt`
4. `CSL/splits/test_split.txt`
5. 二选一：`CSL/color/...` 视频文件 或 `CSL/preprocessed/...` 预处理缓存
6. 可选：`module/*.pth` 预训练权重

目录结构建议保持为：

```text
/content/project/
├── CSL/
│   ├── corpus.txt
│   ├── splits/
│   ├── color/
│   └── preprocessed/
├── module/
├── params/
└── *.py
```

## 4. 运行方式

### 4.0 推荐：先在本地预处理视频

如果你想减少 Colab 上的实时视频解码开销，可以在本地项目根目录执行：

```bash
python preprocess_csl_videos.py --frame-sample-stride 4 --output-root CSL/preprocessed
```

这个脚本会对 [`CSL/splits/train_split.txt`](CSL/splits/train_split.txt)、[`CSL/splits/valid_split.txt`](CSL/splits/valid_split.txt)、[`CSL/splits/test_split.txt`](CSL/splits/test_split.txt) 里引用的视频做：

- 抽帧
- 缩放到 `224x224`
- 保存成 `.npy` 缓存

训练时在 [`params/config.ini`](params/config.ini) 中启用：

```ini
usePreprocessed = 1
preprocessedRoot = CSL/preprocessed
```

这样 [`DataProcessMoudle.MyDataset`](DataProcessMoudle.py:185) 会优先读取预处理缓存，找不到时才回退到原始视频读取。

训练：

```python
!python SLR.py
```

导出测试集预测：

```python
!python export_test_predictions.py
```

## 5. 当前项目入口关系

- [`SLR.py`](SLR.py) 是训练入口
- [`SLR.main()`](SLR.py:4) 调用 [`readConfig()`](ReadConfig.py:10)
- [`readConfig()`](ReadConfig.py:10) 读取 [`params/config.ini`](params/config.ini)
- [`train()`](Train.py:91) 是核心训练逻辑
- [`Net.moduleNet`](Net.py:8) 是模型定义
- [`DataProcessMoudle.MyDataset`](DataProcessMoudle.py:185) 是数据集加载逻辑

## 6. 上传前注意事项

1. [`requirements.txt`](requirements.txt) 已补充 `Pillow` 和 `scipy`，避免 Colab 缺包。
2. [`.gitignore`](.gitignore) 已排除权重、日志、临时评估文件和打包产物。
3. 当前 [`params/config.ini`](params/config.ini) 使用的是 `CSL` 数据集路径，上传后只要目录结构不变即可直接运行。
