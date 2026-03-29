# Mobile-TFNet Cloud Deployment Guide

本文档旨在指导如何将 `Mobile-TFNet` 项目部署至恒源云 (GPU Cloud) 进行训练。

## 1. 上传方法 (OSS)
请使用恒源云提供的 `ossutil` 或网页端 OSS 工具，将本目录 (`TEST`) 打包上传至云服务器的 `/hy-tmp/` 目录。
推荐目录结构：
```bash
/hy-tmp/TEST/
    params/
    CE-CSL/
    ... (所有代码文件)
```

## 2. 环境安装
在云端终端 (Terminal) 中执行以下命令：

```bash
# 1. 基础依赖
cd /hy-tmp/TEST
pip install -r requirements.txt

# 2. 安装 ctcdecode (关键步骤)
# 由于 ctcdecode 没有发布到 PyPI，只能源码编译安装
git clone https://github.com/parlance/ctcdecode.git
cd ctcdecode
pip install .
cd ..
```

## 3. 服务器推荐执行命令

安装完成后，推荐先生成训练集压缩缓存，再启动训练：

```bash
cd /hy-tmp/TEST

# 1) 预热训练集缓存
python preprocess_csl_videos.py --source-mode split --cache-splits train --output-root CSL/cache_v1 --frame-sample-stride 4 --resize 224 --cache-format npz

# 2) 启动训练
python SLR.py
```

若想直接开跑，也可以只执行 [`python SLR.py`](SLR.py)。在默认 [`videoCacheMode=lazy`](params/config.ini) 下，训练首轮会自动边读原视频边写缓存。

## 4. 缓存方案说明

- 默认缓存目录：[`CSL/cache_v1`](params/config.ini)
- 默认缓存格式：压缩 [`*.npz`](DEPLOY_README.md)
- 默认策略：仅缓存训练集 [`cacheTrainOnly=1`](params/config.ini)
- 兼容旧版 [`*.npy`](DEPLOY_README.md) 缓存读取，但不再建议继续生成无压缩全量缓存

## 5. 常见问题
- **OOM (显存溢出)**: 
    - 修改 `params/config.ini` 中的 `batchSize`，尝试减小为 2 或 1。
    - MobileNetV2 占用显存较小，一般 3060 (12G) 可支持 Batch=4。
- **GPU 利用率低 / CPU 解码占满**:
    - 先执行 [`preprocess_csl_videos.py`](preprocess_csl_videos.py) 预热训练集缓存。
    - 保留 [`persistentWorkers=1`](params/config.ini) 与较高的 [`numWorkers`](params/config.ini)。
    - 默认从 [`frameSampleStride=4`](params/config.ini) 起步，再按显存与精度需求微调。
- **ctcdecode 安装失败**:
    - 确保系统已安装 gcc/g++ 编译器 (恒源云默认已安装)。
    - 如果依然无法安装，代码会自动回退到 `Greedy Search`，不影响训练，但 WER 指标会略低。
