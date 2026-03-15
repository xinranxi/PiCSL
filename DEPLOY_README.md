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

## 3. 启动训练
在安装完环境后，修改 `params/config.ini` 中的路径配置（如果尚未修改为云端路径），然后运行：

```bash
# 确保 PYTHONPATH 包含当前目录
export PYTHONPATH=$PYTHONPATH:/hy-tmp/TEST

# 启动训练
python SLR.py
```

## 4. 常见问题
- **OOM (显存溢出)**: 
    - 修改 `params/config.ini` 中的 `batchSize`，尝试减小为 2 或 1。
    - MobileNetV2 占用显存较小，一般 3060 (12G) 可支持 Batch=4。
- **ctcdecode 安装失败**:
    - 确保系统已安装 gcc/g++ 编译器 (恒源云默认已安装)。
    - 如果依然无法安装，代码会自动回退到 `Greedy Search`，不影响训练，但 WER 指标会略低。
