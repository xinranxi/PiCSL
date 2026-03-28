import configparser
import os
import torch

# Make config path robust: locate params/config.ini relative to this file
def get_config_path():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "params", "config.ini")

def readConfig():
    # 默认配置参数
    configParams = {
        "trainDataPath": "CE-CSL/train",
        "validDataPath": "CE-CSL/dev",
        "testDataPath": "CE-CSL/test",
        "trainLabelPath": "CE-CSL/train.csv",
        "validLabelPath": "CE-CSL/dev.csv",
        "testLabelPath": "CE-CSL/test.csv",
        "bestModuleSavePath": "module/bestMoudleLightTFNet.pth",
        "currentModuleSavePath": "module/currentMoudleLightTFNet.pth",
        "device": 1, # 0:CPU  1:GPU
        "hiddenSize": 512,
        "lr": 0.0001,
        "batchSize": 2,
        "numWorkers": 2,
        "pinmMemory": 1,
        "moduleChoice": "LightTFNet",
        "dataSetName": "CE-CSL",
        "gradAccumSteps": 8,
        "maxGradNorm": 5.0,
        "freezeBackboneEpochs": 3,
        "backboneLrScale": 0.1,
        "maxEpochs": 55,
        "maxTrainBatches": 0,
        "maxValidBatches": 0,
        "maxTestBatches": 0,
        "frameSampleStride": 1,
        "cnnChunkSize": 8,
        "useAmp": 1,
        "usePreprocessed": 0,
        "preprocessedRoot": "CSL/preprocessed",
    }

    configPath = get_config_path()
    if os.path.exists(configPath):
        print("开始读取配置参数")
        cf = configparser.ConfigParser()
        # 显式使用 UTF-8 读取，避免 Windows 默认 GBK 导致解码失败。
        try:
            cf.read(configPath, encoding="utf-8")
        except UnicodeDecodeError:
            # 兼容历史本地文件编码。
            cf.read(configPath, encoding="gbk")

        # 读取路径参数
        configParams["trainDataPath"] = cf.get("Path", "trainDataPath")
        configParams["validDataPath"] = cf.get("Path", "validDataPath")
        configParams["testDataPath"] = cf.get("Path", "testDataPath")
        configParams["trainLabelPath"] = cf.get("Path", "trainLabelPath")
        configParams["validLabelPath"] = cf.get("Path", "validLabelPath")
        configParams["testLabelPath"] = cf.get("Path", "testLabelPath")
        configParams["bestModuleSavePath"] = cf.get("Path", "bestModuleSavePath")
        configParams["currentModuleSavePath"] = cf.get("Path", "currentModuleSavePath")
        # 读取数值参数
        configParams["device"] = cf.get("Params", "device")
        configParams["hiddenSize"] = cf.get("Params", "hiddenSize")
        configParams["lr"] = cf.get("Params", "lr")
        configParams["batchSize"] = cf.get("Params", "batchSize")
        configParams["numWorkers"] = cf.get("Params", "numWorkers")
        configParams["pinmMemory"] = cf.get("Params", "pinmMemory")
        configParams["moduleChoice"] = cf.get("Params", "moduleChoice")
        configParams["dataSetName"] = cf.get("Params", "dataSetName")

        # 可选参数（兼容旧配置）
        optional_params = [
            "gradAccumSteps", "maxGradNorm", "freezeBackboneEpochs", "backboneLrScale",
            "maxEpochs", "maxTrainBatches", "maxValidBatches", "maxTestBatches",
            "frameSampleStride", "cnnChunkSize", "useAmp", "usePreprocessed", "preprocessedRoot"
        ]
        for k in optional_params:
            if cf.has_option("Params", k):
                configParams[k] = cf.get("Params", k)

        cuda_available = torch.cuda.is_available()
        print("GPU is %s" % cuda_available)
        if 1 == int(configParams["device"]) and cuda_available:
            configParams["device"] = torch.device("cuda:0")
        else:
            if 1 == int(configParams["device"]) and not cuda_available:
                print("CUDA 不可用，自动回退到 CPU 运行")
            configParams["device"] = torch.device("cpu")
    else:
        print("配置文件不存在 %s" % (configPath))
        print("使用默认参数")

    if not isinstance(configParams["device"], torch.device):
        if 1 == int(configParams["device"]) and torch.cuda.is_available():
            configParams["device"] = torch.device("cuda:0")
        else:
            if 1 == int(configParams["device"]) and not torch.cuda.is_available():
                print("CUDA 不可用，自动回退到 CPU 运行")
            configParams["device"] = torch.device("cpu")

    for key in configParams:
        print("%s: %s" %(key, configParams[key]))

    return configParams
