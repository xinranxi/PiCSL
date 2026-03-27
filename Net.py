import torch.nn as nn
import torch
import torchvision.models as models
import numpy as np
import Module
from BiLSTM import BiLSTMLayer

class moduleNet(nn.Module):
    def __init__(self, hiddenSize, wordSetNum, moduleChoice="Seq2Seq", device=torch.device("cuda:0"), dataSetName='RWTH', isFlag=False):
        super().__init__()
        self.device = device
        self.moduleChoice = moduleChoice
        self.outDim = wordSetNum
        self.dataSetName = dataSetName
        self.logSoftMax = nn.LogSoftmax(dim=-1)
        self.softMax = nn.Softmax(dim=-1)
        self.isFlag = isFlag
        self.probs_log = []

        if "LightTFNet" == self.moduleChoice:
            # 轻量化模型：MobileNetV2 + GRU
            hidden_size = hiddenSize
            # 使用预训练的MobileNetV2作为特征提取器
# 优化为使用最新 API：
            # self.conv2d = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
            self.conv2d = models.mobilenet_v2(pretrained=True)
            
            # 保持预训练特征分布，避免随机线性层在小数据上破坏 backbone。
            self.conv2d.classifier = nn.Identity()

            # MobileNetV2 预训练权重对应的输入标准化参数。
            self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
            
            self.conv1d = Module.TemporalConv(input_size=self.conv2d.last_channel,
                                           hidden_size=hidden_size,
                                           conv_type=1)

            # 使用 GRU 代替 LSTM，减少参数量
            # BiLSTMLayer 内部封装了 nn.GRU (通过 rnn_type='GRU' 指定)
            self.temporal_model = BiLSTMLayer(rnn_type='GRU', input_size=hidden_size, hidden_size=hidden_size,
                                              num_layers=2, bidirectional=True)

            self.classifier = Module.NormLinear(hidden_size, self.outDim)
        else:
             print(f"Model {self.moduleChoice} is not implemented in this minimal Net.py")

    def pad(self, tensor, length):
        return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

    def forward(self, seqData, dataLen=None, isTrain=True):
        outData1 = None
        outData2 = None
        outData3 = None
        logProbs1 = None
        logProbs2 = None
        logProbs3 = None
        logProbs4 = None
        logProbs5 = None

        if "LightTFNet" == self.moduleChoice:
            # 轻量化处理逻辑
            len_x_int = [int(x.item()) if torch.is_tensor(x) else int(x) for x in dataLen]
            len_x = [
                (x.reshape(-1)[0].to(seqData.device).float() if torch.is_tensor(x) else torch.tensor(float(x), device=seqData.device))
                for x in dataLen
            ]
            batch, temp, channel, height, width = seqData.shape
            
            # 优化：仅提取有效帧
            inputs = seqData.reshape(batch * temp, channel, height, width)

            valid_frames_list = []
            for i, length in enumerate(len_x_int):
                start = i * temp
                end = start + length
                valid_frames_list.append(inputs[start:end])
            
            x = torch.cat(valid_frames_list, dim=0)

            # 数据预处理中输入范围是 [-1, 1]，先映射到 [0, 1] 再按 ImageNet 统计量归一化。
            x = ((x + 1.0) * 0.5).clamp(0.0, 1.0)
            x = (x - self.imagenet_mean) / self.imagenet_std

            # 2D CNN 特征提取 (MobileNetV2)
            # 返回大小为 512 的特征向量（来自我们的自定义分类头）
            # x = self.conv2d(x) # 原代码可能导致 OOM

            # 使用 Chunk 处理来避免 OOM (Unable to find a valid cuDNN algorithm)
            cnn_features = []
            chunk_size = 64  # 安全的批次大小
            for i in range(0, x.size(0), chunk_size):
                end_idx = min(i + chunk_size, x.size(0))
                # 保持外层的 autocast 上下文 (FP16)，节省显存
                chunk = x[i:end_idx]
                chunk_feat = self.conv2d(chunk)
                cnn_features.append(chunk_feat)
            
            x = torch.cat(cnn_features, dim=0)

            # 重构 batch 并填充以匹配时间维度 'temp'
            framewise_list = []
            current_idx = 0
            for length in len_x_int:
                seg = x[current_idx : current_idx + length]
                current_idx += length
                
                # 用零填充到长度 'temp'
                if length < temp:
                    zeros = torch.zeros((temp - length, x.size(1)), device=x.device)
                    padded_seg = torch.cat([seg, zeros], dim=0)
                else:
                    padded_seg = seg
                
                framewise_list.append(padded_seg)
            
            # (Batch, Time, Feature)
            framewise = torch.stack(framewise_list)
            
            # (Batch, Feature, Time) 用于 TemporalConv
            framewise = framewise.transpose(1, 2)

            # 1D 时序卷积
            conv1d_outputs = self.conv1d(framewise, len_x)
            
            x = conv1d_outputs['visual_feat']
            lgt = conv1d_outputs['feat_len'] 
            
            # RNN 格式: (Time, Batch, Channel)
            x = x.permute(2, 0, 1)
            # 使用 stack 而不是 cat，因为 lgt 是标量张量列表
            # Net.py logic fix
            if isinstance(lgt, list):
                lgt = torch.tensor([
                    int(v.item()) if torch.is_tensor(v) else int(v) for v in lgt
                ], dtype=torch.long)

            # GRU 时序建模
            outputs = self.temporal_model(x, lgt)

            logProbs1 = self.classifier(outputs['predictions'])

        return logProbs1, logProbs2, logProbs3, logProbs4, logProbs5, lgt, outData1, outData2, outData3
