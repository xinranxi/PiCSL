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
            
            # 替换分类层为 Sequential(Dropout, Linear)
            # 原始 Linear 是 (1280, 1000)
            # 我们将其替换为 (1280, 512) 
            self.conv2d.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.conv2d.last_channel, 512), 
            )
            
            self.conv1d = Module.TemporalConv(input_size=512,
                                           hidden_size=hidden_size,
                                           conv_type=2)

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
            len_x = dataLen
            batch, temp, channel, height, width = seqData.shape
            
            # 优化：仅提取有效帧
            inputs = seqData.reshape(batch * temp, channel, height, width)

            valid_frames_list = []
            for i, length in enumerate(len_x):
                start = i * temp
                end = start + length
                valid_frames_list.append(inputs[start:end])
            
            x = torch.cat(valid_frames_list, dim=0)

            # 2D CNN 特征提取 (MobileNetV2)
            # 返回大小为 512 的特征向量（来自我们的自定义分类头）
            x = self.conv2d(x)

            # 重构 batch 并填充以匹配时间维度 'temp'
            framewise_list = []
            current_idx = 0
            for length in len_x:
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
            if isinstance(lgt, list) and len(lgt) > 0 and lgt[0].ndim == 0:
                lgt = torch.stack(lgt)
            elif isinstance(lgt, list):
                 lgt = torch.cat(lgt, dim=0)

            # GRU 时序建模
            outputs = self.temporal_model(x, lgt)

            logProbs1 = self.classifier(outputs['predictions'])

        return logProbs1, logProbs2, logProbs3, logProbs4, logProbs5, lgt, outData1, outData2, outData3
