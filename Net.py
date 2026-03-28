import torch.nn as nn
import torch
import torchvision.models as models
import numpy as np
import Module
from BiLSTM import BiLSTMLayer

class moduleNet(nn.Module):
    def __init__(self, hiddenSize, wordSetNum, moduleChoice="Seq2Seq", device=torch.device("cuda:0"), dataSetName='RWTH', isFlag=False, cnnChunkSize=64):
        super().__init__()
        self.device = device
        self.moduleChoice = moduleChoice
        self.outDim = wordSetNum
        self.dataSetName = dataSetName
        self.logSoftMax = nn.LogSoftmax(dim=-1)
        self.softMax = nn.Softmax(dim=-1)
        self.isFlag = isFlag
        self.probs_log = []
        self.cnnChunkSize = max(1, int(cnnChunkSize))

        if "LightTFNet" == self.moduleChoice:
            hidden_size = hiddenSize

            # ========== Backbone: ResNet18 (预训练) ==========
            self.conv2d = models.resnet18(pretrained=True)
            self.conv2d.fc = nn.Identity()  # 移除分类头，输出 512-d 特征

            # ImageNet 标准化参数
            self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

            # ========== 特征投影: 512 -> hidden_size ==========
            self.feature_proj = nn.Sequential(
                nn.Linear(512, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
            )

            # ========== 时序建模: BiLSTM ==========
            # 不再使用 TemporalConv，避免时间步被池化缩短
            self.temporal_model = BiLSTMLayer(
                rnn_type='LSTM',
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=2,
                bidirectional=True,
                dropout=0.3,
            )

            # ========== 分类头 ==========
            self.classifier = Module.NormLinear(hidden_size, self.outDim)
            self.classifier_bias = nn.Parameter(torch.zeros(self.outDim))
            with torch.no_grad():
                # CTC 在长序列+小样本下极易先塌到 blank，
                # 这里显式压低 blank 初始偏置，给非 blank 一个更可学习的起点。
                self.classifier_bias[0] = -2.0
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
            len_x_int = [int(x.item()) if torch.is_tensor(x) else int(x) for x in dataLen]
            batch, temp, channel, height, width = seqData.shape
            left_pad = 6

            # ===== 仅提取有效帧 =====
            inputs = seqData.reshape(batch * temp, channel, height, width)

            valid_frames_list = []
            for i, length in enumerate(len_x_int):
                start = i * temp
                valid_start = start + left_pad
                valid_end = valid_start + length
                valid_frames_list.append(inputs[valid_start:valid_end])

            x = torch.cat(valid_frames_list, dim=0)

            # 输入归一化: [-1, 1] -> [0, 1] -> ImageNet normalize
            x = ((x + 1.0) * 0.5).clamp(0.0, 1.0)
            x = (x - self.imagenet_mean) / self.imagenet_std

            # ===== ResNet18 特征提取 (分块处理防 OOM) =====
            cnn_features = []
            chunk_size = self.cnnChunkSize
            for i in range(0, x.size(0), chunk_size):
                end_idx = min(i + chunk_size, x.size(0))
                chunk = x[i:end_idx]
                chunk_feat = self.conv2d(chunk)
                cnn_features.append(chunk_feat)

            x = torch.cat(cnn_features, dim=0)  # (total_valid_frames, 512)

            # ===== 特征投影: 512 -> hidden_size =====
            x = self.feature_proj(x)  # (total_valid_frames, hidden_size)

            # ===== 重构 batch 并填充 =====
            framewise_list = []
            current_idx = 0
            for length in len_x_int:
                seg = x[current_idx : current_idx + length]
                current_idx += length

                if length < temp:
                    zeros = torch.zeros((temp - length, x.size(1)), device=x.device)
                    padded_seg = torch.cat([seg, zeros], dim=0)
                else:
                    padded_seg = seg

                framewise_list.append(padded_seg)

            # (Batch, Time, Feature)
            framewise = torch.stack(framewise_list)

            # ===== BiLSTM 时序建模 =====
            # 转为 RNN 格式: (Time, Batch, Feature)
            x = framewise.permute(1, 0, 2)

            # lgt 直接为有效帧长度（不再经过 TemporalConv 缩减）
            lgt = torch.tensor(len_x_int, dtype=torch.long)

            outputs = self.temporal_model(x, lgt)

            logProbs1 = self.classifier(outputs['predictions']) + self.classifier_bias

        return logProbs1, logProbs2, logProbs3, logProbs4, logProbs5, lgt, outData1, outData2, outData3
