#实现CTC输出解码（优先使用 torchaudio beam search，失败时回退到 greedy），映射索引与词表，并提供一个自定义CTC前向/后向损失实现。
torchaudio_ctc_decoder = None


def _load_torchaudio_ctc_decoder():
    try:
        from torchaudio.models.decoder import ctc_decoder as decoder_impl
        return decoder_impl
    except Exception as e:
        print(f"Warning: torchaudio ctc_decoder import failed: {e}")
        return None
from itertools import groupby
import torch.nn.functional as F
import torch
from six.moves import xrange

class Decode(object):# CTC解码类
    def __init__(self, gloss_dict, num_classes, search_mode, blank_id=0):
        # self.i2g_dict = dict((v[0], k) for k, v in gloss_dict.items())
        # self.g2i_dict = {v: k for k, v in self.i2g_dict.items()}
        self.g2i_dict = {}
        for k, v in gloss_dict.items():
            if v == 0:
                continue
            self.g2i_dict[k] = v
        self.i2g_dict = {v: k for k, v in self.g2i_dict.items()}
        self.num_classes = num_classes
        self.search_mode = search_mode
        self.blank_id = blank_id
        self.log_probs_input = True
        self.blank_token = "<blank>"
        self.sil_token = "<sil>"
        self.tokens = [self.blank_token]
        for idx in range(1, num_classes):
            self.tokens.append(self.i2g_dict.get(idx, f"<tok_{idx}>"))
        self.ctc_decoder = None
        decoder_factory = _load_torchaudio_ctc_decoder()
        if decoder_factory is not None:
            try:
                self.ctc_decoder = decoder_factory(
                    lexicon=None,
                    tokens=self.tokens,
                    lm=None,
                    nbest=1,
                    beam_size=10,
                    beam_threshold=50,
                    blank_token=self.blank_token,
                    sil_token=self.sil_token,
                    unk_word="<unk>",
                )
                print("torchaudio ctc_decoder initialized successfully with beam_size=10")
            except Exception as e:
                print(f"Warning: Failed to initialize torchaudio ctc_decoder: {e}")
                self.ctc_decoder = None
        
        if self.ctc_decoder is None:
            # 如果本地调试或云端安装失败，自动降级为 Greedy Search
            print("Warning: torchaudio beam decoder not available. Using greedy search (max) instead of beam search.")
            self.search_mode = "max"

    def decode(self, nn_output, vid_lgt, batch_first=True, probs=False):
        if not batch_first:
            nn_output = nn_output.permute(1, 0, 2)
        if self.search_mode == "max":
            return self.MaxDecode(nn_output, vid_lgt)
        else:
            return self.BeamSearch(nn_output, vid_lgt, probs)

    def BeamSearch(self, nn_output, vid_lgt, probs=False):
        if self.log_probs_input:
            # Train.py already feeds LogSoftmax output; do NOT apply softmax again.
            if probs:
                nn_output = (nn_output + 1e-8).log()
            emissions = nn_output.cpu()
        else:
            if not probs:
                nn_output = nn_output.softmax(-1)
            emissions = nn_output.cpu().log()
        lengths = vid_lgt.cpu()
        decoder_results = self.ctc_decoder(emissions, lengths)
        ret_list = []
        last_result = torch.tensor([])  # Track last valid result (fix: avoid reusing reference)
        for batch_result in decoder_results:
            if len(batch_result) > 0:
                first_tokens = batch_result[0].tokens
            else:
                first_tokens = torch.tensor([], dtype=torch.long)

            if len(first_tokens) != 0:
                first_result = torch.stack([x[0] for x in groupby(first_tokens)])
                last_result = first_result.clone()  # Clone to avoid reference issues
            else:
                first_result = first_tokens

            tmp = [(self.i2g_dict[int(gloss_id)], idx) for idx, gloss_id in
                             enumerate(first_result) if int(gloss_id) in self.i2g_dict]
            if len(tmp) > 0:
                ret_list.append(tmp)
            else:
                # Keep empty decode for this sample; DO NOT copy previous output (corrupts WER)
                ret_list.append([])

        # 在 Beam Search 情况下，返回预测序列列表 + 最后一条有效的 token 索引（已克隆，避免引用问题）
        return ret_list, last_result

    def MaxDecode(self, nn_output, vid_lgt):
        index_list = torch.argmax(nn_output, axis=2)
        batchsize, lgt = index_list.shape
        ret_list = []
        # result_list = []
        for batch_idx in range(batchsize):
            group_result = [x[0] for x in groupby(index_list[batch_idx][:vid_lgt[batch_idx]])]
            filtered = [*filter(lambda x: x != self.blank_id, group_result)]
            if len(filtered) > 0:
                max_result = torch.stack(filtered)
                max_result = [x[0] for x in groupby(max_result)]
            else:
                max_result = filtered
            ret_list.append([(self.i2g_dict[int(gloss_id)], idx) for idx, gloss_id in
                             enumerate(max_result)])
            # result_list.append(max_result)
        # return ret_list, result_list
        return ret_list, index_list

def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0):
    """
    params: [vocab_size, T], logits.softmax(-1). T 是输入序列的长度，vocab_size是词表大小。
    seq: [seq_len] 输出序列的长度。

    CTC loss function.
    params - n x m matrix of n-D probability distributions over m frames.
    seq - sequence of phone id's for given example.
    is_prob - whether params have already passed through a softmax
    Returns objective and gradient.
    """
    batchSize = len(target_lengths)
    numphones = log_probs.shape[-1]  # Number of labels
    n = 0

    for i in range(batchSize):
        seqLen = target_lengths[i]  # Length of label sequence (# phones)
        L = 2 * seqLen + 1  # Length of label sequence with blanks, 拓展后的 l'.
        T = input_lengths[i]  # Length of utterance (time)

        # 建立表格 l' x T.
        alphas = torch.zeros((L, T))  # 前向概率
        betas = torch.zeros((L, T))  # 后向概率

        # 这里dp的map：
        # 横轴为 2*seq_len+1, 也就是 ground truth label中每个token前后插入 blank
        # 纵轴是 T frames

        log_probs = F.softmax(log_probs, dim=-1)

        # 初始条件：T=0时，只能为 blank 或 seq[0]
        alphas[0, 0] = log_probs[0, i,  blank]
        alphas[1, 0] = log_probs[0, i,  targets[n]]
        # T=0， alpha[:, 0] 其他的全部为 0

        c = torch.sum(alphas[:, 0])
        alphas[:, 0] = alphas[:, 0] / c  # 这里 T=0 时刻所有可能节点的概率要归一化

        llForward = torch.log(c)  # 转换为log域

        for t in xrange(1, T):
            # 第一个循环： 计算每个时刻所有可能节点的概率和
            start = max(0, L - 2 * (T - t))  # 对于时刻 t, 其可能的节点.与公式2一致。
            end = min(2 * t + 2, L)  # 对于时刻 t，最大节点范围不可能超过 2t+2
            for s in xrange(start, L):
                l = int((s - 1) / 2)
                # blank，节点s在偶数位置，意味着s为blank
                if s % 2 == 0:
                    if s == 0: # 初始位置，单独讨论
                        alphas[s, t] = alphas[s, t - 1] * log_probs[t, i, blank]
                    else:
                        alphas[s, t] = (alphas[s, t - 1] + alphas[s - 1, t - 1]) * log_probs[t, i, blank]
                # s为奇数，非空
                # l = (s-1/2) 就是 s 所对应的 lable 中的字符。
                # ((s-2)-1)/2 = (s-1)/2-1 = l-1 就是 s-2 对应的lable中的字符
                elif s == 1 or targets[l] == targets[l - 1]:
                    alphas[s, t] = (alphas[s, t - 1] + alphas[s - 1, t - 1]) * log_probs[t, i, targets[l]]
                else:
                    alphas[s, t] = (alphas[s, t - 1] + alphas[s - 1, t - 1] + alphas[s - 2, t - 1]) \
                                   * log_probs[t, i, targets[l]]

            # normalize at current time (prevent underflow)
            c = torch.sum(alphas[start:end, t])
            alphas[start:end, t] = alphas[start:end, t] / c
            llForward = llForward + torch.log(c)

        n = n + target_lengths[i]
        sumN = torch.sum(input_lengths)

    return llForward / sumN
