import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMLayer(nn.Module):
    """
    双向 LSTM 封装层 (BiLSTMLayer)

    提供：
    - 支持 pack_padded_sequence / pad_packed_sequence 的前向接口
    - 当 bidirectional=True 时，会将多个方向的 hidden 合并返回
    - 对 LSTM 的 hidden tuple 进行拆分与合并处理，兼容外部传入的 stacked hidden
    """

    def __init__(self, input_size, debug=False, hidden_size=512, num_layers=1, dropout=0,
                 bidirectional=True, rnn_type='LSTM', num_classes=-1):
        super(BiLSTMLayer, self).__init__()

        # RNN dropout, 图层间的 dropout
        self.dropout = dropout
        self.num_layers = num_layers
        self.input_size = input_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = int(hidden_size / self.num_directions)
        self.rnn_type = rnn_type
        self.debug = debug
        # 创建 RNN 实例（LSTM 或 GRU）
        self.rnn = getattr(nn, self.rnn_type)(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional)
        # for name, param in self.rnn.named_parameters():
        #     if name[:6] == 'weight':
        #         nn.init.orthogonal_(param)

    def forward(self, src_feats, src_lens=None, hidden=None):
        """
        Args:
            - src_feats: (max_src_len, batch_size, D)
            - src_lens: (batch_size)
        Returns:
            - outputs: (max_src_len, batch_size, hidden_size * num_directions)
            - hidden : (num_layers, batch_size, hidden_size * num_directions)
        """
        # 输入: src_feats shape=(max_src_len, batch_size, D)
        # src_lens: 每个样本的序列长度 (batch_size,)
        # 先将变长序列打包（packed），以供 PyTorch RNN 高效处理
        # (max_src_len, batch_size, D)
        if src_lens is not None and torch.is_tensor(src_lens):
            src_lens = src_lens.detach().cpu()
        packed_emb = nn.utils.rnn.pack_padded_sequence(src_feats, src_lens, enforce_sorted=False)

        # rnn(gru) returns:
        # - packed_outputs: shape same as packed_emb
        # - hidden: (num_layers * num_directions, batch_size, hidden_size)
        # 如果外部传入 hidden（stacked 格式），且类型为 LSTM，则需要拆分成 (h, c)
        if hidden is not None and self.rnn_type == 'LSTM':
            half = int(hidden.size(0) / 2)
            hidden = (hidden[:half], hidden[half:])
        packed_outputs, hidden = self.rnn(packed_emb, hidden)

        # outputs: (max_src_len, batch_size, hidden_size * num_directions)
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)

        # 如果为双向 RNN，需要将 hidden 的方向维度合并
        if self.bidirectional:
            # (num_layers * num_directions, batch_size, hidden_size)
            # => (num_layers, batch_size, hidden_size * num_directions)
            hidden = self._cat_directions(hidden)

        # LSTM 的 hidden 为 tuple(h, c)，这里将两者按第一维拼接为统一返回格式
        if isinstance(hidden, tuple):
            # cat hidden and cell states
            hidden = torch.cat(hidden, 0)

        return {
            "predictions": rnn_outputs,
            "hidden": hidden
        }

    def _cat_directions(self, hidden):
        """ If the encoder is bidirectional, do the following transformation.
            Ref: https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/models/DecoderRNN.py#L176
            -----------------------------------------------------------
            In: (num_layers * num_directions, batch_size, hidden_size)
            (ex: num_layers=2, num_directions=2)

            layer 1: forward__hidden(1)
            layer 1: backward_hidden(1)
            layer 2: forward__hidden(2)
            layer 2: backward_hidden(2)

            -----------------------------------------------------------
            Out: (num_layers, batch_size, hidden_size * num_directions)

            layer 1: forward__hidden(1) backward_hidden(1)
            layer 2: forward__hidden(2) backward_hidden(2)
        """

        # 将 PyTorch 默认的 [f1, b1, f2, b2, ...] 排列转换成 [(f1, b1), (f2, b2), ...]
        # 并在最后一个维度上拼接 forward 与 backward 的 hidden，输出维度变为 hidden_size * num_directions
        def _cat(h):
            return torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)

        if isinstance(hidden, tuple):
            # LSTM hidden contains a tuple (hidden state, cell state)
            hidden = tuple([_cat(h) for h in hidden])
        else:
            # GRU hidden
            hidden = _cat(hidden)

        return hidden
