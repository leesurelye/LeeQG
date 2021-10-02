# -*- coding: utf-8 -*-
# @Time    : 2021/9/19 14:40
# @Author  : Leesure
# @File    : Encoder.py
# @Software: PyCharm
# import torch
import torch.nn as nn


class RNNEncoder(nn.Module):
    """
        @output:    (1)output: 包含了每一个时间步的隐藏状态

    """

    def __init__(self, input_size, emb_size, hid_size, n_layers=1, dropout=0.5):
        super(RNNEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hid_size
        self.embed_size = emb_size

        self.embedding = nn.Embedding(input_size, emb_size)
        # rnn 的 output 包含所有时间步的隐藏状态，hn 是最后一个时间步的隐藏状态
        # output.size = (seq_len, batch_size, hidden_size) | hn.size = (num_layers, batch_size, hidden_size )
        self.rnn = nn.GRU(emb_size, hid_size, n_layers, dropout=dropout, bidirectional=True)

    def forward(self, inputs, hidden=None):
        """
            input: (1)inputs (max_len, batch_size)

        """
        # embedded = (max_len, batch_size, emb_size)
        embedded = self.embedding(inputs)
        # outputs = (max_len, batch_size, hid_size*bidirectional)
        # hidden = (num_layers*bidirectional, batch_size, hid_size)

        outputs, hidden = self.rnn(embedded, hidden)
        # outputs = (max_len, batch_size, hid_size)
        outputs = (outputs[:, :, :self.hidden_size] +
                   outputs[:, :, self.hidden_size:])
        return outputs, hidden
