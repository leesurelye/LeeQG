# -*- coding: utf-8 -*-
# @Time    : 2021/9/19 14:41
# @Author  : Leesure
# @File    : Decoder.py
# @Software: PyCharm
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleDecoder(nn.Module):
    def __init__(self, vocab_size, hid_dim, dropout):
        super(SimpleDecoder, self).__init__()
        self.output_dim = vocab_size
        self.rnn = nn.GRU(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, context, hidden):
        out, hidden = self.rnn(context, hidden)
        output = torch.cat((out.squeeze(0), hidden.squeeze(0)), dim=1)
        prediction = self.fc(output)
        return prediction, hidden


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        # init value : random
        self.v = nn.Parameter(torch.rand(hidden_size))
        std_v = 1.0 / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-std_v, std_v)

    def forward(self, hidden, encoder_outputs):
        time_step = encoder_outputs.size(0)
        h = hidden.repeat(time_step, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        attn_energies = self.score(h, encoder_outputs)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        # [B*T*2H]->[B*T*H]
        energy = F.relu(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size, n_layers=1, dropout=0.2):
        super(Decoder, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embed = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.attention = Attention(hidden_size)
        self.rnn = nn.GRU(embed_size * 3, hidden_size, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, inputs, last_hidden, encoder_outputs):
        """
        input:  (1) input: [batch_size]
        """

        embedded = self.embed(inputs)
        # (1, B, N)  N is the embedding size
        embedded = self.dropout(embedded)
        # combine embedded input word and context
        attn_weights = self.attention(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N)
        context = context.transpose(0, 1)  # (1,B,N)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat([embedded, context], 2)
        output, hidden = self.rnn(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,N) -> (B,N)
        context = context.squeeze(0)
        output = self.fc_out(torch.cat([output, context], 1))
        output = F.log_softmax(output, dim=1)
        # output = [batch_size, vocab_size]
        return output, hidden


class RNNDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout, teach_forcing: bool):
        """

            output_dim: tgt_vocab_size
        """
        super(RNNDecoder, self).__init__()
        self.teach = teach_forcing
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim)
        self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, hidden, context):
        if self.teach:
            inputs = inputs.unsqueeze(0)
            embedded = self.dropout(self.embedding(inputs))
        else:
            embedded = inputs
        emb_con = torch.cat((embedded, context), dim=2)
        output, hidden = self.rnn(emb_con, hidden)
        output = torch.cat((embedded.squeeze(0), hidden.squeeze(0), context.squeeze(0)), dim=1)
        prediction = self.fc_out(output)
        return prediction, hidden

    def set_teach(self, teach):
        self.teach = teach
