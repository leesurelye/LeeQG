# -*- coding: utf-8 -*-
# @Time    : 2021/9/19 17:27
# @Author  : Leesure
# @File    : Attention.py
# @Software: PyCharm
import torch
import torch.nn as nn


class GatedSelfAttention(nn.Module):
    def __init__(self, dim, attn_dim=64, dropout=0.1):
        super(GatedSelfAttention, self).__init__()

        self.m_translate = nn.Linear(dim, attn_dim)
        self.q_translate = nn.Linear(dim, attn_dim)

        self.update = nn.Linear(2 * dim, dim, bias=False)

        self.gate = nn.Linear(2 * dim, dim, bias=False)
        self.dropout = dropout > 0
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)

    def forward(self, query, mask):
        raw = query

        memory = self.m_translate(query)
        query = self.q_translate(query)
        # A@B
        energy = torch.bmm(query, memory.transpose(1, 2))
        energy = energy.masked_fill(mask, value=-1e12)
        score = torch.softmax(energy, dim=2)
        if self.dropout:
            score = self.dropout(score)
        context = torch.bmm(score, raw)

        inputs = torch.cat((raw, context), dim=2)
        f_t = torch.tanh(self.update(inputs))
        g_t = torch.tanh(self.gate(inputs))

        output = f_t * g_t + (1 - g_t) * raw
        return output, score


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, query len, key len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)

        # attention = [batch size, n heads, query len, key len]

        x = torch.matmul(self.dropout(attention), V)

        # x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)

        # x = [batch size, query len, hid dim]

        x = self.fc_o(x)

        # x = [batch size, query len, hid dim]

        return x, attention
