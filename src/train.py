# -*- coding: utf-8 -*-
# @Time    : 2021/9/18 10:20
# @Author  : Leesure
# @File    : train.py
# @Software: PyCharm


# parameter file
import os

import torch
import torch.nn as nn
from torch import optim
from dataset_process.DataSet import DRCD_DataSet
import torch.utils.data as torch_utils
from models.Model import Seq2Seq
from models.modules.Decoder import Decoder
from models.modules.Encoder import RNNEncoder
from transformers import generation_beam_search
from utils.vis_util import plot_loss

from tqdm import tqdm
from LeeQG import CONSTANT


def train(model, data, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for _, src, _, tgt in tqdm(data, desc="train"):
        optimizer.zero_grad()
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)

        out = model(src, tgt)
        # output = [batch_size, tgt_len, vocab_size]
        output_dim = out.shape[-1]
        out = out[1:].view(-1, output_dim)
        tgt = tgt[1:].contiguous().view(-1)

        loss = criterion(out, tgt)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(data)


def evaluate(model, data, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for _, src, _, tgt in tqdm(data, desc="validate"):
            src = src.transpose(0, 1)
            tgt = tgt.transpose(0, 1)
            output = model(src, tgt)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            tgt = tgt[1:].contiguous().view(-1)

            loss = criterion(output, tgt)
            epoch_loss += loss.item()
    return epoch_loss / len(data)


def get_generate(model: Seq2Seq, data, max_len, tokenizer):
    sos_token_ids = tokenizer.convert_tokens_to_ids('[SOS]')
    eos_token_ids = tokenizer.convert_tokens_to_ids('[EOS]')
    pad_token_ids = tokenizer.convert_tokens_to_ids('[PAD]')
    with torch.no_grad():
        for src_text, src, tgt_text, tgt in tqdm(data, desc="Generate"):
            src = src.transpose(0, 1)
            sentence = model.generate(src, max_len, 3, torch.device('cuda'),
                                      eos_token_ids, pad_token_ids, sos_token_ids)
            for i in range(max_len):
                print("=======================================")
                print("<src>", src_text[i])
                print("<gold>", tgt_text[i])
                print('<pred>', tokenizer.convert_ids_to_tokens(sentence[i]))
                print("======================================")

    print("Finnish")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_path = {
        'src': '/home1/liyue/NLP_QG/RL-for-Question-Generation/LeeQG/dataset/preprocess/train/para-train.txt',
        'tgt': '/home1/liyue/NLP_QG/RL-for-Question-Generation/LeeQG/dataset/preprocess/train/tgt-train.txt'
    }
    valid_path = {
        'src': '/home1/liyue/NLP_QG/RL-for-Question-Generation/LeeQG/dataset/preprocess/dev/para-dev.txt',
        'tgt': '/home1/liyue/NLP_QG/RL-for-Question-Generation/LeeQG/dataset/preprocess/dev/tgt-dev.txt'
    }

    train_data = DRCD_DataSet(train_path, device, pre_trained='bert-base-chinese', src_max_len=500, tgt_max_len=50)
    valid_data = DRCD_DataSet(valid_path, device, pre_trained='bert-base-chinese', src_max_len=500, tgt_max_len=50)
    # plot_sen_len_freq(train_data.samples_src, data_type="train_src")
    # plot_sen_len_freq(train_data.sample_tgt, data_type="train_target")
    dl_train = torch_utils.DataLoader(train_data, batch_size=32, shuffle=True, drop_last=True)
    dl_valid = torch_utils.DataLoader(valid_data, batch_size=32, shuffle=False, drop_last=True)
    # prepare model
    vocab_size = train_data.get_vocab_size()
    embedded_size = 256
    hidden_size = 512
    encoder = RNNEncoder(vocab_size, embedded_size, hidden_size, n_layers=2, dropout=0.5)
    decoder = Decoder(embedded_size, hidden_size, vocab_size, n_layers=1, dropout=0.5)

    model = Seq2Seq(encoder, decoder).to(device)
    print(model)
    # Training

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.NLLLoss()
    best_val_loss = float('inf')
    epochs = 20
    train_loss_history = []
    valid_loss_history = []
    # print("start training ....")
    # if not os.path.exists(CONSTANT.CHECKPOINT):
    #     os.mkdir(CONSTANT.CHECKPOINT)
    #
    # for epoch in range(epochs):
    #
    #     train_loss = train(model, dl_train, optimizer, criterion)
    #     valid_loss = evaluate(model, dl_valid, criterion)
    #     train_loss_history.append(train_loss)
    #     valid_loss_history.append(valid_loss)
    #     if valid_loss < best_val_loss:
    #         best_val_loss = valid_loss
    #         torch.save(model.state_dict(), f'{CONSTANT.CHECKPOINT}/test_{best_val_loss}.pt')
    #     print(f"Epoch:{epoch} | train_loss : {train_loss:.3f} | valid_loss : {valid_loss:.3f} ")
    # print("End training.")
    # plot_loss(epochs, train_loss_history, valid_loss_history)

    # generation
    state_dict = torch.load(
        '/home1/liyue/NLP_QG/RL-for-Question-Generation/LeeQG/checkpoint/test_4.934656737067483.pt',
        map_location=device)
    model.load_state_dict(state_dict)
    get_generate(model, dl_valid, max_len=50, tokenizer=valid_data.tokenizer)

    # Validation


    #
    # for src, src_ids, tgt, tgt_ids in dl_valid:
    #     out = model(src_ids, tgt_ids)
    #     out = out.argmax(-1)
    #     print("Sentence: ", src)
    #     print("Real Question: ", tgt)
    #     for i, ele in enumerate(out):
    #         sen = valid_data.tokenizer.convert_ids_to_tokens(ele)
    #         print(sen)
