# -*- coding: utf-8 -*-
# @Time    : 2021/9/19 14:02
# @Author  : Leesure
# @File    : DataSet.py
# @Software: PyCharm
import torch
import torch.utils.data as data_utils
from transformers import AutoTokenizer

special_tokens = {
    "eos_token": '[EOS]',
    "bos_token": '[BOS]',

}


def load_processed_data(file_path_dict: dict):
    """
        load processed data
        @return {
            src:[]
            tgt:[]
            ans:[](options)
        }
    """
    res = {}
    for k, v in file_path_dict.items():
        with open(v, 'r', encoding='utf-8') as f:
            data = f.read()
        data = data.split('\n')
        res[k] = data
    assert len(res['tgt']) == len(res['src']), "the length of tgt and src is not equal"
    return res['src'], res['tgt']


def _padding_zero(tokens, max_len: int):
    if len(tokens) > max_len:
        return tokens[:max_len]
    else:
        return tokens + [0] * (max_len - len(tokens))


def pad_special_words(text: list):
    return ['[BOS]'] + text + ['[EOS]']


def height_light_answer(sentence: str, answer: str):
    # TODO height light answer position
    pass


class DRCD_DataSet(data_utils.Dataset):
    def __init__(self, samples_dir: dict, device, pre_trained='bert-base-chinese', **kwargs):
        self.samples_dir: samples_dir
        self.samples_src, self.sample_tgt = load_processed_data(samples_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(pre_trained)
        self.special_words = {'bos_token': '[BOS]', 'eos_token': '[EOS]'}
        self.add_special_word(self.special_words)
        self.src_max_length = kwargs['src_max_len']
        self.tgt_max_length = kwargs['tgt_max_len']
        self.device = device

    def __len__(self):
        return len(self.samples_src)

    def add_special_word(self, special_dict):
        self.tokenizer.add_special_tokens(special_dict)

    def sentence_to_token(self, sentence: str):
        tokens = self.tokenizer.tokenize(sentence)
        final_tokens = pad_special_words(tokens)
        return self.tokenizer.convert_tokens_to_ids(final_tokens)

    def __getitem__(self, index):
        src_sentence = self.samples_src[index]
        tgt_sentence = self.sample_tgt[index]

        src_ids = _padding_zero(self.sentence_to_token(src_sentence), self.src_max_length)
        tgt_ids = _padding_zero(self.sentence_to_token(tgt_sentence), self.tgt_max_length)

        return src_sentence, torch.LongTensor(src_ids).to(self.device), tgt_sentence, torch.LongTensor(tgt_ids).to(
            self.device)

    def get_vocab_size(self):
        return self.tokenizer.vocab_size + len(self.special_words)
