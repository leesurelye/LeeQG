# -*- coding: utf-8 -*-
"""
@Time    : 2021/9/14 19:29
@Author  : Le Yu E
@FileName: Vocab.py
@Software: PyCharm
"""
import logging

import torch
import functools
from typing import List, Dict
from transformers import AutoTokenizer
from tqdm import tqdm
from dataset_process import Constant
from exception.GlobalException import ParameterError, CheckParameterError


class Vocab(object):
    """
    Class for get vocabulary from dataset, ONLY for english and chinese language
    Note: Class for getting vocabulary from dataset
    Contain Bert Tokenizer
    """

    def __init__(self, special_words: Dict, language='en', lower=True, checkpoint='default'):
        """
        `type` is the  pretrained tokenizer
        (*) [checkpoint:str]  :  pre_trained model in transformers `default` means don't use transformers
        """
        self.checkpoint = checkpoint if checkpoint else 'default'
        self.language = language
        # dict{'pad_token':'[PAD]'}
        self.special_words = special_words
        self.lower = lower if self.language == 'en' else False
        # store the idx of special words

        if self.checkpoint == 'default':
            # don't using pre trained model
            self.idx_to_token_dict = {}
            self.token_to_idx_dict = {}
            # this store idx of special words
            self.special = []
            if len(special_words) > 0:
                self.add_special_words([v for _, v in special_words.items()])
        else:
            # use pre trained model
            self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
            self.tokenizer.add_special_tokens(special_words)
            self.token_to_idx_F = functools.partial(self.tokenizer.convert_tokens_to_ids)
            self.idx_to_token_F = functools.partial(self.tokenizer.convert_ids_to_tokens)
            self.special = self.token_to_idx_F([v for _, v in special_words.items()])
        # {idx:freq}
        self.frequencies = {k: 1 for k in self.special}

    @classmethod
    def from_opt(cls, corpus=None, pretrained=None, **kwargs):
        # static method
        special_word = {'pad_token': Constant.PAD['token'], 'unk_token': Constant.UNK['token']}
        if kwargs['trans']:
            special_word['cls_token'] = Constant.CLS['token']
        if kwargs['tgt'] and not pretrained:
            special_word['bos_token'] = Constant.BOS['token']
            special_word['eos_token'] = Constant.EOS['token']
        if kwargs['separate']:
            special_word['sep_token'] = Constant.SEP['token']

        if pretrained:
            return cls(special_word, language=kwargs['language'], lower=kwargs['lower'], checkpoint=pretrained)
        else:
            if corpus is None:
                raise ParameterError("corpus")
            vocab = cls(special_word, language=kwargs['language'], lower=kwargs['lower'])
            if special_word != vocab.special_words:
                raise ParameterError("special_word !=vocab.special_words")
            for sent in corpus:
                for word in sent:
                    vocab.add_to_dict(word, lower=kwargs['lower'])
            original_size = vocab.size
            vocab = vocab.prune(kwargs['size'], kwargs['frequency'], kwargs['mode'])

        logging.info(f"Truncate vocabulary size from {original_size} to {vocab.size}")
        return vocab

    @property
    def size(self):
        """
        return the size of vocabulary
        """
        if self.checkpoint == 'default':
            return len(self.idx_to_token_dict)
        return self.tokenizer.vocab_size

    def get_label(self, idx, default=Constant.UNK['token']):
        if self.checkpoint != "default":
            return self.idx_to_token_F(idx)
        try:
            return self.idx_to_token_dict[idx]
        except KeyError:
            return default

    def get_idx(self, label, default=Constant.UNK['id']):
        if self.checkpoint != 'default':
            return self.token_to_idx_F(label)
        try:
            return self.token_to_idx_dict[label]
        except KeyError:
            return default

    def add_to_dict(self, label, idx=None, lower=False):
        """
        Add `label` in the dictionary. Use `idx` as its index in the directory
        Return: idx: the index in the directory
        """
        if lower and self.language == 'en':
            label = label.lower()

        if not idx:
            idx = len(self.idx_to_token_dict)
        if not idx and label in self.token_to_idx_dict:
            idx = self.token_to_idx_dict[label]

        self.idx_to_token_dict[idx] = label
        self.token_to_idx_dict[label] = idx

        if idx not in self.frequencies:
            self.frequencies[idx] = 1
        else:
            self.frequencies[idx] += 1
        return idx

    def add_special_word(self, label: str, idx=None):
        idx = self.add_to_dict(label, idx)
        self.special.append(idx)

    def add_special_words(self, labels: List[str]):
        for label in labels:
            self.add_special_word(label)

    def prune(self, size: int, frequency: int, mode='size'):
        """
        there are two model
        return the size most frequency entries with the mode `size`
        mode = [size, frequency]
        """

        # frequency{idx:freq}
        if mode not in ['size', 'frequency']:
            raise CheckParameterError(mode, ['size', 'frequency'])
        if size >= self.size or frequency <= 1:
            return self
        freq = torch.Tensor([v for _, v in self.frequencies])
        freq, idx = torch.sort(freq, dim=0, descending=True)
        new_vocab = Vocab(self.special_words, language='en', lower=self.lower, checkpoint=self.checkpoint)
        if mode == 'size':
            for i in idx[:size]:
                new_vocab.add_to_dict(self.idx_to_token_dict[i.item()])
            return new_vocab
        elif mode == 'frequency':
            for i, ids in enumerate(idx):
                if freq[i] < frequency: break
                new_vocab.add_to_dict(self.idx_to_token_dict[idx.item()])
                new_vocab.frequencies[idx.item()] = self.frequencies[idx.item()]
        return new_vocab

    def lookup(self, key: str, default=Constant.UNK['id']):
        """
            Use token to find idx
        """
        if self.checkpoint == 'default':
            try:
                return self.token_to_idx_dict[key]
            except KeyError:
                return default
        else:
            return self.token_to_idx_F(key)

    def convert_to_indices(self, labels):
        """
        Convert a list token to indices
        """
        indexes = [self.lookup(label) for label in labels]
        return torch.LongTensor(indexes)

    def convert_to_labels(self, ids, stop_list: List[str] = None):
        if not stop_list: stop_list = [Constant.PAD['token'], Constant.EOS['token']]
        return [self.get_label(i) for i in ids if i not in stop_list]

    def word_count(self, corpus):
        for sent in tqdm(corpus, desc="[Counting words]"):
            for word in sent:
                word = word.item() if isinstance(word, torch.Tensor) else word
                if word not in self.frequencies:
                    self.frequencies[word] = 0
                else:
                    self.frequencies[word] += 1
