# -*- coding: utf-8 -*-
"""
@Time    : 2021/9/13 20:00
@Author  : Le Yu E
@FileName: preprocess.py
@Software: PyCharm
"""
import argparse
import logging
import math

import spacy
# since we don't need process raw data, we got the processed data
# import json
import torch
# python process bar
from tqdm import tqdm
from typing import Dict
from LeeQG import global_logger
from dataset_process import Constant
from dataset_process.Vocab import Vocab
from exception.GlobalException import ParameterError

# from transformers import AutoTokenizer
from typing import List


# import pargs
from arugments import pargs_preprocess_test

logger = global_logger.get_logger("preprocess.py")
en_nlp = spacy.blank("en")
zh_nlp = spacy.blank("zh")


# def process_file(filename, config=None, word_counter=None, char_counter=None):
#     data = json.load(open(filename, 'r'))
#
#     examples = []
#     eval_examples = {}
#  TODO i don't know how to do this
#  output = Parallel(n_jobs=12,verbose=10)(delayed())


def load_file(filename):
    """
        load file data into list
        @return type
        [
            [a,b,c]
            [a,b,c]
            ...
        ]
        NOTE that: this function just use for english language
    """
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.read().strip()
    data = data.split('\n')
    data = [sent.split(' ') for sent in data]
    return data


def filter_data(data_list, args):
    """
        pre process sentence add special word at begin and end of sentence
        For example :
        this is a sentence ===> [BOS]this is a sentence[EOS]

        @input: data_list: [src, tgt,[ans, [feat, ]

        @Return res, indexes
    """
    data_num = len(data_list)
    idx = list(range(len(data_list[0])))
    res = [[] for _ in range(data_num)]
    indexes = []
    for i, src, tgt in zip(idx, data_list[0], data_list[1]):
        src_len = len(src)
        tgt_len = len(tgt)
        if src_len - 1 <= args.src_seq_length and tgt_len - 1 <= args.tgt_seq_length \
                and src_len >= 10 and tgt_len > 0:
            res[0].append(src)
            res[1].append([Constant.BOS['token']] + tgt + [Constant.EOS['token']])
            indexes.append(i)
            for j in range(2, data_num):
                sent = data_list[j][i]
                res[j].append(sent)
    logger.info(f"change data size from {len(data_list[0])} to {len(res[0])}")
    return res, indexes


def get_data(file_dict:Dict):
    res = {'src': None, 'tgt': None, 'ans': None, 'feature': None, 'ans_feature': None}
    for k, v in file_dict.items():
        if isinstance(v, list):
            res[k] = [load_file(f) for f in v]
        else:
            res[k] = load_file(v)

    return res


def load_vocab(filename: str):
    """
    load vocabulary from given dataset file
    """
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read().strip().split('\n')
    text = [word.split(' ') for word in text]
    vocab_dict = {word[0]: word[1:] for word in text}
    vocab_dict = {k: [float(d) for d in v] for k, v in vocab_dict.items()}
    return vocab_dict


# def _process_article(article, config=None):
#     paragraphs = article['context']
#     # some article in the full wiki dev/test sets have zero paragraphs
#     if len(paragraphs) == 0:
#         paragraphs = [['some title', 'some random stuff']]
#     text_context, context_tokens, context_chars = '', [], []
# TODO since this part is to process the raw data, but we already got the processed data


def word_tokenize(sent):
    """chinese works tokenizer"""
    doc = en_nlp(sent)
    return [token.text for token in doc]


def _merge_ans(src, ans):
    """
     src , ans ==> src [SEP] ans [SEP]
    """
    return [s + [Constant.SEP['token']] + a + [Constant.SEP['token']] for a, s in zip(src, ans)]


def process_data(args):
    """
        get train data and validation data
    """
    train_files = {'src': args.train_src, 'tgt': args.train_target}
    valid_files = {'src': args.valid_src, 'tgt': args.valid_target}

    if args.answer != 'none':
        if not args.train_ans and not args.valid_ans:
            raise ParameterError("Answer files of train and valid must be given")
        train_files['ans'], valid_files['ans'] = args.train_ans, args.valid_ans

    if args.feature:
        if len(args.train_feats) != len(args.valid_feats) or len(args.train_feats) <= 0:
            raise ParameterError("the length of train_features and valid_feature isn't equal")
        train_files['feats'], valid_files['feats'] = args.train_feats, args.valid_feats

    train_data = get_data(train_files)
    valid_data = get_data(valid_files)
    return train_data, valid_data


def build_vocabulary(train_data, valid_data, args):
    """
        Build vocabulary
    """
    logger.info('Begin build vocabulary...')
    vocabularies = {}
    if args.answer == 'con':  # chain with answer
        train_data['src'] = _merge_ans(train_data['src'], train_data['ans'])
        valid_data['src'] = _merge_ans(train_data['src'], train_data['ans'])

    trans = True if args.answer == 'con' else False
    sep = True if args.answer == 'con' else False
    if args.bert_tokenizer:  # e.g args.bert_tokenizer = 'bert-base-uncased'
        bert_vocab = Vocab.from_opt(pretrained=args.bert_tokenizer, trans=True, separate=sep,
                                    tgt=True, language='en', lower=True)
        if args.share_vocab:
            logger.info(" Building src & target vocabulary...")
            vocabularies['src'] = vocabularies['tgt'] = bert_vocab
        else:
            logger.info("Building src vocabulary...")
            vocabularies['src'] = bert_vocab
            logging.info("Building tgt vocabulary...")
            vocabularies['tgt'] = Vocab.from_opt(corpus=train_data['tgt'], lower=True, mode='size',
                                                 separate=False, tgt=True, trans=False, size=args.src_vocab_size,
                                                 frequency=args.tgt_words_min_frequency)
    else:
        if args.share_vocab:
            logging.info("Building src & target vocabulary...")
            corpus = train_data['src'] + train_data['tgt']
            default_vocab = Vocab.from_opt(corpus=corpus, lower=True, trans=trans, separate=sep, mode='size')
            vocabularies['src'] = vocabularies['tgt'] = default_vocab
        else:
            logging.info("Building src vocabulary ...")
            vocabularies['src'] = Vocab.from_opt(corpus=train_data['src'], lower=True, mode='size',
                                                 separate=sep, tgt=False, trans=trans,
                                                 size=args.src_vocab_size, frequency=args.src_words_min_frequency)
            logger.info("Building tgt vocabulary...")
            vocabularies['tgt'] = Vocab.from_opt(corpus=train_data['tgt'], lower=True, mode='size',
                                                 separate=False, tgt=True, trans=False, size=args.src_vocab_size,
                                                 frequency=args.tgt_words_min_frequency)
    if args.feature and train_data['feats']:
        vocabularies['feats'] = [
            Vocab.from_opt(corpus=feat, lower=False, mode='size', trans=trans, separate=sep, tgt=False,
                           size=args.feat_vocab_size, frequency=args.feat_words_min_frequency)
            for feat in train_data['feats']]
    if args.ans_feature and train_data['ans_feature']:
        vocabularies['ans_feats'] = [
            Vocab.from_opt(corpus=feat, lower=False, mode='size', trans=trans, separate=sep, tgt=False,
                           size=args.feat_vocab_size, frequency=args.feat_words_min_frequency)
            for feat in train_data['ans_feats']]

    return vocabularies


def convert_word_to_idx(data, vocabularies, args):
    def lower_sentence(sent: List[str]):
        sent = [word.lower() if word not in Constant.SPECIAL_TOKEN_LIST else word
                for word in sent]
        return sent

    indexes, tokens = {}, {}
    for k, v in data.items():
        indexes[k] = [] if v else None
        tokens[k] = [] if v else None
        indexes[k] = [[] for _ in v] if k.count('feature') and v else indexes[k]
        tokens[k] = [[] for _ in v] if k.count('feature') and v else tokens[k]

    for i in tqdm(range(len(data['src'])), desc='[Convert tokens to ids]'):
        src, tgt = data['src'][i], data['tgt'][i]
        if args.bert_tokenizer:
            src = vocabularies['src'].tokenizer.tokenize(' '.join(src))
            if args.share_vocab:
                tgt = [Constant.CLS['token']] + vocabularies['tgt'].tokenizer.tokenize(' '.join(tgt)) + [
                    Constant.SEP['token']]
            else:
                tgt = [Constant.BOS['token']] + tgt + [Constant.EOS['token']]
        else:
            src = lower_sentence(src)
            tgt = lower_sentence(tgt)
            tgt = [Constant.BOS['token']] + tgt + [Constant.EOS['token']]

        # TODO why this range? len(src) is the single sequence length
        if len(src) in range(8, args.src_seq_length) and len(tgt) in range(5, args.tgt_seq_length):
            for k, text in data.items():
                if text:
                    if k == 'feats':
                        for j in range(len(indexes[k])):
                            indexes[k][j].append(vocabularies[k][j].convert_to_indices(text[j][i]))
                    else:
                        text = text[i]
                        text = src if k == 'src' else text
                        text = tgt if k == 'tgt' else text

                        tokens[k].append(text)
                        indexes[k].append(vocabularies[k].convert_to_indices(text))
    logger.info(f"change data size from {len(data['src'])} to {len(indexes['src'])}")
    return indexes, tokens


def wrap_copy_idx(split, tgt, tgt_vocab):
    """
    TODO
    """
    def warp_sent(s, t):
        sp_dict = {w: idx for idx, w in enumerate(s)}
        swt, cpt = [0 for _ in t], [0 for _ in t]
        for i, w in enumerate(t):
            # TODO why? <=10?
            if w in sp_dict and tgt_vocab.frequencies[tgt_vocab.lookup(w)] <= 10:
                swt[i] = 1
                cpt[i] = sp_dict[w]
        return torch.Tensor(swt), torch.Tensor(cpt)
    copy = [warp_sent(s,t) for s,t in zip(split, tgt)]
    switch, cp_tgt = [c[0] for c in copy], [c[1] for c in copy]
    return {'switch':switch,'tgt':tgt}



def word_to_index(train_data, valid_data, vocabulary, args):
    """
    TODO
    """
    train_indexes, train_tokens = convert_word_to_idx(train_data, vocabulary, args)
    valid_indexes, valid_tokens = convert_word_to_idx(valid_data, vocabulary, args)
    vocabulary['tgt'].word_count(train_indexes['tgt'] + valid_indexes['tgt'])
    if args.copy:
        train_indexes['copy'] = wrap_copy_idx(train_tokens['src'], train_tokens['tgt'], vocabulary['tgt'])
        valid_indexes['copy'] = wrap_copy_idx(valid_tokens['src'], valid_tokens['tgt'], vocabulary['tgt'])
    else:
        train_indexes['copy'] = {'switch':None, 'tgt':None}
        valid_indexes['copy'] = {'switch':None, 'tgt':None}
    return train_indexes, valid_indexes


def get_embedding(vocab_dict:Dict, vocab:Vocab, args):

    def get_vector(idx):
        word = vocab.idx_to_token_dict[idx]
        if idx in vocab.special or word not in vocab_dict:
            vector = torch.normal( 0, math.sqrt(6 / (1+args.word_vec_size)), size=args.word_vec_size)
        else:
            vector = torch.Tensor(vocab_dict[word])
        return vector

    embedding = [get_vector(idx) for idx in range(vocab.size)]
    embedding = torch.stack(embedding)

    logger.info(embedding.size())
    return embedding


def save_data(data,filepath):
    torch.save(data, filepath)


def prepare_pretrained_vectors(vocabularies:Dict , args):
    pre_trained_vocab = load_vocab(args.pre_trained_vocab) if args.pre_trained_vocab else None
    if not args.bert_tokenizer and pre_trained_vocab:
        pre_trained_src_vocab = get_embedding(pre_trained_vocab,vocabularies['src'], args)
        pre_trained_tgt_vocab = get_embedding(pre_trained_vocab, vocabularies['tgt'], args)
        if args.answer =='sep':
            pre_trained_ans_vocab = get_embedding(pre_trained_vocab, vocabularies['ans'], args)
        else:
            pre_trained_ans_vocab = None
    else:
        pre_trained_src_vocab = None
        pre_trained_tgt_vocab = None
        pre_trained_ans_vocab = None

    pre_trained_vector = {'src':pre_trained_src_vocab, 'tgt':pre_trained_tgt_vocab, 'ans':pre_trained_ans_vocab}
    vocabularies['pre_trained'] = pre_trained_vector




def main(args):
    train_data, valid_data = process_data(args)
    vocabulary = build_vocabulary(train_data, valid_data, args)
    train_indexes, valid_indexes = word_to_index(train_data, valid_data, vocabulary, args)
    prepare_pretrained_vectors(vocabulary, args)
    data = {'setting':args, 'dict':vocabulary,'train':train_indexes, 'valid':valid_indexes}
    save_data(data,args.save_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters for running preprocess.py')
    pargs_preprocess_test.add_options(parser)
    parameter = parser.parse_args()
    main(parameter)
