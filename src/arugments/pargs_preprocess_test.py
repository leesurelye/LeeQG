# -*- coding: utf-8 -*-
# @Time    : 2021/9/17 17:59
# @Author  : Lees ure
# @File    : pargs_preprocess_test.py
# @Software: PyCharm

"""
this is the parameter of test this model
"""
import argparse

def add_options(parser:argparse.ArgumentParser):
    parser.add_argument('-train_src',default="/home1/liyue/NLP_QG/RL-for-Question-Generation/src/RL-for-QG/data/train/train.src.txt",
                        help="training source data")

    parser.add_argument('-train_target', default="/home1/liyue/NLP_QG/RL-for-Question-Generation/src/RL-for-QG/data/train/train.tgt.txt",
                        help='training target data')

    parser.add_argument('-valid_src',default="/home1/liyue/NLP_QG/RL-for-Question-Generation/src/RL-for-QG/data/dev/dev.src.txt",
                        help='validation source data')

    parser.add_argument('-valid_target', default="/home1/liyue/NLP_QG/RL-for-Question-Generation/src/RL-for-QG/data/dev/dev.tgt.txt",
                        help="validation target data")

    parser.add_argument('-train_dataset', help="Path to the training dataset")
    parser.add_argument('-valid_dataset', help="Path to the validation dataset")

    parser.add_argument('-train_graph', help="Path to the training source graph data")
    parser.add_argument('-valid_graph', help="Path to the validation source graph data")

    parser.add_argument('-train_ans', default="/home1/liyue/NLP_QG/RL-for-Question-Generation/src/RL-for-QG/data/train/train.ans.txt",
                        help="Path to training answer")

    parser.add_argument('-valid_ans',default="/home1/liyue/NLP_QG/RL-for-Question-Generation/src/RL-for-QG/data/dev/dev.ans.txt",
                        help="Path to the validation answer")

    parser.add_argument('-src_vocab_size', type=int, default=50000)
    parser.add_argument('-tgt_vocab_size', type=int, default=50000)

    parser.add_argument('-src_words_min_frequency', type=int, default=1)
    parser.add_argument('-tgt_words_min_frequency', type=int, default=1)

    parser.add_argument('-feature', default=False, action='store_true')
    parser.add_argument('-node_feature', default=False, action='store_true')

    parser.add_argument('-train_feats', default=[], nargs='+', type=str, help="Train files of source features")
    parser.add_argument('-valid_feats', default=[], nargs='+', type=str, help="Valid files of source features")

    parser.add_argument('-answer', default='none', choices=['con', 'sep', 'none'],
                        help="the way insert answer information: `con` for concatenating in src; "
                             "`sep` for separate ans encoder; `none` for no answer information")

    parser.add_argument('-ans_feature', default=False, action='store_true', help="answer features")

    parser.add_argument('-train_ans_feats', default=[], nargs='+', type=str, help="Train files of answer features")
    parser.add_argument('-valid_ans_feats', default=[], nargs='+', type=str, help="Valid files of answer features")

    parser.add_argument('-share_vocab', default=True, action='store_true',
                        help="Share the source and valid vocabulary")

    parser.add_argument('-bert_tokenizer',default="bert-base-cased",
                        help="the pretrained model")
    parser.add_argument('-src_seq_length', default=256, help="source sequence length")
    parser.add_argument('-tgt_seq_length', default=64, help="target sequence length")

    # TODO copy mechanism
    parser.add_argument('-copy', default=True, help="I don't know")
    parser.add_argument('-pre_trained_vocab', default=None, help="Path to the pretrained vocab file")
    parser.add_argument('-word_vec_size', default=300, help="word vector size")
    parser.add_argument('-save_path', help="Output file for the pretrained data")

