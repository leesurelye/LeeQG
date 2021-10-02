# -*- coding: utf-8 -*-
# @Time    : 2021/9/15 12:19
# @Author  : Le Yu E
# @File    : pargs.py
# @Software: PyCharm

import argparse


def add_options(parser: argparse.ArgumentParser):
    parser.add_argument('-train_src', help="training source data")
    parser.add_argument('-train_target', help='training target data')

    parser.add_argument('-valid_src', help='validation source data')
    parser.add_argument('-valid_target', help="validation target data")

    parser.add_argument('-train_dataset', help="Path to the training dataset")
    parser.add_argument('-valid_dataset', help="Path to the validation dataset")

    parser.add_argument('-train_graph', help="Path to the training source graph data")
    parser.add_argument('-valid_graph', help="Path to the validation source graph data")

    parser.add_argument('-train_ans', help="Path to training answer")
    parser.add_argument('-valid_ans', help="Path to the validation answer")

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

    parser.add_argument('-share_vocab', default=False, action='store_true', help="Share the source and valid vocabulary")

