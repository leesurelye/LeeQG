# -*- coding: utf-8 -*-
# @Time    : 2021/9/18 10:45
# @Author  : Leesure
# @File    : pargs_train_test.py
# @Software: PyCharm

import argparse

def add_data_options(parser:argparse.ArgumentParser):
    # the author add those data in the pt file
    parser.add_argument('-data', required=True)
    # TODO complete parameter
