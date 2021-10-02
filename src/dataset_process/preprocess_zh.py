# -*- coding: utf-8 -*-
# @Time    : 2021/9/19 12:51
# @Author  : Leesure
# @File    : preprocess_zh.py
# @Software: PyCharm
import json
import os
from tqdm import tqdm

"""
    extract sentences question and answer from raw data without tokenize
"""
# json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)


def get_data(file_path: str):
    file = open(file_path, encoding='UTF-8')
    save_file_name = file_path.split("_")[1] + '.json'
    with open(save_file_name, mode='a+') as save_file:
        data_raw = json.load(file)['data']
        save_file.write(json.dumps(data_raw, indent=2, ensure_ascii=False))
    save_file.close()
    file.close()


def get_sentence(context: str, index: int):
    """
     TODO get_sentence 存在一些问题
    """
    begin = -1
    end = -1
    i = 0
    while i < len(context):
        if context[i] == "。" and i < index:
            begin = i
        if i >= index and context[i] == "。":
            end = i
            break
        i += 1
    return context[begin + 1: end + 1]


def split_data_into_questions(data: list, data_type: str, output_folder: str):
    """
        @param data: 传入的json 对象，train.json | test.json | test.json
        @param data_type: 传入的数据集类型
        @param output_folder: the output file path
    """
    # 开始写入文件
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    tgt_ = open(f'{output_folder}/tgt-{data_type}.txt', mode='a', encoding='utf-8')
    para_ = open(f'{output_folder}/para-{data_type}.txt', mode='a', encoding='utf-8')
    src_ = open(f'{output_folder}/src-{data_type}.txt', mode='a', encoding='utf-8')
    ans_ = open(f'{output_folder}/ans-{data_type}.txt', mode='a', encoding='utf-8')
    for item in tqdm(data, desc=f"[Processing {data_type} data]"):
        paragraphs = item['paragraphs']
        for para in paragraphs:
            context = para['context']
            qas = para["qas"]
            for question in qas:
                ques = question["question"]
                ans = question['answers'][0]['text']
                index = question["answers"][0]["answer_start"]
                sentence = get_sentence(context, index)
                src_.write(sentence + '\n')
                para_.write(context + '\n')
                tgt_.write(ques + '\n')
                ans_.write(ans + '\n')
    tgt_.close()
    para_.close()
    src_.close()
    ans_.close()


def process_into_json(data: list, data_type: str, output_folder: str):
    """
        将数据处理成json文本
    """
    if not os.path.exists(f"{output_folder}"):
        os.mkdir(output_folder)

    store_file = open(f'{output_folder}/{data_type}.json', mode='w', encoding='utf-8')
    json_data = []
    for item in tqdm(data, desc=f"[Processing {data_type} data]"):
        paragraphs = item['paragraphs']
        for para in paragraphs:
            context = para['context']
            qas = para["qas"]
            for question in qas:
                ques = question["question"]
                ans = question['answers'][0]['text']
                index = question["answers"][0]["answer_start"]
                sentence = get_sentence(context, index)
                # src_.write(sentence + '\n')
                json_tmp = {'sentence': sentence, 'context': context, 'ques': ques, 'ans': ans}
                json_data.append(json_tmp)

    json.dump(json_data, store_file, ensure_ascii=False)
    store_file.close()


def process_raw_data(**kwargs):
    assert kwargs['source_folder'], "the source folder is None"
    assert kwargs['output_folder'], "the output folder is None"

    train_file = open(f"{kwargs['source_folder']}/DRCD_training_simplified.json", encoding='utf-8')
    train_list = json.load(train_file)["data"]
    split_data_into_questions(train_list, data_type="train", output_folder=kwargs['output_folder']+"/train")
    # process_into_json(train_list, data_type="train", output_folder=kwargs['output_folder']+"/train")
    train_file.close()

    dev_file = open(f"{kwargs['source_folder']}/DRCD_dev_simplified.json", encoding='utf-8')
    dev_list = json.load(dev_file)["data"]
    split_data_into_questions(dev_list, data_type="dev", output_folder=kwargs['output_folder']+"/dev")
    # process_into_json(dev_list, data_type="dev", output_folder=kwargs['output_folder']+"/dev")
    dev_file.close()

    test_file = open(f"{kwargs['source_folder']}/DRCD_test_simplified.json", encoding='utf-8')
    test_list = json.load(test_file)["data"]
    split_data_into_questions(test_list, data_type="test", output_folder=kwargs['output_folder']+"/test")
    # process_into_json(test_list, data_type="test", output_folder=kwargs['output_folder']+"/test")
    test_file.close()


if __name__ == '__main__':
    process_raw_data(source_folder="/home1/liyue/NLP_QG/RL-for-Question-Generation/LeeQG/dataset/raw",
                     output_folder="/home1/liyue/NLP_QG/RL-for-Question-Generation/LeeQG/dataset/preprocess")

