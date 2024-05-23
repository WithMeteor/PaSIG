#!/user/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(current_path, '..'))
sys.path.append(root_path)

import re
import sys
import json
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from nltk.corpus import stopwords
from transformers import BertTokenizer
from proc.dataset_config import data_args, get_dataset


def load_data(dataset):
    datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr', 'AGNews', 'WebKB', 'dblp', 'SST2']

    if dataset not in datasets:
        sys.exit("wrong dataset name")

    doc_list = []
    with open('./data/raw/{}.texts.txt'.format(dataset), 'rb') as f:
        for line in f.readlines():
            doc_list.append(line.strip().decode('latin1'))

    label_list = []
    with open('./data/raw/{}.labels.txt'.format(dataset), 'r', encoding='utf-8') as f:
        for line in f.readlines():
            label_list.append(line.strip())
    return doc_list, label_list


def save_doc(dataset, tokens_list):
    with open('./data/temp/{}.doc.json'.format(dataset), 'w', encoding='utf-8') as data_file:
        json.dump(tokens_list, data_file, ensure_ascii=False, indent=4)


def save_interm_doc(dataset, tokens_list):
    with open('./data/temp/{}.doc.interm.json'.format(dataset), 'w', encoding='utf-8') as data_file:
        json.dump(tokens_list, data_file, ensure_ascii=False, indent=4)


def save_vocab(dataset, vocab, mode='train'):
    vocab_str = '\n'.join(vocab)
    with open('./data/temp/{}.vocab.{}.txt'.format(dataset, mode), 'w') as f:
        f.write(vocab_str)


def save_labels(dataset, labels_list, mode='train'):
    # labels 2 targets
    label2index = {lb: i for i, lb in enumerate(sorted(set(labels_list)))}
    targets_list = [label2index[lb] for lb in labels_list]
    np.save("./data/temp/{}.labels.{}.npy".format(dataset, mode), targets_list)


def save_tokens(dataset, tokens_list, mode='train'):
    with open('./data/temp/{}.tokens.{}.json'.format(dataset, mode), 'w', encoding='utf-8') as data_file:
        json.dump(tokens_list, data_file, ensure_ascii=False, indent=4)


def save_inputs(dataset, inputs_list, mode='train'):
    np.save('./data/temp/{}.inputs.{}.npy'.format(dataset, mode), inputs_list)


def save_masks(dataset, masks_list, mode='train'):
    np.save('./data/temp/{}.masks.{}.npy'.format(dataset, mode), masks_list)


def filter_doc(doc_list, dataset):
    new_doc_list = []
    for doc in doc_list:
        doc = doc.lower()
        if dataset == '20ng':
            doc = doc.replace('\n', '')
            regexes_to_remove = [r'from:', r're:', r'subject:', r'distribution:', r'organization:',
                                 r'lines:', r'writes:', r'reply-to:']  # remove email head
            for r in regexes_to_remove:
                doc = re.sub(r, '', doc)
            doc = re.sub(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", "", doc)  # remove email address
            doc = re.sub(r"(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]", "", doc)  # rm url
            doc = re.sub(r"\d+(-\d+)", "", doc)  # remove tel
            doc = re.sub(r"\(\d+\)", "", doc)  # remove tel
            doc = re.sub(r"\d+(\.\d+)", "", doc)  # remove decimal
            doc = re.sub(r"[^A-Za-z0-9().,!?\']", " ", doc)

        else:
            doc = re.sub(r"[^A-Za-z0-9().,!?\']", " ", doc)

        new_doc_list.append(doc)
    return new_doc_list


def tokenize_doc(doc_list, max_len=512):
    bert_path = './ptm/bert-base-uncased'
    print('Tokenizing docs...')
    tokenizer = BertTokenizer.from_pretrained(bert_path)

    tokens_list = []

    for doc in tqdm(doc_list):
        doc = "[CLS] " + doc + " [SEP]"
        tokenized_doc = tokenizer.tokenize(doc)
        if len(tokenized_doc) > max_len:  # cut the document to prevent exceeding the maximum length
            tokenized_doc = tokenized_doc[:max_len - 1]
            tokenized_doc.append("[SEP]")  # add [SEP] in the end
        tokens_list.append(tokenized_doc)
    return tokens_list


def get_word_freq(tokens_list):
    word_freq = {}
    for word_list in tokens_list:
        for word in word_list:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
    return word_freq


def filter_words(word_list, word_freq, stop_words, least_freq):
    new_word_list = []
    for word in word_list:
        if word_freq[word] >= least_freq and word not in stop_words:  # remove stop words and rare words
            new_word_list.append(word)
    return new_word_list


def process_documents(dataset, do_filter=True):
    print('Dataset:', dataset)
    # cut sentence to max length 152 for 20ng, 512 for other dataset
    cut_len = 512
    if dataset == '20ng':
        cut_len = 152
    elif dataset == 'WebKB':
        cut_len = 256
    doc_list, label_list = load_data(dataset)

    if do_filter:
        stop_words = set(stopwords.words('english'))  # stop words
        least_freq = 5  # Minimum occurrences of words
    else:
        stop_words = {}
        least_freq = 0

    new_doc_list = []
    doc_len_list = []

    filtered_doc_list = filter_doc(doc_list, dataset)
    # save_interm_doc(dataset, filtered_doc_list)
    tokens_list = tokenize_doc(filtered_doc_list, max_len=cut_len)
    word_freq = get_word_freq(tokens_list)
    for tokens in tokens_list:
        word_list = filter_words(tokens, word_freq, stop_words, least_freq)
        new_doc = ' '.join(word_list)
        doc_size = len(word_list)

        new_doc_list.append(new_doc)
        doc_len_list.append(doc_size)

    save_doc(dataset, new_doc_list)
    split_shuffle_data(dataset)
    save_data(dataset)

    return doc_len_list


def token2id(tokens_list):
    bert_path = './ptm/bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(bert_path)

    inputs_list = []
    for tokens in tokens_list:
        tokens_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens))
        inputs_list.append(tokens_ids)
    inputs_array = nn.utils.rnn.pad_sequence(inputs_list, batch_first=True).numpy()

    max_seq_len = inputs_array.shape[1]
    masks_list = []
    for tokens in tokens_list:
        tokens_masks = [1] * len(tokens)
        padding_len = max_seq_len - len(tokens)
        tokens_masks += ([0] * padding_len)
        masks_list.append(tokens_masks)
    masks_array = np.array(masks_list)
    # print(inputs_array.shape, masks_array.shape)
    return inputs_array.tolist(), masks_array.tolist()


def save_data(dataset):
    with open('./data/temp/{}.doc.json'.format(dataset), 'r', encoding='utf-8') as data_file:
        doc_list = json.load(data_file)
    with open('./data/raw/{}.labels.txt'.format(dataset), 'r', encoding='utf-8') as f:
        labels_list = f.readlines()

    tokens_list = [doc.split() for doc in doc_list]
    labels_list = [lb.strip() for lb in labels_list]
    inputs_list, masks_list = token2id(tokens_list)

    train_size = data_args[dataset]['train_size']
    train_tokens = tokens_list[:train_size]

    all_tokens = list()
    for tokens in train_tokens:
        all_tokens.extend(tokens)
    vocab = set(all_tokens)
    vocab.remove('[CLS]')
    vocab.remove('[SEP]')
    vocab = list(vocab)

    save_vocab(dataset, vocab)
    save_tokens(dataset, tokens_list, 'all')
    save_labels(dataset, labels_list, 'all')
    save_inputs(dataset, inputs_list, 'all')
    save_masks(dataset, masks_list, 'all')


def split_shuffle_data(dataset):
    train_size = data_args[dataset]['train_size']
    valid_size = data_args[dataset]['valid_size']
    test_size = data_args[dataset]['test_size']

    train_valid_idx = list(range(train_size))
    random.shuffle(train_valid_idx)
    train_idx = train_valid_idx[:train_size - valid_size]
    valid_idx = train_valid_idx[train_size - valid_size:train_size]
    test_idx = list(range(train_size, train_size + test_size))

    np.save('./data/temp/{}.idx.train.npy'.format(dataset), train_idx)
    np.save('./data/temp/{}.idx.valid.npy'.format(dataset), valid_idx)
    np.save('./data/temp/{}.idx.test.npy'.format(dataset), test_idx)


if __name__ == '__main__':
    Dataset = get_dataset()

    # todo: 这里先去停用词和稀疏词，看一下是否会对 BERT 的微调结果有影响
    # todo: 需要对比不去停用词的条件下的 BERT 的训练效果，如果效果相同，建议去停用词/稀疏词
    # todo: 因为去停用词对传导式文本图的构建帮助很大
    SentLen = process_documents(Dataset, do_filter=True)
