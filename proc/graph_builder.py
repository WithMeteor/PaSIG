#!/user/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(current_path, '..'))
sys.path.append(root_path)

import json
import numpy as np
# from tqdm import tqdm
import scipy.sparse as sp
from proc.my_tfidf import TfIdf
# from proc.my_pmi import calculate_pmi
from proc.dataset_config import get_dataset


def load_data(dataset):
    with open('./data/temp/{}.tokens.all.json'.format(dataset), 'r', encoding='utf-8') as data_file:
        ori_tokens_list = json.load(data_file)

    tokens_list = [tokens[1: -1] for tokens in ori_tokens_list]  # remove special tokens [CLS] & [SEP]

    train_idx = np.load('./data/temp/{}.idx.train.npy'.format(dataset))
    valid_idx = np.load('./data/temp/{}.idx.valid.npy'.format(dataset))
    test_idx = np.load('./data/temp/{}.idx.test.npy'.format(dataset))

    train_tokens = [tokens_list[i] for i in train_idx]
    valid_tokens = [tokens_list[i] for i in valid_idx]
    test_tokens = [tokens_list[i] for i in test_idx]

    vocab = []
    with open('./data/temp/{}.vocab.train.txt'.format(dataset), 'r', encoding='utf-8') as data_file:
        for line in data_file.readlines():
            vocab.append(line.strip())

    return train_tokens, valid_tokens, test_tokens, vocab


def calculate_tfidf(train_tokens):
    train_corpus = []
    for tokens in train_tokens:
        train_corpus.append(" ".join(tokens))
    tfidf_model = TfIdf()
    tfidf_model.build_corpus(train_corpus)
    return tfidf_model


def build_train_graph(train_tokens, vocab, tfidf_model, threshold=0.01):
    """
    build train graph
    :param train_tokens: train corpus
    :param vocab: vocabulary
    :param tfidf_model: tf-idf values calculated by train corpus
    :param threshold: the minimum threshold value of tfidf
    :return:
    """
    train_row = []  # row of co-occurrence matrix
    train_col = []  # column of co-occurrence matrix
    train_weight = []  # weight of co-occurrence matrix

    train_size = len(train_tokens)
    vocab_size = len(vocab)
    token2id = dict()
    for i in range(vocab_size):
        token2id[vocab[i]] = i

    tfidf_values = tfidf_model.get_tfidf()

    # doc-word edges
    for i in range(train_size):
        tokens = train_tokens[i]
        for token in tokens:
            j = token2id[token]
            tf_idf = tfidf_values[i][token]
            if tf_idf < threshold:  # filter too-small-weight edges
                continue
            train_row.append(vocab_size + i)
            train_col.append(j)
            train_weight.append(tf_idf)
            # bi-directional message passing between word-nodes and train-doc-nodes in train graph
            train_row.append(j)
            train_col.append(vocab_size + i)
            train_weight.append(tf_idf)

    # word-word edges
    # train_row.extend(pmi_row)
    # train_col.extend(pmi_col)
    # train_weight.extend(pmi_weight)

    return train_row, train_col, train_weight


def build_eval_graph(train_tokens, test_tokens, vocab, tfidf_model, threshold=0.01):
    """
    build test/valid graph
    :param train_tokens: train corpus
    :param test_tokens: test/valid corpus
    :param vocab: vocabulary
    :param tfidf_model: tf-idf values calculated by train corpus
    :param threshold: the minimum threshold value of tfidf
    :return:
    """
    test_row = []  # row of co-occurrence matrix
    test_col = []  # column of co-occurrence matrix
    test_weight = []  # weight of co-occurrence matrix

    train_size = len(train_tokens)
    test_size = len(test_tokens)
    vocab_size = len(vocab)
    token2id = dict()
    for i in range(vocab_size):
        token2id[vocab[i]] = i

    # doc-word edges
    for i in range(test_size):
        tokens = test_tokens[i]
        sentence = " ".join(tokens)
        tfidf_model.add_sentence(sentence)
        tfidf_value = tfidf_model.get_sentence_tfidf(sentence)
        for token in tokens:
            # if test token not appears in the train corpus, then do not add this edge
            if token not in token2id:
                continue
            j = token2id[token]
            tf_idf = tfidf_value[token]
            if tf_idf < threshold:  # filter too-small-weight edges
                continue
            # one-directional message passing from word-nodes to test-doc-nodes in test graph
            test_row.append(j)
            test_col.append(vocab_size + train_size + i)
            test_weight.append(tf_idf)

    # doc-doc edges
    # todo 1: 这里暂时不考虑在测试文档节点和训练文档节点之间直接连边

    return test_row, test_col, test_weight


def get_svg_len(dataset, tokens_list):
    len_list = [len(tokens) for tokens in tokens_list]
    print(dataset)
    print(np.mean(len_list))


def build_graph(dataset, save=True):
    print('Dataset:', dataset)
    # Load and shuffle data
    train_tokens, valid_tokens, test_tokens, vocab = load_data(dataset)

    # get_svg_len(dataset, train_tokens+valid_tokens+test_tokens)

    # Calculating TF-IDF
    print('Calculating TF IDF...')
    tfidf_model = calculate_tfidf(train_tokens)

    # # Calculating PMI
    # print('Calculating PMI...')
    # pmi_row, pmi_col, pmi_weight = calculate_pmi(train_tokens, vocab)

    print('Building train graph...')
    train_row, train_col, train_weight = build_train_graph(train_tokens, vocab, tfidf_model)
    # [Train Graph] node_size = vocab_size + train_size
    node_size = len(vocab) + len(train_tokens)
    train_adj = sp.csr_matrix(
        (train_weight, (train_row, train_col)), shape=(node_size, node_size), dtype=np.float32)
    print('Train doc num:', node_size - len(vocab))
    print('Train graph edge num:', train_adj.nnz)

    print('Building valid graph...')
    valid_row, valid_col, valid_weight = build_eval_graph(train_tokens, valid_tokens, vocab, tfidf_model)
    valid_row, valid_col, valid_weight = train_row + valid_row, train_col + valid_col, train_weight + valid_weight
    # [Valid Graph] node_size = vocab_size + train_size + valid_size
    node_size = len(vocab) + len(train_tokens) + len(valid_tokens)
    valid_adj = sp.csr_matrix(
        (valid_weight, (valid_row, valid_col)), shape=(node_size, node_size), dtype=np.float32)
    print('Valid doc num:', node_size - len(vocab))
    print('Valid graph edge num:', valid_adj.nnz)

    print('Building test graph...')
    test_row, test_col, test_weight = build_eval_graph(train_tokens+valid_tokens, test_tokens, vocab, tfidf_model)
    test_row, test_col, test_weight = valid_row + test_row, valid_col + test_col, valid_weight + test_weight
    # [Test Graph] node_size = vocab_size + train_size + valid_size + test_size
    node_size = len(vocab) + len(train_tokens + valid_tokens) + len(test_tokens)
    test_adj = sp.csr_matrix(
        (test_weight, (test_row, test_col)), shape=(node_size, node_size), dtype=np.float32)
    print('Test doc num:', node_size - len(vocab))
    print('Test graph edge num:', test_adj.nnz)
    print('Vocab size:', len(vocab))

    # todo: 注意邻接矩阵的索引顺序：先是单词索引，再是训练节点索引，其次是验证节点索引，最后是测试节点索引

    if save:
        sp.save_npz('./data/graph/{}.adj.train.npz'.format(dataset), train_adj)  # only-tfidf
        sp.save_npz('./data/graph/{}.adj.valid.npz'.format(dataset), valid_adj)  # only-tfidf
        sp.save_npz('./data/graph/{}.adj.test.npz'.format(dataset), test_adj)  # only-tfidf


if __name__ == '__main__':
    Dataset = get_dataset()
    build_graph(Dataset, save=True)  # False
