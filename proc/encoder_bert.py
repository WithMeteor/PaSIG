#!/user/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(current_path, '..'))
sys.path.append(root_path)

import json
import math
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from src.bert_model import MyBERT
from transformers import BertTokenizer  # , BertModel
from proc.dataset_config import data_args, get_dataset


def load_bert_tokenizer(bert_model_path):
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    return tokenizer


def load_bert_model(bert_model_path, dataset):
    # If you want to obtain the values of each hidden layer, you need to set it this way
    # model = BertModel.from_pretrained(bert_model_path, output_hidden_states=True).cuda()
    hid_dim = 768
    num_classes = data_args[dataset]['num_classes']
    model = MyBERT(hid_dim, num_classes).cuda()
    # todo: Annotate next line of code, if you want to use the BERT before fine-tuning
    model.load_state_dict(torch.load(bert_model_path))
    model.eval()
    return model


def load_text(dataset):
    with open('./data/temp/{}.tokens.all.json'.format(dataset), 'r', encoding='utf-8') as data_file:
        tokens_list = json.load(data_file)
    return tokens_list


def load_word(dataset):
    tokens = []
    with open('./data/temp/{}.vocab.train.txt'.format(dataset), 'r', encoding='utf-8') as data_file:
        for line in data_file.readlines():
            this_token = line.strip()
            tokens.append(['[CLS]', this_token, '[SEP]'])
    return tokens


def get_batch_data(data_list, batch_size=32):
    all_num = len(data_list)
    batch_num = math.ceil(all_num / batch_size)
    batch_list = []
    for i in range(batch_num):
        batch_list.append(data_list[i * batch_size: (i + 1) * batch_size])
    return batch_list


def encode_batch_text(batch_list, bert_model, bert_tokenizer):
    encode_list = []
    for batch in tqdm(batch_list):
        # Convert token to vocabulary indices
        tokens_ids = [torch.tensor(bert_tokenizer.convert_tokens_to_ids(tokens)) for tokens in batch]
        attn_masks = [torch.ones(len(tokens)).bool() for tokens in batch]

        # Convert inputs to PyTorch tensors
        tokens_tensor = nn.utils.rnn.pad_sequence(tokens_ids, batch_first=True).to('cuda')
        masks_tensor = nn.utils.rnn.pad_sequence(attn_masks, batch_first=True).to('cuda')
        act_output, cls_output, word_output = bert_model(tokens_tensor, masks_tensor)
        text_encoded = cls_output.cpu().detach().numpy()
        encode_list.extend(text_encoded)
    encode_array = np.array(encode_list)
    return encode_array


def encode_batch_word(batch_list, bert_model, bert_tokenizer):
    encode_list = []
    for batch in tqdm(batch_list):
        # Convert token to vocabulary indices
        tokens_ids = [torch.tensor(bert_tokenizer.convert_tokens_to_ids(tokens)) for tokens in batch]
        attn_masks = [torch.ones(len(tokens)).bool() for tokens in batch]

        # Convert inputs to PyTorch tensors
        tokens_tensor = nn.utils.rnn.pad_sequence(tokens_ids, batch_first=True).to('cuda')
        masks_tensor = nn.utils.rnn.pad_sequence(attn_masks, batch_first=True).to('cuda')
        act_output, cls_output, word_output = bert_model(tokens_tensor, masks_tensor)
        word_encoded = word_output.cpu().detach().numpy()
        encode_list.extend(word_encoded)
    encode_array = np.array(encode_list)
    return encode_array


def save_embed(dataset, encode_array, mode='text'):
    np.save("./data/embed_bert/{}.embed.{}.cls.in.npy".format(dataset, mode), encode_array)
    # np.save("./data/embed_bert/{}.embed.{}.nofine.npy".format(dataset, mode), encode_array)


if __name__ == '__main__':
    Dataset = get_dataset()
    BertModelPath = f"./out/{Dataset}_bert.pth"
    print(f'Loading model from file: {BertModelPath} ...')
    BertTokenizerPath = './ptm/bert-base-uncased'
    BertModel = load_bert_model(BertModelPath, Dataset)
    BertTokenizer = load_bert_tokenizer(BertTokenizerPath)

    BatchSize = 8

    Texts = load_text(Dataset)
    Batch = get_batch_data(Texts, batch_size=BatchSize)

    print('Encoding texts...')
    Encode = encode_batch_text(Batch, BertModel, BertTokenizer)
    save_embed(Dataset, Encode, mode='text')

    Words = load_word(Dataset)
    Batch = get_batch_data(Words, batch_size=BatchSize)

    print('Encoding words...')
    Encode = encode_batch_word(Batch, BertModel, BertTokenizer)
    save_embed(Dataset, Encode, mode='word')
