#!/user/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as fn
from transformers import BertModel


class MyBERT(nn.Module):
    def __init__(self, hidden_dim, output_dim, bert_dir='./ptm/bert-base-uncased'):
        super(MyBERT, self).__init__()
        self.hidden_dim = hidden_dim
        self.bert = BertModel.from_pretrained(bert_dir)
        self.classifier = nn.Linear(hidden_dim, output_dim)
        self.activation = fn.softmax

    def forward(self, seq_token_ids, seq_attention_masks):
        seq_bert_output = self.bert(
            input_ids=seq_token_ids,
            attention_mask=seq_attention_masks
        )

        # # 使用 [CLS]表示 作为分类器的输入
        # seq_embed = seq_bert_output[0]  # (batch_size, seq_len, hidden_dim)
        # cls_input = seq_embed[:, 0].reshape(-1, self.hidden_dim).contiguous()
        seq_embed = seq_bert_output.last_hidden_state
        cls_input = seq_embed[:, 0]
        word_input = seq_embed[:, 1]

        # 使用 pooler output 作为分类器的输入
        # cls_input = seq_bert_output.pooler_output

        # print('cls input shape:', cls_input.shape)

        cls_output = self.classifier(cls_input)
        # word_output = self.classifier(word_input)
        return self.activation(cls_output, dim=1), cls_input, word_input
