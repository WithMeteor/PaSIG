#!/user/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(current_path, '..'))
sys.path.append(root_path)

import time
import torch
import datetime
# import numpy as np
from torch import nn
# from tqdm import tqdm
from copy import deepcopy
from sklearn import metrics
from src.bert_model import MyBERT
from src.utils import setup_logger
from argparse import ArgumentParser
from proc.dataset_config import data_args
from src.dataset_text import get_data_loader


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', help='Dataset name.', default='R52')  # mr, ohsumed, 20ng, R8, R52
    parser.add_argument('--gpu', help='ID of available gpu.', type=int, default=0)
    # epoch set: 75 for ohsumed, 20 for mr & R8, 30 for 20ng & R52
    parser.add_argument('--epochs', help='Number of epochs to train.', type=int, default=20)
    parser.add_argument('--batch_size', help='Size of batch for backpropagation.', type=int, default=8)
    parser.add_argument('--hidden_dim', help='Number of units in hidden layer.', type=int, default=768)
    parser.add_argument('--learning_rate', help='Initial learning rate.', type=float, default=5e-6)  # 1e-6
    parser.add_argument('--dropout', help='Dropout rate (1 - keep probability).', type=float, default=0.0)
    parser.add_argument('--weight_decay', help='Weight for L2 loss on embedding matrix.', type=float, default=0.0)
    parser.add_argument('--early_stopping', help='Tolerance for early stopping (# of epochs).', type=int, default=60)
    parser.add_argument('--fix_seed', help='Fix the random seed.', action='store_true')
    parser.add_argument('--seed', help='The random seed.', default=123)
    parser.add_argument('--log_dir', help='Log file path.', default='./log')
    parser.add_argument('--out_dir', help='Model save path.', default='./out')

    model_args = parser.parse_args()
    return model_args


def train_eval(cate, model, data_loader, criterion, optimizer, device):
    model.train() if cate == "train" else model.eval()

    preds = []
    labels = []
    loss_sum = 0

    for batch_input, batch_mask, batch_label in data_loader:

        # Train on BERT
        act_output, cls_input, word_input = model(batch_input.to(device), batch_mask.to(device))

        batch_loss = criterion(act_output, batch_label.to(device))

        if cate == 'train':
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        preds.append(act_output.max(dim=1)[1].data)
        labels.append(batch_label.data)
        loss_sum += batch_loss.data

    loss = loss_sum / len(data_loader)
    preds = torch.cat(preds).tolist()
    labels = torch.cat(labels).tolist()
    acc = metrics.accuracy_score(labels, preds) * 100
    return acc, loss, preds, labels


def main():
    args = get_args()
    dataset = args.dataset  # ohsumed
    batch_size = args.batch_size  # 32
    num_epoch = args.epochs
    log_dir = args.log_dir
    out_dir = args.out_dir
    current_time = datetime.datetime.now()
    current_time_str = current_time.strftime("%Y%m%d%H%M%S")
    logger = setup_logger('Vanilla BERT', f'{log_dir}/{dataset}_bert_{current_time_str}.log')

    logger.info(f"load dataset: {dataset}.")
    train_data, valid_data, test_data = get_data_loader(dataset, batch_size=batch_size)
    logger.info(f"train size:{len(train_data)}, valid size:{len(valid_data)}, test size:{len(test_data)}")

    hid_dim = args.hidden_dim
    num_classes = data_args[dataset]['num_classes']

    gpu_id = args.gpu
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    bert_model = MyBERT(hid_dim, num_classes)
    bert_model = bert_model.to(device)

    logger.info(bert_model)

    learning_rate = args.learning_rate
    loss_func = nn.CrossEntropyLoss()
    optimizer_bert = torch.optim.Adam(bert_model.parameters(), lr=learning_rate)

    best_acc = 0.
    best_param = None

    for epoch in range(num_epoch):

        start_time = time.time()
        train_acc, train_loss, _, _ = train_eval('train', bert_model, train_data, loss_func, optimizer_bert, device)
        valid_acc, valid_loss, _, _ = train_eval('valid', bert_model, valid_data, loss_func, optimizer_bert, device)
        test_acc, test_loss, _, _ = train_eval('test', bert_model, test_data, loss_func, optimizer_bert, device)
        cost = time.time() - start_time

        if best_acc < valid_acc:  # save the best model on the validation set
            best_acc = valid_acc
            best_param = deepcopy(bert_model.state_dict())

        logger.info((f"epoch={epoch + 1}/{num_epoch}, cost={cost:.2f}, "
                     f"train:[{train_loss:.4f}, {train_acc:.2f}%], "
                     f"valid:[{valid_loss:.4f}, {valid_acc:.2f}%], "
                     f"test:[{test_loss:.4f}, {test_acc:.2f}%], "
                     f"best_acc={best_acc:.2f}%"))

    bert_model.load_state_dict(best_param)  # load best param
    test_loss, test_acc, test_preds, test_labels = train_eval(
        'test', bert_model, test_data, loss_func, optimizer_bert, device)

    logger.info("Test Precision, Recall and F1-Score...")
    logger.info(metrics.classification_report(test_labels, test_preds, digits=4))
    logger.info("Macro average Test Precision, Recall and F1-Score...")
    logger.info(metrics.precision_recall_fscore_support(test_labels, test_preds, average='macro'))
    logger.info("Micro average Test Precision, Recall and F1-Score...")
    logger.info(metrics.precision_recall_fscore_support(test_labels, test_preds, average='micro'))

    model_path = os.path.join(out_dir, f'{dataset}_bert.pth')
    torch.save(best_param, model_path)


if __name__ == '__main__':
    main()
