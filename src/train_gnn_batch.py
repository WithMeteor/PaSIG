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
from src.utils import setup_logger
from argparse import ArgumentParser
from proc.dataset_config import data_args
from torch.nn.functional import log_softmax
from src.dataset_graph_batch import get_data_loader
from src.gnn_model_batch import GCN, SAGE, GIN, GCAT


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', help='Dataset name.', type=str, default='ohsumed')  # mr, ohsumed, R8
    parser.add_argument('--gpu', help='ID of available gpu.', type=int, default=0)
    parser.add_argument('--epochs', help='Number of epochs to train.', type=int, default=200)  # 500
    parser.add_argument('--batch_size', help='Size of batch for backpropagation.', default=512)
    parser.add_argument('--gnn_model', help='Choose gnn model.', type=str, default='gcn')  # gcn, sage, gin, gcat
    parser.add_argument('--num_features', help='Number of input features.', type=int, default=768)
    parser.add_argument('--num_layers', help='Number of GNN layers.', type=int, default=2)
    parser.add_argument('--first_neigh', help='First order neighbor size for sampling.', type=int, default=10)
    parser.add_argument('--second_neigh', help='Second order neighbor size for sampling.', type=int, default=10)
    parser.add_argument('--hidden_dim', help='Number of units in hidden layer.', type=int, default=256)
    parser.add_argument('--learning_rate', help='Initial learning rate.', type=float, default=1e-3)
    parser.add_argument('--dropout', help='Dropout rate (1 - keep probability).', type=float, default=0.5)
    parser.add_argument('--weight_decay', help='Weight for L2 loss on embedding matrix.', type=float, default=0)
    parser.add_argument('--early_stopping', help='Tolerance for early stopping (# of epochs).', type=int, default=60)
    parser.add_argument('--fix_seed', help='Fix the random seed.', action='store_true')
    parser.add_argument('--seed', help='The random seed.', default=123)
    parser.add_argument('--log_dir', help='Log file path.', default='./log')
    parser.add_argument('--out_dir', help='Model save path.', default='./out')
    parser.add_argument('--add_edge', help='Add doc-doc edges to graph.', action='store_true')

    model_args = parser.parse_args()
    return model_args


def train_eval(cate, model, data, loader, criterion, optimizer, device):
    model.train() if cate == "train" else model.eval()

    preds, labels, loss_sum = [], [], 0.

    for batch_size, n_id, adjs in loader:
        adjs = [adj.to(device) for adj in adjs]
        cls_output = model(data.x[n_id], adjs, data.edge_attr)
        act_output = log_softmax(cls_output, dim=-1)
        loss = criterion(act_output[:batch_size], data.y[n_id[:batch_size]])

        if cate == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_sum += loss.data
        preds.append(act_output[:batch_size].max(dim=1)[1].data)
        labels.append(data.y[n_id[:batch_size]].data)

    preds = torch.cat(preds).tolist()
    labels = torch.cat(labels).tolist()
    loss = loss_sum / len(loader)
    acc = metrics.accuracy_score(labels, preds) * 100
    return acc, loss, preds, labels


def main():
    args = get_args()
    dataset = args.dataset
    batch_size = args.batch_size
    num_epoch = args.epochs
    log_dir = args.log_dir
    out_dir = args.out_dir

    first_neigh = args.first_neigh
    second_neigh = args.second_neigh

    gpu_id = args.gpu
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    num_classes = data_args[dataset]['num_classes']

    if args.gnn_model == 'sage':
        gnn_model = SAGE(args.num_features, args.hidden_dim,
                         num_classes, args.num_layers,
                         args.dropout)
    elif args.gnn_model == 'gin':
        gnn_model = GIN(args.num_features, args.hidden_dim,
                        num_classes, args.num_layers,
                        args.dropout)
    elif args.gnn_model == 'gcat':
        gnn_model = GCAT(args.num_features, args.hidden_dim,
                         num_classes, args.num_layers,
                         args.dropout)
    else:
        gnn_model = GCN(args.num_features, args.hidden_dim,
                        num_classes, args.num_layers,
                        args.dropout)

    gnn_model = gnn_model.to(device)

    current_time = datetime.datetime.now()
    current_time_str = current_time.strftime("%Y%m%d%H%M%S")
    logger = setup_logger('GNN', f'{log_dir}/{dataset}_batch_{first_neigh}_{second_neigh}_'
                                 f'{args.gnn_model}_{current_time_str}.log')

    logger.info(f"load dataset: {dataset}.")
    train_data, train_loader, valid_data, valid_loader, test_data, test_loader = get_data_loader(
        dataset, batch_size, first_neigh, second_neigh, add_edge=args.add_edge)

    logger.info(f"train size:{len(train_loader)}, valid size:{len(valid_loader)}, test size:{len(test_loader)}")

    train_data = train_data.to(device)
    valid_data = valid_data.to(device)
    test_data = test_data.to(device)

    logger.info(gnn_model)

    learning_rate = args.learning_rate
    loss_func = nn.NLLLoss()
    optimizer_gnn = torch.optim.Adam(gnn_model.parameters(), lr=learning_rate)

    best_acc = 0.
    best_param = None

    for epoch in range(num_epoch):

        start_time = time.time()
        train_acc, train_loss, _, _ = train_eval('train', gnn_model, train_data, train_loader,
                                                 loss_func, optimizer_gnn, device)
        valid_acc, valid_loss, _, _ = train_eval('valid', gnn_model, valid_data, valid_loader,
                                                 loss_func, optimizer_gnn, device)
        test_acc, test_loss, _, _ = train_eval('test', gnn_model, test_data, test_loader,
                                               loss_func, optimizer_gnn, device)
        cost = time.time() - start_time
        # torch.cuda.empty_cache()

        if best_acc < valid_acc:  # save the best model on the validation set
            best_acc = valid_acc
            best_param = deepcopy(gnn_model.state_dict())

        logger.info((f"epoch={epoch + 1}/{num_epoch}, cost={cost:.2f}, "
                     f"train:[{train_loss:.4f}, {train_acc:.2f}%], "
                     f"valid:[{valid_loss:.4f}, {valid_acc:.2f}%], "
                     f"test:[{test_loss:.4f}, {test_acc:.2f}%], "
                     f"best_acc={best_acc:.2f}%"))

    gnn_model.load_state_dict(best_param)  # load best param
    test_loss, test_acc, test_preds, test_labels = train_eval(
        'test', gnn_model, test_data, test_loader, loss_func, optimizer_gnn, device)

    logger.info("Test Precision, Recall and F1-Score...")
    logger.info(metrics.classification_report(test_labels, test_preds, digits=4))
    logger.info("Macro average Test Precision, Recall and F1-Score...")
    logger.info(metrics.precision_recall_fscore_support(test_labels, test_preds, average='macro'))
    logger.info("Micro average Test Precision, Recall and F1-Score...")
    logger.info(metrics.precision_recall_fscore_support(test_labels, test_preds, average='micro'))

    # model_path = os.path.join(out_dir, f'{dataset}_{args.gnn_model}_batch.pth')
    # torch.save(best_param, model_path)


if __name__ == '__main__':
    main()
