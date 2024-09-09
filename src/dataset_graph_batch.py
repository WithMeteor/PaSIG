import torch
import numpy as np
import scipy.sparse as sp
from torch_geometric.data import Data
from torch_geometric.loader import NeighborSampler
from proc.dataset_config import get_dataset
from torch.sparse import FloatTensor
from torch_sparse import SparseTensor


def sparse_mx_to_torch_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch.sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = sparse_mx.shape
    graph_tensor = FloatTensor(indices, values, shape)
    return graph_tensor


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch_sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    spt = FloatTensor(indices, values, shape).coalesce()
    row, col = spt.indices()
    edge_index = torch.vstack((row, col))
    edge_weight = spt.values()
    graph_tensor = SparseTensor(row=row, col=col, value=edge_weight, sparse_sizes=shape)
    return edge_index, edge_weight, graph_tensor


def get_idx_split(dataset):
    split_idx = dict()

    vocab = []
    with open('./data/temp/{}.vocab.train.txt'.format(dataset), 'r', encoding='utf-8') as data_file:
        for line in data_file.readlines():
            vocab.append(line.strip())
    vocab_size = len(vocab)

    train_size = np.load('./data/temp/{}.idx.train.npy'.format(dataset)).shape[0]
    valid_size = np.load('./data/temp/{}.idx.valid.npy'.format(dataset)).shape[0]
    test_size = np.load('./data/temp/{}.idx.test.npy'.format(dataset)).shape[0]

    train_idx = vocab_size + np.arange(train_size)
    valid_idx = vocab_size + train_size + np.arange(valid_size)
    test_idx = vocab_size + train_size + valid_size + np.arange(test_size)

    split_idx['train'] = torch.tensor(train_idx, dtype=torch.long)
    split_idx['valid'] = torch.tensor(valid_idx, dtype=torch.long)
    split_idx['test'] = torch.tensor(test_idx, dtype=torch.long)
    return split_idx


def get_data_loader(dataset, batch_size, first_neighbor=10, second_neighbor=10, add_edge=False):
    # load data

    train_idx = np.load('./data/temp/{}.idx.train.npy'.format(dataset))
    valid_idx = np.load('./data/temp/{}.idx.valid.npy'.format(dataset))
    test_idx = np.load('./data/temp/{}.idx.test.npy'.format(dataset))
    # let text embed and label follow the order of shuffling
    tv_idx = np.concatenate((train_idx, valid_idx))
    all_idx = np.concatenate((train_idx, valid_idx, test_idx))

    word_embed = np.load('./data/embed_bert/{}.embed.word.cls.in.npy'.format(dataset))
    text_embed = np.load('./data/embed_bert/{}.embed.text.cls.in.npy'.format(dataset))
    train_embed = torch.tensor(np.vstack((word_embed, text_embed[train_idx])), dtype=torch.float)
    valid_embed = torch.tensor(np.vstack((word_embed, text_embed[tv_idx])), dtype=torch.float)
    test_embed = torch.tensor(np.vstack((word_embed, text_embed[all_idx])), dtype=torch.float)

    vocab_size = word_embed.shape[0]
    text_label = np.load('./data/temp/{}.labels.all.npy'.format(dataset))
    word_label = np.zeros(vocab_size)
    train_label = torch.tensor(np.concatenate((word_label, text_label[train_idx])), dtype=torch.long)
    valid_label = torch.tensor(np.concatenate((word_label, text_label[tv_idx])), dtype=torch.long)
    test_label = torch.tensor(np.concatenate((word_label, text_label[all_idx])), dtype=torch.long)

    train_ntype = torch.cat((torch.zeros(word_label.shape[0]), torch.ones(train_idx.shape[0])))
    valid_ntype = torch.cat((torch.zeros(word_label.shape[0]), torch.ones(tv_idx.shape[0])))
    test_ntype = torch.cat((torch.zeros(word_label.shape[0]), torch.ones(all_idx.shape[0])))

    if add_edge:
        train_adj = sp.load_npz('./data/graph/{}.adj.train.add.npz'.format(dataset))
        valid_adj = sp.load_npz('./data/graph/{}.adj.valid.add.npz'.format(dataset))
        test_adj = sp.load_npz('./data/graph/{}.adj.test.add.npz'.format(dataset))
    else:
        train_adj = sp.load_npz('./data/graph/{}.adj.train.npz'.format(dataset))
        valid_adj = sp.load_npz('./data/graph/{}.adj.valid.npz'.format(dataset))
        test_adj = sp.load_npz('./data/graph/{}.adj.test.npz'.format(dataset))

    train_edge, train_weight, train_adj = sparse_mx_to_torch_sparse_tensor(train_adj)  # .coalesce()
    valid_edge, valid_weight, valid_adj = sparse_mx_to_torch_sparse_tensor(valid_adj)  # .coalesce()
    test_edge, test_weight, test_adj = sparse_mx_to_torch_sparse_tensor(test_adj)  # .coalesce()

    train_data = Data(x=train_embed, edge_index=train_edge, edge_attr=train_weight, y=train_label, node_type=train_ntype)
    valid_data = Data(x=valid_embed, edge_index=valid_edge, edge_attr=valid_weight, y=valid_label, node_type=valid_ntype)
    test_data = Data(x=test_embed, edge_index=test_edge, edge_attr=test_weight, y=test_label, node_type=test_ntype)

    # first_neighbor, second_neighbor = 10, 10  # 25, 10
    split_idx = get_idx_split(dataset)
    train_loader = NeighborSampler(train_edge, node_idx=split_idx['train'],
                                   sizes=[first_neighbor, second_neighbor],
                                   batch_size=batch_size, shuffle=False, num_workers=4)
    valid_loader = NeighborSampler(valid_edge, node_idx=split_idx['valid'],
                                   sizes=[first_neighbor, second_neighbor],
                                   batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = NeighborSampler(test_edge, node_idx=split_idx['test'],
                                  sizes=[first_neighbor, second_neighbor],
                                  batch_size=batch_size, shuffle=False, num_workers=4)

    return train_data, train_loader, valid_data, valid_loader, test_data, test_loader


if __name__ == '__main__':
    Dataset = get_dataset()
    BatchSize = 256
    TrainData, TrainLoader, ValidData, ValidLoader, TestData, TestLoader = get_data_loader(Dataset, BatchSize)
    print(TrainLoader)
    print(ValidLoader)
    print(TestLoader)
    print(next(iter(TrainLoader)))

