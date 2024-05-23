#!/user/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as func
# from torch_geometric.nn import GINConv, SAGEConv
from src.gnn_layer import GCNConv, SAGEConv, GINConv, GCATConv


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.):
        super(GCN, self).__init__()

        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))  # , normalize=False, add_self_loops=False
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))  # , normalize=False, add_self_loops=False
        self.convs.append(GCNConv(hidden_channels, out_channels))  # , normalize=False, add_self_loops=False
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight):
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index, edge_weight)
            x = func.relu(x)
            x = func.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight)
        # return x.log_softmax(dim=-1), x
        return x


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.):
        super(SAGE, self).__init__()

        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight):  # note that GraphSAGE do not use edge weight in propagation
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = func.relu(x)
            x = func.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        # return x.log_softmax(dim=-1), x
        return x


class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.):
        super(GIN, self).__init__()

        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.convs = torch.nn.ModuleList()
        self.convs.append(GINConv(in_channels, hidden_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GINConv(hidden_channels, hidden_channels, hidden_channels))
        self.convs.append(GINConv(hidden_channels, hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight):
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index, edge_weight)
            x = func.relu(x)
            x = func.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight)
        # return x.log_softmax(dim=-1), x
        return x


class GCAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.):
        super(GCAT, self).__init__()

        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCATConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCATConv(hidden_channels, hidden_channels))
        self.convs.append(GCATConv(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight):
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index, edge_weight)
            x = func.relu(x)
            x = func.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight)
        # return x.log_softmax(dim=-1), x
        return x
