#!/user/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as func

from torch_scatter import scatter_add
from torch_geometric.nn.inits import zeros
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.typing import Adj, OptTensor


def edge_norm(edge_index: Tensor, edge_weight=None, num_nodes=None, improved=False,
              add_self_loops=True, dtype=None, degree='row'):

    fill_value = 2. if improved else 1.

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    if add_self_loops:
        edge_index, tmp_edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]
    if degree == 'col':
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)  # for GCAT
    else:  # degree == 'row'
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)  # for GCN & GIN
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


class GCNConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self.lin = Linear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: OptTensor = None) -> Tensor:

        if self.normalize:
            edge_index, edge_weight = edge_norm(
                edge_index, edge_weight, x.size(self.node_dim),
                self.improved, self.add_self_loops, degree='row')

        x = self.lin(x)

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j


class SAGEConv(MessagePassing):

    def __init__(self, in_channels: int,
                 out_channels: int, normalize: bool = False,
                 root_weight: bool = True, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'mean')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight

        self.lin_l = Linear(in_channels, out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:

        x_r = x.clone()

        out = self.propagate(edge_index, x=x)
        out = self.lin_l(out)

        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = func.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j


class GINConv(MessagePassing):

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 eps: float = 0., train_eps: bool = True, normalize: bool = True, add_self_loops: bool = True,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(in_channels, hidden_channels),
                                 nn.ReLU(),
                                 nn.Linear(hidden_channels, out_channels))
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.reset_parameters()

    def reset(self, value):
        if hasattr(value, 'reset_parameters'):
            value.reset_parameters()
        else:
            for child in value.children() if hasattr(value, 'children') else []:
                self.reset(child)

    def reset_parameters(self):
        self.reset(self.mlp)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: OptTensor = None) -> Tensor:

        x_r = x.clone()

        if self.normalize:
            edge_index, edge_weight = edge_norm(
                edge_index, edge_weight, add_self_loops=self.add_self_loops, degree='row')
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)

        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.mlp(out)

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j


class GINConvOri(MessagePassing):

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 eps: float = 0., train_eps: bool = True,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(in_channels, hidden_channels),
                                 nn.ReLU(),
                                 nn.Linear(hidden_channels, out_channels))
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

        self.reset_parameters()

    def reset(self, value):
        if hasattr(value, 'reset_parameters'):
            value.reset_parameters()
        else:
            for child in value.children() if hasattr(value, 'children') else []:
                self.reset(child)

    def reset_parameters(self):
        self.reset(self.mlp)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: OptTensor = None) -> Tensor:

        x_r = x.clone()

        out = self.propagate(edge_index, x=x)

        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.mlp(out)

    def message(self, x_j: Tensor) -> Tensor:
        return x_j


class GCATConvOld(MessagePassing):

    def __init__(self, in_channels: int,
                 out_channels: int, normalize: bool = False,
                 bias: bool = False, **kwargs):
        kwargs.setdefault('aggr', 'mean')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        self.lin_c = Linear(in_channels * 2, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_c.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor) -> Tensor:

        x_r = x.clone()

        agg = self.propagate(edge_index, x=x, edge_weight=edge_weight)

        out = torch.hstack((agg, x_r))
        out = self.lin_c(out)

        if self.normalize:
            out = func.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j


class GCATConv(MessagePassing):

    def __init__(self, in_channels: int,
                 out_channels: int, normalize: bool = True, add_self_loops: bool = True,
                 bias: bool = False, **kwargs):
        kwargs.setdefault('aggr', 'add')  # 'mean'
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.add_self_loops = add_self_loops
        self.lin_c = Linear(in_channels * 2, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_c.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor) -> Tensor:

        x_r = x.clone()

        if self.normalize:
            edge_index, edge_weight = edge_norm(
                edge_index, edge_weight, add_self_loops=self.add_self_loops, degree='col')

        agg = self.propagate(edge_index, x=x, edge_weight=edge_weight)

        out = torch.hstack((agg, x_r))
        out = self.lin_c(out)

        # if self.normalize:
        #     out = func.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
