import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GraphConv, GINConv


class GCN(nn.Module):
    def __init__(self, dim_input, dim_hidden, num_classes, dropout=0, num_layers=3, mode='node'):
        super(GCN, self).__init__()
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.num_classes = num_classes
        self.dropout = dropout
        self.num_layers = num_layers
        self.mode = mode

        convlayers = []
        conv_input = GraphConv(dim_input, dim_hidden[0], norm='both', allow_zero_in_degree=True)
        convlayers.append(conv_input)

        i = -1
        for i in range(num_layers-1):
            conv = GraphConv(dim_hidden[i], dim_hidden[i+1], norm='both', allow_zero_in_degree=True)
            convlayers.append(conv)

        self.convs = nn.ModuleList(convlayers)
        self.classify = nn.Linear(dim_hidden[i+1], num_classes)


    def forward(self, graph, feat, eweight=None):
        h = self.convs[0](graph, feat, edge_weight=eweight)
        h = F.dropout(h, self.dropout, training=self.training)
        h = F.relu(h)
        for i in range(self.num_layers-1):
            h = self.convs[i+1](graph, h, edge_weight=eweight)
            h = F.dropout(h, self.dropout, training=self.training)
            h = F.relu(h)

        if self.mode == 'node':
            h = self.classify(h)

        if self.mode == 'graph':
            with graph.local_scope():
                graph.ndata['f'] = h
                hg = dgl.readout_nodes(graph, 'f', op='sum')
                h = self.classify(hg)

        return h

    def loss(self, pred, label):
        return F.cross_entropy(pred, label)

    def accuracy(self, pred, label):
        pred = pred.max(1)[1].type_as(label)
        correct = pred.eq(label).double()
        correct = correct.sum()
        return correct / len(label)


class GIN(nn.Module):
    def __init__(self, dim_input, dim_hidden, num_classes, dropout=0, num_layers=3, mode='node'):
        super(GIN, self).__init__()
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.num_classes = num_classes
        self.dropout = dropout
        self.num_layers = num_layers
        self.mode = mode

        convlayers = []
        mlp_input = nn.Linear(dim_input, dim_hidden[0])
        conv_input = GINConv(mlp_input)
        convlayers.append(conv_input)

        i = -1
        for i in range(num_layers-1):
            mlp = nn.Linear(dim_hidden[i], dim_hidden[i+1])
            conv = GINConv(mlp)
            convlayers.append(conv)

        self.convs = nn.ModuleList(convlayers)
        self.classify = nn.Linear(dim_hidden[i+1], num_classes)


    def forward(self, graph, feat, eweight=None):
        h = self.convs[0](graph, feat, edge_weight=eweight)
        h = F.dropout(h, self.dropout, training=self.training)
        h = F.relu(h)
        for i in range(self.num_layers-1):
            h = self.convs[i+1](graph, h, edge_weight=eweight)
            h = F.dropout(h, self.dropout, training=self.training)
            h = F.relu(h)

        if self.mode == 'node':
            h = self.classify(h)

        if self.mode == 'graph':
            with graph.local_scope():
                graph.ndata['f'] = h
                hg = dgl.readout_nodes(graph, 'f', op='sum')
                h = self.classify(hg)

        return h

    def loss(self, pred, label):
        return F.cross_entropy(pred, label)

    def accuracy(self, pred, label):
        pred = pred.max(1)[1].type_as(label)
        correct = pred.eq(label).double()
        correct = correct.sum()
        return correct / len(label)

