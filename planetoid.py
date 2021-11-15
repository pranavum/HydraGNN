#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 11:38:24 2021

@author: 7ml
"""
##############################################################################
# Copyright (c) 2021, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################

import sys, os, json

import torch
import shutil
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader

import torch.nn.functional as F
from torch_geometric.utils import degree
from torch_geometric.nn import GATConv, GCNConv, GINConv, PNAConv

import hydragnn, hydragnn.unit_tests
from hydragnn.models import GINStack, GATStack, PNAStack, MFCStack, CGCNNStack

dataset = Planetoid(root='/tmp/Cora', name='Cora')

class GAT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(dataset.num_node_features, 16)
        self.conv2 = GATConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
        #return x

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

class GIN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ginconv = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(dataset.num_node_features, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, dataset.num_classes),
                torch.nn.Sigmoid()
            ),
            eps=100.0,
            train_eps=True,
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        return self.ginconv(x, edge_index)

class PNA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.aggregators = ["mean", "min", "max", "std"]
        self.scalers = [
            "identity",
            "amplification",
            "attenuation",
            "linear",
        ]
        deg = torch.zeros(dataset[0].num_edges, dtype=torch.long)
        for data in dataset:
            d = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())
        self.conv1 = PNAConv(dataset.num_node_features, 6, self.aggregators, self.scalers, deg)
        self.conv2 = PNAConv(6, dataset.num_classes, self.aggregators, self.scalers, deg)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

# Main unit test function called by pytest wrappers.
def planetoid(model_type):

    world_size, rank = hydragnn.utils.get_comm_size_and_rank()

    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    device, device_name = hydragnn.utils.device.get_device()

    if model_type == "GIN":
       basic_model = GIN().to(device)

    elif model_type == "GAT":
       basic_model = GAT().to(device)

    elif model_type == "PNA":
       basic_model = PNA().to(device)

    optimizer = torch.optim.Adam(basic_model.parameters(), lr=0.01, weight_decay=5e-4)

    basic_model.train()
    for epoch in range(20):
        for data in loader:
            optimizer.zero_grad()
            out = basic_model(data)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

    basic_model.eval()
    pred = basic_model(data).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    basic_model_acc = int(correct) / int(data.test_mask.sum())

    graph_head = {
        "graph":{
            "num_sharedlayers": 2,
            "dim_sharedlayers": 4,
            "num_headlayers": 2,
            "dim_headlayers": [10,10]
        }
    }

    if model_type == "GAT":
        hydragnn_model = GATStack(
            input_dim=data.num_node_features,
            output_dim=[data.num_nodes*dataset.num_classes],
            hidden_dim=16,
            num_nodes=data.num_nodes,
            num_conv_layers=2,
            output_type=['graph'],
            config_heads=graph_head,
            loss_weights=[1.0],
        ).to(device)

    elif model_type == "GIN":
        hydragnn_model = GINStack(
            input_dim=dataset.num_node_features,
            output_dim=[dataset.num_nodes*dataset.num_classes],
            hidden_dim=16,
            num_nodes=data.num_nodes,
            num_conv_layers=2,
            output_type=['graph'],
            config_heads=graph_head,
            loss_weights=[1.0],
        ).to(device)

    elif model_type == "PNA":
        deg = torch.zeros(dataset[0].num_edges, dtype=torch.long)
        for data in dataset:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())
        hydragnn_model = PNAStack(
            deg=deg,
            input_dim=dataset.num_node_features,
            output_dim=[dataset.num_nodes*dataset.num_classes],
            hidden_dim=16,
            num_nodes=data.num_nodes,
            num_conv_layers=2,
            output_type=['graph'],
            config_heads=graph_head,
            loss_weights=[1.0],
        ).to(device)

    optimizer = torch.optim.Adam(hydragnn_model.parameters(), lr=0.01, weight_decay=5e-4)

    hydragnn_model.train()
    for epoch in range(50):
        for data in loader:
            optimizer.zero_grad()
            out = hydragnn_model(data)
            out = out.squeeze()
            out = out.reshape((data.num_nodes,dataset.num_classes))
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

    hydragnn_model.eval()
    pred = hydragnn_model(data)
    pred = pred.reshape((data.num_nodes,dataset.num_classes))
    pred = pred.argmax(axis=1) 
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    hydragnn_model_acc = int(correct) / int(data.test_mask.sum())

    print("MASSI: ", basic_model_acc)
    print("MASSI: ", hydragnn_model_acc)

    assert basic_model_acc < hydragnn_model_acc, "Accuracy of PyG model is higher than accuracy of HydraGNN model!"


if __name__ == "__main__":
    planetoid("GAT")
