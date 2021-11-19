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
from torch_geometric.datasets import MD17
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import RadiusGraph

import torch.nn.functional as F
from torch_geometric.utils import degree
from torch_geometric.nn import GATConv, GCNConv, GINConv, PNAConv

import hydragnn, hydragnn.unit_tests
from hydragnn.models import GINStack, GATStack, PNAStack, MFCStack, CGCNNStack

dataset = MD17(root="/tmp/aspirin_CCSD", name="aspirin CCSD", train=True)

compute_edges = RadiusGraph(r=10, loop=True, max_num_neighbors=10)


class GAT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(1, 20)
        self.conv2 = GATConv(20, 20)
        self.linear = torch.nn.Linear(20, 1)

    def forward(self, data):
        x, edge_index = data.z, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.linear(x)

        return x


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(1, 20)
        self.conv2 = GCNConv(20, 20)
        self.linear = torch.nn.Linear(20, 1)

    def forward(self, data):
        x, edge_index = data.z, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.linear(x)

        return x


class GIN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ginconv = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(1, 20),
                torch.nn.ReLU(),
                torch.nn.Linear(20, 20),
                torch.nn.ReLU(),
                torch.nn.Linear(20, 1),
            ),
            eps=100.0,
            train_eps=True,
        )

    def forward(self, data):
        x, edge_index = data.z, data.edge_index
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
        data = compute_edges(dataset[0])
        deg = torch.zeros(data.num_edges, dtype=torch.long)
        for data in dataset:
            data = compute_edges(data)
            d = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())
        self.conv1 = PNAConv(
            dataset.num_node_features, 20, self.aggregators, self.scalers, deg
        )
        self.conv2 = PNAConv(20, 20, self.aggregators, self.scalers, deg)
        self.linear = torch.nn.Linear(20, 1)

    def forward(self, data):
        x, edge_index = data.pos, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.linear(x)

        return x


# Main unit test function called by pytest wrappers.
def md17(model_type):

    world_size, rank = hydragnn.utils.get_comm_size_and_rank()

    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for data in loader:
        data = compute_edges(data)

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
            data = compute_edges(data)
            out = basic_model(data)
            loss = torch.nn.MSELoss(out, data.energy)
            loss.backward()
            optimizer.step()

    basic_model.eval()
    pred = basic_model(data)
    basic_error = (pred - data.energy).sum()

    graph_head = {
        "graph": {
            "num_sharedlayers": 2,
            "dim_sharedlayers": 4,
            "num_headlayers": 2,
            "dim_headlayers": [10, 10],
        }
    }

    if model_type == "GAT":
        hydragnn_model = GATStack(
            input_dim=data.num_node_features,
            output_dim=[1],
            hidden_dim=20,
            num_nodes=data.num_nodes,
            num_conv_layers=2,
            output_type=["graph"],
            config_heads=graph_head,
            loss_weights=[1.0],
        ).to(device)

    elif model_type == "GIN":
        hydragnn_model = GINStack(
            input_dim=dataset.num_node_features,
            output_dim=[1],
            hidden_dim=20,
            num_nodes=data.num_nodes,
            num_conv_layers=2,
            output_type=["graph"],
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
            output_dim=[1],
            hidden_dim=20,
            num_nodes=data.num_nodes,
            num_conv_layers=2,
            output_type=["graph"],
            config_heads=graph_head,
            loss_weights=[1.0],
        ).to(device)

    optimizer = torch.optim.Adam(
        hydragnn_model.parameters(), lr=0.01, weight_decay=5e-4
    )

    hydragnn_model.train()
    for epoch in range(20):
        for data in loader:
            optimizer.zero_grad()
            out = hydragnn_model(data)
            loss = torch.nn.MSELoss(out, data.energy)
            loss.backward()
            optimizer.step()

    hydragnn_model.eval()
    pred = hydragnn_model(data)
    hydragnn_error = (pred - data.energy).sum()

    assert (
        basic_error < hydragnn_error
    ), "Accuracy of PyG model is higher than accuracy of HydraGNN model!"


if __name__ == "__main__":
    md17("PNA")
