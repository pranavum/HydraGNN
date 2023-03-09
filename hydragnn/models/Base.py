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

import torch
from torch.nn import ModuleList, Sequential, ReLU, Linear, Module
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, BatchNorm
from torch.nn import GaussianNLLLoss
from hydragnn.utils.model import loss_function_selection
import sys
from hydragnn.utils.distributed import get_device

from .AbstractBase import AbstractBase


class Base(AbstractBase):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )

    def forward(self, data):
        x, edge_index, batch = (
            data.x,
            data.edge_index,
            data.batch,
        )
        use_edge_attr = False
        if (data.edge_attr is not None) and (self.use_edge_attr):
            use_edge_attr = True

        ### encoder part ####
        if use_edge_attr:
            for conv, batch_norm in zip(self.convs, self.batch_norms):
                c = conv(x=x, edge_index=edge_index, edge_attr=data.edge_attr)
                x = F.relu(batch_norm(c))
        else:
            for conv, batch_norm in zip(self.convs, self.batch_norms):
                c = conv(x=x, edge_index=edge_index)
                x = F.relu(batch_norm(c))

        #### multi-head decoder part####
        # shared dense layers for graph level output
        if batch is None:
            x_graph = x.mean(dim=0, keepdim=True)
        else:
            x_graph = global_mean_pool(x, batch.to(x.device))
        outputs = []
        for head_dim, headloc, type_head in zip(
            self.head_dims, self.heads_NN, self.head_type
        ):
            if type_head == "graph":
                x_graph_head = self.graph_shared(x_graph)
                outputs.append(headloc(x_graph_head))
            else:
                if self.node_NN_type == "conv":
                    for conv, batch_norm in zip(headloc[0::2], headloc[1::2]):
                        x_node = F.relu(batch_norm(conv(x=x, edge_index=edge_index)))
                else:
                    x_node = headloc(x=x, batch=batch)
                outputs.append(x_node)
        return outputs

    def loss(self, pred, value, head_index):
        if self.ilossweights_nll == 1:
            return self.loss_nll(pred, value, head_index)
        elif self.ilossweights_hyperp == 1:
            return self.loss_hpweighted(pred, value, head_index)

    def loss_nll(self, pred, value, head_index):
        # negative log likelihood loss
        # uncertainty to weigh losses in https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf
        # fixme: Pei said that right now this is never used
        raise ValueError("loss_nll() not ready yet")
        nll_loss = 0
        tasks_mseloss = []
        loss = GaussianNLLLoss()
        for ihead in range(self.num_heads):
            head_pre = pred[ihead][:, :-1]
            pred_shape = head_pre.shape
            head_val = value[head_index[ihead]]
            value_shape = head_val.shape
            if pred_shape != value_shape:
                head_val = torch.reshape(head_val, pred_shape)
            head_var = torch.exp(pred[ihead][:, -1])
            nll_loss += loss(head_pre, head_val, head_var)
            tasks_mseloss.append(F.mse_loss(head_pre, head_val))

        return nll_loss, tasks_mseloss, []

    def loss_hpweighted(self, pred, value, head_index):
        # weights for different tasks as hyper-parameters
        tot_loss = 0
        tasks_loss = []
        for ihead in range(self.num_heads):
            head_pre = pred[ihead]
            pred_shape = head_pre.shape
            head_val = value[head_index[ihead]]
            value_shape = head_val.shape
            if pred_shape != value_shape:
                head_val = torch.reshape(head_val, pred_shape)

            tot_loss += (
                self.loss_function(head_pre, head_val) * self.loss_weights[ihead]
            )
            tasks_loss.append(self.loss_function(head_pre, head_val))

        return tot_loss, tasks_loss

    def __str__(self):
        return "Base"


class MLPNode(Module):
    def __init__(self, input_dim, output_dim, num_mlp, hidden_dim_node, node_type):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.node_type = node_type
        self.num_mlp = num_mlp

        self.mlp = ModuleList()
        for _ in range(self.num_mlp):
            denselayers = []
            denselayers.append(Linear(self.input_dim, hidden_dim_node[0]))
            denselayers.append(ReLU())
            for ilayer in range(len(hidden_dim_node) - 1):
                denselayers.append(
                    Linear(hidden_dim_node[ilayer], hidden_dim_node[ilayer + 1])
                )
                denselayers.append(ReLU())
            denselayers.append(Linear(hidden_dim_node[-1], output_dim))
            self.mlp.append(Sequential(*denselayers))

    def node_features_reshape(self, x, batch):
        """reshape x from [batch_size*num_nodes, num_features] to [batch_size, num_features, num_nodes]"""
        num_features = x.shape[1]
        batch_size = batch.max() + 1
        out = torch.zeros(
            (batch_size, num_features, self.num_nodes),
            dtype=x.dtype,
            device=x.device,
        )
        for inode in range(self.num_nodes):
            inode_index = [i for i in range(inode, batch.shape[0], self.num_nodes)]
            out[:, :, inode] = x[inode_index, :]
        return out

    def forward(self, x: torch.Tensor, batch: torch.Tensor):
        if self.node_type == "mlp":
            outs = self.mlp[0](x)
        else:
            outs = torch.zeros(
                (x.shape[0], self.output_dim),
                dtype=x.dtype,
                device=x.device,
            )
            x_nodes = self.node_features_reshape(x, batch)
            for inode in range(self.num_nodes):
                inode_index = [i for i in range(inode, batch.shape[0], self.num_nodes)]
                outs[inode_index, :] = self.mlp[inode](x_nodes[:, :, inode])
        return outs

    def __str__(self):
        return "MLPNode"
