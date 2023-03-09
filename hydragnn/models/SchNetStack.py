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

from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import ModuleList
from torch_geometric.typing import OptTensor
from torch_geometric.nn import SchNet, BatchNorm, global_mean_pool

from .AbstractBase import AbstractBase


class SchNetStack(AbstractBase):
    def __init__(
        self,
        num_filters: int,
        num_interactions: int,
        num_gaussians: int,
        cutoff: float,
        interaction_graph: Optional[Callable],
        max_num_neighbors: int,
        readout: str,
        dipole: bool,
        mean: Optional[float],
        std: Optional[float],
        atomref: OptTensor,
        *args,
        **kwargs
    ):
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.interaction_graph = interaction_graph
        self.max_num_neighbors = max_num_neighbors
        self.readout = readout
        self.dipole = dipole
        self.mean = mean
        self.std = std
        self.atomref = atomref

        super().__init__(*args, **kwargs)

    def forward(self, data):
        x, pos, batch = (
            data.x,
            data.pos,
            data.batch,
        )
        use_edge_attr = False

        ### encoder part ####
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            c = conv(z=x.long(), pos=pos)
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
                        x_node = F.relu(batch_norm(conv(z=x, pos=pos)))
                else:
                    x_node = headloc(x=x, batch=batch)
                outputs.append(x_node)
        return outputs

    def get_conv(self, input_dim, output_dim):
        return SchNet(
            num_filters=self.num_filters,
            num_interactions=self.num_interactions,
            num_gaussians=self.num_gaussians,
            cutoff=self.cutoff,
            interaction_graph=self.interaction_graph,
            max_num_neighbors=self.max_num_neighbors,
            readout=self.readout,
            dipole=self.dipole,
            mean=self.mean,
            std=self.std,
            atomref=self.atomref,
        )

    def __str__(self):
        return "SchNetStack"
