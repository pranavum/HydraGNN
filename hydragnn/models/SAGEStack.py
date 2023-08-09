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
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import ModuleList
from torch_geometric.nn import SAGEConv, BatchNorm

from .Base import Base


class SAGEStack(Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_conv(self, input_dim, output_dim):
        sage = SAGEConv(
            in_channels=input_dim,
            out_channels=output_dim,
        )

        base_args = "x, pos, edge_index"
        if self.use_edge_attr:
            base_args += ", edge_attr"

        return Sequential(
            base_args,
            [
                (interaction, base_args + " -> x"),
            ],
        )

    def __str__(self):
        return "SAGEStack"
