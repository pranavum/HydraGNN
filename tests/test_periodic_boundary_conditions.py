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
import pytest

import torch
from torch_geometric.data import Data
from hydragnn.preprocess.utils import (
    get_radius_graph_config,
    get_radius_graph_pbc_config,
)


def unittest_periodic_boundary_conditions():
    config_file = "./tests/inputs/ci_periodic.json"
    config = {}
    with open(config_file, "r") as f:
        config = json.load(f)

    compute_edges = get_radius_graph_config(config['Architecture'])
    compute_edges_pbc = get_radius_graph_pbc_config(config['Architecture'])

    # Create two nodes with arbitrary values.
    data = Data()
    data.supercell_size = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    data.atom_types = [1, 1] # Hydrogen
    data.pos = torch.tensor([[0, 0, 0], [0.5, 0.5, 0.5]])
    data.x = torch.tensor([[3, 5, 7], [9, 11, 13]])
    data.y = torch.tensor([[99]])

    data = compute_edges(data)
    data_periodic = compute_edges_pbc(data)

    # check that the periodic boundary condition introduces additional edges
    assert data.edge_index.size(0) < data_periodic.edge_index.size(0)

    # Check that there's still two nodes.
    assert data_periodic.edge_index.size(0) == 2
    # Check that there's one "real" bond and 26 ghost bonds (for both nodes).
    assert data_periodic.edge_index.size(1) == 1 + 26 * 2

    # Check the nodes were not modified.
    for i in range(2):
        for d in range(3):
            assert data_periodic.pos[i][d] == data.pos[i][d]
        assert data_periodic.x[i][0] == data.x[i][0]
    assert data_periodic.y == data.y


def pytest_train_model():
    unittest_periodic_boundary_conditions()
