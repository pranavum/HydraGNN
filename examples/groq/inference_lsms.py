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

import json, os
import matplotlib.pyplot as plt

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import Distance

from hydragnn.preprocess.load_data import dataset_loading_and_splitting
from hydragnn.utils.distributed import setup_ddp, get_device
from hydragnn.utils.model import load_existing_model
from hydragnn.utils.config_utils import (
    update_config,
    get_log_name_config,
)
from hydragnn.utils.model import load_existing_model_config
from hydragnn.models.create import create_model_config

from hydragnn.preprocess.utils import (
    get_radius_graph,
)

import time

plt.rcParams.update({"font.size": 16})
#########################################################

CaseDir = os.path.join(
    os.path.dirname(__file__),
    "logs_lsms",
)


def create_bct_sample(config):
    # Create BCT lattice structure with 32 nodes
    uc_x = 12
    uc_y = 6
    uc_z = 6
    lxy = 5.218
    lz = 7.058
    count = 0
    number_atoms = 2 * uc_x * uc_y * uc_z
    positions = torch.zeros(number_atoms, 3)
    for x in range(uc_x):
        for y in range(uc_y):
            for z in range(uc_z):
                positions[count][0] = x * lxy
                positions[count][1] = y * lxy
                positions[count][2] = z * lz
                count += 1
                positions[count][0] = (x + 0.5) * lxy
                positions[count][1] = (y + 0.5) * lxy
                positions[count][2] = (z + 0.5) * lz
                count += 1

    data = Data()
    number_fe_atoms = torch.randint(0,number_atoms, (1,)).item()
    number_pt_atoms = number_atoms - number_fe_atoms
    atom_configuration = torch.cat((torch.ones(number_fe_atoms, 1), torch.zeros(number_pt_atoms, 1)), 0)
    idx = torch.randperm(atom_configuration.shape[0])
    randomized_atom_configuration = atom_configuration[idx,:]
    data.x = randomized_atom_configuration
    data.pos = positions

    compute_edges = get_radius_graph(
        radius=config["radius"],
        loop=False,
        max_neighbours=config["max_neighbours"],
    )

    data = compute_edges(data)

    # edge lengths already added manually if using PBC.
    compute_edge_lengths = Distance(norm=False, cat=True)
    data = compute_edge_lengths(data)

    max_edge_length = torch.Tensor([float("-inf")])

    data.edge_attr = data.edge_attr / max_edge_length

    return data

######################################################################
for irun in range(1, 2):

    number_data_samples = 1000
    start_time = 0.0
    finish_time = 0.0

    for icase in range(1):
        config_file = CaseDir + "/energy/" + "config"
        with open(config_file + ".json", "r") as f:
            config = json.load(f)

        os.environ["SERIALIZED_DATA_PATH"] = CaseDir

        world_size, world_rank = setup_ddp()

        train_loader, val_loader, test_loader = dataset_loading_and_splitting(
            config=config
        )

        config = update_config(config, train_loader, val_loader, test_loader)

        model = create_model_config(
            config=config["NeuralNetwork"],
            verbosity=config["Verbosity"]["level"],
        )

        log_name = get_log_name_config(config)
        model_name = "energy"

        model = torch.nn.parallel.DistributedDataParallel(
            model
        )

        load_existing_model(model, model_name, path="logs_lsms/")

        data_list = []

        for iter in range (0,number_data_samples):
            # generate sequence of randomized atomic configurations
            data_list.append(create_bct_sample(config["NeuralNetwork"]["Architecture"]))

        for data_item in data_list:
            start_time += time.time()
            pred = model(data_item.to(get_device()))
            finish_time += time.time()

        #Average time per batch-1 inference
        average_time = (finish_time - start_time) / number_data_samples

        print("Average wall-clock time on classical device: ", average_time)




