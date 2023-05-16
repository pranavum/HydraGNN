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


import os, json
import matplotlib.pyplot as plt
import random
import pickle, csv
import pandas as pd

import logging
import sys
from tqdm import tqdm
from mpi4py import MPI
from itertools import chain
import argparse
import time

import torch_geometric

import hydragnn
from hydragnn.utils.distributed import setup_ddp, get_device
from hydragnn.utils.print_utils import print_distributed, iterate_tqdm
from hydragnn.utils.time_utils import Timer
from hydragnn.utils.pickledataset import SimplePickleDataset
from hydragnn.utils.model import print_model, load_existing_model
from hydragnn.utils.smiles_utils import (
    get_node_attribute_name,
    generate_graphdata_from_smilestr,
)
from hydragnn.preprocess.utils import get_radius_graph

from hydragnn.models.create import create_model_config
from hydragnn.utils.config_utils import (
    update_config,
    get_log_name_config,
    save_config
)

import numpy as np

try:
    from hydragnn.utils.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    pass

import torch_geometric.data
import torch
import torch.distributed as dist

import time

plt.rcParams.update({"font.size": 16})
#########################################################

CaseDir = os.path.join(
    os.path.dirname(__file__),
    "logs",
)

DatasetDir = os.path.join(
    os.path.dirname(__file__),
    "dataset",
)


######################################################################
for irun in range(1, 2):

    number_data_samples = 1000
    start_time = 0.0
    finish_time = 0.0

    config_file = CaseDir + "/spectrum/" + "config"
    with open(config_file + ".json", "r") as f:
        config = json.load(f)

    os.environ["SERIALIZED_DATA_PATH"] = CaseDir

    world_size, world_rank = setup_ddp()

    pickle_path = DatasetDir + "/serialized_dataset/"

    ##set directory to load processed pickle files, train/validate/test
    testset = []
    with open("dataset/serialized_dataset/GDB-9-Ex-testset.pkl", 'rb') as pickle_file:
        _ = pickle.load(pickle_file)
        _ = pickle.load(pickle_file)
        testset = pickle.load(pickle_file)

    model = create_model_config(
        config=config["NeuralNetwork"],
        verbosity=config["Verbosity"]["level"],
    )

    model = torch.nn.parallel.DistributedDataParallel(
        model
    )

    model_name = "spectrum"

    load_existing_model(model, model_name, path="./logs/")

    num_test_samples = len(testset)
    mol_ID = [item.ID for item in testset]

    lower = 0.0
    upper = 750.0
    length = 37500
    bins = [lower + x * (upper - lower) / length for x in range(length)]

    for sample_id, test_data in enumerate(tqdm(testset)):
        fig, ax = plt.subplots(1, 1, figsize=(18, 6))
        prediction = model(test_data.to(get_device()))[0].squeeze().detach().to('cpu')
        true_values = test_data.to(get_device()).detach().to('cpu').y
        ax.plot(bins, true_values, label="TD-DFTB+")
        ax.plot(bins, prediction, label="HydraGNN")
        plt.title("molecule ID: "+mol_ID[sample_id])
        plt.legend()
        plt.draw()
        plt.tight_layout()
        plt.ylim([-0.2, max(true_values)+0.2])
        plt.savefig(f"logs/sample_{sample_id}.png")
        plt.close(fig)


