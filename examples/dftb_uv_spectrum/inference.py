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
import pickle

from tqdm import tqdm

from hydragnn.utils.distributed import setup_ddp, get_device
from hydragnn.utils.model import load_existing_model

from hydragnn.models.create import create_model_config

try:
    from hydragnn.utils.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    pass

import torch

from ae_vibrational_spectrum.reduction_models import autoencoder

ae_model = autoencoder(input_dim=37500, reduced_dim=50, hidden_dim_ae=[250], PCA=False).to(get_device())
log_model = "nz-" + str(50) + "-PCA-" + str(False)
model_dir = os.path.join("ae_vibrational_spectrum/logs_GDB-9-Ex-TDDFTB/", log_model)
device = next(ae_model.parameters()).device
path_name = os.path.join(model_dir, "model.pk")
ae_model.load_state_dict(torch.load(path_name, map_location=device))

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
        compressed_prediction = model(test_data.to(get_device()))[0]
        prediction = ae_model.decoder(compressed_prediction).squeeze().detach().to('cpu')
        true_values_AE = ae_model.decoder(test_data.to(get_device()).y).detach().to('cpu')
        ax.plot(bins, true_values_AE, label="TD-DFTB+_AE", linewidth=5)
        ax.plot(bins, test_data.to(get_device()).full_spectrum.detach().to('cpu'), label="TD-DFTB+_original", linewidth=5)
        ax.plot(bins, prediction, label="HydraGNN", linewidth=2)
        plt.title("molecule ID: "+mol_ID[sample_id])
        plt.legend()
        plt.draw()
        plt.tight_layout()
        plt.ylim([-0.2, max(true_values_AE)+0.2])
        plt.savefig(f"logs/sample_{sample_id}.png")
        plt.close(fig)


