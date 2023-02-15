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
import pickle
from tqdm import tqdm

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.transforms import Distance

from hydragnn.preprocess.load_data import dataset_loading_and_splitting
from hydragnn.preprocess.load_data import load_train_val_test_sets
from hydragnn.utils.distributed import setup_ddp
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

from hydragnn.preprocess import create_dataloaders

import time

from scipy.interpolate import griddata

plt.rcParams.update({"font.size": 16})

def getcolordensity(xdata, ydata):
    ###############################
    nbin = 20
    hist2d, xbins_edge, ybins_edge = np.histogram2d(
        x=xdata, y=ydata, bins=[nbin, nbin]
    )
    xbin_cen = 0.5 * (xbins_edge[0:-1] + xbins_edge[1:])
    ybin_cen = 0.5 * (ybins_edge[0:-1] + ybins_edge[1:])
    BCTY, BCTX = np.meshgrid(ybin_cen, xbin_cen)
    hist2d = hist2d / np.amax(hist2d)
    print(np.amax(hist2d))

    bctx1d = np.reshape(BCTX, len(xbin_cen) * nbin)
    bcty1d = np.reshape(BCTY, len(xbin_cen) * nbin)
    loc_pts = np.zeros((len(xbin_cen) * nbin, 2))
    loc_pts[:, 0] = bctx1d
    loc_pts[:, 1] = bcty1d
    hist2d_norm = griddata(
        loc_pts,
        hist2d.reshape(len(xbin_cen) * nbin),
        (xdata, ydata),
        method="linear",
        fill_value=0,
    )  # np.nan)
    return hist2d_norm



#########################################################

CaseDir = os.path.join(
    os.path.dirname(__file__),
    "logs",
)


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

        (
            trainset,
            valset,
            testset,
        ) = load_train_val_test_sets(config, isdist=True)

        #train_loader, val_loader, test_loader = dataset_loading_and_splitting(
        #    config=config
        #)
        (train_loader, val_loader, test_loader,) = create_dataloaders(
            trainset, valset, testset, config["NeuralNetwork"]["Training"]["batch_size"]
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

        load_existing_model(model, model_name, path="./logs/")
        model.eval()

        data_list = []

        test_MAE = 0.0
        test_MAE_per_atom = 0.0

        minmax_graph_feature = []

        with open("./dataset/serialized_dataset/FePt_32atoms_0_train.pkl", "rb") as f:
            _ = pickle.load(f)
            minmax_graph_feature = pickle.load(f)

        num_samples = len(test_loader.dataset)
        true_formation_energy_values = []
        predicted_formation_energy_values = []

        for data_id, data in enumerate(tqdm(test_loader.dataset)):

            if data_id > 1000:
                break
            scaled_predicted_formation_energy = model(data)
            predicted_formation_energy = scaled_predicted_formation_energy[0].item() * (minmax_graph_feature[1, 0] - minmax_graph_feature[0, 0]) + minmax_graph_feature[0, 0]
            predicted_formation_energy_values.append(predicted_formation_energy)
            #predicted_formation_energy_values.append(scaled_predicted_formation_energy[0].item())
            scaled_true_formation_energy = data.y[0].item()
            true_formation_energy = scaled_true_formation_energy * (minmax_graph_feature[1, 0] - minmax_graph_feature[0, 0]) + minmax_graph_feature[0, 0]
            true_formation_energy_values.append(true_formation_energy)
            #true_formation_energy_values.append(scaled_true_formation_energy)
            num_atoms = data.x.shape[0]
            MAE = abs(true_formation_energy - predicted_formation_energy)
            MAE_per_atom = MAE/num_atoms
            test_MAE += MAE/num_samples
            test_MAE_per_atom += MAE_per_atom/num_samples

        print("Test MAE: ", test_MAE)
        print("Test MAE per atom: ", test_MAE_per_atom)

        hist2d_norm = getcolordensity(true_formation_energy_values, predicted_formation_energy_values)

        fig, ax = plt.subplots()
        plt.scatter(
            true_formation_energy_values, predicted_formation_energy_values, s=8, c=hist2d_norm, vmin=0, vmax=1
        )
        #plt.xlim([0.0, 0.1])
        #plt.ylim([0.0, 0.1])
        plt.clim(0, 1)
        plt.colorbar()
        plt.xlabel("True values (eV)")
        plt.ylabel("Predicted values (eV)")
        plt.title("NiPt - Formation energy")
        plt.draw()
        plt.tight_layout()
        plt.savefig("./Formation_energy_scatterplot" + ".png", dpi=400)




