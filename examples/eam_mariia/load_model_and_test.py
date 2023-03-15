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

from hydragnn.preprocess.load_data import load_train_val_test_sets
from hydragnn.utils.distributed import setup_ddp, get_device
from hydragnn.utils.model import load_existing_model
from hydragnn.utils.config_utils import (
    update_config,
    get_log_name_config,
)
from hydragnn.models.create import create_model_config
from hydragnn.preprocess import create_dataloaders

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

    model_name = "bulk"
    pickle_dataset_path = "train_dataset/serialized_dataset"

    for icase in range(1):
        config_file = CaseDir + "/"+model_name+"/" + "config"
        with open(config_file + ".json", "r") as f:
            config = json.load(f)

        os.environ["SERIALIZED_DATA_PATH"] = CaseDir
        config["Dataset"]["path"] = {}
        config["Dataset"]["path"]["train"] = "train_dataset/serialized_dataset/NiPt-trainset.pkl"
        config["Dataset"]["path"]["validate"] = "train_dataset/serialized_dataset/NiPt-valset.pkl"
        config["Dataset"]["path"]["test"] = "train_dataset/serialized_dataset/NiPt-testset.pkl"

        world_size, world_rank = setup_ddp()

        (
            trainset,
            valset,
            testset,
        ) = load_train_val_test_sets(config, isdist=True)

        (train_loader, val_loader, test_loader,) = create_dataloaders(
            trainset, valset, testset, config["NeuralNetwork"]["Training"]["batch_size"]
        )

        config = update_config(config, train_loader, val_loader, test_loader)

        model = create_model_config(
            config=config["NeuralNetwork"],
            verbosity=config["Verbosity"]["level"],
        )

        log_name = get_log_name_config(config)

        model = torch.nn.parallel.DistributedDataParallel(
            model
        )

        load_existing_model(model, model_name, path="./logs/")
        model.eval()

        data_list = []

        test_MAE = 0.0
        test_MAE_per_atom = 0.0

        minmax_graph_feature = []

        with open(pickle_dataset_path+"/NiPt-trainset.pkl", "rb") as f:
            _ = pickle.load(f)
            minmax_graph_feature = pickle.load(f)

        num_samples = len(test_loader.dataset)
        true_values = []
        predicted_values = []

        for data_id, data in enumerate(tqdm(test_loader.dataset)):

            scaled_predicted = model(data.to(get_device()))
            predicted = scaled_predicted[0].item() * (minmax_graph_feature[1, 0] - minmax_graph_feature[0, 0]) + minmax_graph_feature[0, 0]
            predicted_values.append(predicted)
            scaled_true = data.y[0].item()
            true = scaled_true * (minmax_graph_feature[1, 0] - minmax_graph_feature[0, 0]) + minmax_graph_feature[0, 0]
            true_values.append(true)
            num_atoms = data.x.shape[0]
            MAE = abs(true - predicted)
            MAE_per_atom = MAE/num_atoms
            test_MAE += MAE/num_samples
            test_MAE_per_atom += MAE_per_atom/num_samples

        print("Test MAE: ", test_MAE)
        print("Test MAE per atom: ", test_MAE_per_atom)

        hist2d_norm = getcolordensity(true_values, predicted_values)

        fig, ax = plt.subplots()
        plt.scatter(
            true_values, predicted_values, s=8, c=hist2d_norm, vmin=0, vmax=1
        )
        plt.clim(0, 1)
        ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", color="red")
        plt.colorbar()
        plt.xlabel("True values")
        plt.ylabel("Predicted values")
        plt.draw()
        plt.tight_layout()
        plt.savefig("./Scatterplot" + ".png", dpi=400)
