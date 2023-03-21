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
import sys
import logging
import pickle
from tqdm import tqdm
from mpi4py import MPI
import argparse

import torch
import numpy as np

import hydragnn
from hydragnn.utils.time_utils import Timer
from hydragnn.preprocess.load_data import load_train_val_test_sets
from hydragnn.utils.distributed import setup_ddp, get_device
from hydragnn.utils.model import load_existing_model
from hydragnn.utils.cfgdataset import CFGDataset
from hydragnn.utils.serializeddataset import SerializedWriter, SerializedDataset
from hydragnn.utils.config_utils import (
    update_config,
    get_log_name_config,
)
from hydragnn.models.create import create_model_config
from hydragnn.preprocess import create_dataloaders

from scipy.interpolate import griddata

try:
    from hydragnn.utils.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    pass

import matplotlib.pyplot as plt
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

def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))

if __name__ == "__main__":

    model_name = "energy"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--loadexistingsplit",
        action="store_true",
        help="loading from existing pickle/adios files with train/test/validate splits",
    )
    parser.add_argument(
        "--preonly",
        action="store_true",
        help="preprocess only. Adios or pickle saving and no train",
    )
    parser.add_argument(
        "--inputfile", help="input file", type=str, default="./logs/energy/config.json"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--adios",
        help="Adios gan_dataset",
        action="store_const",
        dest="format",
        const="adios",
    )
    group.add_argument(
        "--pickle",
        help="Pickle gan_dataset",
        action="store_const",
        dest="format",
        const="pickle",
    )
    parser.set_defaults(format="pickle")

    args = parser.parse_args()

    dirpwd = os.path.dirname(os.path.abspath(__file__))
    input_filename = os.path.join(dirpwd, args.inputfile)
    with open(input_filename, "r") as f:
        config = json.load(f)
    hydragnn.utils.setup_log(get_log_name_config(config))
    ##################################################################################################################
    # Always initialize for multi-rank training.
    comm_size, rank = hydragnn.utils.setup_ddp()
    ##################################################################################################################
    comm = MPI.COMM_WORLD

    datasetname = config["Dataset"]["name"]

    comm.Barrier()
    if args.preonly:
        sys.exit(0)

    timer = Timer("load_data")
    timer.start()

    model = create_model_config(
        config=config["NeuralNetwork"],
        verbosity=config["Verbosity"]["level"],
    )

    model = torch.nn.parallel.DistributedDataParallel(
        model
    )

    load_existing_model(model, model_name, path="./logs/")
    model.eval()

    test_MAE = 0.0
    test_MAE_per_atom = 0.0

    training_minmax_graph_feature = []

    with open("small_crystal_dataset/serialized_dataset/NiPt-trainset.pkl", "rb") as f:
        _ = pickle.load(f)
        training_minmax_graph_feature = pickle.load(f)

    test_minmax_graph_feature = []

    with open("large_crystal_dataset/serialized_dataset/NiPt-test.pkl", "rb") as f:
        _ = pickle.load(f)
        test_minmax_graph_feature = pickle.load(f)

    with open("large_crystal_dataset/serialized_dataset/NiPt-testset.pkl", "rb") as f:
        _ = pickle.load(f)
        _ = pickle.load(f)
        dataset = pickle.load(f)

    num_samples = len(dataset)
    true_values = []
    predicted_values = []

    for data_id, data in enumerate(tqdm(dataset)):
        scaled_predicted = model(data.to(get_device()))
        predicted = scaled_predicted[0].item() * (training_minmax_graph_feature[1, 0] - training_minmax_graph_feature[0, 0]) + \
                    training_minmax_graph_feature[0, 0]
        predicted_values.append(predicted)
        scaled_true = data.y[0].item()
        true = scaled_true * (test_minmax_graph_feature[1, 0] - test_minmax_graph_feature[0, 0]) + test_minmax_graph_feature[0, 0]
        true_values.append(true)
        num_atoms = data.x.shape[0]
        MAE = abs(true - predicted)
        MAE_per_atom = MAE / num_atoms
        test_MAE += MAE / num_samples
        test_MAE_per_atom += MAE_per_atom / num_samples

    print("Test MAE: ", test_MAE)
    print("Test MAE per atom: ", test_MAE_per_atom)

    hist2d_norm = getcolordensity(true_values, predicted_values)

    fig, ax = plt.subplots()
    plt.scatter(
        true_values, predicted_values, s=8, c=hist2d_norm, vmin=0, vmax=1
    )
    plt.clim(0, 1)
    ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", color="red")
    plt.xlim([training_minmax_graph_feature[0, 0], training_minmax_graph_feature[1, 0]])
    plt.ylim([training_minmax_graph_feature[0, 0], training_minmax_graph_feature[1, 0]])
    plt.colorbar()
    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    plt.draw()
    plt.tight_layout()
    plt.savefig("./Scatterplot" + ".png", dpi=400)
