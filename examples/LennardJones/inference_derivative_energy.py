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
from hydragnn.utils.distributed import get_device
from hydragnn.utils.model import load_existing_model
from hydragnn.utils.pickledataset import SimplePickleDataset
from hydragnn.utils.config_utils import (
    update_config,
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


def get_log_name_config(config):
    return (
        config["NeuralNetwork"]["Architecture"]["model_type"]
        + "-r-"
        + str(config["NeuralNetwork"]["Architecture"]["radius"])
        + "-ncl-"
        + str(config["NeuralNetwork"]["Architecture"]["num_conv_layers"])
        + "-hd-"
        + str(config["NeuralNetwork"]["Architecture"]["hidden_dim"])
        + "-ne-"
        + str(config["NeuralNetwork"]["Training"]["num_epoch"])
        + "-lr-"
        + str(config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"])
        + "-bs-"
        + str(config["NeuralNetwork"]["Training"]["batch_size"])
        + "-node_ft-"
        + "".join(
            str(x)
            for x in config["NeuralNetwork"]["Variables_of_interest"][
                "input_node_features"
            ]
        )
        + "-task_weights-"
        + "".join(
            str(weigh) + "-"
            for weigh in config["NeuralNetwork"]["Architecture"]["task_weights"]
        )
    )

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
#def predict_derivative_test(argv=None):

    modelname = "LJ"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile", help="input file", type=str, default="./logs/LJ/config.json"
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

    datasetname = "LJ"

    comm.Barrier()

    timer = Timer("load_data")
    timer.start()
    if args.format == "pickle":
        info("Pickle load")
        basedir = os.path.join(
            os.path.dirname(__file__), "dataset", "%s.pickle" % modelname
        )
        trainset = SimplePickleDataset(basedir=basedir, label="trainset", var_config=config["NeuralNetwork"]["Variables_of_interest"])
        valset = SimplePickleDataset(basedir=basedir, label="valset", var_config=config["NeuralNetwork"]["Variables_of_interest"])
        testset = SimplePickleDataset(basedir=basedir, label="testset", var_config=config["NeuralNetwork"]["Variables_of_interest"])
        pna_deg = trainset.pna_deg
    else:
        raise NotImplementedError("No supported format: %s" % (args.format))

    model = create_model_config(
        config=config["NeuralNetwork"],
        verbosity=config["Verbosity"]["level"],
    )

    model = torch.nn.parallel.DistributedDataParallel(
        model
    )

    load_existing_model(model, modelname, path="./logs/")
    model.eval()

    variable_index = 0
    #for output_name, output_type, output_dim in zip(config["NeuralNetwork"]["Variables_of_interest"]["output_names"], config["NeuralNetwork"]["Variables_of_interest"]["type"], config["NeuralNetwork"]["Variables_of_interest"]["output_dim"]):

    test_MAE = 0.0

    num_samples = len(testset)
    true_values = []
    predicted_values = []
    forces = []
    deriv_energy = []

    for data_id, data in enumerate(tqdm(testset)):
        data.pos.requires_grad = True
        predicted = model(data.to(get_device()))
        predicted = predicted[variable_index].flatten()
        start = data.y_loc[0][variable_index].item()
        end = data.y_loc[0][variable_index + 1].item()
        true = data.y[start:end, 0]
        test_MAE += torch.norm(predicted - true, p=1).item() / len(testset)
        #predicted.backward(retain_graph=True)
        #gradients = data.pos.grad
        grads_energy = torch.autograd.grad(outputs=predicted, inputs=data.pos,
                                           grad_outputs=data.num_nodes * torch.ones_like(predicted),
                                           retain_graph=False)[0]
        predicted_values.extend(predicted.tolist())
        true_values.extend(true.tolist())
        deriv_energy.extend(grads_energy.flatten().tolist())
        forces.extend(data.forces_pre_scaled.flatten().tolist())

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
    plt.title(f"energy")
    plt.draw()
    plt.tight_layout()
    plt.savefig(f"./energy_Scatterplot" + ".png", dpi=400)

    print(f"Test MAE energy: ", test_MAE)

    hist2d_norm = getcolordensity(deriv_energy, forces)
    fig, ax = plt.subplots()
    plt.scatter(
        deriv_energy, forces, s=8, c=hist2d_norm, vmin=0, vmax=1
    )
    plt.clim(0, 1)
    ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", color="red")
    plt.colorbar()
    plt.xlabel("Derivatives of energy")
    plt.ylabel("Forces")
    plt.title(f"derivative_energy")
    plt.draw()
    plt.tight_layout()
    plt.savefig(f"./Forces_Scatterplot" + ".png", dpi=400)
