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

def compute_sensitivity_grad(model, pred, input):
    #output = model(input.to(get_device()))
    sensitivities = []
    for layer_name, layer in model.named_modules():
        layer_sensitivities = []
        for param_name, param in layer.named_parameters():
            # Find output gradient wrt parameter
            param_grad = torch.autograd.grad(pred, param, grad_outputs=torch.ones_like(pred), retain_graph=True, allow_unused=True)[0]
            if param_grad != None:
                print(param_grad.shape)
                # Add value of gradient to list of gradients in current layer
                print(torch.mean(param_grad).item())
                layer_sensitivities.append(torch.mean(param_grad).item()) #.detach().abs().mean().item())
        # Add average gradient value per layer to list of layers
        sensitivities.append(sum(layer_sensitivities) / len(layer_sensitivities) if layer_sensitivities else 0)
    return sensitivities

def mean_sensitivity(sensitivity_list):
    sum_sensitivity = {}
    for sensitivity_data_batch in sensitivity_list:
        for layer_sensitivity_index in range(len(sensitivity_data_batch)):
            if layer_sensitivity_index in sum_sensitivity:
                sum_sensitivity[layer_sensitivity_index].append(sensitivity_data_batch[layer_sensitivity_index])
            else:
                sum_sensitivity[layer_sensitivity_index] = [sensitivity_data_batch[layer_sensitivity_index]]
    mean_sensitivity = []
    for index, lis in sum_sensitivity.items():
        mean_sensitivity.append(sum(lis) / len(lis))
    return mean_sensitivity

#if __name__ == "__main__":
def find_sensitivity(argv=None):

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
    
    total_sensitivities = {}

    for output_name, output_type, output_dim in zip(config["NeuralNetwork"]["Variables_of_interest"]["output_names"], config["NeuralNetwork"]["Variables_of_interest"]["type"], config["NeuralNetwork"]["Variables_of_interest"]["output_dim"]):

        num_samples = len(testset)
        sensitivities = []

        for data_id, data in enumerate(tqdm(testset)):
            predicted = model(data.to(get_device()))
            predicted = predicted[variable_index] #.flatten()
            sensitivities.append(compute_sensitivity_grad(model, predicted, data))
        
        sensitivities = mean_sensitivity(sensitivities)

        print(f"Test sensitivities {output_name}: ", list(zip(sensitivities,
                                                         [name for name, layer in model.named_modules()])))
        
        plt.figure(figsize=(50, 8))
        plt.bar([name for name, layer in model.named_modules()], sensitivities, color="blue")
        plt.xlabel('Module Names')
        plt.ylabel('Sensitivity')
        plt.title('Sensitivity of Model Modules')

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"Layer_Sensitivities_{output_name}.png")
        total_sensitivities[output_name] = sensitivities

        variable_index += 1

    return
    sys.exit(0)

    variable_index = 0
    sensitivities = {}

    perturbation = 1e-7

    for output_name, output_type, output_dim in zip(config["NeuralNetwork"]["Variables_of_interest"]["output_names"], config["NeuralNetwork"]["Variables_of_interest"]["type"], config["NeuralNetwork"]["Variables_of_interest"]["output_dim"]):

        test_sensitivities = 0.0

        num_samples = len(testset)
        true_values = []
        predicted_values = []

        for data_id, data in enumerate(tqdm(testset)):
            predicted = model(data.to(get_device()))
            predicted = predicted[variable_index].flatten()
            start = data.y_loc[0][variable_index].item()
            end = data.y_loc[0][variable_index + 1].item()
            true = data.y[start:end, 0]
            #test_MAE += torch.norm(predicted - true, p=1).item()/len(testset)

            predicted_values.extend(predicted.tolist())
            true_values.extend(true.tolist())

            perturbed_data = data.clone()
            perturbed_data.pos[:, 1:] += perturbation
            perturbed_output = model(perturbed_data.to(get_device()))
            perturbed_output = perturbed_output[variable_index].flatten()
            start = perturbed_data.y_loc[0][variable_index].item()
            end = perturbed_data.y_loc[0][variable_index + 1].item()
            true = perturbed_data.y[start:end, 0]

            output_change = (perturbed_output - predicted).abs().mean().item()

            test_sensitivities += output_change / perturbation / num_samples

        print(f"Test sensitivities {output_name}: ", test_sensitivities)
        sensitivities[output_name] = test_sensitivities

        variable_index += 1


    # for name, layer in model.named_modules():
    #     print(name)
    #     for param_name, param in layer.named_parameters():
    #         print(param_name, param)