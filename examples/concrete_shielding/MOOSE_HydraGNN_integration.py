import os, json

import numpy
import torch

from concrete_shielding_utils import generate_graph_input

import hydragnn
from hydragnn.utils.distributed import setup_ddp
from hydragnn.utils.model import load_existing_model

def interrogate_hydragnn_model(mesh_file, input_features_numpy_array):

    CaseDir = os.path.join(
        os.path.dirname(__file__),
        "logs",
    )

    maximum_input = torch.load('maximum_input.pt')
    maximum_output = torch.load('maximum_output.pt')

    dirpwd = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(dirpwd, "dataset/concrete_shielding")
    ##################################################################################################################
    config_filename = CaseDir + "/concrete_shielding_fullx/" + "config.json"
    ##################################################################################################################

    # Configurable run choices (JSON file that accompanies this example script).
    with open(config_filename, "r") as f:
        config = json.load(f)

    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=config["Verbosity"]["level"],
    )

    model = torch.nn.parallel.DistributedDataParallel(
        model
    )

    model_name = "concrete_shielding_fullx"
    load_existing_model(model, model_name, path="./logs/")

    # Generate graph for input data
    input_graph = generate_graph_input(mesh_file, input_features_numpy_array)

    # Rescale input features between [0,1]
    input_graph.x = torch.matmul(input_graph.x.float(), torch.diag(1. / maximum_input))

    # Generate prediction
    prediction_list = model(input_graph)

    prediction_torch = torch.cat([prediction_list[0], prediction_list[1]], dim=1)

    # Remap predictions back to the physical domain
    prediction_physical_domain = torch.matmul(prediction_torch, torch.diag(1. / maximum_output))

    # Convert Pytorch tensor into numpy arracy that can be handles by C++ wrapper for MOOSE
    numpy_prediction = prediction_physical_domain.detach().numpy()

    return numpy_prediction


if __name__ == "__main__":
    world_size, world_rank = setup_ddp()
    mesh_file = "PointData_0.csv"
    dataframe = numpy.genfromtxt("example_input_array.csv",
                        delimiter=",", dtype=float)
    numpy_input_features = dataframe[:, [1, 2, 3, 4, 5]]
    numpy_prediction = interrogate_hydragnn_model(mesh_file, numpy_input_features)
    print(numpy_input_features)

