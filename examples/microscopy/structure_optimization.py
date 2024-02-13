import torch
from torch_geometric.data import Data
from ase.io import read, write

from hydragnn.preprocess.utils import get_radius_graph_pbc

from torch_geometric.transforms import LocalCartesian
transform_coordinates = LocalCartesian(norm=False, cat=False)

# Example usage:
# Define your energy function
# def energy_function(positions):
#     ...

# Define initial atomic positions
# initial_positions = ...

# Call BFGS optimization
# optimized_positions, final_energy = structure_optimization_BFGS_pytorch(energy_function, initial_positions)


def poscar_to_torch_geometric(filename):
    """
    Read atomistic structure from a POSCAR file and generate a torch_geometric.data object.

    Parameters:
    filename (str): The name of the POSCAR file.

    Returns:
    data (torch_geometric.data.Data): The atomistic structure represented as a torch_geometric data object.
    """
    # Read the POSCAR file
    ase_object = read(filename, format='vasp')

    # Extract atomic positions
    pos = torch.tensor(ase_object.positions, dtype=torch.float)

    # Extract atomic numbers
    atomic_numbers = ase_object.get_atomic_numbers()
    atomic_numbers_torch = torch.tensor(atomic_numbers, dtype=torch.long).unsqueeze(1)

    x = torch.cat((atomic_numbers_torch, pos), dim=1)

    # Create the torch_geometric data object
    data = Data(pos=pos, x=x, supercell_size=torch.tensor(ase_object.cell.array).float())

    return data


def conjugate_gradient_pytorch(data_object, hydragnn_model, tol=1e-5, max_iter=1000):
    """
    Conjugate gradient optimization for structure optimization of atomistic structures using PyTorch tensors.

    Parameters:
        energy_function (callable): A function that computes the energy given atomic positions.
        gradient_function (callable): A function that computes the gradient of energy given atomic positions.
        initial_positions (torch.Tensor): Initial atomic positions.
        tol (float): Tolerance for convergence.
        max_iter (int): Maximum number of iterations.

    Returns:
        torch.Tensor: Optimized atomic positions.
        float: Final energy.
    """

    # Extract atomic positions
    positions = torch.tensor(data_object.pos, dtype=torch.float)

    def energy_function(positions):
        global data_object
        data_object.pos = positions

        data_object = transform_coordinates(data_object)
        data_object.x[:, 1:] = positions
        data_object = transform_coordinates(data_object)
        predictions = hydragnn_model(data_object)
        energy = predictions[0]

        return energy

    def gradient_function(positions):
        global data_object
        data_object.pos = positions
        data_object.x[:, 1:] = positions
        data_object = transform_coordinates(data_object)
        predictions = hydragnn_model(data_object)
        forces = predictions[0]

        return forces

    #energy = energy_function(positions)
    gradient = gradient_function(positions)
    conjugate_direction = -gradient

    for _ in range(max_iter):
        # Line search
        alpha = torch.dot(gradient, gradient) / torch.dot(conjugate_direction,
                                                          gradient_function(positions + alpha * conjugate_direction))

        # Update position
        new_positions = positions + alpha * conjugate_direction

        # Update gradient
        new_gradient = gradient_function(new_positions)

        # Calculate beta (Fletcher-Reeves update)
        beta = torch.dot(new_gradient, new_gradient) / torch.dot(gradient, gradient)

        # Update conjugate direction
        conjugate_direction = -new_gradient + beta * conjugate_direction

        # Check convergence
        if torch.norm(new_positions - positions) < tol:
            break

        # Update positions and gradient
        positions = new_positions.clone().requires_grad_(True)
        gradient = new_gradient
        energy = energy_function(positions)

    return positions.detach(), energy


from torch.optim import LBFGS


def structure_optimization_BFGS_pytorch(energy_function, initial_positions, tol=1e-5, max_iter=1000):
    """
    Structure optimization using BFGS method for atomistic structures using PyTorch tensors.

    Parameters:
        energy_function (callable): A function that computes the energy given atomic positions and returns the energy.
        initial_positions (torch.Tensor): Initial atomic positions.
        tol (float): Tolerance for convergence.
        max_iter (int): Maximum number of iterations.

    Returns:
        torch.Tensor: Optimized atomic positions.
        float: Final energy.
    """
    # Convert initial positions to a tensor requiring gradients
    positions = initial_positions.clone().requires_grad_()

    # Create LBFGS optimizer
    optimizer = LBFGS([positions], lr=1)

    # Define closure for LBFGS optimizer
    def closure():
        optimizer.zero_grad()
        energy = energy_function(positions)
        return energy

    # Optimization loop
    for _ in range(max_iter):
        # Perform optimization step
        optimizer.step(closure)

        # Check convergence
        if optimizer.state[positions]['n_iter'] > 1 and torch.abs(
                optimizer.state[positions]['func_vals'][0] - optimizer.state[positions]['func_vals'][-1]) < tol:
            break

    # Retrieve optimized positions and energy
    optimized_positions = positions.detach()
    final_energy = energy_function(optimized_positions).item()

    return optimized_positions, final_energy



import json, os
import logging
from mpi4py import MPI
import argparse

import hydragnn
from hydragnn.utils.time_utils import Timer
from hydragnn.utils.model import load_existing_model
from hydragnn.models.create import create_model_config

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


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


if __name__ == "__main__":

    modelname = "MO2"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile", help="input file", type=str, default="./logs/MO2/config.json"
    )
    group = parser.add_mutually_exclusive_group()

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

    comm.Barrier()

    timer = Timer("load_data")
    timer.start()

    hydragnn_model = create_model_config(
        config=config["NeuralNetwork"],
        verbosity=config["Verbosity"]["level"],
    )

    hydragnn_model = torch.nn.parallel.DistributedDataParallel(
        hydragnn_model
    )

    # Read the POSCAR file
    poscar_filename = "./POSCAR_1"

    atoms = read(poscar_filename, format='vasp')

    # Convert ASE object into PyTorch-Geometric object
    data_object = poscar_to_torch_geometric(poscar_filename)

    add_edges_pbc = get_radius_graph_pbc(radius=config["NeuralNetwork"]["Architecture"]["radius"], max_neighbours=config["NeuralNetwork"]["Architecture"]["max_neighbours"])
    data_object = add_edges_pbc(data_object)

    data_object = transform_coordinates(data_object)

    load_existing_model(hydragnn_model, modelname, path="./logs/")
    hydragnn_model.eval()

    optimized_positions = conjugate_gradient_pytorch(data_object, hydragnn_model, tol=1e-5, max_iter=1000)

    # Convert PyTorch tensor to NumPy array
    optimized_positions_np = optimized_positions.numpy()

    # Update positions of atoms in the ASE object
    atoms.set_positions(optimized_positions_np)

    # Write the updated positions to a new POSCAR file with direct coordinates
    write('new_poscar.vasp', atoms, format='vasp', direct=True)

