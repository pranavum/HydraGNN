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
from hydragnn.utils.print_utils import print_distributed, iterate_tqdm
from hydragnn.utils.time_utils import Timer
from hydragnn.utils.pickledataset import SimplePickleDataset
from hydragnn.utils.model import print_model
from hydragnn.preprocess.utils import get_radius_graph
from torch_geometric.transforms import Distance

import numpy as np

try:
    from hydragnn.utils.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    pass

import torch_geometric.data
import torch
import torch.distributed as dist


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


def nsplit(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m): (i + 1) * k + min(i + 1, m)] for i in range(n))


def calculate_maximum_inputs_and_outputs(dataset):
    local_maximum_input = torch.zeros((len(dataset), dataset[0].x.shape[1]))
    local_maximum_output = torch.zeros((len(dataset), len(config['NeuralNetwork']['Variables_of_interest']['output_index'])))

    for item_index, data_item in enumerate(dataset):
        local_maximum_input[item_index,:] = dataset[item_index].x.abs().max(0)[0]
        output_index_count = 0
        for var_index in range(local_maximum_output.shape[1]):
            # FIXME: this loop works only it the nodal feature has length equal to 1
            local_maximum_output[item_index, var_index] = dataset[item_index].y[output_index_count:(output_index_count+dataset[item_index].num_nodes)].abs().max(0)[0]
            output_index_count = output_index_count+dataset[item_index].num_nodes

    maximum_input = local_maximum_input.max(0)[0]
    maximum_output = local_maximum_output.max(0)[0]

    return maximum_input, maximum_output

def normalize_data_sample(data, input_scaling_tensor, output_scaling_tensor):

    assert data.x.shape[1] == input_scaling_tensor.shape[0]

    # FIXME: this loop works only it the nodal feature has length equal to 1
    assert data.y.shape[0]/data.num_nodes == output_scaling_tensor.shape[0]

    data.x = torch.matmul(data.x, torch.diag(1./input_scaling_tensor))

    output_index_count = 0
    # FIXME: this loop works only it the nodal feature has length equal to 1
    for output_index in range(0,output_scaling_tensor.shape[0]):
        data.y[output_index_count:(output_index_count+data.num_nodes)] = data.y[output_index_count:(output_index_count+data.num_nodes)] * 1/output_scaling_tensor[output_index]
        output_index_count = output_index_count + data.num_nodes

    return data

def normalize_data_sample_log_scale_fluence(data, input_scaling_tensor, output_scaling_tensor):

    assert data.x.shape[1] == input_scaling_tensor.shape[0]

    # FIXME: this loop works only it the nodal feature has length equal to 1
    assert data.y.shape[0]/data.num_nodes == output_scaling_tensor.shape[0]

    data.x[:,[0,2,3,4]] = torch.matmul(data.x[:,[0,2,3,4]], torch.diag(1./input_scaling_tensor[[0,2,3,4]]))
    data.x[:,[1]] = torch.log(data.x[:,[1]]+1)

    output_index_count = 0
    # FIXME: this loop works only it the nodal feature has length equal to 1
    for output_index in range(0,output_scaling_tensor.shape[0]):
        data.y[output_index_count:(output_index_count+data.num_nodes)] = data.y[output_index_count:(output_index_count+data.num_nodes)] * 1/output_scaling_tensor[output_index]
        output_index_count = output_index_count + data.num_nodes

    return data


def read_mesh_coordinates_and_nodal_features_from_csv_file(time_step_index):
    #file_relative_path = "examples/concrete_shielding/dataset/concrete_shielding/nodal_info_time/PointData_" + str(time_step_index) + '.csv'
    file_relative_path = "examples/concrete_shielding/dataset/concrete_shielding/inputs/PointData_" + str(
        time_step_index) + '.csv'
    absolute_path = os.path.abspath(os.getcwd())
    df = pd.read_csv(absolute_path + '/' + file_relative_path)
    x = torch.tensor(df['Points:0']).unsqueeze(1).float()
    y = torch.tensor(df['Points:1']).unsqueeze(1).float()
    z = torch.tensor(df['Points:2']).unsqueeze(1).float()

    torch_coordinates = torch.cat([x, y, z], dim=1)

    torch_temperature = torch.tensor(df['temperature']).unsqueeze(1).float()
    torch_fluence = torch.tensor(df['fluence']).unsqueeze(1).float()
    #torch_axial_stress = torch.tensor(df['axial_stress']).unsqueeze(1).float()
    torch_hoop_stress = torch.tensor(df['hoop_stress']).unsqueeze(1).float()
    torch_bc_r = torch.tensor(df['BC_r']).unsqueeze(1).float()
    torch_bc_z = torch.tensor(df['BC_z']).unsqueeze(1).float()

    #return torch_coordinates, torch_temperature, torch_fluence, torch_axial_stress, torch_hoop_stress
    return torch_coordinates, torch_temperature, torch_fluence, torch_hoop_stress, torch_bc_r, torch_bc_z


def read_node_information_for_time_step(time_step_index, vertex_index):
    #file_relative_path = 'examples/concrete_shielding/dataset/concrete_shielding/training_data' + '/' + 'workdir.' + str(
    #    vertex_index) + '/' + 'moose_out.csv'
    file_relative_path = 'examples/concrete_shielding/dataset/concrete_shielding' + '/' + 'workdir.' + str(
        vertex_index) + '/' + 'moose_out.csv'
    absolute_path = os.path.abspath(os.getcwd())
    df = pd.read_csv(absolute_path + '/' + file_relative_path)
    average_linear_expansion = df['average_linear_expansion'][time_step_index]
    #average_damage_hcp_x = df['average_damage_hcp_x'][time_step_index]
    #average_damage_hcp_y = df['average_damage_hcp_y'][time_step_index]
    average_damage_hcp = df['average_damage_hcp'][time_step_index]

    return average_linear_expansion, average_damage_hcp


def generate_graphdata(time_step_index):
    #torch_coordinates, torch_temperature, torch_fluence, torch_axial_stress, torch_hoop_stress = read_mesh_coordinates_and_nodal_features_from_csv_file(
    #    time_step_index)
    torch_coordinates, torch_temperature, torch_fluence, torch_hoop_stress, torch_bc_r, torch_bc_z = read_mesh_coordinates_and_nodal_features_from_csv_file(
        time_step_index)

    average_linear_expansion_list = [read_node_information_for_time_step(time_step_index, vertex_index)[0] for
                                     vertex_index in range(torch_coordinates.shape[0])]
    average_damage_list = [read_node_information_for_time_step(time_step_index, vertex_index)[1] for vertex_index in
                           range(torch_coordinates.shape[0])]

    torch_average_linear_expansion = torch.tensor(average_linear_expansion_list).unsqueeze(1).float()
    torch_average_damage = torch.tensor(average_damage_list).unsqueeze(1).float()

    compute_edges = get_radius_graph(
        radius=0.30,
        loop=False,
        max_neighbours=100,
    )
    compute_edges_lengths = Distance(norm=False, cat=True)

    #x = torch.cat([torch_temperature, torch_fluence, torch_axial_stress, torch_hoop_stress], dim=1)
    x = torch.cat([torch_temperature, torch_fluence, torch_hoop_stress, torch_bc_r, torch_bc_z], dim=1)
    y = torch.cat([torch_average_linear_expansion, torch_average_damage], dim=0)

    data_object = torch_geometric.data.data.Data(x=x, y=y)
    data_object.pos = torch_coordinates
    data_object.y_loc = torch.tensor([0, x.shape[0], x.shape[0] * 2]).reshape(1, 3)
    data_object = compute_edges(data_object)
    data_object = compute_edges_lengths(data_object)
    data_object.time_step_index = time_step_index

    return data_object


def concrete_datasets_load(datadir_path, seed=None, frac=[0.9, 0.05, 0.05]):
    if seed is not None:
        random.seed(seed)
    times_all = []
    #for subdir, dirs, files in os.walk(datadir_path + '/nodal_info_time'):
    for subdir, dirs, files in os.walk(datadir_path + '/inputs'):
        for file in files:
            if not file.startswith('.'):
                time_step = int(file[10:].split('.')[0])
                times_all.append(time_step)
    print("Total:", len(times_all))

    a = list(range(len(times_all)))
    ix0, ix1, ix2 = np.split(
        a, [int(frac[0] * len(a)), int((frac[0] + frac[1]) * len(a))]
    )

    trainset = []
    valset = []
    testset = []

    for i in ix0:
        trainset.append(times_all[i])

    for i in ix1:
        valset.append(times_all[i])

    for i in ix2:
        testset.append(times_all[i])

    return (
        [torch.tensor(trainset), torch.tensor(valset), torch.tensor(testset)]
    )


class ConcreteRawDatasetFactory:
    def __init__(self, datafile, var_config, seed=43):
        self.var_config = var_config

        ## Read full data
        times_sets = concrete_datasets_load(
            datafile, seed=seed
        )

        info([len(x) for x in times_sets])
        self.dataset_lists = list()
        for idataset, valueset in enumerate(zip(times_sets)):
            self.dataset_lists.append(valueset)

    def get(self, label):
        ## Set only assigned label data
        labelnames = ["trainset", "valset", "testset"]
        index = labelnames.index(label)

        valueset = self.dataset_lists[index]
        return valueset


class ConcreteRawDataset(torch.utils.data.Dataset):
    def __init__(self, datasetfactory, label):
        self.valueset = datasetfactory.get(label)

    def __len__(self):
        return len(self.valueset)

    def __getitem__(self, idx):
        data = generate_graphdata(idx)
        return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampling", type=float, help="sampling ratio", default=None)
    parser.add_argument(
        "--preonly",
        action="store_true",
        help="preprocess only. Data saving and no train",
    )
    parser.add_argument("--shmem", action="store_true", help="use shmem")
    parser.add_argument("--mae", action="store_true", help="do mae calculation")

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--adios",
        help="Adios dataset",
        action="store_const",
        dest="format",
        const="adios",
    )
    group.add_argument(
        "--pickle",
        help="Pickle dataset",
        action="store_const",
        dest="format",
        const="pickle",
    )
    group.add_argument(
        "--csv", help="CSV dataset", action="store_const", dest="format", const="csv"
    )
    parser.set_defaults(format="pickle")
    args = parser.parse_args()

    node_output_feature_names = ["average_linear_expansion", "average_damage"]
    node_output_feature_dim = [1, 1]
    dirpwd = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(dirpwd, "dataset/concrete_shielding")
    ##################################################################################################################
    input_filename = os.path.join(dirpwd, "concrete_shielding.json")
    ##################################################################################################################
    # Configurable run choices (JSON file that accompanies this example script).
    with open(input_filename, "r") as f:
        config = json.load(f)
    verbosity = config["Verbosity"]["level"]
    var_config = config["NeuralNetwork"]["Variables_of_interest"]
    var_config["output_names"] = [
        node_output_feature_names[item]
        for ihead, item in enumerate(var_config["output_index"])
    ]
    var_config["input_node_feature_names"] = ["temperature", "fluence", "axial_stress", "hoop_stress"]
    var_config["input_node_feature_dims"] = [1, 1, 1, 1]
    ##################################################################################################################
    # Always initialize for multi-rank training.
    comm_size, rank = hydragnn.utils.setup_ddp()
    ##################################################################################################################

    comm = MPI.COMM_WORLD

    ## Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%%(levelname)s (rank %d): %%(message)s" % (rank),
        datefmt="%H:%M:%S",
    )

    log_name = "concrete_shielding_fullx"
    hydragnn.utils.setup_log(log_name)
    writer = hydragnn.utils.get_summary_writer(log_name)
    with open("./logs/" + log_name + "/config.json", "w") as f:
        json.dump(config, f)

    if args.preonly:
        norm_yflag = True  # True
        time_sets = concrete_datasets_load(
            datadir, seed=43
        )
        info([len(x) for x in time_sets])
        dataset_lists = [[] for dataset in time_sets]
        for idataset, timeset in enumerate(time_sets):

            rx = list(nsplit(range(len(timeset)), comm_size))[rank]
            info("subset range:", idataset, len(timeset), rx.start, rx.stop)
            ## local portion
            _timeset = timeset[rx.start: rx.stop]
            info("local smileset size:", len(_timeset))

            setname = ["trainset", "valset", "testset"]
            if args.format == "pickle":
                dirname = "dataset/pickle"
                if rank == 0:
                    if not os.path.exists(dirname):
                        os.makedirs(dirname)
                    with open("%s/%s.meta" % (dirname, setname[idataset]), "w") as f:
                        f.write(str(len(timeset)))

            for i, time in iterate_tqdm(
                    enumerate(_timeset), verbosity, total=len(_timeset)
            ):
                data = generate_graphdata(time.item())
                dataset_lists[idataset].append(data)

                ## (2022/07) This is for testing to compare with Adios
                ## pickle write
                if args.format == "pickle":
                    fname = "%s/concrete_shielding-%s-%d.pk" % (
                        dirname,
                        setname[idataset],
                        rx.start + i,
                    )
                    with open(fname, "wb") as f:
                        pickle.dump(data, f)

        ## local data
        if args.format == "adios":
            _trainset = dataset_lists[0]
            _valset = dataset_lists[1]
            _testset = dataset_lists[2]

            adwriter = AdiosWriter("dataset/concrete_shielding.bp", comm)
            adwriter.add("trainset", _trainset)
            adwriter.add("valset", _valset)
            adwriter.add("testset", _testset)
            adwriter.save()

        sys.exit(0)

    timer = Timer("load_data")
    timer.start()
    if args.format == "adios":
        opt = {"preload": True, "shmem": False}
        if args.shmem:
            opt = {"preload": False, "shmem": True}
        trainset = AdiosDataset(
            "dataset/concrete_shielding.bp", "trainset", comm, opt
        )
        valset = AdiosDataset("dataset/concrete_shielding.bp", "valset", comm, opt)
        testset = AdiosDataset("dataset/concrete_shielding.bp", "testset", comm, opt)
    elif args.format == "csv":
        fact = ConcreteRawDatasetFactory(
            "dataset/concrete_shielding.csv",
            var_config=var_config,
            sampling=args.sampling,
        )
        trainset = ConcreteRawDataset(fact, "trainset")
        valset = ConcreteRawDataset(fact, "valset")
        testset = ConcreteRawDataset(fact, "testset")
    elif args.format == "pickle":
        dirname = "dataset/pickle"
        trainset = SimplePickleDataset(dirname, "concrete_shielding", "trainset")
        valset = SimplePickleDataset(dirname, "concrete_shielding", "valset")
        testset = SimplePickleDataset(dirname, "concrete_shielding", "testset")
    else:
        raise NotImplementedError("No supported format: %s" % (args.format))

    info("Data load")
    info(
        "trainset,valset,testset size: %d %d %d"
        % (len(trainset), len(valset), len(testset))
    )

    maximum_input, maximum_output = calculate_maximum_inputs_and_outputs(trainset)

    torch.save(maximum_input, 'maximum_input.pt')
    torch.save(maximum_output, 'maximum_output.pt')

    normalized_trainset = [normalize_data_sample_log_scale_fluence(data_item, maximum_input, maximum_output) for data_item in trainset]
    normalized_valset = [normalize_data_sample_log_scale_fluence(data_item, maximum_input, maximum_output) for data_item in valset]
    normalized_testset = [normalize_data_sample_log_scale_fluence(data_item, maximum_input, maximum_output) for data_item in testset]

    (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
        normalized_trainset, normalized_valset, normalized_testset, config["NeuralNetwork"]["Training"]["batch_size"]
    )

    config = hydragnn.utils.update_config(config, train_loader, val_loader, test_loader)
    timer.stop()

    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=verbosity,
    )
    model = hydragnn.utils.get_distributed_model(model, verbosity)

    if rank == 0:
        print_model(model)
    dist.barrier()

    learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )

    hydragnn.utils.load_existing_model_config(
        model, config["NeuralNetwork"]["Training"], optimizer=optimizer
    )

    ##################################################################################################################

    hydragnn.train.train_validate_test(
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        writer,
        scheduler,
        config["NeuralNetwork"],
        log_name,
        verbosity,
        create_plots=False,
    )

    hydragnn.utils.save_model(model, optimizer, log_name)
    hydragnn.utils.print_timers(verbosity)

    if args.mae:
        ##################################################################################################################
        for isub, (loader, setname) in enumerate(
                zip([train_loader, val_loader, test_loader], ["train", "val", "test"])
        ):
            error, rmse_task, true_values, predicted_values = hydragnn.train.test(
                loader, model, verbosity
            )
            for ihead in range(0,len(var_config["output_names"])):
                fig, axs = plt.subplots(1, 3, figsize=(18, 6))

                head_true = np.asarray(true_values[ihead].cpu()).squeeze()
                head_pred = np.asarray(predicted_values[ihead].cpu()).squeeze()
                ifeat = var_config["output_index"][ihead]
                outtype = var_config["type"][ihead]
                varname = node_output_feature_names[ifeat]

                ax = axs[isub]
                error_mae = np.mean(np.abs(head_pred - head_true))
                error_rmse = np.sqrt(np.mean(np.abs(head_pred - head_true) ** 2))
                print(varname, ": mae=", error_mae, ", rmse= ", error_rmse)

                ax.scatter(
                    head_true,
                    head_pred,
                    s=7,
                    linewidth=0.5,
                    edgecolor="b",
                    facecolor="none",
                )
                minv = np.minimum(np.amin(head_pred), np.amin(head_true))
                maxv = np.maximum(np.amax(head_pred), np.amax(head_true))
                ax.plot([minv, maxv], [minv, maxv], "r--")
                ax.set_title(setname + "; " + varname , fontsize=16)
                ax.text(
                    minv + 0.1 * (maxv - minv),
                    maxv - 0.1 * (maxv - minv),
                    "MAE: {:.2f}".format(error_mae),
                )
                if rank == 0:
                    fig.savefig("./logs/" + log_name + "/" + varname + "_all.png")
                plt.close()

    if args.shmem:
        trainset.unlink()

    sys.exit(0)
