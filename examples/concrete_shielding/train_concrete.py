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

import numpy as np

try:
    from hydragnn.utils.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    pass

import torch_geometric.data
import torch
import torch.distributed as dist

from concrete_shielding_utils import calculate_maximum_inputs_and_outputs, normalize_data_sample, \
    normalize_data_sample_log_scale_fluence, read_mesh_coordinates_and_nodal_features_from_csv_file, \
    read_node_information_for_time_step, generate_graphdata


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


def nsplit(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m): (i + 1) * k + min(i + 1, m)] for i in range(n))


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
    a = random.sample(a, len(a))
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
    var_config["input_node_feature_names"] = ["time", "temperature", "fluence", "hoop_stress", "bc_r", "bc_z"]
    var_config["input_node_feature_dims"] = [1, 1, 1, 1, 1, 1]
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
        ##set directory to load processed pickle files, train/validate/test
        trainset = []
        valset = []
        testset = []
        for dataset_type in ["train", "val", "test"]:
            with open(f'dataset/pickle/{dataset_type}set.meta') as f:
                num_samples = int(f.readline().strip('\n')) 
                for sample_count in range(0,num_samples):
                    with open("dataset/pickle/concrete_shielding-"+dataset_type+"set-"+str(sample_count)+".pk", 'rb') as pickle_file:
                        data_object = pickle.load(pickle_file)
                        if "train"==dataset_type:
                            trainset.append(data_object)
                        elif "val"==dataset_type:
                            valset.append(data_object)
                        elif "test"==dataset_type:
                            testset.append(data_object)

    else:
        raise ValueError("Unknown data format: %d" % args.format)

    info("Data load")
    info(
        "trainset,valset,testset size: %d %d %d"
        % (len(trainset), len(valset), len(testset))
    )

    maximum_input, maximum_output = calculate_maximum_inputs_and_outputs(trainset, config)

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
        create_plots=True,
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
