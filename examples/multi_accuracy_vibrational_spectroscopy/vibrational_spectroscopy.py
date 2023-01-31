import os, json
import matplotlib.pyplot as plt
import random
import pickle

import logging
import sys
from tqdm import tqdm
from mpi4py import MPI
from itertools import chain
import argparse
import time

from rdkit.Chem.rdmolfiles import MolFromPDBFile

import hydragnn
from hydragnn.utils.distributed import get_device
from hydragnn.utils.print_utils import print_distributed, iterate_tqdm
from hydragnn.utils.time_utils import Timer
#from hydragnn.utils.adiosdataset import AdiosWriter, AdiosDataset
from hydragnn.utils.pickledataset import SimplePickleDataset
from hydragnn.utils.smiles_utils import (
    get_node_attribute_name,
    generate_graphdata_from_smilestr,
)

import numpy as np
#import adios2 as ad2

import torch_geometric.data
import torch
import torch.distributed as dist

import warnings

warnings.filterwarnings("error")

# FIXME: this works fine for now because we train on GDB-9 molecules
# for larger chemical spaces, the following atom representation has to be properly expanded
atom_types = {"C": 0, "F": 1, "H": 2, "N": 3, "O": 4, "S": 5}


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


def nsplit(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def datasets_load(list_dirpaths, sampling=None, seed=None, frac=[0.9, 0.05, 0.05]):
    if seed is not None:
        random.seed(seed)
    data_ID_all = []
    smiles_all = []
    values_all = []
    source_labels_all = []
    source_labels_indices_all = []
    source_label = None
    for dirpath in list_dirpaths:
        for subdir, dirs, files in os.walk(dirpath):
            for dir in dirs:

                if "HQ" in dirpath:
                    source_label = "HQ"

                elif "LQ" in dirpath:
                    source_label = "LQ"

                # check that the spectrum has been calculated. Otherwise, ingore the molecule case
                spectrum_filename_list = [f for f in os.listdir(dirpath + '/' + dir + '/') if
                                 f.endswith('.csv')]

                if len(spectrum_filename_list) == 0:
                    continue

                smilestr_files_list = [f for f in os.listdir(dirpath + '/' + dir + '/') if f.endswith('.dat')]

                if len(smilestr_files_list) == 0:
                    continue

                # collect information about molecular structure and chemical composition
                try:
                    smilestr_file = dirpath + '/' + dir + '/' + smilestr_files_list[0]
                    with open(smilestr_file) as f:
                        first_line = f.readlines()[0]
                        smilestr = first_line.strip()
                # file not found -> exit here
                except IOError:
                    print(f"'{smilestr_file}'" + " not found")
                    sys.exit(1)

                try:
                    assert len(spectrum_filename_list) > 0, "spectrum file missing from directory: " + dirpath + '/' + dir + '/'
                    assert len(spectrum_filename_list) < 2, "too many spectrum files within directory: " + dirpath + '/' + dir + '/'
                    spectrum_filename = dirpath + '/' + dir + '/' + spectrum_filename_list[0]
                    spectrum_energies = list()
                    with open(spectrum_filename, "r") as input_file:
                        count_line = 0
                        for line in input_file:
                            if 500<=count_line<=1000:
                                spectrum_energies.append(float(line.strip().split(',')[1]))
                            elif count_line>505:
                                break
                            count_line = count_line + 1

                    # FIXME: for now, we replicate the spectrum twice, and then we will use it or not for the loss function of each head depending on the label
                    spectrum_energies.extend(spectrum_energies)

                    assert len(spectrum_energies) == (501 * 2)

                # file not found -> exit here
                except IOError:
                    print(f"'{spectrum_filename}'" + " not found")
                    sys.exit(1)

                data_ID_all.append(dir)
                smiles_all.append(smilestr)
                values_all.append(spectrum_energies)
                source_labels_all.append(source_label)
                source_labels_indices_all.append([source_label]*501)

    print("Total:", len(smiles_all))

    a = list(range(len(smiles_all)))
    a = random.sample(a, len(a))
    ix0, ix1, ix2 = np.split(
        a, [int(frac[0] * len(a)), int((frac[0] + frac[1]) * len(a))]
    )

    traindataIDs = []
    valdataIDs = []
    testdataIDs = []
    trainsmiles = []
    valsmiles = []
    testsmiles = []
    trainset = []
    valset = []
    testset = []
    trainlabels = []
    vallabels = []
    testlabels = []
    trainlabels_indices = []
    vallabels_indices = []
    testlabels_indices = []

    for i in ix0:
        traindataIDs.append(data_ID_all[i])
        trainsmiles.append(smiles_all[i])
        trainset.append(values_all[i])
        trainlabels.append(source_labels_all[i])
        trainlabels_indices.append(source_labels_indices_all[i])

    for i in ix1:
        valdataIDs.append(data_ID_all[i])
        valsmiles.append(smiles_all[i])
        valset.append(values_all[i])
        vallabels.append(source_labels_all[i])
        vallabels_indices.append(source_labels_indices_all[i])

    for i in ix2:
        testdataIDs.append(data_ID_all[i])
        testsmiles.append(smiles_all[i])
        testset.append(values_all[i])
        testlabels.append(source_labels_all[i])
        testlabels_indices.append(source_labels_indices_all[i])

    return (
        [traindataIDs, valdataIDs, testdataIDs],
        [trainsmiles, valsmiles, testsmiles],
        [torch.tensor(trainset), torch.tensor(valset), torch.tensor(testset)],
        [trainlabels, vallabels, testlabels],
        [trainlabels_indices, vallabels_indices, testlabels_indices]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampling", type=float, help="sampling ratio", default=None)
    parser.add_argument(
        "--preonly",
        action="store_true",
        help="preprocess only (no training)",
    )
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
    parser.set_defaults(format="pickle")
    args = parser.parse_args()

    graph_feature_names = ["LQ_spectrum", "HQ_spectrum"]
    graph_feature_dim = [501, 501]
    dirpwd = os.path.dirname(os.path.abspath(__file__))
    datadir_low_accuracy = os.path.join(dirpwd, "dataset/QM8-LQ")
    datadir_high_accuracy = os.path.join(dirpwd, "dataset/QM8-HQ")
    ##################################################################################################################
    input_filename = os.path.join(dirpwd, "vibrational_spectroscopy.json")
    ##################################################################################################################
    # Configurable run choices (JSON file that accompanies this example script).
    with open(input_filename, "r") as f:
        config = json.load(f)
    verbosity = config["Verbosity"]["level"]
    var_config = config["NeuralNetwork"]["Variables_of_interest"]
    var_config["output_names"] = [
        graph_feature_names[item]
        for ihead, item in enumerate(var_config["output_index"])
    ]
    var_config["graph_feature_names"] = graph_feature_names
    var_config["graph_feature_dims"] = graph_feature_dim
    (
        var_config["input_node_feature_names"],
        var_config["input_node_feature_dims"],
    ) = get_node_attribute_name(atom_types)
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

    log_name = "vibrational_spectroscopy_fullx"
    hydragnn.utils.setup_log(log_name)
    writer = hydragnn.utils.get_summary_writer(log_name)

    if args.preonly:

        (
            dataIDs_sets,
            smilestr_sets,
            values_sets,
            source_label_sets,
            source_label_indices_sets,
        ) = datasets_load([datadir_low_accuracy, datadir_high_accuracy], sampling=args.sampling, seed=43)

        os.makedirs("dataset/pickle/", exist_ok=True)

        info([len(x) for x in values_sets])
        dataset_lists = [[] for dataset in values_sets]
        for idataset, (dataIDset, smilestrset, valueset, sourceset, source_indices_set) in enumerate(zip(dataIDs_sets, smilestr_sets, values_sets, source_label_sets, source_label_indices_sets)):

            rx = list(nsplit(range(len(smilestrset)), comm_size))[rank]
            info("subset range:", idataset, len(smilestrset), rx.start, rx.stop)
            ## local portion
            _dataIDset = dataIDset[rx.start: rx.stop]
            _smilestrset = smilestrset[rx.start : rx.stop]
            _valueset = valueset[rx.start : rx.stop]
            _sourceset = sourceset[rx.start: rx.stop]
            _sourceset_indices = source_indices_set[rx.start: rx.stop]
            info("local molecule set size:", len(_smilestrset))

            setname = ["trainset", "valset", "testset"]
            if args.format == "pickle":
                if rank == 0:
                    with open(
                        "dataset/pickle/%s.meta" % (setname[idataset]),
                        "w+",
                    ) as f:
                        f.write(str(len(smilestrset)))

            for i, (dataID, smilestr, ytarget, source, source_indices) in iterate_tqdm(
                enumerate(zip(_dataIDset, _smilestrset, _valueset, _sourceset, _sourceset_indices)), verbosity, total=len(_smilestrset)
            ):
                data = generate_graphdata_from_smilestr(
                    smilestr, ytarget, atom_types, var_config
                )
                data.ID = dataID
                data.source = source
                data.source_indices = source_indices
                dataset_lists[idataset].append(data)

                ## (2022/07) This is for testing to compare with Adios
                ## pickle write
                if args.format == "pickle":
                    fname = "dataset/pickle/vibrational_spectrum-%s-%d.pk" % (
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

            adwriter = AdiosWriter("dataset/vibrational_spectrum.bp", comm)
            adwriter.add("trainset", _trainset)
            adwriter.add("valset", _valset)
            adwriter.add("testset", _testset)
            adwriter.save()

        sys.exit(0)

    timer = Timer("load_data")
    timer.start()
    if args.format == "adios":
        trainset = AdiosDataset(
            "dataset/vibrational_spectrum.bp",
            "trainset",
            comm,
            preload=False,
            shmem=True,
        )
        valset = AdiosDataset("dataset/vibrational_spectrum.bp", "valset", comm)
        testset = AdiosDataset("dataset/vibrational_spectrum.bp", "testset", comm)
    elif args.format == "pickle":
        trainset = SimplePickleDataset(
            "dataset/pickle", "vibrational_spectrum", "trainset"
        )
        valset = SimplePickleDataset(
            "dataset/pickle", "vibrational_spectrum", "valset"
        )
        testset = SimplePickleDataset(
            "dataset/pickle", "vibrational_spectrum", "testset"
        )
    else:
        raise NotImplementedError("No supported format: %s" % (args.format))

    info("Adios load")
    info(
        "trainset,valset,testset size: %d %d %d"
        % (len(trainset), len(valset), len(testset))
    )

    (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
        trainset, valset, testset, config["NeuralNetwork"]["Training"]["batch_size"]
    )

    config = hydragnn.utils.update_config(config, train_loader, val_loader, test_loader)

    with open("./logs/" + log_name + "/config.json", "w") as f:
        json.dump(config, f)

    timer.stop()

    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=verbosity,
    )
    model = hydragnn.utils.get_distributed_model(model, verbosity)

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

    if args.mae and rank == 0:
        ##################################################################################################################
        for isub, (loader, setname) in enumerate(
            zip([test_loader], ["test"])
        ):

            num_samples = len(loader.dataset)
            colors = ["blue", "red"]

            for data_index in range(0, num_samples):

                data_sample = loader.dataset[data_index]
                pred = model.module(data_sample.to(get_device()))

                fig, ax = plt.subplots()

                true_value = []
                if data_sample.source == "LQ":
                    true_value = data_sample.y[0:graph_feature_dim[0]].cpu()
                    ax.plot(true_value, color="blue", linestyle='solid')
                elif data_sample.source == "HQ":
                    true_value = data_sample.y[graph_feature_dim[0]:].cpu()
                    ax.plot(true_value, color="red", linestyle='solid')

                head_index = 0
                for ihead in range(0,2):
                    pred_head = pred[head_index].detach().cpu()
                    ax.plot(pred_head.t(), color=colors[ihead], linestyle='dashed')

                plt.ylim([-0.2, max(true_value) + 0.2])
                plt.title("Molecule ID: "+f"{data_sample.ID}")
                plt.tight_layout()
                plt.draw()
                plt.savefig("logs/"+setname+f"_sample_{data_sample.ID}.png")
                plt.close(fig)

    if args.format == "adios":
        trainset.unlink()

    sys.exit(0)
