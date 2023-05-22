import os, json
import matplotlib.pyplot as plt
import logging
import sys
from mpi4py import MPI
import argparse


import hydragnn
from hydragnn.utils.time_utils import Timer
from hydragnn.utils.config_utils import get_log_name_config
from hydragnn.utils.model import print_model
from DFTB_UV_Dataset_AE_compressed import DFTB_UV_Dataset_AE_compressed
from hydragnn.utils.serializeddataset import SerializedWriter, SerializedDataset
from hydragnn.preprocess.load_data import split_dataset
from hydragnn.utils.print_utils import log

try:
    from hydragnn.utils.adiosdataset import AdiosWriter, AdiosDataset
except ImportError:
    pass

import numpy as np

import torch
import torch.distributed as dist


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


if __name__ == "__main__":
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
    parser.add_argument("--inputfile", help="input file", type=str, default="dftb_smooth_uv_spectrum.json")
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

    dirpwd = os.path.dirname(os.path.abspath(__file__))
    input_filename = os.path.join(dirpwd, args.inputfile)
    with open(input_filename, "r") as f:
        config = json.load(f)

    datasetname = config["Dataset"]["name"]
    graph_feature_names = config["Dataset"]["graph_features"]["name"]
    graph_feature_dim = config["Dataset"]["graph_features"]["dim"]

    hydragnn.utils.setup_log(datasetname)

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

    for dataset_type, raw_data_path in config["Dataset"]["path"].items():
        config["Dataset"]["path"][dataset_type] = os.path.join(dirpwd, raw_data_path)

    if not args.loadexistingsplit and rank == 0:
        ## Only rank=0 is enough for pre-processing
        total = DFTB_UV_Dataset_AE_compressed(config)

        trainset, valset, testset = split_dataset(
            dataset=total,
            perc_train=config["NeuralNetwork"]["Training"]["perc_train"],
            stratify_splitting=False,
        )
        print(len(total), len(trainset), len(valset), len(testset))

        if args.format == "adios":
            fname = os.path.join(
                os.path.dirname(__file__), "./dataset/%s.bp" % datasetname
            )
            adwriter = AdiosWriter(fname, MPI.COMM_SELF)
            adwriter.add("trainset", trainset)
            adwriter.add("valset", valset)
            adwriter.add("testset", testset)
            adwriter.add_global("minmax_node_feature", total.minmax_node_feature)
            adwriter.add_global("minmax_graph_feature", total.minmax_graph_feature)
            adwriter.save()
        elif args.format == "pickle":
            basedir = os.path.join(
                os.path.dirname(__file__), "dataset", "serialized_dataset"
            )
            SerializedWriter(
                trainset,
                basedir,
                datasetname,
                "trainset",
                minmax_node_feature=None,
                minmax_graph_feature=None,
            )
            SerializedWriter(
                valset,
                basedir,
                datasetname,
                "valset",
            )
            SerializedWriter(
                testset,
                basedir,
                datasetname,
                "testset",
            )
    comm.Barrier()
    if args.preonly:
        sys.exit(0)

    timer = Timer("load_data")
    timer.start()
    if args.format == "adios":
        info("Adios load")
        opt = {
            "preload": True,
            "shmem": False,
        }
        fname = os.path.join(os.path.dirname(__file__), "./dataset/%s.bp" % datasetname)
        trainset = AdiosDataset(fname, "trainset", comm, **opt)
        valset = AdiosDataset(fname, "valset", comm, **opt)
        testset = AdiosDataset(fname, "testset", comm, **opt)
    elif args.format == "pickle":
        info("Pickle load")
        basedir = os.path.join(
            os.path.dirname(__file__), "dataset", "serialized_dataset"
        )
        trainset = SerializedDataset(basedir, datasetname, "trainset")
        valset = SerializedDataset(basedir, datasetname, "valset")
        testset = SerializedDataset(basedir, datasetname, "testset")
    else:
        raise ValueError("Unknown data format: %d" % args.format)
    ## Set minmax
    config["NeuralNetwork"]["Variables_of_interest"][
        "minmax_node_feature"
    ] = trainset.minmax_node_feature
    config["NeuralNetwork"]["Variables_of_interest"][
        "minmax_graph_feature"
    ] = trainset.minmax_graph_feature

    info(
        "trainset,valset,testset size: %d %d %d"
        % (len(trainset), len(valset), len(testset))
    )

    (train_loader, val_loader, test_loader,) = hydragnn.preprocess.create_dataloaders(
        trainset, valset, testset, config["NeuralNetwork"]["Training"]["batch_size"]
    )
    timer.stop()

    config = hydragnn.utils.update_config(config, train_loader, val_loader, test_loader)
    var_config = config["NeuralNetwork"]["Variables_of_interest"]
    config["NeuralNetwork"]["Variables_of_interest"].pop("minmax_node_feature", None)
    config["NeuralNetwork"]["Variables_of_interest"].pop("minmax_graph_feature", None)

    verbosity = config["Verbosity"]["level"]
    model = hydragnn.models.create_model_config(
        config=config["NeuralNetwork"],
        verbosity=verbosity,
    )
    if rank == 0:
        print_model(model)
    comm.Barrier()

    model = hydragnn.utils.get_distributed_model(model, verbosity)

    learning_rate = config["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )

    log_name = get_log_name_config(config)
    writer = hydragnn.utils.get_summary_writer(log_name)

    if dist.is_initialized():
        dist.barrier()

    hydragnn.utils.save_config(config, log_name)

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
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        for isub, (loader, setname) in enumerate(
            zip([train_loader, val_loader, test_loader], ["train", "val", "test"])
        ):
            error, rmse_task, true_values, predicted_values = hydragnn.train.test(
                loader, model, verbosity
            )
            ihead = 0
            head_true = np.asarray(true_values[ihead].cpu()).squeeze()
            head_pred = np.asarray(predicted_values[ihead].cpu()).squeeze()
            ifeat = var_config["output_index"][ihead]
            outtype = var_config["type"][ihead]
            varname = graph_feature_names[ifeat]

            ax = axs[isub]

            num_test_samples = len(test_loader.dataset)
            mol_ID = [item.ID for item in test_loader.dataset]
            error_mae = 0.0
            error_mse = 0.0

            for sample_id in range(0,num_test_samples):
                lower = 0.0
                upper = 750.0
                length = 37500
                bins = [lower + x * (upper - lower) / length for x in range(length)]
                error_mae += np.sum(np.abs(head_pred[(sample_id*graph_feature_dim[0]):(sample_id+1)*graph_feature_dim[0]] - head_true[(sample_id*graph_feature_dim[0]):(sample_id+1)*graph_feature_dim[0]]))
                error_mse += np.sum((head_pred[(sample_id*graph_feature_dim[0]):(sample_id+1)*graph_feature_dim[0]] - head_true[(sample_id*graph_feature_dim[0]):(sample_id+1)*graph_feature_dim[0]]) ** 2)

                fig, ax = plt.subplots()
                true_sample = head_true[(sample_id * graph_feature_dim[0]):(sample_id + 1) * graph_feature_dim[0]]
                pred_sample = head_pred[(sample_id * graph_feature_dim[0]):(sample_id + 1) * graph_feature_dim[0]]
                ax.plot(bins, true_sample, label="TD-DFTB+")
                ax.plot(bins, pred_sample, label="HydraGNN")
                plt.title("molecule ID: "+mol_ID[sample_id])
                plt.legend()
                plt.draw()
                plt.tight_layout()
                plt.ylim([-0.2, max(true_sample)+0.2])
                plt.savefig(f"logs/sample_{sample_id}.png")
                plt.close(fig)

            error_mae /= num_test_samples
            error_mse /= num_test_samples
            error_rmse = np.sqrt(error_mse)

            print(varname, ": ev/cm, mae=", error_mae, ", rmse= ", error_rmse)

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
            ax.set_title(setname + "; " + varname + " (eV/cm)", fontsize=16)
            ax.text(
                minv + 0.1 * (maxv - minv),
                maxv - 0.1 * (maxv - minv),
                "MAE: {:.2f}".format(error_mae),
            )
        if rank == 0:
            fig.savefig("./logs/" + log_name + "/" + varname + "_all.png")
        plt.close()

    if args.format == "adios":
        trainset.unlink()

    sys.exit(0)
