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

import os
import numpy as np

import random

import torch
from torch_geometric.data import Data
from torch import tensor

from ase.io.cfg import read_cfg

from hydragnn.utils.cfgdataset import CFGDataset

import pickle

class MariiaDataset(CFGDataset):

    def __init__(self, config, dist=False):
        super(MariiaDataset, self).__init__(config, dist)

    def _AbstractRawDataset__load_raw_data(self):
        """Loads the raw files from specified path, performs the transformation to Data objects and normalization of values.
        After that the serialized data is stored to the serialized_dataset directory.
        """

        #serialized_dir = os.environ["SERIALIZED_DATA_PATH"] + "/serialized_dataset"
        #if not os.path.exists(serialized_dir):
        #    os.makedirs(serialized_dir, exist_ok=True)

        for dataset_type, raw_data_path in self.path_dictionary.items():
            pure_Ni_object = self.__transform_ASE_object_to_data_object(raw_data_path + '/' + 'Ni_ground_state.cfg')
            pure_Ni_object.y = self.extract_formation_energy_value(raw_data_path + '/' + "Ni_ground_state_formation_energy.txt")
            self.dataset.append(pure_Ni_object)

            pure_Pt_object = self.__transform_ASE_object_to_data_object(raw_data_path + '/' + 'Pt_ground_state.cfg')
            pure_Pt_object.y = self.extract_formation_energy_value(raw_data_path + '/' + "Pt_ground_state_formation_energy.txt")
            self.dataset.append(pure_Pt_object)

            for _, dirs, _ in os.walk(raw_data_path):
                for dir in dirs:
                    for _, subdirs, _ in os.walk(raw_data_path + '/' + dir):
                        for subdir in subdirs:
                            for _, _, files in os.walk(raw_data_path + '/' + dir + '/' + subdir):
                                for filename in files:
                                    if '.cfg' in filename:
                                        try:
                                            filename_without_extension = filename.rsplit(".", 1)[0]
                                            data_object = self.__transform_ASE_object_to_data_object(
                                                raw_data_path + '/' + dir + '/' + subdir + '/' + filename)
                                            data_object.y = self.extract_formation_energy_value(raw_data_path + '/' + dir + '/' + subdir + '/' + filename_without_extension+"_formation_energy.txt")
                                            self.dataset.append(data_object)
                                        except:
                                            print(raw_data_path + '/' + dir + '/' + filename,
                                                  "could not be converted in torch_geometric.data "
                                                  "object")

                        break

                break

            if self.dist:
                torch.distributed.barrier()

        # scaled features by number of nodes
        self.__scale_features_by_num_nodes()

        self.__normalize_dataset()

    def extract_formation_energy_value(self, filename):
        formation_energy_file = open(filename, 'r')
        Lines = formation_energy_file.readlines()

        # Strips the newline character
        for line in Lines:
            return tensor([float(line.strip())])

    def __transform_ASE_object_to_data_object(self, filepath):
        ase_object = read_cfg(filepath)

        data_object = Data()

        filename_without_extension = filepath.replace('.cfg', '')

        data_object.filename = os.path.basename(filename_without_extension)
        data_object.supercell_size = tensor(ase_object.cell.array).float()
        data_object.pos = tensor(ase_object.arrays["positions"]).float()
        proton_numbers = np.expand_dims(ase_object.arrays["numbers"], axis=1)
        masses = np.expand_dims(ase_object.arrays["masses"], axis=1)
        node_feature_matrix = np.concatenate(
            (proton_numbers, masses), axis=1
        )
        data_object.x = tensor(node_feature_matrix).float()

        formation_energy_file = open(filename_without_extension + '_formation_energy.txt', 'r')
        Lines = formation_energy_file.readlines()

        # Strips the newline character
        for line in Lines:
            data_object.y = tensor([float(line.strip())])

        return data_object



