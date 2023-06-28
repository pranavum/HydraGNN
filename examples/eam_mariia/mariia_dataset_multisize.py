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

class MariiaMultisizeDataset(CFGDataset):

    def __init__(self, config, dist=False):
        super(MariiaMultisizeDataset, self).__init__(config, dist)

    def _AbstractRawDataset__load_raw_data(self):
        """Loads the raw files from specified path, performs the transformation to Data objects and normalization of values.
        After that the serialized data is stored to the serialized_dataset directory.
        """

        for dataset_type, raw_data_path in self.path_dictionary.items():
            for atom_system_dir in os.listdir(raw_data_path):
                atom_system_dir_path = raw_data_path + '/' + atom_system_dir
                pure_Ni_object = self.__transform_ASE_object_to_data_object(atom_system_dir_path + '/' + 'Ni_ground_state.cfg')
                self.dataset.append(pure_Ni_object)

                pure_Pt_object = self.__transform_ASE_object_to_data_object(atom_system_dir_path + '/' + 'Pt_ground_state.cfg')
                self.dataset.append(pure_Pt_object)

                for _, dirs, _ in os.walk(atom_system_dir_path):
                    for dir in dirs:
                        for _, subdirs, _ in os.walk(atom_system_dir_path + '/' + dir):
                            for subdir in subdirs:
                                for _, _, files in os.walk(atom_system_dir_path + '/' + dir + '/' + subdir):
                                    for filename in files:
                                        if '.cfg' in filename:
                                            try:
                                                filename_without_extension = filename.rsplit(".", 1)[0]
                                                data_object = self.__transform_ASE_object_to_data_object(
                                                    atom_system_dir_path + '/' + dir + '/' + subdir + '/' + filename)
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
        self._AbstractRawDataset__scale_features_by_num_nodes()

        self._AbstractRawDataset__normalize_dataset()

    def __transform_ASE_object_to_data_object(self, filepath):
        ase_object = read_cfg(filepath)

        data_object = Data()

        filename_without_extension = filepath.replace('.cfg', '')

        data_object.filename = os.path.basename(filename_without_extension)
        data_object.supercell_size = tensor(ase_object.cell.array).float()
        data_object.pos = tensor(ase_object.arrays["positions"]).float()
        proton_numbers = np.expand_dims(ase_object.arrays["numbers"], axis=1)
        c_peratom = np.expand_dims(ase_object.arrays["c_peratom"], axis=1)
        masses = np.expand_dims(ase_object.arrays["masses"], axis=1)
        fx = np.expand_dims(ase_object.arrays["fx"], axis=1)
        fy = np.expand_dims(ase_object.arrays["fy"], axis=1)
        fz = np.expand_dims(ase_object.arrays["fz"], axis=1)
        node_feature_matrix = np.concatenate(
            (proton_numbers, c_peratom, masses, fx, fy, fz), axis=1
        )
        data_object.x = tensor(node_feature_matrix).float()

        formation_energy_file = open(filename_without_extension + '_formation_energy.txt', 'r')
        Lines = formation_energy_file.readlines()

        # Strips the newline character
        for line in Lines:
            num_atoms = data_object.pos.shape[0]
            data_object.y = tensor([float(line.strip())])/num_atoms

        return data_object



