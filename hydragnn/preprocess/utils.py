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


def check_if_graph_size_constant(train_loader, val_loader, test_loader):
    graph_size_variable = False
    nodes_num_list = []
    for loader in [train_loader, val_loader, test_loader]:
        for data in loader.dataset:
            nodes_num_list.append(data.num_nodes)
            if len(list(set(nodes_num_list))) > 1:
                graph_size_variable = True
                return graph_size_variable
    return graph_size_variable


import ase
import torch
from torch_geometric.transforms import RadiusGraph, Distance


class RadiusGraphPBC(RadiusGraph):
    r"""Creates edges based on node positions :obj:`pos` to all points within a
    given distance.

    Args:
        r (float): The distance.
        loop (bool, optional): If :obj:`True`, the graph will contain
            self-loops. (default: :obj:`False`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            return for each element in :obj:`y`.
            This flag is only needed for CUDA tensors. (default: :obj:`32`)
        flow (string, optional): The flow direction when using in combination
            with message passing (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
        periodic_boundary_conditions (bool, optional): If :obj:`True`,
            periodic boundary conditions will be applied. (default: :obj:`False`)
    """

    def __init__(
        self,
        r,
        loop=False,
        max_num_neighbors=32,
        flow="source_to_target",
        periodic_boundary_conditions=False,
    ):
        super().__init__(r, loop, max_num_neighbors, flow)
        self.periodic_boundary_conditions = periodic_boundary_conditions

    def __call__(self, data):
        data.edge_attr = None
        if self.periodic_boundary_conditions:
            data.pbc = True
            assert (
                "batch" not in data
            ), "periodic boundary conditions not currently supported on batches"
            assert hasattr(
                data, "unit_cell"
            ), "The data must contain information about the size of the unit cell to apply the periodic boundary conditions"
            assert hasattr(
                data, "atom_types"
            ), "The data must contain information about the atoms at each location of the lattice. Can be a chemical symbol (str) or an atomic number (int)."
            ase_atom_object = ase.Atoms(
                symbols=data.atom_types,
                positions=data.pos,
                cell=data.unit_cell,
                pbc=True,
            )
            # ‘i’ : first atom index
            # ‘j’ : second atom index
            # https://wiki.fysik.dtu.dk/ase/ase/neighborlist.html#ase.neighborlist.neighbor_list
            edge_src, edge_dst = ase.neighborlist.neighbor_list(
                "ij", a=ase_atom_object, cutoff=self.r, self_interaction=self.loop
            )
            distance_matrix = ase_atom_object.get_all_distances(mic=True)
            data.edge_index = torch.stack(
                [torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0
            )
            data.edge_shift = torch.tensor(edge_shift)
            data.edge_attr = torch.zeros(
                ase_atom_object.get_global_number_of_atoms(), 1
            )
            for index in range(0, edge_src.shape[0]):
                data.edge_attr[index, 1] = distance_matrix[
                    edge_src[index], edge_dst[index]
                ]
        else:
            data = super().__call__(data)

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(r={self.r})"


class DistancePBC(Distance):
    r"""Saves the Euclidean distance of linked nodes in its edge attributes.

    Args:
        norm (bool, optional): If set to :obj:`False`, the output will not be
            normalized to the interval :math:`[0, 1]`. (default: :obj:`True`)
        max_value (float, optional): If set and :obj:`norm=True`, normalization
            will be performed based on this value instead of the maximum value
            found in the data. (default: :obj:`None`)
        cat (bool, optional): If set to :obj:`False`, all existing edge
            attributes will be replaced. (default: :obj:`True`)
    """

    def __call__(self, data):
        if (not hasattr(data, "pbc")) or (not data.pbc):
            super().__call__(data)
        else:
            # If the connectivity of the data uses boundary conditions, then the edge length has already been added as an attribute
            assert hasattr(data, "attr")

        return data
