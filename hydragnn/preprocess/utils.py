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


import torch_geometric
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data
from torch_geometric.transforms import RadiusGraph, Distance


class RadiusGraphPBC(BaseTransform):
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
    """
    def __init__(self, r, loop=False, max_num_neighbors=32, flow='source_to_target', periodic_boundary_conditions=False):
        self.r = r
        self.loop = loop
        self.max_num_neighbors = max_num_neighbors
        self.flow = flow
        self.periodic_boundary_conditions = periodic_boundary_conditions

    def __call__(self, data):
        data.edge_attr = None
        batch = data.batch if 'batch' in data else None
        if periodic_boundary_conditions:
            assert(hasattr(data, 'unit_cell'), "The data must contain information about the size of the unit cell to apply the periodic boundary conditions")
            assert(hasattr(data, 'atom_types'), "The data must contain information about the atoms at each location of the lattice. Can be a chemical symbol (str) or an atomic number (int).")
            ase_atom_object = ase.Atoms(symbols=data.atom_types, positions=data.pos, cell=data.unit_cell, pbc=True)
            edge_src, edge_dst, edge_shift = ase.neighborlist.neighbor_list("ijS", a=ase_atom_object, cutoff=self.r, self_interaction=self.loop)

            #create masks to remove replications of the same edge
            edge_src_mask = numpy.full(edge_src.shape, True)
            edge_dst_mask = numpy.full(edge_dst.shape, True)
            edge_shift_mask = numpy.full(edge_shift.shape, True)

            # the ASE environment enforces periodic boundary conditions so that the same edge may be listed multiple times
            # we need to remove multiple replicas of the same edge
            # for each source node in the graph, we create a unique set that lists all the nodes connected to it (including nodes that connet through periodic boundary conditions)

            src_dictionary = {}
            for index in range(edge_src.shape[0]):
                if edge_src[index] in src_dictionary:
                   if edge_dst[index] in src_dictionary[edge_src[index]]:
                      edge_src_mask[index] = False
                      edge_dst_mask[index] = False
                      edge_shift_mask[index][0] = False
                      edge_shift_mask[index][1] = False
                      edge_shift_mask[index][2] = False
                   else:
                      src_dictionary[edge_src[index]].add(edge_dst[index])
                else:
                   src_dictionary[edge_src[index]] = set()
                   src_dictionary[edge_src[index]].add(edge_dst[index])

            edge_src = edge_src[edge_src_mask]
            edge_dst = edge_dst[edge_dst_mask]
            edge_shift = edge_shift[edge_shift_mask]
            edge_shift = edge_shift.reshape(-1,3)

            data.edge_index=torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0),
            data.edge_shift=torch.tensor(edge_shift, dtype=default_dtype),
        else:
            data.edge_index = torch_geometric.nn.radius_graph(
                data.pos, self.r, batch, self.loop, self.max_num_neighbors,
                self.flow)
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(r={self.r})'

