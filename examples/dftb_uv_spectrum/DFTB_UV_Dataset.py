import os
import sys
import torch
from torch import tensor
from torch_geometric.data import Data
from torch_geometric.transforms import Spherical
from hydragnn.utils.abstractrawdataset import AbstractRawDataset
from hydragnn.utils.smiles_utils import (
    get_node_attribute_name,
    generate_graphdata_from_rdkit_molecule,
)
from hydragnn.utils.atomicdescriptors import atomicdescriptors

from ase import io

from rdkit.Chem.rdmolfiles import MolFromPDBFile

# FIXME: this works fine for now because we train on GDB-9 molecules
# for larger chemical spaces, the following atom representation has to be properly expanded
dftb_node_types = {"C": 0, "F": 1, "H": 2, "N": 3, "O": 4, "S": 5}


class DFTB_UV_Dataset:
    def __init__(self, config, dist=False, sampling=None):
        # self.serial_data_name_list = []
        self.graph_feature_name = (
            config["Dataset"]["graph_features"]["name"]
            if config["Dataset"]["graph_features"]["name"] is not None
            else None
        )

        # atomic descriptors
        atomicdescriptor = atomicdescriptors(
            "./embedding_onehot.json",
            overwritten=True,
            element_types=list(dftb_node_types.keys()),
            one_hot=True,
        )
        self.valence_electrons = atomicdescriptor.get_valence_electrons()
        self.electron_affinity = atomicdescriptor.get_electron_affinity()

        self.graph_feature_dim = config["Dataset"]["graph_features"]["dim"]
        self.raw_dataset_name = config["Dataset"]["name"]
        self.data_format = config["Dataset"]["format"]
        self.path_dictionary = config["Dataset"]["path"]

        assert len(self.graph_feature_name) == len(self.graph_feature_dim)

        self.sampling = sampling
        self.dist = dist
        if self.dist:
            assert torch.distributed.is_initialized()
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()

        self._AbstractRawDataset__load_raw_data()

        ## SerializedDataLoader
        self.verbosity = config["Verbosity"]["level"]
        self.variables = config["NeuralNetwork"]["Variables_of_interest"]
        self.variables_type = config["NeuralNetwork"]["Variables_of_interest"]["type"]
        self.output_index = config["NeuralNetwork"]["Variables_of_interest"][
            "output_index"
        ]

    def _AbstractRawDataset__load_raw_data(self):
        self.dataset = []
        for dataset_type, raw_data_path in self.path_dictionary.items():
            for subdir, dirs, files in os.walk(raw_data_path):
                for dir in dirs:
                    data_object = self.transform_input_to_data_object_base(raw_data_path, dir)
                    if data_object is not None:
                        self.dataset.append(data_object)

            if self.dist:
                torch.distributed.barrier()

    def transform_input_to_data_object_base(self, raw_data_path, dir):
        data_object = self.__transform_DFTB_UV_input_to_data_object_base(raw_data_path, dir)

        return data_object

    def __transform_DFTB_UV_input_to_data_object_base(self, raw_data_path, dir):
        """Transforms lines of strings read from the raw data LSMS file to Data object and returns it.

        Parameters
        ----------
        lines:
          content of data file with all the graph information
        Returns
        ----------
        Data
            Data object representing structure of a graph sample.
        """

        data_object = None

        # collect information about molecular structure and chemical composition
        try:
            pdb_filename = raw_data_path + '/' + dir + '/' + 'smiles.pdb'
            mol = MolFromPDBFile(pdb_filename, sanitize=False, proximityBonding=True,
                                 removeHs=True)  # , sanitize=False , removeHs=False)
        # file not found -> exit here
        except IOError:
            print(f"'{pdb_filename}'" + " not found")
            sys.exit(1)

        try:
            spectrum_filename = raw_data_path + '/' + dir + '/' + 'EXC-smooth.DAT'
            spectrum_energies = list()
            with open(spectrum_filename, "r") as input_file:
                count_line = 0
                for line in input_file:
                    spectrum_energies.append(float(line.strip().split()[1]))

            valence_electrons_list = []
            for atom in mol.GetAtoms():
                valence_electrons_list.append(
                    self.valence_electrons[dftb_node_types[atom.GetSymbol()]].item()
                )
            electron_affinity_list = []
            for atom in mol.GetAtoms():
                electron_affinity_list.append(
                    self.electron_affinity[dftb_node_types[atom.GetSymbol()]].item()
                )

            atomic_descriptors_list = [valence_electrons_list, electron_affinity_list]
            num_manually_constructed_atomic_descriptors = len({len(i) for i in atomic_descriptors_list})

            # The list is empty if there are no atomic descirptors manually constructed, otherwise it shoulb have length=1
            assert num_manually_constructed_atomic_descriptors <= 1, "manually constructed lists of atomic descriptors are not consistent in length"

            if num_manually_constructed_atomic_descriptors == 1:
                atomicdescriptors_torch_tensor = torch.cat([torch.tensor([descriptor]) for descriptor in atomic_descriptors_list],
                               dim=0).t().contiguous()

            data_object = generate_graphdata_from_rdkit_molecule(mol, torch.tensor(spectrum_energies), dftb_node_types, atomicdescriptors_torch_tensor)
            atoms = io.read(raw_data_path + '/' + dir + '/' + 'geo_end.xyz')
            data_object.pos = torch.from_numpy(atoms.positions)
            spherical_transform = Spherical(norm=False)
            data_object = spherical_transform(data_object)
            data_object.pos = data_object.pos.to(torch.float32)
            data_object.x = data_object.x.to(torch.float32)
            data_object.edge_attr = data_object.edge_attr.to(torch.float32)
            data_object.ID = dir.replace('mol_', '')

        except:
            print(f"Graph sample not created for {dir}")

        """
        # file not found -> exit here
        except IOError:
            print(f"'{spectrum_filename}'" + " not found")
            sys.exit(1)

        data_object = generate_graphdata_from_rdkit_molecule(mol, spectrum_energies, dftb_node_types)
        """

        return data_object

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        return self.dataset[idx]

    def apply(self, func):
        for data in self.dataset:
            func(data)

    def map(self, func):
        for data in self.dataset:
            yield func(data)

    def __len__(self):
        return self.len()

    def __getitem__(self, idx):
        return self.get(idx)

    def __iter__(self):
        for idx in range(self.len()):
            yield self.get(idx)

