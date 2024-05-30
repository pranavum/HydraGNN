import torch

class LJpotential():

    def __init__(self, epsilon, sigma, list_atom_types, bravais_lattice_constants, radius_cutoff):

        self.epsilon = epsilon
        self.sigma = sigma
        self.bravais_lattice_constants = bravais_lattice_constants
        self.radius_cutoff = radius_cutoff

    def compute_potential(self, data):

        assert(data.pos.shape[0] == data.x.shape[0])

        interatomic_potential = torch.zeros([data.pos.shape[0],1])
        interatomic_forces = torch.zeros([data.pos.shape[0], 3])

        for node_id in range(data.pos.shape[0]):

            neighbor_list_indices = torch.where(data.edge_index[0,:] == node_id)[0].tolist()
            neighbor_list = data.edge_index[1,neighbor_list_indices]

            for neighbor_id, edge_id in zip(neighbor_list, neighbor_list_indices):

                neighbor_pos_x = data.pos[neighbor_id, 0]
                neighbor_pos_y = data.pos[neighbor_id, 1]
                neighbor_pos_z = data.pos[neighbor_id, 2]

                pair_distance = data.edge_attr[edge_id].item()
                interatomic_potential[node_id] += 4 * self.epsilon * ((self.sigma/pair_distance)**12 - (self.sigma/pair_distance)**6)

                radial_derivative = 4 * self.epsilon * (-12 * (self.sigma/pair_distance)**12 * 1/pair_distance + 6 * (self.sigma/pair_distance)**6 * 1/pair_distance )

                distance_vector = data.pos[neighbor_id, :] - data.pos[node_id, :]

                # if one of the following conditions is true, the connection of an atom with its neighbor is the result of PBC
                if abs(distance_vector[0]) > self.radius_cutoff:
                    if neighbor_pos_x < data.pos[node_id, 0]:
                        neighbor_pos_x = neighbor_pos_x + data.supercell_size[0, 0]
                    elif neighbor_pos_x > data.pos[node_id, 0]:
                        neighbor_pos_x = neighbor_pos_x - data.supercell_size[0, 0]

                if abs(distance_vector[1]) > self.radius_cutoff:
                    if neighbor_pos_y < data.pos[node_id, 1]:
                        neighbor_pos_y = neighbor_pos_y + data.supercell_size[1, 1]
                    elif neighbor_pos_y > data.pos[node_id, 1]:
                        neighbor_pos_y = neighbor_pos_y - data.supercell_size[1, 1]

                if abs(distance_vector[2]) > self.radius_cutoff:
                    if neighbor_pos_z < data.pos[node_id, 2]:
                        neighbor_pos_z = neighbor_pos_z + data.supercell_size[2, 2]
                    elif neighbor_pos_z > data.pos[node_id, 2]:
                        neighbor_pos_z = neighbor_pos_z - data.supercell_size[2, 2]

                derivative_x = radial_derivative * (neighbor_pos_x - data.pos[node_id, 0]) / pair_distance
                derivative_y = radial_derivative * (neighbor_pos_y - data.pos[node_id, 1]) / pair_distance
                derivative_z = radial_derivative * (neighbor_pos_z - data.pos[node_id, 2]) / pair_distance

                interatomic_forces_contribution_x = - derivative_x
                interatomic_forces_contribution_y = - derivative_y
                interatomic_forces_contribution_z = - derivative_z

                interatomic_forces[node_id, 0] += interatomic_forces_contribution_x
                interatomic_forces[node_id, 1] += interatomic_forces_contribution_y
                interatomic_forces[node_id, 2] += interatomic_forces_contribution_z

        data.x = torch.cat(
            (
                data.x,
                interatomic_potential,
                interatomic_forces
            ),
            1,
        )

        return data





