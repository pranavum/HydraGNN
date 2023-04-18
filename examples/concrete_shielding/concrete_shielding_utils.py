import os
import pandas as pd
import torch
import torch_geometric

from hydragnn.preprocess.utils import get_radius_graph
from torch_geometric.transforms import Distance

def calculate_maximum_inputs_and_outputs(dataset, config):
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

def normalize_input(data, input_scaling_tensor):

    assert data.x.shape[1] == input_scaling_tensor.shape[0]
    data.x = torch.matmul(data.x, torch.diag(1./input_scaling_tensor))

    return data


def normalize_log_scale_input(data, input_scaling_tensor):
    assert data.x.shape[1] == input_scaling_tensor.shape[0]
    data.x[:, [0, 2, 3, 4]] = torch.matmul(data.x[:, [0, 2, 3, 4]], torch.diag(1. / input_scaling_tensor[[0, 2, 3, 4]]))
    data.x[:, [1]] = torch.log(data.x[:, [1]] + 1)

    return data


def normalize_data_sample(data, input_scaling_tensor, output_scaling_tensor):

    # FIXME: this loop works only it the nodal feature has length equal to 1
    assert data.y.shape[0]/data.num_nodes == output_scaling_tensor.shape[0]

    data = normalize_input(data, input_scaling_tensor)

    output_index_count = 0
    # FIXME: this loop works only it the nodal feature has length equal to 1
    for output_index in range(0,output_scaling_tensor.shape[0]):
        data.y[output_index_count:(output_index_count+data.num_nodes)] = data.y[output_index_count:(output_index_count+data.num_nodes)] * 1/output_scaling_tensor[output_index]
        output_index_count = output_index_count + data.num_nodes

    return data

def normalize_data_sample_log_scale_fluence(data, input_scaling_tensor, output_scaling_tensor):

    # FIXME: this loop works only it the nodal feature has length equal to 1
    assert data.y.shape[0]/data.num_nodes == output_scaling_tensor.shape[0]

    data = normalize_log_scale_input(data, input_scaling_tensor)

    output_index_count = 0
    # FIXME: this loop works only it the nodal feature has length equal to 1
    for output_index in range(0,output_scaling_tensor.shape[0]):
        data.y[output_index_count:(output_index_count+data.num_nodes)] = data.y[output_index_count:(output_index_count+data.num_nodes)] * 1/output_scaling_tensor[output_index]
        output_index_count = output_index_count + data.num_nodes

    return data


def read_mesh_coordinates_and_nodal_features_from_csv_file(time_step_index, moose_integration=False):

    if moose_integration:
        file_relative_path = time_step_index
    else:
        #file_relative_path = "examples/concrete_shielding/dataset/concrete_shielding/nodal_info_time/PointData_" + str(time_step_index) + '.csv'
        file_relative_path = "dataset/concrete_shielding/inputs/PointData_" + str(
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
    file_relative_path = 'dataset/concrete_shielding' + '/' + 'workdir.' + str(
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

def generate_graph_input(mesh_file, input_features_numpy_array):

    absolute_path = os.path.abspath(os.getcwd())
    df = pd.read_csv(absolute_path + '/' + mesh_file)
    x = torch.tensor(df['Points:0']).unsqueeze(1).float()
    y = torch.tensor(df['Points:1']).unsqueeze(1).float()
    z = torch.tensor(df['Points:2']).unsqueeze(1).float()
    torch_coordinates = torch.cat([x, y, z], dim=1)
    x = torch.from_numpy(input_features_numpy_array)

    compute_edges = get_radius_graph(
        radius=0.30,
        loop=False,
        max_neighbours=100,
    )
    compute_edges_lengths = Distance(norm=False, cat=True)

    data_object = torch_geometric.data.data.Data(x=x)
    data_object.pos = torch_coordinates
    data_object = compute_edges(data_object)
    data_object = compute_edges_lengths(data_object)

    return data_object
