from pytorch_interatomic_potentials.configurational_data import generate_data
from train import train_model
from train_vlad_total_energy import train_model as train_model_energy
from inference import predict_test
import json
import matplotlib.pyplot as plt
import numpy as np

def create_and_train(data_size, num_layers, num_channels_per_layer, epochs):
    if data_size:
        generate_data(data_size)
    with open("LJ_multitask.json", "r+") as f:
        config = json.load(f)
        config["NeuralNetwork"]["Architecture"]["num_conv_layers"] = num_layers
        config["NeuralNetwork"]["Architecture"]["hidden_dim"] = num_channels_per_layer
        config["NeuralNetwork"]["Training"]["num_epoch"] = epochs
        f.seek(0)
        json.dump(config, f, indent=3)
    print("preprocessing")
    train_model(["--preonly"])
    print("training")
    train_model(["--pickle"])
    print("predicting")
    return predict_test()

def increment_data_size(size_range, num_layers, num_channels_per_layer, epochs):
    outputs = {}
    for data_size in size_range:
        outputs[data_size] = create_and_train(data_size=data_size, num_layers=num_layers, num_channels_per_layer=num_channels_per_layer, epochs=epochs)
    return outputs

def increment_num_layers(data_size, layers_range, num_channels_per_layer, epochs):
    outputs = {}
    for num_layers in layers_range:
        outputs[num_layers] = create_and_train(data_size=data_size, num_layers=num_layers, num_channels_per_layer=num_channels_per_layer, epochs=epochs)
    return outputs

def increment_num_channels_per_layer(data_size, num_layers, channels_range, epochs):
    outputs = {}
    for num_channels_per_layer in channels_range:
        outputs[num_channels_per_layer] = create_and_train(data_size=data_size, num_layers=num_layers, num_channels_per_layer=num_channels_per_layer, epochs=epochs)
    return outputs

def increment_architecture(data_size, layers_range, channel_scaling, epochs):
    outputs = {}
    for num_layers in layers_range:
        outputs[num_layers] = create_and_train(data_size=data_size, num_layers=num_layers, num_channels_per_layer=num_layers*channel_scaling, epochs=epochs)
    return outputs

def increment_epochs(data_size, num_layers, num_channels_per_layer, epoch_range):
    outputs = {}
    for epoch in epoch_range:
        outputs[epoch] = create_and_train(data_size=data_size, num_layers=num_layers, num_channels_per_layer=num_channels_per_layer, epochs=epoch)
    return outputs

def plot_outputs(outputs, x_name):
    x_values = []
    energy = []
    forces = []
    for x_value, output in outputs.items():
        energy_mae = output["total_energy"]
        forces_mae = output["atomic_forces"]
        x_values.append(x_value)
        energy.append(energy_mae)
        forces.append(forces_mae)
    energy, forces = np.array(energy), np.array(forces)
    
    plt.clf()
    plt.xlabel(x_name)
    plt.ylabel("MAE")
    plt.plot(x_values, energy)
    plt.title("Energy MAE over " + x_name)
    plt.savefig("energy_over_" + x_name + ".png")

    plt.clf()
    plt.xlabel(x_name)
    plt.ylabel("MAE")
    plt.plot(x_values, forces)
    plt.title("Forces MAE over " + x_name)
    plt.savefig("forces_over_" + x_name + ".png")

if __name__ == "__main__":
    #generate_data(10_000)
    #outputs = increment_num_layers(5, range(1, 5), 2)
    #outputs = increment_architecture(None, range(1, 11), 10)
    outputs = increment_epochs(None, 4, 20, range(10, 60, 10))
    plot_outputs(outputs, "Epochs")
    #output = create_and_train(None, 4, 20, 50)
    with open("outputs.txt", "w") as file:
        print(outputs)
        print(outputs, file=file)
        file.close()
