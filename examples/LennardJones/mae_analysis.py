from pytorch_interatomic_potentials.configurational_data import generate_data
from train import train_model
from train_vlad_total_energy import train_model as train_model_energy
from inference import predict_test
from inference_derivative_energy import predict_derivative_test
import json
import matplotlib.pyplot as plt
import numpy as np
import os

def create_and_train(config_file, data_size, num_layers, num_channels_per_layer, epochs, batch_size, alpha_values):
    if data_size:
        generate_data(data_size)
        train_model(["--preonly"]) if "energy" in config_file else train_model(["--preonly"])
    dirpwd = os.path.dirname(os.path.abspath(__file__))
    input_filename = os.path.join(dirpwd, config_file)
    with open(input_filename, "r+") as f:
        config = json.load(f)
        config["NeuralNetwork"]["Architecture"]["num_conv_layers"] = num_layers
        config["NeuralNetwork"]["Architecture"]["hidden_dim"] = num_channels_per_layer
        config["NeuralNetwork"]["Training"]["num_epoch"] = epochs
        config["NeuralNetwork"]["Training"]["batch_size"] = batch_size
        config["NeuralNetwork"]["Architecture"]["alpha_values"] = alpha_values
        f.seek(0)
        json.dump(config, f, indent=3)
    train_model_energy(["--pickle"]) if "energy" in config_file else train_model(["--pickle"])
    predict_derivative_test()
    return predict_test()

def increment_data_size(
        config_file,
        size_range,
        num_layers,
        num_channels_per_layer,
        epochs,
        batch_size,
        alpha_values
        ):
    outputs = {}
    for data_size in size_range:
        outputs[data_size] = create_and_train(
            config_file,
            data_size=data_size,
            num_layers=num_layers,
            num_channels_per_layer=num_channels_per_layer,
            epochs=epochs,
            batch_size=batch_size,
            alpha_values=alpha_values
            )
    return outputs

def increment_num_layers(
        config_file,
        data_size,
        layers_range,
        num_channels_per_layer,
        epochs,
        batch_size,
        alpha_values
        ):
    outputs = {}
    for num_layers in layers_range:
        outputs[num_layers] = create_and_train(
            config_file,
            data_size=data_size,
            num_layers=num_layers,
            num_channels_per_layer=num_channels_per_layer,
            epochs=epochs,
            batch_size=batch_size,
            alpha_values=alpha_values
            )
    return outputs

def increment_num_channels_per_layer(
        config_file,
        data_size,
        num_layers,
        channels_range,
        epochs,
        batch_size,
        alpha_values
        ):
    outputs = {}
    for num_channels_per_layer in channels_range:
        outputs[num_channels_per_layer] = create_and_train(
            config_file,
            data_size=data_size,
            num_layers=num_layers,
            num_channels_per_layer=num_channels_per_layer,
            epochs=epochs,
            batch_size=batch_size,
            alpha_values=alpha_values
            )
    return outputs

def increment_architecture(
        config_file,
        data_size,
        layers_range,
        channel_scaling,
        epochs,
        batch_size,
        alpha_values
        ):
    outputs = {}
    for num_layers in layers_range:
        outputs[num_layers] = create_and_train(
            config_file,
            data_size=data_size,
            num_layers=num_layers,
            num_channels_per_layer=num_layers*channel_scaling,
            epochs=epochs,
            batch_size=batch_size,
            alpha_values=alpha_values
            )
    return outputs

def increment_epochs(
        config_file,
        data_size,
        num_layers,
        num_channels_per_layer,
        epoch_range,
        batch_size,
        alpha_values
        ):
    outputs = {}
    for epoch in epoch_range:
        outputs[epoch] = create_and_train(
            config_file,
            data_size=data_size,
            num_layers=num_layers,
            num_channels_per_layer=num_channels_per_layer,
            epochs=epoch,
            batch_size=batch_size,
            alpha_values=alpha_values
            )
    return outputs

def repeat_conditions(
        config_file,
        num_repeats,
        data_size,
        num_layers,
        num_channels_per_layer,
        epochs,
        batch_size,
        alpha_values
        ):
    outputs = {}
    base_output_dir = "repeated_results"
    if not os.path.exists(base_output_dir): os.makedirs(base_output_dir)
    for model_index in range(num_repeats):
        run_dir = os.path.join(base_output_dir, f"run{model_index+1}")
        os.makedirs(run_dir)
        outputs[model_index] = create_and_train(
            config_file,
            data_size=data_size,
            num_layers=num_layers,
            num_channels_per_layer=num_channels_per_layer,
            epochs=epochs,
            batch_size=batch_size,
            alpha_values=alpha_values
        )
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
    #predict_derivative_test()
    #generate_data(300_000)
    #train_model(["--preonly"])
    config_file = "LJ_vlad_total_energy.json"
    data_size = 10_000
    num_layers = 5
    num_channels_per_layer = 139
    epochs = 1
    batch_size = 64
    alpha_values = [["constant", 0.0], ["constant", 0.0]]
    output = create_and_train(
        config_file=config_file,
        data_size=None,
        num_layers=num_layers,
        num_channels_per_layer=num_channels_per_layer,
        epochs=epochs,
        batch_size=batch_size,
        alpha_values=alpha_values
    )
    with open("output.txt", "w") as file:
        print(f"config_file={config_file},\ndata_size={data_size},\nnum_layers={num_layers},\nnum_channels_per_layer={num_channels_per_layer},\nepochs={epochs},\nalpha_values={alpha_values}")
        print(f"config_file={config_file},\ndata_size={data_size},\nnum_layers={num_layers},\nnum_channels_per_layer={num_channels_per_layer},\nepochs={epochs},\nalpha_values={alpha_values}", file=file)
        print("\n\n\n")
        print("\n\n\n", file=file)
        print(output)
        print(output, file=file)
        file.close()
