from numpy import save
import pandas as pd
import pickle
import os
from solver import stats_dict

def save_to_dest(to_save : dict, folder : str, name : str):
    with open(folder + '/' + name, 'wb+') as fb:
        pickle.dump(to_save, fb)

def learning_process(
    model_options : dict,
    model_layers : dict,
    machine_options : dict,
    run_options : dict,
    result : stats_dict,
    run_name : str,
    ) -> str:

    file_name = f"{model_options['name']}/"
    for key, value in model_options["args"].items():
        if key != file_name:
            file_name += f"[{key}={value:.1g}]"
    file_name += f"[n={machine_options['visual_n']}]"
    file_name += f"[hn={machine_options['hidden_n']}]"
    file_name += f"[e={run_options['epochs']}]"
    file_name += f"[m={run_options['monte_carlo']['type']}]"
    file_name += f"[c={run_options['monte_carlo']['cycles']}]"


    folder_path = '../Results/' + run_name + r'/' + file_name
    try:
        os.mkdir(folder_path)
    except FileExistsError:
        pass

    save_to_dest(run_options, folder_path, "run_options")
    save_to_dest(result, folder_path, "result")
    save_to_dest(model_options, folder_path, "model_options")
    save_to_dest(model_layers, folder_path, "model_layers")
    save_to_dest(machine_options, folder_path, "machine_options")

    return folder_path

def load_learning_process(path : str):

    loaded = {}

    for file in os.listdir(path):
        with open(f"{path}/{file}", "rb") as fb:
            exec(f"{file} = pickle.load(fb)")
        exec(f"loaded['{file}'] = {file}")

    return loaded



