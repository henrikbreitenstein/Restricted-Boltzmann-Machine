import matplotlib.pyplot as plt
import numpy as np
import torch

from RBMmodules import hamiltonian
import main

models = {
    "Lipkin"     : {
        "name" : "Lipkin",
        "hamiltonian" : hamiltonian.lipkin_local,
        "args" : {
            "eps" : 1,
            "V"   : 0,
            "W"   : 0
        }
    },
    "Ising"      : {},
    "Pairing"    : {},
    "Heisenberg" : {}
}

run_options = {
    "epochs"      : 500,
    "monte_carlo" : {
        "type"   : 5,
        "cycles" : 10_000
    },
    "learning_rate"     : 0.5,
    "adaptive_function" : None
    }
for n_particles in [2, 4, 8, 16]:
    model_options = models["Lipkin"]

    machine_options = {
        "visual_n" : n_particles,
        "hidden_n" : 2*n_particles,
        "precision" : torch.float64,
        "device" : torch.device('cuda')
    }

    result_path = main.run(
        model_options,
        machine_options,
        run_options,
        "Test_noninteractive\\runs"
    )
    
    print(result_path)



