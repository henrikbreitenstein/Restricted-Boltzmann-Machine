import torch
import numpy as np
import gridsearch as gs
import search_analysis as sa
import sys

sys.path.append('../')
sys.path.append('../RBMmodules')
from RBMmodules import hamiltonian, adaptives

lipkin_results_grid = {
    "n" : np.array([2, 4]),
    "learning_rate" : np.array([0.1, 0.07878, 0.0725]),
    "gamma" : np.array([0.184, 0.2875, 0.542])
}

run_options = {
    "epochs"      : 1000,
    "monte_carlo" : {
        "type"   : 2,
        "cycles" : 100_000
    },
    "learning_rate"     : None,
    "adaptive_function" : adaptives.nop
    }

n_particles = 16
machine_options = {
    "visual_n" : n_particles,
    "hidden_n" : n_particles,
    "precision" : torch.float64,
    "device" : torch.device('cuda')
}
basis = hamiltonian.create_basis(
    n_particles,
    machine_options['precision'],
    machine_options['device']
)

eps = -2 ; V=0; W = -0
model_options = {
    "name" : "Lipkin",
    "hamiltonian" : hamiltonian.lipkin_local,
    "basis" : basis,
    "args" : {
        "eps" : eps,
        "V"   : V,
        "W"   : W
    }
}

H = hamiltonian.lipkin_hamiltonian(
    n_particles,
    eps,
    V,
    W
)

model_options["masking_func"] = hamiltonian.lipkin_amps
model_options["hamiltonian"] = H

true_val = hamiltonian.ground_state(H)

resolution = 1
search_x = np.linspace(0.1, 0.04, resolution)
search_y = np.linspace(0.005, 0.05, resolution)
repeats = 1

if __name__ == "__main__":
    if 1:
        gs.gridsearch(
            repeats,
            resolution,
            search_x,
            search_y,
            run_options,
            model_options,
            machine_options,
            true_val,
            plot = True,
            verbose = True
        )

    if 0:
        sa.plot_mesh(
            search_x,
            search_y,
            run_options,
            model_options,
            machine_options
        )
    
