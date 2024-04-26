import torch
import numpy as np
import gridsearch as gs
import search_analysis as sa
import sys

sys.path.append('../')
sys.path.append('../RBMmodules')
from functools import partial
from RBMmodules import adaptives, hamiltonian

pairing_results_grid = {
    "n" : np.array([2, 4]),
    "learning_rate" : np.array([0.1, 0.07878, 0.0725]),
    "gamma" : np.array([0.184, 0.2875, 0.542])
}

run_options = {
    "epochs"      : 100,
    "monte_carlo" : {
        "type"   : 5,
        "cycles" : 50_000
    },
    "learning_rate"     : None,
    "adaptive_function" : adaptives.nop
    }

n_particles = 2; P = 3
machine_options = {
    "visual_n" : P,
    "hidden_n" : 2*P,
    "precision" : torch.float64,
    "device" : torch.device('cuda')
}

basis = hamiltonian.create_basis(
    P,
    machine_options['precision'],
    machine_options['device']
)

eps = -1; g = -2; 
H = hamiltonian.pairing_hamiltonian(
    basis,
    P,
    n_particles,
    eps,
    g
)

print(H)

model_options = {
    "name" : "Pairing",
    "hamiltonian" : H,
    "masking_func" : partial(hamiltonian.amplitudes, basis=basis),
    "basis" : basis,
    "args" : {
        "eps" : eps,
        "g"   : g,
        "P"   : P,
        "n"   : n_particles
    }
}

true_val = hamiltonian.ground_state(H)

resolution = 1
search_x = np.linspace(0.8, 0.95, resolution)
search_y = np.linspace(2, 0.05, resolution)
repeats = 1
if __name__ == "__main__":
    if not False:
        gs.gridsearch(
            repeats,
            resolution,
            search_x,
            search_y,
            run_options,
            model_options,
            machine_options,
            true_val,
            verbose=True,
            plot=True
        )

    if not True:
        sa.plot_mesh(
            search_x,
            search_y,
            run_options,
            model_options,
            machine_options
        )
    
