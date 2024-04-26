import torch
import numpy as np
import gridsearch as gs
import search_analysis as sa
import sys

sys.path.append('../')
sys.path.append('../RBMmodules')
from functools import partial
from RBMmodules import hamiltonian, adaptives

lipkin_results_grid = {
    "n" : np.array([2, 4, 6, 8, 10]),
    "lr" : np.array([8.92]),
    "gibbs_k" : np.array([2]),
    "hidden_n" : np.array([6])
}

n_particles = 2
run_options = {
    "epochs"      : 500,
    "monte_carlo" : {
        "type"   : 2,
        "cycles" : 50_000
    },
    "learning_rate"     : 8.92,
    "adaptive_function" : adaptives.nop,
    "gamma" : 0
}

machine_options = {
    "visual_n" : n_particles,
    "hidden_n" : 6,
    "precision" : torch.float64,
    "device" : torch.device('cuda')
}

basis = hamiltonian.create_basis(
    n_particles,
    machine_options['precision'],
    machine_options['device']
)

eps = 1.5 -np.sqrt(3)/2 ; V=-1; W = 0
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

search= "learning_rate"
resolution = 50
search_x = np.linspace(7, 9, resolution)

if 0:
    resolution = 10
    start = 2
    search = "hidden_n"
    search_x = np.arange(start, start+resolution, dtype=int)
repeats = 5

if 1:
    resolution = 10
    start = 1
    search = "gibbs_k"
    search_x = np.arange(start, start+resolution, dtype=int)

if __name__ == "__main__":
    if 1: 
        gs.linesearch(
            repeats,
            resolution,
            search_x,
            search,
            run_options,
            model_options,
            machine_options,
            true_val
        )
    if 1:
        sa.plot_line(
            repeats,
            search_x,
            search,
            run_options,
            model_options,
            machine_options
        )
