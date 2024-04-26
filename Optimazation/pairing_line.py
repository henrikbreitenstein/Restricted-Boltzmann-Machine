import torch
import numpy as np
import gridsearch as gs
import search_analysis as sa
import sys

sys.path.append('../')
sys.path.append('../RBMmodules')
from functools import partial
from RBMmodules import hamiltonian, adaptives
pairing_results_line = {
    "n" : np.array([2, 4, 6, 8, 10]),
    "lr" : np.array([]),
    "hidden_n" : np.array([2]),
    "gibbs_k" : np.array([])
}

n_particles = 4
run_options = {
    "epochs"      : 1000,
    "monte_carlo" : {
        "type"   : 2,
        "cycles" : 50_000
    },
    "learning_rate"     : None,
    "adaptive_function" : adaptives.nop
    }

P = 4; n_particles = 2
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

eps= 1; g = 0.5
H = hamiltonian.pairing_hamiltonian(basis, P, n_particles, eps, g)

model_options = {
    "name" : "Pairing",
    "masking_func" : partial(hamiltonian.amplitudes, basis),
    "basis" : basis,
    "args" : {
        "eps" : eps,
        "g" : g
    },
    "hamiltonian" : H
}

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
