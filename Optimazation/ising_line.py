import torch
import numpy as np
import gridsearch as gs
import search_analysis as sa
import sys

sys.path.append('../')
sys.path.append('../RBMmodules')
from functools import partial
from RBMmodules import hamiltonian, adaptives

ising_results_grid = {
    "n" : np.array([2, 4, 6, 8, 10]),
    "lr" : np.array([4.18, 4.1, 4.9, 5.5, 5.75]),
    "gibbs_k" : np.array([7, 1, 8, 5, 6]),
    "hidden_n" : np.array([7, 7, 10, 14, 8])
}

ising_results_2d = {
    "n" : np.array([2, 4, 6, 8, 10]),
    "lr" : np.array([4.18, 4.1, 4.9, 5.5, 5.75]),
    "gibbs_k" : np.array([7, 1, 8, 5, 6]),
    "hidden_n" : np.array([7, 7, 10, 14, 8])
}


run_options = {
    "epochs"      : 500,
    "monte_carlo" : {
        "type"   : 3,
        "cycles" : 50_000
    },
    "learning_rate"     : 2.9,
    "adaptive_function" : adaptives.nop,
    "gamma" : 0
    }

N = 10
M = 1
n_particles = N*M
machine_options = {
    "visual_n" : n_particles,
    "hidden_n" : 14,
    "precision" : torch.float64,
    "device" : torch.device('cuda')
}
basis = hamiltonian.create_basis(
    n_particles,
    machine_options['precision'],
    machine_options['device']
)

J = 0.1
L = -0.2
H = hamiltonian.ising_hamiltonian(N, M, J, L)

model_options = {
    "name" : "Ising",
    "masking_func" : partial(hamiltonian.amplitudes, basis=basis),
    "basis" : basis,
    "args" : {
        "J" : J,
        "L" : L
    },
    "hamiltonian" : H
}

if M>1:
    model_options['name'] = "Ising2d"

true_val = hamiltonian.ground_state(H)
search= "learning_rate"
resolution = 50
search_x = np.linspace(2, 6, resolution)

if 1:
    resolution = 15
    start = 5
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

    sa.plot_line(
        repeats,
        search_x,
        search,
        run_options,
        model_options,
        machine_options
    )
