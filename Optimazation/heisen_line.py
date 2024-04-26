import torch
import numpy as np
import gridsearch as gs
import search_analysis as sa
import sys

sys.path.append('../')
sys.path.append('../RBMmodules')
from functools import partial
from RBMmodules import hamiltonian, adaptives

heisen_results_line = {
    "n" : np.array([2, 4, 6, 8, 10]),
    "lr" : np.array([]),
    "hidden_n" : np.array([2]),
    "gibbs_k" : np.array([])
}


heisen_results_2d = {
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
    "learning_rate"     : lr,
    "adaptive_function" : adaptives.nop
    }

N = 2
M = 1
n_particles = N*M
machine_options = {
    "visual_n" : n_particles,
    "hidden_n" : 2*n_particles,
    "precision" : torch.float64,
    "device" : torch.device('cuda')
}
basis = hamiltonian.create_basis(
    n_particles,
    machine_options['precision'],
    machine_options['device']
)

J = -0.3
L = 0.2
H = hamiltonian.heisen_hamiltonian(N, M, J, L)

model_options = {
    "name" : "Heisenberg",
    "masking_func" : partial(hamiltonian.amplitudes, basis=basis),
    "basis" : basis,
    "args" : {
        "J" : J,
        "L" : L
    },
    "hamiltonian" : H
}

if M>1:
    model_options['name'] = "Heisenberg2d"

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
