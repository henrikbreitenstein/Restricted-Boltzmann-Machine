import torch
import numpy as np
import gridsearch as gs
import search_analysis as sa
import sys

sys.path.append('../')
sys.path.append('../RBMmodules')
from functools import partial
from RBMmodules import hamiltonian, adaptives

heisen_results_grid = {
    "n" : np.array([]),
    "learning_rate" : np.array([]),
    "gamma" : np.array([])
}

run_options = {
    "epochs"      : 1000,
    "monte_carlo" : {
        "type"   : 2,
        "cycles" : 500_000
    },
    "learning_rate"     : None,
    "adaptive_function" : adaptives.nop,
    }

N = 3
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

J = 0.3
L = -0.4
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

true_val = hamiltonian.ground_state(H)


resolution = 1
search_x = np.linspace(0.0, 1, resolution)
search_y = np.linspace(1, 2, resolution)


repeats = 1
if __name__ == "__main__":
    if True:
        gs.gridsearch(
            repeats,
            resolution,
            search_x,
            search_y,
            run_options,
            model_options,
            machine_options,
            true_val,
            plot=True,
            verbose=True
        )

    if not True:
        sa.plot_mesh(
            search_x,
            search_y,
            run_options,
            model_options,
            machine_options
        )
