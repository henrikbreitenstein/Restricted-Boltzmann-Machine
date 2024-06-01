import torch
import numpy as np
import gridsearch as gs
import search_analysis as sa
from lipkin_line import lipkin_results_grid
import sys

sys.path.append('../')
sys.path.append('../RBMmodules')
from functools import partial
from RBMmodules import hamiltonian, adaptives

ising_results_grid = {
    "n" : np.arange(4, 15),
    "lr" : np.array([
        0.03812,
        0.03567,
        0.00846,
        0.00795,
        0.00724,
        0.00720,
        0.00608,
        0.00620,
        0.00578,
        0.00589,
        0.00548
    ]),
    "hidden_n" : np.array([
        7,
        9,
        10,
        11,
        9,
        12,
        10,
        14,
        16,
        17,
        18,
    ])
}

ising2d_results_grid = {
    "M" : np.array([2, 3]),
    "lr" : np.array([0.04013208, 0.00770]),
    "hidden_n" : np.array([3, 13])
}

if __name__ == "__main__":
        
    M = 1
    for M in np.arange(2, 4, dtype=int):
        M = int(M)
        N = M

        run_options = {
        "epochs"      : 500,
        "monte_carlo" : {
            "type"   : 2,
            "cycles" : 50_000
        },
        "learning_rate"     : ising_results_grid['lr'][N*M-4],
        "adaptive_function" : adaptives.nop,
        "gamma" : 0
        }



        n_particles = N*M
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

        J = 1
        L = -0.5
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
        start = lipkin_results_grid['lr'][n_particles-4]*0.9
        end = lipkin_results_grid['lr'][n_particles-4]*1.4
        search_x = np.linspace(start, end, resolution)
        
        if 1:
            start = n_particles-2
            end = n_particles+5
            search = "hidden_n"
            search_x = np.arange(start, end, dtype=int)
            resolution = len(search_x)
        repeats = 5

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
            machine_options,
            show=False
        )
