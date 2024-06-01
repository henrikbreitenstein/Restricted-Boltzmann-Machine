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

heisen_results_grid = {
    "n" : np.arange(4, 13),
    "lr" : np.array([
        0.04180,
        0.03233,
        0.00839,
        0.00738,
        0.00580,
        0.00760,
        0.00512,
        0.00680,
        0.00576,
        0.00386
    ]),
    "hidden_n" : np.array([8, 7, 9, 7, 6, 11, 11, 10, 16, 12])
}

heisen2d_results_grid = {
    "M" : np.array([2, 3]),
    "lr" : np.array([0.04180, 0.00770]),
    "hidden_n" : np.array([2, 10])
}

if __name__ == "__main__":
        
    M = 1
    for i, M in enumerate([2, 3]):#for N in np.arange(12, 14, dtype=int):
        N = M = int(M)

        n_particles = N*M
        run_options = {
        "epochs"      : 500,
        "monte_carlo" : {
            "type"   : 2,
            "cycles" : 100_000
        },
        "learning_rate"     : heisen2d_results_grid['lr'][i],
        "adaptive_function" : adaptives.nop,
        "gamma" : 0
        }



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

        J = -1
        L = -0.5
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
        start = lipkin_results_grid['lr'][n_particles-4]*0.9
        end = lipkin_results_grid['lr'][n_particles-4]*1.4
        search_x = np.linspace(start, end, resolution)
        
        if 1:
            start = n_particles-2
            end = n_particles+5
            search = "hidden_n"
            search_x = np.arange(start, end, dtype=int)
            resolution = len(search_x)
        repeats = 3

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
