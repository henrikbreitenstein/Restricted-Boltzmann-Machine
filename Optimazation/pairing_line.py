import torch
import numpy as np
import gridsearch as gs
import search_analysis as sa
import sys

sys.path.append('../')
sys.path.append('../RBMmodules')
from functools import partial
from lipkin_line import lipkin_results_grid
from RBMmodules import hamiltonian, adaptives
pairing_results_line = {
    "n" : np.arange(4, 15),
    "lr" : np.array([
        0.04253,
        0.03602,
        0.00871,
        0.00796,
        0.00738,
        0.00693,
        0.00633,
        0.00625,
        0.00551,
        0.00557,
        0.00527
    ]),
    "hidden_n" : np.array([9, 10, 9, 11, 13, 13, 15, 15, 17, 18, 17]),
}

if __name__ == "__main__":
        
    for i, P in enumerate(np.arange(4, 15)):
        P = int(P)

        run_options = {
            "epochs"      : 1000,
            "monte_carlo" : {
                "type"   : 2,
                "cycles" : 50_000
            },
            "learning_rate"     : pairing_results_line['lr'][P-4],
            "adaptive_function" : adaptives.nop,
            "gamma" : 0
            }

        n_particles = int(P/2)
        machine_options = {
            "visual_n" : P,
            "hidden_n" : P,
            "precision" : torch.float64,
            "device" : torch.device('cuda')
        }
        basis = hamiltonian.create_basis(
            P,
            machine_options['precision'],
            machine_options['device']
        )

        eps= -0.3; g = -1
        H = hamiltonian.pairing_hamiltonian(P, n_particles, eps, g)

        model_options = {
            "name" : "Pairing",
            "masking_func" : partial(hamiltonian.amplitudes, basis=basis),
            "basis" : basis,
            "args" : {
                "eps" : eps,
                "g" : g
            },
            "hamiltonian" : H
        }

        true_val = hamiltonian.ground_state(H)

        search= "learning_rate"
        resolution = 20
        start = lipkin_results_grid['lr'][P-4]*0.7
        end = lipkin_results_grid['lr'][P-4]*1.3
        search_x = np.linspace(start, end, resolution)

        if 1:
            resolution = 9
            search = "hidden_n"
            search_x = np.arange(P-3, P+6, dtype=int)
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
        if 1:
            sa.plot_line(
                repeats,
                search_x,
                search,
                run_options,
                model_options,
                machine_options,
                show=False
            )
