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
    "n" : np.arange(4, 26),
    "lr" : np.array([0.03272,
                     0.02824,
                     0.00689796,
                     0.00612245,
                     0.00573469,
                     0.00554082,
                     0.00495918,
                     0.00485714,
                     0.00444898,
                     0.00428571,
                     0.00415132,
                     0.00370466,
                     0.00365,
                     0.003,
                     0.0028,
                     0.0025,
                     0.0024,
                     0.0021,
                     0.0020,
                     0.0020,
                     0.0020,
                     0.0020]
                ),
    "hidden_n" : np.array([
        1,
        4,
        7,
        8,
        8,
        9,
        10,
        16,
        15,
        16,
        17,
        16,
        13,
        19,
        17,
        20,
        23,
        23,
        23,
        24,
        25,
        25
    ])
}
if __name__ == "__main__":

    n_particles = 5
    low_array=np.zeros(25)
    for n_particles in np.arange(4, 26):

        run_options = {
            "epochs"      : 500,
            "monte_carlo" : {
                "type"   : 2,
                "cycles" : 100_000
            },
            "learning_rate"     : lipkin_results_grid['lr'][n_particles-4],
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

        eps = 2 ; V=-1; W = 0
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
        resolution = 20
        search_x = np.linspace(0.002, 0.006, resolution)
        
        if n_particles > 16:
            search_x = np.linspace(0.001, 0.0045)
        if n_particles > 20:
            search_x = np.linspace(0.0012, 0.002, resolution)

        if 1:
            resolution = 9
            search = "hidden_n"
            search_x = np.arange(n_particles-3, n_particles+6, dtype=int)
        repeats = 5 

                    
        if 1: 
            lowest =gs.linesearch(
                repeats,
                resolution,
                search_x,
                search,
                run_options,
                model_options,
                machine_options,
                true_val
            )
            low_array[n_particles-1] = lowest
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
    print(low_array)
