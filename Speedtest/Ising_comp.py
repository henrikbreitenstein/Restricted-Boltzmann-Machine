import torch
import numpy as np
import sys
import compare_time

sys.path.append('../')
sys.path.append('../RBMmodules')
sys.path.append('../Optimazation')
from functools import partial
from RBMmodules import hamiltonian, adaptives
from Optimazation import ising_line

params = ising_line.ising_results_grid
N = 2; M = 1; n_particles = N*M

if M>1:
    params = ising_line.ising2d_results_grid


n_particles = 2
run_options = {
    "epochs"      : 500,
    "monte_carlo" : {
        "type"   : 2,
        "cycles" : 100_000
    },
    "learning_rate"     : None,
    "adaptive_function" : adaptives.nop,
    "gamma" : 0
    }

machine_options = {
    "visual_n" : None,
    "hidden_n" : None,
    "precision" : torch.float64,
    "device" : torch.device('cuda')
}


J = 0.3; L=-0.4
model_options = {
    "name" : "Ising",
    "H_func" : hamiltonian.ising_hamiltonian,
    "args" : {
        "J" : 0.3,
        "L"   : -0.4,
    }
}

if M>1:
    model_options['name'] = "Ising2d"

search = "particles"
sarray = np.arange(4, 14)
points = len(sarray)

if 1:
    compare_time.linesearch(
        run_options,
        machine_options,
        model_options,
        search,
        sarray,
        params
    )
if 1:
    compare_time.plot(
        run_options,
        model_options,
        search,
        sarray
    )

