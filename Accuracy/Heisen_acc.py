import torch
import numpy as np
import sys
import model_analysis

sys.path.append('../')
sys.path.append('../RBMmodules')
sys.path.append('../Optimazation')
from functools import partial
from RBMmodules import hamiltonian, adaptives
from Optimazation import heisen_line

params = heisen_line.heisen_results_grid
N = 3; M = 3; n_particles = N*M

if M>1:
    params = heisen_line.heisen2d_results_grid

run_options = {
    "epochs"      : 1000,
    "monte_carlo" : {
        "type"   : 2,
        "cycles" : 500_000
    },
    "learning_rate"     : params['lr'][1],
    "adaptive_function" : adaptives.nop,
    "gamma" : 0
    }

machine_options = {
    "visual_n" : n_particles,
    "hidden_n" : params['hidden_n'][1],
    "precision" : torch.float64,
    "device" : torch.device('cuda')
}

basis = hamiltonian.create_basis(
    n_particles,
    machine_options['precision'],
    machine_options['device']
)

J = -1; L=-0.5
model_options = {
    "name" : "Heisenberg",
    "H_func" : hamiltonian.heisen_hamiltonian,
    "masking_func" : partial(hamiltonian.amplitudes, basis=basis),
    "basis" : basis,
    "args" : {
        "N" : N,
        "M" : M,
        "J" : J,
        "L"   : L
    }
}

if M>1:
    model_options["name"] = "Heisenberg2d"

search = "particles"
sarray = np.arange(4, 15)
points = len(sarray)

if 1:
    search = "J"
    points = 20
    sarray = np.linspace(-1, 0, points)

if 1:
    search = "L"
    points = 20
    sarray = np.linspace(-0.5, 0, points)

if 1:
    model_analysis.linesearch(
        run_options,
        machine_options,
        model_options,
        search,
        sarray,
        params,
        opt=False
    )
if 1:
    model_analysis.plot(
        run_options,
        machine_options,
        model_options,
        search,
        sarray
    )

