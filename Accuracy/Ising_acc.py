import torch
import numpy as np
import sys
import model_analysis

sys.path.append('../')
sys.path.append('../RBMmodules')
sys.path.append('../Optimazation')
from functools import partial
from RBMmodules import hamiltonian, adaptives
from Optimazation import ising_line

params = ising_line.ising_results_grid
N = 10; M = 1
n_particles = N*M

if M>1:
    params = ising_line.ising2d_results_grid

run_options = {
    "epochs"      : 500,
    "monte_carlo" : {
        "type"   : 2,
        "cycles" : 100_000
    },
    "learning_rate"     : params['lr'][n_particles-4],
    "adaptive_function" : adaptives.nop,
    "gamma" : 0
    }

machine_options = {
    "visual_n" : n_particles,
    "hidden_n" : params['hidden_n'][n_particles-4],
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
    "name" : "Ising",
    "H_func" : partial(hamiltonian.ising_hamiltonian, N=N, M=M),
    "masking_func" : partial(hamiltonian.amplitudes, basis=basis),
    "basis" : basis,
    "args" : {
        "J" : J,
        "L"   : L,
    }
}

if M>1:
    model_options['name'] = "Ising2d"

search = "particles"
points = 20
sarray = np.arange(4, points+2)

if 1:
    search = "J"
    points = 50
    sarray = np.linspace(-1, 0, points)

if 0:
    search = "L"
    points = 50
    sarray = np.linspace(-0.5, 0, points)

if 1:
    model_analysis.linesearch(
        run_options,
        machine_options,
        model_options,
        search,
        sarray,
        params,
    )
if 1:
    model_analysis.plot(
        run_options,
        machine_options,
        model_options,
        search,
        sarray
    )

