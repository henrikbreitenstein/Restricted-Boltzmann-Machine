import torch
import numpy as np
import sys
import compare_time

sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../RBMmodules')
sys.path.append('../../Optimazation')
from functools import partial
from RBMmodules import hamiltonian, adaptives
from Optimazation import heisen_line

params = heisen_line.heisen_results_grid
N = 2; M = 1; n_particles = N*M

if M>1:
    params = heisen_line.heisen2d_results_grid

run_options = {
    "epochs"      : 1000,
    "monte_carlo" : {
        "type"   : params['lr'][n_particles-2],
        "cycles" : 500_000
    },
    "learning_rate"     : params['lr'][n_particles-2],
    "adaptive_function" : adaptives.nop
    }

machine_options = {
    "visual_n" : n_particles,
    "hidden_n" : params['hidden_n'][n_particles-2],
    "precision" : torch.float64,
    "device" : torch.device('cuda')
}
basis = hamiltonian.create_basis(
    n_particles,
    machine_options['precision'],
    machine_options['device']
)

J = 0.3; L=-0.4
model_options = {
    "name" : "Heisenberg",
    "H_func" : hamiltonian.heisen_hamiltonian,
    "masking_func" : partial(hamiltonian.amplitudes, basis),
    "basis" : basis,
    "args" : {
        "J" : 0.3,
        "L"   : -0.4,
    }
}
if M>1:
    model_options['name'] = "Heisenberg2d"

search = "particles"
points = 20
sarray = np.arange(2, points+2)


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

