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
from Optimazation import lipkin_line

params = lipkin_line.lipkin_results_grid

n_particles = 2
run_options = {
    "epochs"      : 500,
    "monte_carlo" : {
        "type"   : params['gibbs_k'][n_particles-2],
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

eps = 1.5 -np.sqrt(3)/2 ; V=-1; W = 0
model_options = {
    "name" : "Lipkin",
    "H_func" : hamiltonian.lipkin_local,
    "masking_func" : hamiltonian.lipkin_amps,
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
model_options['hamiltonian'] = H

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