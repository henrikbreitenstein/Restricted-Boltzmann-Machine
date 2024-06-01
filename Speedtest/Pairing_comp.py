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
from Optimazation import pairing_line

params = pairing_line.pairing_results_line

P=4; n_particles = 2
run_options = {
    "epochs"      : 500,
    "monte_carlo" : {
        "type"   : params['gibbs_k'][P-2],
        "cycles" : 500_000
    },
    "learning_rate"     : params['lr'][P-2],
    "adaptive_function" : adaptives.nop
    }

machine_options = {
    "visual_n" : n_particles,
    "hidden_n" : params['hidden_n'][P-2],
    "precision" : torch.float64,
    "device" : torch.device('cuda')
}

basis = hamiltonian.create_basis(
    P,
    machine_options['precision'],
    machine_options['device']
)

eps = -1; g = 2; 
H = hamiltonian.pairing_hamiltonian(
    basis,
    P,
    n_particles,
    eps,
    g
)

model_options = {
    "name" : "Pairing",
    "hamiltonian" : H,
    "masking_func" : partial(hamiltonian.amplitudes, basis=basis),
    "basis" : basis,
    "args" : {
        "eps" : eps,
        "g"   : g,
        "P"   : P,
        "n"   : n_particles
    }
}

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
