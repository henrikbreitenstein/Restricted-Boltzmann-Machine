import torch
import numpy as np
import sys
import model_analysis

sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../RBMmodules')
sys.path.append('../../Optimazation')
from functools import partial
from RBMmodules import hamiltonian
from Optimazation import pairing_line

params = pairing_line.pairing_results_line

n_particles = 2; P = 4
run_options = {
    "epochs"      : 500,
    "monte_carlo" : {
        "type"   : params['gibbs_k'][P-2],
        "cycles" : 500_000
    },
    "learning_rate"     : params['lr'][P-2],
    "adaptive_function" : None
    }

machine_options = {
    "visual_n" : P,
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
    search = "eps"
    points = 20
    sarray = np.linspace(0, 1, points)

if 1:
    search = "V"
    points = 20
    sarray = np.linspace(0, 1, points)

if 1:
    search = "W"
    points = 20
    sarray = np.linspace(0, 1, points)

if 1:
    model_analysis.linesearch(
        run_options,
        machine_options,
        model_options,
        search,
        sarray
    )
if 1:
    model_analysis.plot(
        run_options,
        machine_options,
        model_options,
        search,
        sarray
    )

