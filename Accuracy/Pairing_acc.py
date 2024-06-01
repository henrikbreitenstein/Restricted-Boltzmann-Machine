import torch
import numpy as np
import sys
import model_analysis

sys.path.append('../')
sys.path.append('../RBMmodules')
sys.path.append('../Optimazation')
from functools import partial
from RBMmodules import adaptives, hamiltonian
from Optimazation import pairing_line

params = pairing_line.pairing_results_line

n_particles =5; P = 10
run_options = {
    "epochs"      : 1000,
    "monte_carlo" : {
        "type"   : 2,
        "cycles" : 50_000
    },
    "learning_rate"     : 0.00551,
    "adaptive_function" : adaptives.nop,
    "gamma" : 0
    }

machine_options = {
    "visual_n" : P,
    "hidden_n" : 17,
    "precision" : torch.float64,
    "device" : torch.device('cuda')
}

basis = hamiltonian.create_basis(
    P,
    machine_options['precision'],
    machine_options['device']
)

eps = -0.3; g = 0;

model_options = {
    "name" : "Pairing",
    "H_func" : hamiltonian.pairing_hamiltonian,
    "masking_func" : partial(hamiltonian.amplitudes, basis=basis),
    "basis" : basis,
    "args" : {
        "basis" : basis,
        "n"   : n_particles,
        "eps" : eps,
        "g"   : g
    }
}

search = "particles"
sarray = np.arange(1, 10)
points = len(sarray)

if 0:
    search = "eps"
    points = 50
    sarray = np.linspace(-0.3, 0, points)

if 0:
    search = "g"
    points = 2
    sarray = np.linspace(-0.5, -0.3, points)

if 1:
    model_analysis.linesearch(
        run_options,
        machine_options,
        model_options,
        search,
        sarray,
        params
    )
if 1:
    model_analysis.plot(
        run_options,
        machine_options,
        model_options,
        search,
        sarray
    )

