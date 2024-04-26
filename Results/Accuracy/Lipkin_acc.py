import torch
import numpy as np
import sys
import model_analysis

sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../RBMmodules')
sys.path.append('../../Optimazation')
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
    "H_func" : hamiltonian.lipkin_hamiltonian,
    "masking_func" : hamiltonian.lipkin_amps,
    "basis" : basis,
    "args" : {
        "eps" : eps,
        "V"   : V,
        "W"   : W
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

