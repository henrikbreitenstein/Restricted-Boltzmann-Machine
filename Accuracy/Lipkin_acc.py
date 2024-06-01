import torch
import numpy as np
import sys
import model_analysis

sys.path.append('../')
sys.path.append('../RBMmodules')
sys.path.append('../Optimazation')
from RBMmodules import hamiltonian, adaptives
from Optimazation import lipkin_line

params = lipkin_line.lipkin_results_grid

n_particles = 16
run_options = {
    "epochs"      : 850,
    "monte_carlo" : {
        "type"   : 2,
        "cycles" : 70_000
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

eps = -2; V=-0.5; W = 0.1
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
points = 22
sarray = np.arange(4, 26)
points = len(sarray)

if 0:
    search = "eps"
    points = 50
    sarray = np.linspace(-2, 2, points)

if 0:
    search = "V"
    points = 50
    sarray = np.linspace(-1, 0, points)

if 0:
    search = "W"
    points = 50
    sarray = np.linspace(-0.3, 0, points)

if 0:
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

