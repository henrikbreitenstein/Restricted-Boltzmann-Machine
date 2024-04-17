import torch
import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.append('../')
sys.path.append('../RBMmodules')
from RBMmodules import hamiltonian, main, adaptives

run_options = {
    "epochs"      : 1,
    "monte_carlo" : {
        "type"   : 3,
        "cycles" : 100_000
    },
    "learning_rate"     : 10,
    "adaptive_function" : adaptives.nop
    }

n_particles = 2
machine_options = {
    "visual_n" : n_particles,
    "hidden_n" : 2*n_particles,
    "precision" : torch.float64,
    "device" : torch.device('cuda')
}

N = 2
V_range = np.linspace(0, 1, N)
est_energy = np.zeros(N)
true_energy = np.zeros(N)
std = np.zeros(N)
for i in range(N):
    model_options = {
        "name" : "Lipkin",
        "hamiltonian" : hamiltonian.lipkin_local,
        "args" : {
            "eps" : 1,
            "V"   : V_range[i],
            "W"   : 0
        }
    }
    result = main.run(
        model_options,
        machine_options,
        run_options,
        "hamiltonian_tests/runs",
        log = False
    )
    est_energy[i] = result["E_mean"][-1]
    std[i] = np.sqrt(result["variance"][-1])
    true_energy[i] = hamiltonian.lipkin_true(
        n_particles,
        model_options['args']['eps'],
        model_options['args']['V'],
        model_options['args']['W']
    )
print(std)
plt.errorbar(V_range, est_energy, yerr=std)
plt.plot(V_range, true_energy)
plt.show()


