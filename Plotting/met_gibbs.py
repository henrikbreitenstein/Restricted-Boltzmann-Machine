import torch
import numpy as np
import sys

sys.path.append('../')
sys.path.append('../RBMmodules')
sys.path.append('../Optimazation')
from RBMmodules import main, hamiltonian, adaptives
from Optimazation import lipkin_line

param = lipkin_line.lipkin_results_grid
n = 2
lr = param["lr"][int(n/2)-1]
hn = param["hidden_n"][int(n/2)-1]
gibbs_k = param["gibbs_k"][int(n/2)-1]

run_options = {
    "epochs"      : 300,
    "monte_carlo" : {
        "type"   : gibbs_k,
        "cycles" : 10_000
    },
    "learning_rate"     : lr,
    "adaptive_function" : adaptives.nop,
    "gamma" : 0
    }

machine_options = {
    "visual_n" : n,
    "hidden_n" : hn,
    "precision" : torch.float64,
    "device" : torch.device('cuda')
}
basis = hamiltonian.create_basis(
    n,
    machine_options['precision'],
    machine_options['device']
)

eps = 1.5 -np.sqrt(3)/2 ; V=-1; W = 0
model_options = {
    "name" : "Lipkin",
    "hamiltonian" : hamiltonian.lipkin_local,
    "basis" : basis,
    "args" : {
        "eps" : eps,
        "V"   : V,
        "W"   : W
    }
}

H = hamiltonian.lipkin_hamiltonian(
    n,
    eps,
    V,
    W
)

model_options["masking_func"] = hamiltonian.lipkin_amps
model_options["hamiltonian"] = H
true_val = hamiltonian.ground_state(H)

repeats = 1
points = 10


search_k = np.linspace(1, 21, points)
search_c = np.linspace(10_000, 500_000, points, dtype=int)
err_g = np.zeros(points)
err_a = np.zeros(points)
var = np.zeros(points)

for repeat in range(repeats):
    for i in range(points):
        run_options['monte_carlo']['cycles'] = search_c[i]
        run_options['monte_carlo']['type'] = gibbs_k
        result = main.run(
            model_options,
            machine_options,
            run_options,
            "",
            log = False
        )
        err_g[i] += abs(result['E_mean'][-1] - true_val)
        run_options['monte_carlo']['type'] = 0
        result = main.run(
            model_options,
            machine_options,
            run_options,
            "",
            log = False
        )
        err_a[i] += abs(result['E_mean'][-1] - true_val)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(font_scale=1.5)

plt.plot(search_c, err_g/repeats, label='Gibbs')
plt.plot(search_c, err_a/repeats, label='Criteria')
plt.xlabel("Samples")
plt.ylabel("Error")
plt.legend()
plt.show()








