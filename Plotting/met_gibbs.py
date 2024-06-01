import torch
import numpy as np
import sys

sys.path.append('../')
sys.path.append('../RBMmodules')
sys.path.append('../Optimazation')
from RBMmodules import main, hamiltonian, adaptives
from Optimazation import lipkin_line

param = lipkin_line.lipkin_results_grid
n = 16

run_options = {
    "epochs"      : 850,
    "monte_carlo" : {
        "type"   : 2,
        "cycles" : 70_000
    },
    "learning_rate"     : param["lr"][n-4],
    "adaptive_function" : adaptives.nop,
    "gamma" : 0
    }

machine_options = {
    "visual_n" : n,
    "hidden_n" : param["hidden_n"][n-4],
    "precision" : torch.float64,
    "device" : torch.device('cuda')
}
basis = hamiltonian.create_basis(
    n,
    machine_options['precision'],
    machine_options['device']
)

eps = -2 ; V=-0.2; W = -0.14
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

repeats = 2
points = 20


search_c = np.linspace(-1, 0, points)
var_g = np.zeros(points)
var_a = np.zeros(points)

for repeat in range(repeats):
    for i in range(points):
        model_options["args"]["V"] = search_c[i]
        run_options['monte_carlo']['type'] = 2
        result = main.run(
            model_options,
            machine_options,
            run_options,
            "",
            log = False
        )
        var_g[i] += result['variance'][-1]
        run_options['monte_carlo']['type'] = 0
        result = main.run(
            model_options,
            machine_options,
            run_options,
            "",
            log = False
        )
        var_a[i] += result['variance'][-1]

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(font_scale=1.5)


fig, ax1 = plt.subplots(figsize=(9,7))
ax1.set_ylabel("$Var[E_{local}]$")
ax1.plot(search_c, var_g/repeats, label='Gibbs', color='tab:red')
ax1.plot(search_c, var_a/repeats, label='Criteria', color='tab:blue')
ax1.set_xlabel("V")
ax1.legend()
plt.savefig(f"[eps={eps}][V={V}][W={W}]met_gibbs.pdf")
plt.show()








