import torch
import numpy as np
import sys
import matplotlib.pyplot as plt


sys.path.append('../')
sys.path.append('../RBMmodules')
from functools import partial
from RBMmodules import hamiltonian, main, adaptives



run_options = {
    "epochs"      : 1000,
    "monte_carlo" : {
        "type"   : 2,
        "cycles" : 100_000
    },
    "learning_rate"     : 2,
    "adaptive_function" : None
    }

machine_options = {
    "visual_n" : 2,
    "hidden_n" : 4,
    "precision" : torch.float64,
    "device" : torch.device('cuda')
}

basis = hamiltonian.create_basis(
    2,
    machine_options['precision'],
    machine_options['device']
)
model_options = {
    "name" : "Lipkin",
    "hamiltonian" : hamiltonian.lipkin_local,
    "basis" : basis,
    "args" : {
        "eps" : 1,
        "V"   : 0.5,
        "W"   : 0.2
    }
}

lr_list = np.linspace(0.07111111, 0.09, 10)
nop_error = np.zeros(len(lr_list))
adapt_error = np.zeros(len(lr_list))

gamma=0.66666667
adapt = partial(adaptives.momentum, gamma=gamma)
for i, lr in enumerate(lr_list):
    
    run_options["learning_rate"] = lr
    
    run_options["adaptive_function"] = adaptives.nop
    nop_result = main.run(
        model_options,
        machine_options,
        run_options,
        "",
        log = False
    )

    run_options['adaptive_function'] = adapt
    adapt_result = main.run(
        model_options,
        machine_options,
        run_options,
        "",
        log = False
    )

    E_true = hamiltonian.lipkin_true(2, 1, 0.5, 0.2)
    print(E_true,
          nop_result['E_mean'][-1],
          adapt_result['E_mean'][-1],
          adapt_result['part_var'][-1]
    )
    nop_error[i] = abs(nop_result['E_mean'][-1] - E_true)
    adapt_error[i] = abs(adapt_result['E_mean'][-1] - E_true)

print(f"Min nop: {min(nop_error)}, Min adapt: {min(adapt_error)}")
nop_median = np.median(nop_error)
nop_error = np.ma.masked_array(nop_error, mask=(nop_error>nop_median))
adapt_median = np.median(adapt_error)
adapt_error = np.ma.masked_array(adapt_error, mask=(adapt_error>adapt_median))


fig, axs = plt.subplots(1, 2)
axs[0].scatter(lr_list, nop_error, label='Constant lr')
axs[1].scatter(lr_list, adapt_error, color='sienna', label='Adaptive lr')

axs[0].set_xlabel(r'$\eta$')
axs[0].set_ylabel('absolute error')
axs[0].legend()

axs[1].set_xlabel(r'$\eta$')
axs[1].set_ylabel('absolute error')
axs[1].legend()

plt.show()
