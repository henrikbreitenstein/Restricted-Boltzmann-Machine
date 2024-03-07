import torch
import sys
import matplotlib.pyplot as plt

sys.path.append('../')
sys.path.append('../RBMmodules')
from RBMmodules import hamiltonian, main, analysis, logger

model_options = {
    "name" : "Lipkin",
    "hamiltonian" : hamiltonian.lipkin_local,
    "args" : {
        "eps" : 1,
        "V"   : 1,
        "W"   : 0
    }
}
run_options = {
    "epochs"      : 500,
    "monte_carlo" : {
        "type"   : 5,
        "cycles" : 10_000
    },
    "learning_rate"     : 0.5,
    "adaptive_function" : None
    }

n_particles = 2
machine_options = {
    "visual_n" : n_particles,
    "hidden_n" : 2*n_particles,
    "precision" : torch.float64,
    "device" : torch.device('cuda')
}

result_path = main.run(
    model_options,
    machine_options,
    run_options,
    "hamiltonian_tests/runs"
)

run = logger.load_learning_process(result_path)
fig = analysis.plot_energy(
    run_options["epochs"],
    run["result"]["E_mean"]
)

plt.show()


