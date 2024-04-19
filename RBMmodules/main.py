from typing import Any, Union
from numpy._typing import _UnknownType
import torch
import warnings
import matplotlib.pyplot as plt
import model, solver, analysis, hamiltonian, logger
from functools import partial
warnings.filterwarnings("ignore", category=UserWarning)

def nop():
    pass

def run(
    model_options   : dict[str, Any],
    machine_options : dict[str, Any],
    run_options     : dict[str, Any],
    run_name        : str = "",
    log             : bool = True,
    verbose         : bool = False
    ) -> Union[str, dict[str, _UnknownType]]:

    vn        = machine_options["visual_n"]
    hn        = machine_options["hidden_n"]
    precision = machine_options["precision"]
    device    = machine_options["device"]

    machine_initialize = model.set_up_model(
        vn,
        hn,
        precision=precision,
        device=device,
        W_scale=0.1
    )

    epochs            = run_options["epochs"]
    learning_rate     = run_options["learning_rate"]
    adapt             = {
        "func" : run_options["adaptive_function"],
        "gamma": run_options["gamma"]
    }
    cycles            = run_options["monte_carlo"]["cycles"]
    monte_carlo       = run_options["monte_carlo"]["type"]
    masking_func      = model_options['masking_func'] 
    
    H = torch.tensor(
        model_options['hamiltonian'],
        dtype=precision,
        device=device
    )

    result = solver.find_min_energy(
        machine_initialize,
        H,
        masking_func,
        cycles,
        monte_carlo,
        epochs,
        learning_rate,
        adapt,
        verbose = verbose
    )

    machine_layers = {
        "visual"  : machine_initialize.visual_bias,
        "hidden"  : machine_initialize.hidden_bias,
        "weights" : machine_initialize.weights
    }
    
    if log == True:
        saved_path = logger.learning_process(
            model_options,
            machine_layers,
            machine_options,
            run_options,
            result,
            run_name
        )
        return saved_path
    else:
        return result


if __name__ == '__main__':

    visual_n = 2
    hidden_n = 5
    precision = torch.float64
    device = None #torch.device('cuda:0')

    init_model = model.set_up_model(visual_n, hidden_n, precision=precision, device=device, W_scale=0.1)
    
    eps = 1
    V = 1
    W = 0
    H = torch.tensor([[eps, 0, -V], [0, 0, 0], [-V, 0, -eps]], dtype=precision, device=device)

    eigvals = torch.real(torch.linalg.eigvals(H))
    min_eigval = torch.min(eigvals)
    
    epochs = 500
    learning_rate = 1
    cycles = 100000
    k = 3
    
    local_energy = partial(hamiltonian.lipkin_local, eps, V, W)
    # local_energy = partial(hamiltonian.ising_local, J=-0.5, L=-1)
    # print(f"Ising: {hamiltonian.ising_true(visual_n, -0.5, -1)}")
    
    stats = solver.find_min_energy(init_model, local_energy, cycles, k, epochs , learning_rate)

    print(f'Standard Methods: {min_eigval}, ({eigvals}) \n RBM_last: {stats["E_mean"][-1]} \n RBM_100 mean : {torch.sum(stats["E_mean"][-100:])/100}')
    print('------Model-------')
    print('VB: ', init_model.visual_bias)
    print('HB: ', init_model.hidden_bias)
    print('W: ', init_model.weights)
    print('-------Dist-------')
    print(stats['Dist'])

    analysis.plot_energy(epochs, stats['E_mean'])
    plt.title(r'$\left  E \right  $')
    plt.show()
