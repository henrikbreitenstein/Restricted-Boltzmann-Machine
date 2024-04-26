import numpy as np
import sys
import os

sys.path.append('../')
sys.path.append('../RBMmodules')
from RBMmodules import main, hamiltonian

def linesearch(
    run,
    machine,
    model,
    search,
    sarray,
    opt_param
    ):
    
    print(f"""
    Running the {model["name"]}, checking {search} 
    from {sarray[0]} to {sarray[-1]} with {len(sarray)} points.
    """)

    error = np.zeros(len(sarray))
    rel_err = np.zeros(len(sarray))
    var_b = np.zeros(len(sarray))

    
    for i in range(len(sarray)):
    
        if search == 'particles':
            machine['visual_n'] = sarray[i]
        else:
            model["args"][search] = sarray[i]

        opt_n = machine['visual_n'] - 2
        machine['hidden_n'] = opt_param['hidden_n'][opt_n]
        run['monte_carlo']['type'] = opt_param['gibbs_k'][opt_n]
        run['learning rate'] = opt_param['learning rate'][opt_n]

        H = model["H_func"](
            n = machine['visual_n']
            **model["args"]
        )
        model["hamiltonian"] = H
        true_val = hamiltonian.ground_state(H)

        result = main.run(
            model,
            machine,
            run,
            "",
            log = False
        )

        var_b[i]  = result['part_var'][-1]
        error[i]  = abs(result['E_mean'][-1] - true_val)
        rel_err[i] = error[i]/true_val

    epochs = run['epochs']
    n = machine['visual_n']
    run_folder = f'[{search}][{sarray[0]}-{sarray[-1]}]'
    run_folder += f'[e={epochs}]'
    if search == 'particles':
        for arg in model["args"].items():
            run_folder += f'[{arg[0]}={arg[1]}]'
    else:
        run_folder += f'[n={n}]'
        for arg in model["args"].items():
            if arg[0] != search:
                run_folder += f'[{arg[0]}={arg[1]}]'
    
    name = model['name']
    folder = f'Saved/{name}/{run_folder}/'

    try:
        os.mkdir(folder)
    except:
        pass

    np.save(folder +'error.npy', error)
    np.save(folder+'var_b.npy', var_b)

def plot(run, machine, model, search, sarray, show=True):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(font_scale=1.5)

    epochs = run['epochs']
    n = machine['visual_n']
    run_folder = f'[{search}][{sarray[0]}-{sarray[-1]}]'
    run_folder += f'[e={epochs}]'
    if search == 'particles':
        for arg in model["args"].items():
            run_folder += f'[{arg[0]}={arg[1]}]'
    else:
        run_folder += f'[n={n}]'
        for arg in model["args"].items():
            if arg[0] != search:
                run_folder += f'[{arg[0]}={arg[1]}]'
    
    name = model['name']
    folder = f'Saved/{name}/{run_folder}/'
    
    error = np.load(folder+'error.npy')
    var_b = np.load(folder+'var_b.npy')
    rel_err = np.load(folder+'rel_err.npy')

    fig, ax1 = plt.subplots(figsize=(9,7))

    if search == 'particles':
        ax1.set_xlabel(r"$h_n$")
    elif search == 'eps':
        ax1.set_xlabel(r"$\varepsilon$")
    else:
        ax1.set_xlabel(f"${search}$")
    ax1.set_ylabel("Error", color='tab:red')
    ax1.plot(sarray, error, label='Error', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    
    ax2= ax1.twinx()
    ax2.set_ylabel("Relative Error", color='tab:blue')
    ax2.plot(sarray, rel_err, color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    if show:
        plt.title(run_folder)
        plt.show()
        plt.title("")
    plt.savefig(f"Figures/{name}/{run_folder}")

