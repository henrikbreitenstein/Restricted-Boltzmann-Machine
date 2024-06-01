import numpy as np
import torch
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
    opt_param,
    opt=True
    ):
    
    print(f"""
    Running the {model["name"]}, checking {search} 
    from {sarray[0]} to {sarray[-1]} with {len(sarray)} points.
    """)

    error = np.zeros(len(sarray))
    rel_err = np.zeros(len(sarray))
    var = np.zeros(len(sarray))
    val = np.zeros(len(sarray))
    
    for i in range(len(sarray)):
    
        if search == 'particles':
            if 0:
                machine['visual_n'] = sarray[i]
            else:
                model["args"]["n"] = sarray[i]
        else:
            model["args"][search] = sarray[i]
        if opt:
            opt_n = machine['visual_n'] - 4
            machine['hidden_n'] = opt_param['hidden_n'][opt_n]
            run['learning_rate'] = opt_param['lr'][opt_n]

        H = model["H_func"](**model["args"])
        model["hamiltonian"] = H
        true_val = hamiltonian.ground_state(H)

        result = main.run(
            model,
            machine,
            run,
            "",
            log = False
        )
        
        argv       = torch.argmin(result["E_mean"])
        var[i]     = result['variance'][argv]
        error[i]   = abs(result['E_mean'][argv] - true_val)
        rel_err[i] = error[i]/true_val
        val[i]     = result['E_mean'][argv]

    epochs = run['epochs']
    n = machine['visual_n']
    run_folder = f'[{search}][{sarray[0]}-{sarray[-1]}]'
    run_folder += f'[e={epochs}]'
    if search == 'particles':
        for arg in model["args"].items():
            if (arg[0] != search) and (arg[0] != 'basis'):
                run_folder += f'[{arg[0]}={arg[1]}]'
    else:
        run_folder += f'[n={n}]'
        for arg in model["args"].items():
            if (arg[0] != search) and (arg[0] != 'basis'):
                run_folder += f'[{arg[0]}={arg[1]}]'
    
    name = model['name']
    folder = f'Saved/{name}/{run_folder}/'

    try:
        os.mkdir(folder)
    except:
        pass

    np.save(folder+'error.npy', error)
    np.save(folder+'var.npy', var)
    np.save(folder+'rel_err.npy', rel_err)
    np.save(folder+'val.npy', val)

def plot(run, machine, model, search, sarray, show=True, toplot="val-true"):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(font_scale=1.5)

    epochs = run['epochs']
    n = machine['visual_n']
    run_folder = f'[{search}][{sarray[0]}-{sarray[-1]}]'
    run_folder += f'[e={epochs}]'
    if search == 'particles':
        for arg in model["args"].items():
            if arg[0] != 'basis':
                run_folder += f'[{arg[0]}={arg[1]}]'
    else:
        run_folder += f'[n={n}]'
        for arg in model["args"].items():
            if (arg[0] != search) and (arg[0] != 'basis'):
                run_folder += f'[{arg[0]}={arg[1]}]'
    
    name = model['name']
    folder = f'Saved/{name}/{run_folder}/'
    
    error = np.load(folder+'error.npy')
    var = np.load(folder+'var.npy')
    rel_err = np.load(folder+'rel_err.npy')
    val = np.load(folder+'val.npy')

    fig, ax1 = plt.subplots(figsize=(9,7))

    if search == 'particles':
        ax1.set_xlabel(r"$n$")
    elif search == 'eps':
        ax1.set_xlabel(r"$\varepsilon$")
    else:
        ax1.set_xlabel(f"${search}$")

    if toplot=="val-var":
        ax1.set_ylabel("$E_{rbm}$", color='tab:red')
        ax1.plot(sarray, val, label='$E_{rbm}$', color='tab:red')
        ax1.tick_params(axis='y', labelcolor='tab:red')
        
        ax2= ax1.twinx()
        ax2.set_ylabel("$Var[E_{local}]$", color='tab:blue')
        ax2.plot(sarray, var, '--', color='tab:blue')
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        ax2.grid(None)

    if toplot=="val-true":
        trueval = 1/(rel_err/error)
        ax1.set_ylabel("$E$")
        ax1.plot(sarray, trueval, label='$E_{true}$', color='tab:blue')
        ax1.scatter(sarray, val, label='$E_{rbm}$', color='tab:red')
        ax1.legend()

    plt.savefig(f"Figures/{name}/{toplot}{run_folder}.pdf")
    if show:
        plt.title(run_folder)
        plt.show()
    else:
        plt.close()

