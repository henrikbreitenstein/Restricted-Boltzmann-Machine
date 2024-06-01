import numpy as np
import time
import sys
import os

sys.path.append('../')
sys.path.append('../RBMmodules')
from RBMmodules import main, hamiltonian
from functools import partial

def linesearch(
    run,
    machine,
    model,
    search,
    sarray,
    opt_param,
    repeats = 20
    ):
    
    print(f"""
    Running the {model["name"]}, comparing computation time
    from n = {sarray[0]} to n = {sarray[-1]}.
    """)

    time_RBM = np.zeros(len(sarray))
    time_NP = np.zeros(len(sarray))
    
    for i in range(len(sarray)):
        
        opt_n = sarray[i] - 4
        machine['visual_n'] = sarray[i]
        machine['hidden_n'] = opt_param['hidden_n'][opt_n]
        run['learning_rate'] = opt_param['lr'][opt_n]
 
        basis = hamiltonian.create_basis(
            sarray[i],
            machine['precision'],
            machine['device']
        )

        model['basis'] = basis
        model["masking_func"] = partial(hamiltonian.amplitudes, basis=basis)

        H = model["H_func"](
            int(sarray[i]),
            1,
            **model["args"]
        )
        model["hamiltonian"] = H
        for _ in range(repeats):

            start = time.time()
            true_val = hamiltonian.ground_state(H)
            time_NP[i] += time.time() - start
            
           # start = time.time()
           # result = main.run(
           #     model,
           #     machine,
           #     run,
           #     "",
           #     log = False
           # )
            time_RBM[i] += time.time() - start

    time_RBM /= repeats
    time_NP /= repeats
    epochs = run['epochs']
    run_folder = f'[{search}][{sarray[0]}-{sarray[-1]}]'
    run_folder += f'[e={epochs}]'
    
    for arg in model["args"].items():
        run_folder += f'[{arg[0]}={arg[1]}]'
    
    name = model['name']
    folder = f'Saved/{name}/{run_folder}/'

    try:
        os.mkdir(folder)
    except:
        pass

    #np.save(folder +'time_RBM.npy', time_RBM)
    np.save(folder+'time_NP.npy', time_NP)

def plot(run, model, search, sarray, show=True):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(font_scale=1.5)

    epochs = run['epochs']
    run_folder = f'[{search}][{sarray[0]}-{sarray[-1]}]'
    run_folder += f'[e={epochs}]'
    
    for arg in model["args"].items():
        run_folder += f'[{arg[0]}={arg[1]}]'
       
    name = model['name']
    folder = f'Saved/{name}/{run_folder}/'
    
    time_RBM = np.load(folder+'time_RBM.npy')
    time_NP = np.load(folder+'time_NP.npy')

    fig, ax1 = plt.subplots(figsize=(9,7))
    ax1.set_ylabel("Time [s]")
    ax1.plot(sarray, time_RBM,  label='RBM')
    ax1.plot(sarray, time_NP, label='Numpy')
    ax1.legend()
    ax1.set_xlabel("$n$")
    plt.savefig(f"Figures/{name}/{run_folder}.pdf")
    if show:
        plt.title(run_folder)
        plt.show()
        plt.title("")

