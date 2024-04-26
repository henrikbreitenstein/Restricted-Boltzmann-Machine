import numpy as np
import sys
import os
import torch

sys.path.append('../')
sys.path.append('../RBMmodules')
from RBMmodules import hamiltonian, main, adaptives


def gridsearch(
    repeats,
    resolution,
    search_x,
    search_y,
    run_options,
    model_options,
    machine_options,
    true_val,
    plot = False,
    verbose = False
    ):
    
    print(f"""
    Gridsearch of the {model_options["name"]} model 
    with {resolution*resolution} total data points
    """)

    var_grid = np.zeros((resolution, resolution))
    part_grid = np.zeros((resolution, resolution))
    error = np.zeros((resolution, resolution))
    for _ in range(repeats):
        for i in range(resolution):
            for j in range(resolution):
                
                run_options['gamma'] = search_x[j]
                  
                run_options['learning_rate'] = search_y[i]

                result = main.run(
                    model_options,
                    machine_options,
                    run_options,
                    "",
                    log = False,
                    verbose = verbose
                )

                var_grid[i, j] += result['variance'][-1]
                part_grid[i, j] += result['part_var'][-1]
                error[i, j] += abs(result['E_mean'][-1] - true_val)
                
                    
    vec = hamiltonian.ground_state_vec(model_options["hamiltonian"])

    if plot:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(font_scale=1.5)

        plt.xlabel("Epoch")
        plt.ylabel(r"$E$")
        plt.plot(
            np.arange(1, run_options['epochs']+1), result['E_mean'],
            label = r'$E_{pred}$'
        )
        plt.plot(
            np.arange(1, run_options['epochs']+1),
            true_val*np.ones(run_options['epochs']),
            label = r'$E_{true}$'
        )
        title = model_options['name']
        for var in model_options['args'].items():
            title += f' {var[0]}={var[1]}'
        plt.title(title)
        plt.legend()
        plt.show()

        vn = machine_options['visual_n']
        for i in range(vn):
            y = result['vb'][:, i]
            plt.plot(np.arange(1, run_options['epochs']+1), y, label=f'v_{i}')
        plt.title(title + ' V Bias')
        plt.legend()
        plt.show()

        hn = machine_options['hidden_n']
        for i in range(hn):
            y = result['hb'][:, i]
            plt.plot(np.arange(1, run_options['epochs']+1), y, label=f'h_{i}')
        plt.title(title + ' H Bias')
        plt.legend()
        plt.show()

        for i in range(vn):
            y = result['dvb'][:, i]
            plt.plot(np.arange(1, run_options['epochs']+1), y, label=f'v_{i}')
        plt.title(title + ' dV Bias')
        plt.legend()
        plt.show()

        hn = machine_options['hidden_n']
        for i in range(hn):
            y = result['dhb'][:, i]
            plt.plot(np.arange(1, run_options['epochs']+1), y, label=f'h_{i}')
        plt.title(title + ' dH Bias')
        plt.legend()
        plt.show()


    if verbose:
        np.set_printoptions(precision=4, suppress=True)
        print("Amplitudes: ", np.array(result['amps'][-1].cpu()))
        print(f"Amps: {vec[1]}, eig: {vec[0]}")
        print(f"Final E_loc: {result['E'][-1]}")
        print(f"Final error = {error[-1,-1]}")
    model = model_options['name']
    epochs = run_options['epochs']
    n_particles = machine_options['visual_n']
    run_folder = f'[{n_particles}][e={epochs}][n{search_y[0]}-{search_y[-1]}][g{search_x[0]}-{search_x[-1]}]'
    folder = f'Saved/{model}/{run_folder}/'

    try:
        os.mkdir(folder)
    except:
        pass
    if resolution > 1:
        np.save(folder +'error.npy', error)
        np.save(folder+'grid.npy', var_grid)
        np.save(folder+'part.npy', part_grid)

def linesearch(
    repeats,
    resolution,
    search_x,
    search,
    run_options,
    model_options,
    machine_options,
    true_val
    ):
    
    print(f"""
    Running {repeats} repeats of a {resolution} points axis line search.
    """)
    var_grid = np.zeros(resolution)
    part_grid = np.zeros(resolution)
    error = np.zeros(resolution)
    for repeat in range(repeats):
        for i in range(resolution):
            
            if search == 'hidden_n':
                machine_options['hidden_n'] = search_x[i]
            elif search == 'learning_rate':
                run_options['learning_rate'] = search_x[i]
            elif search == 'gibbs_k':
                run_options['monte_carlo']['type'] = search_x[i]

            result = main.run(
                model_options,
                machine_options,
                run_options,
                "",
                log = False
            )
        
            var_grid[i]   += result['variance'][-1]
            part_grid[i]  += result['part_var'][-1]
            error[i]      += abs(result['E_mean'][-1] - true_val)

    model = model_options['name']
    epochs = run_options['epochs']
    n_particles = machine_options['visual_n']
    run_folder = f'[{n_particles}][{search}][e={epochs}][{search_x[0]}-{search_x[-1]}]'
    folder = f'Saved/{model}/{run_folder}/'

    try:
        os.mkdir(folder)
    except:
        pass

    np.save(folder +'error.npy', error)
    np.save(folder+'array.npy', var_grid)
    np.save(folder+'part.npy', part_grid)

