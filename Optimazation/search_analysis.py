import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
font = {'size'   : 20}

matplotlib.rc('font', **font)

# Load wanted run of gridsearch.py
def plot_mesh(
    search_x,
    search_y,
    run_options,
    model_options,
    machine_options
):
    particles = machine_options['visual_n']
    epochs = run_options['epochs']

    model = model_options['name']
    run_folder = f'[{particles}][e={epochs}][n{search_y[0]}-{search_y[-1]}][g{search_x[0]}-{search_x[-1]}]'
    folder = f'Saved/{model}/{run_folder}/'

    dE_grid = np.load(folder+'dE.npy')
    var_grid = np.load(folder+'grid.npy')
    part_grid = np.load(folder+'part.npy')
    dist = np.load(folder+'dist.npy')
    error = np.load(folder+'error.npy')
# Remove states dominated by a single basis state.

    split = dist > 0.98
    masked_var = np.ma.masked_array(var_grid, mask=split)
    masked_part = np.ma.masked_array(part_grid, mask=split)


    part_cutoff = part_grid > 2.5*np.median(np.ma.filled(masked_part, fill_value=999999))
    masked_part = np.ma.masked_array(part_grid, mask=part_cutoff)

    var_cutoff = var_grid > 2.5*np.median(np.ma.filled(masked_var, fill_value=999999))
    masked_var = np.ma.masked_array(var_grid, mask=part_cutoff)

# Print stats

    min = np.nanmin(masked_part)
    min_arg = np.where(masked_part==min)
    g_min = search_x[min_arg[1]]
    n_min = search_y[min_arg[0]]

    min_s = np.nanmin(masked_var)
    min_arg_s = np.where(masked_var==min_s)
    g_min_s = search_x[min_arg_s[1]]
    n_min_s = search_y[min_arg_s[0]]

    print(f"""
    Minimum variance basis   : {min} at g = {g_min} and n = {n_min} with error {error[min_arg]}
    Minimum variance samples : {min_s} at g = {g_min_s} and n = {n_min_s} with error {error[min_arg_s]}
    Minimum mean lr          : {search_y[np.argmin(np.mean(masked_part, 1))]}
    Minimum mean gamma       : {search_x[np.argmin(np.mean(masked_part, 0))]}
    Minimum error            : {np.nanmin(error)}
    """)
# Plot colormesh
    X, Y = np.meshgrid(search_x, search_y)

    plt.pcolormesh(X, Y, error)
    plt.colorbar()
    plt.show()

    fig = plt.figure(figsize=(9,7))
    ax0 = fig.add_subplot(111)
    mesh1 = ax0.pcolormesh(X, Y, masked_var, cmap='cividis')
    ax0.set_xlabel(r'$\gamma$')
    ax0.set_ylabel(r'$\eta$')

    divider = make_axes_locatable(ax0)
    cax = divider.append_axes('right', size='5%', pad=0.07)
    fig.colorbar(mesh1, cax=cax, orientation='vertical', label=r'$Var[E(S)]$')

#ax1 = fig.add_subplot(122)
#mesh2 = ax1.pcolormesh(X, Y, masked_part, cmap='cividis')
#ax1.set_xlabel(r'$\gamma$')
#ax1.set_ylabel(r'$\eta$')
#
#divider = make_axes_locatable(ax1)
#cax = divider.append_axes('right', size='5%', pad=0.07)
#fig.colorbar(mesh2, cax=cax, orientation='vertical', label=r'$Var[E(B)]$')
    name = model_options['name']
    fig.tight_layout()
    plt.savefig(f'Figures/{name}/'+run_folder + '.pdf')
    plt.show()

    plt.plot()

def plot_line(
    repeats,
    search_x,
    search,
    run_options,
    model_options,
    machine_options,
    show=True
    ):

    particles = machine_options['visual_n']
    epochs = run_options['epochs']

    model = model_options['name']
    run_folder = f'[{particles}][{search}][e={epochs}][{search_x[0]}-{search_x[-1]}]'
    folder = f'Saved/{model}/{run_folder}/'

    var_grid = np.load(folder+'array.npy')
    part_grid = np.load(folder+'part.npy')
    error = np.load(folder+'error.npy')

    min = np.nanmin(part_grid)
    min_arg = np.where(part_grid==min)
    h_min = search_x[min_arg]

    min_s = np.nanmin(var_grid)
    min_arg_s = np.where(var_grid==min_s)
    h_min_s = search_x[min_arg_s]

    part_grid /= np.sum(part_grid)
    part_grid *= np.sum(var_grid)

    print(f"""
    Minimum variance basis: {min} at hn = {h_min} with error {error[min_arg]}
    Minimum variance samples: {min_s} at hn = {h_min_s} and with error {error[min_arg_s]}
    Minimum error : {np.nanmin(error)}
    """)

    fig = plt.figure(figsize=(9,7))
    
    if search == 'hidden_n':
        label = r"$h_n$"
    elif search == 'learning_rate':
        label = r"$\eta$"
    elif search == 'gibbs_k':
        label = r"$k$"
    else:
        label = ""

    plt.xlabel(label)
    plt.ylabel(r'$Var[E_{local}]$')
    
    plt.scatter(search_x, var_grid/repeats, label=r'$Var[E_{local}]$')
    plt.plot(search_x, part_grid/repeats, 'r--', label=r'$Var[E_{basis}]$')
    plt.legend()
    fig.tight_layout()
    name = model_options['name']
    plt.savefig(f'Figures/{name}/'+run_folder + '.pdf')
    plt.title(run_folder)
    if show:
        plt.show()


