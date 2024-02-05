import os
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import re

from RBMmodules import  logger, analysis
params = {
    'axes.labelsize': 20,
    'axes.titlesize':20, 
    'legend.fontsize': 20, 
    'xtick.labelsize': 20, 
    'ytick.labelsize': 20}
mpl.rcParams.update(params)

from_cwd = "./Results/Test_noninteractive/runs/Lipkin/"
n_runs = len(os.listdir(from_cwd))

ordering = np.empty(n_runs)
convergences = np.empty(n_runs, dtype=object)
errors = np.empty(n_runs, dtype=object)

for i, run_dir in enumerate(os.listdir(from_cwd)):
    n_particles = re.search(r"n=(?P<n>\d+)", run_dir).group('n')
    ordering[i] = int(n_particles)
    run = logger.load_learning_process(from_cwd+run_dir)
    convergences[i] = run["result"]["E_mean"]
    errors[i] = run["result"]["std"]


sorted = ordering.argsort(axis=0)

ordering = ordering[sorted]
convergences = convergences[sorted]
errors = errors[sorted]

figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(13, 6))
std_fig, std_axe = plt.subplots(nrows=1, ncols=1, figsize=(7.5, 6))

#True values as dashed lines
epochs = len(convergences[0])
cut_off = 2
axes[0].plot([], [], color='black', linestyle='dashed', label='Analytical')
true_energies = [-1,-2,-4,-8]
for true_energy in true_energies:
    axes[0].plot(
        np.arange(0, epochs)[cut_off:],
        np.full(epochs-cut_off, true_energy),
        color='black',
        linestyle='dashed'
    )

colors = ["blue", "green", "red", "purple"]
for n, line, error, color in zip(ordering, convergences, errors, colors):
    analysis.convergence(
        axes[0],
        np.arange(0, len(line), dtype=np.float64)[cut_off:],
        line[cut_off:],
        error[cut_off:],
        color=color,
        label=f"n={n:.0f}",
        x_label="Epoch",
        y_label=r"Predicted $\langle E \rangle$",
    )
std_axe.plot(
    ordering,
    -ordering/2,
    color='black',
    linestyle='dashed',
    zorder=5
)
axes[1].plot(
    ordering,
    -ordering/2,
    color='black',
    linestyle='dashed',
    zorder=5
)

for line_number, line in enumerate(convergences):
    axes[1].scatter(
        ordering[line_number],
        line[-1],
        color=colors[line_number],
        zorder=10
    )
    std_axe.scatter(
        ordering[line_number],
        line[-1],
        label=f"{ordering[line_number]:.0f}",
        color=colors[line_number],
        zorder=10
    )

figure.legend(loc='upper center', ncol=5, columnspacing=0.7, fancybox=True)
std_fig.legend(loc='upper center', ncol=5, columnspacing=0.7, fancybox=True)
error = [abs(line[-1]/(n/2)+1) for n, line in zip(ordering, convergences)]
stds = [std[-1]/(n/2) for n, std in zip(ordering, errors)]

twin_axe = axes[1].twinx()
twin_axe.grid(False)
twin_axe.ticklabel_format(
    axis='y',
    style='sci',
    scilimits=(0, 0)
)
std_twaxe = std_axe.twinx()
std_twaxe.grid(False)
std_twaxe.ticklabel_format(
    axis='y',
    style='sci',
    scilimits=(0, 0)
)
twin_axe.set_ylabel("Error")
twin_axe.plot(ordering, error, label="Relative error", alpha=0.8)
twin_axe.legend(loc='upper center')

std_twaxe.plot(ordering, stds, label="Relative std", alpha=0.6, color='red')
std_axe.set_xlabel("n (particles)")
std_axe.set_ylabel(r"Predicted $\langle E \rangle$")
std_twaxe.set_ylabel("Error")
std_twaxe.legend(loc='upper center')

axes[1].set_xlabel("n (particles)")
axes[1].set_ylabel("Final prediction")
figure.savefig("Results/Test_noninteractive/plots/Lipkin2-16.pdf", bbox_inches="tight")
figure.savefig("C:\\Users\\breit\\_BASE\\Restricted-Boltzman-Machines-Latex\\Figures\\Plots\\Implementation Test\\Lipkin2-16.pdf", bbox_inches="tight")
std_fig.savefig("Results/Test_noninteractive/plots/Lipkin_relative_std.pdf", bbox_inches="tight")
std_fig.savefig("C:\\Users\\breit\\_BASE\\Restricted-Boltzman-Machines-Latex\\Figures\\Plots\\Implementation Test\\Lipkin_relative_std.pdf", bbox_inches="tight")

