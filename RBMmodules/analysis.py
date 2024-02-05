from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy.typing as npt

sns.set_theme(font_scale=1.5)

def plot_energy(epochs, energy_array):
    epochs_array = np.linspace(0, epochs, epochs)
    fig = plt.figure()
    plot_array = np.zeros(epochs)
    for i in range(epochs):
        plot_array[i] = energy_array[i]
    plt.plot(epochs_array, plot_array)
    return fig

def convergence(
    axe         : plt.Axes,
    x           : npt.NDArray[np.float64],
    y           : npt.NDArray[np.float64],
    error       : Any,
    color       : str = "blue",
    label       : str = "",
    x_label     : str = "x",
    y_label     : str = "y",
    error_color : str = "red") -> None:

    #plot convergence line
    axe.plot(x, y, linewidth=1.5, color=color, label=label)
    axe.set_xlabel(x_label)
    axe.set_ylabel(y_label)

    #plot error field
    axe.fill_between(
        x,
        y+error,
        y-error,
        color = error_color,
        alpha = 0.25)

