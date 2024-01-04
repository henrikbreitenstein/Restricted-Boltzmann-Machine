import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme(font_scale=1.5)

def plot_energy(epochs, energy_array):
    epochs_array = np.linspace(0, epochs, epochs)
    fig = plt.figure()
    plot_array = np.zeros(epochs)
    for i in range(epochs):
        plot_array[i] = energy_array[i]
    plt.plot(epochs_array, plot_array)
    return fig


