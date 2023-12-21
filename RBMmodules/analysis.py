import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme(font_scale=1.5)

def plot_energy(epochs, energy_array):
    epochs_array = np.linspace(0, epochs, epochs)
    fig = plt.figure()
    plt.plot(epochs_array, energy_array)
    return fig


