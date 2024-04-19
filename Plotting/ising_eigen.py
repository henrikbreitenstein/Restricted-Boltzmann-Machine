import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(font_scale=1.5)


N = 2
M = 1
L = 1

eigenvalues = np.zeros((100, 4))

J_range= np.linspace(-1, 1, 100)

for i, J in enumerate(J_range):
    
    H = np.array([
        [J, 1, 1, 0],
        [1, -J,0, 1],
        [1, 0, -J, 1],
        [0, 1, 1, J]
    ])

    eig = np.linalg.eigvals(H)
    eig.sort()
    eigenvalues[i] = eig

fig = plt.figure(figsize=(9,7))
plt.xlabel("J")
plt.ylabel("E")

for i in range(4):
    plt.plot(J_range, eigenvalues[:, i], label=rf"$E_{i}$")

plt.legend()
plt.show()
