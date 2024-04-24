import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
sys.path.append('../')
sys.path.append('../RBMmodules')

from RBMmodules import hamiltonian
sns.set_theme(font_scale=1.5)

eps = 0.5
P = 3
n = 1

for g in [-1, -0.7, -0.2, 0.2, 0.7, 1]:
    test = hamiltonian.pairing_hamiltonian(P, n, eps, g)
    eig = np.unique((np.linalg.eigvals(test).real).round(decimals=5))
    print(eig)

eig_len = len(eig)
eigenvalues = np.zeros((100, eig_len))

J_range= np.linspace(-1, 1, 100)

for i, g in enumerate(J_range):
    
    H = hamiltonian.pairing_hamiltonian(P, n, eps, g)
    eig = np.unique((np.linalg.eigvals(H).real).round(decimals=5))
    eig.sort()
    eigenvalues[i] = eig

fig = plt.figure(figsize=(9,7))
plt.xlabel("g")
plt.ylabel("E")

for i in range(eig_len):
    plt.plot(J_range, eigenvalues[:, i], label=r"$E_{" + f"{i}" + r"}$")

plt.legend()
plt.show()
