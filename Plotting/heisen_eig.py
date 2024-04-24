import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
sys.path.append('../')
sys.path.append('../RBMmodules')

from RBMmodules import hamiltonian
sns.set_theme(font_scale=1.5)


N = 2
M = 2
L = 1

test = hamiltonian.heisen_hamiltonian(N, M, 1, L)
n = 2**(N*M)
eigenvalues = np.zeros((100, n))

J_range= np.linspace(-1, 1, 100)

for i, J in enumerate(J_range):
    
    H = hamiltonian.heisen_hamiltonian(N, M, J, L)
    eig = (np.linalg.eigvals(H).real).round(decimals=5)
    eig.sort()
    eigenvalues[i] = eig

fig = plt.figure(figsize=(9,7))
plt.xlabel("J")
plt.ylabel("E")

for i in range(n):
    plt.plot(J_range, eigenvalues[:, i], label=r"$E_{" + f"{i}" + r"}$")

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
