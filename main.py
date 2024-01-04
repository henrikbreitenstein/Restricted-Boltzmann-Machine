import torch
import warnings
import matplotlib.pyplot as plt
from functools import partial
from RBMmodules import model, solver, analysis, hamiltonian
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == '__main__':

    visual_n = 2
    hidden_n = 5
    precision = torch.float64
    device = None #torch.device('cuda:0')

    init_model = model.set_up_model(visual_n, hidden_n, precision=precision, device=device, W_scale=0.1)
    
    eps = 1
    V = 1
    W = 0
    H = torch.tensor([[eps, 0, -V], [0, 0, 0], [-V, 0, -eps]], dtype=precision, device=device)

    eigvals = torch.real(torch.linalg.eigvals(H))
    min_eigval = torch.min(eigvals)
    
    epochs = 500
    learning_rate = 1
    cycles = 100000
    k = 3
    
    local_energy = partial(hamiltonian.lipkin_local, eps=eps, V=V, W=W)
    # local_energy = partial(hamiltonian.ising_local, J=-0.5, L=-1)
    # print(f"Ising: {hamiltonian.ising_true(visual_n, -0.5, -1)}")
    
    stats = solver.find_min_energy(init_model, local_energy, cycles, k, epochs , learning_rate)

    print(f'Standard Methods: {min_eigval}, ({eigvals}) \n RBM_last: {stats["E_mean"][-1]} \n RBM_100 mean : {torch.sum(stats["E_mean"][-100:])/100}')
    print('------Model-------')
    print('VB: ', init_model.visual_bias)
    print('HB: ', init_model.hidden_bias)
    print('W: ', init_model.weights)
    print('-------Dist-------')
    print(stats['Dist'])

    analysis.plot_energy(epochs, stats['E_mean'])
    plt.title('$\left < E \\right > $')
    plt.show()
