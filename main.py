import torch
import warnings
import matplotlib.pyplot as plt
from functools import partial
from RBMmodules import model, solver, analysis
warnings.filterwarnings("ignore", category=UserWarning)

def lipkin_local(dist_s, uniform_s, eps, V, W):
    mask = torch.all(dist_s == uniform_s, dim=1)
    H_0 = torch.zeros_like(dist_s[:, 0])
    H_0[mask] = 2*eps*(torch.sum(dist_s[mask] == 1, dim=1) - torch.sum(dist_s[mask] == 0, dim=1)).to(dtype=H_0.dtype)
    doubles = torch.sum(dist_s-uniform_s, dim=1)
    doubles[torch.abs(doubles) != 2] = 0
    H_1 = 0.5*V*torch.abs(doubles)
    #H_1 = -V*(torch.abs(torch.sum(dist_s-uniform_s, dim=1)) == 2)
    H_2 = W*(torch.abs(torch.sum(dist_s-uniform_s, dim=1)) == 1)
    sum = H_0 + H_1
    return sum

if __name__ == '__main__':

    visual_n = 2
    hidden_n = 5
    precision = torch.float64
    device = torch.device('cuda:0')

    init_model = model.set_up_model(visual_n, hidden_n, precision=precision, device=device, W_scale=0.1)
    
    eps = 0.2
    v = 1
    W = 0
    H = torch.tensor([[eps, 0, -v], [0, 0, 0], [-v, 0, -eps]], dtype=precision, device=device)
    eigvals = torch.real(torch.linalg.eigvals(H))  
    min_eigval = torch.min(eigvals)
    
    epochs = 4000
    learning_rate = 1
    cycles = 100000
    k = 3
    local_energy = partial(lipkin_local, eps=eps, V=v, W=W)
    stats = solver.find_min_energy(init_model, local_energy, cycles, k, epochs , learning_rate)

    print(f'Standard Methods: {min_eigval}, ({eigvals}) \n RBM: {stats["E_mean"][-1]}')
    print('------Model-------')
    print('VB: ', init_model.visual_bias)
    print('HB: ', init_model.hidden_bias)
    print('W: ', init_model.weights)

    print('-------Dist-------')
    print(stats['Dist'])

    analysis.plot_energy(epochs, stats['E_mean'])
    plt.title('$\left < E \\right > $')
    plt.show()
