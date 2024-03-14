import torch
import numpy as np
import itertools

def dd(a, b):
    if a==b:
        return 1
    else:
        return 0

def construct_spin(S):

    size = int(2*S + 1)
    J_pluss = np.zeros((size, size))
    J_minus = np.zeros((size, size))
    J_z = np.zeros((size, size))

    for i in range(size):
        for k in range(size):
            m = i-S
            n = k-S

            pm_factor = np.sqrt(S*(S+1) -m*n)

            J_pluss[i, k] = dd(m, n+1)*pm_factor
            J_minus[i, k] = dd(m+1, n)*pm_factor
            J_z[i, k] = dd(m,n)*m
    
    return J_z, J_pluss, J_minus

def construct_lipkin(S, eps, V):
    
    J_z, J_pluss, J_minus = construct_spin(S)

    H = eps*J_z - 1/2*V*(J_pluss@J_pluss + J_minus@J_minus)

    return H

def lipkin_true(n, eps, V):
    H = torch.tensor(construct_lipkin(n/2, eps, V))
    eigvals = torch.real(torch.linalg.eigvals(H))
    min_eigval = torch.min(eigvals)
    return min_eigval

def lipkin_local(eps, V, W, samples):

    size = samples.shape[0]
    unique = torch.tensor(
        list(itertools.product([0, 1], repeat=samples[0].shape[0])),
        dtype = samples.dtype,
        device= samples.device
    )
    weight = torch.sum(torch.all(samples[:, None] == unique, dim = -1), dim=0)
    div_weight = weight
    div_weight[weight==0] = 1
   # unique, weight = torch.unique(samples, dim=0, return_counts=True)
   # weight = torch.sqrt(weight/size)
    mask = torch.where(torch.all(samples[:, None] == unique, dim = -1))[1]
    
    N_0 = torch.sum(unique == 0, dim=-1)
    N_1 = torch.sum(unique == 1, dim=-1)

    diff_unique = torch.sum(torch.logical_xor(unique[:, None], unique), dim=-1)

    H_0 = 0.5*eps*(N_1-N_0)
    H_eps = H_0[mask]
    
    H_1 = V*torch.sum(weight[:, None]*(diff_unique == 2), dim=0)/div_weight
    H_V = H_1[mask]
    
    H_2 = W*torch.sum(weight[:, None]*(diff_unique == 1), dim=0)/div_weight
    H_W = H_2[mask]
    
    E = (H_eps - H_V - H_W)
    return E

def ising_local(samples, uniform_s, J, L):
    pm = 2*samples - 1
    shift = torch.roll(pm, 1)
    H_J = J*torch.sum(shift*pm, dim=1)
    H_L = L*torch.sum(pm, dim=1)
    return H_L + H_J

#def ising_true(N, J, L):
    hi = nk.hilbert.Spin(s=1/2, N=N) 
    H = sum([L*sigmax(hi,i) for i in range(N)])
    H += sum([J*sigmaz(hi,i)*sigmaz(hi,(i+1)%N) for i in range(N)])
    sp_h=H.to_sparse()
    from scipy.sparse.linalg import eigsh
    eig_vals, eig_vecs = eigsh(sp_h, k=2, which="SA")
    E_gs = eig_vals[0]
    return E_gs

def heisenberg_local():
    ...


if __name__ == "__main__":
    dist = torch.tensor(([[0, 0]]*24 + [[1, 1]]*4))
    print(torch.mean(lipkin_local(dist, 1, 1, 0)))
