import torch
import numpy as np
import itertools
from scipy.sparse.linalg import eigsh
from functools import partial

from torch.cuda import is_available

def ground_state(H):
    eigvals = np.linalg.eigvals(H)
    return min(eigvals)

def local_energy(H, amplitudes):
    weight, non_zero_mask, mask = amplitudes
    non_zero = weight[non_zero_mask]
    non_zero_H = H[non_zero_mask]
    amps = torch.zeros_like(weight)
    amps[non_zero_mask] = weight[non_zero_mask]

    E = torch.sum(non_zero[:,None]*non_zero_H, dim=0)/weight
    return E[mask], torch.var(E), E, amps

# Generic

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
    J_y = np.zeros((size, size))
    J_x = np.zeros((size, size))

    for i in range(size):
        for k in range(size):
            m = i-S
            n = k-S

            pm_factor = np.sqrt(S*(S+1) -m*n)

            J_pluss[i, k] = dd(m, n+1)*pm_factor
            J_minus[i, k] = dd(m+1, n)*pm_factor
            J_z[i, k] = dd(m,n)*m
            J_x[i, k] = (dd(m, n+1)+dd(m+1,n))*pm_factor
            J_y[i, k] = (dd(m, n+1)-dd(m+1,n))*pm_factor

    
    return J_z, J_pluss, J_minus, 0.5*J_x, (0.5+0.5j)*J_y

def create_basis(n, dtype, device):
    basis = torch.unique(torch.tensor(
        list(itertools.product([0, 1], repeat=n)),
        dtype = dtype,
        device= device
    ), dim=0)
    
    return basis

def amplitudes(samples, basis):
    size = samples.shape[0]
    
    weight = torch.sum(torch.all(samples[:, None] == basis, dim = -1), dim=0)
    weight = weight/size
    weight = torch.sqrt(weight)
    non_zero_mask = torch.where(weight>0)
    weight[weight==0] = 1/size
    mask = torch.where(torch.all(samples[:, None] == basis, dim = -1))[1] 
    return weight, non_zero_mask, mask

#Lipkin

def lipkin_hamiltonian(n, eps, V, W):
    J_z, J_pluss, J_minus = construct_spin(n/2)[:3]
    N = np.eye(n+1)*n
    H = eps*J_z 
    H += V/2*(J_pluss@J_pluss + J_minus@J_minus)
    H += W/2*(-N + J_pluss@J_minus+J_minus@J_pluss)
    return H

def lipkin_amps(samples):
    size, n = samples.shape
    basis_m = torch.arange(-n, n+1, 2, dtype=samples.dtype, device=samples.device)
    samples_m = torch.sum(2*samples-1, dim=-1)
    weight = torch.sum(samples_m[:, None] == basis_m, dim=0)
    weight = weight/size
    weight = torch.sqrt(weight)
    non_zero_mask = torch.where(weight>0)
    weight[weight==0] = np.sqrt(1/size)
    mask = torch.where(samples_m[:, None] == basis_m)[1]
    
    return weight, non_zero_mask, mask

def lipkin_local(eps, V, W, samples, basis):
    weight, non_zero_mask, mask = lipkin_amps(samples)
    non_zero = weight[non_zero_mask]
    
    H = lipkin_hamiltonian(samples.shape[1], eps, V, W)
    H = torch.tensor(H, dtype=samples.dtype, device=samples.device)
    non_zero_H = H[non_zero_mask]
    E = torch.sum(non_zero[:, None]*non_zero_H, dim=0)/weight

    return E[mask], torch.var(E), E, weight

def ising_hamiltonian(N, M, J, L):
    import netket.hilbert as hb
    from netket.operator.spin import sigmax,sigmaz
    # From https://netket.readthedocs.io/en/v3.11.4/tutorials/gs-ising.html
    hi = hb.Spin(s=1 / 2, N=N*M)
    H = sum([L*sigmax(hi,i) for i in range(N)])
    if M == 1:
        H += sum([J*sigmaz(hi,i)*sigmaz(hi,(i+1)%N) for i in range(N)])
    #--------------------------------------------------------------------
    else:
        for _ in range(M):
            for i in range(N):
                H += J*sigmaz(hi,N*i)*sigmaz(hi,(N*i+1)%N)
                H += J*sigmaz(hi,N*i)*sigmaz(hi, (N*i+N)%(N*M))
    return np.array(H.to_dense())

#Pairing

def pairing_cutoff(P, dtype, device):
    basis, diff = create_basis(P, dtype, device)
    mask = torch.sum(basis, dim=-1) == 2*P
    basis = basis[mask]
    return basis, diff

def pairing_local_energy(eps, g, samples, basis):
    basis, diff_basis = basis
    weight, non_zero_mask, mask = amplitudes(samples, basis)
    non_zero = weight[non_zero_mask]
    non_zero_diff = diff_basis[non_zero_mask]

    H_0 = 2*eps*torch.sum(torch.where(basis==1)[0], dim=-1)
    H_eps = torch.zeros_like(H_0)
    H_eps[non_zero_mask] = H_0[non_zero_mask]


    H_1 = 0.5*g*torch.sum(non_zero[:, None]*(non_zero_diff==1), dim=0)/weight
    H_V = H_1[mask]

    E = H_eps - H_V
    return E

if __name__ == "__main__":
    basis = create_basis(2, torch.float64, torch.device('cuda'))
    ising = ising_basis_set(2, 1, -0.3, 0.2, basis)
    eq = 7
    op = 20
    samples = torch.tensor([[1,1]]*op + [[0,1]]*eq + [[1,0]]*eq + [[0,0]]*op, dtype=torch.float64, device=torch.device('cuda'))
    energy = ising_local(-0.3, 0.2, samples, ising)
    print(energy[0])
    print(torch.mean(energy[0]))
    #print(netket_ising_true(2, 1, -0.3, 0.2))
    #a = torch.tensor([[1, 1, 0, 0], [1, 0, 1, 0]])
    eps = 2; V = -1/3; W = -1/4
    print(lipkin_true(2, eps, V, W))
    print(basis[1])



