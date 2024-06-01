from numpy.ma import where
import torch
import numpy as np
import itertools
from scipy.sparse.linalg import eigsh
from functools import partial

from torch.cuda import is_available

def ground_state_vec(H):
    vecs = [np.linalg.eig(H)[1][:, i] for i in range(len(H))]
    eig = []
    for vec in vecs:
        eig.append(((vec@H@vec)/(vec@vec), vec**2))

    eig = np.array(eig, dtype=object)
    argmin = np.argmin(eig[:, 0])
    return eig[argmin]

def ground_state(H):
    eigvals = np.linalg.eigvalsh(H)
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
    all = torch.all(samples[:, None] == basis, dim = -1)
    weight = torch.sum(all, dim=0)
    non_zero_mask = torch.where(weight>0)
    weight = torch.sqrt(weight/size)
    weight[weight==0] = 1/size
    mask = torch.where(all)[1] 
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
    weight = torch.sqrt(weight/size)
    non_zero_mask = torch.where(weight>0)
    weight[weight==0] = np.sqrt(1/size)
    mask = torch.where(samples_m[:, None] == basis_m)[1]
    
    return weight, non_zero_mask, mask


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
        for i in range(M):
            for j in range(N):
                H += J*sigmaz(hi,N*i+j)*sigmaz(hi,(N*i+j+1)%N)
                H += J*sigmaz(hi,N*i+j)*sigmaz(hi, (N*i+j+N)%(N*M))
    return np.array(H.to_dense())

#Heisenberg

def heisen_amps(samples, basis):
    print(samples)
    total = torch.sum(samples, dim=-1)
    spin = 2*samples-1
    binary = spin >= 0
    weighted = total[:, None]*torch.all(binary[:, None] == basis, dim=-1)
    weight = torch.sum(weighted, dim=0)
    weight = weight/samples.shape[0]
    weight = torch.sqrt(weight)
    non_zero_mask = torch.where(weight>0)
    weight[weight==0] = 1/samples.shape[0]
    mask = torch.where(torch.all(binary[:, None] == basis, dim = -1))[1]

    return weight, non_zero_mask, mask

def heisen_hamiltonian(J, L, N, M):
    import netket.hilbert as hb
    from netket.operator.spin import sigmax, sigmay, sigmaz
    hi = hb.Spin(s=1/2, N=N*M)
    H = sum([(L+0j)*sigmax(hi,i) for i in range(N)])
    if M == 1:
        for i in range(N):
            H += J*sigmax(hi, i)*sigmax(hi, (i+1)%N)
            H += J*sigmay(hi, i)*sigmay(hi, (i+1)%N)
            H += J*sigmaz(hi, i)*sigmaz(hi, (i+1)%N)
    else:
        for i in range(M):
            for j in range(N):
                H += J*sigmax(hi, N*i+j)*sigmax(hi, (N*i+j+1)%N)
                H += J*sigmax(hi, N*i+j)*sigmax(hi, (N*i+j+N)%(N*M))
                H += J*sigmay(hi, N*i+j)*sigmay(hi, (N*i+j+1)%N)
                H += J*sigmay(hi, N*i+j)*sigmay(hi, (N*i+j+N)%(N*M))
                H += J*sigmaz(hi, N*i+j)*sigmaz(hi, (N*i+j+1)%N)
                H += J*sigmaz(hi, N*i+j)*sigmaz(hi, (N*i+j+N)%(N*M))
    return np.array(H.to_dense().real)

#Pairing

def pairing_hamiltonian(basis, n, eps, g):
    mask_non = torch.where(torch.sum(basis, dim=-1) != n)[0]
    where = torch.where(basis==1)
    B_0 = torch.zeros_like(basis[:, 0])
    for i, a_one in enumerate(where[0]):
        B_0[a_one] += where[1][i]

    diff = torch.sum(
        torch.logical_xor(basis[:, None], basis),
        dim=-1
    )
    B_1 = -0.5*g*(diff==2)
    H = eps*torch.diagflat(B_0) + B_1
    for i in mask_non:
        H[:, i] = 0
        H[i, :] = 0
    return np.array(H.cpu())

if __name__ == "__main__":
    basis = create_basis(3, torch.float64, torch.device('cuda'))
    print(basis)
    print(pairing_hamiltonian(basis, 3, 2, 1, 1))



