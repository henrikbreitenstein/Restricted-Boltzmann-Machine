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

def construct_lipkin(S, eps, V):
    
    J_z, J_pluss, J_minus = construct_spin(S)[:3]

    H = eps*J_z - 1/2*V*(J_pluss@J_pluss + J_minus@J_minus)

    return H

def construct_ising(N, M):
    
    J_z, _, _, J_x, _ = construct_spin(0.5)
    I = np.eye(2)
    right_list = []
    up_list = []
    single_list = []
    for i in range(M):
        for j in range(N):
            mat_list_right = [I]*N*M
            mat_list_up = [I]*N*M
            mat_list_single = [I]*N*M

            mat_list_right[N*i+j] = J_x
            mat_list_right[N*i+(j+1)%N] = J_x

            mat_list_up[N*i+j] = J_x
            mat_list_up[N*((i+1)%M)+j] = J_x
            
            mat_list_single[N*i+j] = J_z

            right = mat_list_right[0]
            up = mat_list_up[0]
            single = mat_list_single[0]

            for mat_right in mat_list_right[1:]:
                right = np.kron(right, mat_right)
            for mat_up in mat_list_up[1:]:
                up = np.kron(up, mat_up)
            for mat_single in mat_list_single[1:]:
                single = np.kron(single, mat_single)
            
            right_list.append(right)
            up_list.append(up)
            single_list.append(single)
    
    return right_list, up_list, single_list

def lipkin_true(n, eps, V):
    H = torch.tensor(construct_lipkin(n/2, eps, V))
    eigvals = torch.real(torch.linalg.eigvals(H))
    min_eigval = torch.min(eigvals)
    return min_eigval

def create_basis(n, dtype, device):
    basis = torch.tensor(
        list(itertools.product([0, 1], repeat=n)),
        dtype = dtype,
        device= device
    )
    return basis

def pairing_cutoff(basis, P):
    mask = torch.sum(basis, dim=-1) == 2*P
    pairing_basis = basis[mask]

    return pairing_basis

def amplitudes(samples, basis):
    weight = torch.sum(torch.all(samples[:, None] == basis, dim = -1), dim=0)
    non_zero_weight = weight[weight!=0]
    weight[weight==0] = 1
    mask = torch.where(torch.all(samples[:, None] == basis, dim = -1))[1]
    
    diff_basis = torch.sum(
        torch.logical_xor(basis[:, None], basis), dim=-1
    )[weight!=0]

    return weight, non_zero_weight, mask, diff_basis

def lipkin_local(eps, V, W, samples, basis):
    weight, non_zero_weight, mask, diff_basis = amplitudes(samples, basis)

    N_0 = torch.sum(basis == 0, dim=-1)
    N_1 = torch.sum(basis == 1, dim=-1)

    diff_basis = torch.sum(
        torch.logical_xor(basis[:, None], basis), dim=-1
    )[weight!=0]
    
    H_0 = 0.5*eps*(N_1-N_0)
    H_eps = H_0[mask]

    H_1 = V*torch.sum(non_zero_weight[:, None]*(diff_basis == 2), dim=0)/weight
    H_V = H_1[mask]
    
    H_2 = W*torch.sum(non_zero_weight[:, None]*(diff_basis == 1), dim=0)/weight
    H_W = H_2[mask]
    
    E = (H_eps - H_V - H_W)
    return E

def ising_1d(samples, J, L):
    pm = 2*samples - 1
    shift_r = torch.roll(pm, 1)
    H_J = J*torch.sum(pm*shift_r)
    H_L = L*torch.sum(pm)
    return H_L + H_J

def ising_1d_true(N, J, L):
    
    right, up, single = construct_ising(N, 1)
    H = np.zeros((2**N, 2**N))
    for mat in right:
        H += J*mat
    for mat in up:
        H += J*mat
    for mat in single:
        H += L*mat
    eig_vals = np.linalg.eigvalsh(H)
    return min(eig_vals)

def ising_2d(samples, J, L):
    
    pm = 2*samples - 1
    size = int(torch.sqrt(pm.shape)[0])
    pm = torch.reshape(pm, (size, size))

    up = torch.roll(pm, 1, 0)
    right = torch.roll(pm, 1, 1)

    H_J = J*torch.sum(up*pm + right*pm)
    H_L = L*torch.sum(pm)

    return H_L + H_J

def pairing_local_energy(eps, g, samples, basis):
    
    weight, non_zero_weight, mask, diff_basis = amplitudes(samples, basis)
    
    H_0 = 2*eps*torch.sum(torch.where(basis==1), dim=-1)
    H_eps = H_0[mask]

    H_1 = 0.5*g*torch.sum(non_zero_weight[:, None]*(diff_basis==1), dim=0)/weight
    H_V = H_1[mask]

    E = H_0 - H_V
    return E

if __name__ == "__main__":
    for N in range(2, 10):
        print(ising_1d_true(N, 1, 0.1))
