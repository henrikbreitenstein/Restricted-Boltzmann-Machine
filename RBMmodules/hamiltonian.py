import torch
import numpy as np
#import netket as nk
#from netket.operator.spin import sigmax,sigmaz

def pairs(N):
    return N*(N-1)/2


# def lipkin_local(dist_s, eps, V, W):

#     shuffle = dist_s[torch.randperm(dist_s.shape[0])]
#     H = torch.zeros_like(dist_s[:, 0])
    
#     mask_0 = torch.where(torch.all(dist_s == shuffle, dim=1))
#     H[mask_0] = 0.5*eps*(torch.sum(dist_s[mask_0] == 1, dim=1) - torch.sum(dist_s[mask_0] == 0, dim=1)).to(dtype=torch.float64)
    
#     mask_1 = torch.where(abs(torch.sum(dist_s-shuffle, dim=-1)) == 2)
#     H[mask_1] = -V
    
#     return H


def lipkin_local(dist_s, eps, V, W):

    size = dist_s.shape[0]
    unique, weight = torch.unique(dist_s, dim=0, return_counts=True)
    weight = torch.sqrt(weight/size)
    mask_u = torch.where(torch.all(dist_s[:, None] == unique, dim = -1))[1]
    
    H_0 = 0.5*eps*(torch.sum(unique == 1, dim=-1)- torch.sum(unique == 0, dim = -1))
    H_1 = V*torch.sum(weight[:, None]*(abs(torch.sum((unique[:, None] - unique), dim=-1)) == 2), dim=0)/weight
    E = (H_0[mask_u] - H_1[mask_u])
    return E

def ising_local(dist_s, uniform_s, J, L):
    pm = 2*dist_s - 1
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