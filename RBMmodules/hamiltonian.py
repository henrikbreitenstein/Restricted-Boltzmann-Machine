import torch
import numpy as np
#import netket as nk
#from netket.operator.spin import sigmax,sigmaz

def lipkin_local(eps, V, W, samples):

    size = samples.shape[0]
    unique, weight = torch.unique(samples, dim=0, return_counts=True)
    weight = torch.sqrt(weight/size)
    mask = torch.where(torch.all(samples[:, None] == unique, dim = -1))[1]
    
    N_0 = torch.sum(unique == 0, dim=-1)
    N_1 = torch.sum(unique == 1, dim=-1)

    diff_unique = abs(torch.sum(unique[:, None] - unique), dim=-1)
    diff_N1 = abs(N_1[:, None] - N_1)

    H_0 = 0.5*eps*(N_1-N_0)
    H_eps = H_0[mask]
    
    one_pair = np.bitwise_and(diff_unique==2, diff_N1==2)
    H_1 = V*torch.sum(weight[:, None]*one_pair, dim=0)/weight
    H_V = H_1[mask]
    
    H_2 = W*torch.sum(weight[:, None]*(diff_unique == 1), dim=0)/weight
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
