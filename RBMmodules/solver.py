import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from RBMmodules import hamiltonian
S = lambda x: 1/(1 + torch.exp(-x))

def net_Energy(v, a, h, b, W):
    energy = -(v@a.T) - (h@b.T) - ((v@W)[:, None, :]@h[:, :, None])[:, :, 0].T
    energy = torch.squeeze(energy)
    return energy

def sample_visual(hidden, visual_bias, W):
    given = S(hidden@W.T + visual_bias)
    binary = torch.bernoulli(given)
    return (given, binary)

def sample_hidden(visual, hidden_bias, W):
    given = S(visual@W + hidden_bias)
    binary = torch.bernoulli(given)
    return given, binary

def gibbs_update(input, visual_bias, hidden_bias, W, k, hidden=None):
    given_h, hidden = sample_hidden(input, hidden_bias, W)
    given_v, visual_k = sample_visual(given_h, visual_bias, W)
    for _ in range(k):
        given_h, hidden_k = sample_hidden(given_v, hidden_bias, W)
        given_v, visual_k = sample_visual(given_h, visual_bias, W)

    return hidden_k, (given_v, visual_k)

def metropolis_hastings(input, visual_bias, hidden_bias, W, hidden=None):
    
    size = input.size(dim=0)
    #given_h, hidden = sample_hidden(input, hidden_bias, W)
    rand_nums = torch.rand(size)
    E_all = net_Energy(input, visual_bias, hidden, hidden_bias, W)
    E_all = E_all.cpu()
    for i in range(1, input.size(dim=0)):
        acc_ratio = E_all[i]/E_all[i-1]
        if acc_ratio <= rand_nums[i]:
            input[i] = input[i-1]
            E_all[i] = E_all[i-1]
    return hidden, (input, torch.bernoulli(input))

def MonteCarlo(cycles, H, masking_func, gibbs_k, model, basis, binary_gaus=1):
 
    vb = model.visual_bias
    hb = model.hidden_bias
    W = model.weights

    hidden_n = hb.size(dim=0)
    p_visual = partial(sample_visual, visual_bias=vb, W=W)

    if gibbs_k == 0:
        p_sampler = partial(
            metropolis_hastings,
            visual_bias=vb,
            hidden_bias=hb,
            W=W
        )
    else:
        p_sampler = partial(
            gibbs_update,
            visual_bias=vb,
            hidden_bias=hb,
            W=W,
            k=gibbs_k
        )

    hidden = torch.rand(
        cycles,
        hidden_n,
        dtype=model.precision,
        device=model.device
    )
    samples = p_visual(hidden)[1]
    hidden, visual = p_sampler(samples, hidden=hidden)
    dist_s = visual[binary_gaus]

    dPsidvb = (dist_s- vb)
    dPsidhb = 1/(torch.exp(-hb-dist_s@W)+ 1)
    dPsidW  = dist_s[:, :, None]*dPsidhb[:, None, :]

    amplitudes = masking_func(dist_s)
    E_local, basis_var, E, weight = hamiltonian.local_energy(H, amplitudes)
    E_mean = torch.mean(E_local)
    E_diff = E_local - E_mean

    DeltaVB = 2*torch.mean(E_diff[:, None]*dPsidvb, axis=0)
    DeltaHB = 2*torch.mean(E_diff[:, None]*dPsidhb, axis=0)
    DeltaW  = 2*torch.mean(E_diff[:, None, None]*dPsidW, axis=0)

   # dPsidvb = (basis-vb)
   # dPsidhb = 1/(torch.exp(-hb-basis@W)+ 1)
   # dPsidW  = basis[:, :, None]*dPsidhb[:, None, :]

   # amplitudes = masking_func(dist_s)
   # E_local, basis_var, E, weight = hamiltonian.local_energy(H, amplitudes)
   # E_diff = E - torch.mean(E[weight!=0])
   # E_diff[weight==0] = 0

   # DeltaVB = torch.mean(E_diff[:, None]*dPsidvb, axis=0)
   # DeltaHB = torch.mean(E_diff[:, None]*dPsidhb, axis=0)
   # DeltaW  = torch.mean(E_diff[:, None, None]*dPsidW, axis=0)


    dE = torch.mean(E_local - E_mean)
    stats = {
        'E_mean'  : E_mean,
        'dE'      : dE,
        'vb'      : vb,
        'hb'      : hb,
        'dvb'     : DeltaVB,
        'dhb'     : DeltaHB,
        'variance': torch.var(E_local),
        'part_var': basis_var,
        'E'       : E,
        'amps'    : weight**2
    }

    return DeltaVB, DeltaHB, DeltaW, stats


from typing import TypedDict
class stats_dict(TypedDict):
    E_mean   : torch.Tensor
    dE       : torch.Tensor
    vb       : torch.Tensor
    hb       : torch.Tensor
    dvb      : torch.Tensor
    dhb      : torch.Tensor
    variance : torch.Tensor
    part_var : torch.Tensor
    E        : torch.Tensor
    amps     : torch.Tensor

def find_min_energy(
    model,
    H,
    masking_func,
    cycles,
    gibbs_k,
    epochs,
    learning_rate,
    adapt,
    binary_gaus=1,
    verbose = False) -> stats_dict:
    vn = len(model.visual_bias)
    hn = len(model.hidden_bias)
    stats_array: stats_dict = {
        'E_mean'   : torch.zeros(epochs),
        'dE'       : torch.zeros(epochs),
        'vb'       : torch.zeros((epochs, vn)),
        'hb'       : torch.zeros((epochs, hn)),
        'dvb'      : torch.zeros((epochs, vn)),
        'dhb'      : torch.zeros((epochs, hn)),
        'variance' : torch.zeros(epochs),
        'part_var' : torch.zeros(epochs),
        'E'        : torch.zeros((epochs, H.shape[0])),
        'amps'     : torch.zeros((epochs, H.shape[0]))
    }
    
    basis = hamiltonian.create_basis(
        len(model.visual_bias),
        dtype = model.precision,
        device = model.device
    )

    adapt_func = adapt['func']
    vv = prev_vv = adapt['gamma']
    hv = prev_hv = adapt['gamma']
    wv = prev_wv = adapt['gamma']
    adapt_v = learning_rate
    adapt_h = learning_rate
    adapt_w = learning_rate

    visual_prev = torch.zeros_like(model.visual_bias)
    hidden_prev = torch.zeros_like(model.hidden_bias)
    weights_prev = torch.zeros_like(model.weights)

    for n in tqdm(range(epochs)):
        DeltaVB, DeltaHB, DeltaW, stats = MonteCarlo(
            cycles,
            H,
            masking_func,
            gibbs_k,
            model,
            basis,
            binary_gaus=binary_gaus,
        )

        for key, stat in stats.items():
            stats_array[key][n] = stat

        if n > 1:
            adapt_v, adapt_h, adapt_w, vv, hv, wv = adapt_func(
                learning_rate = learning_rate,
                prev_vv       = prev_vv,
                prev_hv       = prev_hv,
                prev_wv       = prev_wv,
                prev_vgrad    = visual_prev,
                prev_hgrad    = hidden_prev,
                prev_wgrad    = weights_prev,
                gamma         = adapt['gamma']
            )

        model.visual_bias -= adapt_v*DeltaVB
        model.hidden_bias -= adapt_h*DeltaHB
        model.weights     -= adapt_w*DeltaW

        visual_prev  = DeltaVB
        hidden_prev  = DeltaHB
        weights_prev = DeltaW
        prev_vv      = vv
        prev_hv      = hv
        prev_wv      = wv
 
    return stats_array
