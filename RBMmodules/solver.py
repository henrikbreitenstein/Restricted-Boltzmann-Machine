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
    return given, binary

def sample_hidden(visual, hidden_bias, W):
    given = S(visual@W + hidden_bias)
    binary = torch.bernoulli(given)
    return given, binary

def gibbs_update(input, visual_bias, hidden_bias, W, k):
    given_h, hidden = sample_hidden(input, hidden_bias, W)
    given_v, visual_k = sample_visual(hidden, visual_bias, W)
    for _ in range(k):
        given_h, hidden_k = sample_hidden(given_v, hidden_bias, W)
        given_v, visual_k = sample_visual(given_h, visual_bias, W)
    
    return hidden_k, visual_k

def metropolis_hastings(input, visual_bias, hidden_bias, W):
    
    size = input.size(dim=0)
    given_h, hidden = sample_hidden(input, hidden_bias, W)
    rand_nums = torch.rand(size)
    previous = input[0]
    E_prev = net_Energy(previous, visual_bias, hidden, hidden_bias, W)
    for i in range(input.size(dim=0)):
        E_current = net_Energy(input[i], visual_bias, hidden, hidden_bias, W)
        acc_ratio = E_current/E_prev
        if acc_ratio <= rand_nums[i]:
            input[i] = previous
        previous = input[i]
        E_prev = E_current

    return input

def MonteCarlo(cycles, H, masking_func, gibbs_k, model):
    
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

    hidden = torch.bernoulli(torch.rand(
        cycles,
        hidden_n,
        dtype=model.precision,
        device=model.device
    ))
    
    _, samples = p_visual(hidden)
    hidden, dist_s = p_sampler(samples)

    dPsidvb = (dist_s - vb)
    dPsidhb = 1/(torch.exp(-hb-dist_s@W)+ 1)
    dPsidW  = dist_s[:, :, None]*dPsidhb[:, None, :]
    
    amplitudes = masking_func(dist_s)
    E_local, basis_var, E, weight = hamiltonian.local_energy(H, amplitudes)
    E_mean = torch.mean(E_local)
    E_diff = E_local - E_mean

    DeltaVB = torch.mean(E_diff[:, None]*dPsidvb, axis=0)
    DeltaHB = torch.mean(E_diff[:, None]*dPsidhb, axis=0)
    DeltaW  = torch.mean(E_diff[:, None, None]*dPsidW, axis=0)

    dE = torch.mean(E_local - E_mean)
    stats = {'E_mean' : E_mean,
            'dE'      : dE,
            'variance': torch.var(E_local),
            'part_var': basis_var,
            }
    
    return DeltaVB, DeltaHB, DeltaW, stats, E, weight

def find_min_energy(
    model,
    H,
    masking_func,
    cycles,
    gibbs_k,
    epochs,
    learning_rate,
    adapt_func):

    stats_array = {'E_mean'   : torch.zeros(epochs),
                   'dE'       : torch.zeros(epochs),
                   'variance' : torch.zeros(epochs),
                   'part_var' : torch.zeros(epochs),
                   'Dist'     : []}
    visual_n = model.visual_bias.size(dim=0)
    
    prev_vv = 1
    prev_hv = 1
    prev_wv = 1

    visual_prev = torch.zeros_like(model.visual_bias)
    hidden_prev = torch.zeros_like(model.hidden_bias)
    weights_prev = torch.zeros_like(model.weights)

    for n in tqdm(range(epochs)):
        DeltaVB, DeltaHB, DeltaW, stats, E, weight = MonteCarlo(
            cycles,
            H,
            masking_func,
            gibbs_k,
            model,
        )
        if n == epochs-1:
            print(E)
            print(weight)
        for key, stat in stats.items():
            stats_array[key][n] = stat

        adapt_v, adapt_h, adapt_w, vv, hv, wv = adapt_func(
            learning_rate = learning_rate,
            prev_vv       = prev_vv,
            prev_hv       = prev_hv,
            prev_wv       = prev_wv,
            prev_vgrad    = visual_prev,
            prev_hgrad    = hidden_prev,
            prev_wgrad    = weights_prev,
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

    N_test = 100000
    hidden = torch.bernoulli(torch.rand(N_test, model.hidden_bias.shape[0], dtype=model.precision, device=model.device))
    import itertools
    all_states = torch.tensor(list(itertools.product([0, 1], repeat=visual_n)), dtype=model.precision, device=model.device)
    hidden_g, visual_g = gibbs_update(sample_visual(hidden, model.visual_bias, model.weights)[1], 
                                      model.visual_bias, model.hidden_bias, model.weights, gibbs_k)
    for state in all_states:
        stats_array['Dist'].append(torch.sum(torch.all(visual_g==state, dim=1)).item()/N_test)

    return stats_array
