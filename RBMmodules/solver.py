import torch
from tqdm import tqdm
from functools import partial
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

def MonteCarlo(cycles, local_energy_func, gibbs_k, model, var_mean_ratio):
    
    vb = model.visual_bias
    hb = model.hidden_bias
    W = model.weights
    visual_n = vb.size(dim=0)
    hidden_n = hb.size(dim=0)
    p_visual = partial(sample_visual, visual_bias=vb, W=W)
    p_hidden = partial(sample_hidden, hidden_bias=hb, W=W)
    p_gibbs = partial(gibbs_update, visual_bias=vb, hidden_bias=hb, W=W, k=gibbs_k)

    hidden = torch.bernoulli(torch.rand(
        cycles, 
        hidden_n, 
        dtype=model.precision, 
        device=model.device
    ))
    
    _, samples = p_visual(hidden)
    hidden, dist_s = p_gibbs(samples)
    
   # uniform_n = int(cycles*var_mean_ratio)
   # dist_s[0:uniform_n] = torch.bernoulli(torch.rand(
   #     uniform_n,
   #     visual_n,
   #     dtype=model.precision,
   #     device=model.device
   # ))

    dPsidvb = (dist_s - vb)
    dPsidhb = 1/(torch.exp(-hb-dist_s@W)+ 1)
    dPsidW = dist_s[:, :, None]*dPsidhb[:, None, :]
    
    E_local = local_energy_func(dist_s)
    E_mean = torch.mean(E_local)
    E_diff = E_local - E_mean

    DeltaVB = torch.mean(E_diff[:, None]*dPsidvb, axis=0)
    DeltaHB = torch.mean(E_diff[:, None]*dPsidhb, axis=0)
    DeltaW = torch.mean(E_diff[:, None, None]*dPsidW, axis=0)

    dE = torch.mean(E_local - E_mean)
    stats = {'E_mean' : E_mean,
            'dE'      : dE,
            'variance': torch.var(E_local)}
    
    return DeltaVB, DeltaHB, DeltaW, stats

def find_min_energy(
    model,
    local_energy_func,
    cycles,
    gibbs_k,
    epochs,
    learning_rate,
    adapt_func):

    stats_array = {'E_mean'   : torch.zeros(epochs),
                   'dE'       : torch.zeros(epochs),
                   'variance' : torch.zeros(epochs),
                   'Dist'     : []}
    visual_n = model.visual_bias.size(dim=0)
    var_mean_ratio = 0

    for n in tqdm(range(epochs)):
        DeltaVB, DeltaHB, DeltaW, stats = MonteCarlo(
            cycles, 
            local_energy_func, 
            gibbs_k, 
            model, 
            var_mean_ratio
        )

        var_mean_ratio = min(0.1, abs((stats['variance']/stats['E_mean']).item()))*(1-n/epochs)**2

        if n/epochs > 0.5:
            var_mean_ratio = 0
        
        for key, stat in stats.items():
            stats_array[key][n] = stat
        
        adapt_lr = adapt_func(
            learning_rate = learning_rate,
            epochs        = epochs,
            n             = n,
        )

        model.visual_bias -= adapt_lr*DeltaVB
        model.hidden_bias -= adapt_lr*DeltaHB
        model.weights -= adapt_lr*DeltaW
    
    N_test = 100000
    hidden = torch.bernoulli(torch.rand(N_test, model.hidden_bias.shape[0], dtype=model.precision, device=model.device))
    import itertools
    all_states = torch.tensor(list(itertools.product([0, 1], repeat=visual_n)), dtype=model.precision, device=model.device)
    hidden_g, visual_g = gibbs_update(sample_visual(hidden, model.visual_bias, model.weights)[1], 
                                      model.visual_bias, model.hidden_bias, model.weights, gibbs_k)
    for state in all_states:
        stats_array['Dist'].append(torch.sum(torch.all(visual_g==state, dim=1)).item()/N_test)

    return stats_array











    




