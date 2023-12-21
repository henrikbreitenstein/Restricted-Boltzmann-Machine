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

def MonteCarlo(cycles, local_energy_func, gibbs_k, model, uniform_s):
    
    visual_n = model.visual_bias.size(dim=0)
    hidden_n = model.hidden_bias.size(dim=0)
    vb = model.visual_bias
    hb = model.hidden_bias
    W = model.weights
    p_visual = partial(sample_visual, visual_bias=vb, W=W)
    p_hidden = partial(sample_hidden, hidden_bias=hb, W=W)
    p_gibbs = partial(gibbs_update, visual_bias=vb, hidden_bias=hb, W=W, k=gibbs_k)

    hidden = torch.bernoulli(torch.rand(cycles, hidden_n, dtype=model.precision, device=model.device))
    _, samples = p_visual(hidden)
    hidden, dist_s = p_gibbs(samples)
    
    probs = net_Energy(dist_s, vb, hidden, hb, W)

    dPsidvb = 0.5*(dist_s - vb)
    dPsidhb = 0.5/(2*(hidden + 1))
    dPsidW = 0.5*torch.matmul(dist_s[:, :, None], dPsidhb[:, None, :])
    
    uniform_s = torch.bernoulli(torch.rand(cycles, visual_n, dtype=model.precision, device=model.device))
    E_local = local_energy_func(dist_s, uniform_s)
    E_mean = torch.mean(E_local)

    DeltaVB = torch.mean((E_local*dPsidvb.T).T, axis=0) - E_mean*torch.mean(dPsidvb, axis=0)
    DeltaHB = torch.mean((E_local*dPsidhb.T).T, axis=0) - E_mean*torch.mean(dPsidhb, axis=0)
    DeltaW = torch.mean((E_local*dPsidW.T).T, axis=0) - E_mean*torch.mean(dPsidW, axis=0)

    dE = torch.mean(E_local - E_mean)
    stats = {'E_mean' : E_mean,
            'dE' : dE}
    
    return DeltaVB, DeltaHB, DeltaW, stats

def find_min_energy(model, local_energy_func, cycles, gibbs_k, epochs, learning_rate):

    stats_array = {'E_mean' : torch.zeros(epochs),
                    'dE' : torch.zeros(epochs),
                    'Dist' : []}
    visual_n = model.visual_bias.size(dim=0)
    uniform_s = torch.bernoulli(torch.rand(cycles, visual_n, dtype=model.precision, device=model.device))
    #uniform_s = torch.tensor([[0,0]], dtype=model.precision, device=model.device)
    for n in tqdm(range(epochs)):
        #print('--------------', n, '-------------------')
        DeltaVB, DeltaHB, DeltaW, stats = MonteCarlo(cycles, local_energy_func, gibbs_k, model, uniform_s)

        for key, stat in stats.items():
            stats_array[key][n] = stat

        model.visual_bias -= learning_rate*DeltaVB
        model.hidden_bias -= learning_rate*DeltaHB
        model.weights -= learning_rate*DeltaW
    
    N_test = 100000
    hidden = torch.bernoulli(torch.rand(N_test, model.hidden_bias.shape[0], dtype=model.precision, device=model.device))
    all_states = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=model.precision, device=model.device)
    hidden_g, visual_g = gibbs_update(sample_visual(hidden, model.visual_bias, model.weights)[1], 
                                      model.visual_bias, model.hidden_bias, model.weights, 100)
    
    for state in all_states:
        stats_array['Dist'].append(torch.sum(torch.all(visual_g==state, dim=1)).item()/N_test)

    return stats_array











    




