def nop(**kwargs):
    lr = kwargs['learning_rate']
    return lr, lr, lr, 0, 0, 0


def deminishing_linear(**kwargs):
    update_lr = 1 - (kwargs['n']/kwargs['epochs'])
    return update_lr*kwargs['learning_rate']

def momentum(gamma, **kwargs):
    vv = gamma*kwargs['prev_vv']*kwargs['learning_rate']
    hv = gamma*kwargs['prev_hv']*kwargs['learning_rate']
    wv = gamma*kwargs['prev_wv']*kwargs['learning_rate']
    adapt_v = vv + kwargs['learning_rate']*kwargs['prev_vgrad']
    adapt_h = hv + kwargs['learning_rate']*kwargs['prev_hgrad']
    adapt_w = wv + kwargs['learning_rate']*kwargs['prev_wgrad']
    return adapt_v, adapt_h, adapt_w, vv, hv, wv


