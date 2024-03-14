def nop(**kwargs):
    return kwargs['learning_rate']

def deminishing_linear(**kwargs):
    update_lr = 1 - (kwargs['n']/kwargs['epochs'])
    return update_lr*kwargs['learning_rate']
