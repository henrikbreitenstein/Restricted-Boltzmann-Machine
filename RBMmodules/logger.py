import os

def create_log_file(model):
    
    log = ''
    for element in model:
        log += f'{element.name} : {element}'

    return log

def log_model(run_name, model):
    os.mkdir(f'./run_logs/{run_name}')

    log_name = f'./run_logs/{run_name}/{run_name}_log.txt'

    with open(log_name, 'w+') as f:
        f.write(create_log_file(model))
    
    model_arrays = [model.visual_bias, model.hidden_bias, model.W]
    for array in model_arrays:
        save(array)


