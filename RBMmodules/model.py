import torch
from dataclasses import dataclass


def get_tensortype(device, dtype):
    if device:
        if (dtype == torch.float64) or (dtype==torch.double):
            return torch.cuda.DoubleTensor
        elif (dtype == torch.float32) or (dtype==torch.float):
            return torch.cuda.FloatTensor
        elif (dtype == torch.float16) or (dtype==torch.half):
            return torch.cuda.HalfTensor     
        else:
            return None
    else:
        if (dtype == torch.float64) or (dtype==torch.double):
            return torch.DoubleTensor
        elif (dtype == torch.float32) or (dtype==torch.float):
            return torch.FloatTensor
        elif (dtype == torch.float16) or (dtype==torch.half):
            return torch.HalfTensor     
        else:
            return None

def create_model_dataclass(precision, device):        
    tensortype = get_tensortype(device, precision)
    @dataclass
    class model:
        visual_bias: tensortype
        hidden_bias: tensortype
        weights: tensortype
        device: torch.device
        precision: torch.dtype
    return model

def set_up_model(visual_n, hidden_n, precision, device, W_scale):
    visual_bias = torch.zeros(visual_n, dtype=precision, device=device)
    hidden_bias = torch.zeros(hidden_n, dtype=precision, device=device)
    W = W_scale*torch.rand(visual_n, hidden_n, dtype=precision, device=device)
    
    model = create_model_dataclass(precision, device)

    init_model = model(
        visual_bias = visual_bias,
        hidden_bias = hidden_bias,
        weights = W,
        device = device, 
        precision = precision)
    
    return init_model

