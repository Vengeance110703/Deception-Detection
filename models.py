import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from torch.hub import load_state_dict_from_url

################## Activation Function ##################
class Mish(nn.Module):
    def __init__(self):
        super().__init__()
        # print("Mish activation loaded...")
    
    def forward(self, x):
        x = x *(torch.tanh(F.softplus(x)))
        return x

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        # print("Swish activation loaded...")
    
    def forward(self, x, beta = 1):
        x = x * F.sigmoid(beta * x)
        return x

class Auto_swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))
