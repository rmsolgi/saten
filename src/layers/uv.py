import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers.templates import UVTemplate
from src.layers import utils



class SVDW(UVTemplate):
    def __init__(self, in_features, out_features, rank, bias=True):
        super(SVDW, self).__init__()
        
        self.rank = rank
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        # Initialize U and V matrices
        self.U = nn.Parameter(torch.randn(in_features, rank))
        self.V = nn.Parameter(torch.randn(rank, out_features))

        # Initialize bias if needed
        if self.use_bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        

    @classmethod
    def from_teacher(cls, teacher_layer, config):
        if config.mode!='teacher':
            raise ValueError('config mode must be teacher')
        if not isinstance(teacher_layer, nn.Linear):
            raise ValueError('The teacher_layer must be a Linear layer')

        device = teacher_layer.weight.device

        weight=teacher_layer.weight.detach()#.to(device)
        if teacher_layer.bias!=None:
            bias=teacher_layer.bias.detach()#.to(device)
        else:
            bias=None
        weight=weight.t()
        if bias==None:
            use_bias=False
        else:
            use_bias=True

        U_new, V_new = utils.svd(weight, config.rank)

        instance = cls(teacher_layer.in_features, teacher_layer.out_features, config.rank, bias=use_bias)

        instance.U = nn.Parameter(U_new.to(device))
        instance.V = nn.Parameter(V_new.to(device))

        if use_bias:
            instance.bias = nn.Parameter(bias.to(device))
        return instance

