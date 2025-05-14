import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from . import utils
from . import saten_utils
import pickle
import math

from . import templates


class SatenTT(templates.SatenTemplate):
    def __init__(self, in_features, out_features, dim_list, rank_list, mask=None, bias=True):
        super(SatenTT, self).__init__()
        
        self.rank_list = list(rank_list)
        
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        self.dim_list = list(dim_list)
        self.dim = len(dim_list)

        self.weight=nn.Parameter(torch.Tensor(self.out_features, self.in_features))

        if mask != None:
            self.register_buffer('mask', mask)
            self.factors = nn.ParameterList([
                nn.Parameter(torch.randn(self.rank_list[i], self.dim_list[i], self.rank_list[i + 1]).contiguous())
                for i in range(self.dim)])
            self.get_weight = self.get_w_from_factors
        else:
            self.register_buffer('mask', None)
            self.register_parameter('factors', None)
            self.get_weight = self.get_w

        # Initialize bias if needed
        if self.use_bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)

        # self.update_params_dict()

    def get_w_from_factors(self):
        weight_low_rank=saten_utils.tt_to_tensor(self.factors).view(self.in_features,self.out_features).t()
        weight_sparse=self.weight*self.mask
        weight=weight_sparse+weight_low_rank
        return weight

    def get_w(self):
        return self.weight


    def forward(self, x):

        device = x.device
        weight = self.get_weight().to(device)
        out = x @ weight.t()
        if self.use_bias:
            out += self.bias
        return out

    @classmethod
    def from_teacher(cls, teacher_layer, config):
        if config.mode!='teacher':
            raise ValueError('config mode must be teacher')
        if not isinstance(teacher_layer, nn.Linear):
            raise ValueError('The teacher_layer must be a Linear layer')

        device = teacher_layer.weight.device

        weight=teacher_layer.weight.detach()#.to(device)
        weight=weight.t()

        if teacher_layer.bias!=None:
            bias=teacher_layer.bias.detach()#.to(device)
        else:
            bias=None

        if bias==None:
            use_bias=False
        else:
            use_bias=True
        in_order = config.in_order
        out_order = config.out_order
        decomp_error = config.decomp_error
        weight_shape = weight.size()
        in_features = weight_shape[0]
        out_features = weight_shape[1]
        dim_list = saten_utils.get_tensorized_layer_balanced_shape(in_features,out_features,in_order,out_order)
        factors, rank_list=saten_utils.get_tt_factors(weight,dim_list,decomp_error)

        tensor_hat=saten_utils.tt_to_tensor(factors)
        weight_hat_lr=tensor_hat.view(in_features,out_features)
        device=weight.device
        error=weight-weight_hat_lr.to(device)

        if config.sparsity=='top_prune':
            flatten_error = torch.abs(error).flatten()
            top_n_threshold = torch.kthvalue(flatten_error, int(len(flatten_error) * (config.sparsity_ratio[0])))[0]
            mask = torch.where(torch.abs(error) < top_n_threshold, torch.tensor(0.0), torch.tensor(1.0)).t().contiguous()
        elif config.sparsity=='m2n':
            mask=saten_utils.create_m_n_sparsity_mask(error, int(config.sparsity_ratio[0]), int(config.sparsity_ratio[1])).t().contiguous()
        else:
            raise ValueError('sparsity is not supported')
        
        total_count, ratio = saten_utils.count_params(in_features, out_features, factors, mask, bias)

        if config.freez:
            weight_low_rank=saten_utils.tt_to_tensor(factors).view(in_features,out_features).t()
            weight_sparse=error.t()*mask
            weight_hat=weight_sparse+weight_low_rank
            mask = None
            instance=cls(in_features, out_features, dim_list, rank_list, mask, use_bias)
            instance.weight = nn.Parameter(weight_hat.contiguous(), requires_grad=False)
        else:
            instance=cls(in_features, out_features, dim_list, rank_list, mask, use_bias)
            instance.factors = nn.ParameterList([nn.Parameter(f.contiguous()) for f in factors])
            instance.weight = nn.Parameter(error.t().contiguous())

        if use_bias:
            instance.bias=nn.Parameter(bias.contiguous())
        # instance.update_params_dict()
        instance.saten_params = total_count
        instance.compression_ratio = ratio
        return instance

    @classmethod
    def from_file(cls, model_params, config):
        params=model_params[config.layer_name]
        bias = params['bias']
        if bias!=None:
            use_bias = True 
        else:
            use_bias = False
        instance=cls(params['in_features'], 
                    params['out_features'], 
                    params['dim_list'], 
                    params['rank_list'], 
                    params['mask'], 
                    use_bias)
        instance.weight = nn.Parameter(params['weight'].contiguous())
        if use_bias:
            instance.bias=nn.Parameter(params['bias'].contiguous())
        if not config.freez:
            instance.factors = nn.ParameterList([nn.Parameter(f.contiguous()) for f in params['factors']])
        else:
            instance.weight.requires_grad = False

        # instance.update_params_dict()
        instance.saten_params = params['num_params']
        instance.compression_ratio = params['params_ratio']
        return instance


