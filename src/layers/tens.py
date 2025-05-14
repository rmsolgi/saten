from . import templates
from . import utils
from . import saten_utils
import torch.nn as nn
import torch

class Ten(templates.TensorizedTemplate):
    def __init__(self, in_features, out_features, dim_list, rank_list, decomp_format, bias=True):
        super(Ten, self).__init__()
        
        self.rank_list = list(rank_list)
        
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        self.dim_list = list(dim_list)
        self.dim = len(dim_list)
        self.decomp_format = decomp_format


        if self.decomp_format == 'tt':
            self.factors = nn.ParameterList([
                nn.Parameter(torch.randn(self.rank_list[i], self.dim_list[i], self.rank_list[i + 1]).contiguous())
                for i in range(self.dim)])
            self.factors_to_tensor = utils.tt_to_tensor
            self.x_contract_factors = utils.tt_contract_x_precompute_einsum

        elif self.decomp_format == 'tucker':
            core = nn.Parameter(torch.randn(*self.rank_list).contiguous())
            factors = [
                nn.Parameter(torch.randn(self.dim_list[i], self.rank_list[i]).contiguous())
                for i in range(self.dim)
            ]
            self.factors = nn.ParameterList([core] + factors)
            self.factors_to_tensor = utils.tucker_to_tensor
            self.x_contract_factors = self.x_dot_weight

        else:
            raise ValueError('decomp_format is not recognized')


        if self.use_bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)


    def x_dot_weight(self, factors, x):
        weight_low_rank = self.factors_to_tensor(factors).reshape(self.in_features, self.out_features)
        out = x @ weight_low_rank
        return out

    def forward(self, x):
        if self.training:
            weight_low_rank = self.factors_to_tensor(self.factors).reshape(self.in_features, self.out_features)
            device = x.device
            out = x @ weight_low_rank.to(device)
            if self.use_bias:
                out += self.bias
            return out
        else:
            out = self.x_contract_factors(self.factors,x)
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
        in_features = weight_shape[1]
        out_features = weight_shape[0]
        
        if config.decomp_format == 'tt':
            factors, rank_list, dim_list = utils.get_tt(weight.t(), decomp_error, in_order, out_order)

        elif config.decomp_format == 'tucker':
            dim_list = saten_utils.get_tensorized_layer_balanced_shape(in_features,out_features,in_order,out_order)
            factors, rank_list = utils.get_tucker_factors(weight.t(),dim_list,decomp_error)

        instance=cls(in_features, out_features, dim_list, rank_list, config.decomp_format, use_bias)
        instance.factors = nn.ParameterList([nn.Parameter(f.contiguous()) for f in factors])
      
        if use_bias:
            instance.bias=nn.Parameter(bias.contiguous())
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
                    params['decomp_format'], 
                    use_bias)
        instance.factors = nn.ParameterList([nn.Parameter(f.contiguous()) for f in params['factors']])
        
        if use_bias:
            instance.bias=nn.Parameter(params['bias'].contiguous())
        return instance

