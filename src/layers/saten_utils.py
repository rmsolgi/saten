import torch
from tensorlearn.tensor_geometry import tensor_geometry_graph as tgg
from tensorlearn.decomposition.factorization import tt_factorization as tt
import os
import pickle



def tt_to_tensor(factors):
    tensor = factors[0]
    for i in range(1, len(factors)):
        tensor = torch.tensordot(tensor, factors[i], [[-1], [0]])
        squeezed_tensor = tensor.squeeze(0).squeeze(-1)
    return squeezed_tensor


def get_tensorized_layer_balanced_shape(in_feature,out_feature,in_order,out_order):
    shapes_list=tgg.dyadic_cartesian(in_feature, out_feature, in_order, out_order)
    optimal_shape = min(shapes_list, key=sum)
    return optimal_shape

def get_tt_factors(matrix, tensor_shape, error):
    tensor=matrix.view(tensor_shape)
    decomp=tt(tensor.cpu().numpy(), error)
    tt_factors=decomp.factors
    factors=[torch.from_numpy(f) for f in tt_factors]
    ranks=decomp.rank
    return factors, ranks



def create_m_n_sparsity_mask(error, m, n):
    if error.shape[0] % n != 0:
        raise ValueError(f"The number of rows ({error.shape[0]}) must be divisible by n={n}.")
    
    num_groups = error.shape[0] // n
    error_reshaped = error.view(num_groups, n, -1)  
    
    top_m_indices = torch.topk(torch.abs(error_reshaped), m, dim=1, largest=True).indices  # Apply along rows
    
    mask = torch.zeros_like(error_reshaped)
    
    mask.scatter_(1, top_m_indices, 1.0)
    
    mask = mask.view_as(error)
    
    return mask



def count_params(in_features, out_features, factors, mask, bias = None):
        sparsity_count=mask.sum()
        factors_count=0
        for f in factors:
            factors_count+=f.numel()
        total_count=factors_count+sparsity_count
        if bias != None:
            total_count+=bias.numel()
        base_total=in_features*out_features
        if bias != None:
            base_total+=bias.numel()
        ratio=total_count/base_total
        return total_count, ratio
        