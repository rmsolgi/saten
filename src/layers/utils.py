import torch
from tensorlearn.tensor_geometry import tensor_geometry_graph as tgg
from tensorlearn.decomposition.factorization import tt_factorization as tt
from tensorlearn.decomposition.factorization import tucker_factorization as tucker
import tensorlearn as tl
from src.layers import saten_utils
import tensorly


def nn_tensor_geometry_optimization(weight,config):
    in_order=config.in_order
    out_order=config.out_order
    error=config.error
    weight_shape=weight.size()
    in_feature=weight_shape[0]
    out_feature=weight_shape[1]
    dim_list=config.get_shape(in_feature,out_feature,in_order,out_order)
    factors, rank_list=config.get_factors(weight,dim_list,error)
    return factors, rank_list, dim_list





def tt_to_tensor(factors):
    tensor = factors[0]
    for i in range(1, len(factors)):
        tensor = torch.tensordot(tensor, factors[i], [[-1], [0]])
        squeezed_tensor = tensor.squeeze(0).squeeze(-1)
    return squeezed_tensor



def tucker_to_tensor(factors, device=None):

    def mode_n_product(tensor, matrix, mode):
        # Move mode axis to front
        tensor_permuted = tensor.permute(mode, *[i for i in range(tensor.ndim) if i != mode])
        original_shape = tensor.shape
        tensor_reshaped = tensor_permuted.reshape(tensor.shape[mode], -1)

        # Matrix multiplication
        result = torch.matmul(matrix, tensor_reshaped)

        # New shape after product
        new_shape = list(original_shape)
        new_shape[mode] = matrix.shape[0]
        result = result.reshape(new_shape)

        # Inverse permutation
        inverse_permute = list(range(1, mode + 1)) + [0] + list(range(mode + 1, tensor.ndim))
        return result.permute(*inverse_permute)

    # core_tensor, factor_matrices = factors
    core_tensor = factors[0]
    factor_matrices = factors[1:]
    if device is None:
        device = core_tensor.device  # Use the device of core tensor

    tensor = core_tensor.to(device)
    for mode, factor in enumerate(factor_matrices):
        tensor = mode_n_product(tensor, factor.to(device), mode)

    return tensor

def get_tt_factors(matrix, tensor_shape, error):
    tensor=matrix.view(tensor_shape)
    decomp=tt(tensor.cpu().numpy(), error)
    tt_factors=decomp.factors
    factors=[torch.from_numpy(f) for f in tt_factors]
    ranks=decomp.rank
    return factors, ranks


def get_tucker_factors(matrix, tensor_shape, error):
    tensor=matrix.view(tensor_shape)
    decomp=tucker(tensor.cpu().numpy(), error)
    tucker_core_factor=decomp.core_factor
    tucker_factor_matrices=decomp.factor_matrices
    core_factor=torch.from_numpy(tucker_core_factor)
    factor_matrices=[torch.from_numpy(f) for f in tucker_factor_matrices]
    ranks=decomp.rank
    return [core_factor] + factor_matrices, ranks

def tt_contract_x_precompute_einsum(factors, x):  # Assumed last x dim is contracted
    # Contract factors[0] and factors[1] along their shared dimension (rank_2)
    matrix_A = torch.einsum('ijk,klm->ijlm', factors[0], factors[1])  # Contract over the 3rd dim of [0] and 1st dim of [1]
    matrix_A = matrix_A.reshape(-1, factors[1].shape[-1])  # Reshape to a 2D matrix
    
    # Contract factors[2] and factors[3] along their shared dimension (rank_3)
    matrix_B = torch.einsum('ijk,klm->ijlm', factors[2], factors[3])  # Contract over the 3rd dim of [2] and 1st dim of [3]
    matrix_B = matrix_B.reshape(matrix_B.shape[0], -1)  # Reshape to (rank_2, input_dim_3 * input_dim_4)
    device = x.device
    # Contract x with matrix_A along the last dimension of x and the first dimension of matrix_A
    partial_result = torch.einsum('...i,ij->...j', x, matrix_A.to(device))  # Contract last dim of x with first dim of matrix_A
    
    # Contract partial_result with matrix_B along the last dimension of partial_result and first dimension of matrix_B
    output = torch.einsum('...i,ij->...j', partial_result, matrix_B.to(device))  # Contract last dim of partial_result with first dim of matrix_B
    
    return output




def svd(weight, rank):

    weight_shape = weight.size()
    if weight_shape[0]>weight_shape[1]:
        U, S, Vt = torch.linalg.svd(weight, full_matrices=False)
        U_r = U[:, :rank]  
        S_r = S[:rank]  
        Vt_r = Vt[:rank, :]  
        U_final = U_r * torch.sqrt(S_r).unsqueeze(0)  # Equivalent to U * sqrt(Sigma)
        V_final = torch.sqrt(S_r).unsqueeze(1) * Vt_r 
    else:
        U, S, Vt = torch.linalg.svd(weight.t(), full_matrices=False)
        U_r = U[:, :rank]  
        S_r = S[:rank]  
        Vt_r = Vt[:rank, :]  
        U_trans = U_r * torch.sqrt(S_r).unsqueeze(0)  # Equivalent to U * sqrt(Sigma)
        V_trans = torch.sqrt(S_r).unsqueeze(1) * Vt_r

        U_final = V_trans.t()
        V_final = U_trans.t()

    return U_final, V_final


def get_tt(weight, error, first_order, second_order):
    weight_shape = weight.size()
    dim_list = saten_utils.get_tensorized_layer_balanced_shape(weight_shape[0],weight_shape[1], first_order, second_order)
    factors, rank_list = get_tt_factors(weight, dim_list, error)
    return factors, rank_list, dim_list

