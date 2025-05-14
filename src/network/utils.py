
from transformers import AutoConfig, AutoModel, AutoTokenizer, PretrainedConfig, PreTrainedModel
from lm_eval import evaluator
import torch
import os
import pickle
from src.layers.config import LayerConfig
from src.network.config import NetConfig
from src import layers
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial


def save_model_params(the_model, net_config_dict, dir):
    model_params_dict={}
    target_layers=net_config_dict['target_layers']
    for name, module in the_model.named_modules():
        if name in target_layers:
            # print(name)
            index = target_layers.index(name)
            key=net_config_dict['layer_configs'][index]['layer_name']
            model_params_dict[key] = module.get_params_dict()
    
    tm_params_path=os.path.join(dir,'tm_params.pth')
    torch.save(model_params_dict, tm_params_path)


def process_net_config(net_config_dict):
    processed_net_config_dict = {
                'target_layers': net_config_dict['target_layers'],
                'target_types': net_config_dict['target_types'],
                # 'layer_configs': layer_configs_list
            }
    layer_config_list = []
    for conf in net_config_dict['layer_configs']:
        conf['layer_class'] = getattr(layers, conf['layer_class_type']) 
        layer_config_list.append(LayerConfig(**conf))
    processed_net_config_dict['layer_configs'] = layer_config_list
    net_config=NetConfig(**processed_net_config_dict)
    return net_config


def get_compression_ratio(net_param_dict):
    total_compressed_params = 0.0
    total_base_params = 0.0

    target_device = None
    for entry in net_param_dict.values():
        for v in entry.values():
            if hasattr(v, 'device'):
                target_device = v.device
                break
        if target_device:
            break
    if target_device is None:
        target_device = torch.device("cpu")

    for key, entry in net_param_dict.items():
        num_params = entry['num_params']
        params_ratio = entry['params_ratio']

        if hasattr(num_params, 'to'):
            num_params = num_params.to(target_device)
        if hasattr(params_ratio, 'to'):
            params_ratio = params_ratio.to(target_device)

        total_compressed_params += num_params
        total_base_params += num_params / params_ratio

    ratio = total_compressed_params / total_base_params
    return total_compressed_params, ratio



def get_activations_empirical_cov(model, dataset, batch_size=1, device="cuda"):
    model.eval()

    def collate_fn(batch):
        return {
            k: torch.stack([d[k].clone().detach() for d in batch])
            for k in dataset.column_names
        }

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    activations = {}

    def hook_fn(module, input, output, layer_name):
        X = input[0]
        X = X.reshape(-1, X.shape[-1])
        XXT = X.T @ X
        if layer_name in activations:
            activations[layer_name] += XXT
        else:
            activations[layer_name] = XXT

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(partial(hook_fn, layer_name=name)))

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing Cov"):
            batch = {k: v.to(device) for k, v in batch.items()}
            _ = model(**batch)

    for hook in hooks:
        hook.remove()

    return activations




def filter_config_by_block(net_config_dict, block_index):
    block_str = f".{block_index}."
    indices = [
        idx for idx, layer_name in enumerate(net_config_dict['target_layers'])
        if block_str in layer_name
    ]

    filtered_dict = {
        'target_layers': [net_config_dict['target_layers'][i] for i in indices],
        'target_types': [net_config_dict['target_types'][i] for i in indices],
        'layer_configs': [net_config_dict['layer_configs'][i] for i in indices],
    }

    return filtered_dict