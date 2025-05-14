


def saten_24_llama_3_3b(ratio=None, info_path=None):

    target_layers_names=[]
    target_types_list=[]
    layer_configs_list=[]
    ######################################################################################
    layer_config_dict_1={
            'layer_class_type': 'SatenTT',
            'mode': 'teacher',
            'in_order': 2,
            'out_order': 2,
            'decomp_error': ratio,
            'sparsity_ratio': [2,4],
            'sparsity': 'm2n'
            }

    for i in range(0,28):
        layer_name='model.layers.'+str(i)+'.mlp.gate_proj'
        target_layers_names.append(layer_name)
        target_types_list.append('linear')
        layer_config_dict_1['layer_name']=layer_name
        layer_configs_list.append(layer_config_dict_1.copy())

        layer_name='model.layers.'+str(i)+'.mlp.up_proj'
        target_layers_names.append(layer_name)
        target_types_list.append('linear')
        layer_config_dict_1['layer_name']=layer_name
        layer_configs_list.append(layer_config_dict_1.copy())

        layer_name='model.layers.'+str(i)+'.mlp.down_proj'
        target_layers_names.append(layer_name)
        target_types_list.append('linear')
        layer_config_dict_1['layer_name']=layer_name
        layer_configs_list.append(layer_config_dict_1.copy())

        layer_name='model.layers.'+str(i)+'.self_attn.o_proj'
        target_layers_names.append(layer_name)
        target_types_list.append('linear')
        layer_config_dict_1['layer_name']=layer_name
        layer_configs_list.append(layer_config_dict_1.copy())

        layer_name='model.layers.'+str(i)+'.self_attn.v_proj'
        target_layers_names.append(layer_name)
        target_types_list.append('linear')
        layer_config_dict_1['layer_name']=layer_name
        layer_configs_list.append(layer_config_dict_1.copy())

        layer_name='model.layers.'+str(i)+'.self_attn.k_proj'
        target_layers_names.append(layer_name)
        target_types_list.append('linear')
        layer_config_dict_1['layer_name']=layer_name
        layer_configs_list.append(layer_config_dict_1.copy())

        layer_name='model.layers.'+str(i)+'.self_attn.q_proj'
        target_layers_names.append(layer_name)
        target_types_list.append('linear')
        layer_config_dict_1['layer_name']=layer_name
        layer_configs_list.append(layer_config_dict_1.copy())


    ############################################################################################
    ############################################################################################

    net_config_dict = {
                'target_layers': target_layers_names,
                'target_types': target_types_list,
                'layer_configs': layer_configs_list
            }
   
    return net_config_dict
######################################################################################
######################################################################################
######################################################################################

