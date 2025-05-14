
def svd_llama_3_3b_uni(ratio, info_path=None):

    target_layers_names=[]
    target_types_list=[]
    layer_configs_list=[]
    ######################################################################################
    mn = 3072*8192
    mpn = 3072+8192
    rank = int ((ratio * mn) / mpn)

    layer_config_dict={
            'layer_class_type': 'SVDW',
            'mode': 'teacher',
            'rank': rank
            }

    for i in range(0,28):
        layer_name='model.layers.'+str(i)+'.mlp.gate_proj'
        target_layers_names.append(layer_name)
        target_types_list.append('linear')
        layer_config_dict['layer_name']=layer_name
        layer_configs_list.append(layer_config_dict.copy())

        layer_name='model.layers.'+str(i)+'.mlp.up_proj'
        target_layers_names.append(layer_name)
        target_types_list.append('linear')
        layer_config_dict['layer_name']=layer_name
        layer_configs_list.append(layer_config_dict.copy())

        layer_name='model.layers.'+str(i)+'.mlp.down_proj'
        target_layers_names.append(layer_name)
        target_types_list.append('linear')
        layer_config_dict['layer_name']=layer_name
        layer_configs_list.append(layer_config_dict.copy())


######################################################################################
    mn = 3072*3072
    mpn = 3072+3072
    rank = int ((ratio * mn) / mpn)

    layer_config_dict={
            'layer_class_type': 'SVDW',
            'mode': 'teacher',
            'rank': rank
            }

    for i in range(0,28):

        layer_name='model.layers.'+str(i)+'.self_attn.o_proj'
        target_layers_names.append(layer_name)
        target_types_list.append('linear')
        layer_config_dict['layer_name']=layer_name
        layer_configs_list.append(layer_config_dict.copy())

        layer_name='model.layers.'+str(i)+'.self_attn.q_proj'
        target_layers_names.append(layer_name)
        target_types_list.append('linear')
        layer_config_dict['layer_name']=layer_name
        layer_configs_list.append(layer_config_dict.copy())

######################################################################################

    mn = 3072*1024
    mpn = 3072+1024
    rank = int ((ratio * mn) / mpn)

    layer_config_dict={
            'layer_class_type': 'SVDW',
            'mode': 'teacher',
            'rank': rank
            }

    for i in range(0,28):
        layer_name='model.layers.'+str(i)+'.self_attn.k_proj'
        target_layers_names.append(layer_name)
        target_types_list.append('linear')
        layer_config_dict['layer_name']=layer_name
        layer_configs_list.append(layer_config_dict.copy())

        layer_name='model.layers.'+str(i)+'.self_attn.v_proj'
        target_layers_names.append(layer_name)
        target_types_list.append('linear')
        layer_config_dict['layer_name']=layer_name
        layer_configs_list.append(layer_config_dict.copy())

######################################################################################

    net_config_dict = {
                'target_layers': target_layers_names,
                'target_types': target_types_list,
                'layer_configs': layer_configs_list
            }

    return net_config_dict

######################################################################################
######################################################################################
######################################################################################
