
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from src.network.remodel import swap_layers
from peft import get_peft_model, LoraConfig, TaskType
from src.network.utils import get_compression_ratio
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from src.applications.generation.gen_benchmarks import get_gen_bench
from src.network.utils import save_model_params
from llama_3_3b_wiki.setup.trainer_setup import TrainModel
from llama_3_3b_wiki.setup import net_configs
from peft import get_peft_model, LoraConfig, TaskType
from pathlib import Path
############################################################################################
DO_COMPRESS = True
DO_TUNE = True
EVAL = True
############################################################################################
ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_NAME = 'wikitext'
BASE_MODEL_NAME = 'llama_3_3b'
DATA_DIR = os.path.join(ROOT_DIR, "artifacts")
COMPRESSION_CONFIG = 'saten_24_llama_3_3b'# change it to 'tt_llama_3_3b' or 'svd_llama_3_3b_uni' for svd and tt respectively.
TEMP_DIR = DATA_DIR
COMPRESSED_DIR = DATA_DIR
init_model_name='meta-llama/Llama-3.2-3B'
############################################################################################
ratios = {
    'tt_llama_3_3b': 0.63,
    'svd_llama_3_3b_uni': 0.6,
    'saten_24_llama_3_3b': 1.0, 
    }
ratio=ratios.get(COMPRESSION_CONFIG)

num_epoch_mapping = {
    "wikitext": 1
}
learning_rate_mapping = {
    "wikitext": 2e-5
}
############################################################################################
############################################################################################
init_model_path=os.path.join(DATA_DIR, BASE_MODEL_NAME, BASE_MODEL_NAME+'_'+DATASET_NAME)
############################################################################################
init_model = AutoModelForCausalLM.from_pretrained(init_model_name, device_map = 'auto')
tokenizer = AutoTokenizer.from_pretrained(init_model_name)
###########################################################################################
model_name=COMPRESSION_CONFIG+'_'+DATASET_NAME
compressed_path=os.path.join(COMPRESSED_DIR, COMPRESSION_CONFIG , model_name)
temp_path=os.path.join(TEMP_DIR, 'temp', model_name)
os.makedirs(compressed_path, exist_ok=True)
os.makedirs(temp_path, exist_ok=True)
###########################################################################################
#############################################################################################

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
bench = get_gen_bench(tokenizer,DATASET_NAME)
peft_task_type=TaskType.CAUSAL_LM

#############################################################################################
#############################################################################################
get_config = getattr(net_configs, COMPRESSION_CONFIG) 
info_path=init_model_path
net_config=get_config(ratio, info_path)
#############################################################################################
#############################################################################################
peft_targets = ["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]

if DO_TUNE:
    peft_config = LoraConfig(
        r=8,  # rank
        lora_alpha=32,
        target_modules=peft_targets,  # update based on model architecture
        lora_dropout=0.05,
        bias="none",
        task_type=peft_task_type
    )

    init_model = get_peft_model(init_model, peft_config)
    for index, element in enumerate(net_config['target_layers']):
        if any(peft_target in element for peft_target in peft_targets):
            peft_revised_name = 'base_model.model.' + element + '.base_layer'
            net_config['target_layers'][index] = peft_revised_name
        else:
            peft_revised_name = 'base_model.model.' + element #+ '.base_layer'
            net_config['target_layers'][index] = peft_revised_name

#############################################################################################
#############################################################################################

if DO_COMPRESS:
    new_model=swap_layers(init_model,net_config)
    save_model_params(new_model, net_config, compressed_path)
else:
    params_dict=torch.load(os.path.join(compressed_path, 'tm_params.pth'))
    new_model=swap_layers(init_model,net_config, params_dict)
#############################################################################################
total_params = sum(p.numel() for p in new_model.parameters())
# print(f"Total parameters: {total_params:,}")
#############################################################################################
num_epochs=num_epoch_mapping.get(DATASET_NAME, "dataset name is not recognized in num_epoch_mapping")
learning_rate = learning_rate_mapping.get(DATASET_NAME, "dataset name is not recognized in learning_rate_mapping")
train_args={
    'num_epochs': num_epochs,
    'temp_path': temp_path,
    'learning_rate': learning_rate
    }
############################################################################################
if DO_TUNE:
    ############################################################################################
    for name, param in new_model.named_parameters():
            if any(layer_name in name for layer_name in net_config['target_layers']):
                param.requires_grad = False
    ############################################################################################
    custom_trainer = TrainModel(new_model, tokenizer, bench, train_args)
    custom_trainer.train_model()
else: 
    train_args['num_epochs'] = 0
    custom_trainer = TrainModel(new_model, tokenizer, bench, train_args)
    custom_trainer.train_model()

if EVAL: 
    new_model.eval()
    eval_results = bench.evaluate(new_model)
    print(eval_results)

params_dict=torch.load(os.path.join(compressed_path, 'tm_params.pth'))
num_params, compression_ratio = get_compression_ratio(params_dict)
print('Compression Ratio: ', compression_ratio)
print('Num Params: ', num_params)
