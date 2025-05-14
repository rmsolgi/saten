import torch
import torch.nn.functional as F
from tqdm import tqdm
from abc import ABC, abstractmethod
import numpy as np

####################################################################################
####################################################################################
####################################################################################
####################################################################################
class GenTaskTemplate(ABC):
    def __init__(self,tokenizer):
        self.tokenizer = tokenizer
        self.seq_len_training = None
        
        self.dataset = None  
        
        self.iterate_key = None
        self.columns_to_be_removed=None
        self.seq_len_filter_max=None
        self.QA=False
        self.generation = True
####################################################################################
    @abstractmethod
    def get_base_prompt(self, example):
        """Return the base prompt structure (varies by dataset)."""
        pass

    def get_answer(self, example):
        """Default implementation: Subclasses can override if needed"""
        raise ValueError(" get_answer need to be override")
        return None  

    def get_full_prompt(self, base_prompt, candid=None):
        """Default implementation: Subclasses can override if needed"""
        raise ValueError(" get_full_prompt need to be override")
        return base_prompt
####################################################################################
    def instantiate(self):
        self.train_dataset = self.get_train_dataset()
        self.eval_dataset = self.dataset['validation']#.select(range(128))
    
####################################################################################
    def get_tokenized(self, inputs, mode):
        """Tokenize input based on mode (training or encoding)."""
        if mode == "training":
            return self.tokenizer(
                inputs,
                truncation=True,
                padding="max_length",
                max_length=self.seq_len_training,
                return_tensors="pt",
            )
        elif mode == "encode":
            return self.tokenizer.encode(
                inputs,
                truncation=False,
                padding=False,
                # max_length=self.seq_len_encode,
                return_tensors="pt",
            )
####################################################################################
    def get_tokenized_w_wo_candidate(self, base_prompt, candidate):
        """Tokenize base prompt and full prompt, computing candidate length."""
        full_prompt = self.get_full_prompt(base_prompt, candidate)
        full_tokenized = self.get_tokenized(full_prompt, mode="encode")
        base_tokenized = self.get_tokenized(base_prompt, mode="encode")

        full_len = full_tokenized.shape[1]
        base_len = base_tokenized.shape[1]

        if full_len > base_len:
            candid_len = full_len - base_len
        else:
            print("base: ", base_prompt)
            print("candid: ", candidate)
            print("full: ", full_prompt)
            print("full_t_len: ", full_len)
            print("base_t_len:", base_len)
            raise ValueError("choice length is not positive")

        return {
            "full_tokenized": full_tokenized,
            "base_tokenized": base_tokenized,
            "candid_len": candid_len,
        }
####################################################################################
    def get_data_stat(self):
        lengths = []
       
        for example in self.dataset["train"]:

            prompt=self.get_base_prompt(example)
            tokenized = self.get_tokenized(prompt, mode='encode')

            lengths.append(tokenized.shape[1])
        

        max_length = max(lengths)

        percentile_95 = np.percentile(lengths, 80)
        
        return {"max_length": max_length, "80_percentile_length": percentile_95}

####################################################################################
    def print_example(self):
        counter=0
        for example in self.dataset['train']:
            prompt=self.get_base_prompt(example)
            if counter==0:
                print(prompt)
####################################################################################               
    def filter_long_prompts(self, example):
            prompt = self.get_base_prompt(example)
            tokenized = self.get_tokenized(prompt, mode='encode')
            # Check if the tokenized prompt is not too long
            return tokenized.shape[1] < self.seq_len_filter_max  # Assuming self.max_length is defined
 
 
####################################################################################
    def qa_preprocess_function(self, examples):
        """Preprocess dataset examples for tokenization and training."""
        inputs = [
            self.get_base_prompt({key: examples[key][i] for key in examples})
            + self.get_answer({key: examples[key][i] for key in examples})
            for i in range(len(examples[self.iterate_key]))
        ]

        tokenized = self.get_tokenized(inputs, mode="training")
        tokenized["labels"] = tokenized["input_ids"].clone()

        for i in range(len(tokenized["labels"])):
            example = {key: examples[key][i] for key in examples}
            base_prompt = self.get_base_prompt(example)
            correct_candidate = self.get_answer(example)
            answer_length = self.get_tokenized_w_wo_candidate(base_prompt, correct_candidate)["candid_len"]

            tokenized["labels"][i, :-answer_length] = -100  # Mask everything except the correct answer
        return tokenized
    
    def s2s_preprocess_function(self, examples):
        """Preprocess dataset examples for tokenization and training."""
        inputs = [
            self.get_base_prompt({key: examples[key][i] for key in examples})
            for i in range(len(examples[self.iterate_key]))
        ]

        tokenized = self.get_tokenized(inputs, mode="training")
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized

####################################################################################
    def get_train_dataset(self):
        """Tokenize the dataset and return train/eval splits."""
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Ensure the subclass defines `self.dataset`.")
        if self.QA is None:
            raise ValueError("QA must be set")

        if self.QA:
            tokenized_train_dataset = self.dataset["train"].map(
                self.qa_preprocess_function, batched=True, remove_columns=self.columns_to_be_removed
            )
        else:
            tokenized_train_dataset = self.dataset["train"].map(
                self.s2s_preprocess_function, batched=True, remove_columns=self.columns_to_be_removed
            )
        tokenized_train_dataset.set_format("torch")

        train_dataset = tokenized_train_dataset
      

        return train_dataset

####################################################################################
####################################################################################
####################################################################################
####################################################################################

class GenTaskTemplateS2S(GenTaskTemplate): #for Question Answering as generative tasks

    def __init__(self,tokenizer):
        super().__init__(tokenizer)
        self.QA=False

    def get_prompt_and_label_for_eval(self, example):
        """Compute tokenized prompts and correct label for evaluation."""
        # tokenized_prompts = []
        base_prompt = self.get_base_prompt(example)
        # all_candidates = self.get_all_candidates(example)
        tokenized = self.get_tokenized(base_prompt, mode="encode")
        labels = tokenized.clone()
        return tokenized, labels
       

    def evaluate(self, model):
        """Computes perplexity for the evaluation dataset with shifted labels."""
        all_log_probs = []
        total_tokens = 0

        
        for example in tqdm(self.eval_dataset, desc="Evaluation"):
            input_ids, labels = self.get_prompt_and_label_for_eval(example)

            # Move input tensors to the same device as the model
            input_ids = input_ids#.to(device)
            labels = labels#.to(device)

            with torch.no_grad():
                outputs = model(input_ids, labels=labels)
                logits = outputs.logits  # Shape: (batch_size, seq_length, vocab_size)

            shift_logits = logits[:, :-1, :]  
            shift_labels = input_ids[:, 1:]

            log_probs = F.log_softmax(shift_logits, dim=-1)

            token_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

            all_log_probs.append(token_log_probs.sum().detach())  # Keep as tensor, no `.item()`
            total_tokens += shift_labels.numel()

        avg_log_prob = sum(all_log_probs) / total_tokens  # Keep computation on same device
        perplexity = torch.exp(-avg_log_prob)  # PPL = exp(-avg_log_prob)

        return {"perplexity": perplexity.item()}  # Convert to Python scalar only at the end


####################################################################################
####################################################################################
####################################################################################
####################################################################################

