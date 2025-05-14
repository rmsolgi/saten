
import os
from transformers import TrainingArguments, DataCollatorWithPadding, Trainer, DataCollatorForSeq2Seq
import torch

class TrainModel():
    def __init__(self, new_model, tokenizer, bench, train_args):
        
        self.new_model = new_model
        self.tokenizer = tokenizer

        self.bench = bench
        self.train_args = train_args
    def train_model(self):   
        training_args = TrainingArguments(
                output_dir=os.path.join(self.train_args['temp_path'], 'results'),
                save_strategy="epoch", 
                learning_rate=self.train_args['learning_rate'],
                per_device_train_batch_size=1,
                num_train_epochs=self.train_args['num_epochs'],
                weight_decay=0.01,
                logging_dir=os.path.join(self.train_args['temp_path'], 'logs'),
                logging_steps=10,
                save_total_limit=1,
                fp16=True,
            )
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.new_model, padding=True)
        trainer_args = {
                        "model": self.new_model,
                        "args": training_args,
                        "train_dataset": self.bench.train_dataset,
                        "tokenizer": self.tokenizer,
                        "data_collator": data_collator,
            }
            
        ######################################################################################################
        self.trainer = Trainer(**trainer_args)
        self.trainer.train()
        #########################################################################################
        ######################################################################################################
        ######################################################################################################
    def evaluate_model(self):
        self.trainer.model.eval()
        eval_results = self.bench.evaluate(self.trainer.model)
        return eval_results


