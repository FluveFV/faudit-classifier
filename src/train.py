import os

from datasets import load_dataset, DatasetDict, Dataset
from dataloader import Dataloader

import torch
import transformers
from transformers import AutoConfig, TrainingArguments, EarlyStoppingCallback, Trainer
from bert_wrapper import BertForSentenceClassification, TrainerHandler
from tokenizer_function import TokenizerFunction

#import wandb

if torch.cuda.is_available() == False:
    raise Exception('CUDA not available for torch')

name = os.path.basename(__file__)

# WANDB config
#project_name = P1
#wandb.init(project="{project_name}", name=f"{name}", config={})

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["TORCH_USE_CUDA_DSA"] = 'true'

print('--- Emptying cache...')
torch.cuda.empty_cache()
print('--- Cache emptied ...')

if 'tuned_model' not in os.listdir():
    os.mkdir('tuned_model')
model_save_path = "./tuned_model"

model_name = "dbmdz/bert-base-italian-xxl-cased"

#Load it
dataloader = Dataloader(file_path='dataset_BERT/addestramento.gzip')
dataloader.load_data()
dataloader.stratified_split()
dataset = dataloader.get_dataset()

#Charge it
class_weights = dataloader.get_class_weights()
num_labels = dataloader.get_num_labels()
label_mapping = dataloader.get_label_mapping()

#Tokenize it
tokenize_function = TokenizerFunction(model_name=model_name, max_length=512)

tokenized_datasets = (dataset.map(tokenize_function, batched=True)
                      .shuffle(seed=25)
                      .remove_columns(['text', 'token_type_ids']))


#print(f'--Loading the model for predicting {num_labels} labels--')
config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)  #for later saving the config
model = (BertForSentenceClassification
         .from_pretrained(pretrained_model_name_or_path=model_name,
                          model_name=model_name,
                          config=config,
                          num_labels=num_labels,
                          class_weights=class_weights))

#transformers.logging.set_verbosity_info()
training_args = TrainingArguments(
    output_dir='results/',
    eval_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=5,
    num_train_epochs=20,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    weight_decay=0.005,
    learning_rate=1e-5,
    lr_scheduler_type='linear',
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.tokenizer = tokenize_function.get_tokenizer()

if __name__ == "__main__":
    handler = TrainerHandler(
        trainer=trainer,
        tokenized_datasets=tokenized_datasets,
        num_labels=num_labels,
        label_mapping=label_mapping,
        model_name=name,
        model_save_path=model_save_path
    )
    handler.run()
