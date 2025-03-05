import os

from datasets import load_dataset, DatasetDict, Dataset
from dataloader import Dataloader

import torch
from torch import nn
import torch.nn.functional as F

import transformers
from transformers import AutoConfig, TrainingArguments, EarlyStoppingCallback, Trainer
from bert_wrapper import BertForSentenceClassification, TrainerHandler
from tokenizer_function import TokenizerFunction

import wandb

if torch.cuda.is_available() == False:
    raise Exception('CUDA not available for torch')
device = torch.device("cuda")
name = os.path.basename(__file__).split('.')[0]

# WANDB config
project_name = "Temperature"
wandb.init(project=f"{project_name}", name=f"{name}", config={})

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["TORCH_USE_CUDA_DSA"] = 'true'

#print('--- Emptying cache...')
#torch.cuda.empty_cache()
#print('--- Cache emptied ...')

model_save_path = "temperature_tuned"
os.makedirs(model_save_path, exist_ok=True)

model_name = "dbmdz/bert-base-italian-xxl-uncased"

#Load it
dataloader = Dataloader(file_path='dataset_BERT/addestramento.gzip')
dataloader.load_data()
dataloader.stratified_split()
dataset = dataloader.get_dataset()

#Charge it
class_weights = dataloader.get_class_weights()
num_labels = dataloader.get_num_labels()
encoding = dataloader.get_encoding()

#Tokenize it
tokenize_function = TokenizerFunction(model_name=model_name, max_length=512)

tokenized_datasets = (dataset.map(tokenize_function, batched=True)
                      .shuffle(seed=25)
                      .remove_columns(['text', 'token_type_ids']))

#print(f'--Loading the model for predicting {num_labels} labels--')
config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)  #for later saving the config

temperature = 1.5

model = (BertForSentenceClassification
         .from_pretrained(pretrained_model_name_or_path=model_name,
                          model_name=model_name,
                          config=config,
                          num_labels=num_labels,
                          class_weights=class_weights,
                          temperature=temperature))
device = torch.device("cuda")
model.to(device)
#transformers.logging.set_verbosity_info()
training_args = TrainingArguments(
    output_dir=model_save_path,
    eval_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=5,
    num_train_epochs=30,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    weight_decay=0.005,
    learning_rate=8e-5,
    lr_scheduler_type='linear',
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    #max_steps=1,
)
from torchmetrics.classification import MulticlassF1Score

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    logits = torch.tensor(logits).to(device)
    scaled_logits = logits / temperature
    labels = torch.tensor(labels).to(device)

    # log-softmax
    log_probs = F.log_softmax(logits, dim=-1)
    nll = F.nll_loss(log_probs, labels, reduction="mean").item()

    #CEL
    loss_fct = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    loss = loss_fct(scaled_logits, labels).item()

    # F1
    f1 = MulticlassF1Score(num_classes=num_labels, average="weighted").to(device)
    f1_score = f1(logits.argmax(dim=1), labels)

    return {"f1": f1_score, "nll": nll, "cel":loss}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    callbacks=[EarlyStoppingCallback(early_stopping_patience=15)],
    compute_metrics=compute_metrics
)

trainer.tokenizer = tokenize_function.get_tokenizer()  #this is done separately to avoid issues of circular import.

if __name__ == "__main__":
    handler = TrainerHandler(
        trainer=trainer,
        tokenized_datasets=tokenized_datasets,
        num_labels=num_labels,
        encoding=encoding,
        model_name=name,
        model_save_path=model_save_path
    )
    handler.run()
