import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchmetrics.classification import MulticlassF1Score, Accuracy

from transformers import AutoModelForSequenceClassification, AutoModel
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from datetime import date, datetime

import pandas as pd

import json
import wandb

class BertForSentenceClassification(PreTrainedModel):

    """
    BERT architecture is intended to be from "dbmdz/bert-base-italian-xxl-cased"
    but other models can be tried.
    """


    def __init__(self, config, model_name, num_labels, class_weights=None):
        super().__init__(config)
        self.num_labels = num_labels
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.class_weights = class_weights
        self.accuracy = Accuracy(num_classes=num_labels, task='multiclass')
        self.f1 = MulticlassF1Score(num_classes=num_labels, average='weighted')

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])

        loss = None
        if labels is not None:

            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights, label_smoothing=0.1)
            loss = loss_fct(logits, labels)

            f1_score = self.f1(logits.argmax(dim=1), labels)
            accuracy_score = self.accuracy(logits.argmax(dim=1), labels)
            wandb.log({
                "f1_score": f1_score,
                "accuracy": accuracy_score,
                'CrossEntropyLoss': loss.item() if loss is not None else None
            })

        return SequenceClassifierOutput(loss=loss, logits=logits)

class TrainerHandler:
    def __init__(self, trainer, tokenized_datasets, num_labels, encoding, model_name,
                 model_save_path):
        """
        Args:
            trainer: Hugging Face Trainer instance.
            tokenized_datasets: DatasetDict with train/val/test splits.
            num_labels: Number of labels in the classification task.
            encoding: Mapping from dense to sparse original labelling.
            model_name
            model_save_path: Path to save the configuration, weights and tokenizer for reproducibility.
        """
        self.trainer = trainer
        self.tokenized_datasets = tokenized_datasets
        self.num_labels = num_labels
        self.encoding = encoding
        self.model_name = model_name
        self.model_save_path = model_save_path

    def compute_f1_score(self, predictions):
        """
        Micro F1 score from

        Args:
            predictions: Output from the trainer.predict() method.

        Returns:
            f1_score: The calculated F1 score.
        """
        logits = torch.tensor(predictions.predictions)
        labels = torch.tensor(predictions.label_ids)
        f1 = MulticlassF1Score(num_classes=self.num_labels, average='weighted')
        f1_score = f1(logits.argmax(dim=1), labels.float())
        return f1_score, logits, labels

    def save_f1_results(self, f1_score):
        """
        Save the F1 score to a CSV file.

        Args:
            f1_score: by default, it's the micro F1 weighted on class frequency.
        """
        now = datetime.today()
        pd.DataFrame({
            'F1': [f1_score.item()],
            'modello': [self.model_name],
            'T': [now],
        }).to_csv(f'{self.model_name}.csv')
        print(f"F1 score saved to {self.model_name}.csv")

    def save_predictions(self, logits, labels):
        """
        Save predicted and true labels to a CSV file.

        Args:
            logits: Model logits from the predictions.
            labels: Ground truth label indexes from the predictions.
        """
        now = datetime.today()
        inverted_encoding = {int(v): k for k, v in self.encoding.items()}
        predicted_indices = logits.argmax(dim=1)
        predicted_labels = [inverted_encoding[idx.item()] for idx in predicted_indices]
        true_labels = [inverted_encoding[idx.item()] for idx in labels]

        original_index = self.tokenized_datasets['test']['index'] #original indexes for test observations in the input df

        results = pd.DataFrame({
            'original_index': original_index,
            'true_label': true_labels,
            'predicted_label': predicted_labels,
        })

        results.true_label, results.predicted_label = results.true_label + 1, results.predicted_label + 1

        file_name = f'{self.model_name}_{now.month}_{now.day}-{now.hour}_{now.minute}.csv'
        results.to_csv(file_name, index=False)
        print(f"Predictions saved to {file_name}")

    def save_model_and_tokenizer(self):
        """
        Save the trained model and tokenizer to the specified path.
        """
        print(f"Saving the trained model as {self.model_name}...")
        self.trainer.model.save_pretrained(self.model_save_path)
        self.trainer.tokenizer.save_pretrained(self.model_save_path)
        print(f"Model and tokenizer saved to {self.model_save_path}")

    def run(self):
        """
        Execute the training and save the outputs.
        """
        print("Starting training...")
        #self.trainer.train(resume_from_checkpoint=True)
        self.trainer.train()
        self.trainer.save_state()

        print("Evaluating test set...")
        predictions = self.trainer.predict(self.tokenized_datasets['test'])

        f1_score, logits, labels = self.compute_f1_score(predictions)
        print(f"F1 score: {f1_score.item()}\n")

        self.save_f1_results(f1_score)
        self.save_predictions(logits, labels)
        self.save_model_and_tokenizer()
        print("Done.")
