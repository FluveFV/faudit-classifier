import torch
import numpy as np
import argparse
import pandas as pd
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from bert_wrapper import BertForSentenceClassification
from dataloader import Dataloader


if not torch.cuda.is_available():
    raise Exception('CUDA apparently not available.')

parser = argparse.ArgumentParser(description="Predict which labels for input text.")

parser.add_argument(
    "-k",
    type=int,
    default=1,
    help="Number of action categories to predict"
)

parser.add_argument(
    "--map",
    action="store_true",
    help="Enable mapping from category of the action to relative macrocategory and field"
)

args = parser.parse_args()
k = args.k
map = args.map

# Model loader
model_path = "tuned_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
config = AutoConfig.from_pretrained(model_path)

model = BertForSentenceClassification.from_pretrained(
    model_path,
    config=config,
    model_name="dbmdz/bert-base-italian-xxl-cased",
    num_labels=config.num_labels
)

# Initialize Dataloader class in case no training has occurred
# (Dataloader automatically creates the mapping file 'reverse_encoding.json')
dataloader = Dataloader(file_path='dataset_BERT/addestramento.csv')
dataloader.load_data()

try:
    with open("reverse_encoding.json", "r") as f:
        label_mapping =  {int(k): v for k, v in json.load(f).items()}
except FileNotFoundError:
    print('The map for the model to predict actual labels has not been found.')
    print('exiting')
    raise FileNotFoundError


correspondence = pd.read_csv("dataset_BERT/correspondence.csv")

# Simple example for use
#    Levels of prediction:
#      - label (one or multiple)
#      - label (one or multiple), macrocategoria (idem), campo (idem)

def correct_input(s=None):
    """Ensures the user inputs 'y' or 'n'."""
    if s is None:
        s = input('y/n: ').strip().lower()

    if s not in ['y', 'n']:
        print(f'Wrong input: {s}.\nPlease insert "y" or "n".')
        return correct_input()
    return s
def get_multiline_input():
    """ Avoid newline being interpreted in the command line."""
    print("Paste the description of the action plan (type 'END' on a new line to finish):")
    lines = []
    while True:
        line = input()
        if line.strip().upper() == 'END':
            break
        lines.append(line)
    return "\n".join(lines).lower()
def one_label(inputs):
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_index = torch.argmax(logits, dim=-1).item()
        result = label_mapping[predicted_index] + 1
        return result
def multiple_labels(inputs, k=3):
    with torch.no_grad():
        logits = model(**inputs).logits
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits.squeeze().cpu())

        indices_above_threshold = (probs >= 0.6).nonzero(as_tuple=True)[0]

        probs_above_threshold = probs[indices_above_threshold]
        sorted_indices = probs_above_threshold.argsort(descending=True)

        top_k_indices = indices_above_threshold[sorted_indices][:k]
        result = [label_mapping[int(idx)] + 1 for idx in top_k_indices]
        return result
def answer(ID_tassonomia):
    """
    Given the label(s), retrieves from the taxonomy the rest of the information about the prediction.
    :param ID_tassonomia: int, list - the ID(s) of the tassonomia.
    :return: List of tuples with (azione, campi, descrizione_codice_campo, macroambiti, descrizione_codice_macro).
    """
    dID, c, cd, m, md = correspondence.loc[
        correspondence.ID_tassonomia == ID_tassonomia, ['azione', 'campi', 'descrizione_codice_campo', 'macroambiti',
                                                        'descrizione_codice_macro']].values[0]
    return dID, c, cd, m, md

print('Start prediction?')
testing_model = correct_input()

while testing_model == 'y':
    description = get_multiline_input()
    inputs = tokenizer(description, return_tensors="pt", truncation=True, padding=True, return_token_type_ids=False)
    if k > 1:
        #Multilabel case
        pred = multiple_labels(inputs=inputs, k=k)
        print('___________  Result:  ___________')
        print('ID tassonomie di azione:')
        print(pred)
        for el in pred:
            dID, c, cd, m, md = answer(el)
            print(f'ID tassonomia di azione: {dID}')
            if map:
                print('Campi:')
                print(c)
                print('Macroambiti:')
                print(m)
        print('__________________________________')

    else:
        #Single label case
        pred = one_label(inputs)
        result = answer(pred)
        print('___________  Result:  ___________')
        dID, c, cd, m, md = result
        print('ID tassonomie di azione:')
        print(pred)
        print(dID)
        if map:
            print(f'Campo: {c}, {cd}')
            print(f'Macroambito: {m}, {md}')
        print('__________________________________')

    print()
    print('Would you like to test another description?')
    testing_model = correct_input()

'''
## debugging set: in case the model isn't loading, you can try to use the following to run the prediction on 300 observations
test = pd.read_csv('dataset_BERT/addestramento.csv').iloc[:100, -2:]
t = test.text.tolist()
pred = []
print()
print(f'Predicting {test.shape[0]} lines of text used in training...')
for text in tqdm(t):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, return_token_type_ids=False)
    outputs = model(**inputs)
    logits = outputs.logits
    #unfreeze and adapt for multi-label prediction!
    #probs = sigmoid(logits.squeeze().cpu())  # Convert logits to probabilities
    #predictions = np.zeros(probs.shape)
    #predictions[np.where(probs >= 0.6)] = 1
    predicted_index = torch.argmax(logits, dim=-1).item()
    # predicted_label = [label_mapping[idx + 1] for idx, label in enumerate(predictions) if label == 1.0]
    predicted_label = label_mapping[predicted_index] + 1

    pred.append(predicted_label)

pd.DataFrame({'Text': t,
              'true_label': test.label.tolist(),
              'predicted_label': pred}).to_csv('test_predictor.csv', index=False)
'''
