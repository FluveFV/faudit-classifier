import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import json

if not torch.cuda.is_available():
    raise Exception('CUDA apparently not available.')

# Model loader
model_path = "./tuned_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

#Levels of the prediction: just the label, macrocategoria, campo, or a combination of the three
correspondence = pd.read_csv("dataset_BERT/correspondence.csv")
with open("tuned_model/label_mapping.json", "r") as f:
    label_mapping = {int(item['Mapped']): item['Original'] for item in json.load(f)}

def answer(ID_tassonomia):
    """
    Given the label, retrieves from the whole taxonomy the rest of the information about the prediction
    :param ID_tassonomia:
    :return: description of ID_tassonomia, field (campi),
             description of field, macrocategory (macrocategoria),
             description of macrocategory
    """
    dID, c, cd, m, md = correspondence.loc[correspondence.ID_tassonomia == ID_tassonomia, ['azione', 'campi', 'descrizione_codice_campo', 'macroambiti', 'descrizione_codice_macro']].values[0]
    return dID, c, cd, m, md

# Simple example for use
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


print('Start prediction?')
testing_model = correct_input()

while testing_model == 'y':
    description = get_multiline_input()
    inputs = tokenizer(description, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_index = torch.argmax(outputs.logits, dim=-1).item()
        pred =  label_mapping[predicted_index]

    print('___________  Result:  ___________')
    dID, c, cd, m, md = answer(pred)
    print(f'ID tassonomia di azione: {pred}, {dID}')
    print(f'Campo: {c}, {cd}')
    print(f'Macroambito: {m}, {md}')
    print('__________________________________')
    print()
    print('Would you like to test another description?')
    testing_model = correct_input()