# Predict

The product is ready for downstream tasks. A simple example for using the predictive capacity is displayed in `predictor.py`. The main elements of said scripts are divided in:

-   loading the model weights and architecture for prediction
-   loading the label mapping from dense to sparse
-   loading additional information regarding the label prediction
-   predicting mechanism
-   simple CLI interface

## Loading model weights, architecture

It can be done by using hugginface's AutoTokenizer and AutoModelForSequenceClassification.

## Label mapping, additional information regarding labels

Since the BERT architecture in use was trained with specific attributes, the predicted labels were mapped in two ways:

-   $f(y) = y-1$ so that they start from 0 since labels originally start from 1
-   reindexing $f(y)$ so that $y$ is contiguous (using reindexing)

Hence, the model learns and evaluates labels specifically moved towards 0 by 1 and mapped to a dense array. For this reason, any downstream task requires to store the mapping back to the original distribution using the inverted indexing. The operation of summing of 1 each label can be done with no further explanation, while `predictor.py` uses the label indexing used in training and saved in `tuned_model/label_mapping.json` to revert the predictions back to original labelling.

Additional information includes:

-   macroambito
-   campo

and the relative descriptions. Those are domain-specific information that is stored in [`correspondences.csv`]()

## Predicting mechanism and CLI usage

After loading the pretrained model, ```predictor.py``` given a text will output its predicted label, map it back to the original distribution, then display the additional information. 
The user can test more texts. There is no lower or upper limit to the size of the input, but the BERT will only use the first 512 tokenized elements of the text. 
From the terminal, the user can insert the text they want to test. 

Examples of usage from terminal (parameters from users may not work when used in a notebook environment)
```bash
python3 predictor.py -k 2 -m
```
will prompt multilabel prediction (with 2 labels as specified after k) and with mapping the ID tassonomia to the respective Campo (field) and Macroambito (macrocategory). 
For no mapping and a single label, by default k = 1, so you can just call:

```
python3 predictor.py
```





