#  FamilyAudit-Classifier
## ⚠️⚠️⚠️ Outdated ⚠️⚠️⚠️

This repository is an older and incompatible version of [P1-ClassificatoreAzioni](https://github.com/FluveFV/P1-ClassificatoreAzioni.git) .  
It is preserved for reference purposes only. Please use the newer version for current work.


#### AIxPA

-   `kind`: product-template
-   `ai`: NLP
-   `domain`: PA

Multiclass sequence classifier based on BERT base italian, fine-tuned on selected corpora from Municipalities's Family Audit plans. 

The context in which this project was developed:
The classifier is trained to suggest one or more labels within the Family Audit framework. More specifically, the classifier can predict the category of the action ("azione") of the text describing it.
Given that the category of action for Municipalities is univocally connected with one macrocategory ("macrocategoria") and one field ("campo"), this classifier can be used to indicate which macrocategory and field of the action the text belongs to. 

The classifier is a tool that can be actually used for **any classification** domain, given similar or bigger amounts of text data. It does not matter which are the labels to predict, but they need to be encoded as integers (e.g. "Ecology" is represented by 1, "Childcare" is represented by 2, etc.). 

The model can be trained for further fine-tuning on new data. 
At the time of writing, this ```README``` is structured on similar tools such as [EuroVoc-Classifier](https://github.com/tn-aixpa/eurovoc-classifier/blob/main/README.md) and its [previous version](https://github.com/bocchilorenzo/AutoEuroVoc/blob/main/README.md).


## Usage

The usage for this template is to facilitate the integration of AI in PA-user interfaces:
- PA operators (from municipalities, regional operators, etc.)
More details in the usage section [here](./docs/howto).


## How To

-   [Preprocess corpora for training](./src/preprocess.ipynb)
-   [Train the classifier model](./docs/howto/train.md)
-   [Predict labels given a new plan](./docs/howto/predict.md)

There is a short version of the data used for training [here](./src/addestramento.gzip).

## License

[Apache License 2.0](./LICENSE)

