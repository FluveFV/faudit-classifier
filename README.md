#  FamilyAudit-Classifier

#### AIxPA

-   `kind`: product-template
-   `ai`: NLP
-   `domain`: PA

Multiclass sequence classifier based on BERT base italian, fine-tuned on selected corpora from Municipalities and organizations. 
The classifier is trained to suggest one or more labels from the taxonomy needed to categorize Family Audit plans in use by Municipalities and organizations involved. 
When given a title, a description and an objective the classifier can predict the appropriate label from the taxonomy in use. 
The model can be trained for further fine-tuning on new data. 
At the time of writing, this README is structured on similar tools such as [EuroVoc-Classifier](https://github.com/tn-aixpa/eurovoc-classifier/blob/main/README.md) and its [previous version](https://github.com/bocchilorenzo/AutoEuroVoc/blob/main/README.md).
This project has been implemented in docker. 

## Usage

The usage for this template is to facilitate the integration of AI in two PA-user interfaces:
- PA operators (from municipalities, regional operators, etc.)
- Organization operators
More details in the usage section [here](./docs/usage.md).

## How To

-   [Set up Docker](./docs/howto/docker.md)
-   [Preprocess corpora for training](./docs/howto/process.md)
-   [Train the classifier model](./docs/howto/train.md)
-   [Predict labels given a new plan](./docs/howto/predict.md)

## License

[Apache License 2.0](./LICENSE)

