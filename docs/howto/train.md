# Training FamilyAudit-Classifier

In case of availability of new data it is advised to first check its compatibility through the specifications of [data preprocessing](https://github.com/FluveFV/faudit-classifier/blob/main/docs/howto/process.md).

The project has been implemented in Docker. In case a proper Docker container hasn't been built, check [set-up for Docker](https://github.com/FluveFV/faudit-classifier/blob/main/docs/howto/docker.md).

Simply running the default training mode can be done from command line as

``` bash
docker run --gpus '"device=*"' --rm -ti --shm-size=32gb \
    -v $PWD:/src \
    --workdir /src \
    dockerimagename \
    python train.py
```

The training was executed on one GPU that exists in a cluster. Only one was specified, as the model does not require exceptional computational power.

All parameters for training can be modified from the training script `train.py`

For example, to train and augment the patience for a later automatic stopping of the model, one can modify:

``` python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]  #Here
)
```

Or similarly, one can modify the epochs etc. in the training arguments within the script

``` python
training_args = TrainingArguments(
    output_dir='results/',
    eval_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=5,
    num_train_epochs=20,  #Here
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    weight_decay=0.005,
    learning_rate=1e-5,
    lr_scheduler_type='linear',
    load_best_model_at_end=True,
)
```

Since the nature of the taxonomy ("ID tassonomia") includes many labels, during training the model can tend to focus on more frequent classes. To avoid this, the training is carried on with the objective to focus on all classes, and the evaluation is weighted on the inverse frequency of the classes with weighted Cross Entropy Loss.

On the other hand, almost absent classes shouldn't hinder the overall evaluation on the test test, as it is expected for them to not appear as much in future input data. For this reason, while training is done to try and learn for all classes, the final evaluation on test set is done considering a weighted per-class average of correctly predicted observations - the micro F1 weighted measure. For short, the training Loss forces the model to be more influenced by smaller classes, while the performance is more influenced by the bigger classes.

Thus, Cross Entropy Loss is used with

-   Weights (computed on the inverse relative frequency of classes in the sample)
-   Label smoothing (0.1) to account for possible mistakes that occur in the data between the text describing the action and the wrong label.

$$
CrossEntropyLoss=−{\Sigma}^{C}​w_i​⋅y_i​⋅log(p_i​)
$$ with **p** as a probability vector, computed as: $$
p_c= \left[\Sigma_{j=1}^{n}j\right]^{-1} , {\forall} {c} \in [1, ..., C]
$$

The metric of evaluation for the performance is micro F1 score, with - Weights (computed on the frequency of classes in the sample)

$$
\text{Micro F1} = \frac{2 \times \text{Micro Precision} \times \text{Micro Recall}}{\text{Micro Precision} + \text{Micro Recall}}
$$

where Micro Precision and Micro Recall are defined as:

$$
\text{Micro Precision} = \frac{\text{TP}_1 + \text{TP}_2 + \cdots + \text{TP}_n}{\text{TP}_1 + \text{TP}_2 + \cdots + \text{TP}_n + \text{FP}_1 + \text{FP}_2 + \cdots + \text{FP}_n}
$$

$$
\text{Micro Recall} = \frac{\text{TP}_1 + \text{TP}_2 + \cdots + \text{TP}_n}{\text{TP}_1 + \text{TP}_2 + \cdots + \text{TP}_n + \text{FN}_1 + \text{FN}_2 + \cdots + \text{FN}_n}
$$ In a multiclass setting, true positives are one entry vs the rest, divided in false positives and false negatives:

-   $\text{TP}_i$: True Positive for class $i$
-   $\text{FP}_i$: False Positive for class $i$
-   $\text{FN}_i$: False Negative for class $i$

Additionally, also standard multiclass accuracy is recorded.

At each step of the training, all the metric were logged into WANDB (Weights & Biases). There are multiple sections that can be unfrozen for that matter, like:

```         
# WANDB config
#project_name = P1
#wandb.init(project="{project_name}", name=f"{name}", config={})
```

# End of training

The F1 score on test data is printed out in the terminal at the end of the process.

If the data is compatible and the choice of parameters does not raise any errors, the training will come to an end, and train.py will automatically save the model configuration (model architecture, weights, etc.) in `/tuned_model`.

The results of training can be further analyzed from the output file inside `/results` that contains the predictions of the test set and the ground truth, along with the positions of the test set observations.
