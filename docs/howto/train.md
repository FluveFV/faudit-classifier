# Training FamilyAudit-Classifier

Italian 🇮🇹 version below.

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

In the command above, the `\*` was replaced with the computing unit's ID and `dockerimagename` was also replaced with the docker image made for the purpose of training.

The training was executed on one GPU that exists in a cluster. Only one was specified, as the model does not require exceptional computational power.

## Model hyperparameters

Moving on, all parameters for training can be modified from the training script `train.py`

For example, to train and augment the patience for a later automated stopping of the model, one can modify:

``` python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]  #Here
)
```

Or similarly, one can modify the epochs in the training arguments within the script, or other arguments.

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

## End of training

The F1 score on test data is printed out in the terminal at the end of the process.

If the data is compatible and the choice of parameters does not raise any errors, the training will come to an end, and train.py will automatically save the model configuration (model architecture, weights, etc.) in `/tuned_model`.

The results of training can be further analyzed from the output file inside `/results` that contains the predictions on the test set and the ground truth, along with the positions of the test set observations that represent the positions of the observations in the input dataset.

### Resume training
As the state of the model is saved at each checkpoint, one can resume the training - or continue training any pre-trained version, in case there is new data available. Inside of bert_wrapper.py this line can be added inside ```TrainerHandler``` under ```def run(self)```:
``` python
trainer.train(resume_from_checkpoint=True)
```


# Addestrare il classificatore per FamilyAudit

Nell'occorrenza di nuovi dati per addestrare il modello, si consiglia di controllare la compatibilità con il documento [data preprocessing](https://github.com/FluveFV/faudit-classifier/blob/main/docs/howto/process.md).

Questo progetto è stato implementato in Docker. In caso un container Docker non sia stato appropriatemente costruito per l'addestramento, controllare il documento [set-up for Docker](https://github.com/FluveFV/faudit-classifier/blob/main/docs/howto/docker.md).

Per addestrare il modello dalla linea di comando si può semplicemente utilizzare

``` bash
docker run --gpus '"device=*"' --rm -ti --shm-size=32gb \
    -v $PWD:/src \
    --workdir /src \
    dockerimagename \
    python train.py
```

Nel comando qui sopra, `\*` è stato rimpiazzato con il nome dell'unità di computazione, e `dockerimagename` con il nome dell'immagine Docker creata per addestrare il modello.

L'addestramento è stato eseguito su una GPU che esiste all'interno di un cluster. Solo una era necessaria data la ridotta necessità computazionale del modello.

In seguito, si descrive come modificare i parametri per l'addestramento all'interno di `train.py`.

Per esempio, è possibile addestrare aumentando la *patience* per l'early stopping dell'apprendimento, modificando:

``` python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]  #Qui
)
```

Alternativamente, è possibile modificare le ere di addestramento negli argomenti all'interno di `training_args`, o gli altri parametri se necessario.

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

A causa della natura della tassonomia (nei dati originali: "ID tassonomia") che include molte voci l'addestramento tende a focalizzarsi di più sulle classi più frequenti. Per evitare che questo succeda, l'addestramento è eseguito con l'obiettivo di apprendere tutte le classi, e la valutazione dell'apprendimento è quindi soppesata dall'inverso delle frequenze delle classi con la weighted Cross Entropy Loss.

D'altro canto, le classi che sono quasi inesistenti nel campione non dovrebbero peggiorare vistosamente la performance, poiché ci si aspetta che esse non compariranno altrettanto in futuro. Per questo motivo, sebbene l'addestramento sia fatto con l'obiettivo di apprendere tutte le classi, la valutazione finale della performance sul test set è fatta considerando soltanto una media soppesata dalla frequenza delle classi per le osservazioni correttamente predette: la weighted micro F1.

In breve, la Cross Entropy Loss forza l'apprendimento ad essere più influenzato dalle classi più infrequenti, mentre la valutazione della performance generale è più influenzata dalle classi più frequenti.

Infine, la Cross Entropy Loss è implementata con:

-   Pesi (calcolati sull'inverso delle frequenze delle classi)
-   Label Smoothing (0.1) per tenere in considerazione i possibili errori che sono presenti nei dati tra un testo e la voce della tassonomia.

$$
CrossEntropyLoss=−{\Sigma}^{C}​w_i​⋅y_i​⋅log(p_i​)
$$ con **p** come vettore di probabilità per cui $$
p_c= \left[\Sigma_{j=1}^{n}j\right]^{-1} , {\forall} {c} \in [1, ..., C]
$$

La metrica di valutazione F1 score è intesa come

$$
\text{Micro F1} = \frac{2 \times \text{Micro Precision} \times \text{Micro Recall}}{\text{Micro Precision} + \text{Micro Recall}}
$$

dove la MicroPrecision e la MicroRecall sono

$$
\text{Micro Precision} = \frac{\text{TP}_1 + \text{TP}_2 + \cdots + \text{TP}_n}{\text{TP}_1 + \text{TP}_2 + \cdots + \text{TP}_n + \text{FP}_1 + \text{FP}_2 + \cdots + \text{FP}_n}
$$

$$
\text{Micro Recall} = \frac{\text{TP}_1 + \text{TP}_2 + \cdots + \text{TP}_n}{\text{TP}_1 + \text{TP}_2 + \cdots + \text{TP}_n + \text{FN}_1 + \text{FN}_2 + \cdots + \text{FN}_n}
$$ Dato il contesto del problema multiclasse, i True Positives sono una voce della confusion matrix, e il resto sono divisi in falsi positivi e falsi negativi:

-   $\text{TP}_i$: True Positive per la classe $i$
-   $\text{FP}_i$: False Positive per la classe $i$
-   $\text{FN}_i$: False Negative per la classe $i$

In aggiunta a queste misure, anche l'*accuracy* è registrata tra le metriche di valutazione dell'addestramento e metriche di performance sul test set.

Ad ogni passo dell'addestramento, tutte le metriche erano pensate per essere registrate e visualizzate tramite WANDB (Weights & Biases). Vi sono multiple sezioni che devono essere sbloccate nel caso si voglia usarle, ad esempio:

```         
# WANDB config
#project_name = P1
#wandb.init(project="{project_name}", name=f"{name}", config={})
```

# Termine addestramento

La misura F1 è stampata nel terminale al termine del processo.

Se i dati sono compatibili e la scelta dei parametri non porta a nessun errore, l'addestramento sarà portato a termine e `train_py` salva automaticamente la configurazione del modello per altre attività downstream (architettura del modello, pesi, etc.) in `/tuned_model`.

I risultati dell'addestramento possono essere analizzati (magari sotto altre metriche) nel file di output all'interno della cartella `/results`, che contiene le predizioni del modello sul test set e la ground truth, assieme agli indici delle osservazioni del test set che rappresentano la loro posizione nel dataset di input.
