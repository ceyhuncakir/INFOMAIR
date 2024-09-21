# INFOMAIR
This project is about the INFOMAIR group assignment.

# Dependecies
Bofore installing the project for usage, please make sure you have installed the following dependecies:

```
pipenv
```

You can install the dependecie by running the following pip3 command 

```
pip3 install pipenv
```

# Instalation
When the right dependecies have been installed, run the following command inside the project directory:

```
pipenv install
```

# Data
The dialog_acts.dat data can be originally be found on the following site:

```
https://www.microsoft.com/en-us/research/event/dialog-state-tracking-challenge/
```

# Usage
Before actually using the act classifiers you should note that for the machine learning models you first need to train the models in order to do any real predictions on the test set. For inference this is the same case. Before using any machine learning model, the dataset must first be vectorized using a chosen vectorizer. It is recommended to fit the vectorizer before applying it to the model. To build your own vectorizer, you can run the following pipenv command.

# 1A
## Baseline's
To run any of the baselines run the following command respectively to each baseline.
### Evaluation
To run evaluation use the following pipenv command:
```
pipenv run a1 [BASELINE E.G: baseline_1, baseline_2] evaluate --dataset_dir_path [DIALOG DATASET PATH]
```

### Inference 
To run inference use the following pipenv command:
```
pipenv run a1 [BASELINE E.G: baseline_1, baseline_2] inference --dataset_dir_path [DIALOG DATASET PATH]
```

## Vectorizer
### Build
To fit the vectorizer and save it to the chosen directory path you can run the following pipenv command:
```
pipenv run 1a vectorizer build --dataset-dir-path [PATH OF DIALOG_ACTS.DAT] --vectorizer-type [E.G: tfidf, count] --checkpoint-dir-path [PATH OF SAVING]
```

## Decision tree (ml)
### Train
To train the decision tree model run the following command:

```
pipenv run 1a decisiontree train --dataset-dir-path [DATASET DIRECTORY PATH] --vectorizer-dir-path [VECTORIZER DIRECTORY PATH] --checkpoint-dir-path [CHECKPOINT DIRECTORY PATH] --experiment-name [EXPERIMENT NAME] --max-depth [MAX-DEPTH] --min-samples-split [MIN-SAMPLES-SPLIT] --min-samples-leaf [MIN-SAMPLES-LEAF] --deduplication [DEDUPLICATION FLAG]
```

### Evaluation
To evaluate the decision tree model run the following pipenv command:

```
pipenv run a1 decisiontree evaluate --dataset-dir-path [DATASET DIRECTORY PATH] --vectorizer-dir-path [VECTORIZER DIRECTORY PATH] --checkpoint-dir-path [CHECKPOINT DIRECTORY PATH] --experiment-name [EXPERIMENT NAME] --deduplication [DEDUPLICATION FLAG]
```

### inference
To run the inference on the trained decision tree model run the following pipenv commnad:
```
pipenv run a1 decisiontree inference --dataset-dir-path [DATASET DIRECTORY PATH] --vectorizer-dir-path [VECTORIZER DIRECTORY PATH] --checkpoint-dir-path [CHECKPOINT DIRECTORY PATH] --experiment-name [EXPERIMENT NAME] --deduplication [DEDUPLICATION FLAG]
```

## Logistic Regression (ml)
### Train
To train the Logistic Regression model run the following command:

```
pipenv run a1 logistic_regression train --dataset-dir-path [DATASET DIRECTORY PATH] --vectorizer-dir-path [VECTORIZER DIRECTORY PATH] --checkpoint-dir-path [CHECKPOINT DIRECTORY PATH] --experiment-name [EXPERIMENT NAME] ----max-iter [MAXIMUM AMOUNT OF ITERATIONS] --verbose [BOOL] --deduplication [DEDUPLICATION FLAG]
```

### Evaluation
To evaluate the Logistic Regression model run the following pipenv command:

```
pipenv run a1 logistic_regression evaluate --dataset-dir-path [DATASET DIRECTORY PATH] --vectorizer-dir-path [VECTORIZER DIRECTORY PATH] --checkpoint-dir-path [CHECKPOINT DIRECTORY PATH] --experiment-name [EXPERIMENT NAME] --deduplication [DEDUPLICATION FLAG]
```

### inference
To run the inference on the trained Logistic Regression model run the following pipenv commnad:
```
pipenv run a1 logistic_regression --dataset-dir-path [DATASET DIRECTORY PATH] --vectorizer-dir-path [VECTORIZER DIRECTORY PATH] --checkpoint-dir-path [CHECKPOINT DIRECTORY PATH] --experiment-name [EXPERIMENT NAME] --deduplication [DEDUPLICATION FLAG]
```

## MLP (ml)
### Train
To train the MLP model run the following pipenv command:
```
pipenv run a1 mlp train --dataset-dir-path [DATASET DIRECTORY PATH] --vectorizer-dir-path [VECTORIZER DIRECTORY PATH] --checkpoint-dir-path [CHECKPOINT DIRECTORY PATH] --experiment-name [EXPERIMENT NAME] --device [DEVICE] --eta [LEARNING RATE] --batch-size [BATCH SIZE] --epochs [EPOCHS] --deduplication [DEDUPLICATION FLAG]
```

### Evaluation
To now evaluate the MLP model run the following command:

```
pipenv run a1 mlp evaluate --dataset-dir-path [DATASET DIRECTORY PATH] --vectorizer-dir-path [VECTORIZER DIRECTORY PATH] --checkpoint-dir-path [CHECKPOINT DIRECTORY PATH] --experiment-name [EXPERIMENT NAME] --device [DEVICE] --deduplication [DEDUPLICATION FLAG]
```
### inference
TO run the Trained MLP model in inference run the following pipenv command
```
pipenv run a1 mlp inference --dataset-dir-path [DATASET DIRECTORY PATH] --vectorizer-dir-path [VECTORIZER DIRECTORY PATH] --checkpoint-dir-path [CHECKPOINT DIRECTORY PATH] --experiment-name [EXPERIMENT NAME] --device [DEVICE] --deduplication [DEDUPLICATION FLAG]
```

# Constributors
```
Ceyhun Cakir (c.cakir@students.uu.nl)
Simon Hart (s.f.hart@students.uu.nl)
Bo van Westerlaak (b.vanwesterlaak@students.uu.nl)
Akshaj Agarwal (a.agarwal2@students.uu.nl)
```
