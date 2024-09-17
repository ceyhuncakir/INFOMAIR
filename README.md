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
The data can be originally be found on the following site:

```
https://www.microsoft.com/en-us/research/event/dialog-state-tracking-challenge/
```

# Usage
Before actually using the act classifiers you should note that for the machine learning models you first need to train the models in order to do any real predictions on the test set. For inference this is the same case. Before training any machine learning model make sure you also train the doc2vec model first before running the train commands. Since we generate embeddings based on the trained do2vec model, we can train the machine learning models before doing so.

## Baseline 1
### Evaluation
To run the evaluation of this baseline you can run the following pipenv command:

```
pipenv run a1 baseline_1 evaluate --dataset_dir_path [DIALOG DATASET PATH]
```
### Inference
To do the inference part of this baseline you can run the following pipenv command:

```
pipenv run a1 baseline_1 inference --dataset_dir_path [DIALOG DATASET PATH]
```

## Baseline 2 (rule-based)
### Evaluation
To run the evaluation of the baseline 2, you can run the following example command.

```
pipenv run a1 baseline_2 evaluate --dataset_dir_path [DIALOG DATASET PATH]
```

### Inference
To do the inference part of this baseline you can run the following pipenv command:

```
pipenv run a1 baseline_2 inference --dataset_dir_path [DIALOG DATASET PATH]
```
## Doc2Vec
### Train
To train the necessary Doc2Vec model you can use the following pipenv command:

```
pipenv run a1 doc2vec train --data-dump-data-dir [DIALOG DATASET PATH] --checkpoint-data-dir [THE PATH OF DOC2VEC DIRECTORY] --experiment-name [THE NAME OF THE EXPERIMENT] --vector-size [THE DIMENSIONS OF THE VECTOR] --min_count [FREQ COUNT OF WORDS] --epochs [THE AMOUNT OF EPOCHS] --preprocess_lower [wHETHER TO HAVE LOWERCASED STRINGS] --preprocess-token-max-len [THE MAXIMUM LENGTH OF A TOKEN] --preprocess-processes [THE AMOUNT OF PROCESSES]
```

## Xgboost (ml)
### Train
To train the xgboost model run the following command:

```
pipenv run a1 xgboost train --dataset-dir-path [DATASET DIRECTORY PATH] --doc2vec-data-dir-path [DOC2VEC DIRECTORY PATH] --checkpoint-dir-path [CHECKPOINT DIRECTORY PATH] --experiment-name [EXPERIMENT NAME] --verbosity [VERBOSITY LEVEL] --num-rounds [BOOSTING ROUNDS] --deduplication [DEDUPLICATION FLAG]
```

### Evaluation
To evaluate the xgboost model run the following pipenv command:

```
pipenv run a1 xgboost evaluate --dataset-dir-path [DATASET DIRECTORY PATH] --doc2vec-data-dir-path [DOC2VEC DIRECTORY PATH] --checkpoint-dir-path [CHECKPOINT DIRECTORY PATH] --experiment-name [EXPERIMENT NAME] --deduplication [DEDUPLICATION FLAG]
```

### inference
To run the inference on the trained xgboost model run the following pipenv commnad:
```
pipenv run a1 xgboost inference --dataset-dir-path [DATASET DIRECTORY PATH] --doc2vec-data-dir-path [DOC2VEC DIRECTORY PATH] --checkpoint-dir-path [CHECKPOINT DIRECTORY PATH] --experiment-name [EXPERIMENT NAME] --deduplication [DEDUPLICATION FLAG]
```

## MLP (ml)
### Train
To train the MLP model run the following pipenv command:
```
pipenv run a1 mlp train --dataset-dir-path [DATASET DIRECTORY PATH] --doc2vec-data-dir-path [DOC2VEC DIRECTORY PATH] --checkpoint-dir-path [CHECKPOINT DIRECTORY PATH] --experiment-name [EXPERIMENT NAME] --device [DEVICE] --eta [LEARNING RATE] --batch-size [BATCH SIZE] --epochs [EPOCHS] --deduplication [DEDUPLICATION FLAG]
```

### Evaluation
To now evaluate the MLP model run the following command:

```
pipenv run a1 mlp evaluate --dataset-dir-path [DATASET DIRECTORY PATH] --doc2vec-data-dir-path [DOC2VEC DIRECTORY PATH] --checkpoint-dir-path [CHECKPOINT DIRECTORY PATH] --experiment-name [EXPERIMENT NAME] --device [DEVICE] --deduplication [DEDUPLICATION FLAG]
```
### inference
TO run the Trained MLP model in inference run the following pipenv command
```
pipenv run a1 mlp inference --dataset-dir-path [DATASET DIRECTORY PATH] --doc2vec-data-dir-path [DOC2VEC DIRECTORY PATH] --checkpoint-dir-path [CHECKPOINT DIRECTORY PATH] --experiment-name [EXPERIMENT NAME] --device [DEVICE] --deduplication [DEDUPLICATION FLAG]
```

# Constributors
```
Ceyhun Cakir (c.cakir@students.uu.nl)
```
