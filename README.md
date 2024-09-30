# INFOMAIR
In our group assignment, we are developing a **text-based dialog system** for restaurant recommendations. The system will interact with users to gather their preferences and suggest suitable restaurants from a database. To do this, we will use an existing dataset of annotated dialogs (**dialog_acts.dat**) in the restaurant domain.

## Project Structure

The project is divided into three key parts:

### 1. Utterance Classification
We will apply **classifiers** both rule based and machine learning models to categorize user utterances based on their role in the dialog (e.g., request, confirmation, etc.).

### 2. Dialog Manager
We will construct a **dialog transition model** and implement a dialog manager to facilitate seamless interactions between the user and the system.

### 3. Reasoning and Configuration
We will add a **reasoning component** to the system and implement configurable variations. This will prepare us for Part 2, where we will conduct a **user evaluation experiment** to assess the system’s usability and performance.
        
# Dependecies
Bofore installing the project for usage, please make sure you have installed the following dependecies:

```
pipenv
```

You can install the dependecies by running the following pip3 command: 

```
pip3 install pipenv
```

# Installation
When the right dependecies have been installed, run the following command inside the project directory:

```
pipenv install
```

# Data
The dialog_acts.dat data can be originally be found on the following site:

```
https://www.microsoft.com/en-us/research/event/dialog-state-tracking-challenge/
```

# Model Checkpoints
When you train any model that needs to be saved in a data directory, it’s important to always provide the correct path. For example:

Assume we have the following project structure:
```
INFOMAIR/
├── 1a/
│   ├── data/
│   │   └── dialog_acts.dat
│   ├── src/
├── 1b/
```

Now, let's say we want to train a vectorizer model and save it within this structure. Here's an example pipenv command:

```
pipenv run 1a vectorizer build --dataset-dir-path ~/INFOMAIR/1a/data/dialog_acts.dat --vectorizer-type tfidf --checkpoint-dir-path ~/INFOMAIR/1a/data/vectorizer
```

After running the above command, the project structure will look like this:

```
INFOMAIR/
├── 1a/
│   ├── data/
│   │   ├── vectorizer/
│   │   │   └── tfidf_vectorizer.pkl
│   │   └── dialog_acts.dat
│   ├── src/
├── 1b/
```

Next, assume you want to train a machine learning model (such as a decision tree) using the saved vectorizer. You can use the following pipenv command:

```
pipenv run 1a decisiontree train --dataset-dir-path ~/INFOMAIR/1a/data/dialog_acts.dat --vectorizer-dir-path ~/INFOMAIR/1a/data/vectorizer/tfidf_vectorizer.pkl --checkpoint-dir-path ~/INFOMAIR/1a/data/decisiontree --experiment-name decisiontree-tfidf --max-depth 15 --min-samples-split 20 --min-samples-leaf 100 --no-deduplication
```

After running this command, the project structure will be updated to:
```
INFOMAIR/
├── 1a/
│   ├── data/
│   │   ├── vectorizer/
│   │   │   └── tfidf_vectorizer.pkl
│   │   ├── decisiontree/
│   │   │   └── decisiontree-tfidf/
│   │   │       └── decision_tree.pkl
│   │   └── dialog_acts.dat
│   ├── src/
├── 1b/
```

This is just one example. Always ensure you use the correct paths when running commands.

# Usage
Before actually using the act classifiers you should note that for the machine learning models you first need to train the models in order to do any real predictions on the test set. For inference this is the same case. Before using any machine learning model, the dataset must first be vectorized using a chosen vectorizer. It is recommended to fit the vectorizer before applying it to the model. To build your own vectorizer, you can run the following pipenv command. The example commands are not necessary due to having default values set for each command, but for specific usage please refer to the example commands that have been defined in each section.


# statistics
To get some statistics from the **dialog_acts.dat** dataset you can use the following pipenv command:
```
pipenv run 1a statistics run --dataset-dir-path [DIALOG DATASET PATH] --deduplication [DEDUPLICATION FLAG]
```


# 1A
## Baseline's
To run any of the baselines run the following command respectively to each baseline.
### Evaluation
To run evaluation use the following pipenv command:
```
pipenv run 1a [BASELINE E.G: baseline_1, baseline_2] evaluate --dataset_dir_path [DIALOG DATASET PATH]
```

### Inference 
To run inference use the following pipenv command:
```
pipenv run 1a [BASELINE E.G: baseline_1, baseline_2] inference --dataset_dir_path [DIALOG DATASET PATH]
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
pipenv run 1a decisiontree evaluate --dataset-dir-path [DATASET DIRECTORY PATH] --vectorizer-dir-path [VECTORIZER DIRECTORY PATH] --checkpoint-dir-path [CHECKPOINT DIRECTORY PATH] --experiment-name [EXPERIMENT NAME] --deduplication [DEDUPLICATION FLAG]
```

### Inference
To run the inference on the trained decision tree model run the following pipenv commnad:
```
pipenv run 1a decisiontree inference --dataset-dir-path [DATASET DIRECTORY PATH] --vectorizer-dir-path [VECTORIZER DIRECTORY PATH] --checkpoint-dir-path [CHECKPOINT DIRECTORY PATH] --experiment-name [EXPERIMENT NAME] --deduplication [DEDUPLICATION FLAG]
```

## Logistic Regression (ml)
### Train
To train the Logistic Regression model run the following command:

```
pipenv run 1a logistic_regression train --dataset-dir-path [DATASET DIRECTORY PATH] --vectorizer-dir-path [VECTORIZER DIRECTORY PATH] --checkpoint-dir-path [CHECKPOINT DIRECTORY PATH] --experiment-name [EXPERIMENT NAME] ----max-iter [MAXIMUM AMOUNT OF ITERATIONS] --verbose [BOOL] --deduplication [DEDUPLICATION FLAG]
```

### Evaluation
To evaluate the Logistic Regression model run the following pipenv command:

```
pipenv run 1a logistic_regression evaluate --dataset-dir-path [DATASET DIRECTORY PATH] --vectorizer-dir-path [VECTORIZER DIRECTORY PATH] --checkpoint-dir-path [CHECKPOINT DIRECTORY PATH] --experiment-name [EXPERIMENT NAME] --deduplication [DEDUPLICATION FLAG]
```

### Inference
To run the inference on the trained Logistic Regression model run the following pipenv commnad:
```
pipenv run 1a logistic_regression --dataset-dir-path [DATASET DIRECTORY PATH] --vectorizer-dir-path [VECTORIZER DIRECTORY PATH] --checkpoint-dir-path [CHECKPOINT DIRECTORY PATH] --experiment-name [EXPERIMENT NAME] --deduplication [DEDUPLICATION FLAG]
```

## MLP (ml)
### Train
To train the MLP model run the following pipenv command:
```
pipenv run 1a mlp train --dataset-dir-path [DATASET DIRECTORY PATH] --vectorizer-dir-path [VECTORIZER DIRECTORY PATH] --checkpoint-dir-path [CHECKPOINT DIRECTORY PATH] --experiment-name [EXPERIMENT NAME] --device [DEVICE] --eta [LEARNING RATE] --batch-size [BATCH SIZE] --epochs [EPOCHS] --deduplication [DEDUPLICATION FLAG]
```

### Evaluation
To now evaluate the MLP model run the following command:

```
pipenv run 1a mlp evaluate --dataset-dir-path [DATASET DIRECTORY PATH] --vectorizer-dir-path [VECTORIZER DIRECTORY PATH] --checkpoint-dir-path [CHECKPOINT DIRECTORY PATH] --experiment-name [EXPERIMENT NAME] --device [DEVICE] --deduplication [DEDUPLICATION FLAG]
```
### inference
TO run the Trained MLP model in inference run the following pipenv command
```
pipenv run 1a mlp inference --dataset-dir-path [DATASET DIRECTORY PATH] --vectorizer-dir-path [VECTORIZER DIRECTORY PATH] --checkpoint-dir-path [CHECKPOINT DIRECTORY PATH] --experiment-name [EXPERIMENT NAME] --device [DEVICE] --deduplication [DEDUPLICATION FLAG]
```

# 1B - 1C
## Logistic Regression based dialogue

To use the Logistic Regression based dialogue manager, run the following command:
```
 pipenv run 1b-c dialog_manager 
```
or 
```
python 1b-c/src/main.py dialog_manager run
```
You can add the following configurability arguments:

### Add 3 second delay before every system utterance
```
--do-delay
```
### Disable or change levenshtein edit distance
```
--levenshtein_dist [int]
```
### Enable continious results. Shows remainig results after every user utterance
```
--do-continious-results
```
### Ase keyword baseline as a classifier, instead of linear regression
```
--use-baseline
```


# Contributors
```
Ceyhun Cakir (c.cakir@students.uu.nl)
Simon Hart (s.f.hart@students.uu.nl)
Bo van Westerlaak (b.vanwesterlaak@students.uu.nl)
Akshaj Agarwal (a.agarwal2@students.uu.nl)
```
