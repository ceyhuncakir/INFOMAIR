import os
from typing import Tuple, List, Union

import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import typer
from typing_extensions import Annotated
from loguru import logger
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from helpers.callbacks import *
from helpers.base import Base
from helpers.evaluation import Evaluate

decisiontree_app = typer.Typer()

class DecisionTree(Base):
    """
    This class houses the training, evaluation and usage of the decision tree model.

    Attributes:
        dataset_dir_path (str): The dataset directory path where the original dialog acts dataset resides in.
        checkpoint_dir_path (str): The checkpoint directory path where the trainable decision tree will be saved in.
        experiment_name (str): The experiment name of the decision tree that will be saved as.
        deduplication (bool): Whether the decision tree should be trained on deduplicated data from dialog acts dataset.
    """

    def __init__(
        self,
        dataset_dir_path: str,
        vectorizer_dir_path: str,
        checkpoint_dir_path: str,
        experiment_name: str,
        deduplication: bool
    ) -> None:
        
        self._dataset_dir_path = dataset_dir_path
        self._vectorizer_dir_path = vectorizer_dir_path
        self._checkpoint_dir_path = checkpoint_dir_path
        self._experiment_name = experiment_name

        self._train, self._test, self._labels, self._majority = self.process(
            deduplication=False
        )

        self._vectorizer = self._load_vectorizer()

        self._train_sparse_matrix, self._test_sparse_matrix = self._vectorize(
            train=self._train,
            test=self._test
        )
        
    @logger.catch
    def _train_model(
        self,
        train: pd.DataFrame,
        max_depth: int,
        min_samples_split: int,
        min_samples_leaf: int
    ) -> DecisionTreeClassifier:
        """
        This function is needed to train the decision tree classifier model.

        Args:
            train (pd.DataFrame): The training set that will be used to train the decision tree classifier model.
            max_depth (int): The max depth of the decision tree model.
            min_samples_split (int): The minimum amount of samples per split.
            min_samples_leaf (int): The minimum amount of samples per leaf.
        
        Returns:
            DecisionTreeClassifier: The trained decision tree classifier model.
        """

        labels_train = train['y_true'].values

        clf = DecisionTreeClassifier(
            criterion="log_loss", 
            max_depth=max_depth, 
            min_samples_split=min_samples_split, 
            min_samples_leaf=min_samples_leaf
        )    

        clf = clf.fit(self._train_sparse_matrix, labels_train)

        return clf

    @logger.catch()
    def _load_vectorizer(
        self
    ) -> Union[CountVectorizer, TfidfVectorizer]:
        """
        This function is needed to load the trained vectorizer model.

        Args:
            None
        
        Returns:
            Union[CountVectorizer, TfidfVectorizer]: The trained vectorizer model
        """
        
        with open(self._vectorizer_dir_path, 'rb') as f:
            vectorizer = pickle.load(f)

        return vectorizer

    @logger.catch
    def _vectorize(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function is needed to vectorize the given dataframes.

        Args:
            train (pd.DataFrame): The training dataframe that needs to be vectorized.
            test (pd.DataFrame): The test dataframe that needs to be vectorized.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: The vectorized training and test dataframes.
        """

        train_corpus = train['utterance'].values
        test_corpus = test['utterance'].values

        train_sparse_matrix = self._vectorizer.transform(train_corpus)
        test_sparse_matrix = self._vectorizer.transform(test_corpus)

        return train_sparse_matrix, test_sparse_matrix
    
    @logger.catch
    def _save_model(
        self,
        model: DecisionTreeClassifier
    ) -> None:
        """
        This function is needed to save the trained decision tree model.

        Args:
            model (DecisionTreeClassifier): The trained decision tree model that will be saved. 
        
        Returns:
            None
        """

        os.makedirs(f"{self._checkpoint_dir_path}/{self._experiment_name}", exist_ok=True)
        
        with open(f'{self._checkpoint_dir_path}/{self._experiment_name}/decision_tree.pkl', 'wb') as f:
            pickle.dump(model, f)

    @logger.catch
    def _load_model(
        self
    ) -> DecisionTreeClassifier:
        """
        This function is needed to load the trained decision tree model.

        Args:
            None
        
        Returns:
            DecisionTreeClassifier: The trained decision tree model that has been loaded.
        """

        with open(f'{self._checkpoint_dir_path}/{self._experiment_name}/decision_tree.pkl', 'rb') as f:
            decisiontree = pickle.load(f)
            
        return decisiontree

    @logger.catch 
    def _evaluate_test(
        self,
        model: DecisionTreeClassifier,
        test: pd.DataFrame,
        labels: List[str]
    ) -> pd.DataFrame:
        """
        This function is needed to evaluate the decision tree model using the test set.

        Args:
            model (DecisionTreeClassifier): The trained decision tree model that will be evaluated.
            test (pd.DataFrame): The test set that will be used to evaluate the decision tree model.
            labels (List[str]): The list of labels that will be used to evaluate the decision tree model.
        
        Returns:
            pd.DataFrame: The evaluated
        """
        
        y_pred = model.predict(self._test_sparse_matrix)
        test["y_pred"] = y_pred

        return test

    @logger.catch
    def inference(
        self,
        utterance: str
    ) -> str:
        """
        This function is needed to make predictions using the decision tree model.

        Args:   
            utterance (str): The utterance that will be used to make predictions.
        
        Returns:
            str: The predicted dialog
        """

        decisiontree = self._load_model()

        utterance = utterance.lstrip()
        
        sparse_matrix = self._vectorizer.transform([utterance])

        y_preds_proba = decisiontree.predict_proba(sparse_matrix)

        index_array = np.argmax(y_preds_proba, axis=-1)

        categorical_pred = self._labels[index_array[0]]

        return categorical_pred, y_preds_proba[0][index_array]
    
    @logger.catch
    def evaluate(
        self
    ) -> None:
        """
        This function is needed to evaluate the decision tree model using the test set.

        Args:   
            None    
        
        Returns:
            None
        """

        decisiontree = self._load_model()
        
        results = self._evaluate_test(
            model=decisiontree,
            test=self._test,
            labels=self._labels
        )
        
        Evaluate(
            experiment=self._experiment_name,
            dataframe=results,
            labels=self._labels
        ).run()

    @logger.catch
    def run(
        self,
        max_depth: int,
        min_samples_split: int,
        min_samples_leaf: int
    ) -> None:
        """
        This function is needed to train the decision tree model.

        Args:
            max_depth (int): The max depth of the decision tree model.
            min_samples_split (int): The minimum amount of samples per split.
            min_samples_leaf (int): The minimum amount of samples per leaf.
        
        Returns:
            None
        """

        decisiontree = self._train_model(
            train=self._train,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )

        self._save_model(model=decisiontree)

@decisiontree_app.command()
def evaluate(
    dataset_dir_path: Annotated[str, typer.Option(help="The dataset directory path where the original dialog acts dataset resides in.", callback=path_valid)] = os.getcwd() + "/data/dialog_acts.dat",
    vectorizer_dir_path: Annotated[str, typer.Option(help="The vectorizer directory path where the trained vectorizer model resides in.", callback=path_valid)] = os.getcwd() + "/data/vectorizer/tfidf_vectorizer.pkl",
    checkpoint_dir_path: Annotated[str, typer.Option(help="The checkpoint directory path where the trainable xgboost model will be saved in.", callback=path_valid)] = os.getcwd() + "/data/decisiontree",
    experiment_name: Annotated[str, typer.Option(help="The experiment name of the xgboost model that will be saved as.", callback=experiment_value)] = "decisiontree-tfidf-dupe",
    deduplication: Annotated[bool, typer.Option(help="Whether the decisiontree model should be trained on deduplicated data from dialog acts dataset.", callback=deduplication_value)] = False,
) -> None:

    DecisionTree(
        dataset_dir_path=dataset_dir_path,
        vectorizer_dir_path=vectorizer_dir_path,
        checkpoint_dir_path=checkpoint_dir_path,
        experiment_name=experiment_name,
        deduplication=deduplication
    ).evaluate()

@decisiontree_app.command()
def inference(
    dataset_dir_path: Annotated[str, typer.Option(help="The dataset directory path where the original dialog acts dataset resides in.", callback=path_valid)] = os.getcwd() + "/data/dialog_acts.dat",
    vectorizer_dir_path: Annotated[str, typer.Option(help="The vectorizer directory path where the trained vectorizer model resides in.", callback=path_valid)] = os.getcwd() + "/data/vectorizer/tfidf_vectorizer.pkl",
    checkpoint_dir_path: Annotated[str, typer.Option(help="The checkpoint directory path where the trainable decisiontree model will be saved in.", callback=path_valid)] = os.getcwd() + "/data/decisiontree",
    experiment_name: Annotated[str, typer.Option(help="The experiment name of the decisiontree model that will be saved as.", callback=experiment_value)] = "decisiontree-tfidf-dupe",
    deduplication: Annotated[bool, typer.Option(help="Whether the decisiontree model should be trained on deduplicated data from dialog acts dataset.", callback=deduplication_value)] = False,
) -> None:
    
    decisiontree = DecisionTree(
        dataset_dir_path=dataset_dir_path,
        vectorizer_dir_path=vectorizer_dir_path,
        checkpoint_dir_path=checkpoint_dir_path,
        experiment_name=experiment_name,
        deduplication=deduplication
    )

    while True:

        utterance = input("Enter your utterance: ")

        categorical_pred, probability = decisiontree.inference(utterance=utterance.lower())

        print(f"act: {categorical_pred}, probability: {probability}\n")

@decisiontree_app.command()
def train(
    dataset_dir_path: Annotated[str, typer.Option(help="The dataset directory path where the original dialog acts dataset resides in.", callback=path_valid)] = os.getcwd() + "/data/dialog_acts.dat",
    vectorizer_dir_path: Annotated[str, typer.Option(help="The vectorizer directory path where the trained vectorizer model resides in.", callback=path_valid)] = os.getcwd() + "/data/vectorizer/tfidf_vectorizer.pkl",
    checkpoint_dir_path: Annotated[str, typer.Option(help="The checkpoint directory path where the trainable decisiontree model will be saved in.", callback=path_valid)] = os.getcwd() + "/data/decisiontree",
    experiment_name: Annotated[str, typer.Option(help="The experiment name of the decisiontree model that will be saved as.", callback=experiment_value)] = "decisiontree-tfidf-dupe",
    max_depth: Annotated[int, typer.Option(help="The max depth of the decision tree model.")] = 10,
    min_samples_split: Annotated[int, typer.Option(help="The minimum amount of samples per split.")] = 10,
    min_samples_leaf: Annotated[int, typer.Option(help="The minimum amount of samples per leaf")] = 10,
    deduplication: Annotated[bool, typer.Option(help="Whether the decisiontree model should be trained on deduplicated data from dialog acts dataset.", callback=deduplication_value)] = False,
) -> None:
    
    DecisionTree(
        dataset_dir_path=dataset_dir_path,
        vectorizer_dir_path=vectorizer_dir_path,
        checkpoint_dir_path=checkpoint_dir_path,
        experiment_name=experiment_name,
        deduplication=deduplication
    ).run(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf
    )

