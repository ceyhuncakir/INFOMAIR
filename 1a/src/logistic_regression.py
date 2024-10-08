import os
from typing import Tuple, List, Union

import pickle
import pandas as pd
import numpy as np
import typer
from typing_extensions import Annotated
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from helpers.callbacks import *
from helpers.base import Base
from helpers.evaluation import Evaluate

logisticreg_app = typer.Typer()

class LogisticRegressionClassifier(Base):
    """
    This class is needed to train the logistic regression model for the dialogue act classification task.

    Attributes:
        dataset_dir_path (str): The path to the dataset directory.
        vectorizer_dir_path (str): The path to the vectorizer directory.
        checkpoint_dir_path (str): The path to the checkpoint directory.
        experiment_name (str): The name of the experiment.
        deduplication (bool): Whether the model should be trained on deduplicated data.
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
        self._deduplication = deduplication
        
        self._train, self._test, self._labels, self._majority, self._class_weight_dict = self.process(
            deduplication=deduplication
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
        max_iter: int,
        verbose: bool
    ) -> LogisticRegression:
        """
        This function is needed to train the logistic regression model.

        Args:
            train (pd.DataFrame): The training dataframe that needs to be trained on.
            max_iter (int): The maximum number of iterations to be run.
            verbose (bool): Whether to print out logs.
        
        Returns:
            LogisticRegression: The trained logistic regression model.
        """

        labels_train = [self._labels.index(act) for act in train['y_true']]

        clf = LogisticRegression(
            max_iter=max_iter,
            verbose=verbose,
        ) 

        clf = clf.fit(self._train_sparse_matrix, labels_train)

        return clf

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
    def _save_model(
        self,
        model: LogisticRegression
    ) -> None:
        """
        This function is needed to save the trained logistic regression model.

        Args:
            model (LogisticRegression): The trained logistic regression model that needs to be saved.
        
        Returns:
            None
        """

        os.makedirs(f"{self._checkpoint_dir_path}/{self._experiment_name}", exist_ok=True)
        
        with open(f'{self._checkpoint_dir_path}/{self._experiment_name}/logistic_regression.pkl', 'wb') as f:
            pickle.dump(model, f)

    @logger.catch
    def _load_model(
        self
    ) -> LogisticRegression:
        """
        This function is needed to load the trained logistic regression model.

        Args:
            None
        
        Returns:
            LogisticRegression: The trained logistic regression model.
        """

        with open(f'{self._checkpoint_dir_path}/{self._experiment_name}/logistic_regression.pkl', 'rb') as f:
            logistic_regression = pickle.load(f)
            
        return logistic_regression

    @logger.catch 
    def _evaluate_test(
        self,
        model: LogisticRegression,
        test: pd.DataFrame,
        labels: List[str]
    ) -> pd.DataFrame:
        """
        This funtion is needed to evaluate the test dataframe.

        Args:
            model (LogisticRegression): The trained logistic regression model.
            test (pd.DataFrame): The test dataframe that needs to be evaluated.
            labels (List[str]): The list of labels.
        
        Returns:
            pd.DataFrame: The evaluated test dataframe.
        """

        y_pred = model.predict(self._test_sparse_matrix)
        test["y_pred"] = [self._labels[i] for i in y_pred]

        return test

    @logger.catch
    def inference(
        self,
        utterance: str
    ) -> Tuple[str, float]:
        """
        This function is needed to make inferences using the trained logistic regression model.

        Args:
            utterance (str): The utterance that needs to be classified.
        
        Returns:
            str: The predicted dialogue act.
        """

        # Load the trained logistic regression model
        logisticregression = self._load_model()

        # Preprocess the utterance (e.g., remove leading spaces)
        utterance = utterance.lstrip()

        # Transform the utterance using the vectorizer to create a sparse matrix
        sparse_matrix = self._vectorizer.transform([utterance])

        # Get the predicted probabilities for each class
        y_preds_proba = logisticregression.predict_proba(sparse_matrix)

        # Find the index of the class with the highest probability
        index_array = np.argmax(y_preds_proba, axis=-1)

        # Get the predicted class label
        categorical_pred = self._labels[index_array[0]]

        # Get the probability for the predicted class
        predicted_proba = y_preds_proba[0][index_array[0]]

        # Return the predicted class label and the corresponding probability
        return categorical_pred, predicted_proba

    @logger.catch
    def evaluate(
        self
    ) -> None:
        """
        This function is needed to evaluate the logistic regression model.

        Args:
            None
        
        Returns:
            None
        """
        
        logisticregression = self._load_model()
        
        results = self._evaluate_test(
            model=logisticregression,
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
        max_iter: int,
        verbose: bool
    ) -> None:
        """
        This function is needed to train the logistic regression model.

        Args:
            max_iter (int): The maximum number of iterations to be run.
            verbose (bool): Whether to print out logs.
        
        Returns:
            None
        """
        
        logisticregression = self._train_model(
            train=self._train,
            max_iter=max_iter,
            verbose=verbose
        )

        self._save_model(model=logisticregression)

@logisticreg_app.command()
def evaluate(
    dataset_dir_path: Annotated[str, typer.Option(help="The dataset directory path where the original dialog acts dataset resides in.", callback=path_valid)] = os.getcwd() + "/data/dialog_acts.dat",
    vectorizer_dir_path: Annotated[str, typer.Option(help="The vectorizer directory path where the trained vectorizer model resides in.", callback=path_valid)] = os.getcwd() + "/data/vectorizer/tfidf_vectorizer.pkl",
    checkpoint_dir_path: Annotated[str, typer.Option(help="The checkpoint directory path where the trainable decisiontree model will be saved in.", callback=path_valid)] = os.getcwd() + "/data/logistic_regression",
    experiment_name: Annotated[str, typer.Option(help="The experiment name of the decisiontree model that will be saved as.", callback=experiment_value)] = "logisticregression-tfidf-dupe",
    deduplication: Annotated[bool, typer.Option(help="Whether the decisiontree model should be trained on deduplicated data from dialog acts dataset.", callback=deduplication_value)] = False,
) -> None:

    LogisticRegressionClassifier(
        dataset_dir_path=dataset_dir_path,
        vectorizer_dir_path=vectorizer_dir_path,
        checkpoint_dir_path=checkpoint_dir_path,
        experiment_name=experiment_name,
        deduplication=deduplication
    ).evaluate()

@logisticreg_app.command()
def inference(
    dataset_dir_path: Annotated[str, typer.Option(help="The dataset directory path where the original dialog acts dataset resides in.", callback=path_valid)] = os.getcwd() + "/data/dialog_acts.dat",
    vectorizer_dir_path: Annotated[str, typer.Option(help="The vectorizer directory path where the trained vectorizer model resides in.", callback=path_valid)] = os.getcwd() + "/data/vectorizer/tfidf_vectorizer.pkl",
    checkpoint_dir_path: Annotated[str, typer.Option(help="The checkpoint directory path where the trainable decisiontree model will be saved in.", callback=path_valid)] = os.getcwd() + "/data/logistic_regression",
    experiment_name: Annotated[str, typer.Option(help="The experiment name of the decisiontree model that will be saved as.", callback=experiment_value)] = "logisticregression-tfidf-dupe",
    deduplication: Annotated[bool, typer.Option(help="Whether the decisiontree model should be trained on deduplicated data from dialog acts dataset.", callback=deduplication_value)] = False,
) -> None:
    
    logisticregression = LogisticRegressionClassifier(
        dataset_dir_path=dataset_dir_path,
        vectorizer_dir_path=vectorizer_dir_path,
        checkpoint_dir_path=checkpoint_dir_path,
        experiment_name=experiment_name,
        deduplication=deduplication
    )

    while True:

        utterance = input("Enter your utterance: ")

        categorical_pred, probability = logisticregression.inference(utterance=utterance.lower())

        print(f"""act: {categorical_pred}, probability: {probability}\n""")

@logisticreg_app.command()
def train(
    dataset_dir_path: Annotated[str, typer.Option(help="The dataset directory path where the original dialog acts dataset resides in.", callback=path_valid)] = os.getcwd() + "/data/dialog_acts.dat",
    vectorizer_dir_path: Annotated[str, typer.Option(help="The vectorizer directory path where the trained vectorizer model resides in.", callback=path_valid)] = os.getcwd() + "/data/vectorizer/tfidf_vectorizer.pkl",
    checkpoint_dir_path: Annotated[str, typer.Option(help="The checkpoint directory path where the trainable decisiontree model will be saved in.", callback=path_valid)] = os.getcwd() + "/data/logistic_regression",
    experiment_name: Annotated[str, typer.Option(help="The experiment name of the decisiontree model that will be saved as.", callback=experiment_value)] = "logisticregression-tfidf-dupe",
    max_iter: Annotated[int, typer.Option(help="The maximum number of iterations to be run.")] = 100,
    verbose: Annotated[bool, typer.Option(help="Whether to print out logs.")] = False,
    deduplication: Annotated[bool, typer.Option(help="Whether the decisiontree model should be trained on deduplicated data from dialog acts dataset.", callback=deduplication_value)] = False,
) -> None:
    
    LogisticRegressionClassifier(
        dataset_dir_path=dataset_dir_path,
        vectorizer_dir_path=vectorizer_dir_path,
        checkpoint_dir_path=checkpoint_dir_path,
        experiment_name=experiment_name,
        deduplication=deduplication
    ).run(
        max_iter=max_iter,
        verbose=verbose  
    )
