import os
from typing import Tuple, List

import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import typer
from typing_extensions import Annotated
from loguru import logger
from sklearn.tree import DecisionTreeClassifier

from helpers.base import Base
from helpers.evaluation import Evaluate

decisiontree_app = typer.Typer()

class DecisionTree(Base):
    """
    This class houses the training, evaluation and usage of the xgboost model.

    Attributes:
        dataset_dir_path (str): The dataset directory path where the original dialog acts dataset resides in.
        checkpoint_dir_path (str): The checkpoint directory path where the trainable xgboost model will be saved in.
        experiment_name (str): The experiment name of the xgboost model that will be saved as.
        deduplication (bool): Whether the xgboost model should be trained on deduplicated data from dialog acts dataset.
    """

    def __init__(
        self,
        dataset_dir_path: str,
        checkpoint_dir_path: str,
        experiment_name: str,
        deduplication: bool
    ) -> None:
        
        self._dataset_dir_path = dataset_dir_path
        self._checkpoint_dir_path = checkpoint_dir_path
        self._experiment_name = experiment_name

        self.df = self._load_data()
        self.df = self.set_columns(df=self.df)
        self.df = self._preprocess(df=self.df, deduplication=deduplication)

        self.labels = self._get_labels(df=self.df)
        self.train, self.model_test = self._split_train_test(df=self.df)
        self.model_train, self.model_validation = self._split_train(df=self.train, labels=self.labels)

    @logger.catch
    def _train_model(
        self,
        train: pd.DataFrame,
        validation: pd.DataFrame,
        max_depth: int,
        min_samples_split: int,
        min_samples_leaf: int
    ) -> None:
        """
        This function is being used as the training function for the xgboost model.

        Args:
            train (pd.DataFrame): A dataframe consisting of the training examples for the xgboost model.
            validation (pd.DataFrame): A dataframe consisting of the validation samples for the xgboost model.
            verbosity (int): The verbosity level thats set for the training phase.
            num_round (int): The number of boosting rounds for the xgboost tree.

        Returns:
            model: The trained xgboost model.
        """

        labels_train = train['y_true'].values
        labels_val = validation['y_true'].values

        # creating input matrix (H X W)
        ndarray_embeddings_train = np.vstack(train['embeddings'].values)
        ndarray_embeddings_val = np.vstack(validation['embeddings'].values)

        clf = DecisionTreeClassifier(
            criterion="log_loss", 
            max_depth=max_depth, 
            min_samples_split=min_samples_split, 
            min_samples_leaf=min_samples_leaf
        )    

        clf = clf.fit(ndarray_embeddings_train, labels_train)

        return clf
    
    @logger.catch
    def _save_model(
        self,
        model
    ) -> None:
        """
        This function is neded to save the trained xgboost model.

        Args:
            model: The trained xgboost model.

        Returns:
            None
        """

        os.makedirs(f"{self._checkpoint_dir_path}/{self._experiment_name}", exist_ok=True)
        
        with open(f'{self._checkpoint_dir_path}/{self._experiment_name}/decision_tree.pkl', 'wb') as f:
            pickle.dump(model, f)

    @logger.catch
    def _load_model(
        self
    ) -> None:
        """
        This function is needed to load in the xgboost model.

        Args:
            None

        Returns:
            None
        """

        with open(f'{self._checkpoint_dir_path}/{self._experiment_name}/decision_tree.pkl', 'rb') as f:
            decisiontree = pickle.load(f)
            
        return decisiontree

    @logger.catch 
    def _evaluate_test(
        self,
        model,
        test: pd.DataFrame,
        labels: List[str]
    ) -> None:
        """
        This function is needed to evaluate / predict the given test set.

        Args:
            model: The trained xgboost model we are using for prediction.
            test (pd.DataFrame): The test set that needs to be evaluated / predicted.
            labels (List[str]): A list containing the labels which will translate the predicted values into categorical values. 

        Returns:
            pd.DataFrame: The test set that has been evaluated based on the trained xgboost model.
        """
        
        ndarray_embeddings_test = np.vstack(test['embeddings'].values)

        y_pred = model.predict(ndarray_embeddings_test)

        test["y_true"] = test['act'].apply(lambda x: labels.index(x))
        test["y_pred"] = y_pred

        return test
    
    @logger.catch
    def _split_train(
        self,
        df: pd.DataFrame,
        labels: List[str] 
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        """
        This function is needed to split the initial splitted dataset into a train and validation set.

        Args:
            df (pd.DataFrame): A pandas dataframe including the initial train set.
            labels (List[str]): A list including all the labels we need to determine the target.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple consisting of two pandas dataframes both the training and validation set for the xgboost training.
        """

        train, validation = train_test_split(df, train_size=0.80, random_state=42)

        train['y_true'] = df['act'].apply(lambda x: labels.index(x))
        validation['y_true'] = df['act'].apply(lambda x: labels.index(x))

        return train, validation

    @logger.catch
    def inference(
        self,
        utterance: str
    ) -> str:
        """
        This Function is used to use the trained xgboost model inference

        Args:
            utterance (str): The utterance thats derived from the users inputs.

        Returns:
            str: A categorical predicted value.
        """

        decisiontree = self._load_model()

        utterance = utterance.lower().lstrip()
        
        # tokenization

        y_preds_proba = decisiontree.predict_proba(embeddings)

        index_array = np.argmax(y_preds_proba, axis=-1)

        categorical_pred = self.labels[index_array[0]]

        print(f"""act: {categorical_pred}, probability: {y_preds_proba[0][index_array]}\n""")

        return categorical_pred
    
    @logger.catch
    def evaluate(
        self
    ) -> None:
        """
        This function is needed to evaluate the xgboost classifier we have trained.

        Args:
            None
        
        Returns:
            None
        """

        decisiontree = self._load_model()
        
        results = self._evaluate_test(
            model=decisiontree,
            test=self.model_test,
            labels=self.labels
        )
        
        Evaluate(
            experiment=self._experiment_name,
            dataframe=results,
            labels=self.labels
        ).run()

    @logger.catch
    def run(
        self,
        max_depth: int,
        min_samples_split: int,
        min_samples_leaf: int
    ) -> None:
        """
        This function is meant to be the main train function used for the xgboost classfiier.

        Args:
            verbosity (int): The verbosity level defined inside the training function of the xgboost classfier.
            num_round (int): The number of rounds we need to boost the classifier for.

        Returns:
            None
        """

        decisiontree = self._train_model(
            train=self.model_train,
            validation=self.model_validation,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )

        self._save_model(model=decisiontree)

@decisiontree_app.command()
def evaluate(
    dataset_dir_path: Annotated[str, typer.Option(help="The dataset directory path where the original dialog acts dataset resides in.")] = None,
    checkpoint_dir_path: Annotated[str, typer.Option(help="The checkpoint directory path where the trainable xgboost model will be saved in.")] = None,
    experiment_name: Annotated[str, typer.Option(help="The experiment name of the xgboost model that will be saved as.")] = None,
    deduplication: Annotated[bool, typer.Option(help="Whether the decisiontree model should be trained on deduplicated data from dialog acts dataset.")] = None,
) -> None:

    DecisionTree(
        dataset_dir_path=dataset_dir_path,
        checkpoint_dir_path=checkpoint_dir_path,
        experiment_name=experiment_name,
        deduplication=deduplication
    ).evaluate()

@decisiontree_app.command()
def inference(
    dataset_dir_path: Annotated[str, typer.Option(help="The dataset directory path where the original dialog acts dataset resides in.")] = None,
    checkpoint_dir_path: Annotated[str, typer.Option(help="The checkpoint directory path where the trainable decisiontree model will be saved in.")] = None,
    experiment_name: Annotated[str, typer.Option(help="The experiment name of the decisiontree model that will be saved as.")] = None,
    deduplication: Annotated[bool, typer.Option(help="Whether the decisiontree model should be trained on deduplicated data from dialog acts dataset.")] = None,
) -> None:
    
    decisiontree = DecisionTree(
        dataset_dir_path=dataset_dir_path,
        checkpoint_dir_path=checkpoint_dir_path,
        experiment_name=experiment_name,
        deduplication=deduplication
    )

    while True:

        utterance = input("Enter your utterance: ")

        decisiontree.inference(utterance=utterance.lower())

@decisiontree_app.command()
def train(
    dataset_dir_path: Annotated[str, typer.Option(help="The dataset directory path where the original dialog acts dataset resides in.")] = None,
    checkpoint_dir_path: Annotated[str, typer.Option(help="The checkpoint directory path where the trainable decisiontree model will be saved in.")] = None,
    experiment_name: Annotated[str, typer.Option(help="The experiment name of the decisiontree model that will be saved as.")] = None,
    max_depth: Annotated[int, typer.Option(help="The max depth of the decision tree model.")] = None,
    min_samples_split: Annotated[int, typer.Option(help="The minimum amount of samples per split.")] = None,
    min_samples_leaf: Annotated[int, typer.Option(help="The minimum amount of samples per leaf")] = None,
    deduplication: Annotated[bool, typer.Option(help="Whether the decisiontree model should be trained on deduplicated data from dialog acts dataset.")] = None,
) -> None:
    
    DecisionTree(
        dataset_dir_path=dataset_dir_path,
        checkpoint_dir_path=checkpoint_dir_path,
        experiment_name=experiment_name,
        deduplication=deduplication
    ).run(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf
    )

