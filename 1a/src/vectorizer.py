import os
from typing import Tuple, List, Union

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import typer
from typing_extensions import Annotated
from loguru import logger
import pickle

from helpers.base import Base

vectorizer_app = typer.Typer()

class Vectorizer(Base):
    """
    This class is responsible for vectorizing the dataset.

    Attributes:
        dataset_dir_path (str): The dataset directory path where the original dialog acts dataset resides in.
        vectorizer_type (str): The vectorizer we want to use to train it based on our data.
        checkpoint_dir_path (str): The checkpoint directory path where the vectorizer will be saved.
    """
    
    def __init__(
        self, 
        dataset_dir_path: str,
        vectorizer_type: str,
        checkpoint_dir_path: str
    ) -> None:

        self._dataset_dir_path = dataset_dir_path
        self._vectorizer_type = vectorizer_type
        self._checkpoint_dir_path = checkpoint_dir_path

        self._train, self._test, _, _ = self.process(
            deduplication=False
        )

        self._vectorizer = self._vectorizer_choice(
            vectorizer_type=vectorizer_type
        )

    @logger.catch
    def _vectorizer_choice(
        self,
        vectorizer_type: str
    ) -> Union[TfidfVectorizer, CountVectorizer]:
        """
        This function is responsible for choosing the vectorizer based on the vectorizer type.

        Args:
            vectorizer_type (str): The vectorizer we want to use to train it based on our data.
        
        Returns:
            Union[TfidfVectorizer, CountVectorizer]: The vectorizer we want to use to train it based on our data.   
        """
        
        if vectorizer_type == "tfidf":
            return TfidfVectorizer()   
        elif vectorizer_type == "count":
            return CountVectorizer()

    @logger.catch
    def _save_vectorizer(
        self,
        vectorizer: Union[TfidfVectorizer, CountVectorizer]
    ) -> None:
        """
        This function is responsible for saving the vectorizer.

        Args:
            vectorizer (Union[TfidfVectorizer, CountVectorizer]): The vectorizer we want to save.
        
        Returns:
            None
        """

        path = f"{self._checkpoint_dir_path}/{self._vectorizer_type}_vectorizer.pkl"

        os.makedirs(self._checkpoint_dir_path, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(vectorizer, f)

    @logger.catch
    def run(
        self
    ) -> None:
        """
        This function is responsible for vectorizing the dataset.

        Args:
            None
        
        Returns:
            None
        """

        dataset = pd.concat([self._train, self._test], axis=0)

        corpus = dataset["utterance"].values

        vectorizer = self._vectorizer.fit(corpus)

        self._save_vectorizer(
            vectorizer=vectorizer
        )

@vectorizer_app.command()
def build(
    dataset_dir_path: Annotated[str, typer.Option(help="The dataset directory path where the original dialog acts dataset resides in.")] = None,
    vectorizer_type: Annotated[str, typer.Option(help="The vectorizer we want to use to train it based on our data.")] = None,
    checkpoint_dir_path: Annotated[str, typer.Option(help="The checkpoint directory path where the vectorizer will be saved.")] = None
) -> None:

    Vectorizer(
        dataset_dir_path=dataset_dir_path,
        vectorizer_type=vectorizer_type,
        checkpoint_dir_path=checkpoint_dir_path
    ).run()