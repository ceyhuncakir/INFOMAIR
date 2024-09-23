import pandas as pd
import numpy as np
import typer
from typing_extensions import Annotated
import math

from helpers.base import Base

statistics_app = typer.Typer()

class Statistics(Base):
    """
    This class is responsible for calculating statistics about the dataset.

    Attributes:
        dataset_dir_path (str): The dataset directory path where the original dialog acts dataset resides in.
        deduplication (bool): Whether to deduplicate the dataset.
    """

    def __init__(
        self,
        dataset_dir_path: str,
        deduplication: bool
    ) -> None:
        
        self._dataset_dir_path = dataset_dir_path

        self._train, self._test, self._labels, self._majority = self.process(
            deduplication=deduplication
        )

        self._dataset = pd.concat([self._train, self._test], axis=0)
        self._dataset['amount_of_words_in_utterance'] = self._dataset['utterance'].apply(lambda x: len(x.split(" ")))
    
    def _act_distribution(
        self
    ) -> pd.Series:
        """
        This function calculates the distribution of dialog acts in the dataset.

        Args:   
            None

        Returns:
            pd.Series: The distribution of dialog acts in the dataset.
        """
        
        distribution = self._dataset['act'].value_counts()
        return distribution

    def _utterance_entropy(
        self
    ) -> float:
        """
        This function calculates the mean entropy of the utterances in the dataset.

        Args:
            None
        
        Returns:
            float: The mean entropy of the utterances in the dataset.
        """
        
        def calculate_entropy(
            utterance: str
        ) -> float:

            """
            This function calculates the entropy of an utterance.

            Args:
                utterance (str): The utterance to calculate the entropy for.    
            
            Returns:
                float: The mean entropy of the utterance.
            """

            # Calculate the probability of each character in the utterance
            probabilities = [utterance.count(char) / len(utterance) for char in set(utterance)]

            # Calculate the entropy using the formula: -sum(p * log2(p))
            entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
            return entropy

        # calculate entropy for each utterance
        entropy = self._dataset['utterance'].apply(calculate_entropy)

        # Return the mean entropy of the utterances
        return entropy.mean()

    def run(
        self
    ) -> None:
        """
        This function runs the statistics calculation and prints the results.

        Args:
            None
        
        Returns:
            None
        """

        distribution = self._act_distribution()
        entropy = self._utterance_entropy()

        print("\n--- Dataset Statistics ---\n")

        # Print Dialog Act Distribution
        print("Dialog Act Distribution:")
        print(distribution)
        print("\n-----------------------------")

        # Print Mean Entropy of Utterances
        print(f"\nMean Entropy of Utterances: {entropy:.4f}")

        # Print Missing Values in Dataset
        print("\nMissing Values in Dataset:")
        print(self._dataset.isna().sum())

        # Print Statistics about Utterance Length
        print("\nStatistics about Utterance Length (Words):")
        print(self._dataset['amount_of_words_in_utterance'].describe())

        print("\n-----------------------------\n")

@statistics_app.command()
def run(
    dataset_dir_path: Annotated[str, typer.Option(help="The dataset directory path where the original dialog acts dataset resides in.")] = None,
    deduplication: Annotated[bool, typer.Option(help="Whether to deduplicate the dataset.")] = None,
) -> None:

    Statistics(
        dataset_dir_path=dataset_dir_path,
        deduplication=deduplication
    ).run()