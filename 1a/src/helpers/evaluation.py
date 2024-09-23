from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report
)

class Evaluate:
    """
    This class is meant as a evaluation class. this will evaluate the selected model based on several key metrics.

    Attributes:
        experiment (str): The experiment name of the evaluation
        dataframe (pd.DataFrame): The dataframe consisting of values to be evaluated.
        labels (List[str]): The labels we have from our dataset.
    """

    def __init__(
        self,
        experiment: str, 
        dataframe: pd.DataFrame,
        labels: List[str]
    ) -> None:
        
        self._experiment = experiment
        self._dataframe = dataframe
        self._labels = labels

    def _confusion_matrix(
        self,
        y_pred: pd.Series,
        y_true: pd.Series
    ) -> str:

        """
        This function is meant to be used as a confusion matrix.

        Args:
            y_pred (pd.Series): A pandas series consisting of predicted values based on the input.
            y_true (pd.Series): A pandas series consisting of predicted values based on the input.

        Returns:
            None
        """
        
        return classification_report(
            y_true, 
            y_pred, 
            target_names=self._labels, 
            zero_division=0
        )
    
    def run(
        self
    ) -> None:
        """
        This function is meant to be the main function thats being called to evaluate the model.

        Args:
            None

        Returns:    
            None
        """
        
        confusion_matrix = self._confusion_matrix(
            y_pred=self._dataframe['y_pred'],
            y_true=self._dataframe['y_true']
        )

        results = f"""
        Experiment: {self._experiment}
        
        confusion matrix:\n {confusion_matrix}
        """

        print(results)

