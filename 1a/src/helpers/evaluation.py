from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)

class Evaluate:
    def __init__(
        self,
        experiment: str, 
        dataframe: pd.DataFrame,
        labels: List[str]
    ) -> None:
        
        self._experiment = experiment
        self._dataframe = dataframe
        self._labels = labels

    def _precision(
        self,
        y_pred: pd.Series,
        y_true: pd.Series
    ) -> None:
        
        return precision_score(
            y_true=y_true, 
            y_pred=y_pred,
            average="micro",
        )

    def _recall(
        self,
        y_pred: pd.Series,
        y_true: pd.Series
    ) -> None:
        
        return recall_score(
            y_true=y_true,
            y_pred=y_pred,
            average="micro",
        )
    
    def _f1_score(
        self,
        y_pred: pd.Series,
        y_true: pd.Series
    ) -> None:
        
        return f1_score(
            y_pred=y_pred,
            y_true=y_true,
            average="micro",
        )
    
    def _accuracy(
        self,
        y_pred: pd.Series,
        y_true: pd.Series
    ) -> None:
        
        return accuracy_score(
            y_pred=y_pred,
            y_true=y_true
        )

    def _confusion_matrix(
        self,
        y_pred: pd.Series,
        y_true: pd.Series
    ) -> None:
        
        return classification_report(
            y_true, 
            y_pred, 
            target_names=self._labels, 
            zero_division=0
        )
    
    def run(
        self
    ) -> pd.DataFrame:

        precision = self._precision(
            y_pred=self._dataframe['y_pred'],
            y_true=self._dataframe['y_true']
        )

        recall = self._recall(
            y_pred=self._dataframe['y_pred'],
            y_true=self._dataframe['y_true']
        )

        f1_score = self._f1_score(
            y_pred=self._dataframe['y_pred'],
            y_true=self._dataframe['y_true']
        )

        accuracy = self._accuracy(
            y_pred=self._dataframe['y_pred'],
            y_true=self._dataframe['y_true']
        )

        confusion_matrix = self._confusion_matrix(
            y_pred=self._dataframe['y_pred'],
            y_true=self._dataframe['y_true']
        )

        results = f"""
        Experiment: {self._experiment}

        precision: {precision}
        recall: {recall}
        f1_score: {f1_score}
        accuracy: {accuracy}
        
        confusion matrix:\n {confusion_matrix}

        """

        print(results)

