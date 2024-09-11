from typing import Tuple, List

from sklearn.model_selection import train_test_split
import pandas as pd

from base import Base
from helpers.evaluation import Evaluate

class Baseline_1(Base):
    """
    This class is the first baseline model for the dialogue act classification task.

    Attributes:
        dataset_dir_path (str): The path to the dataset directory.
    """

    def __init__(
        self,
        dataset_dir_path: str
    ) -> None: 
        
        self._dataset_dir_path = dataset_dir_path

    def _get_majority_class(
        self,
        df: pd.DataFrame
    ) -> str:

        return df['act'].value_counts().idxmax()

    def _forward(
        self,
        labels: List[str],
        majority: str,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        df['y_true'] = df['act'].apply(lambda x: labels.index(x))
        df['y_pred'] = labels.index(majority)
        return df

    def run(
        self
    ) -> None:

        df = self._load_data()
        df = self.set_columns(df=df)
        df = self._preprocess(df=df)

        majority = self._get_majority_class(df=df)

        labels = self._get_labels(df=df)

        train, test = self._split_train_test(df=df)

        results = self._forward(
            labels=labels,
            majority=majority, 
            df=test
        )

        Evaluate(
            experiment="baseline 1",
            dataframe=results,
            labels=labels
        ).run()