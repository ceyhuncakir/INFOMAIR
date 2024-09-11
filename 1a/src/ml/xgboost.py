from typing import Tuple, List

from sklearn.model_selection import train_test_split
import pandas as pd
import xgboost as xgb
import gensim

from base import Base
from helpers.evaluation import Evaluate

class Xgboost(Base):
    def __init__(
        self,
        dataset_dir_path: str
    ) -> None:
        
        self._dataset_dir_path = dataset_dir_path

    def train(
        self,
        train: pd.DataFrame,
        validation: pd.DataFrame
    ) -> None:

        pass

    def doc2vec(
        self,
        train: pd.DataFrame,
        validation: pd.DataFrame
    ) -> None:
        
        pass

    def _split_train(
        self,
        df: pd.DataFrame,
        labels: List[str] 
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        train, validation = train_test_split(df, train_size=0.80, random_state=42)

        train['y_true'] = df['act'].apply(lambda x: labels.index(x))
        validation['y_true'] = df['act'].apply(lambda x: labels.index(x))

        return train, validation

    def run(
        self
    ) -> None:
        
        df = self._load_data()
        df = self.set_columns(df=df)
        df = self._preprocess(df=df)

        labels = self._get_labels(df=df)

        train, model_test = self._split_train_test(df=df, labels=labels)

        model_train, model_validation = self._split_train(df=train)
