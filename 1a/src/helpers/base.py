from typing import Tuple, List

from sklearn.model_selection import train_test_split
import pandas as pd
from loguru import logger

from helpers.evaluation import Evaluate

class Base:
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

    @logger.catch
    def _load_data(
        self
    ) -> pd.DataFrame:

        return pd.read_table(
            filepath_or_buffer=self._dataset_dir_path, 
            header=None, 
            names=['data']
        )

    @logger.catch
    def _get_features(
        self,
        row: str
    ) -> None:
    
        seperate_words = row.split(" ")

        act, utterance = seperate_words[0], ' '.join(seperate_words[1:])
        
        return pd.Series([act, utterance])

    @logger.catch
    def set_columns(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        df[['act', 'utterance']] = df.apply(
            lambda row: self._get_features(row=row['data']), axis=1
        )

        return df

    @logger.catch
    def _preprocess(
        self, 
        df: pd.DataFrame,
        deduplication: bool = False
    ) -> pd.DataFrame:

        df['act'] = df['act'].str.lower()
        df['utterance'] = df['utterance'].str.lower()
        df['utterance'] = df['utterance'].str.lstrip()
        
        if deduplication:
            df = df.drop_duplicates(subset=['utterance'])
            counts = list(df['act'].value_counts().index)
            df = df[df['act'].isin(counts[:-1])]
                        
        return df

    @logger.catch
    def _get_labels(
        self,
        df: pd.DataFrame
    ) -> List[str]:
        
        return list(df['act'].unique())

    @logger.catch
    def _split_train_test(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        
        train, test = train_test_split(
            df[['act', 'utterance']], 
            train_size=0.85, 
            random_state=42            
        )

        return train, test

    @logger.catch
    def _get_majority_class(
        self,
        df: pd.DataFrame
    ) -> str:
        """
        This function gets the majority class based on the dataset distribution.

        Args:
            df (pd.DataFrame): A dataframe consisting of data which is needed to determine the majority.

        Returns:
            str: A string containing the majority class.
        """

        return df['act'].value_counts().idxmax()

    @logger.catch
    def process(
        self,
        deduplication: bool
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], str]:

        df = self._load_data()
        df = self.set_columns(df=df)
        df = self._preprocess(df=df, deduplication=deduplication)

        majority = self._get_majority_class(df=df)

        labels = self._get_labels(df=df)

        train, test = self._split_train_test(df=df)

        return train, test, labels, majority