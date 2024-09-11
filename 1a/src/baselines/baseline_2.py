from typing import Tuple, List

from sklearn.model_selection import train_test_split
import pandas as pd

from base import Base
from helpers.evaluation import Evaluate

class Baseline_2(Base):
    def __init__(
        self,
        dataset_dir_path: str
    ) -> None: 
        
        self._dataset_dir_path = dataset_dir_path

    def _rule_based(
        self,
        utterance: str
    ) -> None:

        keyword_matching = {
            "affirm": ["yes", "right"],
            "inform": ["nort", "east", "south", "west", "any", "middle", "cheap", "spanish", "american", "afghan", "chinese", "expensive"],
            "thankyou": ["thank you", "thank you and good bye", "thank you good bye"],
            "bye": ["and good bye", "goodbye", "good bye", "bye"],
            "null": ["sill", "cough", "unintelligible", "sil", "noise"],
            "ack": ["okay", "kay", "okay and", "okay uhm"],
            "request": ["phone number", "address", "postcode", "phone", "post code"],
            "repeat": ["repeat", "again", "repeat that", "go back"],
            "reqmore": ["more"],
            "restart": ["start over", "reset", "start again"],
            "negate": ["no"],
            "hello": ["hi", "hello"], 
            "deny": ["wrong", "dont want"],
            "confirm": ["is it", "is that", "does it", "is there a", "do they"],
            "reqalts": ["how about", "what about", "anything else", "are there", "uh is", "is there", "any more"]
        }

        for act, keyword in keyword_matching.items():

            for keys in keyword:
                if keys in utterance or keys == utterance:
                    return act
                else:
                    continue

        return "inform"


    def _forward(
        self,
        labels: List[str],
        df: pd.DataFrame
    ) -> pd.DataFrame:
        
        df['y_true'] = df['act'].apply(
            lambda x: labels.index(x)
        )

        df['y_pred'] = df['utterance'].apply(
            lambda x: labels.index(self._rule_based(x))
        )

        return df

    def run(
        self
    ) -> None:

        df = self._load_data()
        df = self.set_columns(df=df)
        df = self._preprocess(df=df)

        labels = self._get_labels(df=df)

        train, test = self._split_train_test(df=df)

        results = self._forward(
            labels=labels,
            df=test
        )

        Evaluate(
            experiment="baseline 2",
            dataframe=results,
            labels=labels
        ).run()