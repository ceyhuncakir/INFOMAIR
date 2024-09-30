import os
from typing import Tuple, List

import pandas as pd
import typer
from typing_extensions import Annotated

from helpers.callbacks import *
from helpers.base import Base
from helpers.evaluation import Evaluate

baseline_2_app = typer.Typer()

class Baseline_2(Base):
    """
    This class is the first baseline model for the dialogue act classification task.

    Attributes:
        dataset_dir_path (str): The path to the dataset directory.
    """

    def __init__(
        self,
        dataset_dir_path: str,
    ) -> None: 
        
        self._dataset_dir_path = dataset_dir_path
        _, self.test, self.labels, self.majority, _ = self.process(
            deduplication=False
        )

    def _rule_based(
        self,
        utterance: str
    ) -> str:
        """
        This function is used as a rule based method, which classifies the act for our utterance.

        Args:
            utterance (str): A string containing the utterance.

        Returns;
            str: A string containing the dialogue act.
        """

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

        return self.majority

    def _forward(
        self,
        labels: List[str],
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        This funciton is used as a model forward function.

        Args;
            labels (List[str]): A list containing the labels in strings.
            df (pd.DataFrame): A dataframe containing the data from the dataset.

        Returns:
            pd.DataFrame: A dataframe containing the dataset information which we have gathered with the new information.
        """

        df['y_pred'] = df['utterance'].apply(
            lambda x: self._rule_based(x)
        )

        return df

    def inference(
        self,
        text: str
    ) -> None:
        """
        This function is needed for the inference part of this class.

        Args:
            text (str): A text most likely a utterance which needs to be classified.

        Returns:
            None
        """

        y_pred = self._rule_based(text.lower())
        
        print(f"""act: {y_pred}\n""")

    def run(
        self
    ) -> None:
        """
        This function is needed for the main evaluation of this baseline.

        Args:
            None
        
        Returns:
            None
        """

        results = self._forward(
            labels=self.labels,
            df=self.test
        )

        Evaluate(
            experiment="baseline 2",
            dataframe=results,
            labels=self.labels
        ).run()

@baseline_2_app.command()
def inference(
    dataset_dir_path: Annotated[str, typer.Option(help="The dataset dir path we want to specify for the dataset.", callback=path_valid)] = os.getcwd() + "/data/dialog_acts.dat",
) -> None:
    """
    This function is needed to run the main inference process.

    Args:
        dataset_dir_path (str): A string defining the dataset directory path
    
    Returns:
        None
    """

    baseline_2 = Baseline_2(
        dataset_dir_path=dataset_dir_path,
    )

    while True:

        text = input("Enter your utterance: ")

        baseline_2.inference(text=text)

@baseline_2_app.command()
def evaluate(
    dataset_dir_path: Annotated[str, typer.Option(help="The dataset dir path we want to specify for the dataset.", callback=path_valid)] = os.getcwd() + "/data/dialog_acts.dat",
) -> None:
    """
    This function is needed to run the main evaluation process.

    Args:
        dataset_dir_path (str): A string defining the dataset directory path
    
    Returns:
        None
    """
    
    Baseline_2(
        dataset_dir_path=dataset_dir_path,
    ).run()