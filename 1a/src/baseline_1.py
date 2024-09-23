from typing import Tuple, List

import pandas as pd
import typer
from typing_extensions import Annotated

from helpers.base import Base
from helpers.evaluation import Evaluate

baseline_1_app = typer.Typer()

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
        _, self.test, self.labels, self.majority = self.process(
            deduplication=False
        )

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

    def _forward(
        self,
        labels: List[str],
        majority: str,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        This funciton is used as a model forward function.

        Args:
            labels (List[str]): A list of strings containing the labels.
            majority (str): The majority class based on the distribution of data.
            df (pd.DataFrame): A dataframe consisting of data based on the loaded in dataset.

        Returns:
            pd.DataFrame: A dataframe consiting of data with the new predictions from the model.
        """

        df['y_true'] = df['act'].apply(lambda x: labels.index(x))
        df['y_pred'] = labels.index(majority)
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

        y_pred = self.majority
        
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
            majority=self.majority, 
            df=self.test
        )

        Evaluate(
            experiment="baseline 1",
            dataframe=results,
            labels=self.labels
        ).run()

@baseline_1_app.command()
def inference(
    dataset_dir_path: Annotated[str, typer.Option(help="The dataset dir path we want to specify for the dataset.")] = None,
) -> None:
    """
    This function is needed to run the main inference process.

    Args:
        dataset_dir_path (str): A string defining the dataset directory path
    
    Returns:
        None
    """

    baseline_1 = Baseline_1(
        dataset_dir_path=dataset_dir_path,
    )

    while True:

        text = input("Enter your utterance: ")

        baseline_1.inference(text=text)

@baseline_1_app.command()
def evaluate(
    dataset_dir_path: Annotated[str, typer.Option(help="The dataset dir path we want to specify for the dataset.")] = None,
) -> None:
    """
    This function is needed to run the main evaluation process.

    Args:
        dataset_dir_path (str): A string defining the dataset directory path
    
    Returns:
        None
    """
    
    Baseline_1(
        dataset_dir_path=dataset_dir_path,
    ).run()