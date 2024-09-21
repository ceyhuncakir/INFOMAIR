import os
import sys
import typer
from typing_extensions import Annotated

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../1a/src')))

from helpers.base import Base
from logistic_regression import LogisticRegressionClassifier

dialog_manager_app = typer.Typer()

class DialogManager(Base):
    def __init__(
        self, 
        model_name: str,
        dataset_dir_path: str,
        vectorizer_dir_path: str,
        checkpoint_dir_path: str,
        experiment_name: str,
        deduplication: bool
    ) -> None:
        
        self.model_name = model_name
        self.dataset_dir_path = dataset_dir_path
        self.vectorizer_dir_path = vectorizer_dir_path
        self.checkpoint_dir_path = checkpoint_dir_path
        self.experiment_name = experiment_name
        self.deduplication = deduplication

        self.model = self._initialize_model()
        self._labels = self.model._labels

        self._turn_index = 0
        self._state = 9 
    
    def hello(
        self
    ) -> str:

        return "Hello , welcome to the Cambridge restaurant system? You can ask for restaurants by area , price range or food type . How may I help you?"
        
    def _step(
        self,
        state: int
    ) -> None:

        self._state = state
        self._turn_index += 1

        if self._state == 9:
            return self._hello(), self._turn_index
        else:
            return 

    def _reset(
        self
    ) -> str:

        return self.hello(), self._turn_index   

    def _initialize_model(
        self
    ) -> None:
        
        if self.model_name == "logistic_regression":

            model = LogisticRegressionClassifier(
                dataset_dir_path=self.dataset_dir_path,
                vectorizer_dir_path=self.vectorizer_dir_path,
                checkpoint_dir_path=self.checkpoint_dir_path,
                experiment_name=self.experiment_name,
                deduplication=self.deduplication
            )

            return model
        else:
            raise ValueError("Model not supported")

    def inference(
        self
    ) -> None:

        resp, turn_index = self._reset()

        print("\nturn index:", turn_index)
        print("system:", resp)

        while True:

            utterance = input("user: ")

            categorical_pred, _ = self.model.inference(utterance=utterance.lower()) 

            resp, turn_index = self._step(state=categorical_pred)
            
            print("turn index:", turn_index)
            print("system:", resp)

@dialog_manager_app.command()
def evaluate(
    dataset_dir_path: Annotated[str, typer.Option(help="The dataset directory path where the original dialog acts dataset resides in.")] = None,
) -> None:
    pass

@dialog_manager_app.command()
def inference(
    model_name: Annotated[str, typer.Option(help="The type of act model we want to load in")] = None,
    dataset_dir_path: Annotated[str, typer.Option(help="The dataset directory path where the original dialog acts dataset resides in.")] = None,
    vectorizer_dir_path: Annotated[str, typer.Option(help="The vectorizer directory path where the trained vectorizer model resides in.")] = None,
    checkpoint_dir_path: Annotated[str, typer.Option(help="The checkpoint directory thats needed to load in the model")] = None,
    experiment_name: Annotated[str, typer.Option(help="The experiment name of the model that will be loaded in")] = None,
    deduplication: Annotated[bool, typer.Option(help="Whether the data should be deduplicated for the act model. (not needed)")] = None,
) -> None:

    dialog_manager = DialogManager(
        model_name=model_name,
        dataset_dir_path=dataset_dir_path,
        vectorizer_dir_path=vectorizer_dir_path,
        checkpoint_dir_path=checkpoint_dir_path,
        experiment_name=experiment_name,
        deduplication=deduplication
    ).inference()

    while True:

        utterance = input("Enter your utterance: ")

        categorical_pred, probability = dialog_manager.inference(utterance=utterance.lower())

        print(f"""act: {categorical_pred}, probability: {probability}\n""")