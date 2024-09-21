import os
import sys
import typer
from typing_extensions import Annotated
import pandas as pd
import re
import spacy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../1a/src')))

from structures import Inform
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
        deduplication: bool,
        restaurant_csv: str
    ) -> None:
        
        self.model_name = model_name
        self.dataset_dir_path = dataset_dir_path
        self.vectorizer_dir_path = vectorizer_dir_path
        self.checkpoint_dir_path = checkpoint_dir_path
        self.experiment_name = experiment_name
        self.deduplication = deduplication
        self._restaurant_csv = restaurant_csv

        self._restaurant_df = pd.read_csv(restaurant_csv)
        self.model = self._initialize_model()
        self._labels = self.model._labels

        self._inform_class = Inform()
        self._chat_history = []
        self._turn_index = 0
        self._state = 9 
    
    def _hello(
        self
    ) -> str:

        return "Hello , welcome to the Cambridge restaurant system? You can ask for restaurants by area , price range or food type . How may I help you?"
        
    def _lookup_restaurant(
        self
    ) -> pd.Series:

        filter_condition = pd.Series([True] * len(self._restaurant_df))
        filter_condition &= (self._restaurant_df['pricerange'] == self._inform_class.pricerange)
        
        if self._inform_class.area.lower() != 'dontcare':
            filter_condition &= (self._restaurant_df['area'] == self._inform_class.area)
    
        filter_condition &= (self._restaurant_df['food'] == self._inform_class.food)
        
        results = self._restaurant_df[filter_condition]
        return results

    def _inform(
        self,
        utterance: str
    ) -> str:

        keywords = self.identify_keywords(utterance=utterance)

        self._inform_class.type = keywords['type'] if self._inform_class.type is None else self._inform_class.type
        self._inform_class.pricerange = keywords['pricerange'] if self._inform_class.pricerange is None else self._inform_class.pricerange
        self._inform_class.area = keywords['area'] if self._inform_class.area is None else self._inform_class.area
        self._inform_class.food = keywords['food'] if self._inform_class.food is None else self._inform_class.food

        if self._inform_class.pricerange and self._inform_class.area and self._inform_class.food:
            results = self._lookup_restaurant()

            if results.empty:
                return f"I'm sorry but there is no restaurant serving {self._inform_class.food} food"

        if self._inform_class.area == None:
            print(f"speech act: inform(type={self._inform_class.type},pricerange={self._inform_class.pricerange}, task=find)")
            return "What part of town do you have in mind?"     
        else:
            print(f"speech act: inform(area={self._inform_class.area})")  

        if self._inform_class.food == None:
            return "What kind of food would you like?"
        else:
            print(f"speech act: inform(food={self._inform_class.food})")

    def identify_keywords(
        self,
        utterance: str
    ) -> dict:
        
        words = utterance.split(" ")

        keywords = {
            "type": None,
            "pricerange": None,
            "task": None,
            "food": None,
            "area": None
        }

        for word in words:
            
            if "priced" == word:
                keywords['pricerange'] = words[words.index(word) - 1]
            if "expensive" == word or "cheap" == word:
                keywords['pricerange'] = words[words.index(word)] 
            if "restaurant" == word:
                keywords['type'] = words[words.index(word)] 
            if "food" == word:
                keywords['food'] = words[words.index(word) - 1]
            if "part" == word:
                if words[words.index(word) - 1] == "any":
                    keywords['area'] = "dontcare"
                else:
                    keywords['area'] = words[words.index(word) - 1]
        
        return keywords

    def _step(
        self,
        state: int,
        utterance: str,
    ) -> None:        

        self._state = self._labels.index(state)
        self._turn_index += 1

        if self._state == 9:
            return self._hello(), self._turn_index
        elif self._state == 0:
            return self._inform(utterance=utterance), self._turn_index
        else:
            return 

    def _reset(
        self
    ) -> str:

        return self._hello(), self._turn_index   

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

        # print(self._labels)

        while True:

            utterance = input("user: ")

            categorical_pred, _ = self.model.inference(utterance=utterance.lower()) 

            resp, turn_index = self._step(state=categorical_pred, utterance=utterance)
            
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
    restaurant_csv: Annotated[str, typer.Option(help="The restaurant csv data directory")] = None,
) -> None:

    dialog_manager = DialogManager(
        model_name=model_name,
        dataset_dir_path=dataset_dir_path,
        vectorizer_dir_path=vectorizer_dir_path,
        checkpoint_dir_path=checkpoint_dir_path,
        experiment_name=experiment_name,
        deduplication=deduplication,
        restaurant_csv=restaurant_csv
    ).inference()

    while True:

        utterance = input("Enter your utterance: ")

        categorical_pred, probability = dialog_manager.inference(utterance=utterance.lower())

        print(f"""act: {categorical_pred}, probability: {probability}\n""")