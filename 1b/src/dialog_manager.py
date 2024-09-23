from typing import Union
import os
import sys
import typer
from typing_extensions import Annotated
import pandas as pd
import re
from Levenshtein import distance

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../1a/src')))

from structures import Data
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

        self._restaurant_results = None
        self._buffer = Data()
        self._current = Data()
        self._corrected_words_buffer = Data()
        self._turn_index = 0
        self._state = 9 
    
    def _hello(
        self
    ) -> str:

        return "Hello , welcome to the Cambridge restaurant system? You can ask for restaurants by area , price range or food type . How may I help you?"
        
    def _lookup_restaurant(
        self,
        data: Data
    ) -> pd.Series:

        filter_condition = pd.Series([True] * len(self._restaurant_df))
        filter_condition &= (self._restaurant_df['pricerange'] == data.pricerange)
        
        if data.area.lower() != 'dontcare':
            filter_condition &= (self._restaurant_df['area'] == data.area)

        if data.food.lower() != 'dontcare':
            filter_condition &= (self._restaurant_df['food'] == data.food)
        
        results = self._restaurant_df[filter_condition]
        return results

    def _request(
        self,
        utterance: str
    ) -> None:

        first_row = self._restaurant_results.iloc[0]
        restaurant_name = first_row['restaurantname']
        
        if "phone" in utterance:
            
            phone_number = first_row['phone']

            print("speech act: request(phone)")
            return f"The phone number of {restaurant_name} is {phone_number}"

    def _goodbye(
        self
    ) -> str:

        return "speech act: thankyou()|bye()"

    def _affirm(
        self,
        utterance: str
    ) -> None:

        if self._corrected_words_buffer.__dict__.items():
            for key, value in self._corrected_words_buffer.__dict__.items():
                if value == None:
                    continue
                else:
                    setattr(self._buffer, key, value)

            self._corrected_words_buffer = Data()

        print("speech act: affirm()")
        self._state = 0
        return self._inform(utterance="")

    def _confirm(
        self,
        utterance: str
    ) -> str:

        keywords = self.identify_keywords(utterance=utterance)

        flag, focus = self._check_keywords(keywords=keywords, data=self._corrected_words_buffer)  

        print(focus)
        print(keywords)
        print(self._current)

        # todo change value from key focus to key keywords only when the value is not none and the value is not the same as the current value

        for key, value in keywords.items():
            if key != "task" and key != "type":
                if value == None:
                    keywords[key] = getattr(self._current, key)

                keywords[key] = getattr(self._current, key)
            else:
                continue
                
        buffer = self._slot_filling(data=self._buffer, keywords=keywords)

        print(f"speech act: confirm(pricerange={buffer.pricerange},type={buffer.type},food={buffer.food})")
        return f"You are looking for a {keywords['food']} {keywords['type']} right?"

    def _inform(
        self,
        utterance: str,
    ) -> str:

        # extract preferences
        keywords = self.identify_keywords(utterance=utterance)      

        # check extracted preferences
        flag, focus = self._check_keywords(keywords=keywords, data=self._corrected_words_buffer)  

        # fill the buffer with the extracted preferences
        buffer = self._slot_filling(data=self._buffer, keywords=keywords)

        if flag == False:

            current_preferences = []
            corrected_preferences = []

            for key, value in focus.items():

                if value[1] == None:
                    continue
                else:
                    current_preferences.append(value[0])
                    corrected_preferences.append(value[1])

            return f"I did not recognize {', '.join(current_preferences)}, did you mean {', '.join(corrected_preferences)}?"

        # check if all preferences are known
        confirmation = self._confirm_options(data=buffer)

        # if all preferences are known, return the restaurant
        if confirmation:
            return confirmation
        else:
            # check if area is known
            area = self._area_not_known(data=buffer)

            if area:    
                return area

            # check if pricerange is known
            pricerange = self._pricerange_not_known(data=buffer)

            if pricerange:
                return pricerange

            # check if food is known
            food = self._food_not_known(data=buffer)

            if food:    
                return food 

    def _check_keywords(
        self,
        keywords: dict,
        data: Data
    ) -> None:

        out_of_bounds_preferes = {}

        # check if all values are None of keywords
        all_none = all(value is None for value in keywords.values())

        # if all values are None, return True
        if all_none:
            return True, data

        # check if the values are not None
        for key, value in keywords.items():

            if key != "task" and key != "type":
                if value:
                    if value == "dontcare":
                        continue
                    else:
                        # check if we can find words that are similar to the given word
                        current_word, corrected_word = self._levenshtein_check(word=value, preference=key)
                        setattr(data, key, corrected_word)

                        # if the corrected word is None, continue
                        if corrected_word == None:
                            continue
                        else: 
                            # we store the current word and the corrected word
                            out_of_bounds_preferes[key] = [current_word, corrected_word]
            else:
                continue

        if out_of_bounds_preferes:
            return False, out_of_bounds_preferes
        else:
            return True, data

    def _slot_filling(
        self,
        data: Data,
        keywords: dict
    ) -> Data:
        
        data.type = keywords['type'] if data.type is None else data.type
        data.pricerange = keywords['pricerange'] if data.pricerange is None else data.pricerange
        data.area = keywords['area'] if data.area is None else data.area
        data.food = keywords['food'] if data.food is None else data.food

        return data

    def _transition_data(
        self,
        data: Data
    ) -> Data:

        self._current = data
        self._buffer = Data()
        return self._current

    def _confirm_options(
        self,
        data: Data  
    ) -> Union[str, None]:
        
        if data.pricerange and data.area and data.food:
            
            # save the buffer to current buffer
            new_data = self._transition_data(data=data)

            # lookup the restaurant
            results = self._lookup_restaurant(
                data=new_data   
            )

            # check if no restaurants are found
            if results.empty:
                return f"I'm sorry but there is no restaurant serving {data.food} food"
            else:

                # store the results
                self._restaurant_results = results
                # Get the first row values
                first_row = results.iloc[0]

                return f"{first_row['restaurantname']} serves {first_row['food']} food in the {first_row['pricerange']} price range"
        else:
            return False
    
    def _area_not_known(
        self,
        data: Data
    ) -> Union[str, None]:
      
        if data.area == None:
            print(f"speech act: inform(type={data.type},pricerange={data.pricerange}, task=find)")
            return "What part of town do you have in mind?"     
        else:
            return False

    def _food_not_known(
        self,
        data: Data
    ) -> Union[str, None]:
        
        if data.food == None:
            print(f"speech act: inform(pricerange={data.pricerange}, type={data.type}, area={data.area})")
            return "What kind of food would you like?"
        else:
            return False

    def _pricerange_not_known(
        self,
        data: Data
    ) -> Union[str, None]:

        if data.pricerange == None:
            print(f"speech act: inform(type={data.type}, area={data.area})")
            return "What is your price range?"
        else:
            return False

    def _levenshtein_check(
        self,
        word: str,
        preference: str
    ) -> Union[str, None]:

        distances = []

        pricerange = self._restaurant_df['pricerange'].unique()
        area = self._restaurant_df['area'].unique()
        food = self._restaurant_df['food'].unique()

        if preference == "pricerange":
            distances = [(option, distance(word, option)) for option in pricerange]
        elif preference == "area":
            distances = [(option, distance(word, option)) for option in area]
        elif preference == "food":
            distances = [(option, distance(word, option)) for option in food]

        # get the minimum distance
        min_value = min(distances, key=lambda scores: scores[1])

        # check if the distance is less than 3 but bigger then 1 (for duplicates)
        if min_value[1] > 1 and min_value[1] <= 3:
            return word, min_value[0]
        else:
            return word, None

    def identify_keywords(
        self,
        utterance: str
    ) -> dict:
        
        # split the utterance into words
        words = utterance.split(" ")

        # initialize the keywords
        keywords = {
            "type": None,
            "pricerange": None,
            "task": None,
            "food": None,
            "area": None
        }

        # fix this becacuse it puts any to any other field
        if utterance == "any":
            keywords['area'] = "dontcare"
            keywords['pricerange'] = "dontcare"
            keywords['food'] = "dontcare"
            return keywords  

        for word in words:
            
            # check if the word is in the list of keywords
            if "priced" == word:
                selected_word = words[words.index(word) - 1]
                keywords['pricerange'] = selected_word
            if "expensive" == word or "cheap" == word:   
                selected_word = words[words.index(word)] 
                keywords['pricerange'] = selected_word
            if "restaurant" == word:
                keywords['type'] = words[words.index(word)] 
            if "food" == word:
                selected_word = words[words.index(word) - 1]
                keywords['food'] = selected_word
            if "part" == word:
                if words[words.index(word) - 1] == "any":
                    keywords['area'] = "dontcare"
                else:
                    selected_word = words[words.index(word) - 1]
                    keywords['area'] = selected_word

        return keywords

    def _step(
        self,
        state: int,
        utterance: str,
    ) -> None:        

        self._state = self._labels.index(state)
        self._turn_index += 1

        if self._state == 9:
            return self._hello(), self._turn_index, False
        elif self._state == 0:
            return self._inform(utterance=utterance), self._turn_index, False
        elif self._state == 1:
            return self._confirm(utterance=utterance), self._turn_index, False
        elif self._state == 2:
            return self._affirm(utterance=utterance), self._turn_index, False
        elif self._state == 4 or self._state == 6:
            return self._goodbye(), self._turn_index, True
        elif self._state == 3:
            return self._request(utterance=utterance), self._turn_index, False
        else:
            return 

    def _rule_based(
        self,
        utterance: str
    ) -> str:
    
        keyword_matching = {
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

        return None

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

        while True:

            # todo add confirm state

            # print(self._labels)

            utterance = input("user: ")

            categorical_pred = self._rule_based(utterance=utterance.lower())
            
            if isinstance(categorical_pred, str):
                resp, turn_index, truncation = self._step(state=categorical_pred, utterance=utterance.lower())
            else:
                categorical_pred, _ = self.model.inference(utterance=utterance.lower()) 
                resp, turn_index, truncation = self._step(state=categorical_pred, utterance=utterance.lower())
            
            print("turn index:", turn_index)
            print("system:", resp)

            if truncation:
                exit()
            

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