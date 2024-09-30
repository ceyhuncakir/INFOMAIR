from keyword_identify import identify_keywords
from lookup import lookup_restaurant
from additional_requirements import append_features, extract_additional_req
from custom_sentences import create_custom_sentence
from typing_extensions import Annotated
import os
from collections import defaultdict
import numpy as np
import pandas as pd
import sys
import typer
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../1a/src')))

from logistic_regression import LogisticRegressionClassifier

# To do: contradicting rules vergelijken
# To do: Phone address fixen

dialog_manager_app = typer.Typer()

def dialog_manager(do_delay = True):
    
    def restart():
        global preferences, additional_requirements, req_idx, state

        print("system: Okay, let's try again")

        preferences = {'area': None, 'food': None, 'pricerange': None}
        additional_requirements = defaultdict(dict)
        
        req_idx = 0
        state = 2

    if not os.path.isfile('data/restaurant_info_extra.csv'):
        append_features()

    classifier = LogisticRegressionClassifier(
        dataset_dir_path="data/dialog_acts.dat",
        vectorizer_dir_path="data/vectorizer/tfidf_vectorizer.pkl",
        checkpoint_dir_path="data/logistic_regression",
        experiment_name="logistic-regression-deduplicated-tfidf-vectorizer",
        deduplication=True
    )

    preferences = {'area': None, 'food': None, 'pricerange': None}
    additional_requirements = defaultdict(dict)
    exclusion_list = []

    req_idx = 0
    state = 1

    print("system: Hello, welcome to the Cambridge restaurant system! You can ask for restaurants by area, price range, or food type. How may I help you?")
    
    while True:
        user_input = input('user: ')
        dialog_act = classifier.inference(user_input.lower())[0]
        
        if do_delay:
            time.sleep(3)

        print(f"dialog act: {dialog_act}")

        identified_keywords = identify_keywords(user_input, state)
        preferences.update({k: v for k, v in identified_keywords.items() if v})

        results = lookup_restaurant(**preferences, exclusion_list=exclusion_list)

        #
        # Ending conversation
        #

        if dialog_act == 'bye':
            state = 10

        #
        # Restarting conversation
        #

        elif dialog_act == 'restart':
            restart()

        #
        # No restaurants found
        #

        elif results.empty:
            state = 7

        #
        # Handling alteration requests
        # 
        
        elif state == 8:
            if dialog_act in ['deny', 'negate']:
                restart()
            elif dialog_act in ['affirm']:
                req_idx = 0
                state = 20

        #
        # Confirm 'dontcare' values
        # 
        
        elif state == 5:
            if dialog_act in ['deny', 'negate']:
                restart()
            elif dialog_act in ['affirm']:
                state = 20
        
        #
        # Collecting additional requirements
        #     

        elif state == 20:
            if dialog_act in ['deny', 'negate']:  # User does not give additional requirements
                state = 6
            else:
                found_reqs = extract_additional_req(user_input)
                if found_reqs:
                    for keyword, requirement in found_reqs:
                        if keyword == 'touristic' and preferences['food'] != 'romanian':
                            exclusion_list.append('romanian')# If 'touristic' is a requirement and user is not specifically looking for romanian food, exclude this from the recommendations
                        for key, value in requirement.items():
                            if key in preferences and preferences[key] != value and preferences[key] != 'any': # Check for contradictions between requirements and previous preferences
                                print(f"system: Although you were initially looking for something {preferences[key]}, I changed this to {value} because you want something {keyword}.")
                            additional_requirements[keyword][key] = value
                            preferences[key] = value # If a contradiction occurs, overwrite the first value with the latter value. I.e., requirements overwrite preferences
                    results = lookup_restaurant(**preferences, exclusion_list=exclusion_list)
                    if results.empty:
                        state = 7
                    else:
                        state = 6

        #
        # All preferences are given
        #

        elif all(preferences.values()):
            if dialog_act == 'reqmore' and len(results.index) > 1:
                req_idx += 1
                state = 6
            elif any(preferences.get(key) == 'any' for key in ['food', 'pricerange', 'area']):
                state = 5
            elif dialog_act == 'reqalts': 
                state = 8
            elif dialog_act == 'request':
                state = 9
            else:
                state = 20
        
        #
        # Collecting preferences
        #

        else:
            if preferences['area'] is None:
                state = 2
            elif preferences['food'] is None:
                state = 3
            elif preferences['pricerange'] is None:
                state = 4
                            
        #
        # Print statements
        #
        
        if state == 2:
            print("system: What part of town do you have in mind?")

        elif state == 3:
            print("system: What kind of food would you like?")

        elif state == 4:
            print("system: Would you like something in the cheap, moderate, or expensive price range?")

        elif state == 5:
            print(f"system: So you are looking for {preferences['food']} food in {preferences['area']} area and in the {preferences['pricerange']} pricerange?")

        elif state == 20:
            print("system: Do you have additional requirements?")

        elif state == 6:
            if results.empty:
                state = 7
            else:
                restaurant = results.iloc[req_idx]['restaurantname']
                food = results.iloc[req_idx]['food']
                pricerange = results.iloc[req_idx]['pricerange']
                phone = results.iloc[req_idx]['phone']
                address = results.iloc[req_idx]['addr']
            
                print(f"system: {restaurant} is a great choice serving {food} food in the {pricerange} price range.")
            
                if any(additional_requirements.values()):
                    print(create_custom_sentence(additional_requirements))

        elif state == 7:
            print(f"system: I'm sorry but there is no restaurant serving {preferences['food']} food that meets your preferences")
            print(f"preferences: {preferences}")

        elif state == 8:
            print(f"system: You are looking for a {preferences['food']} restaurant right?")

        elif state == 9:
            if 'phone' in user_input:
                print(f'The phone number of {restaurant} is {phone}')
            elif 'address' in user_input:   
                print(f'Sure, {restaurant} is on {address}') 
            else:
                print('system: Do you want to know their address or phone number?') 

        elif state == 10:
            break 

@dialog_manager_app.command()
def run(do_delay: Annotated[bool, typer.Option("--do-delay")] = False):

    dialog_manager(do_delay)