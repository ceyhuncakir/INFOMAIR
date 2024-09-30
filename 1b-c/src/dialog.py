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
from baseline_2 import Baseline_2

dialog_manager_app = typer.Typer()

def dialog_manager(do_delay, levenshtein_dist, do_continious_results, use_baseline):      
    if not os.path.isfile('data/restaurant_info_extra.csv'):
        append_features() # Create new csv file with three extra attributes: foodquality, crowdedness, lengthofstay
    
    classifier_baseline = Baseline_2(dataset_dir_path="data/dialog_acts.dat")
   
    classifier_lg = LogisticRegressionClassifier(
        dataset_dir_path="data/dialog_acts.dat",
        vectorizer_dir_path="data/vectorizer/tfidf_vectorizer.pkl",
        checkpoint_dir_path="data/logistic_regression",
        experiment_name="logisticregression-tfidf-dupe",
        deduplication=True
    )

    preferences = {'area': None, 'food': None, 'pricerange': None}
    additional_requirements = defaultdict(dict) 
    exclusion_list = [] # Foodtypes excluded from the results (e.g., Romanian when requirement is touristic)

    reqmore_keywords = ['more', 'other'] # Keyword matching for the 'reqmore' dialog act

    req_idx = 0 # Keep track of last recommended restaurant. Goes up when user asks for next restaurant recommendation within the same preferences.
    state = 1

    print("system: Hello, welcome to the Cambridge restaurant system! You can ask for restaurants by area, price range, or food type. How may I help you?")
    
    while True:
        user_input = input('user: ')

        if use_baseline:
            dialog_act = classifier_baseline.inference(user_input.lower()) # Configurability option: use baseline classifier instead of logistic regression
        else:
            dialog_act = classifier_lg.inference(user_input.lower())[0]
        
        if do_delay:
            time.sleep(3) # Configurability option: add a delay before every system response

        identified_keywords = identify_keywords(user_input, state, levenshtein_dist)
        preferences.update({k: v for k, v in identified_keywords.items() if v})

        results = lookup_restaurant(**preferences, exclusion_list=exclusion_list)
        print(dialog_act)

        if do_continious_results:
            if len(results) > 1:
                print("system: So far, these are some of the restaurants that meet your preferences:\n")
                print(f"{'Name':<50} {'Food':<25} {'Price':<25} {'Area':<25}")
                print("-" * 125)
                count = 0
                for name, food, price, area in zip(results['restaurantname'], results['food'], results['pricerange'], results['area']):
                    if count < 10:
                        print(f"{name:<50} {food:<25} {price:<25} {area:<25}")  # Configurability option: continiously print remaining results
                        count += 1
                    else:
                        break
                print("-" * 125)

        #
        # Resetting conversation
        #

        if state == 11:
            preferences = {'area': None, 'food': None, 'pricerange': None}
            additional_requirements = defaultdict(dict)
        
            req_idx = 0
            state = 2
        #
        # Ending conversation
        #

        elif dialog_act == 'bye':
            state = 10

        #
        # Restarting conversation
        #

        elif dialog_act == 'restart':
            state = 11

        #
        # No restaurants found
        #

        elif results.empty:
            state = 7

        #
        # Requesting more restaurants
        #

        elif any(phrase in reqmore_keywords for phrase in user_input.lower().split()) and all(preferences.values()): # Perform keyword matching for the 'reqmore' dialog act. Regular classifier fails to recognize this class.
            if(len(results) > 1):
                dialog_act = 'reqmore'
                req_idx += 1
    
                if req_idx >= len(results): # If there are no more results, reset the index to 0.
                    req_idx = 0 
                    print("system: Sorry, there are no more restaurants that meet your preferences.")
                    state = 6

        #
        # Providing phone number, address
        #

        elif dialog_act == 'request' and all(preferences.values()):
                state = 9

        #
        # Handling alteration requests
        #

        elif dialog_act == 'reqalts' and all(preferences.values()):
            state = 8

            if dialog_act in ['deny', 'negate']:
                state = 11
            elif dialog_act in ['affirm']:
                req_idx = 0 # Reset request index after changing preference.
                state = 12

        #
        # All preferences are given
        #

        elif state == 6:
            if any(preferences.get(key) == 'any' for key in ['food', 'pricerange', 'area']):
                state = 5
            else:
                continue

        #
        # Confirm 'dontcare' values
        # 
        
        elif state == 5:
            if dialog_act in ['deny', 'negate']:
                state = 11
            elif dialog_act in ['affirm']:
                state = 12
        
        #
        # Collecting additional requirements
        #     

        elif state == 12:
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
        # Collecting preferences
        #

        else:
            if preferences['area'] is None:
                state = 2
            elif preferences['food'] is None:
                state = 3
            elif preferences['pricerange'] is None:
                state = 4
            else:
                state = 5
                            
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

        elif state == 12:
            print("system: Do you have additional requirements?")

        elif state == 6:
            if results.empty:
                state = 7
            else:
                if req_idx >= len(results): # Reset req index to 0 to prevent running out of boundaries after changing a preference
                    req_idx = 0
                restaurant = results.iloc[req_idx]['restaurantname']
                food = results.iloc[req_idx]['food']
                pricerange = results.iloc[req_idx]['pricerange']
                phone = results.iloc[req_idx]['phone']
                address = results.iloc[req_idx]['addr']
            
                print(f"system: {restaurant} is a great choice serving {food} food in the {pricerange} price range.")
            
                if any(additional_requirements.values()):
                    print(create_custom_sentence(additional_requirements))

        elif state == 7:
            print(f"system: I'm sorry but there is no restaurant serving {preferences['food']} food that meets your preferences.")
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
        
        elif state == 11:
            print("system: Okay, let's try again.")

        elif state == 10:
            print("system: Goodbye.")
            break 

@dialog_manager_app.command()
def run(do_delay: Annotated[bool, typer.Option("--do-delay")] = False,
        levenshtein_dist: Annotated[int, typer.Option("--levenshtein_dist")] = 3,
        do_continious_results: Annotated[bool, typer.Option("--do-continious-results")] = False,
        use_baseline: Annotated[bool, typer.Option("--use-baseline")] = False):

    dialog_manager(do_delay, levenshtein_dist, do_continious_results, use_baseline)