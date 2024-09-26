from keyword_identify import identify_keywords
from lookup import lookup_restaurant
from additional_requirements import append_features, extract_additional_req
import joblib
import os

# To do: allow for 'any' area, food and pricerange

if __name__ == "__main__":
    if not os.path.isfile('1b/svm/data/restaurant_info_extra.csv'):
        append_features()

    vectorizer = joblib.load(os.path.join('1b/svm/models', 'tfidf_vectorizer.pkl'))
    classifier = joblib.load(os.path.join('1b/svm/models', 'svm_classifier.pkl'))

    preferences = {'area': None, 'food': None, 'pricerange': None}
    additional_requirements = {}

    req_idx = 0
    state = 1

    print("system: Hello, welcome to the Cambridge restaurant system! You can ask for restaurants by area, price range, or food type. How may I help you?")
    
    while True:
        user_input = input('user: ')
        vec_input = vectorizer.transform([user_input])
        dialog_act = classifier.predict(vec_input)[0]
        print(f"Dialog act: {dialog_act}")

        identified_keywords = identify_keywords(user_input)
        preferences.update({k: v for k, v in identified_keywords.items() if v})

        lookup_args = {**preferences, **additional_requirements}
        results = lookup_restaurant(**lookup_args)

        #
        # Ending conversation
        #

        if dialog_act == 'bye':
            state = 10

        #
        # No restaurants found
        #

        if results.empty:
            state = 7

        #
        # Parsing restaurant after all preferences are given
        #

        elif all(preferences.values()):
            if dialog_act == 'reqmore' and len(results.index) > 1:
                req_idx += 1
                state = 6
            elif dialog_act == 'reqalts': 
                state = 8
            elif dialog_act == 'request':
                state = 9
            else:
                state = 11
        
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
        # Collecting additional requirements
        #     

        if state == 11:
            if dialog_act in ['deny', 'negate']: # User does not give additional requirements
                state = 6 
            else:
                found_reqs = extract_additional_req(user_input)
                if found_reqs:
                    for keyword, requirement in found_reqs:
                        for key, value in requirement.items():
                            if key in preferences and preferences[key] != value:
                                print(f"system: Although you were intially looking for somehting {preferences[key]}. I changed this to {value} as you want something {keyword}")
                            additional_requirements[key] = value
                    lookup_args = {**preferences, **additional_requirements}
                    results = lookup_restaurant(**lookup_args)
                    if results.empty:
                        state = 7
                    else:
                        state = 6
                            
        #
        # Print statements
        #
        
        if state == 2:
            print("system: What part of town do you have in mind?")
        elif state == 3:
            print("system: What kind of food would you like?")
        elif state == 4:
            print("system: Would you like something in the cheap, moderate, or expensive price range?")
        elif state == 11:
            print("system: Do you have additional requirements?")
        elif state == 6:
            restaurant = results.iloc[req_idx]['restaurantname']
            food = results.iloc[req_idx]['food']
            pricerange = results.iloc[req_idx]['pricerange']
            phone = results.iloc[req_idx]['phone']
            address = results.iloc[req_idx]['addr']
            
            print(f"system: {restaurant} is a great choice serving {food} food in the {pricerange} price range.")
            for requirement, value in additional_requirements.items():
                if value:
                    print(f"system: This restaurant meets your {requirement} requirement, because it is known to be {value}")
        elif state == 7:
            print(f"system: I'm sorry but there is no restaurant serving {preferences['food']} food")
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
        
