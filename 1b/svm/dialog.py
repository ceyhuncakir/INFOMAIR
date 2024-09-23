from keyword_identify import identify_keywords
from lookup import lookup_restaurant
import joblib
import os

# To do: Allow for any area, any food, any pricerange

if __name__ == "__main__":
    vectorizer = joblib.load(os.path.join('models', 'tfidf_vectorizer.pkl'))
    classifier = joblib.load(os.path.join('models', 'svm_classifier.pkl'))
    
    preferences = {'area': None, 'food': None, 'pricerange': None}
    req_idx = 0

    print("system: Hello, welcome to the Cambridge restaurant system! You can ask for restaurants by area, price range, or food type. How may I help you?") # STATE 1

    while True:
        user_input = input('user: ')
        vec_input = vectorizer.transform([user_input])
        dialog_act = classifier.predict(vec_input)[0]
        print(f"Dialog act: {dialog_act}")

        identified_keywords = identify_keywords(user_input)
        preferences.update({k: v for k, v in identified_keywords.items() if v})

        results = lookup_restaurant(**preferences)
        
        #
        # Ending conversation
        # (State 10)
        #

        if dialog_act == 'bye': # STATE 10
            break

        #
        # No restaurants meet user preferences
        # (State 7)
        #

        if results.empty:
            print(f"system: I'm sorry but there is no restaurant serving {preferences['food']} food") # STATE 7
        
        #
        # All preferences are given
        # (State 6, 7, 8, 9)
        #

        elif all(preferences.values()):
            restaurant = results.iloc[req_idx]['restaurantname']
            food = results.iloc[req_idx]['food']
            pricerange = results.iloc[req_idx]['pricerange']
            phone = results.iloc[req_idx]['phone']
            address = results.iloc[req_idx]['addr']

            if dialog_act == 'reqmore':
                if len(results.index) > 1:
                    req_idx += 1
                else:
                    print(f"system: There are no other restaurants serving {preferences['food']} food that meet your criteria") # STATE 7    
            
            elif dialog_act == 'reqalts': 
                print(f"system: You are looking for a {preferences['food']} restaurant right?") # STATE 8

            elif dialog_act == 'request': 
                if 'phone' in user_input:
                    print(f'The phone number of {restaurant} is {phone}') # STATE 9
                elif 'address' in user_input:
                    print(f'Sure, {restaurant} is on {address}') # STATE 9
                else:
                    print('system: Do you want to know their address or phone number?') 

            else:
                print(f"system: {restaurant} is a great choice serving {food} food in the {pricerange} price range.") # STATE 6
        
        #
        # Collecting preferences
        # (State 2, 3, 4)
        #

        else:
            if preferences['area'] is None:
                print("system: What part of town do you have in mind?") # STATE 2
            elif preferences['food'] is None:
                print("system: What kind of food would you like?") # STATE 3
            elif preferences['pricerange'] is None:
                print("system: Would you like something in the cheap, moderate, or expensive price range?") # STATE 4