from keyword_identify import identify_keywords
from lookup import lookup_restaurant
import joblib
import os

if __name__ == "__main__":
    vectorizer = joblib.load(os.path.join('models', 'tfidf_vectorizer.pkl'))
    classifier = joblib.load(os.path.join('models', 'svm_classifier.pkl'))
    
    preferences = {'area': None, 'food': None, 'pricerange': None}

    print("Hello, welcome to the Cambridge restaurant system! You can ask for restaurants by area, price range, or food type. How may I help you?") # STATE 1

    while True:
        user_input = input()
        vec_input = vectorizer.transform([user_input])
        dialog_act = classifier.predict(vec_input)[0]
        print(f"Dialog act: {dialog_act}")

        identified_keywords = identify_keywords(user_input)
        preferences.update({k: v for k, v in identified_keywords.items() if v})

        results = lookup_restaurant(**preferences)

        if results.empty:
            print("Sorry, no restaurants match your criteria.") # STATE 7
        elif all(preferences.values()):
            restaurant_idx = 0
            restaurant = results.iloc[restaurant_idx]['restaurantname']
            food = results.iloc[restaurant_idx]['food']
            pricerange = results.iloc[restaurant_idx]['pricerange']
            # To do: Confirm selection
            print(f"{restaurant} is a great choice serving {food} food in the {pricerange} price range.") # STATE 6
        else:
            if preferences['area'] is None:
                print("What part of town do you have in mind?") # STATE 2
            elif preferences['food'] is None:
                print("What kind of food would you like?") # STATE 3
            elif preferences['pricerange'] is None:
                print("Would you like something in the cheap, moderate, or expensive price range?") # STATE 4