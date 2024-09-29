import pandas as pd

def lookup_restaurant(pricerange=None, area=None, food=None, foodquality=None, crowdedness=None, lengthofstay=None):
    
    def filter_any(value):
        return None if value is not None and value == 'any' else value

    pricerange = filter_any(pricerange)
    area = filter_any(area)
    food = filter_any(food)
    foodquality = filter_any(foodquality)
    crowdedness = filter_any(crowdedness)
    lengthofstay = filter_any(lengthofstay)

    df = pd.read_csv('1b/svm/data/restaurant_info_extra.csv')
    results = df

    if pricerange:
        results = results[results['pricerange'] == pricerange.lower()]
    if area:
        results = results[results['area'] == area.lower()]
    if food:
        results = results[results['food'] == food.lower()]
    if foodquality:
        results = results[results['foodquality'] == foodquality.lower()]
    if crowdedness:
        results = results[results['crowdedness'] == crowdedness.lower()]
    if lengthofstay:
        results = results[results['lengthofstay'] == lengthofstay.lower()]

    return results
