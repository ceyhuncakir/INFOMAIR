import pandas as pd

def lookup_restaurant(pricerange=None, area=None, food=None, foodquality=None, crowdedness=None, lengthofstay=None):

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
