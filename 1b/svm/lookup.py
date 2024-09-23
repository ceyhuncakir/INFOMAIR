import pandas as pd

def lookup_restaurant(pricerange=None, area=None, food=None):

    df = pd.read_csv('1b/data/restaurant_info.csv')
    results = df

    if pricerange:
        results = results[results['pricerange'] == pricerange.lower()]
    if area:
        results = results[results['area'] == area.lower()]
    if food:
        results = results[results['food'] == food.lower()]

    return results
