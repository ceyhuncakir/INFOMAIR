import pandas as pd
import numpy as np
import os

np.random.seed(4)

def append_features():
    df = pd.read_csv('1b-c/data/restaurant_info.csv')

    df['foodquality'] = np.random.choice(['good', 'normal'], size=len(df))
    df['crowdedness'] = np.random.choice(['busy', 'normal'], size=len(df))
    df['lengthofstay'] = np.random.choice(['short', 'long'], size=len(df))

    df.to_csv('1b-c/data/restaurant_info_extra.csv')


def extract_additional_req(sentence):
       
    rules = [
        ('touristic', {'pricerange': 'cheap', 'foodquality': 'good'}),
        ('assigned seats', {'crowdedness': 'busy'}),
        ('children', {'lengthofstay': 'short'}),
        ('romantic', {'crowdedness': 'normal'}),
        ('romantic', {'lengthofstay': 'long'})
    ]

    words = set(sentence.lower().split())
    requirements_found = []

    for keyword, value in rules:
        keywords = set(keyword.split())
        if keywords <= words:
            requirements_found.append((keyword, value))
            
    return requirements_found