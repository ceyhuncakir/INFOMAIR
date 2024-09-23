import pandas as pd

def identify_keywords(sentence):
    df = pd.read_csv('data/restaurant_info.csv')
    
    keyword_dict = {
    'pricerange': df['pricerange'].unique(), 
    'area': df['area'].unique(),
    'food': df['food'].unique()
    }

    words = sentence.lower().split()
    keywords_found = {}

    for n in range(1, 3): # scan for keyword phrases consisting of one or two words. Some keyword phrases consists of two words, such as 'asian oriental'
        for i in range(len(words) - n + 1):
            phrase = ' '.join(words[i:i + n])
            for key, keywords in keyword_dict.items():
                if phrase in keywords: # if a word is directly found in keyword dict, return that keyword.
                    keywords_found[key] = phrase
                else:
                    # to do: calculate Levenshtein distance here for words not directly included in dict
                    continue

    return keywords_found
            