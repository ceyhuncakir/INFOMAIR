import pandas as pd
from Levenshtein import distance

def identify_keywords(sentence, state):
    df = pd.read_csv('1b/svm/data/restaurant_info_extra.csv')

    common_words = ['I', 'the', 'an', 'in', 'is', 'for', 'can', 'have', 'am', 'need', 'want', 'what', 'that', 'this', 'their', 'its', 'yes', 'no', 'eat', 'else']
    any_conditions = {
        'food': ['any food'],
        'area': ['anywhere', 'any place', 'any area'],
        'pricerange': ['any price', 'any pricerange'],
        'general': ['any', 'anything', 'dont care', 'doesnt matter', 'you choose']
    }

    keyword_dict = {
    'pricerange': df['pricerange'].dropna().astype(str).unique(), 
    'area': df['area'].dropna().astype(str).unique(),
    'food': df['food'].dropna().astype(str).unique()
    }

    words = sentence.lower().split()
    keywords_found = {}
    min_distance = {} 

    for attribute, phrases in any_conditions.items(): # hanle more specific 'dont care' values that can be given anytime. E.g., "anywhere"
        if attribute == 'general':
            continue
        for phrase in phrases:
            if phrase in sentence.lower():
                keywords_found[attribute] = 'any'
                break

    for phrase in any_conditions['general']: # handle 'dontcare' value per state. E.g., if user input includes 'any' when asked for food in state 3. Assume user is referring to food.
        if phrase in sentence.lower():
            if state == 2:
                return {'area': 'any'}
            elif state == 3:
                return {'food': 'any'}
            elif state == 4:
                return {'pricerange': 'any'}
            break

    for n in range(1, 3): # scan for keyword phrases consisting of one or two words. Some keyword phrases consists of two words, such as 'asian oriental'
        for i in range(len(words) - n + 1):
            phrase = ' '.join(words[i:i + n])
            for key, keywords in keyword_dict.items():
                if phrase in common_words: # skip common words to prevent confusion
                    continue
                elif phrase in keywords: # if phrase is directly found in the keywords, return that keyword
                    keywords_found[key] = phrase
                    min_distance[key] = 0
                else:
                    for keyword in keywords: 
                        phrase_dist = distance(phrase, keyword) # if not found diretly, calculate Levenshtein distance for each phrase and each keyword
                        if phrase_dist < 3:
                            if key not in min_distance or phrase_dist < min_distance[key]: # keep only smallest distance and distances below 3
                                min_distance[key] = phrase_dist
                                keywords_found[key] = keyword

    return keywords_found